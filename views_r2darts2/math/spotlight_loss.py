import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Asymmetric magnitude-aware temporal loss for imbalanced time-series regression
    on asinh-transformed targets.

    SpotlightLoss "shines a spotlight" on the signal that matters most: it amplifies
    loss contributions from high-magnitude events (via a clamped cosh magnitude
    weight), penalises under-prediction more harshly than over-prediction (via a smooth
    sigmoid asymmetry gate), and optionally regularises the temporal gradient so the
    model tracks the *shape* of the true series, not just its level.

    Components
    ----------
    1. **Magnitude weighting** — ``w_mag = clamp(cosh(alpha * |y_true|), max=W_MAX)``.
       Exponential cosh weight ensures the model pays strong attention to extreme
       conflict cells (e.g. ~23× for Ukraine at alpha=0.4).  A hard clamp at
       ``W_MAX=1000`` prevents runaway gradients if alpha drifts high during sweeps.
       The recommended alpha range is 0.2–0.5 (smol_cat best: 0.387).
    2. **Scaled log-cosh base loss** — parameter-free replacement for Huber.
       ``L_i = s_i * log(cosh(e_i / s_i))`` where ``s_i = 1 + |y_true| / (1 + |y_true|)``.
       Behaves quadratically for small errors, linearly for large errors, with a smooth
       transition (no curvature discontinuity). Gradient bounded at ±1 always.
       The scale ``s_i`` widens the quadratic zone for conflict cells (up to |e|≈2)
       while keeping it tight for peace cells (|e|≈1).
    3. **Directional Basu DPD residual dampening** — a Gaussian gate
       ``exp(-alpha/2 * z^2)`` that only activates for **underprediction**
       (``y_pred < y_true``).  When the model underpredicts a conflict cell by
       a large margin, the gate dampens w_mag toward 1.0, preventing hard
       outliers from destabilising early training (original DPD robustness).
       When the model **overpredicts**, the gate is held at 1.0 so the full
       cosh weight is preserved — the loss provides maximum corrective gradient
       pushing predictions back down.  This breaks the symmetric Basu attractor
       that previously allowed high-alpha configs to park in an overprediction
       basin where ``w_mag → 1`` killed the recovery signal.
    4. **Asymmetric modulation** — a sigmoid ``sigma(-kappa * e)`` activates when the
       model under-predicts (``y_pred < y_true``). The extra penalty scales with
       ``beta`` and is proportional to the *relative* magnitude of the true value,
       ``|y_true| / (1 + |y_true|)``, so asymmetry matters most for real events.
    5. **Temporal gradient term** — optional scaled log-cosh on first-order differences
       ``Delta y_pred - Delta y_true``, weighted by ``gamma``. Encourages the model to
       reproduce step-to-step dynamics, not just pointwise targets. Only first-order
       (velocity); second-order curvature matching was removed to prevent compound
       escalation during autoregressive rollout.  Uses the same directional Basu
       gate: full weight when predicted dynamics overshoot truth, dampened when
       they undershoot.

    Parameters
    ----------
    alpha : float, default 0.4
        Dual-purpose parameter (magnitude amplification + residual dampening).

        **Magnitude amplification:** rate for the cosh weight ``cosh(alpha * |y|)``.
        Controls how much more the loss attends to extreme conflict cells vs peace.
        At alpha=0.4, Ukraine-level conflict (~10k fatalities, asinh≈9.9) receives
        ~23× the weight of a peace cell.

        **Basu residual dampening (underprediction only):** also serves as the
        robustness parameter in the directional DPD gate.  For underprediction
        (``e < 0``), ``exp(-alpha/2 * z^2)`` dampens the cosh weight — higher
        alpha means stronger dampening and a narrower influence peak at
        ``z = 1/sqrt(alpha)``.  For overprediction (``e > 0``), the gate is 1.0
        and the full cosh weight applies regardless of alpha.

        Recommended range: 0.2–0.5 (smol_cat best: 0.387, influence peak at z≈1.6).
    beta : float, default 0.2
        Asymmetry strength. Maximum extra multiplier applied when the model
        under-predicts a non-zero true value. 0.2 = FN costs 1.2× FP on events.
    kappa : float, default 10.0
        Sharpness of the sigmoid transition that activates the asymmetric penalty.
        Higher values create a near-binary switch around ``e = 0``.
    gamma : float, default 0.1
        Weight for the temporal gradient regularisation term. Set to 0 to disable.

    Forward signature
    -----------------
    y_pred : torch.Tensor
        Predicted values, shape ``(batch, seq_len)`` or ``(batch, seq_len, 1)``.
    y_true : torch.Tensor
        Ground-truth values, same shape as ``y_pred``.

    Returns
    -------
    torch.Tensor
        Scalar loss value.

    Raises
    ------
    NumericalSanityError
        If NaN or Inf is detected in inputs or in the computed loss.

    Examples
    --------
    >>> loss_fn = SpotlightLoss(alpha=1.0, beta=1.5, kappa=10.0, gamma=0.1)
    >>> pred = torch.randn(32, 36)
    >>> true = torch.randn(32, 36)
    >>> loss = loss_fn(pred, true)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        gamma: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.gamma = gamma

        logger.info(
            "SpotlightLoss initialised | alpha=%.4f  beta=%.4f  kappa=%.4f  "
            "gamma=%.4f",
            self.alpha,
            self.beta,
            self.kappa,
            self.gamma,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh_scaled(error: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Scaled log-cosh loss: s * log(cosh(e / s)).

        Parameter-free base loss. Quadratic for |e| << s, linear for |e| >> s,
        with a smooth transition. Gradient = tanh(e/s), hard-bounded at ±1.

        Uses the numerically stable identity:
            log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        which avoids overflow for large |x|.
        """
        z = error / scale
        abs_z = torch.abs(z)
        # Stable: for large |z|, exp(-2|z|) → 0, so this → |z| - ln2
        return scale * (abs_z + torch.nn.functional.softplus(-2.0 * abs_z) - 0.6931471805599453) # log(2) ≈ 0.6931471805599453, hardcoded for precision and to avoid an extra function call yeet

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the SpotlightLoss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions, shape ``(batch, seq_len)`` or ``(batch, seq_len, 1)``.
        y_true : torch.Tensor
            Ground-truth targets, same shape as ``y_pred``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        from views_r2darts2.infrastructure.exceptions import NumericalSanityError

        # --- Input sanity checks ---
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            raise NumericalSanityError(
                "Numerical Sanity Violation: NaN or Inf detected in predictions."
            )
        if torch.isnan(y_true).any() or torch.isinf(y_true).any():
            raise NumericalSanityError(
                "Numerical Sanity Violation: NaN or Inf detected in targets."
            )

        # Squeeze trailing singleton (common for Darts models)
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        # Pointwise error
        e = y_pred - y_true

        # ---- 1. Magnitude weight ----
        # Clamped cosh: cosh(alpha * |y_true|), hard-capped at W_MAX.
        _W_MAX = 1e3
        abs_y = torch.abs(y_true)
        w_mag_raw = torch.cosh(self.alpha * abs_y).clamp(max=_W_MAX)

        # ---- 2. Scaled log-cosh base loss ----
        # s_i ∈ [1, 2): peace cells ≈ 1 (tight quadratic), conflict → 2 (wider)
        scale = 1.0 + abs_y / (1.0 + abs_y)
        base_loss = self._log_cosh_scaled(e, scale)

        # ---- 3. Directional Basu DPD residual dampening ----
        # Only dampen for underprediction (e < 0, z < 0).  For overprediction
        # (e > 0), gate = 1.0 → full cosh weight → maximum corrective gradient.
        # clamp(z, max=0) zeros out positive z so exp(0)=1.  At z=0 both
        # branches give 1.0 and derivative 0 → continuous + smooth.
        z = e / scale
        z_under = z.clamp(max=0.0)
        basu_gate = torch.exp(-0.5 * self.alpha * z_under * z_under)
        w_mag = 1.0 + (w_mag_raw - 1.0) * basu_gate

        # ---- 4. Asymmetric modulation ----
        # s_neg ≈ 1 when y_pred < y_true (under-prediction), ≈ 0 otherwise
        s_neg = torch.sigmoid(-self.kappa * e)
        true_mag_ratio = torch.abs(y_true) / (1.0 + torch.abs(y_true))
        w_asym = 1.0 + self.beta * s_neg * true_mag_ratio

        # ---- Combined pointwise loss ----
        loss_pointwise = (w_mag * base_loss * w_asym).mean()

        # ---- 5. Magnitude-weighted temporal gradient term ----
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            # First-order: match step-to-step dynamics
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            e_grad = diff_pred - diff_true

            # Weight by max magnitude of adjacent true values —
            # conflict transitions get amplified, peaceful noise suppressed
            mag_grad = torch.max(
                torch.abs(y_true[:, 1:]), torch.abs(y_true[:, :-1])
            )
            w_grad_raw = torch.cosh(self.alpha * mag_grad).clamp(max=_W_MAX)
            scale_grad = 1.0 + mag_grad / (1.0 + mag_grad)

            # Directional Basu DPD gate on temporal weight:
            # e_grad > 0 = predicted dynamics overshoot truth → full weight
            # e_grad < 0 = predicted dynamics undershoot truth → dampened
            z_grad = e_grad / scale_grad
            z_grad_under = z_grad.clamp(max=0.0)
            basu_gate_grad = torch.exp(-0.5 * self.alpha * z_grad_under * z_grad_under)
            w_grad = 1.0 + (w_grad_raw - 1.0) * basu_gate_grad

            loss_grad_1 = (w_grad * self._log_cosh_scaled(e_grad, scale_grad)).mean()

            loss_grad = self.gamma * loss_grad_1
        else:
            loss_grad = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        total_loss = loss_pointwise + loss_grad

        # --- Output sanity check ---
        if torch.isnan(total_loss).any():
            raise NumericalSanityError(
                "Numerical Sanity Violation: NaN detected in computed SpotlightLoss."
            )

        logger.debug(
            "SpotlightLoss | pointwise=%.6f  grad=%.6f  total=%.6f",
            loss_pointwise.item(),
            loss_grad.item() if isinstance(loss_grad, torch.Tensor) else loss_grad,
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLoss(alpha={self.alpha}, beta={self.beta}, "
            f"kappa={self.kappa}, gamma={self.gamma})"
        )