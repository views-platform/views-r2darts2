import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Asymmetric magnitude-aware temporal loss for imbalanced time-series regression
    on asinh-transformed targets.

    SpotlightLoss "shines a spotlight" on the signal that matters most: it amplifies
    loss contributions from high-magnitude events (via a cosh-based magnitude weight),
    penalises under-prediction more harshly than over-prediction (via a smooth sigmoid
    asymmetry gate), and optionally regularises the temporal gradient so the model
    tracks the *shape* of the true series, not just its level.

    Components
    ----------
    1. **Magnitude weighting** — ``w_mag = cosh(alpha * |y_true|)``,
       clamped to ``1e6`` for numerical safety. High-magnitude targets receive
       exponentially higher weight based solely on ground truth — predictions
       cannot inflate their own importance.
    2. **Scaled log-cosh base loss** — parameter-free replacement for Huber.
       ``L_i = s_i * log(cosh(e_i / s_i))`` where ``s_i = 1 + |y_true| / (1 + |y_true|)``.
       Behaves quadratically for small errors, linearly for large errors, with a smooth
       transition (no curvature discontinuity). Gradient bounded at ±1 always.
       The scale ``s_i`` widens the quadratic zone for conflict cells (up to |e|≈2)
       while keeping it tight for peace cells (|e|≈1), preventing sweep
       misconfiguration that previously caused 1B+ OOD blowups.
    3. **Asymmetric modulation** — a sigmoid ``sigma(-kappa * e)`` activates when the
       model under-predicts (``y_pred < y_true``). The extra penalty scales with
       ``beta`` and is proportional to the *relative* magnitude of the true value,
       ``|y_true| / (1 + |y_true|)``, so asymmetry matters most for real events.
    4. **Temporal gradient term** — optional scaled log-cosh on first-order differences
       ``Delta y_pred - Delta y_true``, weighted by ``gamma``. Encourages the model to
       reproduce step-to-step dynamics, not just pointwise targets.

    Parameters
    ----------
    alpha : float, default 1.0
        Magnitude amplification strength. Larger values increase the weight gap between
        high-magnitude and near-zero samples.
    beta : float, default 1.0
        Asymmetry strength. Maximum extra multiplier applied when the model
        under-predicts a non-zero true value.
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
        # Ground-truth only: conflict samples get exponentially higher weight.
        # Using only y_true prevents a feedback loop where overshooting
        # predictions inflate their own weight (via detached-max), which
        # caused runaway OOD blowups during out-of-sample forecasting.
        # mag = torch.max(torch.abs(y_true), torch.abs(y_pred.detach())) # test
        # w_mag = torch.cosh(self.alpha * mag).clamp(max=1e6)
        w_mag = torch.cosh(self.alpha * torch.abs(y_true)).clamp(max=1e6)

        # ---- 2. Scaled log-cosh base loss ----
        # s_i ∈ [1, 2): peace cells ≈ 1 (tight quadratic), conflict → 2 (wider)
        scale = 1.0 + torch.abs(y_true) / (1.0 + torch.abs(y_true))
        base_loss = self._log_cosh_scaled(e, scale)

        # ---- 3. Asymmetric modulation ----
        # s_neg ≈ 1 when y_pred < y_true (under-prediction), ≈ 0 otherwise
        s_neg = torch.sigmoid(-self.kappa * e)
        true_mag_ratio = torch.abs(y_true) / (1.0 + torch.abs(y_true))
        w_asym = 1.0 + self.beta * s_neg * true_mag_ratio

        # ---- Combined pointwise loss ----
        loss_pointwise = (w_mag * base_loss * w_asym).mean()

        # ---- 4. Magnitude-weighted temporal gradient term ----
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
            w_grad = torch.cosh(self.alpha * mag_grad).clamp(max=1e6)
            scale_grad = 1.0 + mag_grad / (1.0 + mag_grad)
            loss_grad_1 = (w_grad * self._log_cosh_scaled(e_grad, scale_grad)).mean()

            # Second-order: match curvature (onset/offset shape)
            if y_pred.size(1) > 2:
                curv_pred = diff_pred[:, 1:] - diff_pred[:, :-1]
                curv_true = diff_true[:, 1:] - diff_true[:, :-1]
                e_curv = curv_pred - curv_true

                mag_curv = torch.max(
                    torch.abs(y_true[:, 2:]),
                    torch.max(
                        torch.abs(y_true[:, 1:-1]),
                        torch.abs(y_true[:, :-2]),
                    ),
                )
                w_curv = torch.cosh(self.alpha * mag_curv).clamp(max=1e6)
                scale_curv = 1.0 + mag_curv / (1.0 + mag_curv)
                loss_grad_2 = 0.5 * (w_curv * self._log_cosh_scaled(e_curv, scale_curv)).mean()
            else:
                loss_grad_2 = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

            loss_grad = self.gamma * (loss_grad_1 + loss_grad_2)
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