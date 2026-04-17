import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Magnitude-aware temporal loss for imbalanced time-series regression
    on asinh-transformed targets.

    Designed for conflict forecasting where targets are zero-inflated (~90%),
    heavy-tailed, and subject to exponential amplification by the sinh()
    inverse transform.

    Components
    ----------
    1. **Batch-normalised log-cosh magnitude weighting** —
       ``w_mag = (1 + log_cosh(alpha * |y_true|)) / mean(...)``.  
       log(cosh(x)) grows quadratically near zero and linearly for large x,
       so conflict cells receive higher weight than peace cells without the
       exponential blow-up of raw cosh.  Per-batch normalisation converts
       weights into relative ratios with mean 1.0.  This avoids two failure
       modes: (a) without normalisation a single extreme cell dominates the
       gradient, and (b) with raw cosh even *after* normalisation the ratio
       is so extreme (~50× at α=0.5 for Ukraine) that one cell still
       captures ~40 % of batch loss.  Log-cosh weighting keeps the ratio
       bounded (~12× at α=1.0) with no hardcoded clamp.

    2. **Scaled log-cosh base loss** — parameter-free replacement for Huber.
       ``L_i = s_i * log(cosh(e_i / s_i))`` where
       ``s_i = 1 + |y_true| / (1 + |y_true|)`` ∈ [1, 2).
       Quadratic for small errors, linear for large, gradient bounded ±1.
       The scale widens the quadratic zone for conflict cells (up to |e|≈2)
       while keeping it tight for peace cells (|e|≈1).

    3. **Bounded symmetric error amplification** (tanh²-based) —
       ``w_amp = 1 + beta * tanh(e)² * mag_ratio``
       where ``mag_ratio = |y_true| / (1 + |y_true|)``.

       ``tanh(e)²`` is smooth, symmetric, and **bounded in [0, 1]**, so the
       amplifier saturates for large errors instead of growing linearly.
       This preserves the linear tail of the log-cosh base loss — the
       previous SiLU sum (≈ |e| for large e) multiplied the linear base
       to give quadratic total growth, partially undoing robustness.

       Properties:
       - ``w_amp(0) = 1`` exactly — no overlap with magnitude weight at minimum.
       - ``w_amp'(0) = 0`` — **no directional bias** at the loss minimum.
       - Symmetric: over- and under-prediction penalised equally.
       - Bounded: ``w_amp ∈ [1, 1 + beta * mag_ratio]``.
       - Inactive on peaceful targets (mag_ratio ≈ 0).

    4. **Temporal gradient term** — optional scaled log-cosh on first-order
       differences ``Δy_pred − Δy_true``, with the same batch-normalised
       magnitude weights.  Encourages the model to track step-to-step
       dynamics (seasonality) rather than just pointwise levels.

    Parameters
    ----------
    alpha : float
        Magnitude amplification rate for ``log_cosh(alpha * |y_true|)``.
        Controls the conflict:peace weight ratio *before* batch normalisation.
        Log-cosh grows linearly for large arguments, so higher alpha values
        are safe (no exponential blow-up).  Recommended: 0.5–2.0.
    beta : float
        Bounded symmetric error amplification strength.  The maximum
        amplification on a conflict cell is ``1 + beta * mag_ratio``.
        Set to 0 to disable.  Recommended: 0.0–1.5.
    gamma : float
        Weight for the temporal gradient regularisation term. 0 disables.

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
    >>> loss_fn = SpotlightLoss(alpha=1.0, beta=0.8, gamma=0.1)
    >>> pred = torch.randn(32, 36)
    >>> true = torch.randn(32, 36)
    >>> loss = loss_fn(pred, true)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        logger.info(
            "SpotlightLoss initialised | alpha=%.4f  beta=%.4f  gamma=%.4f",
            self.alpha,
            self.beta,
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

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable ``log(cosh(x))``.

        Uses identity: ``log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)``.
        Grows quadratically near 0, linearly for large |x|.  Never overflows.
        """
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - 0.6931471805599453

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
        abs_y = torch.abs(y_true)

        # ---- 1. Batch-normalised log-cosh magnitude weight ----
        # 1 + log(cosh(alpha * |y_true|)) grows linearly for large |y|,
        # so no clamp is needed.  Per-batch normalisation converts to
        # relative ratios with mean 1.0.
        w_mag = 1.0 + self._log_cosh(self.alpha * abs_y)
        w_mag = w_mag / (w_mag.mean() + 1e-8)

        # ---- 2. Scaled log-cosh base loss ----
        # s_i ∈ [1, 2): peace cells ≈ 1 (tight quadratic), conflict → 2 (wider)
        scale = 1.0 + abs_y / (1.0 + abs_y)
        base_loss = self._log_cosh_scaled(e, scale)

        # ---- 3. Bounded symmetric error amplification (tanh²) ----
        # tanh(e)² ∈ [0, 1]: zero at e=0, zero derivative at e=0,
        # symmetric, saturates to 1 for large |e|.
        # Total amplification bounded: w_amp ∈ [1, 1 + beta * mag_ratio].
        # Preserves the linear tail of log-cosh (SiLU sum grew ~|e|,
        # giving quadratic total loss and undoing robustness).
        if self.beta > 0.0:
            mag_ratio = abs_y / (1.0 + abs_y)
            error_amplifier = torch.tanh(e).square()
            w_amp = 1.0 + self.beta * error_amplifier * mag_ratio
        else:
            w_amp = 1.0

        # ---- Combined pointwise loss ----
        loss_pointwise = (w_mag * base_loss * w_amp).mean()

        # ---- 4. Magnitude-weighted temporal gradient term ----
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            e_grad = diff_pred - diff_true

            mag_grad = torch.max(
                torch.abs(y_true[:, 1:]), torch.abs(y_true[:, :-1])
            )
            w_grad = 1.0 + self._log_cosh(self.alpha * mag_grad)
            w_grad = w_grad / (w_grad.mean() + 1e-8)

            scale_grad = 1.0 + mag_grad / (1.0 + mag_grad)
            loss_grad = self.gamma * (w_grad * self._log_cosh_scaled(e_grad, scale_grad)).mean()
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
            f"SpotlightLoss(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"
        )