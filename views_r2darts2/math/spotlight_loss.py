import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Asymmetric magnitude-aware temporal Huber loss for imbalanced time-series regression
    on asinh-transformed targets.

    SpotlightLoss "shines a spotlight" on the signal that matters most: it amplifies
    loss contributions from high-magnitude events (via a cosh-based magnitude weight),
    penalises under-prediction more harshly than over-prediction (via a smooth sigmoid
    asymmetry gate), and optionally regularises the temporal gradient so the model
    tracks the *shape* of the true series, not just its level.

    Components
    ----------
    1. **Magnitude weighting** — ``w_mag = cosh(alpha * max(|y_true|, |y_pred|))``,
       clamped to ``1e6`` for numerical safety. Events with large absolute values in
       either the target or the prediction receive exponentially higher weight.
    2. **Huber base loss** — standard Huber/smooth-L1 with configurable ``delta``,
       combining MSE sensitivity near zero with linear-regime robustness for outliers.
    3. **Asymmetric modulation** — a sigmoid ``sigma(-kappa * e)`` activates when the
       model under-predicts (``y_pred < y_true``). The extra penalty scales with
       ``beta`` and is proportional to the *relative* magnitude of the true value,
       ``|y_true| / (1 + |y_true|)``, so asymmetry matters most for real events.
    4. **Temporal gradient term** — optional Huber loss on first-order differences
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
    delta : float, default 1.0
        Huber loss threshold. Errors below ``delta`` are penalised quadratically;
        above ``delta``, linearly.
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
    >>> loss_fn = SpotlightLoss(alpha=1.0, beta=1.5, kappa=10.0, delta=1.0, gamma=0.1)
    >>> pred = torch.randn(32, 36)
    >>> true = torch.randn(32, 36)
    >>> loss = loss_fn(pred, true)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        delta: float,
        gamma: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.delta = delta
        self.gamma = gamma

        logger.info(
            "SpotlightLoss initialised | alpha=%.4f  beta=%.4f  kappa=%.4f  "
            "delta=%.4f  gamma=%.4f",
            self.alpha,
            self.beta,
            self.kappa,
            self.delta,
            self.gamma,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _huber(self, error: torch.Tensor) -> torch.Tensor:
        """Element-wise Huber loss with the instance's delta."""
        abs_e = torch.abs(error)
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_e - 0.5 * self.delta)
        return torch.where(abs_e <= self.delta, quadratic, linear)

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
        m = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        w_mag = torch.cosh(self.alpha * m).clamp(max=1e6)

        # ---- 2. Base Huber loss ----
        huber = self._huber(e)

        # ---- 3. Asymmetric modulation ----
        # s_neg ≈ 1 when y_pred < y_true (under-prediction), ≈ 0 otherwise
        s_neg = torch.sigmoid(-self.kappa * e)
        true_mag_ratio = torch.abs(y_true) / (1.0 + torch.abs(y_true))
        w_asym = 1.0 + self.beta * s_neg * true_mag_ratio

        # ---- Combined pointwise loss ----
        loss_pointwise = (w_mag * huber * w_asym).mean()

        # ---- 4. Temporal gradient term ----
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            e_grad = diff_pred - diff_true
            loss_grad = self.gamma * self._huber(e_grad).mean()
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
            f"kappa={self.kappa}, delta={self.delta}, gamma={self.gamma})"
        )
