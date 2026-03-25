import torch
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Symmetric magnitude-aware temporal Huber loss for imbalanced time-series regression
    on asinh-transformed targets.

    SpotlightLoss "shines a spotlight" on the signal that matters most: it amplifies
    loss contributions from high-magnitude events (via a cosh-based magnitude weight),
    applies a symmetric magnitude-gated penalty in conflict zones, and optionally
    regularises the temporal gradient so the model tracks the *shape* of the true
    series, not just its level.

    Components
    ----------
    1. **Magnitude weighting** — ``w_mag = cosh(alpha * |y_true|)``,
       clamped to ``1e6`` for numerical safety. High-magnitude targets receive
       exponentially higher weight based solely on ground truth — predictions
       cannot inflate their own importance. No per-batch normalisation is
       applied; gradient clipping handles cross-batch magnitude stability.
    2. **Huber base loss** — standard Huber/smooth-L1 with configurable ``delta``,
       combining MSE sensitivity near zero with linear-regime robustness for outliers.
    3. **Symmetric conflict-zone amplification** — ``w_amp = 1 + beta * mag_ratio``
       where ``mag_ratio = |y_true| / (1 + |y_true|)``.  Both over- and
       under-prediction receive the same extra penalty, scaled by how much
       conflict signal exists in the true target.  Saturates at ``1 + beta``
       for high-magnitude targets.
    4. **Magnitude-weighted temporal gradient** — Huber loss on first-order
       (velocity) and second-order (curvature) differences, weighted by the
       same cosh magnitude scheme so that dynamics during conflict matter
       more than noise during peace.  Second-order matching penalises
       onset/offset shape errors (acceleration), at half weight since
       curvature is inherently noisier.

    Parameters
    ----------
    alpha : float
        Magnitude amplification strength. Larger values increase the weight gap between
        high-magnitude and near-zero samples.
    beta : float
        Symmetric amplification strength. Maximum extra multiplier applied to
        errors on conflict-zone targets (approaches ``1 + beta`` for high-magnitude targets).
    delta : float
        Huber loss threshold. Errors below ``delta`` are penalised quadratically;
        above ``delta``, linearly.
    gamma : float
        Weight for the temporal gradient regularisation term. Set to 0 to disable.

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
    >>> loss_fn = SpotlightLoss(alpha=1.0, beta=1.5, delta=1.0, gamma=0.1)
    >>> pred = torch.randn(32, 36)
    >>> true = torch.randn(32, 36)
    >>> loss = loss_fn(pred, true)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        delta: float,
        gamma: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma

        logger.info(
            "SpotlightLoss initialised | alpha=%.4f  beta=%.4f  "
            "delta=%.4f  gamma=%.4f",
            self.alpha,
            self.beta,
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
        # Ground-truth only: conflict samples get exponentially higher weight.
        # Using only y_true avoids a feedback loop where overshooting
        # predictions inflate their own weight via detached-max, creating
        # runaway gradient amplification during training.
        w_mag = torch.cosh(self.alpha * torch.abs(y_true)).clamp(max=1e6)

        # ---- 2. Base Huber loss ----
        huber = self._huber(e)

        # ---- 3. Symmetric conflict-zone amplification ----
        # Both over- and under-prediction receive the same extra penalty,
        # scaled by how much conflict signal exists in the true target.
        # Saturates at (1 + beta) for high-magnitude targets.
        true_mag_ratio = torch.abs(y_true) / (1.0 + torch.abs(y_true))
        w_amp = 1.0 + self.beta * true_mag_ratio

        # ---- Combined pointwise loss ----
        loss_pointwise = (w_mag * huber * w_amp).mean()

        # ---- 4. Magnitude-weighted temporal gradient ----
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            # First-order: match step-to-step dynamics
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            e_grad = diff_pred - diff_true

            # Weight by max magnitude of adjacent true values —
            # transitions into/out of conflict get amplified
            mag_grad = torch.max(
                torch.abs(y_true[:, 1:]), torch.abs(y_true[:, :-1])
            )
            w_mag_grad = torch.cosh(self.alpha * mag_grad).clamp(max=1e6)

            loss_grad_1 = (w_mag_grad * self._huber(e_grad)).mean()

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
                w_mag_curv = torch.cosh(self.alpha * mag_curv).clamp(max=1e6)

                loss_grad_2 = 0.5 * (w_mag_curv * self._huber(e_curv)).mean()
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
            f"delta={self.delta}, gamma={self.gamma})"
        )
