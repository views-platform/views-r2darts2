import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Scale-invariant asymmetric Huber loss for imbalanced time-series regression
    on asinh-transformed targets.

    SpotlightLoss operates on *relative* error — ``(y_pred - y_true) / (1 + |y_true|)``
    — so that all samples contribute on a common scale regardless of magnitude.
    This eliminates the gradient-concentration problem that magnitude weighting
    causes on highly skewed conflict data, while still learning conflict dynamics
    (a 20% error on Ukraine matters as much as a 20% error on a peaceful country).

    Components
    ----------
    1. **Scale normalisation** — error is divided by ``(1 + |y_true|)`` before
       any loss computation. High-magnitude targets no longer dominate the gradient
       through raw error size; instead every sample's loss reflects *relative* accuracy.
    2. **Optional magnitude boost** — ``w_mag = (1 + |y_true|) ** alpha``.
       At alpha=0 (recommended starting point) this is uniform. Positive alpha
       gives conflict events extra weight *on top of* scale-invariance.
    3. **Huber base loss** — standard Huber/smooth-L1 on the relative error.
       ``delta`` controls the quadratic-to-linear transition in relative-error space.
    4. **Asymmetric modulation** — sigmoid gate penalises under-prediction of
       non-zero events, operating on relative error so the asymmetry threshold
       adapts to each sample's scale.
    5. **Temporal gradient term** — optional Huber loss on scale-normalised
       first-order differences, weighted by ``gamma``.

    Parameters
    ----------
    alpha : float, default 0.0
        Power-law exponent for optional magnitude boost after normalisation.
        0 = pure scale-invariance, 0.5 = mild conflict upweight.
    beta : float, default 0.0
        Asymmetry strength on relative error. Extra penalty when under-predicting
        non-zero events.
    kappa : float, default 10.0
        Sigmoid sharpness for the asymmetric gate.
    delta : float, default 1.0
        Huber threshold in relative-error space. ``delta=1.0`` means the
        quadratic-to-linear transition happens at 100% relative error.
    gamma : float, default 0.0
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

        # ---- Scale normalisation ----
        # Divide error by target scale so all samples contribute proportionally.
        # Ukraine (|y|=10): error of 2 → relative error 0.18
        # Peaceful (|y|=0): error of 0.1 → relative error 0.10
        # Both now comparable — no gradient concentration.
        scale = 1.0 + torch.abs(y_true)
        e_rel = e / scale

        # ---- 1. Optional magnitude boost (power-law) ----
        # At alpha=0 this is 1.0 everywhere (pure scale-invariance).
        w_mag = scale.pow(self.alpha)

        # ---- 2. Base Huber loss on relative error ----
        huber = self._huber(e_rel)

        # ---- 3. Asymmetric modulation on relative error ----
        s_neg = torch.sigmoid(-self.kappa * e_rel)
        true_mag_ratio = torch.abs(y_true) / scale
        w_asym = 1.0 + self.beta * s_neg * true_mag_ratio

        # ---- Combined pointwise loss ----
        loss_pointwise = (w_mag * huber * w_asym).mean()

        # ---- 4. Temporal gradient term (also scale-normalised) ----
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            scale_grad = 1.0 + 0.5 * (torch.abs(y_true[:, 1:]) + torch.abs(y_true[:, :-1]))
            e_grad_rel = (diff_pred - diff_true) / scale_grad
            loss_grad = self.gamma * self._huber(e_grad_rel).mean()
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
