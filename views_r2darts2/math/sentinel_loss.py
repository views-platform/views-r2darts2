import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SentinelLoss(torch.nn.Module):
    """
    Adaptive-robust bidirectional magnitude-aware temporal loss for imbalanced
    time-series regression on asinh-transformed targets.

    Designed for conflict forecasting where targets are zero-inflated (~90%),
    heavy-tailed, and subject to exponential amplification by the sinh()
    inverse transform.

    Components
    ----------
    1. **Adaptive-robust base loss** (generalised Charbonnier):

           L(e) = (delta^2 / alpha) * [(1 + e^2/delta^2)^(alpha/2) - 1]

       ``alpha=2`` -> MSE.  ``alpha=1`` -> pseudo-Huber / Charbonnier.
       ``alpha < 1`` -> Cauchy-like, saturating.
       Always quadratic at e=0 (no dead zones).  Gradient bounded for
       alpha < 2: gradient ~ |e|^(alpha-1) for large |e|.

       NOTE: at alpha=2, delta cancels entirely (L=e^2/2).  Avoid sweeping
       both simultaneously near alpha=2; fix delta and sweep alpha only.

    2. **Power-law magnitude weighting** — ``(1 + |y_true|)^kappa``,
       normalised per-batch.  Polynomial growth gives controllable
       conflict:peace weight ratios without the exponential instability
       of cosh.  Pair kappa with gradient_clip_val: higher kappa needs
       higher clip to preserve within-conflict proportionality.

       ====  ========================  ========================
       κ     conflict:peace ratio      suggested gradient_clip
       ====  ========================  ========================
       0.5   3:1                       5
       1.0   10:1                      5–10
       1.5   32:1                      10–15
       2.0   100:1                     15–25
       ====  ========================  ========================

    3. **Symmetric error amplification** — SiLU-based penalty for both
       over- and under-prediction, magnitude-gated to conflict zones:

           w_asym = 1 + beta * [SiLU(e) + SiLU(-e)] * mag_ratio

       where ``SiLU(x) = x * sigmoid(x)`` and
       ``mag_ratio = |y_true| / (1 + |y_true|)``.

       ``SiLU(e) + SiLU(-e) = e * (2*sigmoid(e) - 1)`` is a smooth,
       symmetric function that grows linearly as |e| for large errors
       in BOTH directions.

       Properties:
       - ``w_asym(0) = 1`` exactly (zero overlap with kappa).
       - ``w_asym'(0) = 0`` (no directional bias at the loss minimum).
       - ``w_asym(e) >= 1`` for all e.
       - Symmetric: both over- and under-prediction penalised equally.
       - Large errors amplified linearly (~|e|) on top of base loss.
       - Inactive on peaceful targets (mag_ratio ~ 0).

       kappa and beta are fully separated: kappa controls WHICH samples
       matter (magnitude weighting), beta controls HOW errors are treated
       directionally (asymmetry).  No interaction at e=0.

    4. **Magnitude-weighted temporal gradient** — first-difference loss on
       consecutive steps, weighted by the same power-law weights so
       dynamics during conflict matter more than noise during peace.

    Parameters
    ----------
    alpha : float
        Base loss shape.  ``alpha=2`` -> pure MSE (delta irrelevant).
        ``alpha=1`` -> Charbonnier / pseudo-Huber.  ``alpha < 1`` -> robust
        / Cauchy-like.
    beta : float
        Symmetric error amplification strength.  Amplifies loss for
        large errors in conflict zones equally in both directions.
        Set to 0 to disable.
    kappa : float
        Power-law magnitude weighting exponent.
    delta : float
        Base loss scale.  Controls quadratic-to-sublinear transition.
        Irrelevant when alpha=2.
    gamma : float
        Temporal gradient regularisation weight.  0 disables.

    Forward signature
    -----------------
    y_pred : torch.Tensor
        shape ``(batch, seq_len)`` or ``(batch, seq_len, 1)``.
    y_true : torch.Tensor
        same shape as y_pred.

    Returns
    -------
    torch.Tensor — scalar loss.
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
            "SentinelLoss initialised | alpha=%.4f  beta=%.4f  kappa=%.4f  "
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

    def _base_loss(self, error: torch.Tensor) -> torch.Tensor:
        """
        Generalised Charbonnier / adaptive robust loss.

        L(e) = (delta^2 / alpha) * [(1 + (e / delta)^2)^(alpha/2) - 1]
        """
        scaled_sq = (error / self.delta) ** 2
        if self.alpha < 0.01:
            return self.delta ** 2 * torch.log1p(scaled_sq)
        return (self.delta ** 2 / self.alpha) * (
            (1.0 + scaled_sq).pow(self.alpha / 2.0) - 1.0
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        from views_r2darts2.infrastructure.exceptions import NumericalSanityError

        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            raise NumericalSanityError(
                "Numerical Sanity Violation: NaN or Inf detected in predictions."
            )
        if torch.isnan(y_true).any() or torch.isinf(y_true).any():
            raise NumericalSanityError(
                "Numerical Sanity Violation: NaN or Inf detected in targets."
            )

        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        e = y_pred - y_true

        # ---- 1. Magnitude weight (power-law, ground-truth only) ----
        w_mag = (1.0 + torch.abs(y_true)).pow(self.kappa)
        w_mag = w_mag / (w_mag.mean() + 1e-8)

        # ---- 2. Base loss (adaptive robust) ----
        base = self._base_loss(e)

        # ---- 3. Symmetric error amplification (SiLU-based) ----
        # SiLU(e) + SiLU(-e) = e * (2*sigmoid(e) - 1):
        #   - Exactly 0 at e=0 (no kappa overlap)
        #   - Zero derivative at e=0 (no directional bias)
        #   - >= 0 everywhere
        #   - Symmetric: same penalty for over- and under-prediction
        #   - Linear growth ~|e| for large errors
        if self.beta > 0.0:
            mag_ratio = torch.abs(y_true) / (1.0 + torch.abs(y_true))
            error_amplifier = F.silu(e) + F.silu(-e)
            w_asym = 1.0 + self.beta * error_amplifier * mag_ratio
        else:
            w_asym = 1.0

        # ---- Combined pointwise loss ----
        loss_pointwise = (w_mag * base * w_asym).mean()

        # ---- 4. Magnitude-weighted temporal gradient ----
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            e_grad = diff_pred - diff_true

            mag_grad = torch.max(
                torch.abs(y_true[:, 1:]), torch.abs(y_true[:, :-1])
            )
            w_mag_grad = (1.0 + mag_grad).pow(self.kappa)
            w_mag_grad = w_mag_grad / (w_mag_grad.mean() + 1e-8)

            loss_grad = self.gamma * (w_mag_grad * self._base_loss(e_grad)).mean()
        else:
            loss_grad = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        total_loss = loss_pointwise + loss_grad

        if torch.isnan(total_loss).any():
            raise NumericalSanityError(
                "Numerical Sanity Violation: NaN detected in computed SentinelLoss."
            )

        logger.debug(
            "SentinelLoss | pointwise=%.6f  grad=%.6f  total=%.6f",
            loss_pointwise.item(),
            loss_grad.item() if isinstance(loss_grad, torch.Tensor) else loss_grad,
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SentinelLoss(alpha={self.alpha}, beta={self.beta}, "
            f"kappa={self.kappa}, delta={self.delta}, gamma={self.gamma})"
        )
