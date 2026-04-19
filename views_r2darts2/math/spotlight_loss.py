import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Magnitude-aware temporal loss for zero-inflated conflict time-series.

    Designed for asinh-transformed targets (UCDP GED state-based conflict),
    where ~90% of cells are structural zeros and a small minority of high-
    magnitude conflict events carry the forecasting signal. Standard mean
    losses collapse to flat-line predictions; this loss prevents that via
    three coordinated mechanisms.

    Mechanism 1 — Truth-only inverse-density weight:
        Each cell is weighted by cosh(alpha * |y|), a parametric approximation
        of inverse label density (Yang et al. 2021, "Delving into Deep
        Imbalanced Regression"). Zeros are most frequent → weight 1.0×.
        High conflict is rare → weight up to ~16× at alpha=0.3, y=11.5.

            w(y) = cosh(alpha * |y|)

        At alpha=0.3 in asinh space:
            y=0  → 1.0×    (peace, ~90% of cells)
            y=5  → 2.4×    (moderate conflict)
            y=10 → 10.1×   (high conflict)

        Truth-only: no prediction-side weight. The gradient is:
            ∂L/∂ŷ = w(y) × tanh(e/s)
        which has a clean separation of concerns: w(y) controls *importance*
        (how much to care about this cell), tanh(e/s) controls *direction*
        (which way to push ŷ). OOD gradient magnitude is bounded by
        w(y) × 1.0 (tanh saturation) and further controlled by gradient
        clipping and the adaptive LR (RAdam/AdamW).

        No pred-side weight avoids:
          - The "shock absorber" interaction with gradient clipping (two
            mechanisms fighting over the same concern)
          - Detach semantics confusion (detached pred weight still scales
            gradient magnitude as a first-order multiplier)
          - Non-detached pred weight creating zero-attracting bias (path 2
            gradient always pushes ŷ toward zero via ∂w/∂ŷ)

        cosh is safe on truth because the data distribution is bounded
        (asinh space max ≈ 11.5 → cosh(3.45) ≈ 15.8 at alpha=0.3).

    Mechanism 2 — Balanced dual-mean aggregation:
        Per-cell weighted losses are aggregated with equal 50/50 weight
        across event (|y| > threshold) and peace (|y| ≤ threshold) groups,
        regardless of their relative frequency. This is the continuous
        regression analogue of class-balanced loss (Cui et al. 2019).

        If one group is absent in a batch, the present group receives 100%
        weight. Balancing is computed per target channel for multi-target
        scenarios. Isolated spikes are naturally diluted: one spike among
        50 event cells contributes ~2% of the event budget.

    Mechanism 3 — Truth-masked TV smoothness regularizer:
        When gamma > 0, a total variation (TV) penalty (Rudin, Osher &
        Fatemi 1992) is applied to the prediction's first differences,
        but only at time steps where the truth is smooth:

            L_smooth = gamma × mean[ 𝟙[|Δy| < θ] × (Δŷ)² ]

        This says: "where the truth doesn't change much, the prediction
        must not change much either." Where the truth has sharp transitions
        (onsets/cessations), the penalty is masked out so the model is
        free to produce large steps.

        Advantages over weighted first-difference reconstruction:
          - No truth-based weighting → no plateau blind spot (velocity
            weight gave plateaus 1× vs 16× for levels)
          - No balanced mean needed → no event/peace misclassification
          - No scale parameter needed → no saturation issues
          - No phase sensitivity → shifted onsets aren't double-penalized
          - One line of math, no hyperparameters beyond gamma and threshold

    Base loss — Scaled log-cosh:
        The per-cell base loss uses a scaled log-cosh::

            L(e, s) = s × log(cosh(e / s)),   s = 1 + |y| / (1 + |y|)

        Quadratic for |e| << s, linear for |e| >> s. The gradient tanh(e/s)
        is hard-bounded in (-1, 1), providing natural gradient clipping
        without a discontinuous Huber transition. The adaptive scale s ∈
        [1, 2) widens the quadratic region for larger targets.

    Args:
        alpha (float): Scale for the truth-only cosh weight. Controls how
            aggressively high-magnitude conflict cells are up-weighted
            relative to peace. Recommended range: 0.2–0.5. At alpha=0.3,
            max UCDP (asinh≈11.5) gets cosh(3.45) ≈ 16× weight.
        gamma (float): Weight for the TV smoothness regularizer. Set to 0.0
            to disable. Higher values enforce smoother predictions where the
            truth is smooth. Recommended range: 0.1–1.0.
        non_zero_threshold (float): asinh-space threshold separating "event"
            from "peace" cells. Also used to mask the TV penalty: steps where
            |Δy_true| ≥ threshold are excluded (onset/cessation). 0.88 ≈
            asinh(1), corresponding to ≥1 battle-related death.

    Example:
        >>> loss_fn = SpotlightLoss(alpha=0.3, gamma=0.5, non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        non_zero_threshold: float,
    ):
        if alpha <= 0.0:
            raise ValueError(f"SpotlightLoss: alpha must be positive, got {alpha}")
        if alpha > 0.7:
            logger.warning(
                "SpotlightLoss: alpha=%.4f > 0.7. cosh weight at max UCDP "
                "(asinh≈11.5) = cosh(%.2f) ≈ %.0f×. May cause instability.",
                alpha, alpha * 11.5, math.cosh(min(alpha * 11.5, 88.0)),
            )
        if gamma < 0.0:
            raise ValueError(f"SpotlightLoss: gamma must be non-negative, got {gamma}")
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"SpotlightLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "SpotlightLoss | alpha=%.4f gamma=%.4f threshold=%.4f",
            alpha, gamma, non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh_scaled(error: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Scaled log-cosh: s × log(cosh(e / s)).

        Quadratic for |e| << s, linear for |e| >> s.
        Gradient = tanh(e/s), hard-bounded in (-1, 1).

        Numerically stable: log(cosh(x)) = |x| + softplus(-2|x|) - ln2.
        """
        z = error / scale
        abs_z = torch.abs(z)
        return scale * (abs_z + F.softplus(-2.0 * abs_z) - math.log(2.0))

    def _balanced_mean(
        self, per_sample: torch.Tensor, is_event: torch.Tensor
    ) -> torch.Tensor:
        """50/50 balanced average over event and peace cells.

        Each group contributes 50% if both present, 100% if only one is
        present. Computed per target channel for multi-target support.
        """
        if per_sample.dim() == 2:
            per_sample = per_sample.unsqueeze(-1)
            is_event = is_event.unsqueeze(-1)

        C = per_sample.size(-1)
        ps_flat = per_sample.reshape(-1, C)
        ie_flat = is_event.reshape(-1, C)

        n_event = ie_flat.sum(0)
        n_peace = (~ie_flat).sum(0)

        loss_event = (ps_flat * ie_flat).sum(0) / n_event.clamp(min=1)
        loss_peace = (ps_flat * ~ie_flat).sum(0) / n_peace.clamp(min=1)

        w_e = 0.5 * (n_event > 0).float()
        w_p = 0.5 * (n_peace > 0).float()
        total_w = (w_e + w_p).clamp(min=1e-8)

        return ((w_e / total_w) * loss_event + (w_p / total_w) * loss_peace).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        e = y_pred - y_true
        abs_y = torch.abs(y_true)

        # ---- 1. Truth-only inverse-density weight ----
        w = torch.cosh(self.alpha * abs_y)

        # ---- 2. Pointwise loss ----
        scale = 1.0 + abs_y / (1.0 + abs_y)
        base_loss = self._log_cosh_scaled(e, scale)
        per_sample = w * base_loss

        is_event = abs_y > self.non_zero_threshold
        loss_pointwise = self._balanced_mean(per_sample, is_event)

        # ---- 3. TV smoothness regularizer ----
        loss_smooth = y_pred.new_tensor(0.0)
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]

            smooth_mask = (torch.abs(diff_true) < self.non_zero_threshold).float()
            tv = smooth_mask * diff_pred.square()
            loss_smooth = tv.mean()

        total_loss = loss_pointwise + self.gamma * loss_smooth

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLoss: pointwise={loss_pointwise.item():.6f} "
                f"smooth={loss_smooth.item():.6f}"
            )

        logger.debug(
            "SpotlightLoss | pw=%.6f tv=%.6f total=%.6f",
            loss_pointwise.item(),
            loss_smooth.item(),
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLoss(alpha={self.alpha}, gamma={self.gamma}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )
