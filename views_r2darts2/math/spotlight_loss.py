import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Magnitude-aware temporal loss for zero-inflated conflict time-series
    on asinh-transformed targets (UCDP GED sb).

    Two mechanisms prevent flat-line forecasts on dynamic series:

    1. **Cosh magnitude weighting** amplifies loss on high-fatality cells.
       At alpha=0.4, asinh(10000)≈9.9 receives ~27× the weight of a
       zero-fatality cell.  This provides implicit asymmetry: missing a
       conflict cell is penalised far more than a false alarm at a peace
       cell, because the weight is conditioned on y_true, not error sign.

    2. **Temporal dynamics term** penalises mismatches in first-order
       differences (escalation and de-escalation), weighted by the
       magnitude of the true change.  A flat forecast on a dynamic series
       produces large diff-errors at conflict transitions, amplified by
       cosh weighting on |Δy_true|.

    Both terms use **balanced dual-mean** averaging: equal-weighted
    average over event / non-event cells (thresholded at
    non_zero_threshold).  This guarantees the model receives meaningful
    conflict gradient even when 90%+ of the batch is zeros.

    No directional asymmetry (beta / kappa) is used.  The cosh magnitude
    weight already creates strong implicit asymmetry, and the balanced
    mean ensures conflict gradients are never diluted.  FN/FP cost
    trade-offs are better handled at the decision layer.

    RevIN-compatible: operates on denormalised asinh-scale predictions
    and targets.  No normalisation assumptions.

    Parameters
    ----------
    alpha : float
        Rate for cosh magnitude weight cosh(alpha * |y|).
        At alpha=0.4: y=0 → w=1, y=5 → w≈3.8, y=10 → w≈27.
        Recommended range 0.2–0.5.
    gamma : float
        Weight for the temporal dynamics term.  Set to 0 to disable.
        Default 1.0 gives dynamics equal footing with pointwise loss
        after both are balanced-mean normalised.  Reduce to 0.5 if
        dynamics tracking dominates pointwise accuracy during tuning.
    non_zero_threshold : float
        asinh threshold separating "event" from "peace" cells for
        balanced averaging.  0.88 ≈ asinh(1), i.e. ≥1 battle-related
        death.  Applied to |y_true| for pointwise and |Δy_true| for
        dynamics.
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        non_zero_threshold: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "SpotlightLoss | alpha=%.4f gamma=%.4f threshold=%.4f",
            self.alpha,
            self.gamma,
            self.non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh_scaled(
        error: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Scaled log-cosh: s * log(cosh(e / s)).

        Quadratic for |e| << s, linear for |e| >> s, smooth transition.
        Gradient = tanh(e/s), hard-bounded in (-1, 1).

        Numerically stable via:
            log(cosh(x)) = |x| + softplus(-2|x|) - ln2
        """
        z = error / scale
        abs_z = torch.abs(z)
        return scale * (abs_z + F.softplus(-2.0 * abs_z) - math.log(2.0))

    def _balanced_mean(
        self, per_sample: torch.Tensor, is_event: torch.Tensor
    ) -> torch.Tensor:
        """Equal-weighted average over event and non-event cells.

        Prevents the majority class (peace) from diluting the minority
        class (conflict) gradient.  Each group contributes 50% of the
        loss regardless of its frequency in the batch.
        """
        n_event = is_event.sum().clamp(min=1)
        n_peace = (~is_event).sum().clamp(min=1)
        loss_event = (per_sample * is_event).sum() / n_event
        loss_peace = (per_sample * ~is_event).sum() / n_peace
        return 0.5 * loss_event + 0.5 * loss_peace

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : (batch, seq_len) or (batch, seq_len, 1)
        y_true : same shape as y_pred

        Returns
        -------
        Scalar loss.
        """
        # Squeeze trailing singleton (Darts convention)
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        e = y_pred - y_true
        abs_y = torch.abs(y_true)

        # ---- Pointwise loss ----
        # cosh magnitude weight: no clamp, no Basu gate.
        # For alpha=0.4 and max realistic asinh≈10, cosh(4)≈27 —
        # well within float32 range.  Only problematic for alpha>1
        # or asinh>20, both outside the operating envelope.
        w_mag = torch.cosh(self.alpha * abs_y)

        # Scale widens quadratic zone for conflict cells:
        # peace (abs_y≈0) → scale≈1, conflict (abs_y≫0) → scale→2
        scale = 1.0 + abs_y / (1.0 + abs_y)

        base_loss = self._log_cosh_scaled(e, scale)
        per_sample_pw = w_mag * base_loss

        is_event = abs_y > self.non_zero_threshold
        loss_pointwise = self._balanced_mean(per_sample_pw, is_event)

        # ---- Temporal dynamics loss ----
        # Penalises mismatched first-order differences: escalation,
        # de-escalation, and flat forecasts on dynamic series.
        loss_dynamics = y_pred.new_tensor(0.0)
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            e_grad = diff_pred - diff_true

            abs_diff_true = torch.abs(diff_true)

            # Weight by magnitude of true change: large transitions
            # (escalation/de-escalation) get amplified.
            w_dyn = torch.cosh(self.alpha * abs_diff_true)
            scale_dyn = 1.0 + abs_diff_true / (1.0 + abs_diff_true)

            per_sample_dyn = w_dyn * self._log_cosh_scaled(e_grad, scale_dyn)

            is_dynamic = abs_diff_true > self.non_zero_threshold
            loss_dynamics = self._balanced_mean(per_sample_dyn, is_dynamic)

        total_loss = loss_pointwise + self.gamma * loss_dynamics

        if torch.isnan(total_loss):
            raise RuntimeError(
                "NaN in SpotlightLoss. "
                f"pointwise={loss_pointwise.item():.6f} "
                f"dynamics={loss_dynamics.item():.6f}"
            )

        logger.debug(
            "SpotlightLoss | pw=%.6f dyn=%.6f total=%.6f",
            loss_pointwise.item(),
            loss_dynamics.item(),
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLoss(alpha={self.alpha}, gamma={self.gamma}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )