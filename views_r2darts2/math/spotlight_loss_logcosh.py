import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    Two terms with distinct roles:

    * **Shape (per‑cell pattern).** Demeaned logcosh residual, gated
      and Hájek‑normalised. logcosh provides bounded, stable gradients
      against noise and spikes. The demeaning ensures this term scores
      temporal dynamics only — it cannot absorb a magnitude error.

    * **Level (aggregate magnitude).** logcosh(mean(pred) − mean(true))
      per series, filtered to exclude pure-peace series (both means < τ).
      
      Logcosh is symmetric: both over and under prediction contribute
      equally to the loss. The gradient tanh(gap) is antisymmetric, so
      it always pushes predictions toward the true mean (no sign flip).
      
      Why filter peace series? In zero-inflated data, most series have
      mean ≈ 0. For these series, |gap| is small (usually < 0.5 asinh
      units), yet logcosh(gap) can still be nonzero. This wastes capacity
      on high-frequency noise. By ignoring series where BOTH y_true and
      y_pred are entirely below τ (no events in either), the level loss
      focuses gradient only on series with meaningful events — where
      calibration and magnitude actually matter.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    Used to define the event gate and to filter peace series.
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float):
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"non_zero_threshold must be positive, got {non_zero_threshold}"
            )
        super().__init__()
        self.non_zero_threshold = non_zero_threshold
        self._last_components: dict | None = None

        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _Logcosh(z: torch.Tensor) -> torch.Tensor:
        a = z.abs()
        return a + F.softplus(-2.0 * a)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold
        T = y_pred.size(1)

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))

        # ── Shape: demeaned logcosh, standardised, Hájek ────────────
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # ── Level: logcosh(mean gap), symmetric, peace-filtered ──────
        # Logcosh is symmetric: penalizes over and under equally.
        # Gradient: tanh(gap) is antisymmetric, so it always pushes toward truth.
        #
        # Filter: ignore series where BOTH y_true and y_pred are entirely
        # below non_zero_threshold (pure peace series). These contribute only
        # noise to the level signal; the shape term handles temporal dynamics.
        #
        # Peace mask: True where series should be ignored (both means < tau)
        series_mean_true = y_true.mean(dim=1)  # (B,) or (B, C)
        series_mean_pred = y_pred.mean(dim=1)
        peace_mask = (series_mean_true.abs() < tau) & (series_mean_pred.abs() < tau)

        gap = series_mean_pred - series_mean_true  # (B,) or (B, C)
        level_cell = self._Logcosh(gap)

        # ── Per-series event weight for level (zero out peace series) ──
        w = gate.amax(dim=1)  # (B,) or (B, C)
        w = w * (~peace_mask).float()  # Mask out peace series

        # ── Combine ──────────────────────────────────────────────────
        # Shape: Hájek (sum/sum) — gradient ~0.5-1.0 per cell
        # Level: SUM over batch, peace-filtered — symmetric logcosh gradient
        #   Peace-filtered reduces noise: no gradient from series where both
        #   y_true and y_pred are entirely below τ. This focuses gradient on
        #   the event series where magnitude calibration matters.
        #   SUM (not mean) preserves gradient scale across batches.
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = (w * level_cell).sum(dim=0)  # SUM over batch

            per_channel = shape + level
            total_loss = per_channel.sum()  # SUM over channels
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)
            level = (w * level_cell).sum()
            total_loss = shape + level
            shape_c = [float(shape.detach())]
            level_c = [float(level.detach())]
            comp = [float(total_loss.detach())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossLogcosh: per_channel={comp}")

        n = len(comp)
        self._last_components = {
            "shape": shape_c,
            "level": level_c,
            "spec": [0.0] * n,
            "weight": [1.0] * n,
            "ema": [float("nan")] * n,
            "cal_ratio": [1.0] * n,
            "cal_score": [1.0] * n,
            "gates": [1.0] * n,
            "contribution": comp,
        }

        logger.debug(
            "SpotlightLossLogcosh | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"
