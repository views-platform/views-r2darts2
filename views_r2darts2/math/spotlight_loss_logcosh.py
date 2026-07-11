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

    * **Level (aggregate magnitude).** ``T × logcosh(mean(pred) − mean(true))``
      per series. This is a SCALAR loss (one value per series per channel)
      that acts as a UNIFORM bias shift across all timesteps:

        gradient per cell = T × tanh(gap) × (1/T) = tanh(gap)

      For a typical gap of 2 asinh units: tanh(2) ≈ 0.96 per cell —
      comparable to the shape gradient (tanh ≈ 0.5‑1.0). The T factor
      cancels the 1/T dilution from the mean operator, so the level
      gradient is NOT diluted.

      Because the gradient is the SAME for every cell in the series,
      it acts as a pure DC offset — it shifts the entire prediction up
      or down without distorting the temporal pattern. This avoids the
      conflict with the shape term (which adjusts individual cells)
      and prevents the oscillation / collapse seen with per‑cell level
      losses where up‑pushes and down‑pushes cancel out.

      Stability: the mean aggregates T=36 timesteps (√T noise
      reduction), and tanh bounds the per‑cell gradient at ±1. Total
      gradient for 36 cells: 36 × 1.0 = 36 (within clip=100). No
      explosion because the mean (unlike the sum) stays in the
      single‑digit range where tanh is well‑behaved.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
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

        # ── Level: T × logcosh(mean gap) — uniform DC bias shift ────
        # The gradient is tanh(gap) per cell — uniform across all
        # timesteps. This shifts the entire prediction up/down as a
        # block without distorting the shape. No per-cell up/down
        # oscillation. No Hájek dilution (T cancels the 1/T from mean).
        # Total gradient: 36 × tanh(gap) ≤ 36 (within clip=100).
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * self._Logcosh(gap)

        # ── Per-series event weight for level ────────────────────────
        w = gate.amax(dim=1)  # (B,) or (B, C)

        # ── Combine ──────────────────────────────────────────────────
        # Shape: Hájek (sum/sum) — gradient ~0.5-1.0 per cell
        # Level: mean over batch, sum over channels — gradient = tanh(gap)
        #   per cell (T cancels 1/T from mean). mean(dim=0) prevents the
        #   5.47× overprediction seen with SUM (which was B×C×tanh = 384).
        #   sum() over channels keeps C×tanh = 3.0 per cell — 3× shape,
        #   strong enough to correct level but not overshoot.
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = (w * level_cell).mean(dim=0)  # mean over batch (not SUM)

            per_channel = shape + level
            total_loss = per_channel.sum()  # sum over channels (not mean)
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
