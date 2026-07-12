import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (per‑cell pattern).** Demeaned logcosh residual, gated
      and Hájek‑normalised. Scores temporal dynamics only.

    * **Level (per‑cell magnitude).** Per‑cell logcosh on the raw error
      (``pred − true``), weighted by ``gate × log1p(abs_max) × focal``,
      summed, and NORMALISED BY EVENT COUNT per series.

      The **focal** weight ``(1 − exp(−|raw_error|))`` down‑weights
      well‑fit cells (small error → focal ≈ 0) and preserves full
      gradient on large errors (spikes/dips → focal ≈ 1). This turns
      the level gradient from a flat push (``tanh ≈ 1`` for all events)
      into a **variable push** that tracks the error magnitude:
      cells that are most wrong get the strongest correction.

      This is fundamentally different from DRO:
      - Focal is a **per‑cell** scaling (0 to 1), no series‑level mean
      - No mean computation → no peace‑noise amplification at 90% zeros
      - Does NOT change relative weights within a series (each cell
        scaled independently by its own error)
      - Shape Hájek is completely unaffected (focal is on level only)

      The result: the model is incentivised to produce **variable**
      forecasts (high at true peaks, low at true valleys) because
      well‑fit cells cost nothing and wrong cells cost proportionally
      more.

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

        # ── Level: per-cell gated logcosh, log-magnitude, focal, /n_event ──
        # focal = (1 - exp(-|raw_error|)) — down-weights well-fit cells,
        # preserves full gradient on large errors. Per-cell scaling (no
        # series-level mean → no peace noise amplification).
        raw_error = y_pred - y_true
        focal = 1.0 - torch.exp(-raw_error.abs())
        mag_weight = torch.log1p(abs_max)
        level_raw = gate * mag_weight * focal * self._Logcosh(raw_error)

        n_event = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        level_cell = level_raw / n_event

        # ── Combine ──────────────────────────────────────────────────
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = level_cell.sum(dim=(0, 1))

            per_channel = shape + level
            total_loss = per_channel.sum()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)
            level = level_cell.sum()
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
