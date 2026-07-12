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

    * **Level (batch magnitude).** Uses the BATCH-LEVEL gated mean gap
      per channel: ``T × logcosh(gated_mean(pred) − gated_mean(true))``.

      The batch-level gap has a SINGLE SIGN per channel — if the batch
      underpredicts, ALL series get pushed up; if it overpredicts, ALL
      get pushed down. This eliminates the per-series cancellation that
      kept the Hájek level stuck at 8.3 while calibration dropped.

      The gated mean uses the event gate as weights, so only event
      cells contribute to the gap. The gradient per event cell is:
        ``tanh(batch_gap) × gate / n_event ≈ 0.33`` per cell
      (for typical gap≈2, n_event≈3), which matches the shape gradient
      (~0.5) without dominating it.

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

        # ── Level: batch-level gated mean gap ────────────────────────
        # ONE gap per channel (not per series). This ensures a single
        # gradient direction: if the batch underpredicts, ALL series
        # get pushed up. No per-series cancellation.
        #
        # Gated mean: only event cells contribute (weighted by gate).
        # T scaling: compensates 1/T dilution from the mean operator.
        # Gradient per event cell ≈ tanh(gap) × gate / n_event ≈ 0.33.
        if multivariate:
            gate_sum = gate.sum(dim=(0, 1)).clamp_min(self._EPS)  # (C,)
            batch_pred_mean = (y_pred * gate).sum(dim=(0, 1)) / gate_sum  # (C,)
            batch_true_mean = (y_true * gate).sum(dim=(0, 1)) / gate_sum  # (C,)
            batch_gap = batch_pred_mean - batch_true_mean  # (C,)
            level = T * self._Logcosh(batch_gap)  # (C,)

            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate_sum
            per_channel = shape + level
            total_loss = per_channel.sum()  # SUM over channels
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            gate_sum = gate.sum().clamp_min(self._EPS)
            batch_pred_mean = (y_pred * gate).sum() / gate_sum
            batch_true_mean = (y_true * gate).sum() / gate_sum
            batch_gap = batch_pred_mean - batch_true_mean
            level = T * self._Logcosh(batch_gap)

            shape = (gate * shape_cell).sum() / gate_sum
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
