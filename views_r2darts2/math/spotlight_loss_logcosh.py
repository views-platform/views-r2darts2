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
      (``pred − true``), weighted by ``gate × (1 + abs_max)`` and
      summed (NOT Hájek‑normalised).

      The signal is simple and direct: each event cell gets gradient
      ``gate × (1 + abs_max) × tanh(e)``. This pushes ONLY event cells
      (gate ≈ 1 for events, ≈ 0.018 for peace) toward their true value,
      with magnitude proportional to ``1 + abs_max`` (Ukraine gets 10×
      the weight of a small event). Peace cells get ≈ 0.01 gradient —
      effectively left alone.

      Previous level formulations (mean gap, batch gap, sum gap) all
      pushed ALL cells uniformly — including peace cells that should
      stay at 0. This created a conflict: level pushes peace up, shape
      pushes peace down → oscillation → calibration stuck.

      The per‑cell approach avoids this entirely: only event cells
      receive level gradient, peace cells are untouched. The SUM (not
      Hájek) prevents the 1/sum(gate) dilution that made the original
      variant 20× too weak. logcosh bounds each cell at ±1, so the
      total gradient for 3–4 events is ≈ 12–16 (within clip=100).

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

        # ── Level: per-cell gated logcosh, SUM ──────────────────────
        # The RIGHT signal: push ONLY event cells toward their true value.
        # gate * logcosh(pred - true), summed (not Hájek).
        #
        # Why this is the right signal (not the wrong one):
        # 1. gate ≈ 1 for events, ≈ 0.018 for peace → only events pushed
        # 2. logcosh bounds per-cell gradient at tanh ≤ 1 → stable
        # 3. SUM (not Hájek) prevents 1/sum(gate) dilution
        # 4. No (1+abs_max) weight → no explosion on Ukraine (was 10× per cell)
        # 5. Magnitude comes from EVENT COUNT: Ukraine (36 events) gets
        #    36× more total gradient than a 1-event series. This IS
        #    magnitude-proportional through count, not through weight.
        #
        # Previous approaches that failed:
        # - Mean/batch gap: pushed ALL cells uniformly (including peace) → conflict
        # - (1+abs_max) weight: exploded on Ukraine (10× per cell → 360 total)
        # - Hájek: diluted by 1/sum(gate) → 20× too weak
        raw_error = y_pred - y_true
        level_cell = gate * self._Logcosh(raw_error)

        # ── Combine ──────────────────────────────────────────────────
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = level_cell.sum(dim=(0, 1))  # SUM, not Hájek

            per_channel = shape + level
            total_loss = per_channel.sum()  # SUM over channels
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
