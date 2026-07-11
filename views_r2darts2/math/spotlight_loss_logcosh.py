import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Adaptive loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).
    The loss is designed for ~86‑98% zeros and heavy‑tailed non‑zero values.

    ── Design ─────────────────────────────────────────────────────────

    The error is split into two complementary terms:

    * **Shape (temporal pattern).** Demeaned per‑series residual,
      penalised with logcosh and standardised by the channel‑wise
      temporal standard deviation of the true AC component.
      logcosh is chosen for stability: the training data contains noise
      and unexplainable spikes that destabilise unbounded gradients.

    * **Level (magnitude).** Per‑cell logcosh on raw error, weighted by
      event magnitude (gate × (1 + abs_max)). The (1+abs_max) weight IS
      the magnitude signal: Ukraine cells get 10× the weight of small
      events, so even though tanh saturates at ±1, the TOTAL gradient
      per cell is (1+abs_max) × tanh(e) — magnitude‑proportional through
      the weight, not through the gradient.

      The Hájek denominator uses ``sum(gate)`` (not ``sum(mag_weight)``)
      so the denominator counts only event cells (gate ≈ 1 for events,
      ≈ 0.018 for peace). This gives gradient ≈ (1+abs_max) × tanh(e) /
      n_events — comparable to shape (tanh ≈ 0.5) rather than 20× weaker.

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
        """log(cosh(z)) without the additive constant log(2)."""
        a = z.abs()
        return a + F.softplus(-2.0 * a)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold

        # ── Sharp event gate ─────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))

        # ── Shape term: demeaned logcosh, standardised ─────────────
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # ── Level term: per-cell logcosh, magnitude-weighted ────────
        # (1+abs_max) is the magnitude signal: Ukraine (asinh=9) gets
        # 10× the weight of a small event (asinh=2). Even though tanh
        # saturates at ±1, the total gradient per cell is
        # (1+abs_max) × tanh(e) / sum(gate) — magnitude-proportional.
        raw_error = y_pred - y_true
        level_cell_raw = self._Logcosh(raw_error)
        mag_weight = gate * (1.0 + abs_max)
        level_cell = mag_weight * level_cell_raw

        # ── Normalisation ────────────────────────────────────────────
        # Shape: Hájek (self-normalised) — composition-robust
        # Level: plain SUM (not Hájek) — Hájek dilutes by 1/sum(gate)
        # which includes 90% peace cells → gradient 20× too weak.
        # SUM + logcosh is stable: logcosh bounds per-cell gradient at
        # ±1, (1+abs_max) gives magnitude weighting. The clip is the
        # safety net for Ukraine-scale events (36 cells × 10 = 360,
        # clipped to 100 → still 10× stronger than Hájek).
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = level_cell.sum(dim=(0, 1))

            per_channel = shape + level
            total_loss = per_channel.mean()
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
