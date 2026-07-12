import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    Minimal symmetric 2-term loss. No DRO, no temporal, no asymmetry.
    Shape = log_cosh on demeaned errors (hard-gated to events only).
    Level = Hájek MSE on raw errors (soft-gated, all cells).
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None

        logger.info("SpotlightLossV8 | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        B, T = y_pred.shape[:2]
        e = y_pred - y_true

        # Signal detection
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate_soft = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.tau))
        signal_mask = abs_max > self.tau

        # ── SHAPE: log_cosh on demeaned errors, HARD gate (events only)
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        # Hard gate: peace cells contribute zero to shape
        gate_shape = signal_mask.float()

        if multivariate:
            loss_shape = (gate_shape * shape_cell).sum(dim=(0, 1)) / gate_shape.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (gate_shape * shape_cell).sum() / gate_shape.sum().clamp_min(self._EPS)

        # ── LEVEL: Hájek MSE, soft gate (all cells contribute)
        mag_weight = torch.log1p(abs_max)
        level_raw = gate_soft * mag_weight * (e ** 2)

        if multivariate:
            loss_level = level_raw.sum(dim=(0, 1)) / gate_soft.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_level = level_raw.sum() / gate_soft.sum().clamp_min(self._EPS)

        # ── Combine ───────────────────────────────────────────────────
        if multivariate:
            per_channel = loss_shape + loss_level
            total_loss = per_channel.sum()

            shape_c = loss_shape.detach().tolist()
            level_c = loss_level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            total_loss = loss_shape + loss_level
            shape_c = [float(loss_shape.detach())]
            level_c = [float(loss_level.detach())]
            comp = [float(total_loss.detach())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV8: per_channel={comp}")

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
            "SpotlightLossV8 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV8(non_zero_threshold={self.tau})"