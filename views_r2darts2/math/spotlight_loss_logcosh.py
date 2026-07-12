import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    3-term loss for zero-inflated conflict fatality forecasting.
    Shape = log_cosh (prevents templating). Level = MSE (fixes underprediction).
    Temporal = adaptive MSE on diffs (fixes persistence). No hyperparameters.
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None

        logger.info("SpotlightLossV3 | threshold=%.4f", non_zero_threshold)

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

        # ── Event gate (soft floor, 5× slope) ─────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.tau))

        # ── SHAPE: log_cosh on full-sequence demeaned errors ────────
        # Demean the ERROR, not the raw signals. Cleaner and numerically stable.
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        # Scale by the true signal's AC std, with a floor to prevent blowup
        # on flat series. Use the true signal's std, not the error's.
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        if multivariate:
            loss_shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)

        # ── LEVEL: MSE on raw errors (unbounded gradient) ───────────
        mag_weight = torch.log1p(abs_max)          # bounded magnitude: Ukraine~2.3, small~1.1
        level_raw = gate * mag_weight * (e ** 2)   # MSE, not log_cosh

        n_event = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        level_cell = level_raw / n_event

        # CRITICAL FIX: mean over batch+time so level is batch-size invariant
        if multivariate:
            loss_level = level_cell.mean(dim=(0, 1))  # (C,)
        else:
            loss_level = level_cell.mean()

        # ── TEMPORAL: MSE on first differences (persistence fix) ──────
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]
        diff_error = (pred_diff - true_diff) ** 2
        gate_diff = gate[:, 1:]

        if multivariate:
            loss_temporal = (gate_diff * diff_error).mean(dim=(0, 1))  # (C,)
        else:
            loss_temporal = (gate_diff * diff_error).mean()

        # ── ADAPTIVE TEMPORAL WEIGHT (hyperparameter-free) ────────────
        # Measure how bursty predictions are relative to targets.
        # Detached so the model cannot game the weight.
        with torch.no_grad():
            if multivariate:
                pred_scale = (gate_diff * pred_diff.abs()).mean(dim=(0, 1))
                true_scale = (gate_diff * true_diff.abs()).mean(dim=(0, 1))
            else:
                pred_scale = (gate_diff * pred_diff.abs()).mean()
                true_scale = (gate_diff * true_diff.abs()).mean()

            scale_floor = 0.1 * self.tau
            true_scale = torch.clamp(true_scale, min=scale_floor)

            burst_ratio = pred_scale / true_scale
            # Soft activation: weak when matched, strong when bursty
            w_t = 0.1 + 0.9 * torch.sigmoid(5.0 * (burst_ratio - 1.2))

        # ── Combine ───────────────────────────────────────────────────
        if multivariate:
            per_channel = loss_shape + loss_level + w_t * loss_temporal
            total_loss = per_channel.sum()

            shape_c = loss_shape.detach().tolist()
            level_c = loss_level.detach().tolist()
            temp_c = loss_temporal.detach().tolist()
            w_t_c = w_t.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            total_loss = loss_shape + loss_level + w_t * loss_temporal
            shape_c = [float(loss_shape.detach())]
            level_c = [float(loss_level.detach())]
            temp_c = [float(loss_temporal.detach())]
            w_t_c = [float(w_t.detach())]
            comp = [float(total_loss.detach())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV3: per_channel={comp}")

        n = len(comp)
        self._last_components = {
            "shape": shape_c,
            "level": level_c,
            "temporal": temp_c,
            "w_temporal": w_t_c,
            "spec": [0.0] * n,
            "weight": [1.0] * n,
            "ema": [float("nan")] * n,
            "cal_ratio": [1.0] * n,
            "cal_score": [1.0] * n,
            "gates": [1.0] * n,
            "contribution": comp,
        }

        logger.debug(
            "SpotlightLossV3 | shape=%s level=%s temporal=%s w_t=%s total=%.6f",
            shape_c, level_c, temp_c, w_t_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV3(non_zero_threshold={self.tau})"