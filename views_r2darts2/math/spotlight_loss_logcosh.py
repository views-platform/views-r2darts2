import torch
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
        3-term loss for zero-inflated conflict fatality forecasting.
        Shape = log_cosh (prevents templating). Level = MSE (fixes underprediction).
        Temporal = MSE on diffs (fixes persistence).
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88, temporal_weight: float = 0.1):
        super().__init__()
        self.tau = non_zero_threshold
        self.lambda_t = temporal_weight

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Squeeze univariate channel dim if present
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
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        if multivariate:
            loss_shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)

        # ── LEVEL: MSE on raw errors (strong gradients) ─────────────
        mag_weight = torch.log1p(abs_max)          # bounded magnitude
        level_raw = gate * mag_weight * (e ** 2)   # MSE, not log_cosh

        n_event = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        level_cell = level_raw / n_event

        # BUG FIX from v1: mean over batch+time so level is batch-invariant
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

        # ── Combine ───────────────────────────────────────────────────
        if multivariate:
            per_channel = loss_shape + loss_level + self.lambda_t * loss_temporal
            total = per_channel.sum()
        else:
            total = loss_shape + loss_level + self.lambda_t * loss_temporal

        return total