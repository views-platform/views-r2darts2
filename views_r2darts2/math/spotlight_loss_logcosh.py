import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    2-term loss for zero-inflated conflict fatality forecasting.
    Shape = log_cosh (prevents templating).
    Level = asymmetric Hájek MSE (fixes underprediction via heavy left-tail penalty).
    No temporal component.
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None

        logger.info("SpotlightLossV10 | threshold=%.4f", non_zero_threshold)

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
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        if multivariate:
            loss_shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)

        # ── LEVEL: Asymmetric Hájek MSE ─────────────────────────────
        # Asymmetric penalty: underprediction (e < 0, pred < true) is penalized
        # more heavily. This directly corrects the systematic underprediction bias.
        # 
        # For e < 0 (underpredict): penalty = 2.0 * e^2  (2x weight)
        # For e > 0 (overpredict):  penalty = 1.0 * e^2  (normal)
        # 
        # The asymmetry factor is applied BEFORE the Hájek normalization so it
        # genuinely shifts the optimal prediction upward.
        mag_weight = torch.log1p(abs_max)
        
        # Asymmetric squared error
        asym_factor = torch.where(e < 0, 2.0, 1.0)  # 2x for underprediction
        level_raw = gate * mag_weight * asym_factor * (e ** 2)

        # Hájek normalization: sum over all event cells in batch, divide by total gate mass
        # This is batch-size invariant and gives each event cell gradient proportional
        # to its error, NOT diluted by n_event per series.
        if multivariate:
            loss_level = level_raw.sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_level = level_raw.sum() / gate.sum().clamp_min(self._EPS)

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
            raise RuntimeError(f"NaN in SpotlightLossV10: per_channel={comp}")

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
            "SpotlightLossV10 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV10(non_zero_threshold={self.tau})"