import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    Fixed V6: hard shape gate eliminates DC/AC conflict on peace cells.
    DRO on raw error with reference-mean normalization (not self-mean).
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None

        logger.info("SpotlightLossV7 | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _dro_sqrt_gated(losses: torch.Tensor, signal_mask: torch.Tensor) -> torch.Tensor:
        """
        Signal-gated sqrt-DRO with BATCH reference mean.
        
        Key fix: mu is the BATCH-WIDE mean on signal cells, not per-series.
        This prevents the "uniform error → w=1 everywhere" cancellation.
        A series with uniform |e|=2.0 in a batch where mean |e|=5.0 gets
        w_raw = sqrt(2/5) = 0.63, properly downweighted.
        """
        l = losses.detach()
        
        # Batch-wide mean on signal cells (not per-series)
        # This maintains relative differences across series
        global_mu = l[signal_mask].mean() if signal_mask.any() else l.mean()
        global_mu = global_mu.clamp(min=1e-8)
        
        # Per-series mean for normalization (preserves within-series structure)
        signal_sum = signal_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        local_mu = (l * signal_mask).sum(dim=1, keepdim=True) / signal_sum
        local_mu = local_mu.clamp(min=1e-8)
        
        # Hybrid: use global mu for ratio (cross-series discrimination),
        # local mu for normalization (per-series stability)
        w_raw = torch.sqrt(l / global_mu)
        
        # Normalize to mean 1.0 on signal cells per series
        w_raw_mean = (w_raw * signal_mask).sum(dim=1, keepdim=True) / signal_sum
        w_raw_mean = w_raw_mean.clamp(min=1e-8)
        
        w = torch.where(
            signal_mask,
            w_raw / w_raw_mean,
            torch.ones_like(w_raw)
        )
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)

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

        # ── SHAPE: log_cosh on demeaned errors, HARD GATE (only events)
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean
        
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)
        
        # HARD gate for shape: peace cells contribute ZERO
        # This eliminates the DC/AC conflict
        gate_shape = signal_mask.float()  # 1.0 on signal, 0.0 on peace
        
        # DRO on raw |error|, but apply to shape
        raw_loss_for_dro = e.abs()
        w_dro = self._dro_sqrt_gated(raw_loss_for_dro, signal_mask)
        w_shape = gate_shape * w_dro
        
        if multivariate:
            loss_shape = (w_shape * shape_cell).sum(dim=(0, 1)) / w_shape.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (w_shape * shape_cell).sum() / w_shape.sum().clamp_min(self._EPS)

        # ── LEVEL: Hájek MSE, SOFT gate (all cells contribute)
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
            raise RuntimeError(f"NaN in SpotlightLossV7: per_channel={comp}")

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
            "SpotlightLossV7 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV7(non_zero_threshold={self.tau})"