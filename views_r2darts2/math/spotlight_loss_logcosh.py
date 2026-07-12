import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    Symmetric 2-term loss with signal-gated sqrt-DRO on raw error.
    DRO operates on raw error magnitude (not demeaned shape) to avoid
    demeaning-oscillation coupling. Gated by signal presence in either
    y_true or y_pred to prevent noise amplification at 97% sparsity.
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None

        logger.info("SpotlightLossV6 | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _dro_sqrt_gated(losses: torch.Tensor, signal_mask: torch.Tensor) -> torch.Tensor:
        """
        Signal-gated sqrt-DRO on raw errors.

        - losses: raw cell losses, shape (B, T) or (B, T, C)
        - signal_mask: bool tensor, True where |y_true|>tau OR |y_pred|>tau

        Design:
        1. Compute per-series mean on SIGNAL cells only (not all cells).
           Using all cells at 97% sparsity gives mu ≈ 0, amplifying noise.
        2. sqrt(l/mu) gives sublinear concentration.
        3. Renormalize to mean 1.0 on SIGNAL cells per series.
           Without this, a series with 1 signal cell gets w≈1 on that cell
           (sqrt(1/1)=1), but the cell's weight mass is 1/36 of the series.
           Normalizing within-signal-region makes the weight mass proper.
        4. Peace cells get w=1.0 (neutral, no DRO noise).

        The detach on losses prevents the model from gaming DRO weights.
        """
        l = losses.detach()

        # Per-series mean on SIGNAL cells only
        signal_sum = signal_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu = (l * signal_mask).sum(dim=1, keepdim=True) / signal_sum
        mu = mu.clamp(min=1e-8)

        # Raw sqrt weights
        w_raw = torch.sqrt(l / mu)

        # Renormalize to mean 1.0 on SIGNAL cells per series
        w_raw_mean = (w_raw * signal_mask).sum(dim=1, keepdim=True) / signal_sum
        w_raw_mean = w_raw_mean.clamp(min=1e-8)

        # Apply normalized weight on signal, 1.0 on peace
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

        # ── Signal detection (for gating and DRO) ───────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.tau))
        signal_mask = abs_max > self.tau  # bool, True if event in true OR pred

        # ── SHAPE: log_cosh on full-sequence demeaned errors, DRO-weighted
        # DRO operates on RAW error magnitude to avoid demeaning coupling
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        # DRO on raw |error| (not shape loss) — avoids demeaning oscillation
        raw_loss_for_dro = e.abs()  # (B, T) or (B, T, C)
        w_dro = self._dro_sqrt_gated(raw_loss_for_dro, signal_mask)

        # Combine: gate suppresses peace, DRO upweights hard event cells
        w_shape = gate * w_dro

        if multivariate:
            loss_shape = (w_shape * shape_cell).sum(dim=(0, 1)) / w_shape.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (w_shape * shape_cell).sum() / w_shape.sum().clamp_min(self._EPS)

        # ── LEVEL: Hájek MSE (symmetric, unbounded, batch-normalized)
        mag_weight = torch.log1p(abs_max)
        level_raw = gate * mag_weight * (e ** 2)

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
            raise RuntimeError(f"NaN in SpotlightLossV6: per_channel={comp}")

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
            "SpotlightLossV6 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV6(non_zero_threshold={self.tau})"