import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    Symmetric 2-term loss with safe per-series shape DRO.
    Shape = log_cosh + soft DRO (hard cell focus without noise amplification).
    Level = Hájek MSE (unbounded, symmetric, batch-normalized).
    No temporal component. No asymmetry. No hyperparameters.
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None

        logger.info("SpotlightLossV5 | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _dro_weights_safe(losses: torch.Tensor, gate: torch.Tensor, n_blocks: int = 3) -> torch.Tensor:
        """
        Safe per-series DRO: soft concentration, gated, block-robust.

        Design:
        1. Split series into n_blocks non-overlapping time chunks.
        2. Compute mean loss per block (robust to single-timestep outliers).
        3. Take median-of-block-means as reference mu (outlier-resistant).
        4. Compute soft concentration: w_raw = 1 + sqrt(loss / mu).
           The +1 ensures baseline weight is 1.0 (additive, not multiplicative).
        5. Gate: w = 1.0 on peace cells, w_raw on event cells.
           Prevents DRO from amplifying noise in the 97% peace regime.
        6. Normalize to mean 1.0 within the event region per series.
           Without this, a series with 1 hard cell gets that cell upweighted 3x,
           but the total weight mass is tiny → gradient gets lost in batch mean.
           Normalizing within-event-region preserves the relative upweight while
           ensuring the DRO contribution scales properly with batch composition.

        Returns weights with shape (B, T) or (B, T, C), all >= 1.0.
        """
        # losses, gate: (B, T) or (B, T, C)
        B, T = losses.shape[:2]
        if T < n_blocks:
            n_blocks = max(1, T // 2)

        # Split into blocks along time
        block_size = T // n_blocks
        remainder = T % n_blocks

        # Handle non-divisible T: distribute remainder to first blocks
        block_means = []
        start = 0
        for i in range(n_blocks):
            end = start + block_size + (1 if i < remainder else 0)
            if end > start:
                block = losses[:, start:end]  # (B, block_len) or (B, block_len, C)
                block_mean = block.mean(dim=1, keepdim=False)  # (B,) or (B, C)
                block_means.append(block_mean)
            start = end

        # Stack and compute median-of-means
        block_means = torch.stack(block_means, dim=1)  # (B, n_blocks) or (B, n_blocks, C)
        mu = block_means.median(dim=1).values  # (B,) or (B, C)
        mu = mu.clamp(min=1e-8)
        if losses.dim() == 3:
            mu = mu.unsqueeze(1)  # (B, 1, C)

        # Soft concentration: w_raw = 1 + sqrt(loss / mu)
        # Additive formulation: baseline is 1.0, hard cells get bonus.
        ratio = losses / mu
        w_raw = 1.0 + torch.sqrt(ratio.clamp(min=0.0))

        # Gate: only apply DRO where events exist
        event_mask = (gate > 0.5).float()  # (B, T) or (B, T, C)
        
        # Normalize w_raw to mean 1.0 within event region per series
        # This is critical: without it, the DRO bonus is diluted by peace cells
        event_sum = event_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        w_raw_mean_event = (w_raw * event_mask).sum(dim=1, keepdim=True) / event_sum
        
        # Avoid division by zero: if no events, keep w_raw as-is (all 1.0)
        w_raw_mean_event = torch.where(event_sum > 0.5, w_raw_mean_event, torch.ones_like(w_raw_mean_event))
        
        # Normalize: w = w_raw / mean(w_raw on events) on events, 1.0 on peace
        w_normalized = torch.where(
            event_mask > 0.5,
            w_raw / w_raw_mean_event.clamp(min=1e-8),
            torch.ones_like(w_raw)
        )

        return w_normalized

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

        # ── SHAPE: log_cosh on full-sequence demeaned errors + safe DRO
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        # Safe DRO on shape: upweights hard pattern cells within each series
        w_dro = self._dro_weights_safe(shape_cell, gate, n_blocks=3)
        
        # Combined shape weight: gate * DRO
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
            raise RuntimeError(f"NaN in SpotlightLossV5: per_channel={comp}")

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
            "SpotlightLossV5 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV5(non_zero_threshold={self.tau})"