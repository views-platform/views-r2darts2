import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (per‑cell pattern).** Demeaned logcosh residual, gated,
      DRO‑weighted, and Hájek‑normalised. The DRO (sqrt concentration)
      upweights the hardest cells per series — a flat forecast has
      large errors at peaks and small errors at valleys; DRO amplifies
      the peak errors, pushing the model to sharpen its temporal pattern
      instead of outputting a flat line.

    * **Level (per‑cell magnitude).** Per‑cell logcosh on the raw error,
      weighted by ``gate × log1p(abs_max)``, normalised by event count,
      and summed. ``log1p`` gives bounded magnitude weighting (Ukraine
      gets 2.3×, small event gets 1.1×). Event‑count normalisation
      prevents explosion (Ukraine 36 events → 0.064/cell).

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

    def _dro_weights(self, losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-series sqrt DRO: upweight hardest cells within each series.

        w = sqrt(loss / median_of_means(loss))
        A cell 4× harder than the series median gets 2× weight (sublinear).
        The median-of-means denominator is robust to outliers.

        Peace cells (mask ≈ 0) keep weight ≈ 1 (neutral).
        """
        l = losses.detach()
        m = mask.detach().to(dtype=l.dtype).clamp(min=0.0, max=1.0)

        def _wmean(x, w):
            den = w.sum(dim=1, keepdim=True).clamp_min(self._EPS)
            return (x * w).sum(dim=1, keepdim=True) / den

        T = int(l.shape[1])
        W = max(6, T // 3)
        n_blocks = max(1, (T + W - 1) // W)
        means = []
        for lb, mb in zip(torch.tensor_split(l, n_blocks, dim=1),
                          torch.tensor_split(m, n_blocks, dim=1)):
            means.append(_wmean(lb, mb))
        mom = torch.cat(means, dim=1)
        mu = mom.median(dim=1, keepdim=True).values.clamp_min(self._EPS)

        w = torch.sqrt(l / mu)
        w_active_mean = _wmean(w, m).clamp_min(self._EPS)
        w = w / w_active_mean  # normalise to mean 1 on active region
        w = 1.0 + m * (w - 1.0)  # peace cells stay at 1
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))

        # ── Shape: demeaned logcosh, standardised, DRO, Hájek ───────
        # DRO upweights hardest cells per series → sharpens flat forecasts.
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # DRO weights on shape (sqrt concentration, median-of-means)
        w_dro = self._dro_weights(shape_cell, gate)
        shape_weighted = gate * w_dro * shape_cell

        # ── Level: per-cell gated logcosh, log-magnitude-weighted,
        #            normalised by event count ────────────────────────
        raw_error = y_pred - y_true
        mag_weight = torch.log1p(abs_max)
        level_raw = gate * mag_weight * self._Logcosh(raw_error)

        n_event = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        level_cell = level_raw / n_event

        # ── Combine ──────────────────────────────────────────────────
        if multivariate:
            shape = shape_weighted.sum(dim=(0, 1)) / (gate * w_dro).sum(dim=(0, 1)).clamp_min(self._EPS)
            level = level_cell.sum(dim=(0, 1))

            per_channel = shape + level
            total_loss = per_channel.sum()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape = shape_weighted.sum() / (gate * w_dro).sum().clamp_min(self._EPS)
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