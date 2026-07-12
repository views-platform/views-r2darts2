import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (per‑cell pattern).** Demeaned logcosh residual, gated,
      event‑only DRO‑weighted, and Hájek‑normalised.

      The DRO (``sqrt(loss / mean)``) is computed with an **event‑only
      mean**: only cells where ``abs_max > tau`` contribute to the
      denominator. This prevents the 90% peace cells (loss ≈ 0) from
      dragging the mean to zero and amplifying noise. The event‑only
      mean is ≈ 0.5–1.0, giving stable DRO weights of 0.5–2×.

      DRO on shape sharpens flat forecasts: a flat line has large
      shape errors at high‑magnitude peaks (``|e_ac|`` proportional to
      ``true − mean``) and small errors at low‑magnitude events. DRO
      upweights the high‑magnitude peaks → the model is pushed to fix
      them first → the forecast becomes variable instead of flat.

    * **Level (per‑cell magnitude).** Per‑cell logcosh on the raw error,
      weighted by ``gate × log1p(abs_max)``, summed, and normalised by
      event count per series. No DRO on level (amplifies peace noise
      because raw‑error mean ≈ 0 at 90% zeros).

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

    @staticmethod
    def _dro_weights(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-series sqrt DRO with event-only mean.

        ``mask`` selects which cells contribute to the mean (event cells
        where ``abs_max > tau``). Peace cells (mask=0) get weight 1.0
        (neutral). This prevents the 90% peace cells from dragging the
        mean to zero and amplifying noise.

        Returns weights with mean ≈ 1 on the active (masked) region.
        """
        l = losses.detach()
        m = mask.detach().to(dtype=l.dtype)

        # Event-only mean (not all-cell mean)
        n_active = m.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mu = (l * m).sum(dim=1, keepdim=True) / n_active

        w = torch.sqrt(l / mu.clamp_min(1e-6))
        # Normalize to mean 1 on active region
        w_active_mean = (w * m).sum(dim=1, keepdim=True) / n_active
        w = w / w_active_mean.clamp_min(1e-8)

        # Peace cells (mask=0) stay at weight 1.0
        w = 1.0 + m * (w - 1.0)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold
        T = y_pred.size(1)

        # ── Event gate and mask ──────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))
        # Hard mask for DRO: only event cells contribute to DRO mean
        event_mask = (abs_max > tau).float()

        # ── Shape: demeaned logcosh, standardised, event-DRO, Hájek ─
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # DRO with event-only mean: prevents peace-noise amplification
        w_dro = self._dro_weights(shape_cell, event_mask)

        # ── Level: per-cell gated logcosh, log-magnitude, /n_event ──
        # NO DRO on level — raw error mean ≈ 0 at 90% zeros → noise amplification
        raw_error = y_pred - y_true
        mag_weight = torch.log1p(abs_max)
        level_raw = gate * mag_weight * self._Logcosh(raw_error)

        n_event = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        level_cell = level_raw / n_event

        # ── Combine ──────────────────────────────────────────────────
        if multivariate:
            shape_w = gate * w_dro
            shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = level_cell.sum(dim=(0, 1))

            per_channel = shape + level
            total_loss = per_channel.sum()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape_w = gate * w_dro
            shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)
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
