import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (per‑cell pattern).** Demeaned logcosh residual, gated,
      DRO‑weighted, and Hájek‑normalised.

    * **Level (per‑cell magnitude).** Per‑cell logcosh on the raw error,
      weighted by ``gate × log1p(abs_max) × DRO``, normalised by event
      count, and summed. DRO on level is what actually sharpens flat
      forecasts: a flat line at the mean has equal-magnitude errors at
      peaks (underprediction) and valleys (overprediction). Without DRO
      these cancel. DRO upweights whichever direction is harder,
      breaking the symmetry and pushing the model to vary its
      predictions.

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
    def _dro_weights(losses: torch.Tensor) -> torch.Tensor:
        """Per-series sqrt self-reweighting.

        w_it = sqrt(loss_it / mean_i(loss))

        Sublinear concentration: a cell 16× harder than average gets 4×
        the gradient (not 16×).  Redistributes enough signal to fix
        systematic bias while still focusing on spikes.

        Returns weights with mean ≈ 1 per series, shape (B, T) or (B, T, C).
        """
        l = losses.detach()
        mu = l.mean(dim=1, keepdim=True).clamp(min=1e-6)
        w = torch.sqrt(l / mu)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)
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
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)
        w_dro_shape = self._dro_weights(shape_cell)

        # ── Level: per-cell gated logcosh, log-magnitude, /n_event ──
        # NO DRO on level — DRO on raw error amplifies peace noise
        # (mean ≈ 0 at 90% zeros → sqrt(loss/mean) explodes for noise).
        raw_error = y_pred - y_true
        mag_weight = torch.log1p(abs_max)
        level_raw = gate * mag_weight * self._Logcosh(raw_error)

        n_event = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        level_cell = level_raw / n_event

        # ── Combine ──────────────────────────────────────────────────
        if multivariate:
            shape_w = gate * w_dro_shape
            shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = level_cell.sum(dim=(0, 1))

            per_channel = shape + level
            total_loss = per_channel.sum()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape_w = gate * w_dro_shape
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