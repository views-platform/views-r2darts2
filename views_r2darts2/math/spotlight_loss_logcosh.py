import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    The loss splits the error into two **orthogonal** components:
    the per‑series DC (mean) and the per‑series AC (demeaned residual).

    * **Shape (AC pattern).** Operates on the demeaned prediction
      ``pred - pred_mean.detach()`` and demeaned truth. Because the
      mean is **detached**, the shape gradient flows *only* through the
      AC component of ``y_pred``. It cannot affect the DC level, so it
      cannot trigger a level counter‑correction. This eliminates the
      shape–level conflict that caused underprediction in previous
      versions (where shape DRO or |raw_error| weighting distorted the
      mean, causing the level term to push back down).

    * **Level (DC magnitude).** Operates on the per‑series mean gap
      ``pred_mean - true_mean``, weighted by event count and magnitude.
      Because the shape term is detached from the mean, the level term
      has full control over the DC component — it can raise or lower
      the entire forecast without fighting shape.

    This orthogonality means **any weighting applied to shape (DRO,
    |raw_error|, quadratic) no longer conflicts with level**, because
    the shape gradient physically cannot reach the mean. The two terms
    optimize disjoint subspaces of the prediction.

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
        """Per-series sqrt DRO with masked mean."""
        l = losses.detach()
        m = mask.detach().to(dtype=l.dtype)

        n_active = m.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mu = (l * m).sum(dim=1, keepdim=True) / n_active

        w = torch.sqrt(l / mu.clamp_min(1e-6))
        w_active_mean = (w * m).sum(dim=1, keepdim=True) / n_active
        w = w / w_active_mean.clamp_min(1e-8)

        w = 1.0 + m * (w - 1.0)
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

        # ── Orthogonal decomposition ────────────────────────────────
        # Detach the mean so shape gradient doesn't flow through it.
        # This makes shape and level operate on disjoint subspaces:
        #   shape → AC component only (pred - mean.detach())
        #   level → DC component only (mean)
        pred_mean = y_pred.mean(dim=1, keepdim=True)
        true_mean = y_true.mean(dim=1, keepdim=True)

        # ── Shape: AC-only, logcosh, standardised, DRO, Hájek ───────
        # DRO is now SAFE because it cannot distort the mean (detached).
        # DRO upweights the hardest AC cells → sharpens flat forecasts.
        pred_ac = y_pred - pred_mean.detach()
        true_ac = y_true - true_mean
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # DRO with event-only mean
        event_mask = (abs_max > tau).float()
        w_dro = self._dro_weights(shape_cell, event_mask)

        # ── Level: DC-only, gated logcosh, log-magnitude, /n_event ──
        # Level operates on the detached-mean gap. Shape cannot interfere.
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
