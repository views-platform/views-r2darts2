import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    The raw error ``e = pred - true`` is split into DC (per‑series mean)
    and AC (demeaned residual). Each term sees ONLY its component:

    * **Shape (AC pattern).** ``|error_ac| × logcosh(error_ac / scale)``
      — the AC component, weighted by its own magnitude.

      The ``|error_ac|`` weight is critical: without it, a flat forecast
      at the correct mean has symmetric errors (``+x`` at valleys, ``−x``
      at peaks) → ``tanh`` gradient cancels → **zero shape gradient** →
      model is stuck flat. With ``|error_ac|``, the gradient becomes
      ``sign(e_ac) × logcosh + |e_ac| × tanh / scale`` — it does NOT
      cancel because the ``sign`` term is antisymmetric but the
      ``|e_ac| × tanh`` term is also antisymmetric and they ADD rather
      than subtract.

      The DC is **detached** in the shape term → shape cannot shift the
      mean → DRO on shape is safe (no level conflict).

    * **Level (DC magnitude).** ``logcosh(error_mean)`` per series — the
      DC component only. The AC is **detached** → level sees only the
      mean error → **uniform gradient** (same sign for all cells) → no
      peak/valley cancellation. Weighted by ``gate × log1p(abs_max) /
      n_event``, SUM.

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
        """Per-series sqrt DRO with masked (event-only) mean."""
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

        # ── Orthogonal DC/AC split ──────────────────────────────────
        raw_error = y_pred - y_true
        error_mean = raw_error.mean(dim=1, keepdim=True)  # DC
        error_ac = raw_error - error_mean                  # AC

        # ── Shape: AC-only, |e_ac|-weighted logcosh, DRO, Hájek ────
        # |error_ac| breaks the symmetry that causes gradient cancellation
        # on flat forecasts. DC is detached → no level conflict.
        ac_scale = error_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = error_ac.abs() * self._Logcosh(error_ac / ac_scale)

        event_mask = (abs_max > tau).float()
        w_dro = self._dro_weights(shape_cell, event_mask)

        # ── Level: DC-only, gated logcosh, log-magnitude, /n_event ─
        # AC is detached → level sees only mean error → uniform gradient.
        level_error = raw_error - error_ac.detach()  # = error_mean (broadcast)
        mag_weight = torch.log1p(abs_max)
        level_raw = gate * mag_weight * self._Logcosh(level_error)

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
