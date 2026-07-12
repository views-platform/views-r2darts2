import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (per‑cell pattern).** Plain demeaned logcosh residual,
      gated and aggregated per series, then reweighted by a per‑series
      DRO scalar. Scores temporal dynamics only.

      The shape CELL is left UNWEIGHTED. A per‑cell ``|raw_error|`` weight
      inflates the shape SCALE at large‑error event cells (the Hájek
      denominator is only ``gate.sum()``, so the extra factor is not
      divided back out). Early in training that balloons shape relative to
      level and steals the fixed gradient budget from the DC/magnitude
      signal → worse magnitude capture. Plain logcosh keeps shape O(1) and
      stable so level keeps its share of the budget.

      **Series‑level DRO** (not per‑cell). A per‑SERIES scalar weight
      ``w_i = sqrt(L_i / mean_active(L))`` upweights harder series
      sublinearly. Being constant across time it commutes with the
      demeaning projection, so it is DC‑neutral and cannot fight level.
      Renormalising to mean 1 over active series keeps the shape scale
      fixed (unlike ``|raw_error|``), so DRO redistributes focus WITHOUT
      stealing gradient budget from level.

      **Peace masking.** Series where BOTH y_true and y_pred are entirely
      below ``tau`` are pure peace → their AC pattern is noise. They are
      excluded from the shape loss and from the DRO statistics.

    * **Level (per‑cell magnitude).** Per‑cell logcosh on the raw error,
      weighted by ``gate × log1p(abs_max)``, summed, and normalised by
      event count per series. No DRO on level.

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

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold
        T = y_pred.size(1)

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))

        # ── Shape: plain demeaned logcosh (DC-neutral by construction) ─
        #   pred_ac = y_pred − mean(y_pred) applies the projection
        #   P = I − (1/T)·11ᵀ. P is symmetric & idempotent, so the gradient
        #   of logcosh(pred_ac − true_ac) w.r.t. y_pred is P·g — ZERO-MEAN
        #   per series. Shape contributes NO DC shift; level owns the mean.
        # The cell is UNWEIGHTED: a per-cell |raw_error| weight inflates the
        # shape scale at big-error cells (not divided out by the gate-only
        # Hájek denominator), stealing gradient budget from level → worse
        # magnitude. Plain logcosh stays O(1) and stable.
        raw_error = y_pred - y_true
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # ── Per-series gated Hájek shape loss (over time) ───────────
        # gate focuses the time-aggregation on event timesteps.
        gate_t = gate.sum(dim=1).clamp_min(self._EPS)            # (B,) or (B, C)
        series_shape = (gate * shape_cell).sum(dim=1) / gate_t   # (B,) or (B, C)

        # ── Peace-series mask ───────────────────────────────────────
        # A series is ACTIVE only if abs_max > tau at some timestep (event
        # in y_true or y_pred). Series where BOTH are entirely below tau are
        # pure peace → excluded from the shape loss and the DRO statistics.
        series_abs_max = abs_max.max(dim=1).values              # (B,) or (B, C)
        series_active = (series_abs_max > tau).float()          # (B,) or (B, C)

        # ── Series-level DRO (DC-neutral, scale-stable) ─────────────
        # Per-SERIES scalar weight (constant across time) commutes with the
        # demeaning projection → DC-neutral, cannot fight level.
        # w_i = sqrt(L_i / mean_active(L)) upweights harder series sublinearly.
        # Renormalising to mean 1 over active series keeps the shape scale
        # fixed, so DRO redistributes focus WITHOUT stealing budget from level.
        sd = series_shape.detach()
        active_sum = series_active.sum(dim=0).clamp_min(1.0)                 # () or (C,)
        mean_active = (sd * series_active).sum(dim=0) / active_sum           # () or (C,)
        w_dro = torch.sqrt(sd / mean_active.clamp_min(self._EPS).unsqueeze(0))
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=1.0, neginf=0.0)
        w_dro = w_dro * series_active                                        # drop peace
        w_dro_mean = (w_dro.sum(dim=0) / active_sum).clamp_min(self._EPS)    # () or (C,)
        w_dro = w_dro / w_dro_mean.unsqueeze(0)                              # mean 1

        # ── Shape: DRO-weighted average over series (Hájek in series space)
        shape = (w_dro * series_shape).sum(dim=0) / w_dro.sum(dim=0).clamp_min(self._EPS)

        # ── Level: per-cell gated logcosh, log-magnitude, /n_event ──
        mag_weight = torch.log1p(abs_max)
        level_raw = gate * mag_weight * self._Logcosh(raw_error)

        n_event = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        level_cell = level_raw / n_event

        # ── Combine ──────────────────────────────────────────────────
        # shape is already computed above (series-DRO weighted, peace-masked).
        if multivariate:
            level = level_cell.sum(dim=(0, 1))

            per_channel = shape + level
            total_loss = per_channel.sum()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
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
