import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = MSE on mean gap (V14 — fixes V13 sporadic spikes & calibration overpred).

    ── V14 Changes ────────────────────────────────────────────────────

    1. **Soft level weighting:** `w_level = gate_true.mean(dim=1)` instead of
       `gate.amax(dim=1)`. Eliminates binary on/off that caused spikes when
       a single cell crossed threshold.

    2. **True-only gate for Level:** `gate_true` uses `y_true` only, not
       `max(y_true, y_pred.detach())`. Removes prediction→gate feedback loop
       that amplified oscillations.

    3. **Peace series get weak Level signal:** With soft mean, a series with
       1 event in 36 cells gets w_level≈0.03 (not 1.0). The model no longer
       overcorrects on marginal event series. Pure-peace series (w_level≈0)
       rely on Shape's DRO-weighted event cells for calibration — but since
       gate_true is 0 for peace cells, Shape also gets 0 weight. So we add
       a tiny peace anchor (see below).

    ── Why V13 Spiked ─────────────────────────────────────────────────

    V13's `w_level = gate.amax(dim=1)` was binary: 1.0 if ANY cell in the
    series had max(|y_true|,|y_pred|) > τ. On calibration (sparse events),
    this created hard transitions:

    - Epoch N:   y_pred=0.5 everywhere → gate≈0 → w_level=0 → no Level loss
    - Epoch N+1: weight update drifts y_pred to 0.95 on some cells
                 → gate flips to 1 → w_level=1 → T×gap² suddenly active
                 → gradient yanks mean hard → overshoot → spike

    The gate also used y_pred.detach(), so the flip was based on stale
    predictions, creating a feedback loop. Validation had dense events (gate
    always 1), so no flip, no spikes.

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Unchanged from V13.
    * **Level (DC magnitude).** `T × gap²` with soft, true-only gating.
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None

        logger.info("SpotlightLossV14 | threshold=%.4f", non_zero_threshold)

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

        self._last_input_grad = None
        if y_pred.requires_grad:
            y_pred.register_hook(lambda g: setattr(self, "_last_input_grad", g.detach()))

        e = y_pred - y_true

        # ── Event gates ──────────────────────────────────────────────
        # Shape gate: uses max(true, pred) — allows model to "discover" events
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate_shape = torch.sigmoid(10.0 * (abs_max - self.tau))

        # Level gate: uses true ONLY — deterministic, no feedback loop
        gate_true = torch.sigmoid(10.0 * (y_true.abs() - self.tau))

        # ── SHAPE: log_cosh on demeaned errors, DRO on |raw_error| ──
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        raw_abs = e.abs().detach()
        event_mask = (abs_max > self.tau).float()
        n_ev = event_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        dro_mu = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = torch.sqrt(raw_abs / dro_mu.clamp_min(1e-6))
        w_dro_mean = (w_dro * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = w_dro / w_dro_mean.clamp_min(1e-8)
        w_dro = 1.0 + event_mask * (w_dro - 1.0)
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=1.0, neginf=0.0)

        if multivariate:
            shape_w = gate_shape * w_dro
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            shape_w = gate_shape * w_dro
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: T × MSE(mean gap), soft true-only gating ─────────
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)

        # V14: soft mean instead of hard amax; true-only instead of max
        w_level = gate_true.mean(dim=1)  # (B,) or (B, C) — proportional to event density

        # Peace anchor: for series with no true events, weakly pull toward zero
        # This prevents pure-peace series from drifting arbitrarily.
        # Computed only where w_level is very small (no true events).
        peace_mask = (w_level < 0.05).float()  # series with <5% event density
        peace_anchor = peace_mask * (y_pred.mean(dim=1) ** 2)  # MSE to zero mean

        level_cell = T * gap ** 2 + 0.5 * peace_anchor  # 0.5 is weak anchor weight

        if multivariate:
            loss_level = (w_level * level_cell).sum(dim=0) / w_level.sum(dim=0).clamp_min(self._EPS)
        else:
            loss_level = (w_level * level_cell).sum() / w_level.sum().clamp_min(self._EPS)

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

        # ── Diagnostic telemetry ──────────────────────────────────────
        with torch.no_grad():
            if multivariate:
                _n_ev   = event_mask.sum(dim=(0, 1)).clamp_min(1.0)
                _w_ev   = w_dro * event_mask
                _dm     = _w_ev.sum(dim=(0, 1)) / _n_ev
                _dw2    = (_w_ev ** 2).sum(dim=(0, 1)) / _n_ev
                _dstd   = (_dw2 - _dm ** 2).clamp_min(0).sqrt()
                dro_wmean_l   = _dm.tolist()
                dro_wstd_l    = _dstd.tolist()
                dro_wmax_l    = w_dro.amax(dim=(0, 1)).tolist()
                dro_frac_up_l = (((w_dro > 1.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                event_frac_l  = event_mask.mean(dim=(0, 1)).tolist()

                _ga    = gap.abs()
                gap_mean_l    = _ga.mean(dim=0).tolist()
                gap_max_l     = _ga.amax(dim=0).tolist()

                # V14: w_level is soft, so "event series" threshold is different
                _ev_mask_s = (w_level > 0.05).float()  # soft threshold
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                # V14: peace anchor diagnostics
                peace_frac_l = peace_mask.mean(dim=0).tolist()
                peace_anchor_mean_l = (peace_anchor * peace_mask).sum(dim=0).clamp_min(0).tolist()

                shape_dc_l    = (gate_shape * e_shape).mean(dim=1).abs().mean(dim=0).tolist()
            else:
                _n_ev   = event_mask.sum().clamp_min(1.0)
                _w_ev   = w_dro * event_mask
                _dm     = (_w_ev.sum() / _n_ev).item()
                _dw2    = ((_w_ev ** 2).sum() / _n_ev).item()
                dro_wmean_l   = [_dm]
                dro_wstd_l    = [max(0.0, _dw2 - _dm ** 2) ** 0.5]
                dro_wmax_l    = [w_dro.max().item()]
                dro_frac_up_l = [((w_dro > 1.0) * event_mask).sum().item() / _n_ev.item()]
                event_frac_l  = [event_mask.mean().item()]
                _ga    = gap.abs()
                gap_mean_l    = [_ga.mean().item()]
                gap_max_l     = [_ga.max().item()]
                _ev_mask_s = (w_level > 0.05).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l  = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                peace_frac_l = [peace_mask.mean().item()]
                peace_anchor_mean_l = [(peace_anchor * peace_mask).sum().clamp_min(0).item()]
                shape_dc_l    = [(gate_shape * e_shape).mean(dim=1).abs().mean().item()]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV14: per_channel={comp}")

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
            "dro_w_mean":     dro_wmean_l,
            "dro_w_std":      dro_wstd_l,
            "dro_w_max":      dro_wmax_l,
            "dro_frac_up":    dro_frac_up_l,
            "event_frac":     event_frac_l,
            "level_gap_mean": gap_mean_l,
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "level_gap_sat":     gap_sat_l,
            # V14 diagnostics
            "peace_frac":        peace_frac_l,
            "peace_anchor_mean": peace_anchor_mean_l,
            "shape_dc":       shape_dc_l,
        }

        logger.debug(
            "SpotlightLossV14 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV14(non_zero_threshold={self.tau})"