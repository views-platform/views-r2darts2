import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × MSE(true-event-mean gap) — V24.

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged.

    * **Level (DC magnitude).** ``T × gap²`` where ``gap`` is the
      **true-event-mean gap** — mean over cells where ``y_true > tau``
      only (NOT y_pred).

      V13 used all-cell mean: gap = mean(y_pred) - mean(y_true) over
      ALL 36 cells. Diluted by 31 peace cells. Gap 7× too small.

      V23 used gate-weighted event mean: gate = sigmoid(max(|y_true|,
      |y_pred|) - tau). The gate depends on y_pred → the model can
      game it by spiking at any cell → gate turns on → mean includes
      spike → gap small. Training/eval gap: training 0.98× but eval
      MCR 0.27× on sb. The model learned to spike, not to calibrate.

      V24 uses TRUE-event mean: mask = (y_true > tau). The mask
      depends ONLY on y_true → FIXED target the model cannot game.
      No dilution (only true event cells counted). No gaming (mask
      doesn't depend on y_pred).

      gap = mean(y_pred, where y_true > tau) - mean(y_true, where y_true > tau)

      This is the mean prediction error AT TRUE EVENT CELLS. The model
      must predict the right level where events actually are — it
      can't satisfy the loss by spiking elsewhere.

      Why this is DC-only:
        gap is a scalar per series (weighted mean → scalar).
        d(gap)/d(y_pred[t]) = mask[t] / sum(mask)  (uniform for true events)
        d(T*gap²)/d(y_pred[t]) = 2*gap * mask[t]/sum(mask)
        This is UNIFORM across true event cells → pure DC, zero AC.

      Why this fixes dilution:
        V13: gap = sum(e over 36 cells) / 36 ≈ 0.2 (diluted)
        V24: gap = sum(e over 5 true events) / 5 ≈ 1.5 (undiluted)
        7× larger gap → 7× stronger DC push.

      Why this fixes gaming:
        V23's gate turned on at y_pred spikes → model could spike
        anywhere → gap stayed small. V24's mask only counts y_true
        events → spiking at a peace cell doesn't change the gap →
        the model MUST predict the right level at true event cells.

      Why ch_1, ch_2 will calibrate (V23 failed):
        V23's gate-weighted mean was dominated by wherever the model
        spiked. For ch_1 (rare events), the model couldn't find the
        events, so it spiked at wrong cells → gate turned on there →
        mean was dominated by wrong cells → 89% AC gradient → never
        learned the level. V24's mask points at TRUE events → the
        model is told "raise your prediction HERE" → DC gradient
        at the right cells → calibration.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None

        logger.info("SpotlightLossV24 | threshold=%.4f", non_zero_threshold)

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

        # ── Event gate (for Shape, uses both y_true and y_pred) ──────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── TRUE-event mask (for Level, uses y_true ONLY) ────────────
        # V24: FIXED mask that the model cannot game.
        # V23's gate depended on y_pred → model could spike anywhere
        # → gaming → training/eval gap.
        # V24's mask depends only on y_true → fixed target → no gaming.
        true_mask = (y_true.abs() > self.tau).float()

        # ── SHAPE: log_cosh on demeaned errors, DRO on |raw_error| ───
        # Unchanged from V13.
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
            shape_w = gate * w_dro
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            shape_w = gate * w_dro
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: T × MSE(true-event-mean gap) — V24 ────────────────
        # Replaces V13's all-cell mean and V23's gate-weighted mean.
        #
        # true_mask depends ONLY on y_true → FIXED target, no gaming.
        # mean over TRUE event cells only → no dilution by peace cells.
        #
        # gap = mean(y_pred at true events) - mean(y_true at true events)
        #     = mean(e at true events)  (since y_pred - y_true = e)
        #
        # Gradient: 2*gap * true_mask[t] / sum(true_mask) per cell.
        # Uniform across true event cells → DC-only → no Shape conflict.
        sum_tm = true_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pred_te = (true_mask * y_pred).sum(dim=1, keepdim=True) / sum_tm
        mean_true_te = (true_mask * y_true).sum(dim=1, keepdim=True) / sum_tm
        gap = (mean_pred_te - mean_true_te).squeeze(1)  # (B,) or (B, C)

        level_cell = T * gap ** 2
        w_level = gate.amax(dim=1)  # per-series event mass (same as V13)

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

                # Gap diagnostics — compare V24's true-event gap to V13's all-cell gap
                gap_v13 = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga     = gap.abs()
                _ga_v13 = gap_v13.abs()
                gap_mean_l     = _ga.mean(dim=0).tolist()
                gap_max_l      = _ga.amax(dim=0).tolist()
                gap_v13_mean_l = _ga_v13.mean(dim=0).tolist()
                gap_v13_max_l  = _ga_v13.amax(dim=0).tolist()
                dilution_l = (_ga.mean(dim=0) / _ga_v13.mean(dim=0).clamp_min(1e-8)).tolist()

                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V24: true-event diagnostics ──
                # True event cells per series
                _n_te_per_series = true_mask.sum(dim=1)  # (B, C)
                _te_series_mask = (true_mask.sum(dim=1) > 0).float()  # (B, C)
                _n_tes = _te_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
                n_te_per_series_l = ((_n_te_per_series * _te_series_mask).sum(dim=0) / _n_tes).tolist()

                # Mean pred/true at true event cells
                _mean_pred_te = mean_pred_te.squeeze(1)  # (B, C)
                _mean_true_te = mean_true_te.squeeze(1)  # (B, C)
                mean_pred_te_l = (_mean_pred_te.mean(dim=0)).tolist()
                mean_true_te_l = (_mean_true_te.mean(dim=0)).tolist()

                # ── V24: spike misplacement diagnostics ──
                # False alarms: y_pred > tau at cells where y_true < tau
                _false_alarm_mask = ((y_pred.abs() > self.tau).float() * (1 - true_mask)) * _te_series_mask.unsqueeze(1)
                _n_fa = _false_alarm_mask.sum(dim=(0, 1)).clamp_min(1.0)
                false_alarm_frac_l = (_false_alarm_mask.sum(dim=(0, 1))
                                      / true_mask.sum(dim=(0, 1)).clamp_min(1.0)).tolist()
                # Mean y_pred at false alarm cells (should be ~0 if no spiking)
                _fa_pred = (y_pred.abs() * _false_alarm_mask).sum(dim=(0, 1)) / _n_fa
                false_alarm_mag_l = _fa_pred.tolist()

                # True events missed: y_pred < tau at cells where y_true > tau
                _missed_mask = (true_mask * (y_pred.abs() < self.tau).float()) * _te_series_mask.unsqueeze(1)
                _n_missed = _missed_mask.sum(dim=(0, 1)).clamp_min(1.0)
                missed_frac_l = (_missed_mask.sum(dim=(0, 1))
                                 / true_mask.sum(dim=(0, 1)).clamp_min(1.0)).tolist()

                sl_ratio_l = (loss_shape.detach() / loss_level.detach().clamp_min(self._EPS)).tolist()
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
                gap_v13 = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga     = gap.abs()
                _ga_v13 = gap_v13.abs()
                gap_mean_l     = [_ga.mean().item()]
                gap_max_l      = [_ga.max().item()]
                gap_v13_mean_l = [_ga_v13.mean().item()]
                gap_v13_max_l  = [_ga_v13.max().item()]
                dilution_l = [(_ga.mean() / max(1e-8, _ga_v13.mean())).item()]
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l  = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                _n_te_per_series = true_mask.sum(dim=1)
                _te_series_mask = (true_mask.sum(dim=1) > 0).float()
                _n_tes = _te_series_mask.sum().clamp_min(1.0)
                n_te_per_series_l = [((_n_te_per_series * _te_series_mask).sum() / _n_tes).item()]
                _mean_pred_te = mean_pred_te.squeeze(1) if mean_pred_te.dim() > 1 else mean_pred_te
                _mean_true_te = mean_true_te.squeeze(1) if mean_true_te.dim() > 1 else mean_true_te
                mean_pred_te_l = [_mean_pred_te.mean().item()]
                mean_true_te_l = [_mean_true_te.mean().item()]
                _false_alarm_mask = ((y_pred.abs() > self.tau).float() * (1 - true_mask)) * _te_series_mask.unsqueeze(1)
                _n_fa = _false_alarm_mask.sum().clamp_min(1.0)
                false_alarm_frac_l = [(_false_alarm_mask.sum() / max(1.0, true_mask.sum().item())).item()]
                _fa_pred = (y_pred.abs() * _false_alarm_mask).sum() / _n_fa
                false_alarm_mag_l = [_fa_pred.item()]
                _missed_mask = (true_mask * (y_pred.abs() < self.tau).float()) * _te_series_mask.unsqueeze(1)
                _n_missed = _missed_mask.sum().clamp_min(1.0)
                missed_frac_l = [(_missed_mask.sum() / max(1.0, true_mask.sum().item())).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV24: per_channel={comp}")

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
            # ── DRO diagnostics ──
            "dro_w_mean":     dro_wmean_l,
            "dro_w_std":      dro_wstd_l,
            "dro_w_max":      dro_wmax_l,
            "dro_frac_up":    dro_frac_up_l,
            "event_frac":     event_frac_l,
            # ── Gap diagnostics ──
            "level_gap_mean": gap_mean_l,
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "level_gap_sat":     gap_sat_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V24: dilution diagnostics ──
            "gap_v13_mean":   gap_v13_mean_l,
            "gap_v13_max":    gap_v13_max_l,
            "dilution":       dilution_l,
            # ── V24: true-event diagnostics ──
            "n_te_per_series": n_te_per_series_l,  # true event cells per series
            "mean_pred_te":   mean_pred_te_l,       # mean y_pred at true events
            "mean_true_te":   mean_true_te_l,       # mean y_true at true events
            # ── V24: spike misplacement diagnostics ──
            # False alarms: y_pred > tau where y_true < tau.
            # V23's gate-weighted mean was gameable because it counted
            # these. V24's true_mask ignores them.
            # If false_alarm_frac is high, the model is spiking at
            # wrong cells. V24 doesn't penalize this directly (Level
            # only looks at true events), but Shape should catch it.
            "false_alarm_frac": false_alarm_frac_l,  # false alarms / true events
            "false_alarm_mag":  false_alarm_mag_l,   # mean |y_pred| at false alarms
            # Missed events: y_pred < tau where y_true > tau.
            # If high, the model is underpredicting at true events.
            "missed_frac":      missed_frac_l,       # missed events / true events
        }

        logger.debug(
            "SpotlightLossV24 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV24(non_zero_threshold={self.tau})"
