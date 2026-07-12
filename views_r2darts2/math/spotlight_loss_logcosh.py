import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × MSE(binary-OR-mask gap) — V25.

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged.

    * **Level (DC magnitude).** ``T × gap²`` where ``gap`` is the mean
      over a **binary OR mask**: cells where ``y_true > tau`` OR
      ``y_pred > tau`` (detached).

      V13 (all-cell mean): diluted by 31 peace cells. Gap 7× too small.
      V23 (soft gate, sigmoid of max(y_true,y_pred)): gameable — the
        model spikes at any cell → gate turns on → spike included in
        mean → gap small. Training 0.98× but eval MCR 0.27× on sb.
      V24 (true_mask only): blind to overpredictions — spikes at peace
        cells (y_true < tau) are invisible → runaway overprediction.

      V25 (binary OR mask): 1 if y_true > tau OR y_pred.detach() > tau,
        else 0. Binary (not soft) → not gameable. Includes false alarms
        → not blind to overpredictions. Excludes peace cells → not
        diluted.

      Why binary (not soft) is the key:
        V23's soft gate gave weight sigmoid(10*(x-tau)), which varies
        with y_pred magnitude. A bigger spike → bigger weight → more
        influence on the mean → gameable.
        V25's binary mask gives weight 1 for all included cells. A
        spike of 5 and a spike of 50 both get weight 1. The model
        cannot gain influence by spiking harder.

      Why detached y_pred in the OR condition:
        The mask is computed from y_pred.detach() → no gradient flows
        through the mask itself. The gradient only flows through y_pred
        in the numerator (sum(mask * y_pred)). This prevents feedback
        loops where the model adjusts predictions to change which cells
        are included.

      Why this catches overpredictions (unlike V24):
        Model spikes at peace cell (y_pred=10, y_true=0).
        y_pred.detach() > tau → mask=1 at that cell.
        mean_pred includes 10, mean_true includes 0.
        gap > 0 → gradient = 2*gap/sum(mask) → pushes y_pred DOWN.
        Every false alarm is caught.

      Why this is DC-only:
        gap is a scalar per series (weighted mean → scalar).
        d(gap)/d(y_pred[t]) = mask[t] / sum(mask)  (uniform for mask=1)
        d(T*gap²)/d(y_pred[t]) = 2*gap * mask[t]/sum(mask)
        Uniform across included cells → pure DC, zero AC, no Shape conflict.

      Limitation (same as all DC approaches):
        When both false alarms and missed events exist, the DC gradient
        pushes all cells in the same direction (the net gap direction).
        It can't push false alarms down AND missed events up
        simultaneously — that's Shape's job. But Level correctly
        detects the NET error and pushes the average in the right
        direction, while Shape's DRO handles the distribution.

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

        logger.info("SpotlightLossV25 | threshold=%.4f", non_zero_threshold)

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

        # ── Event gate (for Shape — soft, uses both y_true and y_pred) ─
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── Binary OR mask (for Level — binary, detached y_pred) ─────
        # V25: 1 if y_true > tau OR y_pred.detach() > tau, else 0.
        # Binary → not gameable (unlike V23's soft gate).
        # Includes false alarms → not blind to overpred (unlike V24).
        # Excludes peace cells → not diluted (unlike V13).
        true_event = (y_true.abs() > self.tau)
        pred_alarm = (y_pred.detach().abs() > self.tau)
        level_mask = (true_event | pred_alarm).float()

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

        # ── LEVEL: T × MSE(binary-OR-mask gap) — V25 ─────────────────
        # Binary OR mask: true events + false alarms, uniform weight.
        # gap = mean(y_pred over mask) - mean(y_true over mask)
        # Gradient: 2*gap * mask[t] / sum(mask) — DC-only, uniform.
        sum_lm = level_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pred_lm = (level_mask * y_pred).sum(dim=1, keepdim=True) / sum_lm
        mean_true_lm = (level_mask * y_true).sum(dim=1, keepdim=True) / sum_lm
        gap = (mean_pred_lm - mean_true_lm).squeeze(1)  # (B,) or (B, C)

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

                # Gap diagnostics
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

                # ── V25: mask composition diagnostics ──
                # How many cells are in the mask, and what fraction are
                # true events vs false alarms?
                _n_true_ev = true_event.sum(dim=(0, 1)).clamp_min(1.0)  # (C,)
                _n_pred_alarm = pred_alarm.sum(dim=(0, 1)).clamp_min(1.0)  # (C,)
                _n_overlap = (true_event & pred_alarm).sum(dim=(0, 1)).clamp_min(1.0)  # (C,)
                _n_mask = level_mask.sum(dim=(0, 1)).clamp_min(1.0)  # (C,)
                # True events that the model also predicted (hits)
                hit_frac_l = (_n_overlap / _n_true_ev).tolist()
                # False alarms (pred > tau but true < tau) as fraction of mask
                _n_false_alarm = (pred_alarm & ~true_event).sum(dim=(0, 1)).clamp_min(1.0)
                false_alarm_of_mask_l = (_n_false_alarm / _n_mask).tolist()
                # Missed events (true > tau but pred < tau) as fraction of true events
                _n_missed = (true_event & ~pred_alarm).sum(dim=(0, 1)).clamp_min(1.0)
                missed_frac_l = (_n_missed / _n_true_ev).tolist()

                # Mean pred/true at mask cells
                _mean_pred_lm = mean_pred_lm.squeeze(1)  # (B, C)
                _mean_true_lm = mean_true_lm.squeeze(1)  # (B, C)
                mean_pred_lm_l = (_mean_pred_lm.mean(dim=0)).tolist()
                mean_true_lm_l = (_mean_true_lm.mean(dim=0)).tolist()

                # Mask cells per series
                _lm_per_series = level_mask.sum(dim=1)  # (B, C)
                _lm_series_mask = (level_mask.sum(dim=1) > 0).float()  # (B, C)
                _n_lms = _lm_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
                lm_per_series_l = ((_lm_per_series * _lm_series_mask).sum(dim=0) / _n_lms).tolist()

                # ── V25: overprediction catch diagnostics ──
                # Error at false alarm cells (should be positive = overpred)
                _e_fa = e * (pred_alarm & ~true_event).float()  # error at false alarms
                _n_fa = (pred_alarm & ~true_event).sum(dim=(0, 1)).clamp_min(1.0)
                e_fa_mean_l = (_e_fa.sum(dim=(0, 1)) / _n_fa).tolist()
                # Error at missed event cells (should be negative = underpred)
                _e_me = e * (true_event & ~pred_alarm).float()  # error at missed events
                _n_me = (true_event & ~pred_alarm).sum(dim=(0, 1)).clamp_min(1.0)
                e_me_mean_l = (_e_me.sum(dim=(0, 1)) / _n_me).tolist()

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
                _n_true_ev = true_event.sum().clamp_min(1.0)
                _n_pred_alarm = pred_alarm.sum().clamp_min(1.0)
                _n_overlap = (true_event & pred_alarm).sum().clamp_min(1.0)
                _n_mask = level_mask.sum().clamp_min(1.0)
                hit_frac_l = [(_n_overlap / _n_true_ev).item()]
                _n_false_alarm = (pred_alarm & ~true_event).sum().clamp_min(1.0)
                false_alarm_of_mask_l = [(_n_false_alarm / _n_mask).item()]
                _n_missed = (true_event & ~pred_alarm).sum().clamp_min(1.0)
                missed_frac_l = [(_n_missed / _n_true_ev).item()]
                _mean_pred_lm = mean_pred_lm.squeeze(1) if mean_pred_lm.dim() > 1 else mean_pred_lm
                _mean_true_lm = mean_true_lm.squeeze(1) if mean_true_lm.dim() > 1 else mean_true_lm
                mean_pred_lm_l = [_mean_pred_lm.mean().item()]
                mean_true_lm_l = [_mean_true_lm.mean().item()]
                _lm_per_series = level_mask.sum(dim=1)
                _lm_series_mask = (level_mask.sum(dim=1) > 0).float()
                _n_lms = _lm_series_mask.sum().clamp_min(1.0)
                lm_per_series_l = [((_lm_per_series * _lm_series_mask).sum() / _n_lms).item()]
                _e_fa = e * (pred_alarm & ~true_event).float()
                _n_fa = (pred_alarm & ~true_event).sum().clamp_min(1.0)
                e_fa_mean_l = [(_e_fa.sum() / _n_fa).item()]
                _e_me = e * (true_event & ~pred_alarm).float()
                _n_me = (true_event & ~pred_alarm).sum().clamp_min(1.0)
                e_me_mean_l = [(_e_me.sum() / _n_me).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV25: per_channel={comp}")

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
            # ── V25: dilution diagnostics ──
            "gap_v13_mean":   gap_v13_mean_l,
            "gap_v13_max":    gap_v13_max_l,
            "dilution":       dilution_l,
            # ── V25: mask composition diagnostics ──
            # How the binary OR mask breaks down:
            "hit_frac":            hit_frac_l,              # true events the model also predicted (hits/true_events)
                                                            # Should increase over training
            "false_alarm_of_mask": false_alarm_of_mask_l,   # false alarms as fraction of mask
                                                            # Should decrease over training (model stops spiking at peace)
            "missed_frac":         missed_frac_l,            # missed events / true events
                                                            # Should decrease over training
            "mean_pred_lm":        mean_pred_lm_l,           # mean y_pred at mask cells
            "mean_true_lm":        mean_true_lm_l,           # mean y_true at mask cells
            "lm_per_series":       lm_per_series_l,          # mask cells per series
            # ── V25: overprediction catch diagnostics ──
            # Error at false alarm cells — V24 was blind to these.
            # V25 catches them. Should be positive (overpred) and
            # decrease over training as the model stops spiking.
            "e_fa_mean":           e_fa_mean_l,              # mean error at false alarm cells
                                                            # Positive = overprediction. Should → 0.
            # Error at missed event cells — should be negative (underpred)
            # and increase toward 0 as the model learns to predict there.
            "e_me_mean":           e_me_mean_l,              # mean error at missed event cells
                                                            # Negative = underprediction. Should → 0.
        }

        logger.debug(
            "SpotlightLossV25 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV25(non_zero_threshold={self.tau})"
