import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × mean(e²) per-cell with per-series Hájek (V22 — catches spikes).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged from
      V13 — this component works.

    * **Level (DC + AC magnitude).** ``T × mean_T(e²)`` per series,
      per-series Hájek-normalised. ``e = y_pred - y_true`` (raw error,
      ALL cells, no cell-level gating).

      V13 used ``T × mean(e)²`` (mean-gap MSE). The mean is a first-
      moment statistic — it cannot distinguish a uniform distribution
      from a spike distribution with the same mean. The model satisfies
      ``mean(y_pred) ≈ mean(y_true)`` with zeros+spikes. On training,
      spikes align with events. On eval, spikes misalign → overprediction
      (V20 eval: sb 1.96× vs training 0.66×).

      V22 uses ``T × mean_T(e²)``:
        mean(e²) = var(e) + mean(e)²

      This sees BOTH the first moment (DC, same as V13) AND the second
      moment (AC variance, catches spikes). A spike distribution has
      34× higher mean(e²) than a uniform distribution with the same mean.

      Why MSE doesn't blow up here (unlike V21):
        V21 used per-CELL Hájek (denominator ≈ 6400 event cells).
        Gradient at a 10000-spike: 2*9.8/5 = 3.9 (concentrated on 5 cells).
        V22 uses per-SERIES Hájek (denominator ≈ 100 event series).
        Gradient at a 10000-spike: 2*9.8/100 = 0.196 (bounded by 1/sum(w)).
        V13's total gradient at severe underprediction: 0.04*36 = 1.44.
        V22's spike gradient (0.196) is 7× WEAKER than V13's worst case.

      DC gradient comparison (per cell, with Hájek 1/sum(w) ≈ 1/100):
        V13: 2*gap/100 = 2*1.24/100 = 0.025 (ALL 36 cells, uniform)
        V22: 2*e[t]/100 = 2*1.24/100 = 0.025 (5 event cells, targeted)
        — Same per-cell DC gradient! V22 just doesn't waste it on peace cells.

      AC gradient (what V13 lacks, V22 adds):
        At event cell: 2*(e[t] - mean(e))/100 ≈ 2*0/100 = 0 (if e[t] = mean)
        At spike cell: 2*(5 - 0.14)/100 = 0.097 (directly pushes spike down)
        At false-alarm peace cell: 2*(0.5 - 0.14)/100 = 0.007 (catches it)

      Why no cell-level gating (unlike V21):
        V21 used ``gate`` as per-cell weight, excluding peace cells.
        This made it blind to false alarms — spikes at peace cells got
        gate ≈ 0 → no penalty. V22 uses V13's series-level gate only
        (``gate.amax(dim=1)``). ALL cells in event series are included.
        A spike at a peace cell gives e = spike → e² = large → loss
        increases → model is penalized for the false alarm.

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

        logger.info("SpotlightLossV22 | threshold=%.4f", non_zero_threshold)

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

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

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

        # ── LEVEL: T × mean_T(e²), per-series Hájek — V22 ────────────
        # Replaces V13's T × mean(e)² (mean-gap MSE).
        #
        # mean(e²) = var(e) + mean(e)² — sees BOTH DC (mean) and AC (variance).
        # A spike distribution has 34× higher mean(e²) than uniform with same mean.
        #
        # No cell-level gating — ALL cells in event series are included.
        # This catches false alarms (spikes at peace cells), which V21's
        # cell-level gating missed.
        #
        # Per-series Hájek (not per-cell) — denominator ≈ 100 event series.
        # Gradient at 10000-spike: 2*9.8/100 = 0.196 (bounded, no blow-up).
        # V21's per-cell Hájek gave 2*9.8/5 = 3.9 (concentrated, unstable).
        #
        # T factor: same as V13. T * mean_T(e²) has gradient 2*e[t] per cell.
        # Without T, gradient would be 2*e[t]/T — diluted by 1/T.
        level_per_series = (e ** 2).mean(dim=1)  # mean over T, ALL cells, (B,) or (B, C)
        level_cell = T * level_per_series
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

                # Gap diagnostics (what V13's mean-gap would see — for comparison)
                gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga    = gap.abs()
                gap_mean_l    = _ga.mean(dim=0).tolist()
                gap_max_l     = _ga.amax(dim=0).tolist()
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V22: mean obfuscation diagnostics ──
                # The deception signal: compare mean(|e|) to |mean(e)|.
                # If mean(|e|) >> |mean(e)|, the mean is hiding error
                # (errors cancel out in the mean but are large individually).
                _mean_abs_e = e.abs().mean(dim=1)  # (B, C) — mean |e| per series
                _mean_e = e.mean(dim=1)  # (B, C) — mean e per series (= gap)
                _deception = _mean_abs_e / _mean_e.abs().clamp_min(1e-6)  # (B, C)
                _deception_ev = _deception * _ev_mask_s  # only event series
                _n_deception = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                deception_mean_l = (_deception_ev.sum(dim=0) / _n_deception).tolist()
                deception_max_l = (_deception_ev.amax(dim=0)).tolist()

                # Per-cell error stats (what V22 sees, V13 doesn't)
                _e_ev = e.abs() * event_mask  # event cells only
                _e_ev_mean = _e_ev.sum(dim=(0, 1)) / _n_ev  # (C,)
                e_ev_mean_l = _e_ev_mean.tolist()
                e_ev_max_l = (_e_ev.amax(dim=(0, 1))).tolist()

                # Peace cell false alarms (spikes at peace cells in event series)
                _peace_mask = (1 - event_mask) * _ev_mask_s.unsqueeze(1)  # (B, T, C)
                # Actually _ev_mask_s is (B, C), need to broadcast to (B, T, C)
                _peace_mask = (1 - event_mask) * _ev_mask_s.unsqueeze(1)
                _e_peace = e.abs() * _peace_mask
                _n_peace = _peace_mask.sum(dim=(0, 1)).clamp_min(1.0)
                e_peace_mean_l = (_e_peace.sum(dim=(0, 1)) / _n_peace).tolist()
                e_peace_max_l = (_e_peace.amax(dim=(0, 1))).tolist()
                # Fraction of peace cells (in event series) with |e| > 0.5 (false alarms)
                false_alarm_frac_l = (((e.abs() > 0.5) * _peace_mask).sum(dim=(0, 1))
                                      / _n_peace).tolist()

                # Spike diagnostics (event cells)
                spike_frac_l = (((e.abs() > 2.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                spike_severe_l = (((e.abs() > 4.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                overpred_frac_l = (((e > 0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                _overpred = (e * (e > 0).float() * event_mask).sum(dim=(0, 1))
                _n_overpred = ((e > 0).float() * event_mask).sum(dim=(0, 1)).clamp_min(1.0)
                overpred_mag_l = (_overpred / _n_overpred).tolist()
                underpred_frac_l = (((e < 0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()

                # mean(e²) decomposition: var(e) + mean(e)²
                _var_e = e.var(dim=1)  # (B, C) — variance of e per series
                _mean_e_sq = gap ** 2  # (B, C) — mean(e)² = V13's signal
                _var_e_ev = _var_e * _ev_mask_s
                _mean_e_sq_ev = _mean_e_sq * _ev_mask_s
                var_e_mean_l = (_var_e_ev.sum(dim=0) / _n_deception).tolist()
                mean_e_sq_mean_l = (_mean_e_sq_ev.sum(dim=0) / _n_deception).tolist()
                # Ratio: var(e) / mean(e)² — if high, AC dominates (spiky).
                # If ~0, DC dominates (uniform, V13 is fine).
                # If >10, V13 was hiding 10× more error than it showed.
                var_mean_ratio_l = ((_var_e_ev.sum(dim=0) / _n_deception)
                                    / (_mean_e_sq_ev.sum(dim=0) / _n_deception).clamp_min(1e-8)).tolist()

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
                gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga    = gap.abs()
                gap_mean_l    = [_ga.mean().item()]
                gap_max_l     = [_ga.max().item()]
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l  = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                _mean_abs_e = e.abs().mean(dim=1)
                _mean_e = e.mean(dim=1)
                _deception = _mean_abs_e / _mean_e.abs().clamp_min(1e-6)
                _deception_ev = _deception * _ev_mask_s
                _n_deception = _ev_mask_s.sum().clamp_min(1.0)
                deception_mean_l = [(_deception_ev.sum() / _n_deception).item()]
                deception_max_l = [_deception_ev.max().item()]
                _e_ev = e.abs() * event_mask
                _e_ev_mean = _e_ev.sum() / _n_ev
                e_ev_mean_l = [_e_ev_mean.item()]
                e_ev_max_l = [_e_ev.max().item()]
                _peace_mask = (1 - event_mask) * _ev_mask_s.unsqueeze(1)
                _e_peace = e.abs() * _peace_mask
                _n_peace = _peace_mask.sum().clamp_min(1.0)
                e_peace_mean_l = [(_e_peace.sum() / _n_peace).item()]
                e_peace_max_l = [_e_peace.max().item()]
                false_alarm_frac_l = [(((e.abs() > 0.5) * _peace_mask).sum() / _n_peace).item()]
                spike_frac_l = [(((e.abs() > 2.0) * event_mask).sum() / _n_ev).item()]
                spike_severe_l = [(((e.abs() > 4.0) * event_mask).sum() / _n_ev).item()]
                overpred_frac_l = [(((e > 0) * event_mask).sum() / _n_ev).item()]
                _overpred = (e * (e > 0).float() * event_mask).sum()
                _n_overpred = ((e > 0).float() * event_mask).sum().clamp_min(1.0)
                overpred_mag_l = [(_overpred / _n_overpred).item()]
                underpred_frac_l = [(((e < 0) * event_mask).sum() / _n_ev).item()]
                _var_e = e.var(dim=1)
                _mean_e_sq = gap ** 2
                _var_e_ev = _var_e * _ev_mask_s
                _mean_e_sq_ev = _mean_e_sq * _ev_mask_s
                var_e_mean_l = [(_var_e_ev.sum() / _n_deception).item()]
                mean_e_sq_mean_l = [(_mean_e_sq_ev.sum() / _n_deception).item()]
                var_mean_ratio_l = [((_var_e_ev.sum() / _n_deception).item()
                                     / max(1e-8, (_mean_e_sq_ev.sum() / _n_deception).item()))]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV22: per_channel={comp}")

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
            # ── Gap diagnostics (what V13's mean-gap would see) ──
            "level_gap_mean": gap_mean_l,
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "level_gap_sat":     gap_sat_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V22: mean obfuscation diagnostics ──
            # The deception ratio: mean(|e|) / |mean(e)|.
            # = 1.0 if all errors have the same sign (no cancellation)
            # = high if errors cancel (spikes offset by zeros)
            # V13 only sees |mean(e)|. V22 sees mean(|e|) via mean(e²).
            # If deception > 3, V13 was hiding 3× more error than it showed.
            "deception_mean":   deception_mean_l,   # mean deception ratio over event series
            "deception_max":    deception_max_l,    # max deception (most obfuscated series)
            # ── V22: mean(e²) decomposition ──
            # mean(e²) = var(e) + mean(e)²
            # var(e) = AC component (what V13 misses, V22 catches)
            # mean(e)² = DC component (what V13 sees)
            # var_mean_ratio = var(e) / mean(e)² — if >10, V13 was hiding 10× more
            "var_e_mean":       var_e_mean_l,       # mean var(e) over event series (AC)
            "mean_e_sq_mean":   mean_e_sq_mean_l,   # mean mean(e)² over event series (DC = V13's signal)
            "var_mean_ratio":   var_mean_ratio_l,    # var(e) / mean(e)² (obfuscation factor)
            # ── V22: per-cell error diagnostics ──
            "e_ev_mean":        e_ev_mean_l,        # mean |e| at event cells
            "e_ev_max":         e_ev_max_l,         # max |e| at event cells (largest spike)
            # ── V22: false alarm diagnostics (spikes at peace cells) ──
            # These are what cause eval overprediction. V13 can't see them
            # (mean dilutes). V21 gated them out. V22 catches them.
            "e_peace_mean":     e_peace_mean_l,     # mean |e| at peace cells in event series
            "e_peace_max":      e_peace_max_l,      # max |e| at peace cells (largest false alarm)
            "false_alarm_frac": false_alarm_frac_l, # frac of peace cells with |e|>0.5 (false alarms)
            # ── V22: spike diagnostics ──
            "spike_frac":       spike_frac_l,       # frac of event cells with |e|>2
            "spike_severe":     spike_severe_l,     # frac with |e|>4 (should decrease over training)
            "overpred_frac":    overpred_frac_l,    # frac of event cells where y_pred > y_true
            "overpred_mag":     overpred_mag_l,     # mean overprediction magnitude
            "underpred_frac":   underpred_frac_l,   # frac where y_pred < y_true
        }

        logger.debug(
            "SpotlightLossV22 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV22(non_zero_threshold={self.tau})"
