import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × weighted dual-gap on event/peace partition (V29).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged.

    * **Level (DC magnitude, two-distribution match).** ``T × [(n_ev/T) × gap_ev² + (n_peace/T) × gap_peace²]``
      where the masks are from ``y_true`` only (not gameable).

      V13 used ``T × mean(e)²`` — a single scalar gap over all cells.
      This is mathematically DC-only, but DC-only ⟹ L = φ(Σy_t), so
      any two trajectories with the same mean are indistinguishable.
      The model satisfies V13 by spiking at event cells on training
      (gap ≈ 0) but spikes misalign on eval → overprediction.

      V29 separates the gap into event and peace components:
        gap_event = mean(y_pred at true events) - mean(y_true at true events)
        gap_peace = mean(y_pred at true peace) - mean(y_true at true peace)
        L = T × [(n_event/T) × gap_event² + (n_peace/T) × gap_peace²]

      This matches the two-point distribution (event mean, peace mean)
      without sorting — no temporal blindness. The event/peace partition
      is the natural structure for zero-inflated data.

      Why this escapes mean obfuscation:
        pred = [5, 0, 0, 0]  true = [0, 0, 0, 0]  (false alarm)
        V13: gap = 1.25 (looks like calibration)
        V29: gap_peace = 1.67 (isolates the false alarm) → penalized

      Why the gradient is DC-dominant:
        grad at event cell: 2 × gap_event (uniform across event cells)
        grad at peace cell: 2 × gap_peace (uniform across peace cells)
        The only AC is the single event/peace partition — one bit of
        structure, not noisy per-cell AC. This is the minimum AC needed
        to escape the DC-only = mean-only limitation.

      Why no conflict (unlike V27):
        V27's level_cell pushed ALL cells up (including peace), while
        peace_cell pushed peace cells down → conflict.
        V29's gap_event only pushes event cells, gap_peace only pushes
        peace cells → no overlap, no conflict.

      Masks from y_true only → not gameable (fixes V23).
      gap_peace catches false alarms → not blind to overpred (fixes V24).
      Gradient uniform within each group → DC-dominant (fixes V22/V25).

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

        logger.info("SpotlightLossV29 | threshold=%.4f", non_zero_threshold)

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

        # ── True event/peace masks (for Level — y_true only, not gameable) ─
        mask_event = (y_true.abs() > self.tau).float()
        mask_peace = 1.0 - mask_event
        n_event = mask_event.sum(dim=1).clamp_min(1.0)   # (B,) or (B, C)
        n_peace = mask_peace.sum(dim=1).clamp_min(1.0)   # (B,) or (B, C)

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

        # ── LEVEL: T × weighted dual-gap (V29) ────────────────────────
        # Event gap: mean(y_pred at true events) - mean(y_true at true events)
        # Peace gap: mean(y_pred at true peace) - mean(y_true at true peace)
        # Weighted by cell fraction (n_event/T, n_peace/T).
        #
        # Gradient at event cell: 2 × gap_event (uniform across events)
        # Gradient at peace cell: 2 × gap_peace (uniform across peace)
        # DC-dominant within each group; only AC is the event/peace split.
        mean_pred_ev = (mask_event * y_pred).sum(dim=1) / n_event
        mean_true_ev = (mask_event * y_true).sum(dim=1) / n_event
        gap_event = mean_pred_ev - mean_true_ev                       # (B,) or (B, C)

        mean_pred_peace = (mask_peace * y_pred).sum(dim=1) / n_peace
        mean_true_peace = (mask_peace * y_true).sum(dim=1) / n_peace
        gap_peace = mean_pred_peace - mean_true_peace                 # (B,) or (B, C)

        # Weights: cell fractions (n_event/T, n_peace/T)
        w_ev = n_event / T
        w_peace = n_peace / T

        level_cell = T * (w_ev * gap_event ** 2 + w_peace * gap_peace ** 2)

        # Hájek normalization (same structure as V13)
        w_level = gate.amax(dim=1)  # per-series event mass

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

                # V13's all-cell gap (for comparison)
                gap_v13 = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga_v13 = gap_v13.abs()
                gap_v13_mean_l = _ga_v13.mean(dim=0).tolist()
                gap_v13_max_l  = _ga_v13.amax(dim=0).tolist()

                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_v13_mean_l = ((_ga_v13 * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_v13_max_l  = ((_ga_v13 * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga_v13 > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V29: dual-gap diagnostics ──
                _gae = gap_event.abs()
                _gap = gap_peace.abs()
                gap_event_mean_l = _gae.mean(dim=0).tolist()
                gap_event_max_l  = _gae.amax(dim=0).tolist()
                gap_peace_mean_l = _gap.mean(dim=0).tolist()
                gap_peace_max_l  = _gap.amax(dim=0).tolist()

                # Event/peace fractions
                ev_frac_l = (n_event.mean(dim=0) / T).tolist()
                peace_frac_l = (n_peace.mean(dim=0) / T).tolist()

                # Mean pred/true at event and peace cells
                mean_pred_ev_l = mean_pred_ev.mean(dim=0).tolist()
                mean_true_ev_l = mean_true_ev.mean(dim=0).tolist()
                mean_pred_peace_l = mean_pred_peace.mean(dim=0).tolist()
                mean_true_peace_l = mean_true_peace.mean(dim=0).tolist()

                # Sign analysis: how many event series are under/overpredicting?
                _gap_ev_neg = (gap_event < 0).float() * _ev_mask_s
                _gap_ev_pos = (gap_event > 0).float() * _ev_mask_s
                underpred_ev_frac_l = (_gap_ev_neg.sum(dim=0) / _n_ev_s).tolist()
                overpred_ev_frac_l = (_gap_ev_pos.sum(dim=0) / _n_ev_s).tolist()

                # Same for peace
                _gap_peace_pos = (gap_peace > 0).float() * _ev_mask_s  # overpred at peace = false alarms
                _gap_peace_neg = (gap_peace < 0).float() * _ev_mask_s
                false_alarm_frac_l = (_gap_peace_pos.sum(dim=0) / _n_ev_s).tolist()
                underpred_peace_frac_l = (_gap_peace_neg.sum(dim=0) / _n_ev_s).tolist()

                # Obfuscation detection: V13 gap vs V29 gaps
                # If |gap_v13| << |gap_event| or |gap_peace|, V13 was hiding error
                _obf_ev = _gae / _ga_v13.clamp_min(1e-8)
                _obf_peace = _gap / _ga_v13.clamp_min(1e-8)
                obf_ev_mean_l = (_obf_ev * _ev_mask_s).sum(dim=0).div(_n_ev_s).tolist()
                obf_peace_mean_l = (_obf_peace * _ev_mask_s).sum(dim=0).div(_n_ev_s).tolist()

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
                _ga_v13 = gap_v13.abs()
                gap_v13_mean_l = [_ga_v13.mean().item()]
                gap_v13_max_l  = [_ga_v13.max().item()]
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_v13_mean_l = [((_ga_v13 * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_v13_max_l  = [((_ga_v13 * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga_v13 > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                _gae = gap_event.abs()
                _gap = gap_peace.abs()
                gap_event_mean_l = [_gae.mean().item()]
                gap_event_max_l  = [_gae.max().item()]
                gap_peace_mean_l = [_gap.mean().item()]
                gap_peace_max_l  = [_gap.max().item()]
                ev_frac_l = [(n_event.mean() / T).item()]
                peace_frac_l = [(n_peace.mean() / T).item()]
                mean_pred_ev_l = [mean_pred_ev.mean().item()]
                mean_true_ev_l = [mean_true_ev.mean().item()]
                mean_pred_peace_l = [mean_pred_peace.mean().item()]
                mean_true_peace_l = [mean_true_peace.mean().item()]
                _gap_ev_neg = (gap_event < 0).float() * _ev_mask_s
                _gap_ev_pos = (gap_event > 0).float() * _ev_mask_s
                underpred_ev_frac_l = [(_gap_ev_neg.sum() / _n_ev_s).item()]
                overpred_ev_frac_l = [(_gap_ev_pos.sum() / _n_ev_s).item()]
                _gap_peace_pos = (gap_peace > 0).float() * _ev_mask_s
                _gap_peace_neg = (gap_peace < 0).float() * _ev_mask_s
                false_alarm_frac_l = [(_gap_peace_pos.sum() / _n_ev_s).item()]
                underpred_peace_frac_l = [(_gap_peace_neg.sum() / _n_ev_s).item()]
                _obf_ev = _gae / _ga_v13.clamp_min(1e-8)
                _obf_peace = _gap / _ga_v13.clamp_min(1e-8)
                obf_ev_mean_l = [((_obf_ev * _ev_mask_s).sum() / _n_ev_s).item()]
                obf_peace_mean_l = [((_obf_peace * _ev_mask_s).sum() / _n_ev_s).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV29: per_channel={comp}")

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
            "level_gap_mean": gap_v13_mean_l,        # V13's all-cell gap (for comparison)
            "level_gap_max":  gap_v13_max_l,
            "level_gap_ev_mean": gap_ev_v13_mean_l,
            "level_gap_ev_max":  gap_ev_v13_max_l,
            "level_gap_sat":     gap_sat_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V29: dual-gap diagnostics ──
            # The two gaps V29 actually optimizes. Compare to V13's gap:
            # If gap_event >> gap_v13, V13 was hiding event underprediction.
            # If gap_peace >> gap_v13, V13 was hiding peace overprediction (false alarms).
            "gap_event_mean":   gap_event_mean_l,     # mean |gap| at true event cells
            "gap_event_max":    gap_event_max_l,
            "gap_peace_mean":   gap_peace_mean_l,     # mean |gap| at true peace cells
            "gap_peace_max":    gap_peace_max_l,
            # Event/peace fractions
            "ev_frac":          ev_frac_l,            # fraction of cells that are true events
            "peace_frac":       peace_frac_l,
            # Mean pred/true at each partition — should converge
            "mean_pred_ev":     mean_pred_ev_l,       # mean y_pred at true events
            "mean_true_ev":     mean_true_ev_l,       # mean y_true at true events
            "mean_pred_peace":  mean_pred_peace_l,    # mean y_pred at true peace (should → 0)
            "mean_true_peace":  mean_true_peace_l,    # mean y_true at true peace (≈ 0)
            # Sign analysis — calibration direction
            "underpred_ev_frac":   underpred_ev_frac_l,   # frac event series underpredicting events
            "overpred_ev_frac":    overpred_ev_frac_l,    # frac event series overpredicting events
            "false_alarm_frac":    false_alarm_frac_l,    # frac event series overpredicting at peace (FALSE ALARMS)
            "underpred_peace_frac": underpred_peace_frac_l,
            # Obfuscation: V29 gap / V13 gap. If >1, V13 was hiding this error.
            "obf_ev_mean":         obf_ev_mean_l,        # event obfuscation (V29 sees more than V13)
            "obf_peace_mean":      obf_peace_mean_l,     # peace obfuscation (V29 sees more than V13)
        }

        logger.debug(
            "SpotlightLossV29 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV29(non_zero_threshold={self.tau})"
