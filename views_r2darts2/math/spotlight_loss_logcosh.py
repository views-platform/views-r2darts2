import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × MSE(gate-weighted event-mean gap) — V23.

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged from
      V13 — this component works.

    * **Level (DC magnitude).** ``T × gap²`` where ``gap`` is the
      **gate-weighted event-mean gap** (not the all-cell mean gap).

      V13 used ``gap = mean(y_pred) - mean(y_true)`` over ALL 36 cells.
      This diluted the gap by 31 peace cells. For 5 events at error 1.5:
        V13 gap = 5×1.5/36 = 0.21  (diluted 7×)
        V23 gap = 5×1.5/5  = 1.50  (undiluted)
      V23's gap is 7× larger → 50× larger gap² → much stronger DC push.

      V23 computes the mean over EVENT cells only, using the same gate
      as Shape (detached to avoid second-order gradients):
        gate_d = gate.detach()
        mean_pred_ev = sum(gate_d * y_pred) / sum(gate_d)
        mean_true_ev = sum(gate_d * y_true) / sum(gate_d)
        gap = mean_pred_ev - mean_true_ev

      Everything else is V13 exactly:
        level_cell = T * gap²
        w_level = gate.amax(dim=1)
        Hájek: (w_level * level_cell).sum / w_level.sum

      Why this is DC-only (no AC entanglement):
        gap is a SCALAR per series (weighted mean → scalar).
        d(gap)/d(y_pred[t]) = gate_d[t] / sum(gate_d)  (uniform for gate>0)
        d(T*gap²)/d(y_pred[t]) = 2*gap * gate_d[t]/sum(gate_d)
        This is UNIFORM across event cells → pure DC, zero AC.
        No Shape conflict (unlike V22's mean(e²) which had AC component).

      Why this solves underprediction:
        V13's diluted gap (0.21) gives gradient 2*0.21 = 0.42 per cell.
        V23's undiluted gap (1.50) gives gradient 2*1.50 = 3.0 per cell.
        7× stronger DC push → model can calibrate.

      Why this is robust to spikes:
        A single huge spike at an event cell raises mean_pred_ev, but
        only by spike/N_ev (not spike/36). The model doesn't need to
        lift ALL cells — just the event cells. Shape (with DRO) then
        distributes the mass correctly.

      Why gate is detached:
        Detaching avoids second-order gradients through the weighting.
        The gate is a function of y_pred (via abs_max), so without
        detach, the gap computation would create a feedback loop.
        Detach keeps the Level gradient flowing only through y_pred
        in the numerator, not through the gate weights.

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

        logger.info("SpotlightLossV23 | threshold=%.4f", non_zero_threshold)

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

        # ── LEVEL: T × MSE(gate-weighted event-mean gap) — V23 ───────
        # Replaces V13's T × mean(e)² (all-cell mean gap).
        #
        # Gate-weighted event mean: mean over EVENT cells only (gate > 0).
        # Detached gate avoids second-order gradients through weighting.
        # Gap is a SCALAR per series → DC-only → no Shape conflict.
        #
        # 7× larger gap than V13 (no peace-cell dilution) → 50× larger
        # gap² → 7× stronger DC gradient → fixes underprediction.
        gate_d = gate.detach()
        sum_g = gate_d.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pred_ev = (gate_d * y_pred).sum(dim=1, keepdim=True) / sum_g
        mean_true_ev = (gate_d * y_true).sum(dim=1, keepdim=True) / sum_g
        gap = (mean_pred_ev - mean_true_ev).squeeze(1)  # (B,) or (B, C)

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

                # Gap diagnostics — compare V23's event-gap to V13's all-cell gap
                gap_v13 = y_pred.mean(dim=1) - y_true.mean(dim=1)  # V13's diluted gap
                _ga     = gap.abs()         # V23's event-gap (undiluted)
                _ga_v13 = gap_v13.abs()     # V13's all-cell gap (diluted)
                gap_mean_l     = _ga.mean(dim=0).tolist()
                gap_max_l      = _ga.amax(dim=0).tolist()
                gap_v13_mean_l = _ga_v13.mean(dim=0).tolist()
                gap_v13_max_l  = _ga_v13.amax(dim=0).tolist()
                # Dilution factor: V23 gap / V13 gap. Should be ~7× (36/5).
                # If low, events are spread across many cells (less dilution).
                dilution_l = (_ga.mean(dim=0) / _ga_v13.mean(dim=0).clamp_min(1e-8)).tolist()

                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V23: event-mean diagnostics ──
                # What the gate-weighted event mean actually sees
                _mean_pred_ev = mean_pred_ev.squeeze(1)  # (B, C)
                _mean_true_ev = mean_true_ev.squeeze(1)  # (B, C)
                mean_pred_ev_l = (_mean_pred_ev.mean(dim=0)).tolist()
                mean_true_ev_l = (_mean_true_ev.mean(dim=0)).tolist()
                # Event cells per series (affects dilution)
                _ev_per_series = event_mask.sum(dim=1)  # (B, C)
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B, C)
                _n_es = _ev_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
                ev_per_series_l = ((_ev_per_series * _ev_series_mask).sum(dim=0) / _n_es).tolist()

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
                _mean_pred_ev = mean_pred_ev.squeeze(1) if mean_pred_ev.dim() > 1 else mean_pred_ev
                _mean_true_ev = mean_true_ev.squeeze(1) if mean_true_ev.dim() > 1 else mean_true_ev
                mean_pred_ev_l = [_mean_pred_ev.mean().item()]
                mean_true_ev_l = [_mean_true_ev.mean().item()]
                _ev_per_series = event_mask.sum(dim=1)
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()
                _n_es = _ev_series_mask.sum().clamp_min(1.0)
                ev_per_series_l = [((_ev_per_series * _ev_series_mask).sum() / _n_es).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV23: per_channel={comp}")

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
            "level_gap_mean": gap_mean_l,         # V23's event-gap (undiluted)
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "level_gap_sat":     gap_sat_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V23: dilution diagnostics ──
            # Compare V23's event-gap to V13's all-cell gap.
            # dilution = V23_gap / V13_gap ≈ 36/N_ev ≈ 7×
            # If dilution > 5, V13 was hiding 5× more error than it showed.
            "gap_v13_mean":   gap_v13_mean_l,     # V13's diluted gap (what V13 saw)
            "gap_v13_max":    gap_v13_max_l,
            "dilution":       dilution_l,         # V23_gap / V13_gap (should be ~7×)
            # ── V23: event-mean diagnostics ──
            "mean_pred_ev":   mean_pred_ev_l,     # gate-weighted mean of y_pred over events
            "mean_true_ev":   mean_true_ev_l,     # gate-weighted mean of y_true over events
            "ev_per_series":  ev_per_series_l,    # mean event cells per series (drives dilution)
        }

        logger.debug(
            "SpotlightLossV23 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV23(non_zero_threshold={self.tau})"
