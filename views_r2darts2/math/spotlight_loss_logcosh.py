import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = T × per-series Hájek of log_cosh DRO (V30 — rebalanced).
    Level = T × weighted dual-gap (V29 — unchanged).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** T-scaled per-series Hájek of demeaned
      log_cosh residual, gated, DRO-weighted.

      V13/V29 used per-cell Hájek (normalize over B×T = ~6400 event
      cells) with no T. Shape ≈ 1.0, gradient ≈ 0.006/cell.
      V29's Level ≈ 88, gradient ≈ 0.4/cell.
      Shape:Level = 1:88 → Shape completely starved → templating.

      V30 uses per-series Hájek (normalize over B = ~100 event series)
      WITH T factor. Shape ≈ T = 36, gradient ≈ T/N_series = 0.36/cell.
      Shape:Level = 36:88 ≈ 1:2.4 → balanced.

      Why per-series Hájek (not per-cell):
        Per-cell Hájek: gradient = w[t] × tanh / sum_BT(w) ≈ 1/6400 per cell
        Per-series Hájek: gradient = w[t] × tanh / (sum_T(w) × N_series) ≈ 1/500 per cell
        With T: per-series becomes T/N_series ≈ 0.36 per cell — comparable to Level

      Why T is safe here (unlike V14):
        V14's explosion: T × log_cosh(e_shape/ac_scale) per cell, gradient
        T × tanh / ac_scale ≈ 24/cell, compounded through 8 layers.
        V30: T × Hájek-normalized loss. Per-cell gradient = T × w[t] × tanh /
        (sum_T(w) × N_series) = T/N_series × (w[t]/sum_T(w)) × tanh.
        The (w[t]/sum_T(w)) term is ≤ 1 (Hájek-normalized within series),
        so per-cell gradient ≤ T/N_series ≈ 0.36. Bounded, no explosion.

    * **Level (DC magnitude, dual-gap).** V29 unchanged.
        T × [(n_ev/T) × gap_ev² + (n_peace/T) × gap_peace²]
      Catches false alarms (gap_peace) and underprediction (gap_event)
      without gaming (y_true-only masks). DC-dominant (76% in V29).

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

        logger.info("SpotlightLossV30 | threshold=%.4f", non_zero_threshold)

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
        n_event = mask_event.sum(dim=1).clamp_min(1.0)
        n_peace = mask_peace.sum(dim=1).clamp_min(1.0)

        # ── SHAPE: T × per-series Hájek of log_cosh DRO (V30) ─────────
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

        shape_w = gate * w_dro  # (B, T) or (B, T, C)

        # V30: per-series Hájek (normalize over T), then Hájek over series, then T-scale
        if multivariate:
            # Per-series mean: (B, T, C) → sum over T → (B, C)
            shape_per_series = (shape_w * shape_cell).sum(dim=1) / shape_w.sum(dim=1).clamp_min(self._EPS)
            # Event series mask: 1 if series has ANY event cell
            ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B, C)
            n_ev_series = ev_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
            # Hájek over event series, then T-scale
            loss_shape = T * (ev_series_mask * shape_per_series).sum(dim=0) / n_ev_series
        else:
            # Per-series mean: (B, T) → sum over T → (B,)
            shape_per_series = (shape_w * shape_cell).sum(dim=1) / shape_w.sum(dim=1).clamp_min(self._EPS)
            ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B,)
            n_ev_series = ev_series_mask.sum().clamp_min(1.0)
            loss_shape = T * (ev_series_mask * shape_per_series).sum() / n_ev_series

        # ── LEVEL: T × weighted dual-gap (V29 — unchanged) ────────────
        mean_pred_ev = (mask_event * y_pred).sum(dim=1) / n_event
        mean_true_ev = (mask_event * y_true).sum(dim=1) / n_event
        gap_event = mean_pred_ev - mean_true_ev

        mean_pred_peace = (mask_peace * y_pred).sum(dim=1) / n_peace
        mean_true_peace = (mask_peace * y_true).sum(dim=1) / n_peace
        gap_peace = mean_pred_peace - mean_true_peace

        w_ev = n_event / T
        w_peace = n_peace / T

        level_cell = T * (w_ev * gap_event ** 2 + w_peace * gap_peace ** 2)

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
                _n_ev = event_mask.sum(dim=(0, 1)).clamp_min(1.0)
                _w_ev = w_dro * event_mask
                _dm = _w_ev.sum(dim=(0, 1)) / _n_ev
                _dw2 = (_w_ev ** 2).sum(dim=(0, 1)) / _n_ev
                _dstd = (_dw2 - _dm ** 2).clamp_min(0).sqrt()
                dro_wmean_l   = _dm.tolist()
                dro_wstd_l    = _dstd.tolist()
                dro_wmax_l    = w_dro.amax(dim=(0, 1)).tolist()
                dro_frac_up_l = (((w_dro > 1.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                event_frac_l  = event_mask.mean(dim=(0, 1)).tolist()

                # V13 gap (for comparison)
                gap_v13 = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga_v13 = gap_v13.abs()
                gap_v13_mean_l = _ga_v13.mean(dim=0).tolist()

                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_v13_mean_l = ((_ga_v13 * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # Dual-gap diagnostics
                _gae = gap_event.abs()
                _gap = gap_peace.abs()
                gap_event_mean_l = _gae.mean(dim=0).tolist()
                gap_peace_mean_l = _gap.mean(dim=0).tolist()

                ev_frac_l = (n_event.mean(dim=0) / T).tolist()
                peace_frac_l = (n_peace.mean(dim=0) / T).tolist()

                mean_pred_ev_l = mean_pred_ev.mean(dim=0).tolist()
                mean_true_ev_l = mean_true_ev.mean(dim=0).tolist()
                mean_pred_peace_l = mean_pred_peace.mean(dim=0).tolist()
                mean_true_peace_l = mean_true_peace.mean(dim=0).tolist()

                _gap_peace_pos = (gap_peace > 0).float() * _ev_mask_s
                false_alarm_frac_l = (_gap_peace_pos.sum(dim=0) / _n_ev_s).tolist()
                _gap_ev_neg = (gap_event < 0).float() * _ev_mask_s
                underpred_ev_frac_l = (_gap_ev_neg.sum(dim=0) / _n_ev_s).tolist()

                sl_ratio_l = (loss_shape.detach() / loss_level.detach().clamp_min(self._EPS)).tolist()
            else:
                _n_ev = event_mask.sum().clamp_min(1.0)
                _w_ev = w_dro * event_mask
                _dm = (_w_ev.sum() / _n_ev).item()
                _dw2 = ((_w_ev ** 2).sum() / _n_ev).item()
                dro_wmean_l   = [_dm]
                dro_wstd_l    = [max(0.0, _dw2 - _dm ** 2) ** 0.5]
                dro_wmax_l    = [w_dro.max().item()]
                dro_frac_up_l = [((w_dro > 1.0) * event_mask).sum().item() / _n_ev.item()]
                event_frac_l  = [event_mask.mean().item()]
                gap_v13 = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga_v13 = gap_v13.abs()
                gap_v13_mean_l = [_ga_v13.mean().item()]
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_v13_mean_l = [((_ga_v13 * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                _gae = gap_event.abs()
                _gap = gap_peace.abs()
                gap_event_mean_l = [_gae.mean().item()]
                gap_peace_mean_l = [_gap.mean().item()]
                ev_frac_l = [(n_event.mean() / T).item()]
                peace_frac_l = [(n_peace.mean() / T).item()]
                mean_pred_ev_l = [mean_pred_ev.mean().item()]
                mean_true_ev_l = [mean_true_ev.mean().item()]
                mean_pred_peace_l = [mean_pred_peace.mean().item()]
                mean_true_peace_l = [mean_true_peace.mean().item()]
                _gap_peace_pos = (gap_peace > 0).float() * _ev_mask_s
                false_alarm_frac_l = [(_gap_peace_pos.sum() / _n_ev_s).item()]
                _gap_ev_neg = (gap_event < 0).float() * _ev_mask_s
                underpred_ev_frac_l = [(_gap_ev_neg.sum() / _n_ev_s).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV30: per_channel={comp}")

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
            "level_gap_mean": gap_v13_mean_l,
            "level_gap_ev_mean": gap_ev_v13_mean_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V29 dual-gap diagnostics ──
            "gap_event_mean":   gap_event_mean_l,
            "gap_peace_mean":   gap_peace_mean_l,
            "ev_frac":          ev_frac_l,
            "peace_frac":       peace_frac_l,
            "mean_pred_ev":     mean_pred_ev_l,
            "mean_true_ev":     mean_true_ev_l,
            "mean_pred_peace":  mean_pred_peace_l,
            "mean_true_peace":  mean_true_peace_l,
            "false_alarm_frac": false_alarm_frac_l,
            "underpred_ev_frac": underpred_ev_frac_l,
        }

        logger.debug(
            "SpotlightLossV30 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV30(non_zero_threshold={self.tau})"
