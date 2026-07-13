import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = T × per-series Hájek of log_cosh DRO (V30 — unchanged).
    Level = T × unweighted dual-gap (V31 — fixes V29/V30 structural zero bias).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** T-scaled per-series Hájek of demeaned
      log_cosh residual, gated, DRO-weighted. Unchanged from V30.

    * **Level (DC magnitude, unweighted dual-gap).** ``T × (gap_event² + gap_peace²) / 2``
      where masks are from ``y_true`` only (not gameable).

      V29/V30 used ``T × [(n_ev/T) × gap_ev² + (n_peace/T) × gap_peace²]``.
      The n/T weighting gave peace cells 97% of the loss (n_peace >> n_ev).
      This created a STRUCTURAL ZERO BIAS: the equilibrium prediction was
      c ≈ 0.14 (near zero) instead of c ≈ 2.5 (midpoint between event
      and peace means). The model was incentivized to predict near-zero
      everywhere to minimize gap_peace², even though this made gap_event²
      large. Underprediction was the optimal strategy.

      V31 drops the n/T weighting:
        level_cell = T × (gap_event² + gap_peace²) / 2

      The /2 is the standard mean normalization (average of 2 terms),
      not a hyperparameter.

      Why this eliminates the zero bias:
        For uniform y_pred = c:
          gap_event = c - μ_event, gap_peace = c - μ_peace
          level = T × ((c-μ_event)² + (c-μ_peace)²) / 2
          d/dc = T × ((c-μ_event) + (c-μ_peace)) = 0
          → c = (μ_event + μ_peace) / 2  (midpoint, NOT zero)

      Per-cell gradient comparison:
        V29 weighted: event=2×gap_ev, peace=2×gap_peace (equal per cell)
          → peace dominates by majority (32 cells vs 4) → zero bias
        V31 unweighted: event=T×gap_ev/n_ev, peace=T×gap_peace/n_peace
          → event cells get T/n_ev ≈ 9× more gradient per cell
          → total event push : peace push = 9:1 (events drive calibration)

      Why Shape and Level stop fighting:
        At event cells: Shape (DRO) pushes UP, Level (gap_event < 0) pushes UP
          → ALIGNED, not fighting
        At peace cells: Shape may push up (false alarm), Level pushes DOWN
          → Level correctly opposes false alarms (weak but right direction)

      V30's tug-of-war was caused by the weighted dual-gap giving equal
      per-cell gradients at events and peace. The model compromised to
      flat-low. V31's unweighted version makes event push 72× stronger
      per cell → Level dominates at events → no compromise needed.

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

        logger.info("SpotlightLossV31 | threshold=%.4f", non_zero_threshold)

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

        # ── Event gate (for Shape) ───────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── True event/peace masks (for Level — y_true only) ─────────
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

        shape_w = gate * w_dro

        if multivariate:
            shape_per_series = (shape_w * shape_cell).sum(dim=1) / shape_w.sum(dim=1).clamp_min(self._EPS)
            ev_series_mask = (event_mask.sum(dim=1) > 0).float()
            n_ev_series = ev_series_mask.sum(dim=0).clamp_min(1.0)
            loss_shape = T * (ev_series_mask * shape_per_series).sum(dim=0) / n_ev_series
        else:
            shape_per_series = (shape_w * shape_cell).sum(dim=1) / shape_w.sum(dim=1).clamp_min(self._EPS)
            ev_series_mask = (event_mask.sum(dim=1) > 0).float()
            n_ev_series = ev_series_mask.sum().clamp_min(1.0)
            loss_shape = T * (ev_series_mask * shape_per_series).sum() / n_ev_series

        # ── LEVEL: T × unweighted dual-gap (V31) ──────────────────────
        # V29/V30 used n/T weighting → structural zero bias (c ≈ 0.14).
        # V31 drops the weighting → equilibrium at midpoint (c ≈ 2.5).
        #
        # Per-cell gradient:
        #   event: T × gap_event / n_event (STRONG, few cells)
        #   peace: T × gap_peace / n_peace (weak, many cells)
        # Events drive calibration 9:1 over peace. No zero bias.
        mean_pred_ev = (mask_event * y_pred).sum(dim=1) / n_event
        mean_true_ev = (mask_event * y_true).sum(dim=1) / n_event
        gap_event = mean_pred_ev - mean_true_ev

        mean_pred_peace = (mask_peace * y_pred).sum(dim=1) / n_peace
        mean_true_peace = (mask_peace * y_true).sum(dim=1) / n_peace
        gap_peace = mean_pred_peace - mean_true_peace

        # Unweighted: (gap_event² + gap_peace²) / 2 — mean of 2 terms
        level_cell = T * (gap_event ** 2 + gap_peace ** 2) / 2.0

        w_level = gate.amax(dim=1)

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

                gap_v13 = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga_v13 = gap_v13.abs()
                gap_v13_mean_l = _ga_v13.mean(dim=0).tolist()

                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_v13_mean_l = ((_ga_v13 * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

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

                # V31: per-cell gradient magnitude comparison
                # event: T * |gap_event| / n_event
                # peace: T * |gap_peace| / n_peace
                _ev_grad_per_cell = T * _gae / n_event
                _peace_grad_per_cell = T * _gap / n_peace
                ev_grad_per_cell_l = (_ev_grad_per_cell.mean(dim=0)).tolist()
                peace_grad_per_cell_l = (_peace_grad_per_cell.mean(dim=0)).tolist()
                # Ratio: how much stronger is event push vs peace push per cell?
                # Should be >> 1 (events dominate)
                grad_ratio_l = (_ev_grad_per_cell.mean(dim=0)
                                / _peace_grad_per_cell.mean(dim=0).clamp_min(1e-8)).tolist()

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
                _ev_grad_per_cell = T * _gae / n_event
                _peace_grad_per_cell = T * _gap / n_peace
                ev_grad_per_cell_l = [_ev_grad_per_cell.mean().item()]
                peace_grad_per_cell_l = [_peace_grad_per_cell.mean().item()]
                grad_ratio_l = [(_ev_grad_per_cell.mean()
                                 / max(1e-8, _peace_grad_per_cell.mean())).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV31: per_channel={comp}")

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
            # ── V31 dual-gap diagnostics ──
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
            # ── V31: per-cell gradient comparison ──
            # The KEY diagnostic: event push should >> peace push per cell.
            # V29/V30 had these equal (→ zero bias).
            # V31 should have ev_grad >> peace_grad (→ no zero bias).
            "ev_grad_per_cell":     ev_grad_per_cell_l,
            "peace_grad_per_cell":  peace_grad_per_cell_l,
            "grad_ratio":           grad_ratio_l,  # ev/peace per-cell ratio (should be >> 1)
        }

        logger.debug(
            "SpotlightLossV31 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV31(non_zero_threshold={self.tau})"
