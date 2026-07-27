import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """SpotlightLossLogcosh — PGM/CM robust loss with guarded background demeaning.

    Design:
    - Two-mask gate: event_mask (y_true-based) for structure, gate (abs_max-based)
      for shape weighting.
    - Background-referenced demeaning (guarded by has_event).
    - Guarded DRO (no NaN for no-event series).
    - Decomposed level loss: gap_event + gap_non_event (prevents mean overshoot).
    - Series-level normalization for shape loss.
    """
    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _tolist(x):
        """Normalize tensor .tolist() to always return a list.
        0-d tensor .tolist() returns a float; 1-d returns a list."""
        val = x.tolist() if isinstance(x, torch.Tensor) else x
        return val if isinstance(val, list) else [val]

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

        # ── Two-mask gate ────────────────────────────────────────────
        event_mask = (y_true.abs() > self.tau).float()
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = (abs_max > self.tau).float()

        n_ev_raw = event_mask.sum(dim=1, keepdim=True)
        has_event = (n_ev_raw > 0.5).float()
        n_ev = n_ev_raw.clamp_min(1.0)

        # ── SHAPE: background-referenced demeaning (guarded) ─────────
        bg_mask = 1.0 - event_mask
        n_bg = bg_mask.sum(dim=1, keepdim=True)
        has_bg = (n_bg > 0.5).float()
        n_bg_safe = n_bg.clamp_min(1.0)

        e_bg_mean = (bg_mask * e).sum(dim=1, keepdim=True) / n_bg_safe
        e_ev_mean = (event_mask * e).sum(dim=1, keepdim=True) / n_ev

        e_mean = has_event * (has_bg * e_bg_mean + (1.0 - has_bg) * e_ev_mean)
        e_shape = e - e_mean
        shape_cell = self._log_cosh(e_shape)

        # ── DRO weighting (guarded) ──────────────────────────────────
        raw_abs = e_shape.abs().detach()
        dro_mu = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev
        valid_dro = (raw_abs > 1e-6).float() * event_mask
        w_dro_raw = torch.sqrt((raw_abs * valid_dro) / dro_mu.clamp_min(self._EPS))
        w_dro_mean = (w_dro_raw * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = w_dro_raw / w_dro_mean.clamp_min(self._EPS)
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=1.0, neginf=0.0)
        w_dro = has_event * (1.0 + event_mask * (w_dro - 1.0)) + (1.0 - has_event) * 1.0

        shape_w = gate * w_dro

        # ── Shape loss: series-level normalization ───────────────────
        if multivariate:
            num = (shape_w * shape_cell).sum(dim=(0, 1))
            den = shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
            per_series = num / den
            has_gated = (shape_w.sum(dim=(0, 1)) > self._EPS).float()
            loss_shape = (per_series * has_gated).sum() / has_gated.sum().clamp_min(1.0)
        else:
            num = (shape_w * shape_cell).sum(dim=1)
            den = shape_w.sum(dim=1).clamp_min(self._EPS)
            per_series = num / den
            has_gated = (shape_w.sum(dim=1) > self._EPS).float()
            loss_shape = (per_series * has_gated).sum() / has_gated.sum().clamp_min(1.0)

        # ── LEVEL: background term (population-wide) + event term
        #           (normalized by event-bearing series only, mirroring Shape) ──
        n_non_ev = (T - n_ev_raw).clamp_min(1.0)
        e_non_event_mean = (bg_mask * e).sum(dim=1, keepdim=True) / n_non_ev

        gap_non_event = e_non_event_mean.squeeze(1)              # (B, C) or (B,)
        gap_event_raw = e_ev_mean.squeeze(1)
        has_event_flat = has_event.squeeze(1)                    # (B, C) or (B,)

        level_bg_cell = self._log_cosh(gap_non_event)
        level_ev_cell = self._log_cosh(gap_event_raw) * has_event_flat

        if multivariate:
            loss_level_bg = level_bg_cell.mean(dim=0)
            n_event_series = has_event_flat.sum(dim=0).clamp_min(1.0)
            loss_level_ev = level_ev_cell.sum(dim=0) / n_event_series
            loss_level = T * (loss_level_bg + loss_level_ev).sum()
        else:
            loss_level_bg = level_bg_cell.mean()
            n_event_series = has_event_flat.sum().clamp_min(1.0)
            loss_level_ev = level_ev_cell.sum() / n_event_series
            loss_level = T * (loss_level_bg + loss_level_ev)

        # ── Diagnostic telemetry ─────────────────────────────────────
        with torch.no_grad():
            if multivariate:
                _n_ev = event_mask.sum(dim=(0, 1)).clamp_min(1.0)
                _w_ev = w_dro * event_mask
                _dm = _w_ev.sum(dim=(0, 1)) / _n_ev
                _dw2 = (_w_ev ** 2).sum(dim=(0, 1)) / _n_ev
                _dstd = (_dw2 - _dm ** 2).clamp_min(0).sqrt()
                dro_wmean_l = self._tolist(_dm)
                dro_wstd_l = self._tolist(_dstd)
                dro_wmax_l = self._tolist(w_dro.amax(dim=(0, 1)))
                dro_frac_up_l = self._tolist(((w_dro > 1.0) * event_mask).sum(dim=(0, 1)) / _n_ev)
                event_frac_l = self._tolist(event_mask.mean(dim=(0, 1)))

                gap_event_tel = has_event.squeeze(1) * e_ev_mean.squeeze(1)
                _ga = gap_event_tel.abs()
                gap_mean_l = self._tolist(_ga.mean(dim=0))
                gap_max_l = self._tolist(_ga.amax(dim=0))
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = self._tolist((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s)
                gap_ev_max_l = self._tolist((_ga * _ev_mask_s).amax(dim=0))
                gap_sat_l = self._tolist(((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s)
                shape_dc_l = self._tolist((gate * e_shape).mean(dim=1).abs().mean(dim=0))

                sl_ratio_l = self._tolist(loss_shape.detach() / loss_level.detach().clamp_min(self._EPS))
            else:
                _n_ev = event_mask.sum().clamp_min(1.0)
                _w_ev = w_dro * event_mask
                _dm = (_w_ev.sum() / _n_ev).item()
                _dw2 = ((_w_ev ** 2).sum() / _n_ev).item()
                dro_wmean_l = [_dm]
                dro_wstd_l = [max(0.0, _dw2 - _dm ** 2) ** 0.5]
                dro_wmax_l = [w_dro.max().item()]
                dro_frac_up_l = [((w_dro > 1.0) * event_mask).sum().item() / _n_ev.item()]
                event_frac_l = [event_mask.mean().item()]

                gap_event_tel = has_event.squeeze(1) * e_ev_mean.squeeze(1)
                _ga = gap_event_tel.abs()
                gap_mean_l = [_ga.mean().item()]
                gap_max_l = [_ga.max().item()]
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l = [(gate * e_shape).mean(dim=1).abs().mean().item()]

                sl_ratio_l = [float((loss_shape.detach() / loss_level.detach().clamp_min(self._EPS)).item())]

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
            "dro_w_mean": dro_wmean_l,
            "dro_w_std": dro_wstd_l,
            "dro_w_max": dro_wmax_l,
            "dro_frac_up": dro_frac_up_l,
            "event_frac": event_frac_l,
            "level_gap_mean": gap_mean_l,
            "level_gap_max": gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max": gap_ev_max_l,
            "level_gap_sat": gap_sat_l,
            "shape_dc": shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
        }

        logger.debug(
            "SpotlightLossLogcosh | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.tau})"