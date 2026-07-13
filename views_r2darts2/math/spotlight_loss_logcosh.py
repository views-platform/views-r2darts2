import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × MSE(mean gap) with series-level DRO (V32 — improves on V13).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. EXACTLY V13.

    * **Level (DC magnitude).** V13's ``T × gap²`` with **series-level
      DRO** that upweights poorly-calibrated series.

      V13 used uniform Hájek weighting (w_level = gate.amax). All event
      series got equal weight regardless of calibration error. Sparse
      channels (ch_1, ch_2) with large gaps got the same gradient share
      as well-calibrated channels (ch_0) with small gaps → ch_1/ch_2
      stuck at 0.51×/0.36×.

      V32 adds series-level DRO:
        w_series = sqrt(|gap| / mean|gap|)  (detached, per-channel)
        w_level = gate.amax × w_series

      Series with 2× the mean gap get sqrt(2) ≈ 1.41× more weight.
      Series with 0.5× the mean gap get 0.71× less weight. The sqrt
      dampens extremes. The Hájek normalization preserves loss scale.

      Why this is safe (preserves V13's strengths):
        1. Still DC-only: gradient is uniform within each series (same
           value at all T cells). No AC introduced. No Shape conflict.
        2. No n_ev instability: no per-cell normalization. The DRO
           operates on per-series |gap| (a scalar), not per-cell errors.
        3. No location leakage: w_series is per-series, not per-cell.
           The model can't learn WHERE events are from Level.
        4. No new constants: DRO is fully adaptive (sqrt of ratio to
           batch mean). No thresholds, no hyperparameters.
        5. Loss scale preserved: Hájek normalization absorbs the DRO
           weight. Shape:Level ratio unchanged from V13.

      Why this helps ch_1/ch_2:
        These channels have large gaps (0.51×, 0.36× calibration) but
        few event series. V13's uniform weighting gives them the same
        gradient share as ch_0 (which has more series but smaller gaps).
        V32's DRO upweights the poorly-calibrated series → stronger
        calibration push where it's needed most.

      What this does NOT fix:
        Mean obfuscation (DC-only = mean-only, mathematical necessity).
        Eval overprediction (generalization problem, not a loss problem).
        These require Shape or model-level changes, not Level changes.

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

        logger.info("SpotlightLossV32 | threshold=%.4f", non_zero_threshold)

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
        # EXACTLY V13 — unchanged.
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

        # ── LEVEL: T × MSE(mean gap) with series-level DRO (V32) ──────
        # V13's pure DC Level + adaptive series-level weighting.
        #
        # gap = mean(y_pred) - mean(y_true) per series (scalar, DC-only)
        # w_series = sqrt(|gap| / mean|gap|)  (detached, per-channel)
        # w_level = gate.amax × w_series
        # level_cell = T × gap²
        # loss = (w_level × level_cell).sum / w_level.sum  (Hájek)
        #
        # Gradient: uniform within series (DC), reweighted across series.
        # Series with large |gap| get more gradient → stronger calibration
        # push on poorly-calibrated channels (ch_1, ch_2).
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)

        # Series-level DRO: upweight poorly-calibrated series
        gap_abs = gap.abs().detach()  # (B,) or (B, C)
        if multivariate:
            gap_mu = gap_abs.mean(dim=0, keepdim=True).clamp_min(1e-8)  # (1, C)
        else:
            gap_mu = gap_abs.mean().clamp_min(1e-8)  # scalar
        w_series = torch.sqrt(gap_abs / gap_mu)  # (B,) or (B, C)
        # Normalize so mean weight = 1 (preserves loss scale)
        if multivariate:
            w_series = w_series / w_series.mean(dim=0, keepdim=True).clamp_min(1e-8)
        else:
            w_series = w_series / w_series.mean().clamp_min(1e-8)

        level_cell = T * gap ** 2
        w_level = gate.amax(dim=1) * w_series  # event mass × series DRO

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
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V32: series-level DRO diagnostics ──
                w_series_mean_l = w_series.mean(dim=0).tolist()  # should be ≈ 1.0
                w_series_max_l = w_series.amax(dim=0).tolist()   # most upweighted series
                # For event series only
                _ws_ev = w_series * _ev_mask_s
                w_series_ev_mean_l = (_ws_ev.sum(dim=0) / _n_ev_s).tolist()
                w_series_ev_max_l = (_ws_ev.amax(dim=0)).tolist()
                # Fraction of event series upweighted (w_series > 1)
                dro_series_up_l = (((w_series > 1.0).float() * _ev_mask_s).sum(dim=0)
                                   / _n_ev_s).tolist()
                # Gap distribution: how variable are gaps across series?
                # High std means some series are much worse → DRO matters more
                _gap_ev = _ga * _ev_mask_s
                _gap_ev_mean_t = _gap_ev.sum(dim=0) / _n_ev_s  # (C,)
                _gap_ev_var = ((_gap_ev ** 2).sum(dim=0) / _n_ev_s
                               - _gap_ev_mean_t ** 2).clamp_min(0)
                gap_ev_std_l = _gap_ev_var.sqrt().tolist()
                # Coefficient of variation (std/mean) — if high, DRO is doing work
                gap_cv_l = (_gap_ev_var.sqrt() / _gap_ev_mean_t.clamp_min(1e-8)).tolist()

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
                _ga    = gap.abs()
                gap_mean_l    = [_ga.mean().item()]
                gap_max_l     = [_ga.max().item()]
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l  = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                w_series_mean_l = [w_series.mean().item()]
                w_series_max_l = [w_series.max().item()]
                _ws_ev = w_series * _ev_mask_s
                w_series_ev_mean_l = [(_ws_ev.sum() / _n_ev_s).item()]
                w_series_ev_max_l = [_ws_ev.max().item()]
                dro_series_up_l = [(((w_series > 1.0).float() * _ev_mask_s).sum() / _n_ev_s).item()]
                _gap_ev = _ga * _ev_mask_s
                _gap_ev_mean_t = _gap_ev.sum() / _n_ev_s
                _gap_ev_var = ((_gap_ev ** 2).sum() / _n_ev_s - _gap_ev_mean_t ** 2).clamp_min(0)
                gap_ev_std_l = [_gap_ev_var.sqrt().item()]
                gap_cv_l = [(_gap_ev_var.sqrt() / max(1e-8, _gap_ev_mean_t.item())).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV32: per_channel={comp}")

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
            # ── DRO diagnostics (Shape) ──
            "dro_w_mean":     dro_wmean_l,
            "dro_w_std":      dro_wstd_l,
            "dro_w_max":      dro_wmax_l,
            "dro_frac_up":    dro_frac_up_l,
            "event_frac":     event_frac_l,
            # ── Gap diagnostics (Level) ──
            "level_gap_mean": gap_mean_l,
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "level_gap_sat":     gap_sat_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V32: series-level DRO diagnostics ──
            # The series-level DRO weight. Mean should be ≈ 1.0 (normalized).
            # Max shows the most upweighted series (should be 1.5-2.5).
            # If max > 3, some series have extreme gaps (outliers).
            "w_series_mean":       w_series_mean_l,       # mean DRO weight (should be ≈ 1.0)
            "w_series_max":        w_series_max_l,        # max DRO weight (most upweighted)
            "w_series_ev_mean":    w_series_ev_mean_l,    # mean over event series
            "w_series_ev_max":     w_series_ev_max_l,     # max over event series
            "dro_series_up":       dro_series_up_l,       # frac event series with w > 1 (upweighted)
            # Gap variability — if high, DRO is doing important work
            # (some series much worse than others)
            "gap_ev_std":          gap_ev_std_l,          # std of |gap| over event series
            "gap_cv":              gap_cv_l,              # coefficient of variation (std/mean)
                                                          # >0.5 = high variability = DRO matters
        }

        logger.debug(
            "SpotlightLossV32 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV32(non_zero_threshold={self.tau})"
