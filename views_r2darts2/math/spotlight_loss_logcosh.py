import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO + gentle eighth-root series boost (V20).
    Level = T × MSE on mean gap (V13 — unchanged, best calibration).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, **eighth-root series boost**,
      Hájek-normalised.

      V13 had some templating — rare-event countries got little Shape
      gradient → model learned common patterns for them.

      V18's sqrt boost (1.58× for templated) shifted Shape:Level
      balance → overprediction. V19's fourth-root (1.26×) + Huber
      constant also failed.

      V20 uses **eighth root**: 1.12× for templated countries (2.5×
      error ratio), 0.89× for well-fit (0.5× error ratio). Truly
      "a tiny boost" — barely 12% for the most templated countries.
      Still inside the Hájek → loss magnitude preserved.

      Boost comparison at 2.5× error ratio:
        V18 sqrt:      1.58×  (broke calibration)
        V19 4th root:  1.26×  (broke calibration)
        V20 8th root:  1.12×  (target: fixes templating without breaking)

    * **Level (DC magnitude).** ``T × gap²`` (symmetric MSE) on the
      per-series mean gap, gate-weighted, Hájek-normalised. UNCHANGED
      from V13 — byte-for-byte identical.

      V13 had the best calibration of all versions (0.84× overall,
      1.13× ch_0, 0.51× ch_1, 0.36× ch_2).

    ── Ranking: Safe Level alternatives to MSE ───────────────────────

    All options below are adaptive (no hardcoded constants). ``delta``
    = batch std of event-series |gap|, computed fresh each forward pass.

    1. **MSE** (current) — gradient ``2*gap``. Proportional, unbounded.
       Best calibration but chases extreme training events.
    2. **Adaptive Welsch** — ``delta²/2 * (1 - exp(-(gap/delta)²))``.
       Gradient ``gap * exp(-(gap/delta)²)``. MSE for small, exponentially
       decays for large. Most graceful extreme handling.
    3. **Adaptive Cauchy** — ``delta² * log(1 + (gap/delta)²)``.
       Gradient ``2*gap / (1 + (gap/delta)²)``. MSE for small, bounded
       at ``2*delta``. Smooth.
    4. **Adaptive Fair** — ``delta² * (|gap|/delta - log(1 + |gap|/delta))``.
       Gradient ``gap / (1 + |gap|/delta)``. Bounded at ``delta``. Simpler.
    5. **Adaptive asinh** — ``2*delta² * (sqrt(1 + (gap/delta)²) - 1)``.
       Gradient ``2*gap / sqrt(1 + (gap/delta)²)``. MSE for small, bounded
       at ``2*delta``. Similar to log_cosh but adaptive.
    6. **Adaptive Tukey biweight** — redescending (zero gradient for
       |gap| > delta). Most aggressive outlier rejection. May ignore
       too much.
    7. **Quantile-clipped MSE** — ``min(gap², p95²)``. Discontinuous
       gradient at p95. Crude.
    8. **Adaptive log-cosh** — ``delta * log_cosh(gap/delta)``.
       Gradient ``tanh(gap/delta)``. Saturates too early (V11 problem).

    Recommendation: keep MSE (#1). It has the best calibration. The
    overprediction in V18 came from the Shape boost, not from MSE.

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

        logger.info("SpotlightLossV20 | threshold=%.4f", non_zero_threshold)

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

        # ── SHAPE: log_cosh on demeaned errors, DRO + eighth-root boost
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

        # ── V20: eighth-root series boost (very gentle) ──────────────
        # V18 sqrt (0.5):  1.58× for templated → broke calibration
        # V19 4th root (0.25): 1.26× → broke calibration
        # V20 8th root (0.125): 1.12× → target: fixes templating without
        # breaking the Shape:Level balance.
        #
        # For 2.5× error ratio: 2.5^0.125 = 1.12
        # For 0.5× error ratio: 0.5^0.125 = 0.92
        # Barely perceptible — "a tiny boost."
        #
        # Detached, inside Hájek → loss magnitude preserved.
        series_err_mean = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev
        global_err_mean = (raw_abs * event_mask).sum() / event_mask.sum().clamp_min(1.0)
        series_boost = (series_err_mean / global_err_mean.clamp_min(1e-8)) ** 0.125
        series_boost = torch.nan_to_num(series_boost, nan=1.0, posinf=1.0, neginf=1.0)

        if multivariate:
            shape_w = gate * w_dro * series_boost
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            shape_w = gate * w_dro * series_boost
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: T × symmetric MSE on mean gap (V13 — UNCHANGED) ───
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * gap ** 2
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

                _ga    = gap.abs()
                gap_mean_l    = _ga.mean(dim=0).tolist()
                gap_max_l     = _ga.amax(dim=0).tolist()
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V20: series boost diagnostics ──
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B, C)
                _n_es = _ev_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
                _boost_per_series = series_boost.squeeze(-1) * _ev_series_mask  # (B, C)
                # Compute mean as tensor first (stays on same device), then
                # convert to list. Avoids CPU/CUDA mismatch from torch.tensor(list).
                _boost_mean_t = _boost_per_series.sum(dim=0) / _n_es  # (C,)
                boost_mean_l = _boost_mean_t.tolist()
                _boost_var = ((_boost_per_series ** 2).sum(dim=0) / _n_es
                              - _boost_mean_t ** 2).clamp_min(0)
                boost_std_l = _boost_var.sqrt().tolist()
                boost_max_l = (_boost_per_series.amax(dim=0)).tolist()
                boost_min_l = (((_boost_per_series + 1e8 * (1 - _ev_series_mask)).amin(dim=0))).tolist()
                boost_up_frac_l = (((series_boost.squeeze(-1) > 1.0).float() * _ev_series_mask).sum(dim=0)
                                   / _n_es).tolist()

                # Per-series error stats (what drives the boost)
                _err_per_series = series_err_mean.squeeze(-1) * _ev_series_mask  # (B, C)
                _err_mean_t = _err_per_series.sum(dim=0) / _n_es  # (C,)
                err_series_mean_l = _err_mean_t.tolist()
                _err_var = ((_err_per_series ** 2).sum(dim=0) / _n_es
                            - _err_mean_t ** 2).clamp_min(0)
                err_series_std_l = _err_var.sqrt().tolist()
                err_series_max_l = (_err_per_series.amax(dim=0)).tolist()

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
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l  = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()
                _n_es = _ev_series_mask.sum().clamp_min(1.0)
                _bps = series_boost.squeeze(1) if series_boost.dim() > 1 and series_boost.size(1) == 1 else series_boost
                _boost_per_series = _bps * _ev_series_mask
                boost_mean_l = [(_boost_per_series.sum() / _n_es).item()]
                _boost_var = ((_boost_per_series ** 2).sum() / _n_es - boost_mean_l[0] ** 2).clamp_min(0)
                boost_std_l = [_boost_var.sqrt().item()]
                boost_max_l = [_boost_per_series.max().item()]
                boost_min_l = [(_boost_per_series + 1e8 * (1 - _ev_series_mask)).min().item()]
                boost_up_frac_l = [(((_bps > 1.0).float() * _ev_series_mask).sum() / _n_es).item()]
                _eps_err = series_err_mean.squeeze(1) if series_err_mean.dim() > 1 and series_err_mean.size(1) == 1 else series_err_mean
                _err_per_series = _eps_err * _ev_series_mask
                err_series_mean_l = [(_err_per_series.sum() / _n_es).item()]
                _err_var = ((_err_per_series ** 2).sum() / _n_es - err_series_mean_l[0] ** 2).clamp_min(0)
                err_series_std_l = [_err_var.sqrt().item()]
                err_series_max_l = [_err_per_series.max().item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV20: per_channel={comp}")

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
            # ── V20: eighth-root series boost diagnostics ──
            # V18 sqrt: boost_max ~1.58 (broke calibration)
            # V19 4th root: boost_max ~1.26 (broke calibration)
            # V20 8th root: boost_max ~1.12 (target: fixes templating)
            # If boost_max > 1.2, the boost is still too aggressive.
            # If boost_max < 1.05, the boost is too gentle to matter.
            "boost_mean":     boost_mean_l,      # mean boost (should be ≈ 1.0)
            "boost_std":      boost_std_l,       # std of boost (V20 << V18)
            "boost_max":      boost_max_l,       # max boost (target: 1.10-1.15)
            "boost_min":      boost_min_l,       # min boost (target: 0.85-0.95)
            "boost_up_frac":  boost_up_frac_l,   # frac with boost > 1 (should be < 0.5)
            # Per-series error stats (templating signal)
            "err_series_mean": err_series_mean_l, # mean |e| per event series
            "err_series_std":  err_series_std_l,  # std of per-series |e| (templating signal)
            "err_series_max":  err_series_max_l,  # max per-series |e| (most templated)
        }

        logger.debug(
            "SpotlightLossV20 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV20(non_zero_threshold={self.tau})"
