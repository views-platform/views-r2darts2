import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO + series boost (V18 — fixes templating).
    Level = T × symmetric MSE on mean gap (V13 — best calibration).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, **series-boosted**,
      Hájek-normalised.

      V13's global Hájek gave each series Shape gradient proportional
      to its event count. Rare-event countries (the templating victims)
      got the least gradient → model learned a common pattern for them.

      V17 tried per-series Hájek (each event country gets equal total
      gradient) but this changed the total Shape gradient magnitude,
      shifting the Shape:Level balance → Level got weaker →
      underprediction regressed.

      V18 keeps V13's global Hájek EXACTLY (preserves loss scale and
      Shape:Level balance) but adds a per-series BOOST inside the
      Hájek weighting:

        series_boost = sqrt(mean_|e|_series / mean_|e|_global)

      This multiplies shape_w for each series. High-error (templated)
      series get a boost, low-error (well-fit) series get a reduction.
      The sqrt prevents over-weighting.

      Because the boost is inside the Hájek (weighted mean), the loss
      MAGNITUDE is preserved — it's still a normalized weighted mean.
      Only the gradient DISTRIBUTION shifts: templated countries get
      more gradient, well-fit countries get less.

      Effect on per-series gradient (batch with 50 event series,
      640 total events, mean 12.8 events/series):

        Templated country (2 events, mean_|e| = 2.5× global):
          V13: 2/640 = 0.0031  →  V18: 2*sqrt(2.5)/640 = 0.0049  (+58%)
        Average country (13 events, mean_|e| = 1.0× global):
          V13: 13/640 = 0.020  →  V18: 13*1.0/640 = 0.020  (same)
        Well-fit country (20 events, mean_|e| = 0.5× global):
          V13: 20/640 = 0.031  →  V18: 20*sqrt(0.5)/640 = 0.022  (-29%)

      Templated countries get 58% more Shape gradient. Well-fit
      countries get 29% less. Total is preserved.

    * **Level (DC magnitude).** ``T × gap²`` (symmetric MSE) on the
      per-series mean gap, gate-weighted, Hájek-normalised. UNCHANGED
      from V13.

      V13 had the best calibration of all versions (0.84× overall,
      1.13× ch_0, 0.51× ch_1, 0.36× ch_2). V16's asymmetric MSE
      caused 3.88× overprediction. Symmetry is correct.

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

        logger.info("SpotlightLossV18 | threshold=%.4f", non_zero_threshold)

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

        # ── SHAPE: log_cosh on demeaned errors, DRO + series boost ───
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

        # ── V18: series boost ────────────────────────────────────────
        # Boost high-error (templated) series, reduce well-fit series.
        # sqrt dampening prevents over-weighting. Detached (no grad).
        # Applied INSIDE the Hájek → loss magnitude preserved.
        #
        # series_boost = sqrt(mean_|e|_series / mean_|e|_global)
        #
        # Templated country (mean_|e| = 2.5× global): boost = 1.58
        # Average country (mean_|e| = 1.0× global):   boost = 1.00
        # Well-fit country (mean_|e| = 0.5× global):  boost = 0.71
        series_err_mean = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev  # (B, 1) or (B, 1, C)
        global_err_mean = (raw_abs * event_mask).sum() / event_mask.sum().clamp_min(1.0)
        series_boost = torch.sqrt(series_err_mean / global_err_mean.clamp_min(1e-8))
        series_boost = torch.nan_to_num(series_boost, nan=1.0, posinf=1.0, neginf=1.0)

        if multivariate:
            shape_w = gate * w_dro * series_boost  # (B, T, C)
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            shape_w = gate * w_dro * series_boost  # (B, T)
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: T × symmetric MSE on mean gap (V13 — unchanged) ───
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

                # ── V18 NEW: series boost diagnostics ──
                # Per-series boost stats over event series only
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B, C)
                _n_es = _ev_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
                _boost_per_series = series_boost.squeeze(-1) * _ev_series_mask  # (B, C)
                boost_mean_l = (_boost_per_series.sum(dim=0) / _n_es).tolist()
                _boost_var = ((_boost_per_series ** 2).sum(dim=0) / _n_es
                              - torch.tensor(boost_mean_l) ** 2).clamp_min(0)
                boost_std_l = _boost_var.sqrt().tolist()
                boost_max_l = (_boost_per_series.amax(dim=0)).tolist()
                boost_min_l = (((_boost_per_series + 1e8 * (1 - _ev_series_mask)).amin(dim=0))).tolist()
                # Fraction of event series with boost > 1 (getting amplified)
                boost_up_frac_l = (((series_boost.squeeze(-1) > 1.0).float() * _ev_series_mask).sum(dim=0)
                                   / _n_es).tolist()

                # Per-series error stats (what drives the boost)
                _err_per_series = series_err_mean.squeeze(-1) * _ev_series_mask  # (B, C)
                err_series_mean_l = (_err_per_series.sum(dim=0) / _n_es).tolist()
                _err_var = ((_err_per_series ** 2).sum(dim=0) / _n_es
                            - torch.tensor(err_series_mean_l) ** 2).clamp_min(0)
                err_series_std_l = _err_var.sqrt().tolist()

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
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B,)
                _n_es = _ev_series_mask.sum().clamp_min(1.0)
                _boost_per_series = series_boost.squeeze(-1) * _ev_series_mask  # (B,) — series_boost is (B,1), squeeze to (B,)
                # Actually series_boost shape is (B, 1) for univariate. Let me handle carefully.
                _bps = series_boost.squeeze(1) if series_boost.dim() > 1 and series_boost.size(1) == 1 else series_boost
                _boost_per_series = _bps * _ev_series_mask
                boost_mean_l = [(_boost_per_series.sum() / _n_es).item()]
                _boost_var = ((_boost_per_series ** 2).sum() / _n_es - boost_mean_l[0] ** 2).clamp_min(0)
                boost_std_l = [_boost_var.sqrt().item()]
                boost_max_l = [_boost_per_series.max().item()]
                boost_min_l = [(_boost_per_series + 1e8 * (1 - _ev_series_mask)).min().item()]
                boost_up_frac_l = [(((_bps > 1.0).float() * _ev_series_mask).sum() / _n_es).item()]
                _err_per_series = series_err_mean.squeeze(1) if series_err_mean.dim() > 1 and series_err_mean.size(1) == 1 else series_err_mean
                _eps = _err_per_series * _ev_series_mask
                err_series_mean_l = [(_eps.sum() / _n_es).item()]
                _err_var = ((_eps ** 2).sum() / _n_es - err_series_mean_l[0] ** 2).clamp_min(0)
                err_series_std_l = [_err_var.sqrt().item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV18: per_channel={comp}")

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
            # ── V18 NEW: series boost diagnostics ──
            # The boost applied to each series' Shape weight.
            # boost > 1 = high-error (templated) series getting amplified.
            # boost < 1 = well-fit series getting reduced.
            # mean should be ≈ 1.0 (preserved by construction).
            "boost_mean":     boost_mean_l,      # mean boost over event series (should be ≈ 1.0)
            "boost_std":      boost_std_l,       # std of boost (high = high variability = boost matters)
            "boost_max":      boost_max_l,       # max boost (most templated series)
            "boost_min":      boost_min_l,       # min boost (best-fit series)
            "boost_up_frac":  boost_up_frac_l,   # frac of event series with boost > 1 (getting amplified)
                                                  # Should be < 0.5 (only high-error series boosted)
            # Per-series error stats (what drives the boost)
            "err_series_mean": err_series_mean_l, # mean |e| per event series
            "err_series_std":  err_series_std_l,  # std of per-series |e| (high = some countries much worse)
                                                   # This is the templating signal — high std = some countries
                                                   # have much higher error = templating candidates.
        }

        logger.debug(
            "SpotlightLossV18 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV18(non_zero_threshold={self.tau})"
