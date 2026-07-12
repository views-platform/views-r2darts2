import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO, PER-SERIES Hájek (V17 — fixes templating).
    Level = T × symmetric MSE on mean gap (V13 style — best calibration).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** ``log_cosh`` on demeaned residual, gated,
      DRO-weighted on ``|raw_error|``, **per-series Hájek-normalised**,
      then averaged over event series.

      V13 used global Hájek over (B, T): divided by total event cells
      across the batch. Countries with 10 event cells got 10× more
      gradient share than countries with 1 event cell. Rare-event
      countries (the ones that template most) got the least Shape
      gradient → the model learned a "common pattern" for them.

      V17 normalizes within each series first (sum_T / sum_T), then
      averages over event series (mean_B). Each event country gets
      EQUAL total Shape gradient regardless of how many event cells
      it has:

        Rare-event country (2 cells): each cell gets 1/(N_ev * 2) ≈ 5× V13
        Frequent-event country (10 cells): each cell gets 1/(N_ev * 10) ≈ 0.5× V13

      This directly targets the templating problem — the model can no
      longer ignore rare-event countries by averaging their patterns
      into a common template.

      Loss scale unchanged (mean of means = weighted mean ≈ 1.0).
      No T factor (V14 proved T causes explosions).
      No new hyperparameters.

    * **Level (DC magnitude).** ``T × gap²`` (symmetric MSE) on the
      per-series mean gap, gate-weighted, Hájek-normalised.

      V13's symmetric MSE had the best calibration of all versions
      (0.84× overall, 1.13× ch_0, 0.51× ch_1, 0.36× ch_2).
      V16's asymmetric MSE caused massive overprediction (3.88× from
      epoch 0). Symmetry is correct — the model needs equal push
      in both directions.

      T compensates 1/T dilution from mean() operator.
      MSE gradient 2*gap is proportional, never saturates (unlike
      V11's tanh, V15's Huber).

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

        logger.info("SpotlightLossV17 | threshold=%.4f", non_zero_threshold)

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

        # ── AC scale ─────────────────────────────────────────────────
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)

        # ── SHAPE: log_cosh on demeaned errors, DRO, per-series Hájek ─
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

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

        # V17: per-series Hájek (normalize over T only), then mean over
        # event series. This gives each event country EQUAL total Shape
        # gradient regardless of how many event cells it has.
        #
        # V13 (global Hájek): sum_BT(w*cell) / sum_BT(w)
        #   → countries with more events get more gradient
        #   → rare-event countries get little gradient → templating
        #
        # V17 (per-series Hájek):
        #   per_series = sum_T(w*cell) / sum_T(w)       # (B,) or (B, C)
        #   loss = mean over event series of per_series  # scalar or (C,)
        #   → each event country gets equal gradient → no templating
        if multivariate:
            # (B, T, C) → sum over T → (B, C)
            shape_per_series = (shape_w * shape_cell).sum(dim=1) / shape_w.sum(dim=1).clamp_min(self._EPS)
            # Event series mask: 1 if series has ANY event cell for that channel
            ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B, C)
            n_ev_series = ev_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
            loss_shape = (shape_per_series * ev_series_mask).sum(dim=0) / n_ev_series  # (C,)
        else:
            # (B, T) → sum over T → (B,)
            shape_per_series = (shape_w * shape_cell).sum(dim=1) / shape_w.sum(dim=1).clamp_min(self._EPS)
            ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B,)
            n_ev_series = ev_series_mask.sum().clamp_min(1.0)
            loss_shape = (shape_per_series * ev_series_mask).sum() / n_ev_series

        # ── LEVEL: T × symmetric MSE on mean gap (V13 style) ─────────
        # V16's asymmetry caused 3.88× overprediction from epoch 0.
        # Symmetric MSE is correct — V13 had best calibration (0.84×).
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

                ac_scale_l = ac_scale.squeeze().tolist() if ac_scale.squeeze().dim() > 0 else [float(ac_scale.squeeze())]

                # ── V17 NEW: per-series shape diagnostics ──
                # These show whether the per-series Hájek is working:
                # rare-event countries should now get more gradient.
                # Event cells per series (for event series only):
                _ev_per_series = event_mask.sum(dim=1)  # (B, C)
                _ev_series_mask = (ev_series_mask > 0).float()  # (B, C)
                _n_es = _ev_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
                # Mean and std of event-cell count per event series
                _ev_per_series_mean = ((_ev_per_series * _ev_series_mask).sum(dim=0) / _n_es)  # (C,)
                ev_per_series_mean_l = _ev_per_series_mean.tolist()
                _evps_var = ((_ev_per_series ** 2 * _ev_series_mask).sum(dim=0) / _n_es
                             - _ev_per_series_mean ** 2).clamp_min(0)
                ev_per_series_std_l = _evps_var.sqrt().tolist()
                # Per-series shape loss stats (how variable is shape loss across countries?)
                _sps = shape_per_series * _ev_series_mask  # zero out peace series
                _shape_per_series_mean = (_sps.sum(dim=0) / _n_es)  # (C,)
                shape_per_series_mean_l = _shape_per_series_mean.tolist()
                _sps_var = ((_sps ** 2).sum(dim=0) / _n_es
                            - _shape_per_series_mean ** 2).clamp_min(0)
                shape_per_series_std_l = _sps_var.sqrt().tolist()
                # Fraction of event series with above-median shape loss
                # (high value = shape loss concentrated in few countries = templating)
                _sps_median = _sps.median(dim=0).values  # (C,) rough median
                shape_hi_frac_l = ((_sps > _sps_median.unsqueeze(0)).float().sum(dim=0)
                                   / _n_es).tolist()

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
                ac_scale_l = [float(ac_scale.squeeze())]
                _ev_per_series = event_mask.sum(dim=1)  # (B,)
                _ev_series_mask = (ev_series_mask > 0).float()  # (B,)
                _n_es = _ev_series_mask.sum().clamp_min(1.0)
                ev_per_series_mean_l = [((_ev_per_series * _ev_series_mask).sum()
                                         / _n_es).item()]
                _evps_var = ((_ev_per_series ** 2 * _ev_series_mask).sum() / _n_es
                             - ev_per_series_mean_l[0] ** 2).clamp_min(0)
                ev_per_series_std_l = [_evps_var.sqrt().item()]
                _sps = shape_per_series * _ev_series_mask
                shape_per_series_mean_l = [(_sps.sum() / _n_es).item()]
                _sps_var = ((_sps ** 2).sum() / _n_es
                            - shape_per_series_mean_l[0] ** 2).clamp_min(0)
                shape_per_series_std_l = [_sps_var.sqrt().item()]
                _sps_median = _sps.median()
                shape_hi_frac_l = [((_sps > _sps_median).float().sum()
                                    / _n_es).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV17: per_channel={comp}")

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
            "ac_scale":         ac_scale_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V17 NEW: per-series shape diagnostics ──
            # Event cells per event series — shows the variability that
            # per-series Hájek addresses. High std/mean ratio = high
            # variability = per-series Hájek matters a lot.
            "ev_per_series_mean": ev_per_series_mean_l,  # mean event cells per event series
            "ev_per_series_std":  ev_per_series_std_l,   # std of event cells per event series
            # Per-series shape loss — if std/mean is high, some countries
            # have much worse patterns than others (templating signal).
            # V13's global Hájek let these be dominated by frequent-event
            # countries. V17 gives them equal weight.
            "shape_per_series_mean": shape_per_series_mean_l,  # mean per-series shape loss
            "shape_per_series_std":  shape_per_series_std_l,   # std of per-series shape loss
            # Fraction of event series with above-median shape loss.
            # Should be ~0.5 if shape loss is symmetrically distributed.
            # If >0.7, shape loss is concentrated in few countries
            # (those are the templating candidates).
            "shape_hi_frac": shape_hi_frac_l,
        }

        logger.debug(
            "SpotlightLossV17 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV17(non_zero_threshold={self.tau})"
