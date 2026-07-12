import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V11 style — no T).
    Level = T × asymmetric MSE on mean gap (V16 — V13 MSE + 2:1 asymmetry).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** ``log_cosh`` on demeaned residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged from
      V11/V13/V15 — no T factor (V14 proved T causes explosions).

    * **Level (DC magnitude).** ``T × asym_w × gap²`` where ``asym_w = 2``
      for underprediction (gap < 0) and ``1`` for overprediction (gap > 0).

      V15 used Huber(gap, delta=ac_scale) — gradient saturates at 1.0
      for |gap| > ac_scale. For ch_1 (ac_scale≈0.88), gap≈1.24 is
      already in the linear regime → gradient stuck at 1.0 → model
      can't push harder → plateaus at 0.40× (vs V13's 0.84×).

      V13 used T × gap² (MSE) — gradient 2*gap is proportional to
      error, never saturates → best calibration of all versions
      (0.84× overall, 1.13× ch_0, 0.51× ch_1, 0.36× ch_2).
      Problems: (1) gradient explosions from outlier series
      (5/18 epochs, max 1452), (2) ch_0 overshoot from 1.13× to 6.64×
      after epoch 12 (MSE pushes equally in both directions, model
      builds momentum, overshoots).

      V16 asymmetric MSE fixes both V13 problems:
        - Underprediction (gap < 0): weight 2.0 → gradient 4*|gap|
          → 2× stronger push → faster calibration of ch_1, ch_2
        - Overprediction (gap > 0): weight 1.0 → gradient 2*|gap|
          → 2× gentler push → prevents ch_0 overshoot

      The 2:1 ratio is the standard asymmetric loss design (same as
      V10 used for the same purpose). Not a tunable hyperparameter.

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

        logger.info("SpotlightLossV16 | threshold=%.4f", non_zero_threshold)

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
        # No T factor (V14 proved T causes explosions: T*tanh/ac_scale
        # bounds at 41/cell for ch_1, compounds to grad_max 2589-3706).
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        ac_scale_1d = ac_scale.squeeze()

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

        # ── LEVEL: T × asymmetric MSE on mean gap ────────────────────
        # V16: V13's MSE + 2:1 asymmetry (underprediction penalized 2×).
        #
        # V13 (symmetric MSE) had best calibration (0.84×) but:
        #   - explosions from outlier gaps (grad max 1452)
        #   - ch_0 overshoot 1.13× → 6.64× (MSE pushes equally both ways)
        #
        # V15 (Huber) was stable but plateaued at 0.40× — gradient
        # saturates at 1.0 for |gap| > ac_scale, can't push harder.
        #
        # V16 asymmetric MSE:
        #   gap < 0 (underpredicting): weight 2.0 → grad 4*|gap| per cell
        #     → 2× stronger push → breaks ch_1, ch_2 out of underprediction
        #   gap > 0 (overpredicting): weight 1.0 → grad 2*|gap| per cell
        #     → 2× gentler push → prevents ch_0 overshoot
        #
        # Still MSE (proportional gradient, no saturation) — just
        # asymmetric. Explosions still possible from extreme outlier
        # gaps, but asymmetry reduces overprediction-side explosions.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        asym_w = torch.where(gap < 0, 2.0, 1.0)
        level_cell = T * asym_w * gap ** 2
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

                ac_scale_l = ac_scale_1d.tolist() if ac_scale_1d.dim() > 0 else [float(ac_scale_1d)]

                # ── V16 NEW: asymmetry diagnostics ──
                # Fraction of event series that are UNDERPREDICTING (gap < 0).
                # These get 2× weight. If this is high, most of the Level
                # push is at 2× strength (strong calibration force).
                _gap_neg = (gap < 0).float() * _ev_mask_s
                underpredict_frac_l = (_gap_neg.sum(dim=0) / _n_ev_s).tolist()
                # Mean |gap| for underpredicting vs overpredicting event series.
                _gap_under = (_ga * _gap_neg).sum(dim=0) / _gap_neg.sum(dim=0).clamp_min(1.0)
                _gap_over  = (_ga * (1 - _gap_neg) * _ev_mask_s).sum(dim=0) / ((1 - _gap_neg) * _ev_mask_s).sum(dim=0).clamp_min(1.0)
                gap_under_mean_l = _gap_under.tolist()
                gap_over_mean_l  = _gap_over.tolist()
                # Mean effective Level gradient per cell (asym_w * 2 * |gap|)
                # — directly shows the push strength. Underpredicting series
                # get 2× this value.
                _eff_grad = asym_w * 2.0 * _ga
                eff_grad_mean_l = ((_eff_grad * _ev_mask_s).sum(dim=0)
                                   / _n_ev_s).tolist()

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
                ac_scale_l = [float(ac_scale_1d)]
                _gap_neg = (gap < 0).float() * _ev_mask_s
                underpredict_frac_l = [(_gap_neg.sum() / _n_ev_s).item()]
                _gap_under = (_ga * _gap_neg).sum() / _gap_neg.sum().clamp_min(1.0)
                _gap_over  = (_ga * (1 - _gap_neg) * _ev_mask_s).sum() / ((1 - _gap_neg) * _ev_mask_s).sum().clamp_min(1.0)
                gap_under_mean_l = [float(_gap_under.item())]
                gap_over_mean_l  = [float(_gap_over.item())]
                _eff_grad = asym_w * 2.0 * _ga
                eff_grad_mean_l = [((_eff_grad * _ev_mask_s).sum()
                                    / _n_ev_s).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV16: per_channel={comp}")

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
            # ── V16 NEW: asymmetry diagnostics ──
            "underpredict_frac":  underpredict_frac_l,  # frac of event series with gap < 0 (getting 2× push)
                                                         # High → strong calibration force active
                                                         # Low → most series overpredicting (gentle 1× push)
            "gap_under_mean":     gap_under_mean_l,      # mean |gap| for underpredicting event series
            "gap_over_mean":      gap_over_mean_l,       # mean |gap| for overpredicting event series
            "eff_grad_mean":      eff_grad_mean_l,       # mean effective Level gradient (asym_w * 2 * |gap|)
                                                         # V13 was 2*|gap| (symmetric)
                                                         # V16 is 4*|gap| for underpredicting, 2*|gap| for overpredicting
        }

        logger.debug(
            "SpotlightLossV16 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV16(non_zero_threshold={self.tau})"
