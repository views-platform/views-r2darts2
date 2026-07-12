import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO + gentle series boost (V19).
    Level = T × scaled Huber on mean gap (MSE-like, bounded for extremes).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, **gentle series-boost** (fourth
      root), Hájek-normalised.

      V13's global Hájek gave rare-event countries little gradient →
      templating. V18's sqrt boost (1.58× for templated countries) was
      too aggressive — shifted Shape:Level balance → overprediction.

      V19 uses fourth root: 1.26× for templated countries, 0.84× for
      well-fit. Gentle redistribution that doesn't shift the balance.
      Still inside the Hájek → loss magnitude preserved.

    * **Level (DC magnitude).** ``T × 2*delta * huber(gap, delta)``
      where ``delta=3.0`` (fixed design constant, not a hyperparameter).

      V13 used MSE → gradient 2*gap is unbounded → extreme training
      events (gap=15) get gradient 30, dominate the batch → model
      learns to predict high everywhere → eval overprediction (V18:
      ch_0 at 3.22×).

      V11 used log_cosh(gap) → tanh(gap) saturates for |gap|>3 →
      can't push hard enough on underpredicted channels.

      V15 used Huber(delta=ac_scale) → delta too small (0.88 for ch_1)
      → quadratic regime too narrow → plateaus at 0.40×.

      V19 scaled Huber: ``2*delta * huber(gap, delta)`` with delta=3.0:
        - |gap| < 3: loss = delta * gap², gradient = 2*delta*gap/delta = 2*gap
          → EXACTLY MSE (proportional, unsaturated) ✓
        - |gap| > 3: loss = 2*delta*|gap| - delta², gradient = 2*delta = 6.0
          → BOUNDED at 6.0 (vs MSE's 2*gap=30 for gap=15) ✓

      Per-channel gradient at typical gaps:
        gap=1.0: 2.0 (MSE=2.0, V11=0.76, V15=0.67) — matches MSE ✓
        gap=2.0: 4.0 (MSE=4.0, V11=0.96, V15=1.33) — matches MSE ✓
        gap=3.0: 6.0 (MSE=6.0) — at boundary, still MSE
        gap=5.0: 6.0 (MSE=10.0) — 1.7× weaker, doesn't chase
        gap=15.0: 6.0 (MSE=30.0) — 5× weaker, doesn't chase extremes

      This is "MSE but not insane" — exactly MSE in the normal range,
      bounded for extreme unpredictable events.

      delta=3.0 is a design constant (like log_cosh's implicit scale of
      1), not a tunable hyperparameter. It covers the typical gap range
      (V18: gap_max=3.35 for ch_0) and transitions to bounded only for
      the extreme tail.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    """

    _EPS = 1e-6
    _LEVEL_DELTA = 3.0  # scaled Huber transition point (design constant)

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None

        logger.info("SpotlightLossV19 | threshold=%.4f delta=%.1f", non_zero_threshold, self._LEVEL_DELTA)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _scaled_huber(x: torch.Tensor, delta: float) -> torch.Tensor:
        """Scaled Huber: MSE for |x| < delta, bounded gradient for |x| > delta.

        loss = 0.5 * delta * x² / delta = 0.5 * x²         (|x| <= delta, gradient = x*... wait)

        Standard Huber:
          |x| <= delta: 0.5 * x² / delta    (gradient = x/delta)
          |x| >  delta: |x| - 0.5 * delta   (gradient = sign(x))

        Scaled by 2*delta:
          |x| <= delta: delta * x²          (gradient = 2*delta*x = 2x when... no)

        Let me redo. To get gradient = 2x in quadratic regime (matching MSE):
          loss = x²                           (gradient = 2x)
        So we want:
          |x| <= delta: x²                    (gradient = 2x = MSE ✓)
          |x| >  delta: 2*delta*|x| - delta²  (gradient = 2*delta*sign(x), bounded ✓)

        This is equivalent to: 2*delta * standard_huber(x, delta).
        """
        abs_x = x.abs()
        return torch.where(
            abs_x <= delta,
            x ** 2,                              # MSE (gradient = 2x)
            2.0 * delta * abs_x - delta ** 2,    # bounded (gradient = 2*delta)
        )

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

        # ── SHAPE: log_cosh on demeaned errors, DRO + gentle boost ───
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

        # ── V19: gentle series boost (fourth root, not sqrt) ─────────
        # V18's sqrt boost was too aggressive (1.58× for templated).
        # Fourth root: 1.26× for templated, 0.84× for well-fit.
        # Gentle redistribution that doesn't shift Shape:Level balance.
        # Detached, inside Hájek → loss magnitude preserved.
        series_err_mean = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev
        global_err_mean = (raw_abs * event_mask).sum() / event_mask.sum().clamp_min(1.0)
        series_boost = (series_err_mean / global_err_mean.clamp_min(1e-8)) ** 0.25
        series_boost = torch.nan_to_num(series_boost, nan=1.0, posinf=1.0, neginf=1.0)

        if multivariate:
            shape_w = gate * w_dro * series_boost
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            shape_w = gate * w_dro * series_boost
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: T × scaled Huber (MSE-like, bounded for extremes) ─
        # V19: replaces V13's T*gap² (MSE).
        #
        # MSE gradient 2*gap is unbounded → extreme training events
        # (gap=15) get gradient 30, dominate batch → model predicts
        # high everywhere → eval overprediction (V18: ch_0 3.22×).
        #
        # Scaled Huber with delta=3.0:
        #   |gap| < 3: gradient = 2*gap (EXACTLY MSE)
        #   |gap| > 3: gradient = 2*delta = 6.0 (BOUNDED)
        #
        # Normal gaps (0-2): identical to V13's MSE → same calibration push.
        # Extreme gaps (5-15): bounded at 6.0 → doesn't chase unpredictable events.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * self._scaled_huber(gap, self._LEVEL_DELTA)
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

                # ── V19 NEW: Huber regime diagnostics ──
                # Fraction of event series in BOUNDED regime (|gap| > delta).
                # These are the extreme events that MSE would chase but
                # scaled Huber bounds. High value = many extreme series
                # = scaled Huber is doing important work.
                huber_bounded_frac_l = (((_ga > self._LEVEL_DELTA) * _ev_mask_s).sum(dim=0)
                                        / _n_ev_s).tolist()
                # Mean effective Level gradient over event series.
                # Scaled Huber: 2*|gap| for |gap|<delta, 2*delta for |gap|>delta.
                _eff_grad = torch.where(
                    _ga <= self._LEVEL_DELTA,
                    2.0 * _ga,
                    torch.full_like(_ga, 2.0 * self._LEVEL_DELTA),
                )
                eff_grad_mean_l = ((_eff_grad * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                eff_grad_max_l = (_eff_grad.amax(dim=0)).tolist()
                # What V13 MSE gradient would have been (for comparison)
                mse_grad_mean_l = ((2.0 * _ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                mse_grad_max_l = (2.0 * _ga.amax(dim=0)).tolist()

                # ── V19 NEW: series boost diagnostics ──
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B, C)
                _n_es = _ev_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
                _boost_per_series = series_boost.squeeze(-1) * _ev_series_mask  # (B, C)
                _boost_mean = _boost_per_series.sum(dim=0) / _n_es
                boost_mean_l = _boost_mean.tolist()
                _boost_var = ((_boost_per_series ** 2).sum(dim=0) / _n_es
                              - _boost_mean ** 2).clamp_min(0)
                boost_std_l = _boost_var.sqrt().tolist()
                boost_max_l = (_boost_per_series.amax(dim=0)).tolist()
                boost_up_frac_l = (((series_boost.squeeze(-1) > 1.0).float() * _ev_series_mask).sum(dim=0)
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
                huber_bounded_frac_l = [(((_ga > self._LEVEL_DELTA) * _ev_mask_s).sum() / _n_ev_s).item()]
                _eff_grad = torch.where(
                    _ga <= self._LEVEL_DELTA,
                    2.0 * _ga,
                    torch.full_like(_ga, 2.0 * self._LEVEL_DELTA),
                )
                eff_grad_mean_l = [((_eff_grad * _ev_mask_s).sum() / _n_ev_s).item()]
                eff_grad_max_l = [_eff_grad.max().item()]
                mse_grad_mean_l = [(2.0 * _ga * _ev_mask_s).sum().item() / _n_ev_s.item()]
                mse_grad_max_l = [(2.0 * _ga.max()).item()]
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()
                _n_es = _ev_series_mask.sum().clamp_min(1.0)
                _bps = series_boost.squeeze(1) if series_boost.dim() > 1 and series_boost.size(1) == 1 else series_boost
                _boost_per_series = _bps * _ev_series_mask
                _boost_mean = _boost_per_series.sum() / _n_es
                boost_mean_l = [_boost_mean.item()]
                _boost_var = ((_boost_per_series ** 2).sum() / _n_es - _boost_mean ** 2).clamp_min(0)
                boost_std_l = [_boost_var.sqrt().item()]
                boost_max_l = [_boost_per_series.max().item()]
                boost_up_frac_l = [(((_bps > 1.0).float() * _ev_series_mask).sum() / _n_es).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV19: per_channel={comp}")

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
            # ── V19 NEW: scaled Huber diagnostics ──
            # Fraction of event series in BOUNDED regime (|gap| > delta=3).
            # These are the extreme events MSE would chase.
            # V18 had gap_max=3.35 for ch_0 — so ~1-5% of series hit bounded.
            # If this is >10%, many extreme events are being clipped (good).
            "huber_bounded_frac": huber_bounded_frac_l,
            # Effective Level gradient (what the model actually sees).
            # Scaled Huber: 2*|gap| for |gap|<3, 6.0 for |gap|>3.
            # Compare to mse_grad_mean to see how much we're clipping.
            "eff_grad_mean":  eff_grad_mean_l,   # mean Level gradient (V19)
            "eff_grad_max":   eff_grad_max_l,    # max Level gradient (should be 6.0)
            "mse_grad_mean":  mse_grad_mean_l,   # what V13 MSE gradient would have been
            "mse_grad_max":   mse_grad_max_l,    # what V13 MSE max would have been
                                                  # If mse_grad_max >> 6.0, V19 is clipping
                                                  # extreme events that V13 chased.
            # ── V19 NEW: gentle series boost diagnostics ──
            # Fourth root boost — should be gentler than V18's sqrt.
            # V18: boost_max ~1.58 for 2.5× error ratio.
            # V19: boost_max ~1.26 for 2.5× error ratio.
            "boost_mean":     boost_mean_l,      # mean boost (should be ≈ 1.0)
            "boost_std":      boost_std_l,       # std of boost (V19 < V18)
            "boost_max":      boost_max_l,       # max boost (V19 ~1.2-1.3, V18 was ~1.5-1.6)
            "boost_up_frac":  boost_up_frac_l,   # frac of event series with boost > 1
        }

        logger.debug(
            "SpotlightLossV19 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV19(non_zero_threshold={self.tau})"
