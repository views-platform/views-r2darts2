import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = T × log_cosh with raw-error DRO (V14 — fixes templating).
    Level = T × Huber(mean gap, delta=ac_scale) (V14 — fixes V13 explosions).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** ``T × log_cosh`` on demeaned residual,
      gated, DRO-weighted on ``|raw_error|``, Hájek-normalised.

      V11/V12/V13 used ``log_cosh`` without T. Shape loss was ~1.0
      while Level (with T) was ~55 → 55:1 loss ratio → 43:1 gradient
      ratio (dcMag=0.013 vs acMag=0.0003) → model spent 87% of
      gradient on DC, only 13% on AC → prediction std collapsed to
      0.04 (epoch 16) → **templating**.

      Adding T scales Shape loss to ~36, comparable to Level. The
      gradient ratio shifts from 1:43 to ~1:1. tanh is still bounded
      at 1, so T*tanh is bounded at T=36 — large but bounded, no
      explosion risk.

    * **Level (DC magnitude).** ``T × Huber(mean gap, delta=ac_scale)``
      per series, gate-weighted, Hájek-normalised.

      V13 used ``T × gap²`` (MSE). Gradient ``2*gap`` is unbounded →
      5 of 18 epochs had grad max > 500 (peaks at 1452) → oscillation
      and ch_0 overprediction (6.64× by epoch 17).

      V11 used ``T × log_cosh(gap)``. Gradient ``tanh(gap)`` saturates
      for |gap|>3 → can't distinguish "off by 5" from "off by 15".

      V14 Huber with ``delta=ac_scale`` (per-channel AC std):
        - |gap| < ac_scale: gradient = gap/ac_scale (quadratic, gentle)
        - |gap| > ac_scale: gradient = sign(gap) (linear, bounded at 1.0)

      Per-channel at typical gap ≈ 1.24:
        ch_0 (ac_scale~2.0): quadratic regime, grad=0.62 (vs V13's 2.48)
        ch_1 (ac_scale~0.88): linear regime, grad=1.00 (vs V11's 0.85)
        ch_2 (ac_scale~1.5): quadratic regime, grad=0.83

      ch_0 gets gentler push (prevents 6× overprediction).
      ch_1 gets stronger push than V11 (fixes 0.55× underprediction).
      All channels bounded at 1.0 (no explosions).
      No new hyperparameter — ac_scale already computed for Shape.

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

        logger.info("SpotlightLossV14 | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _huber(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Huber loss with gradient bounded at 1.0.

        |x| <= delta: 0.5 * x² / delta   (gradient = x/delta, → 1.0 at boundary)
        |x| >  delta: |x| - 0.5 * delta  (gradient = sign(x), bounded at 1.0)

        ``delta`` broadcasts against ``x``. Both must be positive.
        """
        abs_x = x.abs()
        return torch.where(
            abs_x <= delta,
            0.5 * x ** 2 / delta,
            abs_x - 0.5 * delta,
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

        # ── AC scale (shared by Shape normalisation and Level delta) ─
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        # Squeeze to broadcast with per-series gap: (C,) for multivariate,
        # scalar for univariate.
        ac_scale_1d = ac_scale.squeeze()

        # ── SHAPE: T × log_cosh on demeaned errors, DRO on |raw_error| ─
        # V14 change: added T factor to balance with Level.
        # Without T, Shape loss ~1.0 vs Level ~55 → 55:1 ratio →
        # acMag/dcMag = 1:43 → templating (prediction std collapses).
        # With T, Shape loss ~36 vs Level ~14-29 → balanced →
        # Shape gradient gets meaningful share of the total.
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        shape_cell = T * self._log_cosh(e_shape / ac_scale)

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

        # ── LEVEL: T × Huber(mean gap, delta=ac_scale), Hájek ────────
        # V14 change: replaces V13's T*gap² (MSE).
        #
        # MSE gradient 2*gap is unbounded → V13 had 5 epochs with
        # grad max > 500 → oscillation, ch_0 overpredicted to 6.64×.
        #
        # Huber with delta=ac_scale (per-channel AC std):
        #   |gap| < ac_scale: grad = gap/ac_scale (quadratic, gentle)
        #   |gap| > ac_scale: grad = sign(gap)    (linear, bounded at 1.0)
        #
        # Per-channel adaptation:
        #   ch_0 (high AC, ac_scale~2.0): gap~1.24 is in quadratic regime
        #     → grad=0.62 → gentle push → prevents overprediction
        #   ch_1 (low AC, ac_scale~0.88): gap~1.24 is in linear regime
        #     → grad=1.0 → strong push → fixes underprediction
        #   ch_2 (mid AC, ac_scale~1.5): gap~1.24 is in quadratic regime
        #     → grad=0.83 → moderate push
        #
        # All channels bounded at 1.0 → no gradient explosions.
        # No new hyperparameter — ac_scale already computed for Shape.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * self._huber(gap, ac_scale_1d)
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

                # ── V14 NEW: Huber regime diagnostics ──
                # Per-channel ac_scale (the Huber delta)
                ac_scale_l = ac_scale_1d.tolist() if ac_scale_1d.dim() > 0 else [float(ac_scale_1d)]
                # Fraction of event series in LINEAR regime (|gap| > ac_scale)
                # High value → most series getting bounded-at-1.0 gradient
                # Low value  → most series in quadratic regime (gentle push)
                huber_linear_l = (((_ga > ac_scale_1d.abs()) * _ev_mask_s).sum(dim=0)
                                  / _n_ev_s).tolist()
                # Mean Huber gradient magnitude over event series
                # (diagnostic for effective Level push strength)
                _huber_grad = torch.where(
                    _ga <= ac_scale_1d.abs(),
                    _ga / ac_scale_1d.abs(),
                    torch.ones_like(_ga),
                )
                huber_grad_mean_l = ((_huber_grad * _ev_mask_s).sum(dim=0)
                                     / _n_ev_s).tolist()

                # ── V14 NEW: Shape/Level loss ratio per channel ──
                # Should be ~0.5-2.0 with T on both terms.
                # <0.3 → Shape starved (templating risk)
                # >5.0 → Level starved (calibration risk)
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
                huber_linear_l = [(((_ga > ac_scale_1d.abs()) * _ev_mask_s).sum()
                                   / _n_ev_s).item()]
                _huber_grad = torch.where(
                    _ga <= ac_scale_1d.abs(),
                    _ga / ac_scale_1d.abs(),
                    torch.ones_like(_ga),
                )
                huber_grad_mean_l = [((_huber_grad * _ev_mask_s).sum()
                                      / _n_ev_s).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV14: per_channel={comp}")

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
            # ── V14 NEW: Huber regime diagnostics ──
            "ac_scale":         ac_scale_l,       # per-channel Huber delta (= AC std of y_true)
            "huber_linear_frac": huber_linear_l,   # frac of event series in LINEAR regime (|gap|>delta)
                                                    # High → bounded-at-1.0 gradient (strong push)
                                                    # Low  → quadratic regime (gentle push)
            "huber_grad_mean":   huber_grad_mean_l, # mean Huber gradient over event series (0-1.0)
            # ── V14 NEW: Shape/Level balance ──
            "shape_level_ratio": sl_ratio_l,       # Shape loss / Level loss per channel
                                                    # V13 was ~0.02 (55:1) → templating
                                                    # V14 target: 0.5-2.0 (balanced)
        }

        logger.debug(
            "SpotlightLossV14 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV14(non_zero_threshold={self.tau})"
