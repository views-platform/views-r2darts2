import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × log_cosh(mean gap) — V34 (log_cosh instead of MSE).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged.

    * **Level (DC magnitude).** ``T × log_cosh(gap)`` on the per-series
      mean gap, gate-weighted, Hájek-normalised.

      V13 used ``T × gap²`` (MSE) → gradient ``2 × gap`` is unbounded.
      For extreme series (gap=15), gradient=30 → gradient explosions
      and runaway overprediction (V32: ch_0 reached 12.25×).

      V34 uses ``T × log_cosh(gap)`` → gradient ``tanh(gap)`` is bounded
      at ±1. For extreme gaps, the push stays gentle — no explosions,
      no runaway.

      Gradient comparison:
        gap=0.2:  V13=0.4,  V34=0.197  (V34 gentler for small gaps)
        gap=1.0:  V13=2.0,  V34=0.762  (V34 much gentler)
        gap=3.0:  V13=6.0,  V34=0.995  (V34 nearly saturated)
        gap=15.0: V13=30.0, V34=1.0    (V34 BOUNDED — no explosion)

      The bounded gradient prevents the runaway dynamics that plagued
      every MSE Level variant (V13's sporadic spikes, V32's 12× overpred).
      The gentler push for normal gaps (0.2-0.5 in RevIN space) means
      slower calibration, but more stable training.

      In RevIN-normalized space, gaps are small (0.1-0.5 typically).
      tanh(0.3) = 0.29 — the gradient is ~75% of the gap value. This
      is weaker than MSE's 2*gap=0.6, but stable. The model calibrates
      more slowly but doesn't overshoot.

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

        logger.info("SpotlightLossV34 | threshold=%.4f", non_zero_threshold)

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
        # Unchanged from V13.
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

        # ── LEVEL: T × log_cosh(mean gap) — V34 ──────────────────────
        # V34: replaces V13's T × gap² (MSE) with T × log_cosh(gap).
        #
        # Gradient: T * tanh(gap) * (1/T) = tanh(gap) per cell.
        # Bounded at ±1 → no explosions on extreme gaps.
        #
        # For small gaps (0.2 in RevIN space): tanh(0.2) = 0.197.
        # This is ~2× weaker than V13's 2*0.2=0.4, but stable.
        # The model calibrates more slowly but doesn't overshoot.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * self._log_cosh(gap)
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

                # V34: gradient comparison (log_cosh vs MSE)
                _tanh_gap = torch.tanh(gap).abs()
                _mse_grad = (2.0 * gap).abs()
                tanh_grad_mean_l = (_tanh_gap.mean(dim=0)).tolist()
                tanh_grad_max_l = (_tanh_gap.amax(dim=0)).tolist()
                mse_grad_mean_l = (_mse_grad.mean(dim=0)).tolist()
                mse_grad_max_l = (_mse_grad.amax(dim=0)).tolist()
                # Saturation: fraction of event series where tanh is saturated (>0.95)
                sat_frac_l = (((_tanh_gap > 0.95).float() * _ev_mask_s).sum(dim=0)
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
                _tanh_gap = torch.tanh(gap).abs()
                _mse_grad = (2.0 * gap).abs()
                tanh_grad_mean_l = [_tanh_gap.mean().item()]
                tanh_grad_max_l = [_tanh_gap.max().item()]
                mse_grad_mean_l = [_mse_grad.mean().item()]
                mse_grad_max_l = [_mse_grad.max().item()]
                sat_frac_l = [(((_tanh_gap > 0.95).float() * _ev_mask_s).sum() / _n_ev_s).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV34: per_channel={comp}")

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
            # ── V34: gradient comparison (log_cosh vs MSE) ──
            # tanh_grad is what V34 actually uses (bounded at 1.0).
            # mse_grad is what V13 would have used (unbounded).
            # If mse_grad_max >> 1.0, V34 is successfully bounding
            # what would have been an explosion in V13.
            "tanh_grad_mean": tanh_grad_mean_l,   # V34's gradient (bounded)
            "tanh_grad_max":  tanh_grad_max_l,    # max V34 gradient (≤ 1.0)
            "mse_grad_mean":  mse_grad_mean_l,    # V13's gradient (for comparison)
            "mse_grad_max":   mse_grad_max_l,     # max V13 gradient (can be large)
            "sat_frac":       sat_frac_l,          # frac event series where tanh > 0.95 (saturated)
                                                   # If high, V34 can't push harder — may underpredict
        }

        logger.debug(
            "SpotlightLossV34 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV34(non_zero_threshold={self.tau})"
