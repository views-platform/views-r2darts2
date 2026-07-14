import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SpotlightLossLogcosh(torch.nn.Module):
    """V63: V58 + Per-series Hájek for Shape (keeps log_cosh).

    ROOT CAUSE OF TEMPLATING (confirmed by comparing V58 vs V62):
    The Hájek denominator for Shape sums over ALL active timesteps
    in the batch (~60). This dilutes each timestep's gradient by 60×:

      V58:  tanh(5)/60 = 0.017  (log_cosh, batch Hájek)
      V62:  asinh_plus'(5)/60 = 0.055  (AsinhPlus, batch Hájek)

    Switching to AsinhPlus only gave 3× stronger gradient (0.017→0.055),
    but the Hájek denominator is STILL 60. The gradient is still 10×
    weaker than Level (0.5). The model still cannot escape the mean.

    From logs:
      V58 and V62 both show grad_ac% = 10% (Shape dominated by Level)
      V58 and V62 both show fc_out collapse (6.37→0.15, 2.38→0.06)
      V58 and V62 both show calibration ratio 0.37-0.43x (mean regression)

    THE FIX (minimal, keeps log_cosh):
    Use per-series Hájek for Shape. The denominator changes from ~60
    (all active timesteps) to ~3 (per-series active timesteps).

      V63:  tanh(5)/3 = 0.33  (log_cosh, per-series Hájek)

    This is 20× stronger than V58 and 6× stronger than V62, while
    STILL bounded by tanh ≤ 1.0 (no outlier chasing).

    WHY V59 (per-series Hájek + AsinhPlus) FAILED:
    V59 combined per-series Hájek with AsinhPlus, giving UNBOUNDED
    gradients for large errors:
      V59 at e=50: asinh_plus'(50)/3 = 5.6/3 = 1.87  (unbounded!)
    This caused the outlier chasing the user observed.

    V63 keeps log_cosh (bounded at 1.0) with per-series Hájek:
      V63 at e=50: tanh(50)/3 = 1.0/3 = 0.33  (bounded, stable)

    Division of labor (unchanged):
    - Shape (log_cosh, per-series Hájek): learns per-timestep patterns
    - Level (AsinhPlus, batch Hájek over series): calibrates mean
    """
    _EPS = 1e-6
    _K = 4  # 4 blocks of 9 months

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None
        logger.info("SpotlightLossV63 | threshold=%.4f K=%d", non_zero_threshold, self._K)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _asinh_plus(x: torch.Tensor) -> torch.Tensor:
        """Loss: x * asinh(x)
        Gradient: asinh(x) + x / sqrt(1 + x^2)
        Matches MSE curvature (2.0) at origin, bends to log(x) for large x.
        """
        return x * torch.asinh(x)

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

        # ── SHAPE: log_cosh on demeaned errors (NO ac_scale) ────────
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        # V63: KEEP log_cosh (intentional stability regularizer).
        # tanh gradient saturates at 1.0 for |e_shape| > 2, which bounds
        # the gradient and prevents outlier chasing.
        shape_cell = self._log_cosh(e_shape)

        # DRO weighting (V60 fix: use e_shape.abs(), not e.abs())
        # This aligns DRO with what Shape actually penalizes (pattern error),
        # preventing dilution when gap ≠ 0.
        event_mask = (abs_max > self.tau).float()
        raw_abs = e_shape.abs().detach()
        n_ev = event_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        dro_mu = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = torch.sqrt(raw_abs / dro_mu.clamp_min(1e-6))
        w_dro_mean = (w_dro * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = w_dro / w_dro_mean.clamp_min(1e-8)
        w_dro = 1.0 + event_mask * (w_dro - 1.0)
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=1.0, neginf=0.0)

        shape_w = gate * w_dro

        # V63 FIX: Per-series Hájek for Shape.
        #
        # V58 used batch Hájek: Σ(shape_w · shape_cell) / Σ(shape_w)
        # The denominator Σ(shape_w) sums over ALL active timesteps in
        # the batch (~60), diluting each timestep's gradient by 60×.
        #
        # Per-series Hájek: compute the Hájek ratio PER SERIES (denominator
        # ~3), then average across series. Each series gets equal weight
        # (1/B) regardless of event count.
        #
        # This gives 20× stronger gradient per timestep while keeping
        # log_cosh's tanh bound (≤ 1.0). No outlier chasing.
        if multivariate:
            # shape_w: (B, T, C), shape_cell: (B, T, C)
            # Per-series Hájek over T, then mean over B
            num_per_series = (shape_w * shape_cell).sum(dim=1)        # (B, C)
            den_per_series = shape_w.sum(dim=1).clamp_min(self._EPS)  # (B, C)
            loss_shape = (num_per_series / den_per_series).mean(dim=0)  # (C,)
        else:
            # shape_w: (B, T), shape_cell: (B, T)
            num_per_series = (shape_w * shape_cell).sum(dim=1)        # (B,)
            den_per_series = shape_w.sum(dim=1).clamp_min(self._EPS)  # (B,)
            loss_shape = (num_per_series / den_per_series).mean()     # scalar

        # ── LEVEL: AsinhPlus on global gap, GATED ───────────────────
        # Level keeps batch Hájek (over series) — this is CORRECT because
        # we want each series to contribute equally to the level calibration.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
        level_cell = T * self._asinh_plus(gap)
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
                dro_wmean_l = _dm.tolist()
                dro_wstd_l = _dstd.tolist()
                dro_wmax_l = w_dro.amax(dim=(0, 1)).tolist()
                dro_frac_up_l = (((w_dro > 1.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                event_frac_l = event_mask.mean(dim=(0, 1)).tolist()

                gap_global = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga = gap_global.abs()
                gap_mean_l = _ga.mean(dim=0).tolist()
                gap_max_l = _ga.amax(dim=0).tolist()
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                shape_dc_l = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                sl_ratio_l = (loss_shape.detach() / loss_level.detach().clamp_min(self._EPS)).tolist()
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

                gap_global = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga = gap_global.abs()
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
            raise RuntimeError(f"NaN in SpotlightLossV63: per_channel={comp}")

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
            "SpotlightLossV63 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV63(non_zero_threshold={self.tau})"
