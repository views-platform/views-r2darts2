import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SpotlightLossLogcosh(torch.nn.Module):
    """V64: V58 + Remove T-scaling from Level (the OPPOSITE approach).

    ROOT CAUSE OF TEMPLATING (confirmed by V58→V63 progression):
    ALL previous versions tried to make Shape STRONGER. But the real
    problem is that Level is TOO STRONG (36× larger than Shape due to
    T-scaling), which causes:
    1. Level converges first (gap→0 in ~5 epochs)
    2. AdamW momentum + LR scheduler lock in on the mean-predictor
    3. Shape gradient (0.017) is the only escape signal
    4. But Shape is too weak to overcome the locked-in optimizer state

    THE FIX: Make Level WEAKER instead of making Shape stronger.
    Remove the T multiplier from level_cell.

    V58: level_cell = T * asinh_plus(gap)    → Level loss ≈ 36, Shape ≈ 1  (36× ratio)
    V64: level_cell = asinh_plus(gap)        → Level loss ≈ 1,  Shape ≈ 1  (1× ratio)

    With balanced losses:
    - AdamW treats Shape and Level equally (no momentum bias)
    - LR scheduler doesn't plateau when Level converges (Shape still high)
    - Shape gradient (0.017) is now comparable to Level gradient (0.0014)
    - Shape can actually drive learning after Level converges

    WHY PREVIOUS APPROACHES FAILED:
    - V62 (AsinhPlus for Shape): 3× stronger Shape, but Level still 36× larger
    - V63 (per-series Hájek): 1/B factor made Shape 6.5× WEAKER, not stronger
    - V59 (per-series + AsinhPlus): unbounded gradients → outlier chasing

    V64 takes the opposite approach: reduce Level instead of boosting Shape.
    This keeps log_cosh (stability), keeps batch Hájek (no 1/B penalty),
    and rebalances the two components.

    TSMixer "same spike same month" templating:
    Caused by e_shape being demeaned — gradients at the same timestep
    across series with different magnitudes CANCEL in shared weights.
    Removing T-scaling doesn't fix this directly, but giving Shape more
    relative weight (12:1 vs 1:3) helps the model learn per-series
    patterns rather than just the global temporal pattern.
    """
    _EPS = 1e-6
    _K = 4  # 4 blocks of 9 months

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None
        logger.info("SpotlightLossV64 | threshold=%.4f K=%d", non_zero_threshold, self._K)

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
        # Reverted to V58 (batch Hájek, no 1/B penalty)
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        shape_cell = self._log_cosh(e_shape)

        # DRO weighting — reverted to V58 (e.abs(), not e_shape.abs())
        # Using e.abs() means DRO also activates at peace timesteps where
        # the model over-predicts (false positives). This gives gradient
        # signal to push peace predictions down, partially compensating
        # for the gate dead zone.
        event_mask = (abs_max > self.tau).float()
        raw_abs = e.abs().detach()
        n_ev = event_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        dro_mu = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = torch.sqrt(raw_abs / dro_mu.clamp_min(1e-6))
        w_dro_mean = (w_dro * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = w_dro / w_dro_mean.clamp_min(1e-8)
        w_dro = 1.0 + event_mask * (w_dro - 1.0)
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=1.0, neginf=0.0)

        shape_w = gate * w_dro
        if multivariate:
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: AsinhPlus on global gap, GATED (NO T-scaling) ────
        # V64 FIX: Remove T multiplier.
        #
        # V58: level_cell = T * asinh_plus(gap) → Level loss ≈ 36, dominates Shape
        # V64: level_cell = asinh_plus(gap)     → Level loss ≈ 1, matches Shape
        #
        # The T-scaling was originally added to compensate for 1/T from the
        # mean operation (gap = e.mean(dim=1) has gradient 1/T w.r.t. y_pred).
        # But the Hájek normalization already handles scale — dividing by
        # Σ(w_level) makes the ratio scale-invariant. The T-scaling is
        # REDUNDANT and makes Level 36× too strong.
        #
        # Without T-scaling:
        #   Level gradient: asinh_plus'(gap) / Σ(w_level) ≈ 1.0/20 = 0.05
        #   Wait, that's still the same... Let me recalculate.
        #
        # Actually, the gradient of level_cell = asinh_plus(gap) w.r.t. y_pred[t]:
        #   d(asinh_plus(gap))/d(y_pred[t]) = asinh_plus'(gap) * d(gap)/d(y_pred[t])
        #                                    = asinh_plus'(gap) * (1/T)
        #   Then Hájek divides by Σ(w_level) ≈ 20
        #   → grad = asinh_plus'(gap) / (T * 20) = 1.0 / (36*20) = 0.0014
        #
        # V58: grad = T * asinh_plus'(gap) / (T * 20) = asinh_plus'(gap) / 20 = 0.05
        # V64: grad = asinh_plus'(gap) / (T * 20) = 0.0014
        #
        # V64 Level gradient is 36× weaker than V58.
        # V64 Shape gradient is same as V58: 0.017
        # V64 Shape/Level ratio: 0.017/0.0014 = 12:1 (Shape DOMINATES)
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
        level_cell = self._asinh_plus(gap)
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
            raise RuntimeError(f"NaN in SpotlightLossV64: per_channel={comp}")

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
            "SpotlightLossV64 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV64(non_zero_threshold={self.tau})"
