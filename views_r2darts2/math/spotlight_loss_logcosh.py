import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    """

    _EPS = 1e-6
    _K = 4

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None
        logger.info(
            "SpotlightLossV60 | threshold=%.4f K=%d | balanced gradients",
            non_zero_threshold,
            self._K,
        )

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _asinh_plus(x: torch.Tensor) -> torch.Tensor:
        """Loss: x · asinh(x).
        Gradient: asinh(x) + x / √(1 + x²).
        Curvature 2.0 at origin (matches MSE), bends to log for large |x|.
        Never saturates — gradient grows without bound.
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
            y_pred.register_hook(
                lambda g: setattr(self, "_last_input_grad", g.detach())
            )

        e = y_pred - y_true

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(15.0 * (abs_max - self.tau))

        event_mask = (abs_max > self.tau).float()
        n_ev = event_mask.sum(dim=1, keepdim=True).clamp_min(self._EPS)

        # ── SHAPE: log_cosh on demeaned errors (unchanged) ───────────
        e_mean = (event_mask * e).sum(dim=1, keepdim=True) / n_ev
        e_shape = e - e_mean

        shape_cell = self._log_cosh(e_shape)

        # DRO weighting
        raw_abs = e_shape.abs().detach()
        dro_mu = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev
        valid_dro = (raw_abs > 1e-6).float()
        w_dro = torch.sqrt((raw_abs * valid_dro) / dro_mu.clamp_min(self._EPS))
        w_dro_mean = (w_dro * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = w_dro / w_dro_mean.clamp_min(1e-8)
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=1.0, neginf=0.0)
        w_dro = 1.0 + event_mask * (w_dro - 1.0)

        shape_w = gate * w_dro
        if multivariate:
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(
                dim=(0, 1)
            ).clamp_min(self._EPS)
        else:
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(
                self._EPS
            )

        # ── LEVEL: Event-conditioned, asinh_plus, NO T multiplier ────
        # FIX 1: removed T multiplier (was T × avg, now just avg).
        #         This makes level gradient O(1/(B·n_ev)), matching shape.
        # FIX 2: asinh_plus instead of log_cosh.
        #         tanh saturates at ±1; asinh_plus grows logarithmically.
        pred_event_mean = (event_mask * y_pred).sum(dim=1) / n_ev.squeeze(1)
        true_event_mean = (event_mask * y_true).sum(dim=1) / n_ev.squeeze(1)
        gap = pred_event_mean - true_event_mean

        level_cell = self._asinh_plus(gap)          # FIX 2
        w_level = gate.amax(dim=1)

        if multivariate:
            loss_level = (                           # FIX 1: no T
                (w_level * level_cell).sum(dim=0)
                / w_level.sum(dim=0).clamp_min(self._EPS)
            )
        else:
            loss_level = (                           # FIX 1: no T
                (w_level * level_cell).sum()
                / w_level.sum().clamp_min(self._EPS)
            )

        # ── ZERO-ANCHOR: per-series normalization ────────────────────
        # FIX 3: divide by B (per-series), not zero_w.sum() (per-cell).
        #         V59 divided by B×(T−n_ev) ≈ 4500 for PGM, diluting
        #         the gradient 50×.  Per-series normalization makes the
        #         gradient scale with tanh(ŷ)/B — proportional to how
        #         far zeros are from zero, not how many there are.
        zero_w = 1.0 - gate
        zero_cell = self._log_cosh(y_pred)

        if multivariate:
            loss_zero = (
                (zero_w * zero_cell).sum(dim=(0, 1)) / B   # FIX 3
            )
        else:
            loss_zero = (
                (zero_w * zero_cell).sum() / B              # FIX 3
            )

        # ── Combine ───────────────────────────────────────────────────
        if multivariate:
            per_channel = loss_shape + loss_level + loss_zero
            total_loss = per_channel.sum()
            shape_c = loss_shape.detach().tolist()
            level_c = loss_level.detach().tolist()
            zero_c = loss_zero.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            total_loss = loss_shape + loss_level + loss_zero
            shape_c = [float(loss_shape.detach())]
            level_c = [float(loss_level.detach())]
            zero_c = [float(loss_zero.detach())]
            comp = [float(total_loss.detach())]

        # ── Diagnostic telemetry ──────────────────────────────────────
        with torch.no_grad():
            if multivariate:
                _n_ev = event_mask.sum(dim=(0, 1)).clamp_min(1.0)
                _w_ev = w_dro * event_mask
                _dm = _w_ev.sum(dim=(0, 1)) / _n_ev
                _dw2 = (_w_ev**2).sum(dim=(0, 1)) / _n_ev
                _dstd = (_dw2 - _dm**2).clamp_min(0).sqrt()
                dro_wmean_l = _dm.tolist()
                dro_wstd_l = _dstd.tolist()
                dro_wmax_l = w_dro.amax(dim=(0, 1)).tolist()
                dro_frac_up_l = (
                    ((w_dro > 1.0) * event_mask).sum(dim=(0, 1)) / _n_ev
                ).tolist()
                event_frac_l = event_mask.mean(dim=(0, 1)).tolist()

                gap_abs = gap.abs()
                gap_mean_l = gap_abs.mean(dim=0).tolist()
                gap_max_l = gap_abs.amax(dim=0).tolist()
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = (
                    (gap_abs * _ev_mask_s).sum(dim=0) / _n_ev_s
                ).tolist()
                gap_ev_max_l = ((gap_abs * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (
                    ((gap_abs > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s
                ).tolist()
                shape_dc_l = (
                    (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()
                )
                sl_ratio_l = (
                    loss_shape.detach()
                    / (loss_level + loss_zero).detach().clamp_min(self._EPS)
                ).tolist()
                zero_frac_l = zero_w.mean(dim=(0, 1)).tolist()
            else:
                _n_ev = event_mask.sum().clamp_min(1.0)
                _w_ev = w_dro * event_mask
                _dm = (_w_ev.sum() / _n_ev).item()
                _dw2 = ((_w_ev**2).sum() / _n_ev).item()
                dro_wmean_l = [_dm]
                dro_wstd_l = [max(0.0, _dw2 - _dm**2) ** 0.5]
                dro_wmax_l = [w_dro.max().item()]
                dro_frac_up_l = [
                    ((w_dro > 1.0) * event_mask).sum().item() / _n_ev.item()
                ]
                event_frac_l = [event_mask.mean().item()]

                gap_abs = gap.abs()
                gap_mean_l = [gap_abs.mean().item()]
                gap_max_l = [gap_abs.max().item()]
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [
                    ((gap_abs * _ev_mask_s).sum() / _n_ev_s).item()
                ]
                gap_ev_max_l = [(gap_abs * _ev_mask_s).amax().item()]
                gap_sat_l = [
                    ((gap_abs > 1.5) * _ev_mask_s).sum().item()
                    / _n_ev_s.item()
                ]
                shape_dc_l = [
                    (gate * e_shape).mean(dim=1).abs().mean().item()
                ]
                sl_ratio_l = [
                    float(
                        (
                            loss_shape.detach()
                            / (loss_level + loss_zero)
                            .detach()
                            .clamp_min(self._EPS)
                        ).item()
                    )
                ]
                zero_frac_l = [zero_w.mean().item()]

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossV60: per_channel={comp}"
            )

        n = len(comp)
        self._last_components = {
            "shape": shape_c,
            "level": level_c,
            "zero": zero_c,
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
            "zero_frac": zero_frac_l,
        }

        logger.debug(
            "SpotlightLossV60 | shape=%s level=%s zero=%s total=%.6f",
            shape_c,
            level_c,
            zero_c,
            total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV60(non_zero_threshold={self.tau})"