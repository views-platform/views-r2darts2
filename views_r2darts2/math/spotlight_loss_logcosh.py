import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SpotlightLossLogcosh(torch.nn.Module):
    """V58: V57 + Remove ac_scale entirely.
    
    Per-series ac_scale gave peaceful countries 5.6× stronger Shape
    gradient than spike countries (1/0.88 vs 1/5.0). This inverted
    priorities — the model was incentivized to ignore spikes.
    
    Removing ac_scale lets tanh(e) naturally prioritize large errors:
    - Spike (e=5): tanh(5)=1.0 (full push)
    - Peace (e=0.3): tanh(0.3)=0.29 (gentle push)
    
    Division of labor:
    - Shape (no ac_scale): aggressively learns spikes
    - Level (AsinhPlus, gate.amax): calibrates mean for active series
    """
    _EPS = 1e-6
    _K = 4  # 4 blocks of 9 months

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None
        logger.info("SpotlightLossV58 | threshold=%.4f K=%d", non_zero_threshold, self._K)

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

    def forward(self, y_pred, y_true):
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        B, T = y_pred.shape[:2]

        e = y_pred - y_true

        # ── Event gate (binary, y_true-based mask + abs_max binary gate) ──
        # event_mask = (y_true.abs() > self.tau).float()
        # abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        # gate = (abs_max > self.tau).float()
        # n_ev = event_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # Alternative block
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())        
        gate = torch.sigmoid(15.0 * (abs_max - self.tau))         
        event_mask = (abs_max > self.tau).float()
        n_ev = event_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # ── SHAPE: background-referenced demeaning ──
        bg_mask = 1.0 - event_mask
        n_bg = bg_mask.sum(dim=1, keepdim=True)
        has_bg = (n_bg > 0).float()
        e_bg_mean = (bg_mask * e).sum(dim=1, keepdim=True) / n_bg.clamp_min(1e-6)
        e_ev_mean = (event_mask * e).sum(dim=1, keepdim=True) / n_ev
        e_mean = has_bg * e_bg_mean + (1.0 - has_bg) * e_ev_mean
        e_shape = e - e_mean

        shape_cell = self._log_cosh(e_shape)

        # DRO weighting (unchanged)
        raw_abs = e_shape.abs().detach()
        dro_mu = (raw_abs * event_mask).sum(dim=1, keepdim=True) / n_ev
        valid_dro = (raw_abs > 1e-6).float()
        w_dro = torch.sqrt((raw_abs * valid_dro) / dro_mu.clamp_min(1e-6))
        w_dro_mean = (w_dro * event_mask).sum(dim=1, keepdim=True) / n_ev
        w_dro = w_dro / w_dro_mean.clamp_min(1e-8)
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=1.0, neginf=0.0)
        w_dro = 1.0 + event_mask * (w_dro - 1.0)

        shape_w = gate * w_dro
        if multivariate:
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: gap_event + gap_non_event (decomposed) ──
        # n_non_ev = (T - event_mask.sum(dim=1, keepdim=True)).clamp_min(1.0)
        # e_non_event_mean = ((1.0 - event_mask) * e).sum(dim=1, keepdim=True) / n_non_ev
        # gap_event = e_ev_mean.squeeze(1)
        # gap_non_event = e_non_event_mean.squeeze(1)
        # level_cell = self._log_cosh(gap_event) + self._log_cosh(gap_non_event)
        # w_level = torch.ones_like(gap_event)

        # if multivariate:
        #     loss_level = T * (w_level * level_cell).sum(dim=0) / w_level.sum(dim=0).clamp_min(self._EPS)
        # else:
        #     loss_level = T * (w_level * level_cell).sum() / w_level.sum().clamp_min(self._EPS)

        n_ev_level = n_ev.squeeze(1).clamp_min(1.0)  # (B,)
        event_pred_mean = (event_mask * y_pred).sum(dim=1) / n_ev_level
        event_true_mean = (event_mask * y_true).sum(dim=1) / n_ev_level
        gap = event_pred_mean - event_true_mean

        level_cell = self._log_cosh(gap)
        w_level = gate.amax(dim=1)

        if multivariate:
            loss_level = T * (w_level * level_cell).sum(dim=0) / w_level.sum(dim=0).clamp_min(self._EPS)
        else:
            loss_level = T * (w_level * level_cell).sum() / w_level.sum().clamp_min(self._EPS)

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

                # gap_global = y_pred.mean(dim=1) - y_true.mean(dim=1)
                # _ga = gap_global.abs()
                gap_event_tel = event_pred_mean - event_true_mean  # (B,) or (B, C)
                _ga = gap_event_tel.abs()
                gap_mean_l = _ga.mean(dim=0).tolist()
                gap_max_l = _ga.amax(dim=0).tolist()
                _ev_mask_s = (gate.amax(dim=1) > 0.88).float()
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

                # gap_global = y_pred.mean(dim=1) - y_true.mean(dim=1)
                # _ga = gap_global.abs()
                gap_event_tel = event_pred_mean - event_true_mean  # (B,) or (B, C)
                _ga = gap_event_tel.abs()
                gap_mean_l = [_ga.mean().item()]
                gap_max_l = [_ga.max().item()]
                _ev_mask_s = (gate.amax(dim=1) > 0.88).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l = [(gate * e_shape).mean(dim=1).abs().mean().item()]

                sl_ratio_l = [float((loss_shape.detach() / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV58: per_channel={comp}")

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
            "SpotlightLossV58 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV58(non_zero_threshold={self.tau})"