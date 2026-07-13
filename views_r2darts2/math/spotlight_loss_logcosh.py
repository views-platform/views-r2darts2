import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """V42: V13 + per-series ac_scale + ungated Level.

    Fixes two overlooked blind spots in V13:
    1. ac_scale computed per-batch → high-variance countries diluted
       Shape penalty for peaceful countries → enabled sporadic spikes.
    2. w_level = gate.amax → peaceful countries got 100× less Level
       gradient → model had no incentive to push peaceful predictions
       to zero → aggregate overprediction.

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** V13's log_cosh, but with **per-series
      ac_scale** (dim=1 instead of dim=(0,1)). Each country's Shape
      loss is scaled by its OWN variance, not contaminated by other
      countries in the batch. A spike on a peaceful country now gets
      the full Shape penalty.

    * **Level (DC magnitude).** V13's MSE on mean gap, but with
      **w_level = 1.0** (constant, not gate-dependent). All series
      get equal Level weight. Peaceful countries now get the same
      calibration push as event countries. The gate remains only for
      Shape (where it's needed to focus on events).

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

        logger.info("SpotlightLossV42 | threshold=%.4f", non_zero_threshold)

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
        # V13 structure, but with PER-SERIES ac_scale (V42 fix).
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        # V42 FIX: per-series ac_scale (dim=1 only, not dim=(0,1))
        # This prevents high-variance countries from diluting the Shape
        # penalty for peaceful countries.
        if multivariate:
            ac_scale = true_ac.std(dim=1, keepdim=True).clamp_min(self.tau)  # (B, 1, C)
        else:
            ac_scale = true_ac.std(dim=1, keepdim=True).clamp_min(self.tau)  # (B, 1)
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

        # ── LEVEL: T × MSE(mean gap), UNGATED (V42 fix) ──────────────
        # V42 FIX: w_level = 1.0 (constant, not gate-dependent).
        # This ensures peaceful countries get the same Level push as
        # event countries. The gate stays only for Shape.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * gap ** 2
        w_level = torch.ones_like(gap)  # V42: constant 1.0, not gate.amax

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
                # Event-only gap
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()  # use gate for diagnostics
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # V42: per-series ac_scale diagnostics
                _ac_scale_mean = ac_scale.mean(dim=0).squeeze(0) if ac_scale.dim() > 2 else ac_scale.mean()
                ac_scale_mean_l = _ac_scale_mean.tolist() if _ac_scale_mean.dim() > 0 else [float(_ac_scale_mean)]
                _ac_scale_max = ac_scale.max(dim=0).values.squeeze(0) if ac_scale.dim() > 2 else ac_scale.max()
                ac_scale_max_l = _ac_scale_max.tolist() if _ac_scale_max.dim() > 0 else [float(_ac_scale_max)]
                _ac_scale_min = ac_scale.min(dim=0).values.squeeze(0) if ac_scale.dim() > 2 else ac_scale.min()
                ac_scale_min_l = _ac_scale_min.tolist() if _ac_scale_min.dim() > 0 else [float(_ac_scale_min)]

                # V42: peaceful country diagnostics
                _peace_mask = (gate.amax(dim=1) < 0.1).float()  # peaceful series
                _n_peace = _peace_mask.sum(dim=0).clamp_min(1.0)
                _gap_peace = (_ga * _peace_mask).sum(dim=0) / _n_peace
                gap_peace_mean_l = _gap_peace.tolist()
                # Mean y_pred for peaceful countries (should be →0)
                _yp_peace = (y_pred.abs() * _peace_mask.unsqueeze(1)).sum(dim=(0,1)) / (_n_peace * T)
                y_pred_peace_l = _yp_peace.tolist()

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
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l  = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                _ac_scale_mean = ac_scale.mean()
                ac_scale_mean_l = [float(_ac_scale_mean)]
                _ac_scale_max = ac_scale.max()
                ac_scale_max_l = [float(_ac_scale_max)]
                _ac_scale_min = ac_scale.min()
                ac_scale_min_l = [float(_ac_scale_min)]
                _peace_mask = (gate.amax(dim=1) < 0.1).float()
                _n_peace = _peace_mask.sum().clamp_min(1.0)
                _gap_peace = (_ga * _peace_mask).sum() / _n_peace
                gap_peace_mean_l = [_gap_peace.item()]
                _yp_peace = (y_pred.abs() * _peace_mask.unsqueeze(1)).sum() / (_n_peace * T)
                y_pred_peace_l = [_yp_peace.item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV42: per_channel={comp}")

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
            "dro_w_mean":     dro_wmean_l,
            "dro_w_std":      dro_wstd_l,
            "dro_w_max":      dro_wmax_l,
            "dro_frac_up":    dro_frac_up_l,
            "event_frac":     event_frac_l,
            "level_gap_mean": gap_mean_l,
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "level_gap_sat":     gap_sat_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V42: per-series ac_scale diagnostics ──
            # Should now vary significantly across series (not a single
            # batch-wide value). Min should be ≈ tau (peaceful countries),
            # max should be >> tau (high-variance countries).
            "ac_scale_mean":  ac_scale_mean_l,
            "ac_scale_max":   ac_scale_max_l,
            "ac_scale_min":   ac_scale_min_l,
            # ── V42: peaceful country diagnostics ──
            # These show whether the blind spot is fixed. If gap_peace_mean
            # decreases over training, Level is now pushing peaceful
            # predictions toward zero. If y_pred_peace → 0, the blind
            # spot is closed.
            "gap_peace_mean":  gap_peace_mean_l,  # mean |gap| for peaceful countries (should →0)
            "y_pred_peace":    y_pred_peace_l,    # mean |y_pred| for peaceful countries (should →0)
        }

        logger.debug(
            "SpotlightLossV42 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV42(non_zero_threshold={self.tau})"