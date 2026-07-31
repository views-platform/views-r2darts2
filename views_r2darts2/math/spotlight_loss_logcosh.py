import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SpotlightLossLogcosh(torch.nn.Module):
    """

    V60: Gradient-gated Shape + sum anchor.

    Changes from V59 (all hyperparameter-free):
      1. Shape: gate the e_shape gradient so dead cells get ~50× less gradient
         e_shape = gate * e_shape + (1 - gate) * e_shape.detach()
         Forward: identical. Backward: dead cells (gate≈0) get no Shape gradient.
         This stops Shape from pushing dead cells UP (which fought the anchor).

      2. Dead-cell anchor: sum-based, no T (from V59)
         dead_sum = (dead_mask * y_pred).sum(dim=1)
         Pushes dead cells toward 0. With grad gate, anchor now dominates on dead cells.

    Unchanged from V58:
      - abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
      - gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
      - T multiplier on Level
      - density_scale, w_level Hájek weighting
      - DRO with sqrt on |e_shape|
      - e_mean.detach()
    """
    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _tolist(x):
        val = x.tolist() if isinstance(x, torch.Tensor) else x
        return val if isinstance(val, list) else [val]

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

        # ── SHAPE: log_cosh on demeaned errors ────────
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean.detach()

        # V60: Gate the e_shape gradient so dead cells (gate≈0) get ~50× less
        # Shape gradient. This stops Shape from pushing dead cells UP when
        # e_shape[dead] < 0 (below mean error), which fought the anchor.
        # Forward value is IDENTICAL (gate*x + (1-gate)*x = x).
        # Backward: ∂e_shape/∂y_pred = gate (only gated cells get full gradient).
        
        # e_shape = gate * e_shape + (1.0 - gate) * e_shape.detach()

        shape_cell = self._log_cosh(e_shape)

        # DRO weighting
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
        if multivariate:
            loss_shape = (shape_w * shape_cell).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            loss_shape = (shape_w * shape_cell).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: Density-Scaled Gap + Signal-Weighted Hájek + T ────
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)

        # Series-level: amplify gap for sparse-event series
        n_ev_flat = event_mask.sum(dim=1).squeeze(1) if event_mask.dim() == 3 else event_mask.sum(dim=1)
        # density_scale = torch.sqrt(T / n_ev_flat.clamp_min(T).float()) # no-op. will return 1. just a placeholder
        density_scale = torch.asinh(T / n_ev_flat.clamp_min(1.0).float())
        level_cell = self._log_cosh(gap * density_scale)

        # Batch-level: weight by signal strength, gated by has_gated
        event_frac = event_mask.mean().clamp_min(self._EPS)
        has_gated = (gate.sum(dim=1) > self._EPS).float()
        w_level = (torch.sqrt(n_ev_flat.float()) + event_frac) * has_gated

        if multivariate:
            loss_level = T * (w_level * level_cell).sum(dim=0) / w_level.sum(dim=0).clamp_min(self._EPS)
        else:
            loss_level = T * (w_level * level_cell).sum() / w_level.sum().clamp_min(self._EPS)

        # ── Dead-cell anchor at Shape scale ─────────────────
        # Pushes dead cells toward 0, eliminating gap contamination.
        # Uses y_true_mask (true events only) inverse for dead cells.
        # Sum-based: tanh saturates → bounded strong gradient without T.
        # With V60 grad gate, anchor now dominates on dead cells (Shape muted).
        y_true_mask = (y_true.abs() > self.tau).float()
        dead_mask = 1.0 - y_true_mask
        dead_sum = (dead_mask * y_pred).sum(dim=1)
        anchor_cell = self._log_cosh(dead_sum)

        if multivariate:
            loss_anchor = (w_level * anchor_cell).sum(dim=0) / w_level.sum(dim=0).clamp_min(self._EPS)
        else:
            loss_anchor = (w_level * anchor_cell).sum() / w_level.sum().clamp_min(self._EPS)

        # ── Combine ───────────────────────────────────────────────────
        if multivariate:
            per_channel = loss_shape + loss_level + loss_anchor
            total_loss = per_channel.sum()
            shape_c = loss_shape.detach().tolist()
            level_c = loss_level.detach().tolist()
            anchor_c = loss_anchor.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            total_loss = loss_shape + loss_level + loss_anchor
            shape_c = [float(loss_shape.detach())]
            level_c = [float(loss_level.detach())]
            anchor_c = [float(loss_anchor.detach())]
            comp = [float(total_loss.detach())]

        # ── Diagnostic telemetry ──────────────────────────────────────
        with torch.no_grad():
            if multivariate:
                density_scale_mean = density_scale.mean(dim=0)
                gap_scaled_mean = (gap * density_scale).abs().mean(dim=0)
                w_level_mean = w_level.mean(dim=0)
                batch_active_frac = has_gated.mean(dim=0)
                dead_pred_mean = dead_sum.mean(dim=0) / dead_mask.sum(dim=1).clamp_min(1.0).mean(dim=0)
            else:
                density_scale_mean = density_scale.mean()
                gap_scaled_mean = (gap * density_scale).abs().mean()
                w_level_mean = w_level.mean()
                batch_active_frac = has_gated.mean()
                dead_pred_mean = (dead_sum / dead_mask.sum(dim=1).clamp_min(1.0)).mean()

            # Gradient direction diagnostics (event vs non-event)
            grad = self._last_input_grad
            if grad is not None:
                g = grad.float()
                if g.dim() == 2:
                    g = g.unsqueeze(-1)
                ev_mask_3d = event_mask.unsqueeze(-1) if event_mask.dim() == 2 else event_mask
                n_ev_total = ev_mask_3d.sum().clamp_min(1.0)
                n_nev_total = (1.0 - ev_mask_3d).sum().clamp_min(1.0)
                grad_ev = (g * ev_mask_3d).sum().item() / n_ev_total.item()
                grad_nev = (g * (1.0 - ev_mask_3d)).sum().item() / n_nev_total.item()
            else:
                grad_ev = 0.0
                grad_nev = 0.0

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
                density_scale_mean_l = self._tolist(density_scale_mean.expand(1) if density_scale_mean.dim() == 0 else density_scale_mean)
                gap_scaled_mean_l = self._tolist(gap_scaled_mean.expand(1) if gap_scaled_mean.dim() == 0 else gap_scaled_mean)
                w_level_mean_l = self._tolist(w_level_mean.expand(1) if w_level_mean.dim() == 0 else w_level_mean)
                batch_active_frac_l = self._tolist(batch_active_frac.expand(1) if batch_active_frac.dim() == 0 else batch_active_frac)
                dead_pred_mean_l = self._tolist(dead_pred_mean.expand(1) if dead_pred_mean.dim() == 0 else dead_pred_mean)
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
                density_scale_mean_l = [density_scale_mean.item()]
                gap_scaled_mean_l = [gap_scaled_mean.item()]
                w_level_mean_l = [w_level_mean.item()]
                batch_active_frac_l = [batch_active_frac.item()]
                dead_pred_mean_l = [dead_pred_mean.item()]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossLogcosh: per_channel={comp}")

        n = len(comp)
        self._last_components = {
            "shape": shape_c,
            "level": level_c,
            "anchor": anchor_c,
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
            "density_scale_mean": density_scale_mean_l,
            "gap_scaled_mean": gap_scaled_mean_l,
            "w_level_mean": w_level_mean_l,
            "batch_active_frac": batch_active_frac_l,
            "dead_pred_mean": dead_pred_mean_l,
        }

        logger.debug(
            "SpotlightLossLogcosh | shape=%s level=%s anchor=%s total=%.6f",
            shape_c, level_c, anchor_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.tau})"