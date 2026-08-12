import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLossLogcosh v66

    Fixes from forensic audit:
      1. DRO now uses |e_shape| (demeaned error) instead of |e| (raw error).
         This stops false-alarm cells (large |e| but detached e_shape) from
         inflating the Hájek denominator and diluting the Shape gradient on
         true events.
      2. Shape gradient gate now uses event_mask (includes false alarms) instead
         of y_true_mask (true events only). This lets Shape push false alarms
         DOWN, closing the loophole where the model could create false alarms
         without any Shape penalty.
      3. Event-only gap falls back to the global mean gap for dead-only series
         (series with zero true events). This prevents Level from providing
         zero gradient on those series, which caused template collapse.
      4. Dead-cell anchor scaled by density_scale (batch mean) instead of
         either 1× (too weak) or T (too strong). This balances the per-cell
         anchor force against Level's per-cell force.

    Unchanged from prior accepted version:
      - _log_cosh numerically stable implementation (|x| + softplus(-2|x|) - log2)
      - Shape/Level decomposition with Hájek self-normalization
      - T multiplier on Level (T=36) for level-dominance
      - Event-only gap (clean signal, no dead-cell contamination in value)
      - e_mean computed on all cells, detached
      - Sum-based dead anchor (tanh saturates → bounded strong gradient)
      - Gate = sigmoid(10 * (max(|y_true|, |y_pred|) - tau))
      - DRO with sqrt(|e_shape| / mu) normalization
    """
    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable log(cosh(x)).

        Uses |x| + softplus(-2|x|) - log(2) which is stable for all x.
        Gradient is tanh(x), which asymptotes to ±1 (never vanishes).
        """
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _tolist(x):
        val = x.tolist() if isinstance(x, torch.Tensor) else x
        return val if isinstance(val, list) else [val]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        B, T = y_pred.shape[:2]

        self._last_input_grad = None
        if y_pred.requires_grad:
            y_pred.register_hook(lambda g: setattr(self, "_last_input_grad", g.detach().cpu()))

        e = y_pred - y_true

        # ── Event gate ───────────────────────────────────────────────
        # Includes false alarms (y_pred > tau) so Shape can push them down.
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── Masks ────────────────────────────────────────────────────
        # y_true_mask: true events only (for gap + dead anchor)
        y_true_mask = (y_true.abs() > self.tau).float()

        # event_mask: true events + false alarms (for Shape grad gate + DRO)
        event_mask = (abs_max > self.tau).float()

        # ── SHAPE: log_cosh on demeaned errors ───────────────────────
        # Demean by all-cell mean (detached). On 98%-zero data e_mean ≈ 0,
        # so e_shape ≈ e. Shape sees the raw pattern error.
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean.detach()

        # FIX #2: Gate gradient using event_mask (includes false alarms) instead
        # of y_true_mask. This lets Shape push false alarms DOWN, closing the
        # loophole where the model could create false alarms without penalty.
        # On true dead cells (y_true ≤ tau AND y_pred ≤ tau), event_mask=0 and
        # gate≈0, so Shape gradient is still zero — no change for true dead cells.
        shape_cell = self._log_cosh(
            event_mask * e_shape + (1.0 - event_mask) * e_shape.detach()
        )

        # FIX #1: DRO uses |e_shape| (demeaned error) instead of |e| (raw error).
        # This stops false-alarm cells (large |e| but detached e_shape) from
        # inflating the Hájek denominator and diluting the Shape gradient on
        # true events.
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

        # ── LEVEL: Event-Only Gap + Hájek + T ────────────────────────
        # FIX #3: Event-only gap with fallback to global gap for dead-only series.
        # For series WITH events: gap = mean(event_pred) - mean(event_true) — clean.
        # For series WITHOUT events: gap = y_pred.mean() - y_true.mean() — prevents
        #   Level from providing zero gradient (which caused template collapse).
        n_ev_safe = y_true_mask.sum(dim=1).clamp_min(1.0)
        gap_event = (y_true_mask * y_pred).sum(dim=1) / n_ev_safe - (y_true_mask * y_true).sum(dim=1) / n_ev_safe
        gap_global = y_pred.mean(dim=1) - y_true.mean(dim=1)
        has_events = (y_true_mask.sum(dim=1) > 0.5).float()
        gap = has_events * gap_event + (1.0 - has_events) * gap_global

        level_cell = self._log_cosh(gap)

        # Batch-level: weight by signal strength, gated by has_gated
        n_ev_flat = event_mask.sum(dim=1).squeeze(1) if event_mask.dim() == 3 else event_mask.sum(dim=1)
        event_frac = event_mask.mean().clamp_min(self._EPS)
        has_gated = (gate.sum(dim=1) > self._EPS).float()
        w_level = (torch.sqrt(n_ev_flat.float()) + event_frac) * has_gated

        if multivariate:
            loss_level = T * (w_level * level_cell).sum(dim=0) / w_level.sum(dim=0).clamp_min(self._EPS)
        else:
            loss_level = T * (w_level * level_cell).sum() / w_level.sum().clamp_min(self._EPS)

        # ── Dead-cell anchor ─────────────────────────────────────────
        # FIX #4: Scale anchor by density_scale (batch mean) instead of 1× (too
        # weak) or T (too strong). This balances the per-cell anchor force
        # against Level's per-cell force:
        #   Level per event cell ≈ T * tanh(gap) / (n_ev * w_sum)
        #   Anchor per dead cell ≈ density_scale * tanh(dead_sum) / w_sum
        # On CM (n_ev=6, density_scale≈2.5): anchor is ~2.5×, Level is ~6× per
        # cell → aggregate forces are balanced (30 dead × 2.5 ≈ 6 events × 6 × T).
        # On PGM (n_ev=1, density_scale≈4.3): anchor per cell is 4.3×, Level per
        # cell is 36× → Level dominates events, anchor holds dead cells.
        density_scale = torch.asinh(T / n_ev_flat.clamp_min(1.0).float())
        density_scale_mean = density_scale.mean()

        dead_mask = 1.0 - y_true_mask
        dead_sum = (dead_mask * y_pred).sum(dim=1)
        anchor_cell = self._log_cosh(dead_sum)

        if multivariate:
            loss_anchor = density_scale_mean * (w_level * anchor_cell).sum(dim=0) / w_level.sum(dim=0).clamp_min(self._EPS)
        else:
            loss_anchor = density_scale_mean * (w_level * anchor_cell).sum() / w_level.sum().clamp_min(self._EPS)

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
        # All telemetry is computed on CPU to avoid allocating large temporary
        # tensors on the GPU when device memory is already near capacity.
        with torch.no_grad():
            _event_mask_cpu = event_mask.cpu()
            _w_dro_cpu      = w_dro.cpu()
            _gate_cpu       = gate.cpu()
            _e_shape_cpu    = e_shape.cpu()
            _gap_cpu        = (y_pred.mean(dim=1) - y_true.mean(dim=1)).cpu()
            _density_scale_cpu = density_scale.cpu()
            _gap_scaled_cpu = (_gap_cpu * _density_scale_cpu).abs()
            _w_level_cpu    = w_level.cpu()
            _has_gated_cpu  = has_gated.cpu()
            _dead_sum_cpu   = dead_sum.cpu()

            # Gradient direction diagnostics (event vs non-event)
            grad = self._last_input_grad
            if grad is not None:
                g = grad.float().cpu()
                if g.dim() == 2:
                    g = g.unsqueeze(-1)
                ev_mask_3d = _event_mask_cpu.unsqueeze(-1) if _event_mask_cpu.dim() == 2 else _event_mask_cpu
                n_ev_total = ev_mask_3d.sum().clamp_min(1.0)
                n_nev_total = (1.0 - ev_mask_3d).sum().clamp_min(1.0)
                grad_ev = (g * ev_mask_3d).sum().item() / n_ev_total.item()
                grad_nev = (g * (1.0 - ev_mask_3d)).sum().item() / n_nev_total.item()
            else:
                grad_ev = 0.0
                grad_nev = 0.0

            if multivariate:
                _n_ev = _event_mask_cpu.sum(dim=(0, 1)).clamp_min(1.0)
                _w_ev = _w_dro_cpu * _event_mask_cpu
                _dm = _w_ev.sum(dim=(0, 1)) / _n_ev
                _dw2 = (_w_ev ** 2).sum(dim=(0, 1)) / _n_ev
                _dstd = (_dw2 - _dm ** 2).clamp_min(0).sqrt()
                dro_wmean_l = _dm.tolist()
                dro_wstd_l = _dstd.tolist()
                dro_wmax_l = _w_dro_cpu.amax(dim=(0, 1)).tolist()
                dro_frac_up_l = (((_w_dro_cpu > 1.0) * _event_mask_cpu).sum(dim=(0, 1)) / _n_ev).tolist()
                event_frac_l = _event_mask_cpu.mean(dim=(0, 1)).tolist()

                _ga = _gap_cpu.abs()
                gap_mean_l = _ga.mean(dim=0).tolist()
                gap_max_l = _ga.amax(dim=0).tolist()
                _ev_mask_s = (_gate_cpu.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                shape_dc_l = (_gate_cpu * _e_shape_cpu).mean(dim=1).abs().mean(dim=0).tolist()

                sl_ratio_l = (loss_shape.detach() / loss_level.detach().clamp_min(self._EPS)).tolist()
                density_scale_mean_l = self._tolist(_density_scale_cpu.mean(dim=0))
                gap_scaled_mean_l    = self._tolist(_gap_scaled_cpu.mean(dim=0))
                w_level_mean_l       = self._tolist(_w_level_cpu.mean(dim=0))
                batch_active_frac_l  = self._tolist(_has_gated_cpu.mean(dim=0))
                dead_sum_mean_l      = self._tolist(_dead_sum_cpu.mean(dim=0))
            else:
                _n_ev = _event_mask_cpu.sum().clamp_min(1.0)
                _w_ev = _w_dro_cpu * _event_mask_cpu
                _dm = (_w_ev.sum() / _n_ev).item()
                _dw2 = ((_w_ev ** 2).sum() / _n_ev).item()
                dro_wmean_l = [_dm]
                dro_wstd_l = [max(0.0, _dw2 - _dm ** 2) ** 0.5]
                dro_wmax_l = [_w_dro_cpu.max().item()]
                dro_frac_up_l = [((_w_dro_cpu > 1.0) * _event_mask_cpu).sum().item() / _n_ev.item()]
                event_frac_l = [_event_mask_cpu.mean().item()]

                _ga = _gap_cpu.abs()
                gap_mean_l = [_ga.mean().item()]
                gap_max_l = [_ga.max().item()]
                _ev_mask_s = (_gate_cpu.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l = [(_gate_cpu * _e_shape_cpu).mean(dim=1).abs().mean().item()]

                sl_ratio_l = [float((loss_shape.detach() / loss_level.detach().clamp_min(self._EPS)).item())]
                density_scale_mean_l = [_density_scale_cpu.mean().item()]
                gap_scaled_mean_l    = [_gap_scaled_cpu.mean().item()]
                w_level_mean_l       = [_w_level_cpu.mean().item()]
                batch_active_frac_l  = [_has_gated_cpu.mean().item()]
                dead_sum_mean_l      = [_dead_sum_cpu.mean().item()]

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
            "dead_sum_mean": dead_sum_mean_l,
        }

        logger.debug(
            "SpotlightLossLogcosh | shape=%s level=%s anchor=%s total=%.6f",
            shape_c, level_c, anchor_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.tau})"
