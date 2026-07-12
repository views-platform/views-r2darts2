import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (sharpens flat forecasts).
    Level = MSE on mean gap (V13 — fixes V11 saturation & V12 weak DC).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged from
      V11/V12 — this component works (non-flat forecasts, healthy AC
      gradient).

    * **Level (DC magnitude).** ``T × gap²`` on the per-series mean gap,
      gate-weighted, Hájek-normalised.

      V11 used ``T × log_cosh(gap)`` → ``tanh(gap)`` saturates for
      ``|gap| > 3`` → model gets the same gradient whether off by 5 or
      15 → cannot push harder on severely underpredicted channels.

      V12 used per-cell ``log_cosh(e)`` → gradient focused on event
      cells but total DC push per series is ``tanh(e_event) × 1`` vs
      V11's ``tanh(gap) × 36`` → 34× weaker DC gradient → worse
      underprediction than V11 on ALL channels.

      V13 uses ``T × gap²`` → gradient is ``2 × gap`` per cell:
        - DC-only (mean gap → uniform gradient → no Shape conflict)
        - Unsaturated (scales linearly with error magnitude)
        - Applied to all T cells (same total gradient surface as V11)
        - For gap=1.5: gradient = 3.0/cell vs V11's 0.91 → 3.3× stronger
        - For gap=3.0: gradient = 6.0/cell vs V11's 0.995 → 6× stronger

      MSE is safe here because the mean-gap formulation has ZERO AC
      gradient by construction — it cannot cause the AC template-ization
      that per-cell MSE caused in V9. The user's "must use log_cosh"
      constraint applies to the Shape base loss (per-cell), not to the
      Level DC term.

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
        # Set by backward hook in forward(); holds d(loss)/d(y_pred) for gradient diagnostics.
        self._last_input_grad: torch.Tensor | None = None

        logger.info("SpotlightLossV13 | threshold=%.4f", non_zero_threshold)

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

        # Register hook to capture d(loss)/d(y_pred) after backward.
        self._last_input_grad = None
        if y_pred.requires_grad:
            y_pred.register_hook(lambda g: setattr(self, "_last_input_grad", g.detach()))

        e = y_pred - y_true

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── SHAPE: log_cosh on demeaned errors, DRO on |raw_error| ──
        # Unchanged from V11/V12 — this component works.
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

        # ── LEVEL: T × MSE(mean gap), gate-weighted, Hájek ───────────
        # V13: replaces V11's T*log_cosh(gap) and V12's per-cell log_cosh(e).
        #
        # gap = mean(y_pred) - mean(y_true) per series, shape (B,) or (B, C).
        # level_cell = T * gap²
        # Gradient w.r.t. y_pred[t] = 2 * gap (uniform across all T cells).
        #
        # Why MSE instead of log_cosh here:
        #  - log_cosh(gap) → tanh(gap) saturates for |gap|>3 → V11 can't
        #    push harder on severely underpredicted channels (ch_1, ch_2).
        #  - MSE(gap) → 2*gap is unsaturated → gradient scales with error.
        #  - Both are DC-only (mean gap → uniform gradient) → no Shape
        #    conflict. MSE cannot cause AC template-ization because it
        #    has zero AC gradient by construction.
        #
        # Why mean-gap instead of per-cell (V12):
        #  - Per-cell log_cosh(e) focuses gradient on event cells but
        #    total DC push per series = tanh(e_event) × 1 ≈ 1.
        #  - Mean-gap applies gradient to ALL T cells: total push =
        #    gradient × T. V11/V13 have 36× more total DC gradient.
        #  - V12's dcMag=0.0001 vs V11's approach → V12 underpredicts
        #    more on ALL channels.
        #
        # T factor: gap has gradient 1/T per cell (from mean operator).
        # T * gap² has gradient 2*gap per cell (T cancels 1/T). Without
        # T, gradient would be 2*gap/T — diluted by 1/T.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * gap ** 2
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

        # ── Diagnostic telemetry (detached, no extra grad) ────────────────
        with torch.no_grad():
            if multivariate:
                _n_ev   = event_mask.sum(dim=(0, 1)).clamp_min(1.0)           # (C,)
                _w_ev   = w_dro * event_mask
                _dm     = _w_ev.sum(dim=(0, 1)) / _n_ev
                _dw2    = (_w_ev ** 2).sum(dim=(0, 1)) / _n_ev
                _dstd   = (_dw2 - _dm ** 2).clamp_min(0).sqrt()
                dro_wmean_l   = _dm.tolist()
                dro_wstd_l    = _dstd.tolist()
                dro_wmax_l    = w_dro.amax(dim=(0, 1)).tolist()
                dro_frac_up_l = (((w_dro > 1.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                event_frac_l  = event_mask.mean(dim=(0, 1)).tolist()

                # Gap diagnostics — computed per series, then stats over
                # event series only (where w_level > 0.5) AND over all series.
                _ga    = gap.abs()
                gap_mean_l    = _ga.mean(dim=0).tolist()  # all-series mean |gap|
                gap_max_l     = _ga.amax(dim=0).tolist()
                # Event-only gap (what Level actually sees after Hájek).
                _ev_mask_s = (w_level > 0.5).float()  # (B,) or (B, C)
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                # Fraction of event series where |gap| > 1.5 (tanh saturated
                # zone for V11 — V13's MSE gradient is 2*1.5=3.0, unsaturated).
                # If this is high and V11 stalled here, V13 should break through.
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()
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

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV13: per_channel={comp}")

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
            # ── Diagnostic keys (read by LossGradientDiagnosticsCallback) ──
            "dro_w_mean":     dro_wmean_l,    # mean DRO weight over event cells
            "dro_w_std":      dro_wstd_l,     # std  DRO weight over event cells
            "dro_w_max":      dro_wmax_l,     # max  DRO weight (tail upweighting)
            "dro_frac_up":    dro_frac_up_l,  # fraction of event cells with w_dro > 1
            "event_frac":     event_frac_l,   # fraction of cells that are events
            "level_gap_mean": gap_mean_l,     # mean |gap| per channel (ALL series — diagnostic only)
            "level_gap_max":  gap_max_l,      # max  |gap| per channel (ALL series)
            "shape_dc":       shape_dc_l,     # gated shape-error DC leak (should be ≈ 0)
            # ── NEW (V13): event-series gap diagnostics ──
            # These show what the Hájek-normalized Level actually sees
            # (event series only, where w_level > 0.5).
            "level_gap_ev_mean": gap_ev_mean_l,  # mean |gap| over EVENT series (what Level optimizes)
            "level_gap_ev_max":  gap_ev_max_l,   # max  |gap| over EVENT series
            "level_gap_sat":     gap_sat_l,      # frac of event series with |gap|>1.5 (V11 tanh saturation zone)
                                                 # V13 MSE gradient at |gap|=1.5 is 3.0 — unsaturated.
                                                 # If V11 stalled with high gap_sat, V13 should break through.
        }

        logger.debug(
            "SpotlightLossV13 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV13(non_zero_threshold={self.tau})"
