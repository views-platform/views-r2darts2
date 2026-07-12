import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (sharpens flat forecasts).
    Level = per-cell log_cosh on raw error (V12 — fixes V11 underprediction).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised.

      The DRO uses ``|raw_error|`` (not shape loss) as the signal.
      On a flat forecast, shape loss is the same at all cells → DRO is
      neutral. But ``|raw_error|`` is large at peaks/valleys and zero
      at the mean → DRO upweights the cells that need sharpening.

    * **Level (DC magnitude).** Per-cell ``log_cosh(e)`` on raw error,
      gate-weighted, Hájek-normalised.

      V11 used ``T × log_cosh(mean gap)`` whose gradient ``tanh(gap)``
      saturates for ``|gap| > 3`` — the model gets the SAME gradient
      whether it is off by 5 or 15 on sparse channels (ch_1, ch_2).
      Per-cell ``log_cosh(e)`` also saturates at ``tanh ≤ 1`` for big
      ``|e|``, BUT the gradient is FOCUSED on event cells (via gate)
      rather than diluted uniformly across all 36 cells. The network
      is told "increase output HERE" instead of "increase mean across
      all cells," which is a different gradient direction that doesn't
      rely on the network translating a bias-push into event-cell
      increases.

      Both terms use log_cosh → both bounded at ``tanh ≤ 1`` → balanced.
      Per-cell Level includes AC components that overlap with Shape —
      this REINFORCES Shape on AC (both push toward y_true at event
      cells), so no conflict. The DC push comes from the per-cell raw
      errors at event cells.

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

        logger.info("SpotlightLossV12 | threshold=%.4f", non_zero_threshold)

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
        # Fires once per backward pass; stores (B,T) or (B,T,C) gradient for
        # LossGradientDiagnosticsCallback to decompose into DC (level) and AC (shape) parts.
        self._last_input_grad = None
        if y_pred.requires_grad:
            y_pred.register_hook(lambda g: setattr(self, "_last_input_grad", g.detach()))

        e = y_pred - y_true

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── SHAPE: log_cosh on demeaned errors, DRO on |raw_error| ──
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        # DRO on |raw_error| (detached) — upweights peaks/valleys on
        # flat forecasts where shape loss alone is uniform.
        # Event-only mean to prevent peace-noise amplification.
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

        # ── LEVEL: per-cell log_cosh on raw error, gate-weighted, Hájek ─
        # V12 change: per-cell log_cosh(e) instead of T * log_cosh(mean gap).
        #
        # V11's mean-gap formulation saturates: tanh(gap) ≈ -1 for |gap| > 3,
        # so the model gets the same gradient whether off by 5 or 15. On sparse
        # channels (ch_1, ch_2) where mean gap = -5 to -15 even after training,
        # the gradient is fully saturated and cannot push harder.
        #
        # Per-cell log_cosh(e) also saturates at tanh ≤ 1 for big |e|, BUT:
        #  (a) no mean dilution — tanh(e_event) saturates at the cell-level
        #      error magnitude, not the per-series mean (which is 36× smaller).
        #  (b) gradient FOCUSED on event cells (gate ≈ 1) and zero on peace
        #      cells (gate ≈ 0) — network is told "increase output HERE."
        #  (c) loss scale preserved: for one event of magnitude M, both V11
        #      (T * log_cosh(-M/36) ≈ M) and V12 (log_cosh(-M) ≈ M) give M.
        #
        # AC/DC overlap with Shape is REINFORCEMENT (both push toward y_true
        # at event cells), not conflict. DC push comes from per-cell raw
        # errors at event cells — the component V11 was missing.
        level_cell = self._log_cosh(e)
        w_level = gate

        if multivariate:
            loss_level = (w_level * level_cell).sum(dim=(0, 1)) / w_level.sum(dim=(0, 1)).clamp_min(self._EPS)
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
        # Computed before NaN check so they're always populated on a valid step.
        with torch.no_grad():
            # Per-series mean gap (still computed for diagnostics even though
            # V12 Level no longer uses it — this is the y_hat_bar/ch_X_ratio signal).
            gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)

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
                _ga    = gap.abs()
                gap_mean_l    = _ga.mean(dim=0).tolist()
                gap_max_l     = _ga.amax(dim=0).tolist()
                # Shape DC: mean over time of gated demeaned error per series, then
                # mean over batch. Pure AC projection gives 0; any leak indicates
                # gate weighting reintroducing a DC component.
                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()
                # ── NEW (V12): per-cell raw-error stats over event cells ──
                # |e| at event cells — directly measures what the new Level sees.
                _e_ev   = e.abs() * event_mask
                level_e_mean_l = (_e_ev.sum(dim=(0, 1)) / _n_ev).tolist()
                level_e_max_l  = (_e_ev.amax(dim=(0, 1))).tolist()
                # Fraction of event cells where |e| > 3 (tanh saturated zone).
                # High value → most event cells are at maximum gradient (can't
                # push harder); low value → headroom for gradient to scale with error.
                level_e_sat_l  = ((e.abs() > 3.0) * event_mask).sum(dim=(0, 1)).div(_n_ev).tolist()
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
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                _e_ev   = e.abs() * event_mask
                level_e_mean_l = [(_e_ev.sum() / _n_ev).item()]
                level_e_max_l  = [_e_ev.max().item()]
                level_e_sat_l  = [((e.abs() > 3.0) * event_mask).sum().item() / _n_ev.item()]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV12: per_channel={comp}")

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
            "level_gap_mean": gap_mean_l,     # mean |gap| per channel (per-series mean calibration; y_hat_bar ratio signal)
            "level_gap_max":  gap_max_l,      # max  |gap| per channel
            "shape_dc":       shape_dc_l,     # gated shape-error DC leak (should be ≈ 0)
            # ── NEW (V12): per-cell raw-error diagnostics ──
            "level_e_mean":   level_e_mean_l, # mean |e| over event cells (what V12 Level actually sees)
            "level_e_max":    level_e_max_l,  # max  |e| over event cells
            "level_e_sat":    level_e_sat_l,  # frac of event cells with |e|>3 (tanh saturated — gradient capped)
        }

        logger.debug(
            "SpotlightLossV12 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV12(non_zero_threshold={self.tau})"
