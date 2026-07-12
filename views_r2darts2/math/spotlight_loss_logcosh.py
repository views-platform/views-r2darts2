import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = V13 mean-gap MSE + V27 peace guard (least invasive).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged.

    * **Level (DC magnitude + peace guard).** V13's ``T × gap²`` plus
      a peace-cell guard ``T × peace_mean²``.

      V13's gap is DC-only but gameable by spiking at peace cells —
      the model satisfies mean(y_pred) ≈ mean(y_true) with zeros+
      spikes. On training, spikes align with events. On eval, spikes
      misalign at peace cells → overprediction (V20 eval: sb 1.96×).

      V27 adds a peace guard:
        peace_mean = mean(y_pred at cells where y_true < tau)
        peace_cell = T × peace_mean²

      The peace mask depends ONLY on y_true (not y_pred) → fixed
      target, not gameable (fixes V23's flaw). The guard catches
      false alarms — a spike at a peace cell raises peace_mean →
      gradient pushes it DOWN (fixes V24's blindness).

      Least-invasive design choices:
        1. peace_cell gradient is ZERO at event cells → does NOT
           disturb V13's DC calibration at events. Only pushes peace
           cells down.
        2. peace_cell is inert when peace_mean ≈ 0 (no false alarms).
           Doesn't interfere when the model is behaving well.
        3. Combined with level_cell via simple ADDITION (not Hájek
           reweighting) — keeps level_cell's scale exactly as V13.
           The peace term is a small additive penalty, not a
           replacement. This preserves V13's Shape:Level balance.
        4. No new constants — uses existing tau for the peace mask.

      The peace guard is essentially a regularizer on peace-cell
      predictions, not a separate loss term. It nudges the model
      toward predicting low at peace cells without changing the
      magnitude calibration at event cells.

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

        logger.info("SpotlightLossV27 | threshold=%.4f", non_zero_threshold)

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

        # ── LEVEL: V13 mean-gap MSE + V27 peace guard ────────────────
        #
        # Part 1 (V13, unchanged): T × gap² on all-cell mean.
        # Pure DC, calibrates magnitude. Gradient = 2*gap at ALL cells.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * gap ** 2

        # Part 2 (V27 peace guard): T × peace_mean².
        # Peace mask from y_true ONLY (not gameable). Catches false
        # alarms — spikes at peace cells raise peace_mean → pushed DOWN.
        # Gradient is ZERO at event cells → doesn't disturb V13 calibration.
        # Inert when peace_mean ≈ 0 (no false alarms).
        gate_true = torch.sigmoid(10.0 * (y_true.abs() - self.tau))  # (B, T) or (B, T, C)
        mask_peace = 1.0 - gate_true
        n_peace = mask_peace.sum(dim=1).clamp_min(1e-6)
        peace_mean = (y_pred * mask_peace).sum(dim=1) / n_peace  # (B,) or (B, C)
        peace_cell = T * peace_mean ** 2

        # Hájek combination (same structure as V13, weight = event mass)
        w_level = gate.amax(dim=1)  # per-series event mass (same as V13)

        if multivariate:
            loss_level = (w_level * (level_cell + peace_cell)).sum(dim=0) / w_level.sum(dim=0).clamp_min(self._EPS)
        else:
            loss_level = (w_level * (level_cell + peace_cell)).sum() / w_level.sum().clamp_min(self._EPS)

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

                # ── V27 peace guard diagnostics ──
                _pm = peace_mean.abs()
                peace_mean_l  = _pm.mean(dim=0).tolist()
                peace_max_l   = _pm.amax(dim=0).tolist()
                # Peace cell fraction (how many cells are "peace")
                peace_frac_l  = mask_peace.mean(dim=(0, 1)).tolist()
                # Fraction of event series with significant false alarms (peace_mean > 0.1)
                _pm_ev = _pm * _ev_mask_s
                false_alarm_frac_l = (((_pm > 0.1).float() * _ev_mask_s).sum(dim=0)
                                      / _n_ev_s).tolist()
                # Mean y_pred at peace cells (what peace_cell is pushing down)
                _yp_peace = (y_pred.abs() * mask_peace).sum(dim=(0, 1)) / mask_peace.sum(dim=(0, 1)).clamp_min(1.0)
                y_pred_peace_l = _yp_peace.tolist()
                # Level sub-component magnitudes (for monitoring)
                level_cell_mean_l = ((level_cell * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                peace_cell_mean_l = ((peace_cell * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                # Ratio: how much of Level loss is the peace guard?
                # Should be small (<0.5) — if >1, peace guard is dominating
                peace_level_ratio_l = ((peace_cell * _ev_mask_s).sum(dim=0)
                                       / (level_cell * _ev_mask_s).sum(dim=0).clamp_min(1e-8)).tolist()

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
                _pm = peace_mean.abs()
                peace_mean_l  = [_pm.mean().item()]
                peace_max_l   = [_pm.max().item()]
                peace_frac_l  = [mask_peace.mean().item()]
                _pm_ev = _pm * _ev_mask_s
                false_alarm_frac_l = [(((_pm > 0.1).float() * _ev_mask_s).sum() / _n_ev_s).item()]
                _yp_peace = (y_pred.abs() * mask_peace).sum() / mask_peace.sum().clamp_min(1.0)
                y_pred_peace_l = [_yp_peace.item()]
                level_cell_mean_l = [((level_cell * _ev_mask_s).sum() / _n_ev_s).item()]
                peace_cell_mean_l = [((peace_cell * _ev_mask_s).sum() / _n_ev_s).item()]
                peace_level_ratio_l = [((peace_cell * _ev_mask_s).sum()
                                        / max(1e-8, (level_cell * _ev_mask_s).sum())).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV27: per_channel={comp}")

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
            # ── V27 peace guard diagnostics ──
            # The mean y_pred at peace cells. This is what the peace guard
            # is pushing toward 0. Should DECREASE over training.
            # If high at start and decreases, the guard is working.
            "peace_mean":           peace_mean_l,        # mean |peace_mean| per channel
            "peace_max":            peace_max_l,         # max |peace_mean| (worst false alarmer)
            "peace_frac":           peace_frac_l,        # fraction of cells that are peace
            # Fraction of event series with significant false alarms.
            # If >50%, most series are spiking at peace cells.
            "false_alarm_frac":     false_alarm_frac_l,  # frac event series with peace_mean > 0.1
            # Mean |y_pred| at peace cells — direct measure of false alarm magnitude.
            "y_pred_peace":         y_pred_peace_l,      # mean |y_pred| at peace cells
            # Level sub-component magnitudes (for monitoring balance).
            # level_cell = V13's gap² term. peace_cell = V27's guard.
            # If peace_cell_mean >> level_cell_mean, the guard is dominating
            # (may be too strong). If <<, it's a minor regularizer (ideal).
            "level_cell_mean":      level_cell_mean_l,   # V13's gap² term (calibration)
            "peace_cell_mean":      peace_cell_mean_l,   # V27's peace guard
            "peace_level_ratio":    peace_level_ratio_l, # peace/level ratio (should be < 0.5)
        }

        logger.debug(
            "SpotlightLossV27 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV27(non_zero_threshold={self.tau})"
