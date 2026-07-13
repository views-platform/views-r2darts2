import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO, y_true-only gate (V33 — fixes spiking).
    Level = MSE on mean gap (V13 — unchanged).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated by
      ``y_true`` only (not ``y_pred``), DRO-weighted on ``|raw_error|``,
      Hájek-normalised.

      V13's gate used ``max(|y_true|, |y_pred|)``. When the model spiked
      at a peace cell (y_pred=5, y_true=0), the gate turned ON → Shape
      treated the spike as an "event" → DRO upweighted it → Shape wasted
      gradient on the spike instead of true events. The bounded log_cosh
      penalty (tanh ≤ 1) couldn't push the spike down hard enough.

      On the calibration set (sparse events), random spikes mostly landed
      on peace cells → overprediction. On validation (dense events),
      spikes accidentally aligned with real events → looked fine.

      V33 changes the gate to ``y_true`` only:
        - True event cells: gate ON (unchanged) → Shape learns patterns
        - Peace cells where model spikes: gate OFF → Shape ignores spike
        - Level unchanged: still sees all cells through the mean

      Why this stops spiking:
        - Spiking at peace cells no longer turns on Shape's attention
        - Level sees the inflated mean → pushes ALL cells down
        - The model can't "cheat" by spiking — must raise event cells
          specifically (Shape guides this)
        - On calibration (sparse events), no random spikes → no overpred
        - On validation (dense events), Shape guides to true events

      Why this is minimally invasive:
        - ONE LINE change from V13 (gate computation)
        - Level completely unchanged (V13's proven mean-gap MSE)
        - Shape's DRO, log_cosh, Hájek all unchanged
        - Only difference: gate no longer responds to y_pred spikes

    * **Level (DC magnitude).** ``T × gap²`` on per-series mean gap,
      gate-weighted, Hájek-normalised. EXACTLY V13 — unchanged.

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

        logger.info("SpotlightLossV33 | threshold=%.4f", non_zero_threshold)

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

        # ── Event gate — V33: y_true ONLY (not y_pred) ───────────────
        # V13 used max(|y_true|, |y_pred|) → model could turn on gate
        # by spiking at peace cells → Shape treated spikes as events.
        # V33 uses y_true only → gate is FIXED, not gameable by y_pred.
        # Shape focuses on TRUE events, ignores model artifacts.
        abs_max = y_true.abs()  # V33: y_true only, not max(y_true, y_pred)
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── SHAPE: log_cosh on demeaned errors, DRO on |raw_error| ───
        # Unchanged from V13 except gate source (y_true only).
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        raw_abs = e.abs().detach()
        event_mask = (abs_max > self.tau).float()  # V33: true events only
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
        # EXACTLY V13 — unchanged.
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * gap ** 2
        w_level = gate.amax(dim=1)  # per-series event mass (now y_true-only)

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
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V33: spike diagnostics ──
                # Compare y_true-only event_mask to V13's max-based mask
                # If there's a big difference, the model is spiking at peace cells
                _v13_mask = (torch.max(y_true.abs(), y_pred.detach().abs()) > self.tau).float()
                _spike_mask = (_v13_mask - event_mask).clamp_min(0)  # cells that V13 would gate but V33 doesn't
                _n_spike = _spike_mask.sum(dim=(0, 1)).clamp_min(1.0)
                spike_frac_l = (_spike_mask.sum(dim=(0, 1)) / _v13_mask.sum(dim=(0, 1)).clamp_min(1.0)).tolist()
                # Mean y_pred at spike cells (should be high — these are false alarms)
                _y_pred_spike = (y_pred.abs() * _spike_mask).sum(dim=(0, 1)) / _n_spike
                spike_mag_l = _y_pred_spike.tolist()
                # Mean y_true at spike cells (should be ~0 — these are peace cells)
                _y_true_spike = (y_true.abs() * _spike_mask).sum(dim=(0, 1)) / _n_spike
                spike_true_l = _y_true_spike.tolist()

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
                _v13_mask = (torch.max(y_true.abs(), y_pred.detach().abs()) > self.tau).float()
                _spike_mask = (_v13_mask - event_mask).clamp_min(0)
                _n_spike = _spike_mask.sum().clamp_min(1.0)
                spike_frac_l = [(_spike_mask.sum() / _v13_mask.sum().clamp_min(1.0)).item()]
                _y_pred_spike = (y_pred.abs() * _spike_mask).sum() / _n_spike
                spike_mag_l = [_y_pred_spike.item()]
                _y_true_spike = (y_true.abs() * _spike_mask).sum() / _n_spike
                spike_true_l = [_y_true_spike.item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV33: per_channel={comp}")

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
            # ── V33: spike diagnostics ──
            # These show cells that V13's gate would have turned ON but
            # V33's y_true-only gate leaves OFF. These are the "spikes at
            # peace cells" that V13 was treating as events.
            #
            # spike_frac: fraction of V13's "events" that are actually
            # model spikes at peace cells. If >10%, V13 was wasting
            # significant Shape gradient on false alarms.
            #
            # spike_mag: mean |y_pred| at spike cells. Should be > tau
            # (these are predictions above threshold at peace cells).
            #
            # spike_true: mean |y_true| at spike cells. Should be ~0
            # (confirming these are peace cells where the model spikes).
            "spike_frac":  spike_frac_l,   # frac of V13 events that are pred spikes at peace
            "spike_mag":   spike_mag_l,    # mean |y_pred| at spike cells (should be > tau)
            "spike_true":  spike_true_l,   # mean |y_true| at spike cells (should be ~0)
        }

        logger.debug(
            "SpotlightLossV33 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV33(non_zero_threshold={self.tau})"
