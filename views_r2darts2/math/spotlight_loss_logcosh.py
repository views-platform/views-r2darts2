import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × per-series Hájek of per-cell log_cosh(e) (V21b — fixes V21).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged.

    * **Level (DC + AC magnitude).** ``T × per-series Hájek(log_cosh(e))``
      — per-cell log_cosh on raw error, normalized per-series first,
      then Hájek-normalized over series, T-scaled.

      V13 used ``T × gap²`` (mean-gap MSE). The mean is deceptive —
      it hides the true distribution. The model satisfies it with
      zeros+spikes. On training, spikes align with events. On eval,
      spikes land on wrong months → overprediction (V20 eval: sb 1.96×
      vs training 0.66×).

      V21 used ``T × per-cell Hájek(log_cosh(e))`` — normalizes over
      (B,T) = 6400 event cells. This dilutes the gradient 64× more
      than V13 (which normalizes over (B) = 100 series). DC gradient
      was 5× weaker than V13 → plateaued at 0.40× underprediction.

      V21b fixes this with **per-series Hájek**: normalize over T
      first (per-series mean of log_cosh(e)), then Hájek over B.
      The denominator is ~100 (event series) like V13, not ~6400
      (event cells) like V21.

      Gradient at event cell:
        V13:   2*gap / N_series = 0.4/100 = 0.004 (all 36 cells)
        V21:   T*tanh(e) / N_cells = 36/6400 = 0.0056 (5 cells)
        V21b:  T*w*tanh(e) / (N_ev_per_series * N_series) = 36/(5*100) = 0.072 (5 cells)

      Total DC push per series:
        V13:   36 * 0.004 = 0.144
        V21:   5 * 0.0056 = 0.028 (5× too weak → underprediction)
        V21b:  5 * 0.072 = 0.36 (2.5× V13 → strong calibration)

      DC component of gradient (what drives calibration):
        V13:   0.004 (uniform)
        V21:   5*0.0056/36 = 0.00078 (3× weaker than V13)
        V21b:  5*0.072/36 = 0.01 (2.5× stronger than V13)

      V21b gives both strong DC (calibration, 2.5× V13) AND strong AC
      at event cells (spike catching, 18× V13 per cell).

      Loss value unchanged (~T * 1.0 = 36) → Shape:Level balance
      preserved. log_cosh bounds gradient at tanh ≤ 1 → no blow-up
      on 10000-fatality spikes.

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

        logger.info("SpotlightLossV21b | threshold=%.4f", non_zero_threshold)

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

        # ── LEVEL: T × per-series Hájek of per-cell log_cosh(e) ──────
        # V21b: fixes V21's weak DC gradient.
        #
        # V21 normalized over (B,T) = 6400 event cells → gradient
        # diluted 64× more than V13 → DC 5× weaker → underprediction.
        #
        # V21b normalizes per-series first (sum_T / sum_T), then
        # Hájek over B (like V13). Denominator is ~100 event series,
        # not ~6400 event cells.
        #
        # Per-series mean: (gate * log_cosh(e)).sum(dim=1) / gate.sum(dim=1)
        # This gives each series' mean log_cosh error ≈ 1.0.
        # Then Hájek over series with w_level = gate.amax(dim=1).
        # Then T scales to ≈ 36.
        level_cell = self._log_cosh(e)  # raw error, no demeaning, no /ac_scale
        w_level = gate  # same gate as Shape

        if multivariate:
            # Per-series mean: (B, T, C) → sum over T → (B, C)
            per_series_level = (w_level * level_cell).sum(dim=1) / w_level.sum(dim=1).clamp_min(self._EPS)
            # Hájek over series with event-mass weight
            w_level_series = gate.amax(dim=1)  # (B, C) — per-series event mass
            loss_level = T * (w_level_series * per_series_level).sum(dim=0) / w_level_series.sum(dim=0).clamp_min(self._EPS)
        else:
            # Per-series mean: (B, T) → sum over T → (B,)
            per_series_level = (w_level * level_cell).sum(dim=1) / w_level.sum(dim=1).clamp_min(self._EPS)
            # Hájek over series
            w_level_series = gate.amax(dim=1)  # (B,)
            loss_level = T * (w_level_series * per_series_level).sum() / w_level_series.sum().clamp_min(self._EPS)

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

                # Gap diagnostics (what V13's mean-gap would see — for comparison)
                gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga    = gap.abs()
                gap_mean_l    = _ga.mean(dim=0).tolist()
                gap_max_l     = _ga.amax(dim=0).tolist()
                _ev_mask_s = (w_level_series > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V21b: per-cell Level diagnostics ──
                _e_ev = e.abs() * event_mask  # zero out peace cells
                _e_ev_mean = _e_ev.sum(dim=(0, 1)) / _n_ev  # (C,)
                e_ev_mean_l = _e_ev_mean.tolist()
                e_ev_max_l = (_e_ev.amax(dim=(0, 1))).tolist()
                _lc_ev = level_cell * event_mask
                lc_ev_mean_l = (_lc_ev.sum(dim=(0, 1)) / _n_ev).tolist()
                lc_ev_max_l = (_lc_ev.amax(dim=(0, 1))).tolist()

                # Effective per-cell gradient for V21b:
                # T * w_level_series * gate * tanh(e) / (sum_T(gate) * sum_B(w_level_series))
                # ≈ T * tanh(e) / (N_ev_per_series * N_series)
                _sum_gate_per_series = w_level.sum(dim=1).clamp_min(1.0)  # (B, C)
                _sum_w_series = w_level_series.sum(dim=0).clamp_min(1.0)  # (C,)
                _eff_grad = T * w_level_series.unsqueeze(1) * w_level * torch.tanh(e) \
                            / (_sum_gate_per_series.unsqueeze(1) * _sum_w_series.unsqueeze(0).unsqueeze(0))
                # _eff_grad shape: (B, T, C) — but need to be careful with broadcasting
                # Actually w_level_series is (B, C), unsqueeze(1) → (B, 1, C)
                # w_level is (B, T, C)
                # _sum_gate_per_series is (B, C), unsqueeze(1) → (B, 1, C)
                # _sum_w_series is (C,), unsqueeze(0).unsqueeze(0) → (1, 1, C)
                _eff_grad_ev = _eff_grad.abs() * event_mask
                eff_grad_mean_l = (_eff_grad_ev.sum(dim=(0, 1)) / _n_ev).tolist()
                eff_grad_max_l = (_eff_grad_ev.amax(dim=(0, 1))).tolist()

                # Spike diagnostics
                spike_frac_l = (((e.abs() > 2.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                spike_severe_l = (((e.abs() > 4.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                overpred_frac_l = (((e > 0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                _overpred = (e * (e > 0).float() * event_mask).sum(dim=(0, 1))
                _n_overpred = ((e > 0).float() * event_mask).sum(dim=(0, 1)).clamp_min(1.0)
                overpred_mag_l = (_overpred / _n_overpred).tolist()
                underpred_frac_l = (((e < 0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()

                # Event cells per series (affects gradient strength)
                _ev_per_series = event_mask.sum(dim=1)  # (B, C)
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()  # (B, C)
                _n_es = _ev_series_mask.sum(dim=0).clamp_min(1.0)  # (C,)
                _eps_mean_t = (_ev_per_series * _ev_series_mask).sum(dim=0) / _n_es
                ev_per_series_l = _eps_mean_t.tolist()

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
                gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga    = gap.abs()
                gap_mean_l    = [_ga.mean().item()]
                gap_max_l     = [_ga.max().item()]
                _ev_mask_s = (w_level_series > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l  = [((_ga * _ev_mask_s).amax()).item()]
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                _e_ev = e.abs() * event_mask
                _e_ev_mean = _e_ev.sum() / _n_ev
                e_ev_mean_l = [_e_ev_mean.item()]
                e_ev_max_l = [_e_ev.max().item()]
                _lc_ev = level_cell * event_mask
                lc_ev_mean_l = [(_lc_ev.sum() / _n_ev).item()]
                lc_ev_max_l = [_lc_ev.max().item()]
                _sum_gate_per_series = w_level.sum(dim=1).clamp_min(1.0)  # (B,)
                _sum_w_series = w_level_series.sum().clamp_min(1.0)  # scalar
                _eff_grad = T * w_level_series.unsqueeze(1) * w_level * torch.tanh(e) \
                            / (_sum_gate_per_series.unsqueeze(1) * _sum_w_series)
                _eff_grad_ev = _eff_grad.abs() * event_mask
                eff_grad_mean_l = [(_eff_grad_ev.sum() / _n_ev).item()]
                eff_grad_max_l = [_eff_grad_ev.max().item()]
                spike_frac_l = [(((e.abs() > 2.0) * event_mask).sum() / _n_ev).item()]
                spike_severe_l = [(((e.abs() > 4.0) * event_mask).sum() / _n_ev).item()]
                overpred_frac_l = [(((e > 0) * event_mask).sum() / _n_ev).item()]
                _overpred = (e * (e > 0).float() * event_mask).sum()
                _n_overpred = ((e > 0).float() * event_mask).sum().clamp_min(1.0)
                overpred_mag_l = [(_overpred / _n_overpred).item()]
                underpred_frac_l = [(((e < 0) * event_mask).sum() / _n_ev).item()]
                _ev_per_series = event_mask.sum(dim=1)
                _ev_series_mask = (event_mask.sum(dim=1) > 0).float()
                _n_es = _ev_series_mask.sum().clamp_min(1.0)
                ev_per_series_l = [((_ev_per_series * _ev_series_mask).sum() / _n_es).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV21b: per_channel={comp}")

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
            # ── Gap diagnostics (what V13's mean-gap would see) ──
            "level_gap_mean": gap_mean_l,
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V21b: per-cell Level diagnostics ──
            # Compare e_ev_mean to gap_ev_mean:
            #   gap_ev_mean ≈ 0.2 (V13's diluted signal)
            #   e_ev_mean ≈ 1.5 (V21b's true per-cell error)
            # If e_ev_mean >> gap_ev_mean, the mean was hiding error.
            "e_ev_mean":        e_ev_mean_l,
            "e_ev_max":         e_ev_max_l,
            "lc_ev_mean":       lc_ev_mean_l,
            "lc_ev_max":        lc_ev_max_l,
            # Effective per-cell gradient (V21b):
            # T * tanh(e) / (N_ev_per_series * N_series)
            # Should be ≈ 0.072 at event cells (vs V13's 0.004, V21's 0.0056)
            "eff_grad_mean":    eff_grad_mean_l,
            "eff_grad_max":     eff_grad_max_l,
            # ── Spike diagnostics ──
            "spike_frac":       spike_frac_l,
            "spike_severe":     spike_severe_l,
            "overpred_frac":    overpred_frac_l,
            "overpred_mag":     overpred_mag_l,
            "underpred_frac":   underpred_frac_l,
            "ev_per_series":    ev_per_series_l,
        }

        logger.debug(
            "SpotlightLossV21b | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV21b(non_zero_threshold={self.tau})"
