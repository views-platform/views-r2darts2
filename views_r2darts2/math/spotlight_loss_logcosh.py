import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged).
    Level = T × Hájek(log_cosh(e)) per-cell (V21 — catches spikes, bounded).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. Unchanged from
      V13 — this component works.

    * **Level (DC + AC magnitude).** ``T × Hájek(gate, log_cosh(e))`` —
      per-cell log_cosh on raw error, gate-weighted, Hájek-normalised,
      T-scaled.

      V13 used ``T × gap²`` (mean-gap MSE). The mean is deceptive —
      it hides the true distribution. A series with one spike of 5.0
      and 35 zeros has gap = 5/36 = 0.14. V13 sees gap=0.14 and can't
      tell the model "don't spike here." The model learns to satisfy
      mean(y_pred) ≈ mean(y_true) with zeros+spikes. On training,
      spikes align with true events. On eval, spikes land on wrong
      months → mean inflates → overprediction (V20 eval: sb 1.96×
      vs training 0.66×).

      V12 used per-cell log_cosh(e) without T. Hájek normalizes to
      ≈1.0 (same as Shape) → Shape:Level = 1:1 → total Level gradient
      3× weaker than V13 → underprediction.

      V21 uses ``T × Hájek(gate, log_cosh(e))``:
        - Per-cell: sees the true distribution, catches spikes directly.
          A spike at a low-magnitude event (e=4.5) gets gradient
          T*tanh(4.5)/n_ev ≈ 7.2 at that cell — directly says "don't
          spike here."
        - log_cosh: gradient = tanh(e), bounded at 1.0. A 10000-fatality
          spike (e≈9.8 in asinh) gets the same gradient as a 5-error
          — doesn't blow up like MSE would (2*9.8 = 19.6).
        - T outside Hájek: scales Level to ≈36 (comparable to V13's
          ≈55). Without T, Hájek normalizes to ≈1.0 = Shape loss →
          not enough signal (V12 lesson). T doesn't get normalized
          away because it's outside the Hájek.
        - No /ac_scale: V14 proved T*log_cosh(x/ac_scale) causes
          explosions (gradient bounds at T/ac_scale ≈ 41/cell).
          Without /ac_scale, bounds at T/sum(gate) ≈ 7/cell — safe.

      Gradient at event cells (n_ev=5, T=36):
        e=0.5:  T*tanh(0.5)/5 = 36*0.46/5 = 3.3  (small error, gentle push)
        e=1.5:  T*tanh(1.5)/5 = 36*0.91/5 = 6.5  (normal error, strong push)
        e=5.0:  T*tanh(5.0)/5 = 36*1.00/5 = 7.2  (large error, max push)
        e=9.8:  T*tanh(9.8)/5 = 36*1.00/5 = 7.2  (10000 spike, BOUNDED)

      Compare V13 mean-gap MSE (gap≈0.2):
        gradient = 2*gap = 0.4 per cell (all 36 cells)
        total = 36*0.4 = 14.4

      V21 total (n_ev=5): 5*6.5 = 32.4 — 2.2× stronger than V13.
      V21 per event cell: 6.5 — 16× stronger than V13's 0.4.

      Not orthogonal to Shape (both push toward y_true at event cells
      → reinforcement, not conflict). User accepts this: "i know its
      not orthogonal but mean is very deceptive."

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

        logger.info("SpotlightLossV21 | threshold=%.4f", non_zero_threshold)

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

        # ── LEVEL: T × Hájek(gate, log_cosh(e)) — per-cell, V21 ──────
        # Replaces V13's T × gap² (mean-gap MSE).
        #
        # Per-cell log_cosh on raw error:
        #   - Sees the true distribution (mean hides spikes)
        #   - log_cosh bounds gradient at tanh ≤ 1 (no blow-up on 10000
        #     fatality spikes, unlike MSE's 2*e)
        #   - T outside Hájek provides enough signal (V12 without T was
        #     3× weaker than V13)
        #   - No /ac_scale (V14 proved T*log_cosh(x/ac_scale) explodes)
        #
        # Gradient at event cell: T * gate * tanh(e) / sum(gate)
        #   - Bounded at T/sum(gate) ≈ 7 per cell (safe)
        #   - 16× stronger than V13's 0.4 at event cells (targeted)
        #   - 2.2× stronger total than V13 (enough signal)
        level_cell = self._log_cosh(e)  # raw error, no demeaning, no /ac_scale
        w_level = gate  # same gate as Shape, no DRO

        if multivariate:
            loss_level = T * (w_level * level_cell).sum(dim=(0, 1)) / w_level.sum(dim=(0, 1)).clamp_min(self._EPS)
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

                # Gap diagnostics (for comparison with V13 — what mean-gap would see)
                gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
                _ga    = gap.abs()
                gap_mean_l    = _ga.mean(dim=0).tolist()
                gap_max_l     = _ga.amax(dim=0).tolist()
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # ── V21: per-cell Level diagnostics ──
                # Per-cell |e| at event cells (what Level actually sees,
                # vs mean-gap which only sees the diluted average)
                _e_ev = e.abs() * event_mask  # zero out peace cells
                _e_ev_mean = _e_ev.sum(dim=(0, 1)) / _n_ev  # (C,)
                e_ev_mean_l = _e_ev_mean.tolist()
                e_ev_max_l = (_e_ev.amax(dim=(0, 1))).tolist()
                # Per-cell log_cosh(e) at event cells (the Level loss per cell)
                _lc_ev = level_cell * event_mask
                lc_ev_mean_l = (_lc_ev.sum(dim=(0, 1)) / _n_ev).tolist()
                lc_ev_max_l = (_lc_ev.amax(dim=(0, 1))).tolist()
                # Effective per-cell gradient: T * gate * tanh(e) / sum(gate)
                # This is what the model actually sees at each event cell.
                _sum_gate = gate.sum(dim=(0, 1)).clamp_min(1.0)  # (C,)
                _eff_grad = T * gate * torch.tanh(e) / _sum_gate
                _eff_grad_ev = _eff_grad.abs() * event_mask
                eff_grad_mean_l = (_eff_grad_ev.sum(dim=(0, 1)) / _n_ev).tolist()
                eff_grad_max_l = (_eff_grad_ev.amax(dim=(0, 1))).tolist()
                # Fraction of event cells with |e| > 2 (potential spikes)
                spike_frac_l = (((e.abs() > 2.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                # Fraction with |e| > 4 (definite spikes — should decrease over training)
                spike_severe_l = (((e.abs() > 4.0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                # Overprediction at event cells (e > 0 = y_pred > y_true)
                overpred_frac_l = (((e > 0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                # Mean overprediction magnitude at overpredicted event cells
                _overpred = (e * (e > 0).float() * event_mask).sum(dim=(0, 1))
                _n_overpred = ((e > 0).float() * event_mask).sum(dim=(0, 1)).clamp_min(1.0)
                overpred_mag_l = (_overpred / _n_overpred).tolist()
                # Underprediction at event cells (e < 0 = y_pred < y_true)
                underpred_frac_l = (((e < 0) * event_mask).sum(dim=(0, 1)) / _n_ev).tolist()
                # Event cells per series (affects gradient strength: T/sum(gate))
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
                _ev_mask_s = (gate.amax(dim=1) > 0.5).float()
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
                _sum_gate = gate.sum().clamp_min(1.0)
                _eff_grad = T * gate * torch.tanh(e) / _sum_gate
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
            raise RuntimeError(f"NaN in SpotlightLossV21: per_channel={comp}")

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
            # ── Gap diagnostics (what V13's mean-gap would see — for comparison) ──
            "level_gap_mean": gap_mean_l,         # mean |gap| — V13's signal (diluted)
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V21: per-cell Level diagnostics (what V21 actually sees) ──
            # Compare e_ev_mean to gap_ev_mean:
            #   gap_ev_mean ≈ 0.2 (diluted by 35 peace cells in the mean)
            #   e_ev_mean ≈ 1.5 (true per-cell error at event cells)
            # If e_ev_mean >> gap_ev_mean, the mean was hiding the true error.
            "e_ev_mean":        e_ev_mean_l,      # mean |e| at event cells (true error)
            "e_ev_max":         e_ev_max_l,       # max |e| at event cells (largest spike)
            "lc_ev_mean":       lc_ev_mean_l,     # mean log_cosh(e) at event cells (Level loss per cell)
            "lc_ev_max":        lc_ev_max_l,      # max log_cosh(e) at event cells
            # Effective per-cell gradient (T * gate * tanh(e) / sum(gate))
            # This is what the model actually sees. Compare to V13's 2*gap ≈ 0.4.
            "eff_grad_mean":    eff_grad_mean_l,  # mean effective gradient at event cells
            "eff_grad_max":     eff_grad_max_l,   # max effective gradient (should be ≤ T/sum(gate) ≈ 7)
            # ── V21: spike diagnostics ──
            # These show whether the model is spiking. Should decrease over training.
            "spike_frac":       spike_frac_l,     # frac of event cells with |e|>2
            "spike_severe":     spike_severe_l,   # frac with |e|>4 (key metric — should decrease)
            "overpred_frac":    overpred_frac_l,  # frac of event cells where y_pred > y_true
                                                   # If >0.5, model is overpredicting at event cells
            "overpred_mag":     overpred_mag_l,   # mean overprediction magnitude at overpredicted cells
            "underpred_frac":   underpred_frac_l,  # frac where y_pred < y_true
            # Event cells per series (affects gradient strength)
            "ev_per_series":    ev_per_series_l,  # mean event cells per event series
                                                   # Gradient bounds at T/sum(gate) ≈ T/ev_per_series
                                                   # If low (rare events), gradient is stronger per cell
        }

        logger.debug(
            "SpotlightLossV21 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV21(non_zero_threshold={self.tau})"
