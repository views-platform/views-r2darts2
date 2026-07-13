import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """V48: V13 + series-level gate.

    The minimal, correct fix: V13 exactly, but with a series-level gate
    that activates the ENTIRE series if ANY cell has activity.

    ── Design ─────────────────────────────────────────────────────────

    * **Series-level gate.** If ANY cell in the series has y_true or
      y_pred > tau, the gate turns ON for ALL 36 cells. Otherwise, it's
      OFF for all 36 cells.

      V13's per-cell gate fragmented learning:
        - Peaceful country, small pred (0.4): gate≈0.008 → trained on noise
        - Event country, peace cell: gate≈0.008 → Shape couldn't see contrast

      V48's series-level gate:
        - Peaceful country, small pred: gate=0 → no training (no noise)
        - Peaceful country, spike (5.0): gate=1 → Shape penalizes hallucination
        - Event country, any cell: gate=1 → Shape sees full pattern
        - Event country, event cell: gate=1 → same as V13

    * **Shape (AC pattern).** V13 EXACTLY — batch ac_scale, global
      demeaning, log_cosh, DRO (max(y_true,y_pred) event_mask), Hájek.

    * **Level (DC magnitude).** V13 EXACTLY — T × gap² MSE, gate.amax
      (now = series gate value, 0 or ~1), Hájek.

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

        logger.info("SpotlightLossV48 | threshold=%.4f", non_zero_threshold)

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

        # ── Series-level gate (V48) ──────────────────────────────────
        # max(y_true, y_pred) per cell, then max over all T cells.
        # If ANY cell has activity, gate turns ON for the ENTIRE series.
        # Otherwise, gate is OFF → no noise training on peaceful series.
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())  # (B, T) or (B, T, C)
        if multivariate:
            series_max = abs_max.amax(dim=1, keepdim=True)  # (B, 1, C)
            gate = torch.sigmoid(10.0 * (series_max - self.tau))  # (B, 1, C)
            gate = gate.expand(B, T, -1)  # broadcast to all cells
        else:
            series_max = abs_max.amax(dim=1, keepdim=True)  # (B, 1)
            gate = torch.sigmoid(10.0 * (series_max - self.tau))  # (B, 1)
            gate = gate.expand(B, T)  # broadcast to all cells

        # ── Event mask (V13 — max(y_true, y_pred) > tau) ────────────
        # Same as V13. NOT y_true-only. The DRO gaming "exploit" was
        # mathematically incorrect — sqrt(x/mu) naturally upweights
        # maximum errors, so hallucinating makes the loss WORSE, not
        # better. V13's original event_mask is correct.
        event_mask = (abs_max > self.tau).float()

        # ── SHAPE: log_cosh on demeaned errors, DRO on |raw_error| ───
        # V13 EXACTLY — batch ac_scale, global demeaning.
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        shape_cell = self._log_cosh(e_shape / ac_scale)

        raw_abs = e.abs().detach()
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
        # V13 EXACTLY. w_level = gate.amax = series gate value (0 or ~1).
        gap = y_pred.mean(dim=1) - y_true.mean(dim=1)  # (B,) or (B, C)
        level_cell = T * gap ** 2
        w_level = gate.amax(dim=1)  # series gate value (0 for inactive, ~1 for active)

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

                # V48: series-level gate diagnostics
                _series_gate = gate[:, 0, :]  # (B, C)
                series_gate_mean_l = _series_gate.mean(dim=0).tolist()
                series_active_frac_l = ((_series_gate > 0.5).float().sum(dim=0)
                                        / _series_gate.size(0)).tolist()

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
                _series_gate = gate[:, 0]
                series_gate_mean_l = [_series_gate.mean().item()]
                series_active_frac_l = [((_series_gate > 0.5).float().sum()
                                         / _series_gate.size(0)).item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV48: per_channel={comp}")

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
            # ── V48: series-level gate diagnostics ──
            # Mean gate value. Should be ~0.1-0.3 (only active series).
            # V13's per-cell gate was ~0.01-0.05 (fragmented).
            "series_gate_mean":     series_gate_mean_l,
            # Fraction of series that are active (gate > 0.5).
            # Should be ~10-20% (event fraction). If >50%, model is
            # hallucinating widely. If <5%, threshold too high.
            "series_active_frac":   series_active_frac_l,
        }

        logger.debug(
            "SpotlightLossV48 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV48(non_zero_threshold={self.tau})"
