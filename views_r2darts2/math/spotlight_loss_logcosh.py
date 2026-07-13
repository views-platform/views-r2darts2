import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = cosh (sinh gradient) + TV smoothness prior (V35).
    Level = MSE on mean gap (V13 — unchanged).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** ``cosh`` on demeaned residual, gated,
      DRO-weighted, Hájek-normalised, plus a Total Variation (TV)
      smoothness prior on adjacent predictions.

      V13 used ``log_cosh`` → gradient ``tanh`` saturates at ±1.0. The
      model could not push hard on large temporal errors (spikes,
      misplacements) → sporadic spikes persisted.

      V35 uses ``cosh`` → gradient ``sinh`` grows exponentially. For
      large errors, the gradient hits the global clip (20), providing
      an aggressive, capped-linear push. For small errors, cosh ≈ x²/2
      (MSE-like). This gives Shape the "teeth" to kill spikes without
      destabilizing normal training.

      V35 also adds a TV prior: ``log_cosh(y_pred[t] - y_pred[t-1])``.
      This penalizes bursty oscillations (e.g., 2000, 49, 2500) and
      encourages blocky, sustained predictions (e.g., 1500, 1500, 1500)
      which match real conflict dynamics. Integrated inside the Shape
      Hájek to maintain the 2-term structure.

    * **Level (DC magnitude).** ``T × gap²`` (V13 MSE — unchanged).
      Pure DC, unsaturated, provides strong calibration push.

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

        logger.info("SpotlightLossV35 | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _cosh(x: torch.Tensor) -> torch.Tensor:
        # Standard cosh. Gradient is sinh, which grows exponentially.
        # For numerical stability with large values, we rely on the
        # global gradient clipping (config: gradient_clip_val=20).
        return torch.cosh(x)

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

        # ── SHAPE: cosh + TV prior ───────────────────────────────────
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(self.tau)
        
        # V35: Use cosh instead of log_cosh for aggressive large-error push
        shape_cell = self._cosh(e_shape / ac_scale)

        # V35: Total Variation (TV) smoothness prior on predictions.
        # Penalizes adjacent differences to prevent bursty oscillations.
        # log_cosh is used here for a smooth, bounded penalty on diffs.
        y_pred_diffs = y_pred[:, 1:] - y_pred[:, :-1]
        tv_prior = self._log_cosh(y_pred_diffs / ac_scale)
        # Pad to match T dimension (first cell has no diff, penalty 0)
        if multivariate:
            tv_pad = torch.zeros(B, 1, y_pred.size(-1), device=y_pred.device, dtype=y_pred.dtype)
        else:
            tv_pad = torch.zeros(B, 1, device=y_pred.device, dtype=y_pred.dtype)
        tv_prior = torch.cat([tv_pad, tv_prior], dim=1)

        # Combine shape error and TV prior
        shape_combined = shape_cell + tv_prior

        # DRO weighting (unchanged)
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
            loss_shape = (shape_w * shape_combined).sum(dim=(0, 1)) / shape_w.sum(dim=(0, 1)).clamp_min(self._EPS)
        else:
            shape_w = gate * w_dro
            loss_shape = (shape_w * shape_combined).sum() / shape_w.sum().clamp_min(self._EPS)

        # ── LEVEL: T × MSE(mean gap) — V13, unchanged ────────────────
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

                # V35: TV diagnostic
                _tv_mean = tv_prior.mean(dim=(0, 1))
                tv_mean_l = _tv_mean.tolist()
                # V35: cosh gradient diagnostic
                _sinh_grad = torch.sinh(e_shape / ac_scale).abs()
                sinh_grad_mean_l = (_sinh_grad.mean(dim=(0, 1))).tolist()
                sinh_grad_max_l = (_sinh_grad.amax(dim=(0, 1))).tolist()

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
                _tv_mean = tv_prior.mean()
                tv_mean_l = [_tv_mean.item()]
                _sinh_grad = torch.sinh(e_shape / ac_scale).abs()
                sinh_grad_mean_l = [_sinh_grad.mean().item()]
                sinh_grad_max_l = [_sinh_grad.max().item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV35: per_channel={comp}")

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
            # ── V35: Shape diagnostics ──
            # The sinh gradient (from cosh loss). V13's tanh saturated at 1.0.
            # V35's sinh grows exponentially. If sinh_grad_max >> 1.0, Shape
            # is actively pushing hard on large errors (what we want).
            "sinh_grad_mean": sinh_grad_mean_l,
            "sinh_grad_max":  sinh_grad_max_l,
            # Total Variation penalty. Measures burstiness of predictions.
            # High TV = oscillating predictions (2000, 49, 2500).
            # Low TV = smooth/blocky predictions (1500, 1500, 1500).
            "tv_mean":        tv_mean_l,
        }

        logger.debug(
            "SpotlightLossV35 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV35(non_zero_threshold={self.tau})"
