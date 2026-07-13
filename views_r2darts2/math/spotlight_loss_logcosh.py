import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """V40: Global Shape + Block Level + AsinhIntegral.

    Combines V36's localized block-level calibration with V38's bounded
    AsinhIntegral to prevent gradient explosions on volatile block gaps.

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** V13's global log_cosh, unchanged. This
      prevents the block-smearing exploit seen in V37/V38.

    * **Level (DC magnitude, windowed).** Splits the T-step horizon into
      K non-overlapping windows. For each window w:
        gap_w = mean(y_pred[w]) - mean(y_true[w])
        level_w = T_w × AsinhIntegral(gap_w)
      Total Level = Σ_w level_w.

      V36 used MSE (gap_w²) which exploded on volatile block gaps.
      V40 uses AsinhIntegral (gradient: asinh(gap_w)) which grows
      logarithmically — strong enough to calibrate, bounded enough to
      prevent explosions.

      Gradient comparison (gap_w=3.0, T_w=9):
        V36 (MSE): 2×3.0 = 6.0 per cell (explosive on volatile blocks)
        V40 (Asinh): asinh(3.0) = 1.82 per cell (bounded, stable)

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    ``K`` — structural (like T=36), not tunable. K=4 for 9-month blocks.
    """

    _EPS = 1e-6
    _K = 4  # Number of windows (structural, like T=36)

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None

        logger.info("SpotlightLossV40 | threshold=%.4f K=%d", non_zero_threshold, self._K)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _asinh_integral(x: torch.Tensor) -> torch.Tensor:
        """Integral of asinh(x): x * asinh(x) - sqrt(1 + x^2) + 1
        Gradient is asinh(x), which grows logarithmically.
        Convex, smooth, unbounded but stable.
        """
        return x * torch.asinh(x) - torch.sqrt(1.0 + x**2) + 1.0

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        B, T = y_pred.shape[:2]
        K = self._K
        T_w = T // K

        if T % K != 0:
            # Fallback to V13 if T doesn't divide evenly
            K = 1
            T_w = T

        self._last_input_grad = None
        if y_pred.requires_grad:
            y_pred.register_hook(lambda g: setattr(self, "_last_input_grad", g.detach()))

        e = y_pred - y_true

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── SHAPE: log_cosh on demeaned errors, DRO on |raw_error| ───
        # EXACTLY V13 — global, unchanged.
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

        # ── LEVEL: windowed AsinhIntegral on per-window mean gaps ────
        # V40: Replaces V36's MSE with AsinhIntegral to bound gradient explosions.
        # Gradient: asinh(gap_w), bounded logarithmically.
        if K > 1:
            if multivariate:
                C = y_pred.size(-1)
                # Reshape: (B, T, C) → (B, K, T_w, C)
                y_pred_win = y_pred.reshape(B, K, T_w, C)
                y_true_win = y_true.reshape(B, K, T_w, C)
                # Per-window means: (B, K, C)
                gap_w = y_pred_win.mean(dim=2) - y_true_win.mean(dim=2)
                # Per-window level: T_w × AsinhIntegral(gap_w), sum across windows
                level_cell = T_w * self._asinh_integral(gap_w).sum(dim=1)  # (B, C)
            else:
                # Reshape: (B, T) → (B, K, T_w)
                y_pred_win = y_pred.reshape(B, K, T_w)
                y_true_win = y_true.reshape(B, K, T_w)
                # Per-window means: (B, K)
                gap_w = y_pred_win.mean(dim=2) - y_true_win.mean(dim=2)
                # Per-window level: T_w × AsinhIntegral(gap_w), sum across windows
                level_cell = T_w * self._asinh_integral(gap_w).sum(dim=1)  # (B,)
        else:
            # Fallback to V13 with AsinhIntegral
            gap_w = None
            gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
            level_cell = T * self._asinh_integral(gap)

        w_level = gate.amax(dim=1)  # per-series event mass (same as V13)

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
            # Global gap (for comparison with V13)
            gap_global = y_pred.mean(dim=1) - y_true.mean(dim=1)

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

                _ga    = gap_global.abs()
                gap_mean_l    = _ga.mean(dim=0).tolist()
                gap_max_l     = _ga.amax(dim=0).tolist()
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum(dim=0).clamp_min(1.0)
                gap_ev_mean_l = ((_ga * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()
                gap_ev_max_l  = ((_ga * _ev_mask_s).amax(dim=0)).tolist()
                gap_sat_l = (((_ga > 1.5) * _ev_mask_s).sum(dim=0) / _n_ev_s).tolist()

                shape_dc_l    = (gate * e_shape).mean(dim=1).abs().mean(dim=0).tolist()

                # V40: windowed gap diagnostics
                if K > 1 and gap_w is not None:
                    gap_w_abs = gap_w.abs()  # (B, K, C)
                    gap_w_mean_l = gap_w_abs.mean(dim=(0, 1)).tolist()
                    gap_w_max_l = gap_w_abs.amax(dim=(0, 1)).tolist()
                    gap_w_std = gap_w_abs.std(dim=1)  # (B, C)
                    gap_w_cv = (gap_w_std.mean(dim=0) / gap_w_abs.mean(dim=(0, 1)).clamp_min(1e-8)).tolist()
                    loc_factor_l = (gap_w_abs.amax(dim=1).mean(dim=0)
                                    / _ga.mean(dim=0).clamp_min(1e-8)).tolist()
                    
                    # V40: AsinhIntegral gradient diagnostics
                    _asinh_grad = torch.asinh(gap_w).abs()
                    asinh_grad_mean_l = _asinh_grad.mean(dim=(0, 1)).tolist()
                    asinh_grad_max_l = _asinh_grad.amax(dim=(0, 1)).tolist()
                else:
                    gap_w_mean_l = gap_mean_l
                    gap_w_max_l = gap_max_l
                    gap_w_cv = [0.0] * len(gap_mean_l)
                    loc_factor_l = [1.0] * len(gap_mean_l)
                    _asinh_grad = torch.asinh(gap_global).abs()
                    asinh_grad_mean_l = _asinh_grad.mean(dim=0).tolist()
                    asinh_grad_max_l = _asinh_grad.amax(dim=0).tolist()

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
                _ga    = gap_global.abs()
                gap_mean_l    = [_ga.mean().item()]
                gap_max_l     = [_ga.max().item()]
                _ev_mask_s = (w_level > 0.5).float()
                _n_ev_s = _ev_mask_s.sum().clamp_min(1.0)
                gap_ev_mean_l = [((_ga * _ev_mask_s).sum() / _n_ev_s).item()]
                gap_ev_max_l  = [((_ga * _ev_mask_s).amax()).item()]
                gap_sat_l = [(((_ga > 1.5) * _ev_mask_s).sum() / _n_ev_s).item()]
                shape_dc_l    = [(gate * e_shape).mean(dim=1).abs().mean().item()]
                if K > 1 and gap_w is not None:
                    gap_w_abs = gap_w.abs()
                    gap_w_mean_l = [gap_w_abs.mean().item()]
                    gap_w_max_l = [gap_w_abs.max().item()]
                    gap_w_std = gap_w_abs.std(dim=1)
                    gap_w_cv = [(gap_w_std.mean() / max(1e-8, gap_w_abs.mean().item())).item()]
                    loc_factor_l = [(gap_w_abs.amax(dim=1).mean().item()
                                     / max(1e-8, _ga.mean().item())).item()]
                    _asinh_grad = torch.asinh(gap_w).abs()
                    asinh_grad_mean_l = [_asinh_grad.mean().item()]
                    asinh_grad_max_l = [_asinh_grad.max().item()]
                else:
                    gap_w_mean_l = gap_mean_l
                    gap_w_max_l = gap_max_l
                    gap_w_cv = [0.0]
                    loc_factor_l = [1.0]
                    _asinh_grad = torch.asinh(gap_global).abs()
                    asinh_grad_mean_l = [_asinh_grad.mean().item()]
                    asinh_grad_max_l = [_asinh_grad.max().item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV40: per_channel={comp}")

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
            # ── V40: windowed gap diagnostics ──
            "gap_w_mean":     gap_w_mean_l,
            "gap_w_max":      gap_w_max_l,
            "gap_w_cv":       gap_w_cv,
            "loc_factor":     loc_factor_l,
            # ── V40: AsinhIntegral gradient diagnostics ──
            "asinh_grad_mean": asinh_grad_mean_l,
            "asinh_grad_max":  asinh_grad_max_l,
        }

        logger.debug(
            "SpotlightLossV40 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV40(non_zero_threshold={self.tau})"
