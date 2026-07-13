import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """2-term loss for zero-inflated conflict fatality forecasting.

    Shape = log_cosh with raw-error DRO (V13 — unchanged, global).
    Level = windowed MSE on per-window mean gaps (V36 — localizes spikes).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (AC pattern).** Demeaned log_cosh residual, gated,
      DRO-weighted on ``|raw_error|``, Hájek-normalised. EXACTLY V13.
      Shape operates on the FULL series (global demeaning) to learn
      long-term temporal patterns. Unchanged.

    * **Level (DC magnitude, windowed).** Splits the T-step horizon into
      K non-overlapping windows. For each window w:
        gap_w = mean(y_pred[w]) - mean(y_true[w])
        level_w = T_w × gap_w²
      Total Level = Σ_w level_w.

      V13 used a single global gap → uniform gradient on all T cells →
      couldn't localize corrections. A spike in month 5 was invisible
      if months 10-36 compensated.

      V36 uses K=4 windows (9 months each for T=36). Each window's gap
      is independent. A spike in window 2 inflates gap_2 → pushes ONLY
      window 2's cells down. Localization without per-cell AC.

      Orthogonality analysis:
        - Within each window: Level is DC (uniform gradient), Shape is
          AC (demeaned). Perfectly orthogonal within window.
        - Across windows: Level has low-frequency AC at K-1=3 boundaries.
          This is NOT the per-cell AC (35 boundaries) that caused V25's
          tug-of-war. The AC is block-level, much coarser than Shape's
          cell-level AC. Minimal interaction.

      Gradient comparison (gap=0.3, T=36, K=4, T_w=9):
        V13: 2×0.3 = 0.60 per cell (all 36 cells, uniform)
        V36: 2×gap_w per cell in window w (9 cells per window)
          If gap_w ≈ gap: 0.60 per cell (same as V13)
          If gap_2 = 0.6 (spike): 1.20 per cell in window 2 (2× stronger)
          If gap_1 = 0.0 (correct): 0.00 per cell in window 1 (no push)
        → Localized push where needed, no push where correct.

      Why K=4 (not adaptive):
        K=4 for T=36 gives 9-month windows. This is a structural choice
        (like T=36 itself), not a tunable hyperparameter. 9 months is
        long enough to capture conflict episodes, short enough to
        localize spikes. K must divide T evenly.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
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

        # Verify T is divisible by K (checked at runtime in forward)
        logger.info("SpotlightLossV36 | threshold=%.4f K=%d", non_zero_threshold, self._K)

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

        # ── LEVEL: windowed MSE on per-window mean gaps (V36) ────────
        # Split into K windows, compute per-window gap, sum T_w × gap_w².
        #
        # Gradient per cell in window w: 2 × gap_w (uniform within window)
        → DC within window, blocky AC across windows.
        # Compare V13: 2 × gap (uniform across ALL cells) → pure DC.
        #
        # If K=1 (fallback): identical to V13.
        if K > 1:
            if multivariate:
                C = y_pred.size(-1)
                # Reshape: (B, T, C) → (B, K, T_w, C)
                y_pred_win = y_pred.reshape(B, K, T_w, C)
                y_true_win = y_true.reshape(B, K, T_w, C)
                # Per-window means: (B, K, C)
                gap_w = y_pred_win.mean(dim=2) - y_true_win.mean(dim=2)
                # Per-window level: T_w × gap_w², sum across windows
                level_cell = T_w * (gap_w ** 2).sum(dim=1)  # (B, C)
            else:
                # Reshape: (B, T) → (B, K, T_w)
                y_pred_win = y_pred.reshape(B, K, T_w)
                y_true_win = y_true.reshape(B, K, T_w)
                # Per-window means: (B, K)
                gap_w = y_pred_win.mean(dim=2) - y_true_win.mean(dim=2)
                # Per-window level: T_w × gap_w², sum across windows
                level_cell = T_w * (gap_w ** 2).sum(dim=1)  # (B,)
        else:
            # Fallback to V13
            gap_w = None
            gap = y_pred.mean(dim=1) - y_true.mean(dim=1)
            level_cell = T * gap ** 2

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

                # V36: windowed gap diagnostics
                if K > 1 and gap_w is not None:
                    # Per-window gap stats
                    gap_w_abs = gap_w.abs()  # (B, K, C)
                    gap_w_mean_l = gap_w_abs.mean(dim=(0, 1)).tolist()  # mean across batch and windows
                    gap_w_max_l = gap_w_abs.amax(dim=(0, 1)).tolist()
                    # Gap variation across windows (AC indicator)
                    # If std across windows is high, windows have different gaps → localized
                    gap_w_std = gap_w_abs.std(dim=1)  # (B, C) — std across windows per series
                    gap_w_cv = (gap_w_std.mean(dim=0) / gap_w_abs.mean(dim=(0, 1)).clamp_min(1e-8)).tolist()
                    # Max window gap / global gap (localization factor)
                    # If >1, some window has worse gap than global → V36 catches what V13 missed
                    loc_factor_l = (gap_w_abs.amax(dim=1).mean(dim=0)
                                    / _ga.mean(dim=0).clamp_min(1e-8)).tolist()
                else:
                    gap_w_mean_l = gap_mean_l
                    gap_w_max_l = gap_max_l
                    gap_w_cv = [0.0] * len(gap_mean_l)
                    loc_factor_l = [1.0] * len(gap_mean_l)

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
                else:
                    gap_w_mean_l = gap_mean_l
                    gap_w_max_l = gap_max_l
                    gap_w_cv = [0.0]
                    loc_factor_l = [1.0]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV36: per_channel={comp}")

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
            # ── Gap diagnostics (global, for comparison with V13) ──
            "level_gap_mean": gap_mean_l,
            "level_gap_max":  gap_max_l,
            "level_gap_ev_mean": gap_ev_mean_l,
            "level_gap_ev_max":  gap_ev_max_l,
            "level_gap_sat":     gap_sat_l,
            "shape_dc":       shape_dc_l,
            "shape_level_ratio": sl_ratio_l,
            # ── V36: windowed gap diagnostics ──
            # Per-window gap stats. Compare gap_w_mean to gap_mean:
            # If gap_w_mean > gap_mean, some windows have worse gaps
            # than the global average → V36 catches what V13 missed.
            "gap_w_mean":     gap_w_mean_l,     # mean |gap| across all windows
            "gap_w_max":      gap_w_max_l,      # max |gap| in any window
            # Coefficient of variation of gaps across windows.
            # If >0.3, windows have significantly different gaps →
            # localization is doing work. If ≈0, all windows similar →
            # V36 ≈ V13 (no benefit from windowing).
            "gap_w_cv":       gap_w_cv,         # std/mean of per-window gaps
            # Localization factor: max window gap / global gap.
            # If >1.5, some window has 1.5× worse gap than global →
            # V13 was hiding that error. V36 catches it.
            "loc_factor":     loc_factor_l,     # max_window_gap / global_gap
        }

        logger.debug(
            "SpotlightLossV36 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV36(non_zero_threshold={self.tau})"
