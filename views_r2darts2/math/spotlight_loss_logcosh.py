import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """V39: Global Shape + Block Level + AsinhIntegral.

    Shape = log_cosh on GLOBAL-demeaned errors (V13 — sees spikes).
    Level = AsinhIntegral on block-mean gaps (V36 — localizes, bounded).

    ── Design ─────────────────────────────────────────────────────────

    V37/V38 used block-demeaned Shape → exact orthogonality → but
    block smearing (model predicts flat block means → both losses = 0).

    V39 uses GLOBAL-demeaned Shape (V13) + block-mean Level (V36) +
    AsinhIntegral (V38's bounded gradient). This:
    - Prevents block smearing (Shape sees spikes vs global mean)
    - Localizes Level (block gaps catch obfuscation)
    - Bounds gradient (AsinhIntegral prevents explosions)
    - Accepts mild AC leakage (much better than block smearing collapse)

    The mild orthogonality break is intentional and necessary:
    - Exact orthogonality (V37/V38) → zero-loss flat block means → MCR=0.18
    - Mild leakage (V39) → Shape forces spike learning → MCR should recover

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    ``K`` — structural (like T=36), not tunable. K=4 for 9-month blocks.
    """

    _EPS = 1e-6
    _K = 4  # Number of blocks (structural, like T=36)

    def __init__(self, non_zero_threshold: float = 0.88):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.tau = non_zero_threshold
        self._last_components: dict | None = None
        self._last_input_grad: torch.Tensor | None = None

        logger.info("SpotlightLossV39 | threshold=%.4f K=%d", non_zero_threshold, self._K)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        a = x.abs()
        return a + F.softplus(-2.0 * a) - math.log(2.0)

    @staticmethod
    def _asinh_integral(x: torch.Tensor) -> torch.Tensor:
        """Integral of asinh(x): x * asinh(x) - sqrt(1 + x^2) + 1
        Gradient is asinh(x), which grows logarithmically.
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
            K = 1
            T_w = T

        self._last_input_grad = None
        if y_pred.requires_grad:
            y_pred.register_hook(lambda g: setattr(self, "_last_input_grad", g.detach()))

        e = y_pred - y_true

        # ── Event gate ───────────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - self.tau))

        # ── SHAPE: log_cosh on GLOBAL-demeaned errors (V13) ──────────
        # CRITICAL: Use global demeaning, NOT block demeaning.
        # Block demeaning → block smearing (zero-loss flat block means).
        # Global demeaning → Shape sees spikes vs 36-month mean → forces
        # the model to learn spike locations.
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

        # ── LEVEL: AsinhIntegral on block-mean gaps ──────────────────
        # Block means localize the Level calibration. AsinhIntegral bounds
        # the gradient to prevent explosions on volatile block gaps.
        if K > 1:
            if multivariate:
                C = y_pred.size(-1)
                y_pred_blocks = y_pred.reshape(B, K, T_w, C)
                y_true_blocks = y_true.reshape(B, K, T_w, C)
                gap_blocks = y_pred_blocks.mean(dim=2) - y_true_blocks.mean(dim=2)  # (B, K, C)
            else:
                y_pred_blocks = y_pred.reshape(B, K, T_w)
                y_true_blocks = y_true.reshape(B, K, T_w)
                gap_blocks = y_pred_blocks.mean(dim=2) - y_true_blocks.mean(dim=2)  # (B, K)
        else:
            if multivariate:
                gap_blocks = (y_pred.mean(dim=1) - y_true.mean(dim=1)).unsqueeze(1)  # (B, 1, C)
            else:
                gap_blocks = (y_pred.mean(dim=1) - y_true.mean(dim=1)).unsqueeze(1)  # (B, 1)

        # AsinhIntegral: bounded gradient (asinh), prevents explosions
        if multivariate:
            level_cell = T_w * self._asinh_integral(gap_blocks).sum(dim=1)  # (B, C)
        else:
            level_cell = T_w * self._asinh_integral(gap_blocks).sum(dim=1)  # (B,)

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

                if K > 1:
                    gap_blocks_abs = gap_blocks.abs()
                    gap_w_mean_l = gap_blocks_abs.mean(dim=(0, 1)).tolist()
                    gap_w_max_l = gap_blocks_abs.amax(dim=(0, 1)).tolist()
                    loc_factor_l = (gap_blocks_abs.amax(dim=1).mean(dim=0)
                                    / _ga.mean(dim=0).clamp_min(1e-8)).tolist()
                    _asinh_grad = torch.asinh(gap_blocks).abs()
                    asinh_grad_mean_l = _asinh_grad.mean(dim=(0, 1)).tolist()
                    asinh_grad_max_l = _asinh_grad.amax(dim=(0, 1)).tolist()
                else:
                    gap_w_mean_l = gap_mean_l
                    gap_w_max_l = gap_max_l
                    loc_factor_l = [1.0] * len(gap_mean_l)
                    _asinh_grad = torch.asinh(gap_blocks).abs()
                    asinh_grad_mean_l = _asinh_grad.mean(dim=(0, 1)).tolist()
                    asinh_grad_max_l = _asinh_grad.amax(dim=(0, 1)).tolist()

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
                if K > 1:
                    gap_blocks_abs = gap_blocks.abs()
                    gap_w_mean_l = [gap_blocks_abs.mean().item()]
                    gap_w_max_l = [gap_blocks_abs.max().item()]
                    loc_factor_l = [(gap_blocks_abs.amax(dim=1).mean().item()
                                     / max(1e-8, _ga.mean().item())).item()]
                    _asinh_grad = torch.asinh(gap_blocks).abs()
                    asinh_grad_mean_l = [_asinh_grad.mean().item()]
                    asinh_grad_max_l = [_asinh_grad.max().item()]
                else:
                    gap_w_mean_l = gap_mean_l
                    gap_w_max_l = gap_max_l
                    loc_factor_l = [1.0]
                    _asinh_grad = torch.asinh(gap_blocks).abs()
                    asinh_grad_mean_l = [_asinh_grad.mean().item()]
                    asinh_grad_max_l = [_asinh_grad.max().item()]
                sl_ratio_l = [float((loss_shape.detach()
                                     / loss_level.detach().clamp_min(self._EPS)).item())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossV39: per_channel={comp}")

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
            # ── V39: block diagnostics ──
            "gap_w_mean":     gap_w_mean_l,
            "gap_w_max":      gap_w_max_l,
            "loc_factor":     loc_factor_l,
            "asinh_grad_mean": asinh_grad_mean_l,
            "asinh_grad_max":  asinh_grad_max_l,
        }

        logger.debug(
            "SpotlightLossV39 | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV39(non_zero_threshold={self.tau})"
