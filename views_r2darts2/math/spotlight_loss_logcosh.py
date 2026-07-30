"""SpotlightLossV63 — windowed Hájek level + windowed DRO shape.

Both components share the same non-overlapping windows of size
W = max(6, T // 3).

Level  : per-window mean errors → log_cosh → Hájek aggregation.
Shape  : per-window event-demeaned errors → DRO → gate → log_cosh.

The per-window demeaning in the shape loss makes its gradient
zero-sum within each window's events, which is orthogonal to the
level loss's uniform per-window gradient.
"""

import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Math primitives
# ═══════════════════════════════════════════════════════════════════

def log_cosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(cosh(x))."""
    a = x.abs()
    return a + F.softplus(-2.0 * a) - math.log(2.0)


def asinh_plus(x: torch.Tensor) -> torch.Tensor:
    """x · asinh(x).  Non-saturating; MSE curvature at origin."""
    return x * torch.asinh(x)


def window_size(T: int) -> int:
    """Non-overlapping window size for horizon T."""
    return max(6, T // 3)


# ═══════════════════════════════════════════════════════════════════
#  Windowed Level Loss
# ═══════════════════════════════════════════════════════════════════

class WindowedLevelLoss:
    """Windowed Hájek level anchor.

    Splits the error into non-overlapping windows, evaluates log_cosh
    on each window's mean error, and aggregates with Hájek
    self-normalized series weighting:

        L = T · Σᵢ(wᵢ · ℓᵢ) / Σᵢ(wᵢ)

    Hájek normalization makes the loss scale invariant to the
    peaceful / event composition of the batch.  At 98 % zero-inflation
    this gives event series ~66 % of the level gradient instead of ~2 %.

    The factor T compensates for the 1/W gradient attenuation of the
    window-mean operator.
    """

    def __init__(self, tau: float, eps: float = 1e-6):
        self.tau = tau
        self.eps = eps

    def __call__(
        self,
        e: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_det: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, dict]:

        W = window_size(T)
        chunks = e.split(W, dim=1)
        n_win = len(chunks)

        # (B, n_win) or (B, n_win, C)
        win_means = torch.stack([c.mean(dim=1) for c in chunks], dim=1)
        win_losses = log_cosh(win_means)

        # ── series magnitude gate ────────────────────────────────────
        abs_max_series = torch.max(y_true.abs(), y_pred_det.abs())
        series_mag = abs_max_series.max(dim=1).values          # (B,) | (B,C)
        series_w = 0.0125 + 0.9875 * torch.sigmoid(
            10.0 * (series_mag - self.tau)
        )

        # ── Hájek aggregation ────────────────────────────────────────
        if win_losses.dim() == 3:                              # multivariate
            num = (series_w.unsqueeze(1) * win_losses).sum(dim=(0, 1))
            den = (series_w.sum(dim=0) * n_win).clamp(min=self.eps)
        else:                                                  # univariate
            num = (series_w.unsqueeze(1) * win_losses).sum()
            den = (series_w.sum() * n_win).clamp(min=self.eps)

        loss = T * num / den

        aux = {
            "win_gap_mean":       win_means.abs().mean().item(),
            "win_gap_max":        win_means.abs().max().item(),
            "series_w_mean":      series_w.mean().item(),
            "series_w_std":       series_w.std().item(),
            "series_w_event_frac": (series_w > 0.5).float().mean().item(),
            "n_windows":          n_win,
            "window_size":        W,
        }
        return loss, aux


# ═══════════════════════════════════════════════════════════════════
#  Windowed Shape Loss
# ═══════════════════════════════════════════════════════════════════

class WindowedShapeLoss:
    """Windowed event-masked demeaned shape loss with DRO.

    Uses the same window grid as the level loss.  Within each window:

      1. event-masked mean   μ_w = Σ(mask·e) / Σ(mask)
      2. demean              r  = e − μ_w
      3. DRO weights         w  = √(|r| / mean|r|), normalised
      4. gate                g  = σ(15·(max|y|−τ))
      5. cell loss           ℓ  = log_cosh(r)

    The per-window demeaning makes the shape gradient zero-sum
    within each window's event cells → orthogonal to the level
    loss's uniform per-window gradient.
    """

    def __init__(self, tau: float, eps: float = 1e-6):
        self.tau = tau
        self.eps = eps

    # ── DRO helper ───────────────────────────────────────────────────
    @staticmethod
    def _dro_weights(
        e_shape: torch.Tensor,
        mask: torch.Tensor,
        n_ev: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        raw = e_shape.abs().detach()
        mu = (raw * mask).sum(dim=1, keepdim=True) / n_ev
        valid = (raw > 1e-6).float()
        w = torch.sqrt((raw * valid) / mu.clamp_min(eps))
        w_bar = (w * mask).sum(dim=1, keepdim=True) / n_ev
        w = w / w_bar.clamp_min(1e-8)
        w = torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)
        return 1.0 + mask * (w - 1.0)

    # ── forward ──────────────────────────────────────────────────────
    def __call__(
        self,
        e: torch.Tensor,
        gate: torch.Tensor,
        event_mask: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, dict]:

        W = window_size(T)
        multivariate = e.dim() == 3

        chunks_e = e.split(W, dim=1)
        chunks_g = gate.split(W, dim=1)
        chunks_m = event_mask.split(W, dim=1)

        # accumulators
        if multivariate:
            num = e.new_zeros(e.shape[2])
            den = e.new_zeros(e.shape[2])
        else:
            num = e.new_zeros(())
            den = e.new_zeros(())

        diag_dro, diag_mask, diag_ge = [], [], []

        for e_w, g_w, m_w in zip(chunks_e, chunks_g, chunks_m):
            # 1–2  event-masked demeaning within window
            n_ev = m_w.sum(dim=1, keepdim=True).clamp_min(self.eps)
            e_mean = (m_w * e_w).sum(dim=1, keepdim=True) / n_ev
            e_shape = e_w - e_mean

            # 3  DRO
            w_dro = self._dro_weights(e_shape, m_w, n_ev, self.eps)

            # 4–5  gate × log_cosh
            sw = g_w * w_dro
            cell = log_cosh(e_shape)

            if multivariate:
                num = num + (sw * cell).sum(dim=(0, 1))
                den = den + sw.sum(dim=(0, 1))
            else:
                num = num + (sw * cell).sum()
                den = den + sw.sum()

            diag_dro.append(w_dro)
            diag_mask.append(m_w)
            diag_ge.append(g_w * e_shape)

        loss = num / den.clamp_min(self.eps)

        # ── diagnostics ──────────────────────────────────────────────
        with torch.no_grad():
            cat_dro  = torch.cat(diag_dro,  dim=1)
            cat_mask = torch.cat(diag_mask, dim=1)
            cat_ge   = torch.cat(diag_ge,   dim=1)

            if multivariate:
                n_tot = cat_mask.sum(dim=(0, 1)).clamp_min(1.0)
                w_ev  = cat_dro * cat_mask
                dm    = w_ev.sum(dim=(0, 1)) / n_tot
                dw2   = (w_ev ** 2).sum(dim=(0, 1)) / n_tot
                dstd  = (dw2 - dm ** 2).clamp_min(0).sqrt()
                aux = {
                    "dro_w_mean":  dm.mean().item(),
                    "dro_w_std":   dstd.mean().item(),
                    "dro_w_max":   cat_dro.amax(dim=(0, 1)).mean().item(),
                    "dro_frac_up": (((cat_dro > 1.0) * cat_mask)
                                    .sum(dim=(0, 1)) / n_tot).mean().item(),
                    "event_frac":  cat_mask.mean(dim=(0, 1)).mean().item(),
                    "shape_dc":    (cat_ge.mean(dim=1).abs()
                                    .mean(dim=0).mean().item()),
                }
            else:
                n_tot = cat_mask.sum().clamp_min(1.0)
                w_ev  = cat_dro * cat_mask
                dm    = (w_ev.sum() / n_tot).item()
                dw2   = ((w_ev ** 2).sum() / n_tot).item()
                aux = {
                    "dro_w_mean":  dm,
                    "dro_w_std":   max(0.0, dw2 - dm ** 2) ** 0.5,
                    "dro_w_max":   cat_dro.max().item(),
                    "dro_frac_up": (((cat_dro > 1.0) * cat_mask)
                                    .sum().item() / n_tot.item()),
                    "event_frac":  cat_mask.mean().item(),
                    "shape_dc":    cat_ge.mean(dim=1).abs().mean().item(),
                }

        return loss, aux


# ═══════════════════════════════════════════════════════════════════
#  Main Loss
# ═══════════════════════════════════════════════════════════════════

class SpotlightLossLogcosh(torch.nn.Module):
    """V63: windowed Hájek level + windowed DRO shape.

    Both components share the same non-overlapping window grid
    (W = max(6, T//3)).

    Level  – per-window mean errors → log_cosh → Hájek aggregation.
             Gradient is uniform within each window  (DC correction).

    Shape  – per-window event-demeaned errors → DRO → gate → log_cosh.
             Gradient is zero-sum within each window's events
             (AC correction).

    Orthogonality:
        ⟨∇L_level^(k), ∇L_shape^(k)⟩ = 0  for every window k,
        because the level gradient is constant and the shape
        gradient sums to zero over events.
    """

    _EPS = 1e-6
    _K   = 4          # kept for callback compat

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self.level_loss = WindowedLevelLoss(tau=self.tau, eps=self._EPS)
        self.shape_loss = WindowedShapeLoss(tau=self.tau, eps=self._EPS)
        self._last_components: dict | None = None
        self._last_input_grad:  torch.Tensor | None = None
        logger.info(
            "SpotlightLossV63 | tau=%.4f | windowed Hájek level "
            "+ windowed DRO shape",
            self.tau,
        )

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:

        # ── input handling ───────────────────────────────────────────
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        B, T = y_pred.shape[:2]

        self._last_input_grad = None
        if y_pred.requires_grad:
            y_pred.register_hook(
                lambda g: setattr(self, "_last_input_grad", g.detach())
            )

        e = y_pred - y_true
        y_pred_det = y_pred.detach()

        # ── shared gate / mask ───────────────────────────────────────
        abs_max    = torch.max(y_true.abs(), y_pred_det.abs())
        gate       = torch.sigmoid(15.0 * (abs_max - self.tau))
        event_mask = (abs_max > self.tau).float()

        # ── component losses ─────────────────────────────────────────
        loss_level, lv_aux = self.level_loss(e, y_true, y_pred_det, T)
        loss_shape, sh_aux = self.shape_loss(e, gate, event_mask, T)

        # ── combine ──────────────────────────────────────────────────
        if multivariate:
            per_channel = loss_shape + loss_level
            total_loss  = per_channel.sum()
            shape_c = loss_shape.detach().tolist()
            level_c = loss_level.detach().tolist()
            comp    = per_channel.detach().tolist()
        else:
            total_loss = loss_shape + loss_level
            shape_c = [float(loss_shape.detach())]
            level_c = [float(loss_level.detach())]
            comp    = [float(total_loss.detach())]

        # ── telemetry ────────────────────────────────────────────────
        n = len(comp)

        sl = loss_shape.detach() / loss_level.detach().clamp_min(self._EPS)
        sl_l = sl.tolist() if multivariate else [float(sl.item())]

        with torch.no_grad():
            gap_g = (y_pred.mean(dim=1) - y_true.mean(dim=1)).abs()
            if multivariate:
                gap_mean_l = gap_g.mean(dim=0).tolist()
                gap_max_l  = gap_g.amax(dim=0).tolist()
            else:
                gap_mean_l = [gap_g.mean().item()]
                gap_max_l  = [gap_g.max().item()]

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossV63: per_channel={comp}"
            )

        def _rep(v: float) -> list:
            return [v] * n

        self._last_components = {
            # per-channel
            "shape":        shape_c,
            "level":        level_c,
            "spec":         _rep(0.0),
            "weight":       _rep(1.0),
            "ema":          [float("nan")] * n,
            "cal_ratio":    _rep(1.0),
            "cal_score":    _rep(1.0),
            "gates":        _rep(1.0),
            "contribution": comp,
            # shape diagnostics
            "dro_w_mean":       _rep(sh_aux["dro_w_mean"]),
            "dro_w_std":        _rep(sh_aux["dro_w_std"]),
            "dro_w_max":        _rep(sh_aux["dro_w_max"]),
            "dro_frac_up":      _rep(sh_aux["dro_frac_up"]),
            "event_frac":       _rep(sh_aux["event_frac"]),
            "shape_dc":         _rep(sh_aux["shape_dc"]),
            "shape_level_ratio": sl_l,
            # level diagnostics (backward-compat keys)
            "level_gap_mean":    gap_mean_l,
            "level_gap_max":     gap_max_l,
            "level_gap_ev_mean": gap_mean_l,
            "level_gap_ev_max":  gap_max_l,
            "level_gap_sat":     _rep(0.0),
            # window diagnostics (new)
            "win_gap_mean":        _rep(lv_aux["win_gap_mean"]),
            "win_gap_max":         _rep(lv_aux["win_gap_max"]),
            "series_w_mean":       _rep(lv_aux["series_w_mean"]),
            "series_w_std":        _rep(lv_aux["series_w_std"]),
            "series_w_event_frac": _rep(lv_aux["series_w_event_frac"]),
            "n_windows":           _rep(lv_aux["n_windows"]),
            "window_size":         _rep(lv_aux["window_size"]),
        }

        logger.debug(
            "SpotlightLossV63 | sh=%s lv=%s total=%.4f "
            "win_gap=%.4f ev_w%%=%.1f",
            shape_c, level_c, total_loss.item(),
            lv_aux["win_gap_mean"],
            lv_aux["series_w_event_frac"] * 100,
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV63(non_zero_threshold={self.tau})"