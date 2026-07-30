"""SpotlightLossV64 — orthogonal mean-gap level + full-horizon DRO shape.

Level : global mean gap → asinh_plus → Hájek aggregation → T-scaled.
        Gradient is uniform across all T cells  (DC correction).

Shape : full-horizon demeaned errors → DRO → gate → log_cosh.
        Gradient is exactly zero-sum across all T cells  (AC correction).

Orthogonality (exact, per series):
    The demeaning is in the forward pass: r_i = e_i − mean(e).
    Therefore d(r_i)/d(e_k) = δ_ik − 1/T, and for any weights w_i:

        Σ_k  ∂L_shape/∂e_k
      = (1/W) Σ_k [ w_k f′(r_k) − (1/T) Σ_i w_i f′(r_i) ]
      = (1/W) [ Σ_k w_k f′(r_k) − Σ_i w_i f′(r_i) ]
      = 0

    This holds for ANY choice of w_i (gate, DRO, etc.).
    The level gradient is constant across t, so:

        ⟨ ∇L_level , ∇L_shape ⟩  ∝  Σ_k ∂L_shape/∂e_k  =  0

Why this fixes the PGM flatline:
    Event-masked demeaning gave e_shape = 0 for series with 0–1 events
    (algebraically: e[ev] − mean({e[ev]}) = 0 when |events| ≤ 1).
    Full-horizon demeaning gives e_shape ≠ 0 for any non-constant
    series, so the shape loss pushes spikes up and zeros down even
    at 98 % zero-inflation, while the level loss calibrates the mean.
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


# ═══════════════════════════════════════════════════════════════════
#  Level Loss
# ═══════════════════════════════════════════════════════════════════

class LevelLoss:
    """Global mean-gap level anchor with Hájek normalization.

        L = T · Σᵢ(wᵢ · φ(gapᵢ)) / Σᵢ(wᵢ)

    gapᵢ = (1/T) Σₜ (ŷᵢₜ − yᵢₜ)      global horizon mean error
    φ    = asinh_plus                     non-saturating
    wᵢ   = sigmoid series-magnitude gate  composition-robust

    The factor T inverts the 1/T gradient attenuation of the mean
    operator, giving per-cell gradient magnitude ≈ φ′(gap) · wᵢ/Σw.

    Hájek normalization Σ(w·ℓ)/Σ(w) makes the loss scale invariant
    to the peaceful/event ratio in the batch.
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

        multivariate = e.dim() == 3

        # ── global mean gap ──────────────────────────────────────────
        gap = e.mean(dim=1)                       # (B,) | (B, C)
        level_cell = asinh_plus(gap)

        # ── series magnitude gate ────────────────────────────────────
        abs_max_series = torch.max(y_true.abs(), y_pred_det.abs())
        series_mag = abs_max_series.max(dim=1).values
        series_w = 0.0125 + 0.9875 * torch.sigmoid(
            10.0 * (series_mag - self.tau)
        )

        # ── Hájek aggregation ────────────────────────────────────────
        if multivariate:
            num = (series_w * level_cell).sum(dim=0)
            den = series_w.sum(dim=0).clamp(min=self.eps)
        else:
            num = (series_w * level_cell).sum()
            den = series_w.sum().clamp(min=self.eps)

        loss = T * num / den

        with torch.no_grad():
            aux = {
                "gap_mean":            gap.abs().mean().item(),
                "gap_max":             gap.abs().max().item(),
                "series_w_mean":       series_w.mean().item(),
                "series_w_std":        series_w.std().item(),
                "series_w_event_frac": (series_w > 0.5).float().mean().item(),
            }
        return loss, aux


# ═══════════════════════════════════════════════════════════════════
#  Shape Loss
# ═══════════════════════════════════════════════════════════════════

class ShapeLoss:
    """Full-horizon demeaned shape loss with DRO and gate.

    Demeaning is over ALL T cells (not event-masked):

        r = e − mean(e)

    This guarantees the shape gradient is exactly zero-sum for any
    weighting scheme, making it orthogonal to the level loss.

    DRO is computed over all cells.  The sigmoid gate suppresses
    non-event cells in the loss aggregation, so the effective
    denominator tracks the number of event cells (implicit Hájek).
    """

    def __init__(self, tau: float, eps: float = 1e-6):
        self.tau = tau
        self.eps = eps

    def __call__(
        self,
        e: torch.Tensor,
        gate: torch.Tensor,
        event_mask: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, dict]:

        multivariate = e.dim() == 3

        # ── full-horizon demeaning ───────────────────────────────────
        e_mean  = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        # ── DRO over all cells ───────────────────────────────────────
        raw   = e_shape.abs().detach()
        mu    = raw.mean(dim=1, keepdim=True).clamp_min(self.eps)
        valid = (raw > 1e-6).float()
        w_dro = torch.sqrt((raw * valid) / mu)
        w_bar = w_dro.mean(dim=1, keepdim=True).clamp_min(1e-8)
        w_dro = w_dro / w_bar
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=1.0, neginf=0.0)

        # ── gate × DRO × log_cosh ────────────────────────────────────
        shape_w = gate * w_dro
        cell    = log_cosh(e_shape)

        if multivariate:
            loss = (
                (shape_w * cell).sum(dim=(0, 1))
                / shape_w.sum(dim=(0, 1)).clamp_min(self.eps)
            )
        else:
            loss = (shape_w * cell).sum() / shape_w.sum().clamp_min(self.eps)

        # ── diagnostics ──────────────────────────────────────────────
        with torch.no_grad():
            if multivariate:
                n_tot = event_mask.sum(dim=(0, 1)).clamp_min(1.0)
                w_ev  = w_dro * event_mask
                dm    = w_ev.sum(dim=(0, 1)) / n_tot
                dw2   = (w_ev ** 2).sum(dim=(0, 1)) / n_tot
                dstd  = (dw2 - dm ** 2).clamp_min(0).sqrt()
                aux = {
                    "dro_w_mean":  dm.mean().item(),
                    "dro_w_std":   dstd.mean().item(),
                    "dro_w_max":   w_dro.amax(dim=(0, 1)).mean().item(),
                    "dro_frac_up": (((w_dro > 1.0) * event_mask)
                                    .sum(dim=(0, 1)) / n_tot).mean().item(),
                    "event_frac":  event_mask.mean(dim=(0, 1)).mean().item(),
                    "shape_dc":    ((gate * e_shape).mean(dim=1).abs()
                                    .mean(dim=0).mean().item()),
                }
            else:
                n_tot = event_mask.sum().clamp_min(1.0)
                w_ev  = w_dro * event_mask
                dm    = (w_ev.sum() / n_tot).item()
                dw2   = ((w_ev ** 2).sum() / n_tot).item()
                aux = {
                    "dro_w_mean":  dm,
                    "dro_w_std":   max(0.0, dw2 - dm ** 2) ** 0.5,
                    "dro_w_max":   w_dro.max().item(),
                    "dro_frac_up": (((w_dro > 1.0) * event_mask)
                                    .sum().item() / n_tot.item()),
                    "event_frac":  event_mask.mean().item(),
                    "shape_dc":    (gate * e_shape).mean(dim=1).abs().mean().item(),
                }

        return loss, aux


# ═══════════════════════════════════════════════════════════════════
#  Main Loss
# ═══════════════════════════════════════════════════════════════════

class SpotlightLossLogcosh(torch.nn.Module):
    """V64: orthogonal mean-gap level + full-horizon DRO shape.

    Level  – global mean gap → asinh_plus → Hájek → T-scaled.
             Gradient is uniform across T  (DC correction).

    Shape  – full-horizon demeaned errors → DRO → gate → log_cosh.
             Gradient is exactly zero-sum across T  (AC correction).

    Orthogonality:
        ⟨ ∇L_level , ∇L_shape ⟩ = 0   (exact, per series,
        for any gate / DRO weighting)
    """

    _EPS = 1e-6
    _K   = 4                              # callback compat

    def __init__(self, non_zero_threshold: float = 0.88):
        super().__init__()
        self.tau = non_zero_threshold
        self.level_loss = LevelLoss(tau=self.tau, eps=self._EPS)
        self.shape_loss = ShapeLoss(tau=self.tau, eps=self._EPS)
        self._last_components: dict | None = None
        self._last_input_grad:  torch.Tensor | None = None
        logger.info(
            "SpotlightLossV64 | tau=%.4f | orthogonal mean-gap level "
            "+ full-horizon DRO shape",
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
        n  = len(comp)
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
                f"NaN in SpotlightLossV64: per_channel={comp}"
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
            # level aux
            "series_w_mean":       _rep(lv_aux["series_w_mean"]),
            "series_w_std":        _rep(lv_aux["series_w_std"]),
            "series_w_event_frac": _rep(lv_aux["series_w_event_frac"]),
        }

        logger.debug(
            "SpotlightLossV64 | sh=%s lv=%s total=%.4f "
            "gap=%.4f ev_w%%=%.1f",
            shape_c, level_c, total_loss.item(),
            lv_aux["gap_mean"],
            lv_aux["series_w_event_frac"] * 100,
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossV64(non_zero_threshold={self.tau})"