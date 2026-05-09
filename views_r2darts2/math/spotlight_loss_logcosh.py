import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v36 — asinh + RevIN compatible, with DRO aggregation.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros, 10% spanning
    four orders of magnitude in raw deaths.

    ── Design rationale ─────────────────────────────────────────────────

    Four orthogonal components, each addressing a specific failure mode:

    1. **DC/AC decomposition** — prevents RevIN DC offset amplification.
       Error is demeaned per series: e_shape = e − mean(e). The shape
       gradient sums to exactly zero per series (structural, not tuned):

           Σᵢ ∂L_shape/∂ŷᵢ = 0    ∀ series

       Proof: e_shape = J·e where J = I − 11ᵀ/T is the centering matrix.
       J has zero column sums → backprop through J zeroes out the DC
       component of the gradient, regardless of per-cell weights.

       Why this matters with RevIN: RevIN denormalizes as ŷ = ẑ·σ + μ.
       A small bias b in normalized space becomes b·σ in asinh space.
       Through sinh (convex for x > 0), Jensen's inequality amplifies
       this to E[sinh(b·σ)] > sinh(E[b·σ]) — exponential overprediction
       in raw counts. The DC/AC split makes it structurally impossible
       for the shape loss to accumulate any DC bias, period.

    2. **Adaptive compound weighting** — parameter-free, replaces alpha.

       Per-cell weight is the product of two bounded signals:

           difficulty  = 1 − exp(−|e_shape|)            ∈ [0, 1)
           importance  = 1 − exp(−max(|y|, |ŷ_sg|))    ∈ [0, 1)
           w_compound  = 1 + difficulty × importance     ∈ [1, 2)

       Both must be present for high weight: a cell must be **both hard
       AND important**. Detached — no gradient coupling. Normalised to
       mean=1 jointly with DRO weights.

       Replaces the alpha-tuned `w = 1 + log_cosh(α · mag)` from v35.
       Alpha was a hyperparameter controlling event budget; compound
       weighting achieves this automatically — cells the model already
       predicts well get difficulty→0 → w→1 regardless of magnitude.

    3. **KL-DRO tail aggregation (log-space)** — parameter-free.

       Instead of a plain mean over weighted shape losses, z-score
       log(cell_loss) and apply concave-compressed DRO weights:

           log_l = log(l + ε)
           z = (log_l − mean(log_l)) / std(log_l)
           dro_w = log1p(clamp(1+z, min=0))
           dro_w = dro_w / mean(dro_w)

       KL-DRO detects *proportional* outliers, not absolute. A
       5-death miss at 10× the median loss gets the same DRO emphasis
       as a 1000-death miss at 10× the median. Aligned with the
       proportional error sensitivity of asinh-space prediction.

       Soft activation α = log_std/(log_std + 1.0) blends toward
       uniform when log-loss variance is small (early training).

       Compound weight and DRO are combined independently (product,
       normalised jointly to mean=1) — they address orthogonal concerns:
       compound steers *which cells matter*, DRO steers *how losses
       are aggregated* across the 90/10 peace/event split.

    4. **Level anchor** — T-scaled log_cosh on per-series mean error.

           L_level = T · mean_over_series[ log_cosh(mean(ŷ) − mean(y)) ]

       The ONLY mechanism that can shift per-series means. Necessary
       because the shape loss (DC/AC decomposed) is structurally blind
       to level. T-scaling compensates for the 1/T chain-rule factor.
       Natural curriculum: large mean error early → level dominates →
       calibrate means first. Small mean error later → shape takes over.
       Not DRO-weighted — operates on a fundamentally different
       aggregation dimension (per-series means, not per-cell losses).

    5. **Spectral regularization** (optional, gated by δ > 0).
       Multi-resolution STFT magnitude comparison with DC bin masked.
       Unchanged from v35. Phase-invariant; log_cosh on magnitude diffs.

    ── Base cell loss: log_cosh ─────────────────────────────────────────

    log_cosh(x) ≈ 0.5x² for |x| < 1, ≈ |x| − ln2 for |x| > 2.
    Gradient = tanh(x) ∈ (−1, +1). Bounded by construction.

    ── Changes from v35 ─────────────────────────────────────────────────

    - `alpha` parameter removed. Compound weighting is parameter-free.
    - KL-DRO aggregation replaces simple weighted mean on shape loss.
    - Compound weight (difficulty × importance) replaces alpha-scaled
      log_cosh importance weight.
    - Level anchor unchanged. Spectral unchanged.

    ─────────────────────────────────────────────────────────────────────

    Args:
        delta: Spectral loss weight. 0.0 = disable.
            0.10–0.15 = spectral is ~15–25% of gradient.
            Range: [0.05, 0.20].
        non_zero_threshold: Transformed-space cutoff for spectral signal
            filtering (which series get spectral comparison).
            Value depends on target scaler:
            - AsinhTransform: 0.88 ≈ asinh(1)
            - FourthRootTransform: 0.19 ≈ (1+1)^0.25 − 1

    Example:
        >>> loss_fn = SpotlightLoss(delta=0.10, non_zero_threshold=0.19)
        >>> y_pred = torch.randn(8, 36)  # transformed-space predictions
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))

    def __init__(
        self,
        delta: float,
        non_zero_threshold: float,
        alpha: float = 0.0,  # deprecated — ignored, kept for backward compat
    ):
        if alpha != 0.0:
            logger.warning(
                "SpotlightLoss v36: alpha is deprecated and ignored. "
                "Compound weighting + KL-DRO replaces alpha-based importance. "
                "Remove alpha from your config. (received alpha=%.4f)",
                alpha,
            )
        if delta < 0.0:
            raise ValueError(f"SpotlightLoss: delta must be non-negative, got {delta}")
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"SpotlightLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.delta = delta
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "SpotlightLoss v36 (DRO) | delta=%.4f threshold=%.4f",
            delta, non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only)."""
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        has_signal = (
            (torch.abs(true) > self.non_zero_threshold)
            | (torch.abs(pred.detach()) > self.non_zero_threshold)
        ).any(dim=1)
        if not has_signal.any():
            return pred.new_tensor(0.0)
        pred = pred[has_signal]
        true = true[has_signal]

        T = pred.size(1)
        total = pred.new_tensor(0.0)
        n_valid = 0

        for n_fft, hop in self.SPECTRAL_RESOLUTIONS:
            if T < n_fft:
                continue

            window = torch.hann_window(n_fft, device=pred.device)
            S_pred = torch.stft(
                pred, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )
            S_true = torch.stft(
                true, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )

            # Safe magnitude: sqrt(re² + im² + ε) — bounded gradient.
            # Do NOT use .abs() on pred side (gradient blows up at |z|→0).
            mag_pred = torch.sqrt(S_pred.real ** 2 + S_pred.imag ** 2 + 1e-8)
            mag_true = S_true.abs()

            # Mask DC bin — level is handled by the level anchor.
            mag_pred = mag_pred.clone()
            mag_true = mag_true.clone()
            mag_pred[:, 0, :] = 0.0
            mag_true[:, 0, :] = 0.0

            total = total + self._log_cosh(mag_pred - mag_true).mean()
            n_valid += 1

        return total / max(n_valid, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        T = y_pred.size(1)
        e = y_pred - y_true

        # ── DC/AC decomposition ───────────────────────────────────────
        # e_shape sums to zero per series → shape gradient is DC-free.
        # This is the structural RevIN safety mechanism.
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        # ── Base cell loss: log_cosh on demeaned error ────────────────
        cell_loss = self._log_cosh(e_shape)

        # ── Adaptive compound weighting (dynamic, self-correcting) ─────
        # difficulty = 1 − exp(−|e_shape|) : how wrong (curriculum)
        # event = 𝟙(|y| > threshold) : binary, all events equal
        # w_compound = 1 + difficulty × event ∈ [1, 2)
        # Self-correcting: as |e|→0, w→1 regardless of |y|.
        # Binary event indicator: 1-death and 100-death events get the
        # same importance — only difficulty differentiates within events.
        abs_e = torch.abs(e_shape.detach())
        abs_y = torch.abs(y_true)

        difficulty = 1.0 - torch.exp(-abs_e)
        event = (abs_y > self.non_zero_threshold).float()
        w_compound = 1.0 + difficulty * event

        # ── KL-DRO tail aggregation (log-space z-scores) ──────────────
        # Z-score log(cell_loss) for proportional outlier detection.
        # Operates on raw cell_loss (before compound weighting).
        # Flattened cross-series: event cells dominate the tail.
        loss_flat = cell_loss.detach().flatten()
        log_loss = torch.log(loss_flat + 1e-8)
        log_std = log_loss.std()

        # CV-based alpha: self-normalizing, naturally bounded.
        # CV=2→α≈0.52, CV=5→α≈0.64, CV=10→α≈0.71. Cannot saturate to 1.
        log_cv = torch.log1p(log_std / (log_loss.mean().abs() + 1e-8))
        dro_alpha = log_cv / (log_cv + 1.0)
        # Clamp at 0.1, not 1e-8: std<0.1 means losses span <1.1×,
        # too little variation for meaningful z-scores. At 1e-8, z can
        # reach 1e8 → log1p(1e8) ≈ 18.4 → NaN after normalisation × 0.
        z = (log_loss - log_loss.mean()) / log_std.clamp(min=0.1)
        w_dro = torch.log1p((1.0 + z).clamp(min=0.0))
        w_dro = w_dro / w_dro.mean().clamp(min=1e-8)
        w_dro = w_dro.view_as(cell_loss)
        w_dro = dro_alpha * w_dro + (1.0 - dro_alpha)

        # Combine compound + DRO additively (mean-preserving average).
        # Both are correlated (respond to same high-conflict cells);
        # multiplication would quadratically suppress low-conflict events.
        w_compound_norm = w_compound / w_compound.mean()
        w_total = 0.5 * w_compound_norm + 0.5 * w_dro
        # Safety: any residual NaN from degenerate batches → weight=1
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)

        loss_shape = (w_total * cell_loss).mean()

        # ── Level anchor: T-scaled log_cosh on per-series mean error ──
        # Only mechanism that can shift per-series means. Shape loss is
        # structurally DC-blind. Not DRO-weighted — different dimension.
        # T scaling: ∂L/∂ŷⱼ = T·tanh(ē)·(1/T) = tanh(ē) per cell.
        # L2 norm across T cells = √T·|tanh(ē)|, matching shape gradient
        # norm ~ √T·avg|ρ'(e_shape)|. Natural curriculum preserved:
        # large |ē| → saturated tanh → strong level signal;
        # small |ē| → tanh ≈ ē → level fades, shape takes over.
        loss_level = T * self._log_cosh(e_mean.squeeze(1)).mean()

        # ── Spectral: AC bins only ────────────────────────────────────
        loss_spectral = y_pred.new_tensor(0.0)
        if self.delta > 0.0 and T >= 6:
            loss_spectral = self._spectral_loss(y_pred, y_true)

        total_loss = loss_shape + loss_level + self.delta * loss_spectral

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLoss: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} spectral={loss_spectral.item():.6f}"
            )

        logger.debug(
            "SpotlightLoss | shape=%.6f level=%.6f spec=%.6f total=%.6f",
            loss_shape.item(),
            loss_level.item(),
            loss_spectral.item(),
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLossLogcosh(delta={self.delta}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )