import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    SpotlightLoss v35 — simplified for asinh + RevIN.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros, 10% spanning
    four orders of magnitude in raw deaths.

    ── Design rationale ─────────────────────────────────────────────────

    Three orthogonal components, each addressing a specific failure mode:

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

    2. **Importance weight** — continuous inverse-density reweighting.

           w = 1 + log_cosh(α · max(|y|, |ŷ_sg|))

       Equivalent to inverse label-density weighting (Yang et al. 2021,
       "Delving into Deep Imbalanced Regression") but without requiring
       explicit density estimation. Since UCDP label density decreases
       monotonically with magnitude, magnitude-based weighting is
       isomorphic to inverse-density weighting.

       Symmetric max(|y|, |ŷ_sg|) gives equal corrective pressure to
       misses and false alarms of equal magnitude. ŷ is detached to
       prevent second-order gradient amplification. log_cosh saturates
       linearly, so extreme events (50k deaths → asinh ≈ 11.5) get at
       most ~3.8× weight (at α=0.3) — cannot hijack shared weights.

       This replaces the old dual_mean/event_weight mechanism. dual_mean
       introduced a hard binary split and two extra hyperparameters, but
       the importance weight already provides soft continuous reweighting.
       At α=0.3 with 90% zeros, the effective event gradient budget is
       ~30% — adequate balance without the overprediction risk of
       dual_mean at event_weight ≥ 0.25. (Ren et al. 2022, "Balanced
       MSE": explicit inverse-density reweighting is a complete
       solution; stacking class-based bucketing on top is redundant.)

    3. **Level anchor** — T-scaled log_cosh on per-series mean error.

           L_level = T · mean_over_series[ log_cosh(mean(ŷ) − mean(y)) ]

       The ONLY mechanism that can shift per-series means. Necessary
       because the shape loss (DC/AC decomposed) is structurally blind
       to level. T-scaling compensates for the 1/T chain-rule factor
       through the mean, making level gradient O(1) per cell — comparable
       to shape gradient magnitude. Natural curriculum: large mean error
       early → tanh saturates → level dominates → calibrate means first.
       Small mean error later → shape gradient takes over → learn detail.
       Bounded at ±1 per series — no overcorrection from outlier batches.

    4. **Spectral regularization** (optional, gated by δ > 0).
       Multi-resolution STFT magnitude comparison with DC bin masked.
       Phase-invariant: timing errors (1 month early) ≈ zero penalty.
       Prevents flat forecasts, hockey sticks, missing seasonality.
       log_cosh on magnitude differences with bounded gradients.

    ── Base cell loss: log_cosh ─────────────────────────────────────────

    log_cosh(x) ≈ 0.5x² for |x| < 1, ≈ |x| − ln2 for |x| > 2.
    Gradient = tanh(x) ∈ (−1, +1). Bounded by construction.
    Max effective gradient per cell: w · tanh(e_shape) ≤ 3.8 (α=0.3).

    ── Gradient analysis ────────────────────────────────────────────────

    Shape gradient per cell i:  w_i · tanh(e_shape_i) − (1/T)·Σⱼ wⱼ · tanh(e_shape_j)
    Sum over series:            exactly 0 (structural guarantee)
    Level gradient per cell:    tanh(μ_e) (uniform, only DC control)
    Total gradient norm:        bounded ≤ α_max + 1 ≈ 4.8 per cell

    ─────────────────────────────────────────────────────────────────────

    Args:
        alpha: Importance weight steepness. Controls how much gradient
            goes to rare high-magnitude cells vs common zero cells.
            0.3 → w ∈ [1.0, 3.8], effective event budget ~30%.
            Range: [0.15, 0.5]. Sweep this.
        delta: Spectral loss weight. 0.0 = disable.
            0.10–0.15 = spectral is ~15–25% of gradient.
            Range: [0.05, 0.20].
        non_zero_threshold: asinh-space cutoff for spectral signal
            filtering (which series get spectral comparison).
            0.88 ≈ asinh(1) ≈ 1 battle death.

    Example:
        >>> loss_fn = SpotlightLoss(alpha=0.3, delta=0.10, non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)  # asinh-space predictions
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))

    def __init__(
        self,
        alpha: float,
        delta: float,
        non_zero_threshold: float,
    ):
        if alpha <= 0.0:
            raise ValueError(f"SpotlightLoss: alpha must be positive, got {alpha}")
        if alpha > 0.7:
            _w = 1.0 + math.log(math.cosh(min(alpha * 11.5, 88.0)))
            logger.warning(
                "SpotlightLoss: alpha=%.4f > 0.7. Max weight at asinh≈11.5 "
                "= %.1f×. Likely to cause overprediction.",
                alpha, _w,
            )
        if delta < 0.0:
            raise ValueError(f"SpotlightLoss: delta must be non-negative, got {delta}")
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"SpotlightLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "SpotlightLoss v35 | alpha=%.4f delta=%.4f threshold=%.4f",
            alpha, delta, non_zero_threshold,
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
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        # ── Importance weight (detached, symmetric) ───────────────────
        abs_y = torch.abs(y_true)
        abs_y_hat = torch.abs(y_pred.detach())
        w = 1.0 + self._log_cosh(self.alpha * torch.max(abs_y, abs_y_hat))

        # ── Shape loss: weighted log_cosh on demeaned error ───────────
        loss_shape = (w * self._log_cosh(e_shape)).mean()

        # ── Level anchor: T-scaled log_cosh on per-series mean error ──
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
            f"SpotlightLoss(alpha={self.alpha}, delta={self.delta}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )
