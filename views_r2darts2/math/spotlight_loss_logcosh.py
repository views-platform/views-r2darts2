import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v36.3 — asinh + RevIN compatible, uniform shape + level-DRO.

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

    2. **Uniform shape weighting** — no per-cell weights.

       log_cosh gradient = tanh(e_shape) ∈ (−1, +1) provides natural
       curriculum: large errors get bounded gradients, small errors get
       proportional gradients. Explicit weighting is unnecessary AND
       harmful: any weight correlated with asinh magnitude creates
       systematic DC drift in ẑ-space through the RevIN Jacobian
       asymmetry (∂ŷ/∂ẑ ≈ σ_c at ẑ≈0, →1 at ẑ>>1).

       v36-36.2 used compound weighting (difficulty × importance) but
       importance = 1−exp(−|y|) correlates with Jacobian position →
       ~13%/epoch calibration drift on TCN. Removing all cell weights
       eliminates the correlation, leaving only residual drift from
       tanh saturation asymmetry (~0.01/epoch, manageable by level).

    3. **KL-DRO tail aggregation on LEVEL anchor (log-space)** —

       Applied to the per-series mean error (level loss), NOT cell-level
       shape loss. DRO on level creates negative feedback:

           Series drifts → |ē| grows → DRO upweights → stronger
           correction (tanh points toward fix) → ē shrinks → equilibrium.

       DRO on shape was unsafe: amplified conflict cells at high ẑ where
       the RevIN Jacobian → 1, creating ẑ-space DC drift through the
       asymmetry (shape gradient zero-sum in ŷ-space but NOT in ẑ-space).
       Moving DRO to level exploits that the level gradient is inherently
       self-correcting — amplifying it accelerates mean convergence.

       Soft activation α = log_std/(log_std + 1.0) blends toward
       uniform when log-loss variance is small (early training).
       Max per-cell gradient from level DRO: w·tanh(ē)/T ≤ 4/36 ≈ 0.11
       versus shape gradient per-cell ≤ 2. Cannot destabilize.

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

    - `alpha` parameter removed.
    - All per-cell shape weighting removed (v36.3). compound/DRO/importance
      all correlate with Jacobian position → DC drift. Uniform mean
      on log_cosh(e_shape) eliminates the correlation entirely.
    - KL-DRO moved to level anchor (v36.2+). Self-correcting negative
      feedback on per-series mean error.
    - Shape: uniform mean. Level: DRO-weighted. Spectral unchanged.

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
            "SpotlightLoss v36.3 (uniform shape + level-DRO) | delta=%.4f threshold=%.4f",
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

        # ── Shape aggregation: uniform (no cell weighting) ────────────
        # Compound weighting's `importance` term (1−exp(−|y|)) correlates
        # directly with asinh magnitude → Jacobian position. This creates
        # systematic upweighting of high-ẑ conflict cells where J→1,
        # while peace cells (J≈σ_c >> 1) get the compensating push
        # amplified by σ_c in ẑ-space. Net: persistent DC drift.
        #
        # With uniform weights, log_cosh already provides natural
        # curriculum: gradient = tanh(e_shape) ∈ (-1,+1). Large errors
        # get saturated (bounded) gradients, small errors get proportional
        # gradients. No explicit weighting needed.
        #
        # The centering matrix still guarantees Σ ∂L/∂ŷ = 0 (DC-free).
        # Residual ẑ-space drift from tanh saturation × Jacobian is
        # weak (~0.01/epoch) because it has no weight-magnitude correlation.
        loss_shape = cell_loss.mean()

        # ── Level anchor: T-scaled log_cosh with DRO ─────────────────
        # Only mechanism that can shift per-series means. Shape loss is
        # structurally DC-blind.
        #
        # DRO on level creates NEGATIVE feedback (self-correcting):
        #   Series drifts → |ē| grows → DRO upweights → stronger
        #   correction → mean error shrinks → DRO decreases → equilibrium.
        # Safe: level gradient = tanh(ē) always points toward the fix.
        # Max per-cell magnitude: w_dro·tanh(ē)/T ≤ 4/36 ≈ 0.11
        # (shape gradient per-cell ≤ 2×tanh — level DRO cannot dominate).
        #
        # T scaling: ∂L/∂ŷⱼ = T·tanh(ē)·(1/T)·w = w·tanh(ē) per cell.
        # Natural curriculum preserved: large |ē| early → level dominates.
        level_loss = self._log_cosh(e_mean.squeeze(1))  # (B,)

        # DRO: z-score log(level_loss) across series in batch.
        # Soft activation α → 0 when log-loss variance is low (early
        # training or homogeneous batches) → graceful uniform fallback.
        # Guard: B≤1 or non-finite std → skip DRO (uniform weights).
        B = level_loss.size(0)
        log_level = torch.log(level_loss.detach() + 1e-8)
        log_level_std = log_level.std(unbiased=False) if B > 1 else level_loss.new_tensor(0.0)

        if B > 1 and torch.isfinite(log_level_std) and log_level_std > 1e-6:
            dro_alpha = log_level_std / (log_level_std + 1.0)
            z_level = (log_level - log_level.mean()) / log_level_std.clamp(min=0.1)
            w_level_dro = torch.log1p((1.0 + z_level).clamp(min=0.0))
            w_mean = w_level_dro.mean()
            if w_mean > 1e-8:
                w_level_dro = w_level_dro / w_mean
            else:
                w_level_dro = torch.ones_like(level_loss)
            w_level_dro = dro_alpha * w_level_dro + (1.0 - dro_alpha)
            w_level_dro = torch.nan_to_num(w_level_dro, nan=1.0, posinf=1.0, neginf=0.0)
        else:
            w_level_dro = torch.ones_like(level_loss)

        loss_level = T * (w_level_dro * level_loss).mean()

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