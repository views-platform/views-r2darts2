import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossHuber(torch.nn.Module):
    """
    SpotlightLoss — Huber variant with adaptive delta.

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

    2. **Adaptive compound weighting** — difficulty-gated, no magnitude
       bias.

       Per-cell weight uses difficulty only, with a threshold gate:

           difficulty  = |e_shape| / (1 + |e_shape|)               ∈ [0, 1)
           w_compound  = 1 + difficulty × 1[abs_max > τ]           ∈ [1, 2)

       No event_mag term — avoids double-stacking magnitude emphasis
       with DRO. A 2-death cell with the same proportional error as
       a 500-death cell gets the same compound weight. The threshold
       gate still provides false-positive discipline: sub-threshold
       cells get w=1. DRO alone handles the event-vs-peace split
       (event cells are naturally loss outliers). The difficulty
       function uses |e|/(1+|e|) instead of 1−exp(−|e|) for slower
       saturation — a 10-unit error gets 1.82× the weight of a 1-unit
       error, vs 1.59× with the exponential form.

    3. **KL-DRO tail aggregation (log-space)** — parameter-free.

       Instead of a plain mean over weighted shape losses, z-score
       log(cell_loss) and apply concave-compressed DRO weights:

           log_l = log(l + ε)
           z = (log_l − mean(log_l)) / std(log_l)
           dro_w = log1p(clamp(1+z, min=0))
           dro_w = dro_w / mean(dro_w)

       Compound weight and DRO are combined independently (product,
       normalised jointly to mean=1) — they address orthogonal concerns:
       compound steers *which cells matter*, DRO steers *how losses
       are aggregated* across the 90/10 peace/event split.

    4. **Windowed level anchor** — T-scaled Huber on per-series,
       per-window mean error, with DRO aggregation.

       The horizon is split into non-overlapping windows of width
       W = max(4, T // 6). Each window's mean error is penalised
       independently via Huber (with a separate adaptive delta):

           ē_w = mean(e[t_start:t_end])         per window w, series s
           l_{s,w} = huber(ē_w, δ_level)
           L_level = T · DRO_mean_{s,w}[ l_{s,w} ]

       Finer windows (T//6 → ~6 windows for T=36) than log_cosh variant
       to catch sub-window timing drift.

    5. **Spectral regularization** (optional, gated by δ_spectral > 0).
       Multi-resolution STFT magnitude comparison with DC bin masked.
       Phase-invariant; log_cosh on magnitude diffs (unchanged).

    ── Base cell loss: Huber with adaptive δ ────────────────────────────

    huber(x, δ) = 0.5x²         if |x| ≤ δ
                  δ(|x| − δ/2)   if |x| > δ

    Gradient = x for |x| ≤ δ, capped at ±δ beyond. Quadratic regime
    extends to δ = median + 2·MAD ≈ 90th percentile of |e|. The vast
    majority of cells get pure quadratic loss aligned with MSE/MSLE,
    while extreme outliers get linearly bounded gradients.

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
        >>> loss_fn = SpotlightLossHuber(delta=0.10, non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
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
                "SpotlightLossHuber: alpha is deprecated and ignored. "
                "Remove alpha from your config. (received alpha=%.4f)",
                alpha,
            )
        if delta < 0.0:
            raise ValueError(f"SpotlightLossHuber: delta must be non-negative, got {delta}")
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"SpotlightLossHuber: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.delta = delta
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "SpotlightLossHuber | delta=%.4f threshold=%.4f",
            delta, non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable. Used only for spectral loss."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    @staticmethod
    def _adaptive_huber(x: torch.Tensor, huber_delta: torch.Tensor) -> torch.Tensor:
        """Huber loss with a per-batch adaptive delta (scalar tensor)."""
        abs_x = torch.abs(x)
        quadratic = 0.5 * x * x
        linear = huber_delta * (abs_x - 0.5 * huber_delta)
        return torch.where(abs_x <= huber_delta, quadratic, linear)

    @staticmethod
    def _compute_adaptive_delta(
        abs_errors: torch.Tensor, min_delta: float = 0.5
    ) -> torch.Tensor:
        """Compute adaptive Huber delta from the error distribution.

        δ = median(|e|) + 2 × MAD(|e|), floored at min_delta.

        The median+2·MAD estimator is robust to outliers and places the
        quadratic-to-linear transition just past the bulk of errors.
        As training progresses and errors shrink, δ tightens.
        """
        flat = abs_errors.detach().flatten()
        median_val = flat.median()
        mad = (flat - median_val).abs().median()
        return torch.clamp(median_val + 2.0 * mad, min=min_delta)

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

        # ── Adaptive Huber delta ──────────────────────────────────────
        abs_e_shape = torch.abs(e_shape.detach())
        huber_delta = self._compute_adaptive_delta(abs_e_shape)

        # ── Base cell loss: Huber on demeaned error ───────────────────
        cell_loss = self._adaptive_huber(e_shape, huber_delta)

        # ── Adaptive compound weighting (difficulty-gated) ────────────
        # difficulty = |e| / (1 + |e|): slower saturation than 1-exp(-|e|).
        #   |e|=1 → 0.50,  |e|=3 → 0.75,  |e|=10 → 0.91
        # No event_mag: avoids double-stacking magnitude bias with DRO.
        # Threshold gate provides false-positive discipline.
        # w_compound = 1 + difficulty × 1[abs_max > τ] ∈ [1, 2)
        abs_y = torch.abs(y_true)
        abs_ypred_sg = torch.abs(y_pred.detach())

        difficulty = abs_e_shape / (1.0 + abs_e_shape)
        abs_max = torch.max(abs_y, abs_ypred_sg)
        above_threshold = (abs_max > self.non_zero_threshold).float()
        w_compound = 1.0 + difficulty * above_threshold

        # ── KL-DRO tail aggregation (log-space z-scores) ──────────────
        loss_flat = cell_loss.detach().flatten()
        log_loss = torch.log(loss_flat + 1e-8)
        log_std = log_loss.std()

        log_cv = torch.log1p(log_std / (log_loss.mean().abs() + 1e-8))
        dro_alpha = log_cv / (log_cv + 1.0)
        z = (log_loss - log_loss.mean()) / log_std.clamp(min=0.1)
        w_dro = torch.log1p((1.0 + z).clamp(min=0.0))
        w_dro = w_dro / w_dro.mean().clamp(min=1e-8)
        w_dro = w_dro.view_as(cell_loss)
        w_dro = dro_alpha * w_dro + (1.0 - dro_alpha)

        # Combine compound × DRO, normalise jointly to mean=1
        w_total = w_compound * w_dro
        w_total = w_total / w_total.mean()
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)

        loss_shape = (w_total * cell_loss).mean()

        # ── Windowed level anchor ──────────────────────────────────────
        # Finer windows than log_cosh variant: W = max(4, T // 6)
        # → ~6 windows for T=36 instead of 3. Better timing sensitivity.
        W = max(4, T // 6)
        e_windows = list(e.split(W, dim=1))
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e_windows], dim=1
        )

        # Adaptive delta for level anchor (separate from cell delta)
        level_huber_delta = self._compute_adaptive_delta(
            torch.abs(window_means.detach()), min_delta=0.1
        )
        level_losses = self._adaptive_huber(window_means, level_huber_delta)

        # Series×window DRO
        level_flat = level_losses.detach().flatten()
        log_level = torch.log(level_flat + 1e-8)
        level_log_std = log_level.std()
        level_log_cv = torch.log1p(
            level_log_std / (log_level.mean().abs() + 1e-8)
        )
        level_dro_alpha = level_log_cv / (level_log_cv + 1.0)

        level_z = (log_level - log_level.mean()) / level_log_std.clamp(min=0.1)
        w_level = torch.log1p((1.0 + level_z).clamp(min=0.0))
        w_level = w_level / w_level.mean().clamp(min=1e-8)
        w_level = level_dro_alpha * w_level + (1.0 - level_dro_alpha)
        w_level = torch.nan_to_num(w_level, nan=1.0, posinf=1.0, neginf=0.0)
        w_level = w_level.view_as(level_losses)

        loss_level = T * (w_level * level_losses).mean()

        # ── Spectral: AC bins only ────────────────────────────────────
        loss_spectral = y_pred.new_tensor(0.0)
        if self.delta > 0.0 and T >= 6:
            loss_spectral = self._spectral_loss(y_pred, y_true)

        total_loss = loss_shape + loss_level + self.delta * loss_spectral

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossHuber: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} spectral={loss_spectral.item():.6f}"
            )

        logger.debug(
            "SpotlightLossHuber | shape=%.6f level=%.6f spec=%.6f total=%.6f "
            "huber_delta=%.4f level_delta=%.4f",
            loss_shape.item(),
            loss_level.item(),
            loss_spectral.item(),
            total_loss.item(),
            huber_delta.item(),
            level_huber_delta.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLossHuber(delta={self.delta}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )