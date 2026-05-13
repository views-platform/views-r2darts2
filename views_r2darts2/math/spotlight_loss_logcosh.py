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

    2. **Adaptive compound weighting** — magnitude-proportional, parameter-free.

       Per-cell weight is the product of two bounded signals:

           difficulty  = 1 − exp(−|e_shape|)                      ∈ [0, 1)
           event_mag   = max(|y|, |ŷ_sg|) / (τ + max(|y|, |ŷ_sg|)) ∈ [0, 1)
           w_compound  = 1 + 2 × difficulty × event_mag            ∈ [1, 3)

       event_mag is a continuous magnitude signal replacing the binary
       event indicator. At threshold τ → 0.5, at 10×τ → 0.91. This
       aligns compound weighting with MSLE's proportional sensitivity:
       Syria (500 deaths, event_mag≈0.998) gets more weight than Chad
       (2 deaths, event_mag≈0.72). Binary event treated both identically.

       Gated: hard zero below τ restores false-positive discipline —
       sub-threshold cells get event_mag=0 → same w=1 as the original
       binary indicator. Above τ, smooth magnitude scaling applies.
       ŷ is stop-gradient. Self-correcting: as |e|→0, w→1.

    3. **Dual DRO tail aggregation (log-space, Tukey-capped)** — parameter-free.

       Two DRO passes operate at orthogonal granularities:

       **Cell-level DRO** — z-scores log(cell_loss) across all B×T cells,
       detects proportional outlier *cells* within the batch:

           z = (log_l − mean) / std
           z = clamp(z, max = Q75(z) + 1.5 × IQR(z))   ← Tukey fence
           dro_w = log1p(clamp(1+z, min=0)) / mean

       The Tukey fence is fully data-driven — no hardcoded scalar. With
       90/10 data, Q25 and Q75 both fall in the peace mass (z ≈ 0), so
       the fence sits at z ≈ 0.5–1.0. All event cells exceed it and
       receive equal cell-DRO weight; the extreme few cannot pull away
       from mid-range events. Differentiation *within* events is left
       to compound weighting.

       **Series-level DRO** — aggregates cell_loss to per-series means
       first, then z-scores across B series. Catches *systematically
       hard series* that cell DRO misses: a country with consistent
       moderate errors has no individual outlier cells, but its series
       mean sits in the event tail. Same Tukey fence applied.
       Broadcasts back to (B, T) before combining.

       Final shape weight = compound × cell_DRO × series_DRO,
       normalised jointly to mean=1. Each DRO operates independently:
       cell_DRO = which timesteps, series_DRO = which series.

       Soft activation α = log_cv/(log_cv + 1.0) blends each DRO toward
       uniform when log-loss variance is small (early training).

    4. **Windowed level anchor** — T-scaled log_cosh on per-series,
       per-window mean error, with DRO aggregation.

       The horizon is split into non-overlapping windows of width
       W = max(6, T // 3). Each window's mean error is penalised
       independently via log_cosh:

           ē_w = mean(e[t_start:t_end])         per window w, series s
           l_{s,w} = log_cosh(ē_w)
           L_level = T · DRO_mean_{s,w}[ l_{s,w} ]

       Window width is dynamic — computed from the actual output length,
       no hyperparameter. Floor of 6 ensures each window mean is
       statistically meaningful. Remainder timesteps form a final
       shorter window.

       Why windowed: a single full-mean anchor is blind to intra-horizon
       drift. A series overpredicted in months 1-12 and underpredicted
       in 25-36 can have ~zero full-mean error. MSLE sees each timestep
       independently and penalises both sub-periods. Windowed anchors
       catch this.

       DRO aggregation operates over all (series × window) level losses,
       upweighting whichever (series, window) pair has proportionally
       worst level error. Same log-space z-score + log1p compression
       as cell-level DRO.

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

        # ── Adaptive compound weighting (difficulty-gated) ────────────
        # difficulty = |e| / (1 + |e|): slower saturation than 1-exp(-|e|).
        #   |e|=1 → 0.50,  |e|=3 → 0.75,  |e|=10 → 0.91
        # No event_mag: avoids double-stacking magnitude bias with DRO.
        # Threshold gate provides false-positive discipline.
        # w_compound = 1 + difficulty × 1[abs_max > τ] ∈ [1, 2)
        abs_e = torch.abs(e_shape.detach())
        abs_y = torch.abs(y_true)
        abs_ypred_sg = torch.abs(y_pred.detach())

        difficulty = abs_e / (1.0 + abs_e)
        abs_max = torch.max(abs_y, abs_ypred_sg)
        above_threshold = (abs_max > self.non_zero_threshold).float()
        w_compound = 1.0 + difficulty * above_threshold

        # ── Cell-level DRO (log-space, Tukey-capped) ────────────────────
        # Z-score log(cell_loss) across all B×T cells.
        # Tukey fence caps z at Q75+1.5×IQR — With 90/10 data Q25/Q75 sit in the peace
        # mass (z≈0), so the fence lands at z≈0.5-1.0. All event cells
        # exceed it and are capped to equal cell-DRO weight, preventing
        # extreme events from pulling away from mid-range ones.
        loss_flat = cell_loss.detach().flatten()
        log_loss = torch.log(loss_flat + 1e-8)
        log_std = log_loss.std()
        if not torch.isfinite(log_std) or log_std < 1e-8:
            log_std = loss_flat.new_tensor(0.1)
        log_cv = torch.log1p(log_std / (log_loss.mean().abs() + 1e-8))
        dro_alpha = log_cv / (log_cv + 1.0)
        z_cell = (log_loss - log_loss.mean()) / log_std.clamp(min=0.1)
        z_q25 = torch.quantile(z_cell, 0.25)
        z_q75 = torch.quantile(z_cell, 0.75)
        z_cell = z_cell.clamp(max=z_q75 + 1.5 * (z_q75 - z_q25))
        w_cell_dro = torch.log1p((1.0 + z_cell).clamp(min=0.0))
        w_cell_dro = w_cell_dro / w_cell_dro.mean().clamp(min=1e-8)
        w_cell_dro = w_cell_dro.view_as(cell_loss)
        w_cell_dro = dro_alpha * w_cell_dro + (1.0 - dro_alpha)

        # ── Series-level DRO (per-series aggregated loss) ────────────────
        # Aggregate to per-series mean loss, then z-score over B series.
        # Catches systematically-hard series (consistent moderate error
        # across 36 timesteps) that cell DRO misses because no individual
        # cell is a proportional outlier. Same Tukey fence.
        series_loss = cell_loss.detach().mean(dim=1)          # (B,)
        log_series = torch.log(series_loss + 1e-8)
        series_log_std = log_series.std()
        if not torch.isfinite(series_log_std) or series_log_std < 1e-8:
            series_log_std = series_loss.new_tensor(0.1)
        series_log_cv = torch.log1p(
            series_log_std / (log_series.mean().abs() + 1e-8)
        )
        series_dro_alpha = series_log_cv / (series_log_cv + 1.0)
        z_series = (log_series - log_series.mean()) / series_log_std.clamp(min=0.1)
        sq25 = torch.quantile(z_series, 0.25)
        sq75 = torch.quantile(z_series, 0.75)
        z_series = z_series.clamp(max=sq75 + 1.5 * (sq75 - sq25))
        w_series_dro = torch.log1p((1.0 + z_series).clamp(min=0.0))
        w_series_dro = w_series_dro / w_series_dro.mean().clamp(min=1e-8)
        w_series_dro = series_dro_alpha * w_series_dro + (1.0 - series_dro_alpha)
        w_series_dro = torch.nan_to_num(w_series_dro, nan=1.0, posinf=1.0, neginf=0.0)
        w_series_dro = w_series_dro.unsqueeze(1).expand_as(cell_loss)  # (B, T)

        # Combine compound × cell_DRO × series_DRO, normalise jointly to mean=1
        w_total = w_compound * w_cell_dro * w_series_dro
        w_total = w_total / w_total.mean().clamp(min=1e-8)
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)

        loss_shape = (w_total * cell_loss).mean()

        # ── Windowed level anchor ──────────────────────────────────────
        # Only mechanism that can shift per-series means. Shape loss is
        # structurally DC-blind.
        #
        # Windowed: instead of a single full-horizon mean, split the T
        # timesteps into non-overlapping windows of width W and compute
        # log_cosh(ē_w) per window per series. This catches intra-horizon
        # level drift invisible to a single full-mean anchor — e.g. a
        # series overpredicted in months 1-12 and underpredicted in 25-36
        # would have ~zero full-mean error but large windowed error.
        #
        # Finer windows: W = max(4, T // 6) → ~6 windows for T=36.
        # Better timing sensitivity than 3 windows.
        W = max(4, T // 6)
        # Split e into windows: (B, T) → list of (B, W_i)
        e_windows = list(e.split(W, dim=1))  # last chunk may be < W
        # Per-window mean error: (B, n_windows)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e_windows], dim=1
        )  # (B, n_windows)
        level_losses = self._log_cosh(window_means)  # (B, n_windows)

        # Series×window DRO
        level_flat = level_losses.detach().flatten()
        log_level = torch.log(level_flat + 1e-8)
        level_log_std = log_level.std()
        if not torch.isfinite(level_log_std) or level_log_std < 1e-8:
            level_log_std = level_flat.new_tensor(0.1)
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