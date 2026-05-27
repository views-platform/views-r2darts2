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

    ── Components ───────────────────────────────────────────────────────

    1. **Full-error cell loss** — direct, non-DC-blind gradient.

       Loss operates on the raw error e = ŷ − y at every cell (no
       DC/AC split).  Earlier versions demeaned the error to defend
       against linear-RevIN's exponential Jensen amplification of DC
       bias.  Hybrid RevIN v3 has bounded Jacobian ∈ [1, σ_c] with
       no exponential amplification, making the defence obsolete —
       and actively harmful.

       Why demeaning was harmful:  the centering matrix J = I − 11'/T
       has zero column sums, so Σᵢ ∂ℒ/∂ŷᵢ = 0 ∀ series.  The shape
       loss could redistribute gradient across timesteps but could not
       change the total magnitude of any series' gradient signal.  For
       a 90%-zero series with 3 spikes, the network was told "push
       spikes up by X" while simultaneously being forced to "push 33
       peace cells down by X/33" — net signal to the bias neuron was
       exactly zero.  This is the structural cause of flat predictions.

       Using the full error restores per-series net gradient and lets
       the model actually increase its prediction variance.

    2. **Event-magnitude compound weighting** — parameter-free.

       Peace cells outnumber spike cells ~30:1.  Compound boosts event
       cells so spike gradients are not drowned by sheer peace-cell
       count: w_compound = 1 + 4 × event_mag, then mean-normalised per
       series.
       event_mag = r/(1+r) where r = log1p(max(|y|,|ŷ_sg|)/τ): log-
       damped continuous magnitude signal ∈ [0, 1).  Syria (500
       deaths) ≈ 0.685, Chad (2 deaths) ≈ 0.493.  Union semantics:
       false positives also receive event weighting.

       Difficulty term (1−exp(−|e|)) removed: the linear-proportional
       cell loss already provides error-magnitude curriculum
       (gradient ≈ 1 + |e|), making difficulty × event_mag redundant
       and risking positive feedback on hard cases.

    3. **Cross-series DRO** — parameter-free.

       Within-series DRO removed (would double-count the error
       curriculum already in cell_loss).  Cross-series DRO retained:
       countries with high aggregated loss (blunted spikes, large
       structural errors) get priority over already well-fit countries.

    4. **Windowed level anchor** — event-aware log_cosh_proportional on
       per-window mean error with hierarchical regret-DRO aggregation.

       Only mechanism that can shift per-series means (shape loss is
       structurally DC-blind).  Windows of width max(6, T//3) (~3 wide
       windows) catch intra-horizon level drift.  Event-aware within-
       series DRO ensures low/medium conflict windows are not treated as
       ordinary peace-window noise.

    5. **Temporal gradient matching** — log_cosh on first-difference
       errors (∂ŷ/∂t − ∂y/∂t).  Soft-weighted, fully data-driven.
       Primary anti-flat mechanism in the time domain: a flat prediction
       always has dy_pred/dt = 0, so every non-plateau in truth fires
       the loss proportionally.

    6. **Multi-resolution STFT loss** — log_cosh on log-magnitude
       differences ``log1p(mag_pred) − log1p(mag_true)`` at three
       (n_fft, hop) resolutions, AC bins only.  DC bin masked (level
       anchor handles DC).  Log-magnitude is unbounded for under-
       prediction (mag_pred→0 ⇒ loss grows with signal energy) and
       MSLE-aligned in the spectral domain.

       **No per-series demeaning** — this is critical.  With demean,
       gradients flow through the centering matrix J = I − 11'/T,
       making STFT structurally DC-blind (Σ ∂L/∂ŷ_i = 0).  Since
       shape loss is already DC-blind by design, demeaned STFT
       provided zero additional force for breaking flat equilibria.
       Without demean, STFT is the only component whose gradients
       can increase prediction variance (not constrained to zero-sum).
       DC leakage into adjacent bins cancels in the difference when
       pred and truth share similar mean (the observed scenario).

       Series weighted by continuous event magnitude; additive unit-
       coefficient contribution.

    ── Base cell loss: log_cosh × (1 + |x|)  (linear-proportional) ───

    Multiplies log_cosh by (1 + |x|) to restore MSE-like gradient
    growth for large errors while preserving log_cosh smoothness at
    the origin.  Previous (1 + log(1+|x|²)) gave sub-linear gradient
    growth (∂/∂e ≈ 1 + 2·ln|e|) that weakened spike-cell pull.  Linear
    correction: gradient ≈ 1 + |e| at large |e| — matches MSE's
    spike-friendly scaling without introducing MSE's instability on
    asinh-space outliers.
    For |x| << 1: ≈ 0.5 x².
    For |x| >> 1: ≈ |x|² (MSE-equivalent leading term).
    Gradient = tanh(x)·(1 + |x|) + log_cosh(x)·sign(x).

    ─────────────────────────────────────────────────────────────────────

    Args:
        non_zero_threshold: Transformed-space cutoff for compound
            weighting gate.
            - AsinhTransform: 0.88 ≈ asinh(1)
            - FourthRootTransform: 0.19 ≈ (1+1)^0.25 − 1

    Example:
        >>> loss_fn = SpotlightLossLogcosh(non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _TEMPORAL_GRADIENT = False
    _STFT = True

    def __init__(
        self,
        non_zero_threshold: float,
    ):
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.non_zero_threshold = non_zero_threshold
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    @staticmethod
    def _log_cosh_proportional(x: torch.Tensor) -> torch.Tensor:
        """log_cosh with linear-proportional sensitivity correction.

        ``log_cosh(x) × (1 + |x|)``.

        Design rationale (revised):
            Previous form used ``× (1 + log1p(|x|²))`` which gives
            sub-linear gradient growth (∂/∂e ≈ 1 + 2·ln|e| for |e|>>1).
            That weakened spike-cell gradients relative to MSE—a 90%-
            zero series with 3 spikes at |e|=5 produced only ~1.2× the
            cumulative gradient over the 33 peace-cell residuals, vs
            ~1.8× under MSE.  The model rationally allocates budget to
            the larger total signal, blunting spikes.

            Linear correction ``(1 + |x|)`` restores MSE-like growth
            (loss ≈ |x|², gradient ≈ 1 + |x| at large |x|) while
            preserving log_cosh smoothness at the origin (continuous
            derivative, no kink).  This gives spike cells proportionally
            stronger pull without introducing the optimisation
            instabilities of raw MSE on asinh-space outliers.

        For |x| << 1:  ≈ 0.5 x²
        For |x| >> 1:  ≈ |x|² (MSE-equivalent leading term)
        Gradient = tanh(x)·(1 + |x|) + log_cosh(x)·sign(x).
        """
        abs_x = torch.abs(x)
        lc = abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)
        return lc * (1.0 + abs_x)

    @staticmethod
    def _dro_weights(losses: torch.Tensor) -> torch.Tensor:
        """Log-space DRO weights, robust to bounded loss scales.

        Given a flat tensor of per-element losses, returns same-shaped
        normalised weights (mean ≈ 1).  Above-mean log-losses get
        weight > 1; below-mean get weight ≈ 1 (uniform floor, not
        suppressed).  ``alpha = tanh(std)`` softly blends toward
        uniform when the log-loss spread is small — no dependence on
        ``|mean|``, so bounded scores (regret ∈ [0,1]) behave correctly.
        """
        log_l = torch.log(losses.detach() + 1e-8)
        std = log_l.std()
        if not torch.isfinite(std) or std < 1e-8:
            return torch.ones_like(losses)
        alpha = torch.tanh(std)                              # ∈ [0, 1)
        z = (log_l - log_l.mean()) / std.clamp(min=1e-3)
        # Symmetric floor: weight ≥ 1 always; tail grows for z > 0
        w = 1.0 + torch.log1p(z.clamp(min=0.0))
        w = w / w.mean().clamp(min=1e-8)
        w = alpha * w + (1.0 - alpha)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    @staticmethod
    def _dro_weights_2d(losses: torch.Tensor) -> torch.Tensor:
        """Batched DRO weights along dim=1 for (B, T) input.

        Vectorised analogue of :meth:`_dro_weights` — one weight
        distribution per row.
        """
        log_l = torch.log(losses.detach() + 1e-8)            # (B, T)
        std = log_l.std(dim=1, keepdim=True)                  # (B, 1)
        std_safe = torch.where(
            torch.isfinite(std) & (std > 1e-8),
            std,
            losses.new_tensor(1e-3),
        )
        alpha = torch.tanh(std_safe)                          # (B, 1)
        mean = log_l.mean(dim=1, keepdim=True)                # (B, 1)
        z = (log_l - mean) / std_safe.clamp(min=1e-3)
        w = 1.0 + torch.log1p(z.clamp(min=0.0))
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)
        w = alpha * w + (1.0 - alpha)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _event_magnitude(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Continuous union eventness in transformed space, bounded [0, 1).

        ``event_mag = r / (1 + r)`` where ``r = log1p(|union| / τ)``.
        Log-damped: spreads dynamic range across the mid-conflict band
        instead of saturating at low-intensity events.
        Examples (τ = 0.88, asinh-space):
            asinh(500) ≈ 6.9 → r=2.18, event_mag≈0.685   (Syria)
            asinh(12)  ≈ 3.2 → r=1.50, event_mag≈0.600   (mid conflict)
            asinh(2)   ≈ 1.4 → r=0.97, event_mag≈0.493   (Chad)
            asinh(0)   = 0.0 → r=0.00, event_mag=0.000   (peace)
        Union semantics: false positives also receive event weighting.
        Prediction branch detached — no second gradient path.
        """
        abs_union = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        log_ratio = torch.log1p(abs_union / self.non_zero_threshold)
        return log_ratio / (1.0 + log_ratio)

    def _windowed_level_loss(
        self,
        e: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Event-weighted windowed level anchor with hierarchical DRO.

        Splits the T-length error into non-overlapping windows of width
        max(6, T//3) (~3 wide windows).

        **Event-weighted window mean** — the per-window error mean uses
        ``(1 + event_cell)`` weights so missed events are not diluted by
        surrounding peace cells.  Without this, a window with one
        missed event of asinh(12)≈3.2 averaged over 12 zeros becomes
        ~0.27 (small loss), while a false-positive flat prediction of
        asinh(1)=0.88 averaged over 12 cells becomes 0.88 (large loss).
        That asymmetry drives the model toward under-prediction.  The
        event weighting restores symmetric treatment of false positives
        and missed events.

        Hierarchical DRO:
          Level 1 (within-series): event-aware — windows containing
              conflict get DRO priority over peace-noise windows.
          Level 2 (cross-series): regret-based — prioritises countries
              proportionally underfit vs their own target signal, with
              baseline floor to prevent peace-country inflation.
        """
        W = max(6, T // 3)

        # Per-cell event weighting (detached: shapes weights, not grads)
        event_cell = self._event_magnitude(y_pred, y_true)
        cell_w = (1.0 + event_cell).detach()

        # Event-weighted per-window mean of error
        e_weighted = e * cell_w
        window_num = torch.stack(
            [ew.sum(dim=1) for ew in e_weighted.split(W, dim=1)], dim=1
        )
        window_den = torch.stack(
            [cw.sum(dim=1) for cw in cell_w.split(W, dim=1)], dim=1
        ).clamp(min=1e-6)
        window_means = window_num / window_den              # (B, n_windows)
        level_losses = self._log_cosh_proportional(window_means)

        # Per-window max event magnitude for DRO scoring
        window_event = torch.stack(
            [ew.amax(dim=1) for ew in event_cell.split(W, dim=1)], dim=1
        )

        # Within-series DRO score: event-aware window difficulty
        level_score = level_losses.detach() * (1.0 + window_event.detach())
        w_within = self._dro_weights_2d(level_score)        # (B, n_windows)
        series_level = (w_within * level_losses).mean(dim=1)  # (B,)

        # Cross-series regret DRO with baseline floor.  Baseline uses
        # the same event-weighted aggregation so it reflects the same
        # signal complexity the loss is penalising.
        y_true_w = y_true * cell_w
        target_num = torch.stack(
            [yw.sum(dim=1) for yw in y_true_w.split(W, dim=1)], dim=1
        )
        target_window_means = target_num / window_den
        level_baseline = self._log_cosh_proportional(target_window_means).mean(dim=1).detach()
        level_baseline = level_baseline.clamp(min=self.non_zero_threshold * 0.01)

        series_score = series_level.detach() / (
            series_level.detach() + level_baseline + 1e-8
        )
        w_series = self._dro_weights(series_score)          # (B,)
        w_series = w_series / w_series.mean().clamp(min=1e-8)

        return (w_series * series_level).mean()

    def _temporal_gradient_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Soft-weighted temporal gradient matching — fully data-driven.

        Replaces all binary thresholds (τ onset, _ESCALATION_FRAC) with a
        single continuous relevance weight derived from the proportional
        raw-space change magnitude:

            r = |Δy_raw| / max(midpoint(|y_raw_a|, |y_raw_b|), 1)
            w_rel = 1 − exp(−max(r_true, r_pred))

        Properties:
        • Plateau (Δy=0)           → w=0.  Silent, no smoothness pressure.
        • Mild escalation (18%)    → w≈0.17.  Proportional penalty.
        • Moderate escalation (27%) → w≈0.24.
        • Onset 0→1 death          → w≈0.63.  (raw_local clamps to 1.)
        • Major event 0→100        → w≈0.86.

        Inflection at ~100% raw change (doubling),
        which is a natural scale anchor.  Combined with log1p error-
        curriculum weighting: total_w = w_rel × (1 + log1p(|de|)).
        """
        _CLAMP = 10.0  # sinh(10) ≈ 11 013, numerical safety only

        # ── Proportional raw-space change magnitude ────────────────────
        def _rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            raw_a = torch.sinh(a.clamp(-_CLAMP, _CLAMP))
            raw_b = torch.sinh(b.clamp(-_CLAMP, _CLAMP))
            raw_mid = (raw_a.abs() + raw_b.abs()).mul(0.5).clamp(min=1.0)
            return (raw_b - raw_a).abs() / raw_mid

        rel_true = _rel(y_true[:, :-1], y_true[:, 1:])
        rel_pred = _rel(y_pred[:, :-1].detach(), y_pred[:, 1:].detach())

        # Soft relevance weight: 0 at plateaus, →1 for large changes.
        soft_w = 1.0 - torch.exp(-torch.max(rel_true, rel_pred))  # (B, T-1)

        if soft_w.sum() < 1e-8:
            return y_pred.new_tensor(0.0)

        # ── Temporal gradient error ────────────────────────────────────
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        de = dy_pred - dy_true

        cell_grad = self._log_cosh(de)

        # Combined weight: data-driven relevance × error curriculum.
        abs_de = torch.abs(de.detach())
        w = soft_w * (1.0 + torch.log1p(abs_de))
        denom = w.sum().clamp(min=1e-8)

        return (w * cell_grad).sum() / denom

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT log-magnitude loss (AC bins only).

        - **No per-series demeaning** — gradient flows directly through
          ŷ (identity Jacobian).  This is critical: with demean, the
          gradient goes through the centering matrix J = I − 11'/T,
          making STFT structurally DC-blind (Σ ∂L/∂ŷ_i = 0).  Since
          the shape loss is already DC-blind by design, demeaned STFT
          provided zero additional force for breaking flat equilibria.
          Without demean, STFT gradient CAN increase prediction
          variance — it's the only component not constrained to zero-
          sum gradients.
        - DC bin 0 is masked (level anchor handles DC).  Any DC leakage
          into bin 1 from Hann windowing cancels in the difference when
          pred and truth share similar mean (the observed scenario).
        - **Log-magnitude comparison** ``log1p(mag_pred) − log1p(mag_true)``
          is unbounded for under-prediction and MSLE-aligned.
        - Continuous event-magnitude weighting per series.
        - Safe magnitude sqrt(re²+im²+ε) bounds gradient at |z|→0.
        """
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        # Continuous series relevance: max event_mag across the series.
        # Bounded in [0, 1) — peace series get near-zero weight, conflict
        # series get up to ~0.7 weight. Smooth, no threshold cliff.
        abs_union = torch.max(torch.abs(true), torch.abs(pred.detach()))
        log_ratio = torch.log1p(abs_union / self.non_zero_threshold)
        series_event = (log_ratio / (1.0 + log_ratio)).amax(dim=1)  # (BC,)
        if series_event.sum() < 1e-8:
            return pred.new_tensor(0.0)

        T = pred.size(1)
        total = pred.new_tensor(0.0)
        n_valid = 0

        for n_fft, hop in self._SPECTRAL_RESOLUTIONS:
            if T < n_fft:
                continue
            window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            S_pred = torch.stft(
                pred, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )
            S_true = torch.stft(
                true, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )
            # Safe magnitude — bounded gradient at |z|→0
            mag_pred = torch.sqrt(S_pred.real ** 2 + S_pred.imag ** 2 + 1e-8)
            mag_true = S_true.abs()
            # Mask DC bin — level is handled by the level anchor
            mag_pred = mag_pred.clone()
            mag_true = mag_true.clone()
            mag_pred[:, 0, :] = 0.0
            mag_true[:, 0, :] = 0.0
            # Log-magnitude difference: unbounded penalty for flat preds,
            # naturally bounded for large magnitudes via log1p compression.
            log_mag_err = torch.log1p(mag_pred) - torch.log1p(mag_true)
            cell_loss = self._log_cosh(log_mag_err)
            # Continuous per-series weighting (peace series ≈ 0 weight)
            sw = series_event.view(-1, 1, 1)
            denom = (sw.sum() * cell_loss.size(1) * cell_loss.size(2)).clamp(min=1e-8)
            total = total + (sw * cell_loss).sum() / denom
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

        # ── Base cell loss on FULL error (no DC/AC split) ─────────────
        # Previous versions demeaned the error (e_shape = e − mean(e))
        # to prevent linear-RevIN's exponential Jensen amplification of
        # DC bias.  Hybrid RevIN v3 has bounded Jacobian ∈ [1, σ_c]
        # with NO exponential amplification, so the DC/AC split is no
        # longer protective — it is actively harmful.  Demeaned
        # gradients are routed through the centering matrix J = I −
        # 11'/T, which has zero column sums:
        #     Σ_i ∂ℒ_shape/∂ŷ_i = 0   ∀ series
        # The model cannot change the total magnitude of any series'
        # gradient signal.  For a 90%-zero series, the network is told
        # "push spike up by X" but simultaneously "push 33 peace cells
        # down by X/33 each" — net signal to the bias neuron is exactly
        # zero.  Using full e restores per-series net gradient and lets
        # the model actually increase its prediction variance.
        cell_loss = self._log_cosh_proportional(e)

        # ── Compound weighting (event-aware) ──────────────────────────
        # Peace cells outnumber spike cells ~30:1 even with full-error
        # loss.  Compound boosts event cells so spike gradients are not
        # drowned by sheer peace-cell count.  Difficulty term removed:
        # the new linear-proportional cell loss already provides error-
        # magnitude curriculum (gradient ~ 1+|e|), making difficulty ×
        # event_mag redundant and over-emphasising spikes that are
        # still poorly fit (positive feedback into hard cases).
        event_mag = self._event_magnitude(y_pred, y_true)
        w_compound = 1.0 + 4.0 * event_mag
        w_compound = w_compound / w_compound.mean(dim=1, keepdim=True).clamp(min=1e-8)
        series_loss = (w_compound * cell_loss).mean(dim=1)        # (B,)

        # ── Cross-series DRO ──────────────────────────────────────────
        # Within-series DRO removed: with full-error cell_loss and
        # event-magnitude weighting, per-cell budget allocation is
        # already correct.  Within-series DRO compounded on top would
        # double-count error curriculum and amplify above-mean residuals
        # the model is already attending to via (1+|e|) gradient growth.
        # Cross-series DRO retained: prioritises countries with high
        # remaining aggregated loss (blunted spikes, large structural
        # errors) over those already well-fit.
        w_series = self._dro_weights(series_loss)                  # (B,)
        w_series = w_series / w_series.mean().clamp(min=1e-8)
        loss_shape = (w_series * series_loss).mean()

        # ── Windowed level anchor (event-weighted) ────────────────────
        # Now overlaps with cell_loss in principle (both DC-aware), but
        # operates on smoothed window means — catches sustained level
        # drift over multiple timesteps that per-cell loss may under-
        # weight when individual residuals are small but biased.
        loss_level = self._windowed_level_loss(e, y_pred, y_true, T)

        # ── Multi-resolution spectral loss (secondary) ────────────────
        # Distributional matching in frequency domain.  No longer the
        # "only non-DC-blind term" — cell_loss now carries that role —
        # but still provides complementary signal for periodic structure
        # and helps the model learn correct spectral energy distribution.
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        total_loss = loss_shape + loss_level + loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f "
            "spec=%.6f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"