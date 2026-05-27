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

       difficulty = 1 − exp(−|e|): how wrong this cell is (curriculum).
       event_mag = max(|y|, |ŷ_sg|) / (τ + max(|y|, |ŷ_sg|)): continuous
       magnitude signal ∈ [0, 1).  Syria (500 deaths) gets ~0.998,
       Chad (2 deaths) ~0.72.  Union semantics: false positives get
       event_mag > 0.5.  w_compound = 1 + 4 × difficulty × event_mag
       ∈ [1, 5).  Self-correcting: as |e|→0, w→1.

    3. **Hierarchical regret-DRO aggregation (log-space)** — parameter-free.

       Within each country, DRO selects difficult timesteps/windows.
       Across countries, DRO is applied to relative regret versus each
       country's own target signal, so high-intensity countries do not
       dominate solely by absolute magnitude.  Baseline floor prevents
       peace countries (zero target) from inflating regret.

    4. **Windowed level anchor** — event-aware log_cosh_proportional on
       per-window mean error with hierarchical regret-DRO aggregation.

       Only mechanism that can shift per-series means (shape loss is
       structurally DC-blind).  Windows of width max(6, T//3) (~3 wide
       windows) catch intra-horizon level drift.  Event-aware within-
       series DRO ensures low/medium conflict windows are not treated as
       ordinary peace-window noise.

    5. **Temporal gradient matching** — log_cosh on first-difference
       errors (∂ŷ/∂t − ∂y/∂t).  Soft-weighted, fully data-driven.

       Continuous relevance weight w = 1 − exp(−r) where
       r = |Δy_raw|/max(midpoint_raw, 1) is the proportional raw-space
       change magnitude (via sinh inversion).  Plateau transitions get
       w≈0 naturally; onset/offset ~0.63; doublings ~0.63; major events
       ~0.86.  No hardcoded thresholds — the raw-space proportional
       change is the only signal.  Combined with log1p error-curriculum.
       O(T) computation.

    6. **Multi-resolution STFT loss** — log_cosh on magnitude-spectrum
       differences at three (n_fft, hop) resolutions, AC bins only.
       DC bin masked (level anchor handles DC).  Safe magnitude
       sqrt(re²+im²+ε) avoids gradient blowup at |z|→0.  Only series
       with signal above τ are included.  Per-series RMS-normalized so
       contribution is invariant to signal magnitude.  Gradient budget
       is tied dynamically to shape loss.

    ── Base cell loss: log_cosh × (1 + log(1+|x|²))  (proportional) ───

    Multiplies log_cosh by (1 + log(1+|x|²)) to restore proportional
    sensitivity for large errors without letting extreme-country errors
    dominate low/medium-intensity conflict.  Asinh-space errors are
    already approximately log-ratio errors, so the squared multiplier is
    MSLE-aligned and milder than the earlier cubic variant.
    For |x|<1: ≈0.5x² with mild proportional correction.
    For |x|>2: ≈|x|·2·ln|x|.
    Gradient = tanh(x)·(1+log1p(|x|²)) + log_cosh(x)·2x/(1+|x|²).

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
        """log_cosh with proportional sensitivity correction.

        log_cosh(x) × (1 + log(1 + |x|²)).

        For |x| < 1: ≈ 0.5x² with mild proportional correction.
        For |x| > 2: ≈ |x| × 2·ln|x|.  Asinh-space errors are already
            approximately log-ratio errors, so this reinforces MSLE
            sensitivity without letting extreme countries monopolise
            gradients.

        Gradient = tanh(x)·(1 + log1p(|x|²))
                   + log_cosh(x)·2x·sign(x)/(1+|x|²).
        """
        abs_x = torch.abs(x)
        lc = abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)
        return lc * (1.0 + torch.log1p(abs_x * abs_x))

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

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only).

        - Per-series demeaning removes DC before transform so spectral
          loss does not redundantly penalise level offsets (handled by
          level anchor).
        - Symmetric RMS normalization ``max(rms_true, rms_pred)`` makes
          loss invariant to magnitude in BOTH directions — false
          positives on flat series no longer dominate.
        - Continuous event-magnitude weighting per series replaces the
          hard threshold gate, eliminating discontinuities for
          borderline countries.
        - DC bin masked; safe magnitude sqrt(re²+im²+ε) bounds gradient
          at |z|→0.
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

        # Per-series demean — kills DC contamination in low-freq AC bins
        pred = pred - pred.mean(dim=1, keepdim=True)
        true = true - true.mean(dim=1, keepdim=True)

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
            # Symmetric per-series RMS: bounds loss in both directions
            rms = torch.max(
                mag_true.norm(dim=(1, 2), keepdim=True),
                mag_pred.detach().norm(dim=(1, 2), keepdim=True),
            ).clamp(min=1e-6)
            cell_loss = self._log_cosh((mag_pred - mag_true) / rms)
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
        # The DC/AC split (e_shape = e − mean(e)) forces Σᵢ ∂L/∂ŷᵢ = 0
        # per series — the model cannot increase overall prediction
        # variance, producing flat forecasts.  Full error restores
        # non-zero net gradient per series.
        #
        # Sub-linear proportional correction (1 + log(1+|x|²)) is kept:
        # it damps gradient growth for extreme errors, preventing event
        # cells (|e|=3–5) from dominating peace cells (|e|≈0) in the
        # aggregate gradient.  This controls overprediction naturally —
        # the model can't just raise its baseline to cheaply reduce
        # event-cell loss without paying proportional penalty on peace.
        cell_loss = self._log_cosh_proportional(e)

        # ── Compound weighting (event-aware, mild) ────────────────────
        # With full error and sub-linear proportional, the gradient
        # already naturally attends to event cells (they have larger
        # |e|).  Compound only needs to prevent the 30:1 peace/event
        # COUNT ratio from completely diluting event signals.
        # event_mag alone (no difficulty, no multiplier) gives mild
        # emphasis: w ∈ [1.0, ~1.7) before normalization.
        event_mag = self._event_magnitude(y_pred, y_true)
        w_compound = 1.0 + event_mag
        w_compound = w_compound / w_compound.mean(dim=1, keepdim=True).clamp(min=1e-8)
        series_loss = (w_compound * cell_loss).mean(dim=1)        # (B,)

        # ── Cross-series DRO (regret-based) ───────────────────────────
        # Prioritises countries with high remaining loss relative to
        # their own target complexity.  Uses series_loss directly (no
        # separate score path needed since cell_loss is already on full
        # error).
        shape_baseline = self._log_cosh_proportional(
            y_true - y_true.mean(dim=1, keepdim=True)
        ).mean(dim=1).detach()
        shape_baseline = shape_baseline.clamp(min=self.non_zero_threshold * 0.01)

        regret = series_loss.detach() / (
            series_loss.detach() + shape_baseline + 1e-8
        )
        w_series = self._dro_weights(regret)                       # (B,)
        w_series = w_series / w_series.mean().clamp(min=1e-8)
        loss_shape = (w_series * series_loss).mean()

        # ── Windowed level anchor (event-weighted) ────────────────────
        loss_level = self._windowed_level_loss(e, y_pred, y_true, T)

        # ── Multi-resolution spectral loss (shape-budgeted) ──────────
        loss_spec = y_pred.new_tensor(0.0)
        spec_coeff = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)
            spec_coeff = loss_shape.detach() / (
                loss_shape.detach() + loss_spec.detach() + 1e-8
            )

        total_loss = loss_shape + loss_level + spec_coeff * loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f "
            "spec=%.6f spec_coeff=%.4f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_spec.item(),
            spec_coeff.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"