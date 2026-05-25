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
        """Log-space KL-DRO weights with soft alpha-blend.

        Given a flat tensor of per-element losses, returns a same-shaped
        tensor of normalised weights (mean ≈ 1).  High-loss elements get
        upweighted proportionally in log-space; soft alpha blends toward
        uniform when log-loss variance is small (early training).
        """
        log_l = torch.log(losses.detach() + 1e-8)
        std = log_l.std()
        if not torch.isfinite(std) or std < 1e-8:
            std = losses.new_tensor(0.1)
        cv = torch.log1p(std / (log_l.mean().abs() + 1e-8))
        alpha = cv / (cv + 1.0)
        z = (log_l - log_l.mean()) / std.clamp(min=0.1)
        w = torch.log1p((1.0 + z).clamp(min=0.0))
        w = w / w.mean().clamp(min=1e-8)
        w = alpha * w + (1.0 - alpha)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    @staticmethod
    def _dro_weights_2d(losses: torch.Tensor) -> torch.Tensor:
        """Batched DRO weights along dim=1 for (B, T) input.

        Equivalent to stacking _dro_weights per row, but fully
        vectorised — no Python loop over the batch dimension.
        """
        log_l = torch.log(losses.detach() + 1e-8)           # (B, T)
        std = log_l.std(dim=1, keepdim=True)                 # (B, 1)
        std = torch.where(
            torch.isfinite(std) & (std > 1e-8),
            std,
            losses.new_tensor(0.1),
        )
        mean = log_l.mean(dim=1, keepdim=True)               # (B, 1)
        cv = torch.log1p(std / (mean.abs() + 1e-8))
        alpha = cv / (cv + 1.0)
        z = (log_l - mean) / std.clamp(min=0.1)
        w = torch.log1p((1.0 + z).clamp(min=0.0))
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)
        w = alpha * w + (1.0 - alpha)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _event_magnitude(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Continuous union eventness in transformed space.

        A cell is event-relevant if conflict appears in either y_true or
        y_pred.  The prediction branch is detached so false positives can
        receive corrective weighting without creating a second gradient
        path through the weights.
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
        """Windowed log_cosh level anchor with event-aware hierarchical DRO.

        Splits the T-length error into non-overlapping windows of width
        max(6, T//3) (~3 wide windows).  Two-level DRO:
          Level 1 (within-series): event-aware — windows containing
              conflict get DRO priority over peace-noise windows.
          Level 2 (cross-series): regret-based — prioritises countries
              proportionally underfit vs their own target signal, with
              baseline floor to prevent peace-country inflation.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )
        # (B, n_windows)
        level_losses = self._log_cosh_proportional(window_means)

        # Event-awareness: per-window max event magnitude
        event_cell = self._event_magnitude(y_pred, y_true)
        window_event = torch.stack(
            [ew.amax(dim=1) for ew in event_cell.split(W, dim=1)], dim=1
        )

        # Within-series DRO score: event-aware window difficulty
        level_score = level_losses.detach() * (1.0 + window_event.detach())
        w_within = self._dro_weights_2d(level_score)  # (B, n_windows)
        series_level = (w_within * level_losses).mean(dim=1)  # (B,)

        # Cross-series regret DRO with baseline floor
        target_window_means = torch.stack(
            [yw.mean(dim=1) for yw in y_true.split(W, dim=1)], dim=1
        )
        level_baseline = self._log_cosh_proportional(target_window_means).mean(dim=1).detach()
        level_baseline = level_baseline.clamp(min=self.non_zero_threshold * 0.01)

        series_score = series_level.detach() / (
            series_level.detach() + level_baseline + 1e-8
        )
        w_series = self._dro_weights(series_score)  # (B,)
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
        """Multi-resolution STFT magnitude comparison (AC bins only).

        Safe magnitude sqrt(re² + im² + ε) avoids gradient blowup at
        |z|→0.  DC bin is masked — level anchor already handles DC.
        Only series with signal above threshold are included.
        """
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
            # Per-series RMS normalization: makes spectral loss measure
            # relative shape mismatch, invariant to signal magnitude.
            rms = mag_true.norm(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            total = total + self._log_cosh((mag_pred - mag_true) / rms).mean()
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

        # ── Base cell loss (proportional variant for MSLE sensitivity) ─
        cell_loss = self._log_cosh_proportional(e_shape)

        # ── Compound weighting ────────────────────────────────────────
        abs_e = torch.abs(e_shape.detach())
        difficulty = 1.0 - torch.exp(-abs_e)
        event_mag = self._event_magnitude(y_pred, y_true)
        w_compound = 1.0 + 4.0 * difficulty * event_mag

        # ── Hierarchical shape DRO ───────────────────────────────────────
        # Level 1: within-series DRO (which timesteps for THIS country)
        w_dro_within = self._dro_weights_2d(cell_loss)            # (B, T)
        w_within = w_compound * w_dro_within
        w_within = w_within / w_within.mean(dim=1, keepdim=True).clamp(min=1e-8)
        series_loss = (w_within * cell_loss).mean(dim=1)          # (B,)

        # Level 2: cross-series regret DRO — decouple compound from
        # the ranking path to prevent high-intensity feedback loop.
        # Score uses DRO-only weights (no compound amplification).
        w_score_within = w_dro_within / w_dro_within.mean(
            dim=1, keepdim=True
        ).clamp(min=1e-8)
        series_score_loss = (w_score_within * cell_loss).mean(dim=1).detach()

        # Regret: how underfit is this country relative to its own
        # target signal complexity?  Baseline floor prevents peace
        # countries (baseline≈0) from getting infinite regret.
        target_shape = y_true - y_true.mean(dim=1, keepdim=True)
        shape_baseline = self._log_cosh_proportional(target_shape).mean(dim=1).detach()
        shape_baseline = shape_baseline.clamp(min=self.non_zero_threshold * 0.01)

        regret = series_score_loss / (series_score_loss + shape_baseline + 1e-8)
        w_series = self._dro_weights(regret)                       # (B,)
        w_series = w_series / w_series.mean().clamp(min=1e-8)
        loss_shape = (w_series * series_loss).mean()

        # ── Windowed level anchor (event-aware hierarchical DRO) ─────
        loss_level = self._windowed_level_loss(e, y_pred, y_true, T)

        # ── Temporal gradient matching ──────────────────────────────
        loss_grad = y_pred.new_tensor(0.0)
        if self._TEMPORAL_GRADIENT and T >= 2:
            loss_grad = self._temporal_gradient_loss(y_pred, y_true)

        # ── Multi-resolution spectral loss (shape-budgeted) ─────────
        loss_spec = y_pred.new_tensor(0.0)
        spec_coeff = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)
            # Dynamic budget: STFT gets more weight as shape converges,
            # shrinks when STFT would overwhelm shape. Stateless.
            spec_coeff = loss_shape.detach() / (
                loss_shape.detach() + loss_spec.detach() + 1e-8
            )

        total_loss = loss_shape + loss_level + loss_grad + spec_coeff * loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} grad={loss_grad.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f grad=%.6f "
            "spec=%.6f spec_coeff=%.4f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_grad.item(), loss_spec.item(),
            spec_coeff.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"