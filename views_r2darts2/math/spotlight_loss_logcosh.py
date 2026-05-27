import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v37 — full-error + inverse-frequency + log-mag STFT.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros, 10% spanning
    four orders of magnitude in raw deaths.

    ── Architecture ─────────────────────────────────────────────────────

    Three components, no hardcoded constants.  All weights are derived
    from per-batch data statistics.

    1. **Full-error cell loss** — `log_cosh_proportional(e)` on the raw
       error `e = ŷ − y` at every (series, timestep) cell.

       No DC/AC decomposition.  Earlier versions demeaned the error to
       prevent RevIN DC offset amplification.  Hybrid RevIN v3 has
       bounded Jacobian ∈ [1, σ_c] with no exponential amplification,
       making that defence obsolete.  The DC/AC split forces
       Σᵢ ∂L/∂ŷᵢ = 0 per series — the model CANNOT increase prediction
       variance.  Flat predictions receive zero net gradient.  Removing
       the split restores non-zero net gradient per series and allows
       the model to learn levels directly from cell_loss.

    2. **Inverse-frequency compound weighting** — parameter-free.

       Equalises total gradient mass between event and peace cells
       within each series, regardless of class imbalance.

       Derivation: given event_mag ∈ [0, 1) per cell, let
       p = mean(event_mag) be the per-series event density.  Then:
           w_i = event_mag_i / p + (1 − event_mag_i) / (1 − p)
       This produces:
           Σ_event w_i ≈ Σ_peace w_i   (equalized mass)
       No hardcoded multiplier.  Automatically adapts to series with
       many events (Sudan: p≈0.4) vs few (Luxembourg: p≈0.01).
       Falls back to uniform (w=2) for series with p→0 or p→1.

    3. **Log-magnitude multi-resolution STFT** — the only
       non-pointwise signal.

       `log_cosh(log1p(mag_pred) − log1p(mag_true))` at three
       spectral resolutions.  Properties:
       - Unbounded penalty for flat predictions (log1p(0)≈0 vs
         log1p(mag_true) which grows with signal energy).
       - Monotonically ordered: perfect < shaped < flat.
       - MSLE-aligned in spectrogram domain.
       - No demean, no DC mask, no RMS normalization.  Gradient flows
         through identity Jacobian — no zero-sum constraint.
       - Breaks the symmetric-pointwise-loss trap: spectral magnitude
         is a distributional match (Parseval), so the model is penalised
         for lacking energy at the right frequencies even when pointwise
         timing is uncertain.
       - Event-magnitude series weighting (continuous, no threshold).

    ── Cross-series DRO (additive regret) ───────────────────────────────

    Regret = relu(series_loss − baseline).  Baseline is the irreducible
    loss on each country's own target shape:
        baseline_i = log_cosh_proportional(y_i − mean(y_i))
    Peace countries have baseline ≈ 0 AND series_loss ≈ 0 → regret ≈ 0.
    Conflict countries with high remaining error get regret > 0 and
    receive DRO upweighting.  No baseline floor needed — additive regret
    naturally handles the zero-target case.

    ── Base cell loss: log_cosh × (1 + log(1+|x|²))  (proportional) ───

    Sub-linear proportional correction.  For |x| > 2, gradient grows as
    ≈ 1 + 2·ln|x| — fast enough to attend to large errors but
    sub-linear enough to prevent extreme-country errors from monopolising
    the aggregate gradient (which would cause overprediction via baseline
    drift).

    ─────────────────────────────────────────────────────────────────────

    Args:
        non_zero_threshold: Transformed-space cutoff for event_magnitude.
            - AsinhTransform: 0.88 ≈ asinh(1)

    Example:
        >>> loss_fn = SpotlightLossLogcosh(non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
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
        """log_cosh(x) × (1 + log(1 + |x|²)).

        Sub-linear proportional correction.  Gradient growth at large
        |x| is ≈ 1 + 2·ln|x|.  Reinforces MSLE sensitivity without
        letting extreme errors monopolise aggregate gradient.
        """
        abs_x = torch.abs(x)
        lc = abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)
        return lc * (1.0 + torch.log1p(abs_x * abs_x))

    @staticmethod
    def _dro_weights(scores: torch.Tensor) -> torch.Tensor:
        """Log-space DRO weights from non-negative scores.

        Returns normalised weights (mean ≈ 1) where above-mean
        log-scores get weight > 1.  `alpha = tanh(std)` blends toward
        uniform when spread is small.
        """
        s = scores.detach()
        s = s - s.min() + 1e-8
        log_s = torch.log(s)
        std = log_s.std()
        if not torch.isfinite(std) or std < 1e-8:
            return torch.ones_like(scores)
        alpha = torch.tanh(std)
        z = (log_s - log_s.mean()) / std.clamp(min=1e-3)
        w = 1.0 + torch.log1p(z.clamp(min=0.0))
        w = w / w.mean().clamp(min=1e-8)
        w = alpha * w + (1.0 - alpha)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _event_magnitude(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Continuous union eventness in transformed space, bounded [0, 1).

        ``event_mag = r / (1 + r)`` where ``r = log1p(|union| / τ)``.
        Union semantics: false positives also receive event weighting.
        Prediction branch detached — no second gradient path.
        """
        abs_union = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        log_ratio = torch.log1p(abs_union / self.non_zero_threshold)
        return log_ratio / (1.0 + log_ratio)

    # ------------------------------------------------------------------
    # Spectral loss
    # ------------------------------------------------------------------

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Log-magnitude multi-resolution STFT loss.

        `log_cosh(log1p(mag_pred) − log1p(mag_true))` — unbounded
        penalty for flat predictions, MSLE-aligned, no demean, no DC
        mask, no RMS normalization.  Gradient flows through identity
        Jacobian (no zero-sum constraint).  Event-magnitude weighting
        per series.
        """
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        # Per-series event relevance (max event_mag over time)
        abs_union = torch.max(torch.abs(true), torch.abs(pred.detach()))
        log_ratio = torch.log1p(abs_union / self.non_zero_threshold)
        series_event = (log_ratio / (1.0 + log_ratio)).amax(dim=1)
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
            mag_true = torch.sqrt(S_true.real ** 2 + S_true.imag ** 2 + 1e-8)

            # Log-magnitude difference: MSLE-aligned, unbounded for
            # under-energy predictions (flat → log1p(ε) vs log1p(large))
            log_mag_err = torch.log1p(mag_pred) - torch.log1p(mag_true)
            cell_loss = self._log_cosh(log_mag_err)

            # Event-weighted series aggregation
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

        # ── Base cell loss on FULL error ──────────────────────────────
        cell_loss = self._log_cosh_proportional(e)

        # ── Inverse-frequency compound weighting ──────────────────────
        # Equalises total gradient mass between event and peace cells.
        # Let p = mean(event_mag) per series.  Then:
        #   w_i = event_mag_i / p + (1 − event_mag_i) / (1 − p)
        # This ensures Σ_event w ≈ Σ_peace w regardless of count
        # imbalance.  No hardcoded multipliers.
        event_mag = self._event_magnitude(y_pred, y_true)
        p_series = event_mag.mean(dim=1, keepdim=True).clamp(min=1e-4, max=1.0 - 1e-4)
        w_compound = event_mag / p_series + (1.0 - event_mag) / (1.0 - p_series)
        # Normalize to mean=1 per series (preserves total loss scale)
        w_compound = w_compound / w_compound.mean(dim=1, keepdim=True).clamp(min=1e-8)
        series_loss = (w_compound * cell_loss).mean(dim=1)        # (B,)

        # ── Cross-series DRO (additive regret) ────────────────────────
        # Baseline: irreducible loss on target shape (cannot do better
        # than reproducing the target's own variability around its mean).
        baseline = self._log_cosh_proportional(
            y_true - y_true.mean(dim=1, keepdim=True)
        ).mean(dim=1).detach()
        # Regret: excess loss above baseline.  Peace countries have
        # series_loss ≈ 0 and baseline ≈ 0 → regret ≈ 0.  Conflict
        # countries with structural errors get regret > 0.
        regret = torch.relu(series_loss.detach() - baseline)
        w_series = self._dro_weights(regret)                       # (B,)
        w_series = w_series / w_series.mean().clamp(min=1e-8)
        loss_cell = (w_series * series_loss).mean()

        # ── Log-magnitude multi-resolution STFT ───────────────────────
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        total_loss = loss_cell + loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: cell={loss_cell.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | cell=%.6f spec=%.6f total=%.6f",
            loss_cell.item(), loss_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"