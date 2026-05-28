import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v42 — event-weighted log_cosh. Pointwise + level + STFT.

    Stripped version: no DRO, no inverse-frequency, no proportional.
    Plain log_cosh everywhere for direct MSLE alignment in asinh space.

    ── Components ───────────────────────────────────────────────────────

    1. Pointwise: log_cosh(ŷ − y) per cell, weighted by event_magnitude.
       Sigmoid mask gives ~58:1 ratio (event vs peace). Primary signal.

    2. Level (W=4, 12, 36): log_cosh(window_mean_error), weighted by
       max(event_mag) per window. Curriculum + boundary gradient leakage.
       Scaled by 0.2 — auxiliary, not dominant.

    3. STFT (DC-masked): log-magnitude spectral loss at 3 resolutions.
       Series-weighted by max event relevance. Scaled by 0.2.

    ── Weighting ────────────────────────────────────────────────────────

    Single mechanism: event_magnitude sigmoid mask.
      event_mag = 0.01 + 0.99 · σ(5·(|union| − τ) / τ)
    Provides sufficient event/peace discrimination without needing
    inverse-frequency (which was double-counting) or DRO (which was
    mostly inactive due to similar series scores).

    ── Eval alignment ───────────────────────────────────────────────────

    log_cosh in asinh space ≈ MSLE on raw for large values (asinh ≈ ln2x
    for x>2). For small values (x<1) asinh ≈ x → log_cosh ≈ x²/2 ≈ MSE.
    No reweighting distortion beyond event_magnitude mask.

    Args:
        non_zero_threshold: Event boundary in transformed space.
            AsinhTransform: 0.88 ≈ asinh(1).
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = True
    _LEVEL_WINDOWS = (4, 12, 36)

    def __init__(self, non_zero_threshold: float):
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"non_zero_threshold must be positive, got {non_zero_threshold}"
            )
        super().__init__()
        self.non_zero_threshold = non_zero_threshold
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable. Gradient saturates at ±1."""
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
    def _cell_dro_weights(losses: torch.Tensor) -> torch.Tensor:
        """Cell-level DRO with bounded amplification

        Returns per-element weights (mean=1, range ~[0.5, 2.0]).

        Triple-bounded against outlier fixation:
        1. log-compression of raw losses: a cell with loss=100 vs loss=1
           becomes log(100)=4.6 vs log(1)=0. Tail outliers don't dominate
           z-score statistics.
        2. tanh(relu(z)) bounds boost ∈ [0, 1]: even infinite z-score
           gives boost=1, so max weight = 2× mean. One catastrophic cell
           can take at most 2× share of gradient, never 10× or 100×.
        3. alpha=tanh(std) soft-blends toward uniform when log-loss
           spread is small. Early training (all cells equally bad) →
           uniform weights → no premature focus on noise.

        Effect: creates a self-correcting curriculum across cells.
        Cells with current biggest residual get up to 2× weight →
        model focuses there → those errors shrink → other cells become
        relatively worst → focus shifts. Drives away from flat solutions
        (flat has uniform error → uniform weights → no anti-flat pressure
        from DRO alone, but combined with event_mag this still works).
        """
        log_l = torch.log(losses.detach() + 1e-8)
        std = log_l.std()
        if not torch.isfinite(std) or std < 1e-6:
            return torch.ones_like(losses)
        z = (log_l - log_l.mean()) / std.clamp(min=1e-3)
        # tanh(relu(z)): max boost = 1 → max weight = 2
        boost = torch.tanh(F.relu(z))
        w = 1.0 + boost
        # alpha-blend: low log-std → mostly uniform
        alpha = torch.tanh(std)
        w = alpha * w + (1.0 - alpha)
        w = w / w.mean().clamp(min=1e-8)
        return torch.nan_to_num(w, nan=1.0, posinf=2.0, neginf=0.0)

    def _event_magnitude(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Sharp sigmoid event/peace mask with 1% floor.

        Union semantics (max of |y_true|, |y_pred.detach()|) so false
        positives also get gradient to push them down.
        """
        abs_union = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        return 0.01 + 0.99 * torch.sigmoid(
            5.0 * (abs_union - self.non_zero_threshold) / self.non_zero_threshold
        )

    # ------------------------------------------------------------------
    # Pointwise loss
    # ------------------------------------------------------------------

    def _pointwise_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Event-weighted log_cosh per cell with cell-level DRO.

        Weighting stack:
          event_mag (sigmoid mask): event/peace discrimination (~58:1)
          cell_dro (bounded): up to 2× boost for currently-worst cells
          combined → applied to log_cosh_proportional per-cell error

        DRO is computed on the RAW per-cell loss (before event_mag) so
        it sees true error magnitudes. Then event_mag gates the combined
        weight: peace cells with high DRO weight still get suppressed
        because event_mag ≈ 0.017.
        """
        cell_err = self._log_cosh_proportional(y_pred - y_true)
        event_mag = self._event_magnitude(y_pred, y_true)
        # Cell DRO on raw loss (bounded, max 2× boost per cell)
        dro_w = self._cell_dro_weights(cell_err)
        combined = event_mag * dro_w
        return (combined * cell_err).sum() / combined.sum().clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Level loss
    # ------------------------------------------------------------------

    def _level_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-scale windowed level loss at W=4, 12, 36.

        Provides smooth curriculum + boundary gradient leakage.
        Weighted by max(event_mag) per window.
        """
        T = y_pred.size(1)
        e = y_pred - y_true
        event_mag = self._event_magnitude(y_pred, y_true)

        all_losses = []
        all_weights = []

        for W in self._LEVEL_WINDOWS:
            if T < W:
                continue
            n_win = T // W
            e_win = e[:, :n_win * W].view(e.size(0), n_win, W)
            window_err = self._log_cosh_proportional(e_win.mean(dim=2))

            em_win = event_mag[:, :n_win * W].view(e.size(0), n_win, W)
            window_event = em_win.amax(dim=2)

            all_losses.append(window_err)
            all_weights.append(window_event)

        if not all_losses:
            return y_pred.new_tensor(0.0)

        cat_loss = torch.cat(all_losses, dim=1)
        cat_weight = torch.cat(all_weights, dim=1)

        return (cat_weight * cat_loss).sum() / cat_weight.sum().clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Spectral loss (DC-masked)
    # ------------------------------------------------------------------

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT loss, DC-masked. Cell-level DRO + series gate.

        Weighting stack:
          series_event: per-series gate (peace-only series ≈ 0). Same as
            before — prevents spectral matching for series with no signal.
          cell_dro (bounded): per (series, freq_bin, frame) boost up to
            2×. Targets specific frequency-time bins where prediction is
            spectrally worst, rather than weighting all bins of a series
            uniformly. "Focus on the frequency components you're missing."

        WHY cell DRO over per-bin matters:
        Without it: model satisfies STFT by matching average spectrum
        across all frames. With it: bins where model is most wrong
        (e.g., missing energy at 6-month cycle, frame 2) get up to 2×
        gradient → forces matching at the SPECIFIC time-frequency
        locations of error.
        """
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        # Series-level event relevance (gate: peace series contribute ≈0)
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
            mag_pred = torch.sqrt(S_pred.real ** 2 + S_pred.imag ** 2 + 1e-8)
            mag_true = torch.sqrt(S_true.real ** 2 + S_true.imag ** 2 + 1e-8)

            log_mag_err = torch.log1p(mag_pred) - torch.log1p(mag_true)
            cell_loss = self._log_cosh_proportional(log_mag_err)
            # DC mask: skip bin 0
            cell_loss = cell_loss[:, 1:, :]

            # Per-cell DRO across (series, freq_bin, frame)
            dro_w = self._cell_dro_weights(cell_loss)
            # Series gate (broadcasts across freq, frame)
            sw = series_event.view(-1, 1, 1)
            combined = sw * dro_w

            total = total + (combined * cell_loss).sum() / combined.sum().clamp(min=1e-8)
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

        loss_pw = self._pointwise_loss(y_pred, y_true)
        # loss_level = self._level_loss(y_pred, y_true)

        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        total_loss = loss_pw + loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: pw={loss_pw.item():.6f} "
            )

        logger.debug(
            "SpotlightLossLogcosh | pw=%.6f spec=%.6f total=%.6f",
            loss_pw.item(), loss_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"