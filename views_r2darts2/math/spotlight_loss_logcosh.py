import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v44 — pointwise log_cosh + CCC shape enforcement.

    ── Components ───────────────────────────────────────────────────────

    1. Pointwise: log_cosh_proportional(ŷ − y) per cell, weighted by
       event_magnitude sigmoid mask (~58:1 event/peace). Per-cell
       accuracy signal: "each timestep should be at the right value."

    2. CCC (Concordance Correlation Coefficient): per-series shape loss.
       1 − CCC penalises flat predictions directly:
         - Flat → var(ŷ) = 0 → CCC = 0 → loss = 1.0 (maximum)
         - Perfect tracking → CCC = 1.0 → loss = 0
       Computed only over event-relevant series (series_event gate).
       Weighted by pointwise_loss.detach() so it always contributes
       at the same magnitude as pointwise — never drowned out.

    ── Why CCC breaks flatness ──────────────────────────────────────────

    CCC = 2·cov(ŷ,y) / (var(ŷ) + var(y) + (μ_ŷ − μ_y)²)

    Flat prediction: var(ŷ) = 0, cov = 0 → CCC = 0 regardless of level.
    The model MUST produce temporal variation correlated with truth to
    reduce CCC loss. This is orthogonal to pointwise which only says
    "be at the right level per cell" and is satisfied by flat-at-mean.

    ── Weighting ────────────────────────────────────────────────────────

    event_magnitude: 0.01 + 0.99 · σ(5·(|union| − τ) / τ)
    series_event gate: log1p ratio, gates CCC to event series only.
    CCC scale: multiplied by pw_loss.detach() for self-balancing.

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
        """Event-weighted log_cosh per cell. Primary training signal."""
        cell_err = self._log_cosh_proportional(y_pred - y_true)
        event_mag = self._event_magnitude(y_pred, y_true)
        # Weighted mean: event cells dominate via ~58:1 sigmoid ratio
        return (event_mag * cell_err).sum() / event_mag.sum().clamp(min=1e-8)

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
    # CCC (Concordance Correlation Coefficient) loss
    # ------------------------------------------------------------------

    def _ccc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Per-series CCC loss, gated to event-relevant series.

        CCC = 2·cov(ŷ,y) / (var(ŷ) + var(y) + (μ_ŷ − μ_y)²)
        Loss = 1 − CCC, bounded [0, 2].

        Flat prediction → var(ŷ)=0, cov=0 → CCC=0 → loss=1.
        Returns event-weighted mean (1 − CCC) across series.
        """
        # Flatten to (N_series, T)
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        # Series-level event gate (same logic as spectral)
        abs_union = torch.max(torch.abs(true), torch.abs(pred.detach()))
        log_ratio = torch.log1p(abs_union / self.non_zero_threshold)
        series_event = (log_ratio / (1.0 + log_ratio)).amax(dim=1)
        if series_event.sum() < 1e-8:
            return pred.new_tensor(0.0)

        # Per-series statistics
        mu_p = pred.mean(dim=1)
        mu_t = true.mean(dim=1)
        # Variance (unbiased not needed — relative magnitudes matter)
        var_p = pred.var(dim=1)
        var_t = true.var(dim=1)
        # Covariance
        cov_pt = ((pred - mu_p.unsqueeze(1)) * (true - mu_t.unsqueeze(1))).mean(dim=1)

        # CCC per series
        denom = var_p + var_t + (mu_p - mu_t) ** 2
        ccc = (2.0 * cov_pt) / denom.clamp(min=1e-8)
        # Clamp to [-1, 1] for numerical safety
        ccc = ccc.clamp(-1.0, 1.0)

        # Loss = 1 - CCC, weighted by series_event
        loss_per_series = 1.0 - ccc
        return (series_event * loss_per_series).sum() / series_event.sum().clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Spectral loss (DC-masked) — retained but unused in forward
    # ------------------------------------------------------------------

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT loss, DC bin masked. Series-weighted."""
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        # Series-level event relevance
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

        loss_pw = self._pointwise_loss(y_pred, y_true)

        # CCC shape loss, scaled by pointwise magnitude so it can never
        # be drowned out regardless of training stage or error scale.
        loss_ccc = self._ccc_loss(y_pred, y_true)
        loss_ccc_scaled = loss_ccc * loss_pw.detach()

        total_loss = loss_pw + loss_ccc_scaled

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: pw={loss_pw.item():.6f} "
                f"ccc={loss_ccc.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | pw=%.6f ccc=%.6f ccc_scaled=%.6f total=%.6f",
            loss_pw.item(), loss_ccc.item(), loss_ccc_scaled.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"