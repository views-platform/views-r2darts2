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
        cell_err = self._log_cosh(y_pred - y_true)
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
            window_err = self._log_cosh(e_win.mean(dim=2))

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
            cell_loss = self._log_cosh(log_mag_err)
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

        T = y_pred.size(1)

        loss_pw = self._pointwise_loss(y_pred, y_true)
        loss_level = self._level_loss(y_pred, y_true)

        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        total_loss = loss_pw + 0.2 * loss_level + 0.2 * loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: pw={loss_pw.item():.6f} "
                f"level={loss_level.item():.6f} spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | pw=%.6f level=%.6f spec=%.6f total=%.6f",
            loss_pw.item(), loss_level.item(),
            loss_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"