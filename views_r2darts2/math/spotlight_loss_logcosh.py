import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v46 — asinh + RevIN compatible, per-series DRO.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros, 10% spanning
    four orders of magnitude in raw deaths.

    ── Components ───────────────────────────────────────────────────────

    1. **DC/AC decomposition** — prevents RevIN DC offset amplification.
       e_shape = e − mean(e).  Shape gradient sums to zero per series.

    2. **Sigmoid event-magnitude weighting** — ~50:1 contrast ratio.
       event_mag = 0.01 + 0.99 × σ(5 × (abs_max − τ)).  Peace → ~0.02,
       conflict → ~1.0.  No model-state dependency.

    3. **Per-series temporal DRO** — within-series shock therapy.
       Z-scores log(cell_loss) along time axis per series.  Upweights
       proportionally harder timesteps *relative to that series*.

    4. **Windowed level anchor** — T-scaled log_cosh on per-window means.

    5. **Multi-resolution STFT loss** — always on, ungated.
       log_cosh on magnitude-spectrum differences.  DC bin masked.

    ── Base cell loss: log_cosh × (1 + log(1+|x|³))  (proportional) ───

    Args:
        non_zero_threshold: Sigmoid center (AsinhTransform: 0.88 ≈ asinh(1))

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
        """log_cosh with proportional sensitivity correction.

        log_cosh(x) × (1 + log(1 + |x|³)).

        For |x| < 1: ≈ 0.5x² (cubic interior shrinks faster toward zero,
            so noise cells are quieter than the old x² variant).
        For |x| > 2: ≈ |x| × 3·ln|x| (~50% steeper than x² formula).

        Gradient = tanh(x)·(1 + log1p(|x|³))
                   + log_cosh(x)·3x²·sign(x)/(1+|x|³).
        """
        abs_x = torch.abs(x)
        lc = abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)
        return lc * (1.0 + torch.log1p(abs_x * abs_x * abs_x))

    @staticmethod
    def _dro_weights_2d(losses: torch.Tensor) -> torch.Tensor:
        """Per-series temporal DRO weights along dim=1 for (B, T) input.

        Within each series, upweights timesteps that are proportionally
        harder relative to that series' own loss distribution.  This avoids
        cross-series interference while providing within-series "shock
        therapy" — each country gets pushed hardest at the timesteps where
        it's failing relative to its own baseline.

        Fully vectorised — no Python loop over the batch dimension.
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

    def _windowed_level_loss(self, e: torch.Tensor, T: int) -> torch.Tensor:
        """Windowed log_cosh level anchor.

        Splits the T-length error into non-overlapping windows of width
        max(6, T//3) (~3 wide windows), computes log_cosh on per-window
        means.  Scaled by T: necessary to overcome the 90% zero-cell
        majority pulling the DC component toward zero.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )
        level_losses = self._log_cosh_proportional(window_means)
        return T * level_losses.mean()

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

        # ── Base cell loss (proportional variant for MSLE sensitivity) ─
        cell_loss = self._log_cosh_proportional(e_shape)

        # ── Sigmoid event-magnitude weighting ─────────────────────────
        # Steep sigmoid: peace cells (abs_max ≈ 0) → ~0.01, moderate
        # events → ~0.5, high-conflict → ~1.0.  Gives ~50:1 contrast
        # ratio between Syria-class and zero cells.
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        event_mag = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.non_zero_threshold))

        # Difficulty: how wrong this cell currently is.  Gives up to 2×
        # boost on top of event_mag for cells the model is struggling with.
        difficulty = 1.0 - torch.exp(-torch.abs(e_shape.detach()))
        event_mag = event_mag * (1.0 + difficulty)

        # ── Per-series temporal DRO ────────────────────────────────────
        # Within each series, upweight the hardest timesteps relative to
        # that series' own loss distribution.  Between-series importance
        # is handled by event_mag above.
        w_dro = self._dro_weights_2d(cell_loss)  # (B, T)
        w_total = event_mag * w_dro
        w_total = w_total / w_total.mean()
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
        loss_shape = (w_total * cell_loss).mean()

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, T)

        # ── Multi-resolution spectral loss (always on) ──────────────
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