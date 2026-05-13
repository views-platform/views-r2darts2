import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossPowerLaw(torch.nn.Module):
    """
    SpotlightLoss (Power-Law) — asinh + RevIN compatible.

    Identical Spotlight architecture as SpotlightLossLogcosh (DC/AC,
    compound weighting, KL-DRO, windowed level anchor, spectral reg)
    but uses a pure power-law base cell loss:

        cell_loss = (e² + ε)^0.75 − ε^0.75,  p = 1.5

    Gradient ∝ e / (e² + ε)^0.25 ≈ sign(e) · |e|^0.5 for large |e|,
    giving scale-aware updates that distinguish 50-death from 500-death
    misses. Peace-cell gradient budget is controlled by compound
    weighting (w_compound = 1 for peace cells, ≤ 2 for event cells)
    and DRO z-scores — no separate loss branch needed.
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
                "SpotlightLossPowerLaw: alpha is deprecated and ignored. "
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
            "SpotlightLossPowerLaw (p=1.5) | delta=%.4f threshold=%.4f",
            delta, non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _P = 1.5
    _EPS = 1e-6
    _EPS_P2 = 1e-6 ** 0.75  # ε^(p/2), precomputed

    @staticmethod
    def _power_law(x: torch.Tensor) -> torch.Tensor:
        """Smoothed power-law loss: (x² + ε)^(p/2) − ε^(p/2), p=1.5."""
        return (x * x + 1e-6).pow(0.75) - 1e-6 ** 0.75

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable. Used for spectral loss only."""
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

        # ── Base cell loss: pure power law ───────────────────────────
        # Gradient ∝ |e|^0.5 for large |e| — distinguishes 50-death from
        # 500-death misses. Peace-cell budget controlled by compound
        # weighting and DRO; no separate loss branch required.
        cell_loss = self._power_law(e_shape)

        # ── Adaptive compound weighting (difficulty-gated) ────────────
        # difficulty = |e| / (1 + |e|): slower saturation than 1-exp(-|e|).
        #   |e|=1 → 0.50,  |e|=3 → 0.75,  |e|=10 → 0.91
        # No event_mag: avoids double-stacking magnitude bias with DRO.
        # Threshold gate provides false-positive discipline.
        # w_compound = 1 + difficulty × 1[abs_max > τ] ∈ [1, 2)
        abs_y = torch.abs(y_true)
        abs_ypred_sg = torch.abs(y_pred.detach())
        abs_max = torch.max(abs_y, abs_ypred_sg)
        above_threshold = (abs_max > self.non_zero_threshold).float()

        abs_e = torch.abs(e_shape.detach())
        difficulty = abs_e / (1.0 + abs_e)
        w_compound = 1.0 + difficulty * above_threshold

        # ── Sqrt-DRO tail aggregation (log-space z-scores) ─────────────
        # Z-score log(cell_loss) for proportional outlier detection.
        # Operates on raw cell_loss (before compound weighting).
        # Flattened cross-series: event cells dominate the tail.
        loss_flat = cell_loss.detach().flatten()
        log_loss = torch.log(loss_flat + 1e-8)
        log_std = log_loss.std()

        # CV-based alpha: self-normalizing, naturally bounded.
        # Clamp at 0.1: std<0.1 means losses span <1.1×, too little
        # variation for meaningful z-scores.
        if not torch.isfinite(log_std) or log_std < 1e-8:
            log_std = loss_flat.new_tensor(0.1)
        log_cv = torch.log1p(log_std / (log_loss.mean().abs() + 1e-8))
        dro_alpha = log_cv / (log_cv + 1.0)
        z = (log_loss - log_loss.mean()) / log_std.clamp(min=0.1)
        w_dro = torch.log1p((1.0 + z).clamp(min=0.0))
        w_dro = w_dro / w_dro.mean().clamp(min=1e-8)
        w_dro = w_dro.view_as(cell_loss)
        w_dro = dro_alpha * w_dro + (1.0 - dro_alpha)

        # Combine compound × DRO, normalise jointly to mean=1
        w_total = w_compound * w_dro
        w_total = w_total / w_total.mean().clamp(min=1e-8)
        # Safety: any residual NaN from degenerate batches → weight=1
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
        level_losses = self._power_law(window_means)

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
            f"SpotlightLossPowerLaw(delta={self.delta}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )