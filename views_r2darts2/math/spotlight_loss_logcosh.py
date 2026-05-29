import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v45 — pointwise accuracy + first-difference dynamics.

    ── Design ───────────────────────────────────────────────────────────

    L = L_pointwise + L_dynamics

    Both terms use identical machinery: log_cosh_proportional per cell,
    weighted by event_magnitude sigmoid mask. This makes them naturally
    commensurable — no scaling coefficient needed.

    ── Components ───────────────────────────────────────────────────────

    1. Pointwise (L_pointwise):
       log_cosh_proportional(ŷ − y), weighted by event_mag.
       "Each cell should be at the right value."

    2. First-difference dynamics (L_dynamics):
       log_cosh_proportional(Δŷ − Δy), weighted by event_mag on Δ.
       "Each transition should match the true transition."

       Anti-flatness guarantee: flat prediction → Δŷ ≡ 0 → loss =
       log_cosh_proportional(Δy) at every onset/offset. Gradient =
       tanh(Δy)·(1+log1p(Δy²)) ≈ 0.96 at event onsets. This is a
       PERMANENT, per-cell, per-series gradient that never vanishes
       regardless of pointwise error magnitude or series variance.

    ── Why this breaks flatness ─────────────────────────────────────────

    Pointwise alone: flat-at-mean minimises average cell error (the
    conditional median/mean is achievable without temporal variation).
    Dynamics term: flat has Δŷ=0, so every cell where truth changes
    (|Δy| > 0) contributes irreducible loss. The model MUST reproduce
    temporal changes to reduce this term. Per-series, per-cell — no
    template risk, no vanishing gradient for high-variance series.

    ── Why no template induction ────────────────────────────────────────

    Every operation is per-cell within each series. No batch-level
    aggregation, no frequency-domain matching, no windowed means.
    Different series can have arbitrarily different temporal structure
    and the loss treats each independently.

    ── Weighting ────────────────────────────────────────────────────────

    event_mag = 0.01 + 0.99 · σ(5·(|union| − τ) / τ)
    Applied to BOTH terms: pointwise uses |y_true|, |y_pred| union;
    dynamics uses |Δy_true|, |Δy_pred| union. A transition of ≥1
    fatality (|Δy| > τ) gets full weight; sub-threshold transitions
    get 1% floor.

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
    # Dynamics loss (first-difference matching)
    # ------------------------------------------------------------------

    def _dynamics_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """First-difference matching weighted by LEVEL at each transition.

        Anti-flatness: flat prediction (Δŷ=0) pays log_cosh_proportional(Δy)
        at every onset/offset with full event-weight.

        Weighting by max(|y_true[t]|, |y_true[t-1]|, |y_pred[t]|, |y_pred[t-1]|)
        ensures within-conflict dynamics (e.g., 100→200 deaths, Δy≈0.7 < τ)
        still get full weight because the LEVEL is above τ. Weighting by
        difference magnitude alone would give such transitions only 1% weight.
        """
        d_pred = y_pred[:, 1:] - y_pred[:, :-1]
        d_true = y_true[:, 1:] - y_true[:, :-1]
        cell_err = self._log_cosh_proportional(d_pred - d_true)
        # Weight by max LEVEL flanking each step (not difference magnitude)
        level_true = torch.max(y_true[:, 1:].abs(), y_true[:, :-1].abs())
        level_pred = torch.max(y_pred[:, 1:].abs(), y_pred[:, :-1].abs())
        event_mag = self._event_magnitude(level_pred, level_true)
        return (event_mag * cell_err).sum() / event_mag.sum().clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        loss_pw = self._pointwise_loss(y_pred, y_true)

        loss_dyn = y_pred.new_tensor(0.0)
        if y_pred.size(1) >= 2:
            loss_dyn = self._dynamics_loss(y_pred, y_true)

        total_loss = loss_pw + loss_dyn

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: pw={loss_pw.item():.6f} "
                f"dyn={loss_dyn.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | pw=%.6f dyn=%.6f total=%.6f",
            loss_pw.item(), loss_dyn.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"