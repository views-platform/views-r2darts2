import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightFocalLoss(torch.nn.Module):
    """
    Focal regression loss for zero-inflated heavy-tailed time series.

    Designed for UCDP GED conflict fatality forecasting, evaluated on MSLE.
    Operates in log1p space (use with LogTransform target scaler).

    ── Core idea ────────────────────────────────────────────────────────

    Borrowed from RetinaNet (Lin et al. 2017) object detection, adapted
    for regression. Instead of weighting by magnitude or class membership,
    weight by DIFFICULTY — how wrong the model is on each cell right now.

    Classification focal loss:
        FL(p_t) = −(1 − p_t)^γ · log(p_t)

    Regression focal loss (this implementation):
        L(e) = (1 − exp(−|e|))^γ · log_cosh(e)

    where e = ŷ − y in log1p space.

    (1 − exp(−|e|)) maps error magnitude to [0, 1):
        |e| = 0   → 0.00  (perfect prediction → zero weight)
        |e| = 0.5 → 0.39  (close → low weight)
        |e| = 1.0 → 0.63  (moderate error → moderate weight)
        |e| = 2.0 → 0.86  (large error → near-full weight)
        |e| = 5.0 → 0.99  (very wrong → full weight)

    Raising to γ sharpens the transition:
        γ = 1.0: linear ramp (mild focusing)
        γ = 2.0: quadratic — easy cells suppressed 4× more (default)
        γ = 3.0: aggressive — only hard cells get meaningful gradient

    ── What this gains over PrismLoss ───────────────────────────────

    No class-specific logic. No event/peace split, no dual_mean, no
    event_weight, no non_zero_threshold for the pointwise loss. The focal
    mechanism handles class imbalance automatically:
    - Peace cells correctly predicted at zero: |e| ≈ 0, weight ≈ 0.
    - Missed small events (most MSLE-sensitive): |e| ≈ 0.69, weight ≈ 0.25.
    - Missed large events: |e| ≈ 5+, weight ≈ 1.0.
    - False alarms on peace cells: |e| > 0, weight > 0 — penalised.

    At initialization (ŷ ≈ 0), the focal weight IS magnitude-proportional
    since |e| ≈ |y|. This naturally escapes the zero-prediction basin. As
    training progresses and the model learns, weights shift to whatever
    cells are currently hardest — transition points, onset timing, small
    events near the decision boundary.

    ── Spectral term ────────────────────────────────────────────────────

    Same multi-resolution STFT magnitude comparison as PrismLoss.
    DC bin (frequency bin 0) is masked — spectral is purely AC temporal
    structure. Prevents flat forecasts and enforces seasonal learning.

    ── Gradient analysis ────────────────────────────────────────────────

    Focal weight is DETACHED — no gradient flows through it. Each forward
    pass recomputes the weight from the current prediction, so it adapts
    across iterations but within a single backward pass it's a fixed
    scalar multiplier. This prevents second-order dynamics where the
    weight's gradient could destabilize training (same principle as the
    detached pred-side importance weight in PrismLoss).

    Per-cell gradient: w_focal · tanh(e)
    Max gradient: 1.0 × 1.0 = 1.0. Always bounded. No clipping needed.

    Args:
        gamma: Focal exponent. Higher = more aggressive focus on hard cells.
            1.0 = mild focusing (all errors weighted somewhat).
            2.0 = standard focusing (easy cells strongly suppressed).
            3.0 = aggressive (only hard cells matter).
            Range: [0.5, 4.0]. At γ=0 this degrades to plain log_cosh.
        delta: Spectral loss weight. Same as PrismLoss.
            0.0 = disable spectral. 0.15 = ~20-30% of gradient.
            Range: [0.05, 0.20].
        non_zero_threshold: For spectral signal filtering only (which
            series get spectral comparison). Not used in the pointwise
            loss — focal weighting handles everything. Default: 0.693
            (log1p(1) ≈ 1 battle death).

    Example:
        >>> loss_fn = SpotlightFocalLoss(gamma=2.0, delta=0.15, non_zero_threshold=0.693)
        >>> y_pred = torch.randn(8, 36)  # log1p-space predictions
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))

    def __init__(
        self,
        gamma: float,
        delta: float,
        non_zero_threshold: float,
    ):
        if gamma < 0.0:
            raise ValueError(f"SpotlightFocalLoss: gamma must be non-negative, got {gamma}")
        if delta < 0.0:
            raise ValueError(f"SpotlightFocalLoss: delta must be non-negative, got {delta}")
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"SpotlightFocalLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "SpotlightFocalLoss v1 (log1p-native) | gamma=%.4f delta=%.4f threshold=%.4f",
            gamma, delta, non_zero_threshold,
        )

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable for any finite x."""
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

            mag_pred = torch.sqrt(S_pred.real ** 2 + S_pred.imag ** 2 + 1e-8)
            mag_true = S_true.abs()

            # Zero DC bin — spectral is purely temporal structure.
            mag_pred = mag_pred.clone()
            mag_true = mag_true.clone()
            mag_pred[:, 0, :] = 0.0
            mag_true[:, 0, :] = 0.0

            total = total + self._log_cosh(mag_pred - mag_true).mean()
            n_valid += 1

        return total / max(n_valid, 1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        e = y_pred - y_true
        cell_loss = self._log_cosh(e)

        # ── Focal modulation ──────────────────────────────────────────
        # (1 − exp(−|e|))^γ: maps error → difficulty in [0, 1).
        # Detached — no gradient through the weight. Adapts across
        # iterations but is a fixed scalar within each backward pass.
        #
        # At init (ŷ ≈ 0): weight ≈ (1 − exp(−|y|))^γ — magnitude-
        # proportional. Large events get full weight, peace gets zero.
        # This naturally escapes the zero-prediction basin.
        #
        # At convergence: weight concentrates on the hardest cells —
        # transition points, onset timing, small events near the
        # decision boundary. Exactly what moves MSLE most.
        abs_e = torch.abs(e).detach()
        focal_weight = (1.0 - torch.exp(-abs_e)) ** self.gamma
        per_sample = focal_weight * cell_loss

        loss_main = per_sample.mean()

        # ── Spectral — AC bins only ───────────────────────────────────
        loss_spectral = y_pred.new_tensor(0.0)
        if self.delta > 0.0 and y_pred.size(1) >= 6:
            loss_spectral = self._spectral_loss(y_pred, y_true)

        total_loss = loss_main + self.delta * loss_spectral

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightFocalLoss: main={loss_main.item():.6f} "
                f"spectral={loss_spectral.item():.6f}"
            )

        logger.debug(
            "SpotlightFocalLoss | main=%.6f spec=%.6f total=%.6f",
            loss_main.item(),
            loss_spectral.item(),
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightFocalLoss(gamma={self.gamma}, delta={self.delta}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )
