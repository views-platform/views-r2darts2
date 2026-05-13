import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossAsinh(torch.nn.Module):
    """
    SpotlightLoss (asinh variant) — native proportional sensitivity.

    Operates in asinh space (AsinhTransform target scaler). Uses the
    integral of asinh as the base cell loss:

        L(x) = x·asinh(x) − √(1+x²) + 1

    Gradient = asinh(x) ≈ ln(2|x|) for large |x|.

    This is the natural distance metric when the prediction space is
    asinh-scaled. Unlike log_cosh (gradient = tanh, saturates at ±1),
    the asinh gradient grows logarithmically:

        |e|   tanh(e)  asinh(e)  ratio
        0.5   0.46     0.48      1.04×
        1.0   0.76     0.88      1.16×
        2.5   0.99     1.65      1.67×
        5.0   1.00     2.31      2.31×
        8.0   1.00     2.78      2.78×

    A 500-death miss (|e|=8 in asinh space) gets 2.78× the gradient of
    a 2-death miss (|e|=2.5). Under log_cosh both get gradient ≈ 1.0.

    Because the loss itself now encodes proportional sensitivity, DRO
    aggregation is removed. DRO compensated for tanh saturation — with
    asinh, cell losses are already proportionally spread:

        asinh_int(8)/asinh_int(1) = 32.4×  vs  log_cosh(8)/log_cosh(1) = 17×

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

    2. **Adaptive compound weighting** — difficulty-gated, parameter-free.

       w_compound = 1 + difficulty × 1[abs_max > τ]     ∈ [1, 2)

       difficulty = |e| / (1 + |e|). Distinguishes *how wrong* from
       *how big* — a correctly-predicted Syria cell (|e|≈0) gets w≈1,
       while a badly-predicted Chad cell (|e|=2) gets w≈1.67. The
       threshold gate provides false-positive discipline: sub-threshold
       cells always get w=1. Self-correcting: as |e|→0, w→1.

       Still needed alongside asinh: the base loss gives proportional
       *magnitude* sensitivity; compound adds *difficulty* sensitivity.

    3. **Windowed level anchor** — T-scaled asinh_integral on per-series,
       per-window mean error.

       Only mechanism that can shift per-series means. Shape loss is
       structurally DC-blind.

       The horizon is split into non-overlapping windows of width
       W = max(4, T // 6). Each window's mean error is penalised
       independently:

           ē_w = mean(e[t_start:t_end])         per window w, series s
           l_{s,w} = asinh_integral(ē_w)
           L_level = T · mean_{s,w}[ l_{s,w} ]

       Window width is dynamic — computed from the actual output length.
       Floor of 4 ensures each window mean is statistically meaningful.

    4. **Spectral regularization** (optional, gated by δ > 0).
       Multi-resolution STFT magnitude comparison with DC bin masked.
       Phase-invariant; log_cosh on magnitude diffs.

    ── asinh_integral properties ────────────────────────────────────────

    L(0) = 0,  L'(0) = 0,  L''(0) = 1   (matches MSE curvature at origin)
    For |x| < 1:  L(x) ≈ 0.5x²           (identical to MSE and log_cosh)
    For |x| > 2:  L(x) ≈ |x|·ln(2|x|) − |x|   (super-linear growth)
    Gradient = asinh(x), bounded by ln(2|x|+1)  (no explosion)

    ─────────────────────────────────────────────────────────────────────

    Args:
        delta: Spectral loss weight. 0.0 = disable.
        non_zero_threshold: Transformed-space cutoff for compound weighting
            gate and spectral signal filtering.
            - AsinhTransform: 0.88 ≈ asinh(1)

    Example:
        >>> loss_fn = SpotlightLossAsinh(delta=0.10, non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))

    def __init__(self, delta: float, non_zero_threshold: float):
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"non_zero_threshold must be positive, got {non_zero_threshold}"
            )
        super().__init__()
        self.delta = delta
        self.non_zero_threshold = non_zero_threshold
        logger.info(
            "SpotlightLossAsinh | delta=%.4f threshold=%.4f",
            delta, non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _asinh_integral(x: torch.Tensor) -> torch.Tensor:
        """x·asinh(x) − √(1+x²) + 1, numerically stable.

        Rewrites √(1+x²)−1 = x²/(√(1+x²)+1) to avoid cancellation
        near zero.
        """
        return x * torch.asinh(x) - x * x / (torch.sqrt(1.0 + x * x) + 1.0)

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

        # ── Input guard: clamp inf → finite ───────────────────────────
        _SAFE = 1e4
        y_pred = y_pred.clamp(-_SAFE, _SAFE)
        y_true = y_true.clamp(-_SAFE, _SAFE)

        T = y_pred.size(1)
        e = y_pred - y_true

        # ── DC/AC decomposition ───────────────────────────────────────
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        # ── Base cell loss: asinh integral on demeaned error ──────────
        cell_loss = self._asinh_integral(e_shape)

        # ── Adaptive compound weighting (difficulty-gated) ────────────
        abs_e = torch.abs(e_shape.detach())
        abs_y = torch.abs(y_true)
        abs_ypred_sg = torch.abs(y_pred.detach())

        difficulty = abs_e / (1.0 + abs_e)
        abs_max = torch.max(abs_y, abs_ypred_sg)
        above_threshold = (abs_max > self.non_zero_threshold).float()
        w_compound = 1.0 + difficulty * above_threshold

        # Normalise to mean=1
        w_total = w_compound / w_compound.mean().clamp(min=1e-8)
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)

        loss_shape = (w_total * cell_loss).mean()

        # ── Windowed level anchor ─────────────────────────────────────
        W = max(4, T // 6)
        e_windows = list(e.split(W, dim=1))
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e_windows], dim=1
        )
        level_losses = self._asinh_integral(window_means)
        loss_level = T * level_losses.mean()

        # ── Spectral: AC bins only ────────────────────────────────────
        loss_spectral = y_pred.new_tensor(0.0)
        if self.delta > 0.0 and T >= 6:
            loss_spectral = self._spectral_loss(y_pred, y_true)

        total_loss = loss_shape + loss_level + self.delta * loss_spectral

        if not torch.isfinite(total_loss):
            logger.warning(
                "SpotlightLossAsinh non-finite: shape=%.6f level=%.6f "
                "spectral=%.6f — returning safe fallback",
                loss_shape.item(), loss_level.item(), loss_spectral.item(),
            )
            return self._asinh_integral(e).mean()

        logger.debug(
            "SpotlightLossAsinh | shape=%.6f level=%.6f spec=%.6f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_spectral.item(), total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLossAsinh(delta={self.delta}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )