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

    2. **Windowed level anchor with optional Tukey-capped DRO** — log_cosh
       on per-series, per-window mean error.

       Only mechanism that can shift per-series means. Shape loss is
       structurally DC-blind.

       log_cosh (not asinh_integral) is used deliberately: the level
       anchor's job is gentle DC correction. tanh saturation at ±1
       prevents the anchor from dominating late in training as shape
       loss decreases.

       When _LEVEL_DRO=True: log-space DRO concentrates the level
       gradient on drifting (series, window) pairs. Without it, the
       ~90% near-zero peace windows dilute the gradient on event-series
       DC bias and it never gets corrected. Same z-score + log1p +
       soft alpha-blend mechanism, no Tukey cap.

    3. **Temporal gradient matching** — asinh_integral on first-difference
       errors (∂ŷ/∂t − ∂y/∂t). Always on, no hyperparameters.

       Penalizes wrong slopes: if the model predicts the right magnitude
       but at the wrong time, the pointwise shape loss may be large but
       the gradient loss directly targets onset/offset timing errors.
       Complementary to the shape loss: shape sees level of error at
       each timestep, gradient sees rate-of-change mismatches.

       O(T) computation, no DP needed. Uses the same asinh_integral
       base loss for consistent gradient scaling.

    4. **Multi-resolution STFT loss** — log_cosh on magnitude-spectrum
       differences at three (n_fft, hop) resolutions, AC bins only.
       DC bin masked (level anchor handles DC).  Safe magnitude
       sqrt(re²+im²+ε) avoids gradient blowup at |z|→0.  Only series
       with signal above τ are included.  Uses log_cosh (not
       asinh_integral) so the regulariser stays bounded.
       Always on, no hyperparameters.

    ── asinh_integral properties ────────────────────────────────────────

    L(0) = 0,  L'(0) = 0,  L''(0) = 1   (matches MSE curvature at origin)
    For |x| < 1:  L(x) ≈ 0.5x²           (identical to MSE and log_cosh)
    For |x| > 2:  L(x) ≈ |x|·ln(2|x|) − |x|   (super-linear growth)
    Gradient = asinh(x), bounded by ln(2|x|+1)  (no explosion)

    ─────────────────────────────────────────────────────────────────────

    Args:
        non_zero_threshold: Transformed-space cutoff for signal filtering.
            - AsinhTransform: 0.88 ≈ asinh(1)

    Example:
        >>> loss_fn = SpotlightLossAsinh(non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _LEVEL_DRO = True

    def __init__(self, non_zero_threshold: float):
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"non_zero_threshold must be positive, got {non_zero_threshold}"
            )
        super().__init__()
        self.non_zero_threshold = non_zero_threshold
        logger.info(
            "SpotlightLossAsinh | threshold=%.4f",
            non_zero_threshold,
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

    @staticmethod
    def _temporal_gradient_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Temporal gradient matching via first-difference errors.

        Computes asinh_integral on (Δŷ − Δy) where Δ is the first
        temporal difference.  Penalises rate-of-change mismatches:
        if the model predicts the right magnitude at the wrong time,
        shape loss sees two large pointwise errors while this term
        directly targets the onset/offset timing error.

        Complexity: O(T), no DP or convolution needed.
        """
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        return SpotlightLossAsinh._asinh_integral(dy_pred - dy_true).mean()

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only).

        Safe magnitude sqrt(re² + im² + ε) avoids gradient blowup at
        |z|→0.  DC bin is masked — level anchor already handles DC.
        Only series with signal above threshold are included.
        log_cosh (not asinh_integral) keeps the regulariser bounded.
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

        for n_fft, hop in self.SPECTRAL_RESOLUTIONS:
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

        # ── Input guard: replace NaN and ±inf before any arithmetic ─────
        # torch.clamp() passes NaN through unchanged — use nan_to_num.
        _SAFE = 1e4
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=_SAFE, neginf=-_SAFE)
        y_true = torch.nan_to_num(y_true, nan=0.0, posinf=_SAFE, neginf=-_SAFE)

        T = y_pred.size(1)
        e = y_pred - y_true

        # ── DC/AC decomposition ───────────────────────────────────────
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        # ── Base cell loss: asinh integral on demeaned error ──────────
        cell_loss = self._asinh_integral(e_shape)

        loss_shape = cell_loss.mean()

        # ── Windowed level anchor ─────────────────────────────────────
        # log_cosh: saturating gradient prevents late-training domination.
        W = max(4, T // 6)
        e_windows = list(e.split(W, dim=1))
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e_windows], dim=1
        )
        level_losses = self._log_cosh(window_means)    # (B, n_windows)

        if self._LEVEL_DRO:
            # DRO: concentrate gradient on drifting (series, window)
            # pairs, suppressed by 90% peace entries.
            level_flat = level_losses.detach().flatten()
            log_lv = torch.log(level_flat + 1e-8)
            lv_std = log_lv.std()
            if not torch.isfinite(lv_std) or lv_std < 1e-8:
                lv_std = level_flat.new_tensor(0.1)
            lv_cv = torch.log1p(lv_std / (log_lv.mean().abs() + 1e-8))
            lv_alpha = lv_cv / (lv_cv + 1.0)
            lv_z = (log_lv - log_lv.mean()) / lv_std.clamp(min=0.1)
            w_lv = torch.log1p((1.0 + lv_z).clamp(min=0.0))
            w_lv = w_lv / w_lv.mean().clamp(min=1e-8)
            w_lv = lv_alpha * w_lv + (1.0 - lv_alpha)
            w_lv = torch.nan_to_num(w_lv, nan=1.0, posinf=1.0, neginf=0.0)
            w_lv = w_lv.view_as(level_losses)
            loss_level = T * (w_lv * level_losses).mean()
        else:
            loss_level = T * level_losses.mean()

        # ── Temporal gradient matching ────────────────────────────────
        loss_grad = self._temporal_gradient_loss(y_pred, y_true) if T >= 2 else y_pred.new_tensor(0.0)

        # ── Multi-resolution spectral loss ────────────────────────────
        loss_spec = self._spectral_loss(y_pred, y_true) if T >= 6 else y_pred.new_tensor(0.0)

        total_loss = loss_shape + loss_level + loss_grad + loss_spec

        if not torch.isfinite(total_loss):
            logger.warning(
                "SpotlightLossAsinh non-finite: shape=%.6f level=%.6f "
                "grad=%.6f spec=%.6f — returning safe fallback",
                loss_shape.item(), loss_level.item(), loss_grad.item(),
                loss_spec.item(),
            )
            safe_e = torch.nan_to_num(e, nan=0.0, posinf=_SAFE, neginf=-_SAFE)
            return self._asinh_integral(safe_e).mean()

        logger.debug(
            "SpotlightLossAsinh | shape=%.6f level=%.6f grad=%.6f "
            "spec=%.6f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_grad.item(), loss_spec.item(), total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLossAsinh("
            f"non_zero_threshold={self.non_zero_threshold})"
        )