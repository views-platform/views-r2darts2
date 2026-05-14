import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v36 — asinh + RevIN compatible, with DRO aggregation.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros, 10% spanning
    four orders of magnitude in raw deaths.

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

    2. **Log-compensated compound weighting** — parameter-free.

       difficulty = log(1 + |e|): logarithmic growth compensates tanh
       saturation in the log_cosh gradient.  Effective gradient ∝
       log1p(|e|) × tanh(|e|) ≈ asinh(|e|) for large |e|, restoring
       proportional sensitivity that raw log_cosh loses.

       Relative sigmoid gate σ(5·(abs_max/series_scale − 1)) fires on
       signals that are unusual *for this series*, not just non-zero in
       absolute terms.  series_scale = mean(|y_true|) per series,
       clamped at τ.  Differentiable, still disciplines false positives.

    3. **Hierarchical DRO tail aggregation** — parameter-free.

       Per-series DRO: within each series, upweight hard timesteps
       relative to that series' own difficulty distribution.
       Cross-series DRO: upweight globally harder series by their
       mean cell loss.  Product of both avoids single-cell dominance
       across series while preserving cross-series attention to
       systematically hard forecasts.

    4. **Windowed level anchor** — T-scaled log_cosh on per-window
       mean error with DRO aggregation.

       Only mechanism that can shift per-series means (shape loss is
       structurally DC-blind).  Windows of width max(4, T//6) catch
       intra-horizon level drift.

    5. **Temporal gradient matching** — log_cosh on first-difference
       errors (∂ŷ/∂t − ∂y/∂t). Always on, no hyperparameters.

       Penalises onset/offset timing errors via rate-of-change
       mismatches.  O(T) computation.

    6. **Multi-resolution STFT loss** — log_cosh on magnitude-spectrum
       differences at three (n_fft, hop) resolutions, AC bins only.
       DC bin masked (level anchor handles DC).  Safe magnitude
       sqrt(re²+im²+ε) avoids gradient blowup at |z|→0.  Only series
       with signal above τ are included.  Always on, no hyperparameters.

    ── Base cell loss: log_cosh ─────────────────────────────────────────

    log_cosh(x) ≈ 0.5x² for |x| < 1, ≈ |x| − ln2 for |x| > 2.
    Gradient = tanh(x) ∈ (−1, +1). Bounded by construction.

    ─────────────────────────────────────────────────────────────────────

    Args:
        non_zero_threshold: Transformed-space cutoff for compound
            weighting gate.
            - AsinhTransform: 0.88 ≈ asinh(1)
            - FourthRootTransform: 0.19 ≈ (1+1)^0.25 − 1

    Example:
        >>> loss_fn = SpotlightLossLogcosh(non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))

    def __init__(
        self,
        non_zero_threshold: float,
        alpha: float = 0.0,  # deprecated — ignored, kept for backward compat
    ):
        if alpha != 0.0:
            logger.warning(
                "SpotlightLossLogcosh: alpha is deprecated and ignored. "
                "Remove alpha from your config. (received alpha=%.4f)",
                alpha,
            )
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
    def _dro_weights(losses: torch.Tensor) -> torch.Tensor:
        """Log-space KL-DRO weights with soft alpha-blend.

        Given a flat tensor of per-element losses, returns a same-shaped
        tensor of normalised weights (mean ≈ 1).  High-loss elements get
        upweighted proportionally in log-space; soft alpha blends toward
        uniform when log-loss variance is small (early training).
        """
        log_l = torch.log(losses.detach() + 1e-8)
        std = log_l.std()
        if not torch.isfinite(std) or std < 1e-8:
            std = losses.new_tensor(0.1)
        cv = torch.log1p(std / (log_l.mean().abs() + 1e-8))
        alpha = cv / (cv + 1.0)
        z = (log_l - log_l.mean()) / std.clamp(min=0.1)
        w = torch.log1p((1.0 + z).clamp(min=0.0))
        w = w / w.mean().clamp(min=1e-8)
        w = alpha * w + (1.0 - alpha)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    @staticmethod
    def _dro_weights_2d(losses: torch.Tensor) -> torch.Tensor:
        """Batched DRO weights along dim=1 for (B, T) input.

        Equivalent to stacking _dro_weights per row, but fully
        vectorised — no Python loop over the batch dimension.
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
        """Windowed log_cosh level anchor with DRO aggregation.

        Splits the T-length error into non-overlapping windows of width
        max(4, T//6), computes log_cosh on per-window means, then
        aggregates with DRO weights.  Scaled by T to keep level gradient
        comparable to shape across different horizons.
        """
        W = max(4, T // 6)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )
        level_losses = self._log_cosh(window_means)
        w = self._dro_weights(level_losses.flatten()).view_as(level_losses)
        return T * (w * level_losses).mean()

    def _temporal_gradient_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """log_cosh on first-difference errors (Δŷ − Δy)."""
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        return self._log_cosh(dy_pred - dy_true).mean()

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

        # ── Base cell loss ────────────────────────────────────────────
        cell_loss = self._log_cosh(e_shape)

        # ── Compound weighting ────────────────────────────────────────
        abs_e = torch.abs(e_shape.detach())
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        # Log-scaled difficulty compensates tanh saturation of log_cosh:
        # effective gradient ∝ log1p(|e|) × tanh(|e|) ≈ asinh(|e|)
        difficulty = torch.log1p(abs_e)
        # Relative gate: normalise by per-series mean intensity so the
        # gate fires on "unusual for this series" not "non-zero in absolute terms".
        # Syria at constant 3 → scale=3, routine cells get gate≈0.5;
        # Thailand with one spike → scale clamped to τ, spike gets gate≈1.0.
        series_scale = torch.abs(y_true).mean(dim=1, keepdim=True).clamp(min=self.non_zero_threshold)
        soft_gate = torch.sigmoid(5.0 * (abs_max / series_scale - 1.0))
        w_compound = 1.0 + difficulty * soft_gate

        # ── Shape DRO (hierarchical: per-series × cross-series) ───────
        # Per-series DRO: within each series, upweight hard timesteps
        w_within = self._dro_weights_2d(cell_loss)            # (B, T)
        # Cross-series DRO: upweight globally harder series
        series_means = cell_loss.detach().mean(dim=1)         # (B,)
        w_across = self._dro_weights(series_means).unsqueeze(1)  # (B, 1)
        w_dro = w_within * w_across
        w_total = w_compound * w_dro
        w_total = w_total / w_total.mean()
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
        loss_shape = (w_total * cell_loss).mean()

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, T)

        # ── Temporal gradient matching ────────────────────────────────
        loss_grad = self._temporal_gradient_loss(y_pred, y_true) if T >= 2 else y_pred.new_tensor(0.0)

        # ── Multi-resolution spectral loss ────────────────────────────
        loss_spec = self._spectral_loss(y_pred, y_true) if T >= 6 else y_pred.new_tensor(0.0)

        total_loss = loss_shape + loss_level + loss_grad + loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} grad={loss_grad.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f grad=%.6f "
            "spec=%.6f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_grad.item(), loss_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"