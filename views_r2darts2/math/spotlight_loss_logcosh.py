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

    2. **Adaptive compound weighting** — magnitude-proportional, parameter-free.

       difficulty = 1 − exp(−|e|): how wrong this cell is (curriculum).
       event_mag = max(|y|, |ŷ_sg|) / (τ + max(|y|, |ŷ_sg|)): continuous
       magnitude signal ∈ [0, 1).  Syria (500 deaths) gets ~0.998,
       Chad (2 deaths) ~0.72.  Union semantics: false positives get
       event_mag > 0.5.  w_compound = 1 + 2 × difficulty × event_mag
       ∈ [1, 3).  Self-correcting: as |e|→0, w→1.

    3. **KL-DRO tail aggregation (log-space)** — parameter-free.

       Z-score log(cell_loss) globally across all B×T cells, apply
       concave log1p weights, soft alpha-blend toward uniform when
       variance is small.  Detects *proportional* outliers across
       the 90/10 peace/event split.

    4. **Windowed level anchor** — T-scaled log_cosh on per-window
       mean error with DRO aggregation.

       Only mechanism that can shift per-series means (shape loss is
       structurally DC-blind).  Windows of width max(6, T//3) (~3 wide
       windows) catch intra-horizon level drift.  Scaled by T so the
       level gradient is strong enough to correct volume against the
       90% peace-cell majority pulling toward zero.

    5. **Temporal gradient matching** — log_cosh on first-difference
       errors (∂ŷ/∂t − ∂y/∂t).  Signal-filtered and compound-weighted.

       Only series with signal above τ are included (peaceful series
       contribute zero-information gradient).  Per-transition log1p
       difficulty weighting ensures spike onsets/offsets get
       proportional attention despite tanh saturation.
       O(T) computation.

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
    _TEMPORAL_GRADIENT = False
    _STFT = True

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
        # Curriculum gating: regularisers activate as core loss converges.
        # persistent=False: excluded from state_dict — resets each training
        # run, no checkpoint mismatch on load.
        self.register_buffer('_core_ema', torch.tensor(float('inf')), persistent=False)
        self.register_buffer('_core_peak', torch.tensor(0.0), persistent=False)
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
        max(6, T//3) (~3 wide windows), computes log_cosh on per-window
        means, then aggregates with DRO weights.  Scaled by T to keep
        the level gradient strong enough to correct volume against the
        90% peace-cell majority.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )
        level_losses = self._log_cosh(window_means)
        w = self._dro_weights(level_losses.flatten()).view_as(level_losses)
        return T * (w * level_losses).mean()

    def _temporal_gradient_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Transition-filtered, compound-weighted temporal gradient matching.

        Computes log_cosh on first-difference errors (Δŷ − Δy).
        Only transitions where at least one endpoint (t or t+1) exceeds τ
        in y_true or y_pred are included.  This removes zero→zero transitions
        entirely, preventing the smoothness penalty from competing with the
        shape loss at peace timesteps within conflict series.
        """
        τ = self.non_zero_threshold
        # Build (B, T-1) signal mask: transition is active if either endpoint
        # has signal in ground truth or (stop-gradient) prediction.
        sig = (
            (torch.abs(y_true) > τ)
            | (torch.abs(y_pred.detach()) > τ)
        )  # (B, T)
        trans_mask = sig[:, :-1] | sig[:, 1:]  # (B, T-1): either endpoint

        if not trans_mask.any():
            return y_pred.new_tensor(0.0)

        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]   # (B, T-1)
        dy_true = y_true[:, 1:] - y_true[:, :-1]   # (B, T-1)
        de = dy_pred - dy_true

        cell_grad = self._log_cosh(de)

        # Zero out non-signal transitions, then average over active ones only.
        cell_grad = cell_grad * trans_mask.float()

        # Compound weighting: upweight spike onsets/offsets.
        abs_de = torch.abs(de.detach())
        w_grad = 1.0 + torch.log1p(abs_de)
        w_grad = w_grad * trans_mask.float()
        denom = w_grad.sum().clamp(min=1e-8)

        return (w_grad * cell_grad).sum() / denom

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
        difficulty = 1.0 - torch.exp(-abs_e)
        event_mag = abs_max / (self.non_zero_threshold + abs_max)
        w_compound = 1.0 + 2.0 * difficulty * event_mag

        # ── Shape DRO (global) ─────────────────────────────────────────
        w_dro = self._dro_weights(cell_loss.flatten()).view_as(cell_loss)
        w_total = w_compound * w_dro
        w_total = w_total / w_total.mean()
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
        loss_shape = (w_total * cell_loss).mean()

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, T)

        # ── Curriculum gate for regularisers ────────────────────────
        # Track EMA of core (shape+level) loss and its peak.
        # gate = fraction of peak loss recovered: 0.05 at start, opens as
        # core converges. If core spikes (bad batch / interference),
        # gate contracts automatically.  Prevents timing/spectral
        # gradients from competing with shape+level during early learning.
        # Leaky peak (×0.999/batch, half-life ≈ 693 batches) avoids
        # permanent inflation from outlier batches.
        core_det = (loss_shape + loss_level).detach()
        if self.training:
            with torch.no_grad():
                if torch.isinf(self._core_ema):
                    self._core_ema.fill_(core_det)
                else:
                    self._core_ema.lerp_(core_det, 0.05)
                self._core_peak.copy_(torch.max(self._core_peak * 0.999, self._core_ema))
        if torch.isinf(self._core_ema) or self._core_peak < 1e-8:
            gate = core_det.new_tensor(0.05)
        else:
            gate = (1.0 - self._core_ema / (self._core_peak + 1e-8)).clamp(0.05, 1.0)

        # ── Temporal gradient matching (gated) ─────────────────────
        loss_grad = y_pred.new_tensor(0.0)
        if self._TEMPORAL_GRADIENT and T >= 2:
            loss_grad = self._temporal_gradient_loss(y_pred, y_true)

        # ── Multi-resolution spectral loss (gated) ─────────────────
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        total_loss = loss_shape + loss_level + gate * (0.5 * loss_grad + 0.5 * loss_spec)

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} grad={loss_grad.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f grad=%.6f "
            "spec=%.6f gate=%.4f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_grad.item(), loss_spec.item(),
            gate.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"