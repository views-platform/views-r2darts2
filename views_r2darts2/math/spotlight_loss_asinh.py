import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossAsinh(torch.nn.Module):
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
       event_mag > 0.5.  w_compound = 1 + 4 × difficulty × event_mag
       ∈ [1, 5).  Self-correcting: as |e|→0, w→1.

    3. **KL-DRO tail aggregation (log-space)** — parameter-free.

       Z-score log(cell_loss) globally across all B×T cells, apply
       concave log1p weights, soft alpha-blend toward uniform when
       variance is small.  Detects *proportional* outliers across
       the 90/10 peace/event split.

    4. **Windowed level anchor** — T-scaled log_cosh on per-window
       mean error with DRO aggregation.

       Only mechanism that can shift per-series means (shape loss is
       structurally DC-blind).  Windows of width max(6, T//3) (~3 wide
       windows) catch intra-horizon level drift.  Scaled by T: necessary
       to overcome the 90% zero-cell majority.  √T empirically caused
       flatlines (TsMixer + TiDE) because DC never converged.  The
       flat-window-mean attractor is broken by ungated STFT instead.

    5. **Temporal gradient matching** — log_cosh on first-difference
       errors (∂ŷ/∂t − ∂y/∂t).  Soft-weighted, fully data-driven.

       Continuous relevance weight w = 1 − exp(−r) where
       r = |Δy_raw|/max(midpoint_raw, 1) is the proportional raw-space
       change magnitude (via sinh inversion).  Plateau transitions get
       w≈0 naturally; onset/offset ~0.63; doublings ~0.63; major events
       ~0.86.  No hardcoded thresholds — the raw-space proportional
       change is the only signal.  Combined with log1p error-curriculum.
       O(T) computation.

    6. **Multi-resolution STFT loss** — log_cosh on magnitude-spectrum
       differences at three (n_fft, hop) resolutions, AC bins only.
       DC bin masked (level anchor handles DC).  Safe magnitude
       sqrt(re²+im²+ε) avoids gradient blowup at |z|→0.  Only series
       with signal above τ are included.  Always on, no hyperparameters.

    ── Base cell loss: Asinh-Integral (O(x log x) tails) ─────────────

    Uses the analytic integral of asinh(x/c):
        L(x) = x · asinh(x/c) - sqrt(x² + c²) + c

    This achieves strictly convex O(x²) behavior near zero to ignore
    measurement noise, seamlessly transitioning to O(x log x) penalty
    in the tails for catastrophic misses. The gradient is simply
    asinh(x/c), mathematically aligning with the target scaler for
    exceptional stability and clean optimization topography.

    ─────────────────────────────────────────────────────────────────────

    Args:
        non_zero_threshold: Transformed-space cutoff for compound
            weighting gate.
            - AsinhTransform: 0.88 ≈ asinh(1)
            - FourthRootTransform: 0.19 ≈ (1+1)^0.25 − 1

    Example:
        >>> loss_fn = SpotlightLossAsinh(non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _TEMPORAL_GRADIENT = False
    _STFT = True
    # Capacity of the per-channel uncertainty-weight bank (see __init__).
    # Covers any realistic multi-target setup (sb/ns/os = 3).
    _MAX_CHANNELS = 8

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

        # Per-channel homoscedastic uncertainty weights (Kendall & Gal 2018),
        # used ONLY on the multivariate (C>1) path to balance the joint
        # sb/ns/os objective so the high-magnitude sb channel cannot dominate
        # the summed loss purely by its raw scale.  s_c = log σ_c²; the
        # combination is  Σ_c [0.5·exp(-s_c)·L_c + 0.5·s_c]  (regression form).
        # Allocated as a fixed-capacity bank at construction time: the channel
        # count C is unknown until the first forward, but the optimizer is
        # built from the parameters present when the torch module is created
        # (the loss is a registered submodule), so a lazily-created parameter
        # would be excluded from optimisation.  Unused bank entries receive no
        # gradient and are skipped by the optimizer.  Init 0 → channels start
        # equally weighted.
        self.channel_log_var = torch.nn.Parameter(torch.zeros(self._MAX_CHANNELS))

        # Curriculum gating: regularisers activate as core loss converges.
        # persistent=False: excluded from state_dict — resets each training
        # run, no checkpoint mismatch on load.
        self.register_buffer('_core_ema', torch.tensor(float('inf')), persistent=False)
        self.register_buffer('_core_peak', torch.tensor(0.0), persistent=False)
        logger.info("SpotlightLossAsinh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    @staticmethod
    def _asinh_integral_loss(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Analytic integral of asinh(x/c).
        
        Yields O(x^2) near 0 and O(x * log(x)) in the tails.
        Strictly convex with clean gradients.
        
        Formula: x * asinh(x/c) - sqrt(x^2 + c^2) + c
        Gradient: asinh(x/c)
        """
        x_c = x / c
        return x * torch.arcsinh(x_c) - torch.hypot(x, torch.tensor(c, device=x.device, dtype=x.dtype)) + c

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

    @staticmethod
    def _dro_weights_channelwise(losses: torch.Tensor) -> torch.Tensor:
        """Per-channel DRO weights for (B, T, C) input.

        Identical log-space KL-DRO math to ``_dro_weights``, but every
        statistic (mean, std, cv, alpha, normalisation) is computed WITHIN
        each channel — reduction over the batch and time axes with the
        channel axis kept.  This makes 'hardest cells' relative to each
        channel's own loss distribution, so the magnitude-dominant sb channel
        can no longer monopolise the DRO tail and starve ns/os.  Returns
        weights with per-channel mean ≈ 1, shape (B, T, C).
        """
        dims = (0, 1)
        log_l = torch.log(losses.detach() + 1e-8)                 # (B, T, C)
        std = log_l.std(dim=dims, keepdim=True)                   # (1, 1, C)
        std = torch.where(
            torch.isfinite(std) & (std > 1e-8),
            std,
            losses.new_tensor(0.1),
        )
        mean = log_l.mean(dim=dims, keepdim=True)                 # (1, 1, C)
        cv = torch.log1p(std / (mean.abs() + 1e-8))
        alpha = cv / (cv + 1.0)
        z = (log_l - mean) / std.clamp(min=0.1)
        w = torch.log1p((1.0 + z).clamp(min=0.0))
        w = w / w.mean(dim=dims, keepdim=True).clamp(min=1e-8)
        w = alpha * w + (1.0 - alpha)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _shape_loss(
        self, e_shape: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compound-weighted, DRO-aggregated shape loss.

        Returns a scalar for 2D (B, T) input (single target) and a
        per-channel vector (C,) for 3D (B, T, C) input.  When 3D, the DRO
        aggregation and weight normalisation are computed per channel so the
        result is each channel's own shape loss, ready for cross-channel
        uncertainty weighting.  The 2D branch is identical to the original
        inlined computation.
        """
        cell_loss = self._asinh_integral_loss(e_shape)
        abs_e = torch.abs(e_shape.detach())
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        difficulty = 1.0 - torch.exp(-abs_e)
        event_mag = abs_max / (self.non_zero_threshold + abs_max)
        w_compound = 1.0 + 4.0 * difficulty * event_mag

        if cell_loss.dim() == 3:
            w_dro = self._dro_weights_channelwise(cell_loss)
            w_total = w_compound * w_dro
            w_total = w_total / w_total.mean(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
            return (w_total * cell_loss).mean(dim=(0, 1))          # (C,)

        w_dro = self._dro_weights(cell_loss.flatten()).view_as(cell_loss)
        w_total = w_compound * w_dro
        w_total = w_total / w_total.mean()
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
        return (w_total * cell_loss).mean()                       # scalar

    def _combine_channels(self, per_channel_loss: torch.Tensor) -> torch.Tensor:
        """Combine per-channel losses with learnable homoscedastic uncertainty
        weights:  Σ_c [0.5·exp(-s_c)·L_c + 0.5·s_c].

        Each channel is normalised by its own learned scale s_c before
        summing, so a high-magnitude channel (sb) cannot dominate the joint
        objective purely by its loss magnitude.  The s_c adapt to training
        dynamics via SGD (a reducible channel shrinks its σ and is
        up-weighted; a noisy channel self-down-weights).  The 0.5·s_c term
        regularises the weights away from the degenerate all-zero solution.
        """
        C = per_channel_loss.shape[0]
        if C > self.channel_log_var.numel():
            # More channels than the bank: fall back to equal weighting.
            return per_channel_loss.sum()
        s = self.channel_log_var[:C]
        return (0.5 * torch.exp(-s) * per_channel_loss + 0.5 * s).sum()

    def _windowed_level_loss(self, e: torch.Tensor, T: int) -> torch.Tensor:
        """Windowed log_cosh level anchor with DRO aggregation.

        Splits the T-length error into non-overlapping windows of width
        max(6, T//3) (~3 wide windows), computes log_cosh on per-window
        means, then aggregates with DRO weights.  Scaled by T: necessary
        to overcome the 90% zero-cell majority pulling the DC component
        toward zero.  Empirically, √T is insufficient — both TsMixer and
        TiDE flatlined with it because level never converged and shape had
        no stable DC base to refine.  The flat-minimum-at-window-mean is
        broken by ungated STFT instead of weakening level.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )
        level_losses = self._asinh_integral_loss(window_means)
        if level_losses.dim() == 3:
            # Per-channel level anchor: DRO and reduction within each channel.
            w = self._dro_weights_channelwise(level_losses)
            return T * (w * level_losses).mean(dim=(0, 1))         # (C,)
        w = self._dro_weights(level_losses.flatten()).view_as(level_losses)
        return T * (w * level_losses).mean()                       # scalar

    def _temporal_gradient_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Soft-weighted temporal gradient matching — fully data-driven.

        Replaces all binary thresholds (τ onset, _ESCALATION_FRAC) with a
        single continuous relevance weight derived from the proportional
        raw-space change magnitude:

            r = |Δy_raw| / max(midpoint(|y_raw_a|, |y_raw_b|), 1)
            w_rel = 1 − exp(−max(r_true, r_pred))

        Properties:
        • Plateau (Δy=0)           → w=0.  Silent, no smoothness pressure.
        • Mild escalation (18%)    → w≈0.17.  Proportional penalty.
        • Moderate escalation (27%) → w≈0.24.
        • Onset 0→1 death          → w≈0.63.  (raw_local clamps to 1.)
        • Major event 0→100        → w≈0.86.

        Inflection at ~100% raw change (doubling),
        which is a natural scale anchor.  Combined with log1p error-
        curriculum weighting: total_w = w_rel × (1 + log1p(|de|)).
        """
        _CLAMP = 10.0  # sinh(10) ≈ 11 013, numerical safety only

        # ── Proportional raw-space change magnitude ────────────────────
        def _rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            raw_a = torch.sinh(a.clamp(-_CLAMP, _CLAMP))
            raw_b = torch.sinh(b.clamp(-_CLAMP, _CLAMP))
            raw_mid = (raw_a.abs() + raw_b.abs()).mul(0.5).clamp(min=1.0)
            return (raw_b - raw_a).abs() / raw_mid

        rel_true = _rel(y_true[:, :-1], y_true[:, 1:])
        rel_pred = _rel(y_pred[:, :-1].detach(), y_pred[:, 1:].detach())

        # Soft relevance weight: 0 at plateaus, →1 for large changes.
        soft_w = 1.0 - torch.exp(-torch.max(rel_true, rel_pred))  # (B, T-1)

        if soft_w.sum() < 1e-8:
            return y_pred.new_tensor(0.0)

        # ── Temporal gradient error ────────────────────────────────────
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        de = dy_pred - dy_true

        cell_grad = self._log_cosh(de)

        # Combined weight: data-driven relevance × error curriculum.
        abs_de = torch.abs(de.detach())
        w = soft_w * (1.0 + torch.log1p(abs_de))
        denom = w.sum().clamp(min=1e-8)

        return (w * cell_grad).sum() / denom

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

        # ── DC/AC decomposition (per channel when multivariate) ───────
        # e.mean over the time axis removes each channel's DC offset.
        e_mean = e.mean(dim=1, keepdim=True)
        e_shape = e - e_mean

        # ── Shape + level losses (per-channel DRO when 3D) ────────────
        loss_shape_pc = self._shape_loss(e_shape, y_true, y_pred)   # scalar | (C,)
        loss_level_pc = self._windowed_level_loss(e, T)            # scalar | (C,)

        # ── Core objective ────────────────────────────────────────────
        # Univariate: original sum (unchanged).  Multivariate: combine the
        # per-channel shape+level objectives with learnable uncertainty
        # weights so the sb channel's magnitude cannot dominate ns/os.
        if loss_shape_pc.dim() == 0:
            loss_shape = loss_shape_pc
            loss_level = loss_level_pc
            core = loss_shape + loss_level
        else:
            core = self._combine_channels(loss_shape_pc + loss_level_pc)
            loss_shape = loss_shape_pc.sum().detach()  # logging only
            loss_level = loss_level_pc.sum().detach()  # logging only

        # ── Curriculum gate for regularisers ────────────────────────
        # Track EMA of core (shape+level) loss and its peak.
        # gate = fraction of peak loss recovered: 0.05 at start, opens as
        # core converges. If core spikes (bad batch / interference),
        # gate contracts automatically.  Prevents timing/spectral
        # gradients from competing with shape+level during early learning.
        # Leaky peak (×0.999/batch, half-life ≈ 693 batches) avoids
        # permanent inflation from outlier batches.
        core_det = core.detach()
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

        total_loss = core + gate * (0.5 * loss_grad + 0.5 * loss_spec)

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossAsinh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} grad={loss_grad.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossAsinh | shape=%.6f level=%.6f grad=%.6f "
            "spec=%.6f gate=%.4f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_grad.item(), loss_spec.item(),
            gate.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossAsinh(non_zero_threshold={self.non_zero_threshold})"