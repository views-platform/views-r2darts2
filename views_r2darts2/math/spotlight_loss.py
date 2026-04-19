import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    SpotlightLoss.

    Built for UCDP GED conflict fatality forecasting where ~90% of data
    is zeros (peace) and the remaining 10% spans four orders of magnitude
    in raw deaths. Every "standard" loss we tried collapsed to predicting
    zero everywhere, because predicting zero IS the correct answer 90% of
    the time. Brilliant.

    Three stages, each born from a specific failure mode that made me
    cry.

    ── Stage 1: "How much do we care?" (inverse-density weight) ──────

    Problem: 90% zeros means 90% of gradient says "be more zero." The 10%
    of cells that actually matter (wars, crises) get outvoted. MSE 
    rewards predicting flat lines.

    Solution: weight each cell by how rare its truth value is::

        w(y) = 1 + log_cosh(alpha * |y|)

    Zeros (very common) → weight 1.0. Major wars (very rare) → weight ~3.8×.
    This is a continuous approximation to 1/p(y) — the theoretically optimal
    weight for imbalanced regression (Liu & Lin 2022, "Balanced MSE").

    Why log_cosh and not just exp() or (1+|y|)^β?
    Because I tried both. exp() lets a single noisy 50k-death event get
    16× weight and hijack shared MLP weights. (1+|y|)^β works but β has
    no principled default. log_cosh grows linearly for large args (Yang
    et al. 2021 "LDS" tail truncation, analytically), so extreme events
    can't hijack training. α is one knob with known bounds.

    Why truth-only (no pred-side weight)?
    Because I spent a week debugging pred-side weight. It amplifies OOD
    predictions (model guesses ŷ=50, gets 50× gradient, explodes), and
    even detached it acts as a first-order gradient multiplier. The shock
    absorber interaction with gradient clipping is a whole separate horror
    story.

    With truth-only weight at α=0.3:
        y=0    → w=1.0   (peace)
        y=5    → w=1.9   (small war)
        y=10   → w=3.3   (big war)
        y=11.5 → w=3.8   (max UCDP, ~100k deaths in raw space)

    Gradient = w(y) × tanh(e/s), hard-bounded at w(y) × 1.0. Max gradient
    magnitude = 3.8 at α=0.3. No explosion possible.

    ── Stage 2: "Equal airtime" (50/50 balanced mean) ────────────────

    Problem: even with Stage 1 weights, 90% peace cells still dominate
    gradient sum via sheer numbers. 1000 peace cells × 1.0 = 1000 gradient.
    100 event cells × 3.0 = 300 gradient. Peace still wins 3:1.

    Solution: split loss into two buckets (event vs peace), average each
    bucket separately, then combine 50/50. Now each bucket gets exactly
    half the gradient budget regardless of headcount. This is the
    continuous regression version of class-balanced loss (Cui et al. 2019).

    Bonus: within the event bucket, a single noisy spike is diluted among
    other events (~2% if 50 event cells). So Stage 1 caps per-cell weight,
    Stage 2 caps per-class dominance.

    ── Stage 3: "Match the vibe" (multi-res spectral) ────────────────

    Problem: 36 output steps produced simultaneously from shared MLP
    weights. No autoregressive coupling. Model has zero incentive to make
    adjacent months look coherent. Result: hockey sticks, wild oscillation,
    completely flat forecasts sometimes when there is clear seasonality.

    Previous attempt: Total Variation (TV) penalty on prediction first
    differences. Worked for oscillation, failed for everything else:
    - Blind to slow drift (hockey stick looks smooth step-by-step)
    - Punished correct sharp spikes (needed truth-masking workaround)
    - Zero awareness of 12-month seasonality
    After 5 versions of TV tuning I gave up and stole from audio synthesis.

    Solution: compare STFT magnitude spectra at 3 resolutions, stolen
    directly from neural vocoder training (Yamamoto et al. 2020 "Parallel
    WaveGAN"; Wang et al. 2019 "Neural Source-Filter")::

        L_spec = (1/K) Σ_k mean[ log_cosh(|S_k(ŷ)| − |S_k(y)|) ]

    Why this is strictly better than TV:

        What it catches                   TV   Spectral
        ──────────────────────────────── ───  ────────
        Adjacent-step oscillation          ✓    ✓
        Smooth hockey-stick drift          ✗    ✓  (low-freq bins)
        Timing errors (1mo early)          ✗    ✓  (phase-invariant!)
        Missing 12-month seasonality       ✗    ✓  (n_fft=12 bin 1)
        Needs truth-masking                ✓    ✗  (not needed ever)

    The phase-insensitivity is the goal. Fourier shift theorem:
    shifting a signal by k steps changes STFT phase but not magnitude.
    So predicting something 1 month early → ~zero spectral penalty.
    Pointwise loss would double-penalize it (miss + false alarm).

    Three resolutions capture different temporal scales:
        n_fft=6,  hop=3  → spike/cessation sharpness    (4 bins × 11 frames)
        n_fft=12, hop=6  → 12-month annual cycle         (7 bins × 5 frames)
        n_fft=24, hop=12 → multi-year trend + fine detail (13 bins × 2 frames)

    Why L1-magnitude (not log-magnitude like audio does)?
    Audio uses log(|S|+ε) because human hearing is logarithmic. Armed
    conflict has no perceptual loudness curve. Also log(|S|+ε) creates
    horrifying ε-sensitivity near zero — log(ε) blows up when both pred
    and truth are zero (90% of data), adding a phantom gradient that
    pulls predictions toward zero. The one thing we don't need more of.
    L1-magnitude via log_cosh has no ε, no zero-bias, tanh-bounded
    gradients. Done.

    Skips resolutions where n_fft > T automatically. Works for T < 36.
    Cost at T=36 B=64: ~6720 log_cosh evals ≈ 3× MSE. Fine.

    ── Base cell loss: Scaled log-cosh ─────────────────────────────────

    L(e, s) = s × log(cosh(e / s)),    s = 1 + |y| / (1 + |y|)

    Quadratic near zero, linear far away. Gradient = tanh(e/s) ∈ (−1, 1).
    Like Huber but smooth — no discontinuous kink at the transition.
    Scale s ∈ [1, 2) widens quadratic basin for high-value targets, so
    the model gets useful gradient direction (not just ±1) when it's close
    to getting a war right. Took me embarrassingly long to realize
    fixed-scale log_cosh was starving high-value cells of gradient signal.

    ─────────────────────────────────────────────────────────────────────

    Args:
        alpha: Inverse-density weight scale. Higher = more weight on rare
            conflict events. 0.3 = max 3.8× weight. Range: 0.15–0.5.
            Below 0.15: model goes flat. Above 0.5: overprediction.
            Ask me how I know.
        delta: Spectral loss weight. 0.0 = disable (pointwise only, bad).
            0.15 = spectral is ~20–30% of gradient. Range: 0.05–0.3.
            Above 0.3: spectral dominates, pointwise accuracy suffers.
        non_zero_threshold: asinh-space cutoff for event vs peace.
            0.88 ≈ asinh(1) = "at least 1 battle death." Don't change
            this unless you have a good reason. I didn't.

    Example:
        >>> loss_fn = SpotlightLoss(alpha=0.3, delta=0.15, non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5 
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    # (n_fft, hop_length) for each STFT resolution.
    # 6/3 = local dynamics, 12/6 = annual cycle, 24/12 = long-term trend.
    # If you're wondering why these specific numbers: n_fft=12 puts the
    # 12-month annual cycle at exactly frequency bin 1. The others provide
    # coarser/finer temporal context.
    SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))

    def __init__(
        self,
        alpha: float,
        delta: float,
        non_zero_threshold: float,
    ):
        if alpha <= 0.0:
            raise ValueError(f"SpotlightLoss: alpha must be positive, got {alpha}")
        if alpha > 0.7:
            # You probably don't want this. At α=0.7, max weight ≈ 8.5×.
            # Model will overpredict everything. Trust me.
            _w = 1.0 + math.log(math.cosh(min(alpha * 11.5, 88.0)))
            logger.warning(
                "SpotlightLoss: alpha=%.4f > 0.7. Weight at max UCDP "
                "(asinh≈11.5) = 1+log_cosh(%.2f) ≈ %.1f×. May cause instability.",
                alpha, alpha * 11.5, _w,
            )
        if delta < 0.0:
            raise ValueError(f"SpotlightLoss: delta must be non-negative, got {delta}")
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"SpotlightLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.non_zero_threshold = non_zero_threshold

        # Pre-register Hann windows as buffers so they follow .to(device).
        # One per STFT resolution. Without this, windows stay on CPU when
        # model moves to GPU.
        for n_fft, _ in self.SPECTRAL_RESOLUTIONS:
            self.register_buffer(f"_hann_{n_fft}", torch.hann_window(n_fft))

        logger.info(
            "SpotlightLoss v20 | alpha=%.4f delta=%.4f threshold=%.4f",
            alpha, delta, non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers — the building blocks I've rewritten 20 times
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh_scaled(error: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Adaptive-width log-cosh: s × log(cosh(e / s)).

        Scale s widens the quadratic basin. For small errors (|e| << s)
        this behaves like 0.5 × e²/s — gentle gradient for fine-tuning.
        For large errors (|e| >> s) it's asymptotically |e| − s×ln2 —
        linear, so no single outlier gets infinite gradient.

        Gradient = tanh(e/s), hard-bounded in (−1, 1). This is why we
        don't need aggressive gradient clipping — tanh does it for free.

        Identity: log(cosh(x)) = |x| + softplus(−2|x|) − ln2
        (numerically stable for any finite x.)
        """
        z = error / scale
        abs_z = torch.abs(z)
        return scale * (abs_z + F.softplus(-2.0 * abs_z) - math.log(2.0))

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Plain log(cosh(x)). Same stable identity, no scale param.

        Used everywhere: base weight, spectral comparisons. 
        I want something between L1 and L2 that won't explode."
        """
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _balanced_mean(
        self, per_sample: torch.Tensor, is_event: torch.Tensor
    ) -> torch.Tensor:
        """50/50 split between event and peace cells.

        Without this, 90% peace gradient drowns 10% event gradient.
        Even with per-cell weighting.

        Each group (event / peace) gets averaged separately, then
        combined with equal 0.5 weight. If one group is empty in a
        batch (rare but happens), the other gets 100%.
        """
        # Handle multi-target: unsqueeze to (N, C) for per-channel balance
        if per_sample.dim() == 2:
            per_sample = per_sample.unsqueeze(-1)
            is_event = is_event.unsqueeze(-1)

        C = per_sample.size(-1)
        ps_flat = per_sample.reshape(-1, C)  # (batch×seq, channels)
        ie_flat = is_event.reshape(-1, C)

        n_event = ie_flat.sum(0)
        n_peace = (~ie_flat).sum(0)

        # clamp(min=1) prevents div-by-zero when a group is empty
        loss_event = (ps_flat * ie_flat).sum(0) / n_event.clamp(min=1)
        loss_peace = (ps_flat * ~ie_flat).sum(0) / n_peace.clamp(min=1)

        # 0.5 each if both present, 1.0 for the survivor if one is absent
        w_e = 0.5 * (n_event > 0).float()
        w_p = 0.5 * (n_peace > 0).float()
        total_w = (w_e + w_p).clamp(min=1e-8)  # paranoia clamp

        return ((w_e / total_w) * loss_event + (w_p / total_w) * loss_peace).mean()

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison.

        From from audio synthesis (neural vocoders). Turns out WAV
        generation and conflict forecasting share a problem: you want
        the model to get the shape right, not just the per-sample value.

        Compares |STFT(pred)| vs |STFT(true)| at 3 window sizes via
        log_cosh. Phase is discarded (magnitude only) — this is the key
        property. Predicting something 1 month early? Same magnitude spectrum.
        Zero penalty. Pointwise loss would give you double penalty and
        the model learns to never predict anything.

        Not using log-magnitude like audio does. log(|S|+ε) creates a
        gradient nightmare near zero: log(tiny + ε) ≈ log(ε) which is
        enormous, so 90% of data (peace, near-zero spectrum) generates
        phantom gradient pulling everything toward zero. We have enough
        zero-bias already.
        """
        # Multi-target: flatten (B, T, C) → (B×C, T) so each channel
        # gets its own STFT. torch.stft wants (batch, signal_length).
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        T = pred.size(1)
        total = pred.new_tensor(0.0)  # new_tensor: inherits device + dtype
        n_valid = 0

        for n_fft, hop in self.SPECTRAL_RESOLUTIONS:
            # Skip if sequence too short for this window. Graceful
            # degradation — no code changes needed for shorter horizons.
            if T < n_fft:
                continue

            # Hann window registered in __init__. Lives on same device
            # as model parameters because register_buffer.
            window = getattr(self, f"_hann_{n_fft}")

            # center=False: no zero-padding. We want exact T coverage.
            # return_complex=True: gives complex tensor, .abs() = magnitude.
            S_pred = torch.stft(
                pred, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )
            S_true = torch.stft(
                true, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )

            # L1 magnitude difference wrapped in log_cosh.
            # NOT log-domain. NOT L2. log_cosh = best of both worlds:
            # quadratic for small diffs (gentle tuning), linear for large
            # diffs (no explosion). Gradient = tanh(...) ∈ (-1, 1).
            # No ε needed.
            total = total + self._log_cosh(S_pred.abs() - S_true.abs()).mean()
            n_valid += 1

        # Average across resolutions. If somehow none were valid (T < 6),
        # returns 0. Shouldn't happen at T=36 but defensive coding keeps
        # me employed.
        return total / max(n_valid, 1)

    # ------------------------------------------------------------------
    # Forward — where the three stages come together
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Darts sometimes passes (B, T, 1) for single-target. Squeeze it.
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        e = y_pred - y_true
        abs_y = torch.abs(y_true)

        # ── Stage 1: How much do we care about each cell? ─────────────
        # w(y) = 1 + log_cosh(α|y|). Peace → 1.0×. Max war → ~3.8×.
        # Truth-only: gradient bounded at w(y) × tanh(...) ≤ 3.8.
        # Tried pred-side weight five different ways. All exploded or
        # created zero-attracting bias. This is the one that works somewhat.
        w = 1.0 + self._log_cosh(self.alpha * abs_y)

        # ── Stage 2: Pointwise loss with balanced aggregation ─────────
        # Scale s ∈ [1, 2) widens the loss's quadratic basin for big
        # targets. Intuition: when truth is y=10, being off by 0.5 should
        # still give gradient proportional to error, not a saturated ±1.
        # Fixed scale was starving high-value cells of gradient direction.
        scale = 1.0 + abs_y / (1.0 + abs_y)

        # Weighted log-cosh per cell. w controls importance, log_cosh
        # controls shape. Separation of concerns that took 18 versions
        # to arrive at.
        base_loss = self._log_cosh_scaled(e, scale)
        per_sample = w * base_loss

        # Hard binary split at threshold. "Event" = at least 1 death.
        # 50/50 balanced mean ensures events get half the gradient budget
        # regardless of being outnumbered 9:1 by peace cells.
        is_event = abs_y > self.non_zero_threshold
        loss_pointwise = self._balanced_mean(per_sample, is_event)

        # ── Stage 3: Do the predicted time series look right? ─────────
        # Match STFT magnitude spectra at 3 resolutions. Catches:
        # - Oscillation between adjacent steps (n_fft=6)
        # - Missing 12-month seasonality (n_fft=12, bin 1)
        # - Slow hockey-stick drift (n_fft=24, low-freq bins)
        # And forgives timing errors via phase invariance.
        loss_spectral = y_pred.new_tensor(0.0)
        if self.delta > 0.0 and y_pred.size(1) >= 6:
            loss_spectral = self._spectral_loss(y_pred, y_true)

        total_loss = loss_pointwise + self.delta * loss_spectral

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLoss: pointwise={loss_pointwise.item():.6f} "
                f"spectral={loss_spectral.item():.6f}"
            )

        logger.debug(
            "SpotlightLoss | pw=%.6f spec=%.6f total=%.6f",
            loss_pointwise.item(),
            loss_spectral.item(),
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLoss(alpha={self.alpha}, delta={self.delta}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )
