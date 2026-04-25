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

    Solution: weight each cell by how important it is — determined by
    whichever is larger, the truth or the (detached) prediction::

        w = 1 + log_cosh(alpha * max(|y|, |ŷ_sg|))

    Zeros (very common, correctly predicted) → weight 1.0.
    Major wars (very rare) → weight ~3.8×.
    False alarms (y=0, ŷ=10) → weight ~3.3× (same as missing a war).
    This is symmetric: misses and false alarms of equal magnitude get
    equal corrective pressure, preventing systematic overprediction.

    Why log_cosh and not just exp() or (1+|y|)^β?
    Because I tried both. exp() lets a single noisy 50k-death event get
    16× weight and hijack shared MLP weights. (1+|y|)^β works but β has
    no principled default. log_cosh grows linearly for large args (Yang
    et al. 2021 "LDS" tail truncation, analytically), so extreme events
    can't hijack training. α is one knob with known bounds.

    Why max(|y|, |ŷ_sg|) and not truth-only?
    Truth-only weight creates asymmetric correction: missing a war
    (y=10, ŷ=0) gets gradient ~3.3×, but a false alarm (y=0, ŷ=10)
    gets only ~1.0×. The model learns to hedge by inflating baselines.
    Symmetric max gives equal pressure to both error directions.

    Why detach the pred side?
    Non-detached pred-side weight lets gradient flow through w, creating
    a second-order term that amplifies OOD predictions (model guesses
    ŷ=50, gets 50× gradient, explodes). Detaching makes w a simple
    scalar multiplier each forward pass. The pred-side influence is a
    first-order gradient multiplier across iterations, but log_cosh's
    linear growth (not exponential) keeps this controllable.

    Near convergence (ŷ→y), max(|y|, |ŷ_sg|) → |y|, so the weight
    degrades to truth-only behavior — stable fine-tuning gradient.

    With symmetric weight at α=0.3:
        y=0,  ŷ=0   → w=1.0   (peace, correct)
        y=5,  ŷ=0   → w=1.9   (missed small war)
        y=0,  ŷ=5   → w=1.9   (false alarm, equally penalized)
        y=10, ŷ=0   → w=3.3   (missed big war)
        y=0,  ŷ=10  → w=3.3   (big false alarm, equally penalized)

    Gradient = w × tanh(e/s), hard-bounded at w × 1.0. For in-distribution
    predictions, max weight ≈ 3.8 at α=0.3. No explosion possible.

    ── Stage 2: "Weighted airtime" (event/peace balanced mean) ──────

    Problem: even with Stage 1 weights, 90% peace cells still dominate
    gradient sum via sheer numbers. 1000 peace cells × 1.0 = 1000 gradient.
    100 event cells × 3.0 = 300 gradient. Peace still wins 3:1.

    Solution: split loss into two buckets (event vs peace), average each
    bucket separately, then combine with event_weight / (1−event_weight).
    This controls how much gradient budget goes to events vs peace.

    At event_weight=0.50 (the old hardcoded default), events get 50% of
    the budget — a ~5.7× per-cell amplification over natural prevalence
    (~10%). This caused systematic overprediction: the positive bias
    leaked into shared MLP weights, RevIN denorm made it multiplicative,
    and sinh() amplified it exponentially. Order-of-magnitude overpredict.

    Lower event_weight reduces this amplification:
        event_weight=0.50 → 5.7× per-cell influence (old default, overpredicts)
        event_weight=0.25 → 2.85× per-cell influence (moderate boost)
        event_weight=0.15 → ~1.6× per-cell influence (light boost)
        event_weight=0.10 → ~1.0× per-cell influence (natural prevalence)

    With v21's symmetric weight (Stage 1), false alarms already get
    corrective pressure — the model has less incentive to hedge upward.
    So events may not need the full 50% budget anymore. Sweeping
    event_weight lets Bayes find the right balance.

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

    ── Base cell loss: log-cosh ─────────────────────────────────────────

    L(e) = log(cosh(e))

    Quadratic near zero, linear far away. Gradient = tanh(e) ∈ (−1, 1).
    Like Huber but smooth — no discontinuous kink at the transition.

    v22 used adaptive scale s = 1 + |y|/(1+|y|) to widen the quadratic
    basin for large targets. But the scale grows fast at low magnitudes
    (s=1.47 at the threshold) while the importance weight barely
    compensates (w≈1.02). This creates a gradient valley at y≈0.88–3
    where effective gain (w/s) drops to 0.67–0.70× — below peace cells.
    MSLE is most sensitive to exactly these small-event cells (1–10
    deaths), which is why MSE could be good but MSLE bad.
    Removing the scale eliminates the valley. The importance weight w
    now handles all magnitude-dependent gradient scaling.

    ─────────────────────────────────────────────────────────────────────

    Args:
        alpha: Inverse-density weight scale. Higher = more weight on rare
            conflict events. 0.3 = max 3.8× weight. Range: 0.15–0.5.
            Below 0.15: model goes flat. Above 0.5: overprediction.
            Ask me how I know.
        delta: Spectral loss weight. 0.0 = disable (pointwise only, bad).
            0.15 = spectral is ~20–30% of gradient. Range: 0.05–0.3.
            Above 0.3: spectral dominates, pointwise accuracy suffers.
        dual_mean: Whether to use the event/peace balanced mean (Stage 2).
            True = split into event/peace buckets, combine with event_weight
            ratio. False = plain per-cell mean (event_weight ignored).
        event_weight: Fraction of gradient budget allocated to event cells
            in the balanced mean. Only used when dual_mean=True.
            0.50 = old 50/50 split (overpredicts).
            0.25 = moderate boost. 0.10 = natural prevalence. Range: 0.10–0.50.
            Interacts with alpha: alpha controls per-cell importance,
            event_weight controls per-class budget. Two orthogonal knobs.
        non_zero_threshold: asinh-space cutoff for event vs peace.
            0.88 ≈ asinh(1) = "at least 1 battle death." Don't change
            this unless you have a good reason. I didn't.

    Example:
        >>> loss_fn = SpotlightLoss(alpha=0.3, delta=0.15, dual_mean=True, event_weight=0.25, non_zero_threshold=0.88)
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
        dual_mean: bool,
        event_weight: float,
        non_zero_threshold: float,
    ):
        if alpha < 0.0:
            raise ValueError(f"SpotlightLoss: alpha must be non-negative, got {alpha}")
        if alpha > 5.0:
            logger.warning(
                "SpotlightLoss: alpha=%.4f > 5.0. Events get %.1f× weight. "
                "May cause instability.",
                alpha, 1.0 + alpha,
            )
        if delta < 0.0:
            raise ValueError(f"SpotlightLoss: delta must be non-negative, got {delta}")
        if dual_mean and not (0.0 < event_weight <= 0.5):
            raise ValueError(
                f"SpotlightLoss: event_weight must be in (0, 0.5], got {event_weight}"
            )
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"SpotlightLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.dual_mean = dual_mean
        self.event_weight = event_weight
        self.non_zero_threshold = non_zero_threshold

        # NO register_buffer for Hann windows. Buffers enter state_dict,
        # which causes checkpoint incompatibility when loss class changes
        # (PL load_from_checkpoint strict=True → RuntimeError on unexpected
        # or missing keys). Windows are created inline in _spectral_loss
        # with device=pred.device instead. Cost: ~3µs per forward. Fine.

        logger.info(
            "SpotlightLoss v32 (log1p-native) | alpha=%.4f delta=%.4f dual_mean=%s event_weight=%.4f threshold=%.4f",
            alpha, delta, dual_mean, event_weight, non_zero_threshold,
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
        """Weighted split between event and peace cells.

        Without this, 90% peace gradient drowns 10% event gradient.
        Even with per-cell weighting.

        Each group (event / peace) gets averaged separately, then
        combined with event_weight / (1−event_weight). If one group
        is empty in a batch (rare but happens), the other gets 100%.
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

        # event_weight for events, (1−event_weight) for peace.
        # If one group is absent, the other gets 100%.
        w_e = self.event_weight * (n_event > 0).float()
        w_p = (1.0 - self.event_weight) * (n_peace > 0).float()
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

        Only computed on sequences with signal in either truth or
        (detached) prediction. Pure-peace sequences where both y≈0 and
        ŷ≈0 have near-zero spectrum — comparing spectra there wastes
        compute and adds zero-attracting bias. Including false alarm
        series (y=0, ŷ>>0) ensures spectral penalty on overprediction,
        matching Stage 1's symmetric weight principle.
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

        # Filter to sequences with signal in either truth or prediction.
        # Peace-only sequences (both y≈0 and ŷ≈0) have near-zero spectrum;
        # including them wastes STFT compute and adds zero-attracting bias.
        # Pred-side check (detached) ensures false alarm series (y=0, ŷ=8)
        # still get spectral penalty — same symmetry principle as Stage 1.
        has_signal = (
            (torch.abs(true) > self.non_zero_threshold)
            | (torch.abs(pred.detach()) > self.non_zero_threshold)
        ).any(dim=1)
        if not has_signal.any():
            return pred.new_tensor(0.0)
        pred = pred[has_signal]
        true = true[has_signal]

        T = pred.size(1)
        total = pred.new_tensor(0.0)  # new_tensor: inherits device + dtype
        n_valid = 0

        for n_fft, hop in self.SPECTRAL_RESOLUTIONS:
            # Skip if sequence too short for this window. Graceful
            # degradation — no code changes needed for shorter horizons.
            if T < n_fft:
                continue

            # Create Hann window inline on same device as pred.
            # Not a buffer (no state_dict entry) — avoids checkpoint
            # incompatibility when loss version changes across deployments.
            window = torch.hann_window(n_fft, device=pred.device)

            # center=False: no zero-padding. We want exact T coverage.
            # return_complex=True: gives complex tensor with .real / .imag.
            S_pred = torch.stft(
                pred, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )
            S_true = torch.stft(
                true, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )

            # Safe magnitude for pred: sqrt(re² + im² + ε).
            # DO NOT use S_pred.abs() here. PyTorch's complex abs backward
            # computes conj(z) / (2|z|), which blows up when |z| → 0.
            # Early in training, y_pred ≈ 0 for ~90% of series → |S_pred| ≈ 0
            # → gradient → ∞ → 3e9 predictions. Found this the hard way.
            # sqrt(re² + im² + ε) gradient = re / sqrt(re² + im² + ε) ∈ (-1,1)
            # always, for any finite re, im. Bounded. No ε bias (same ε on both
            # terms in the difference cancels — but we only need it on pred
            # since S_true has no gradient).
            mag_pred = torch.sqrt(S_pred.real ** 2 + S_pred.imag ** 2 + 1e-8)
            mag_true = S_true.abs()  # truth: not in graph, no gradient, .abs() fine

            # Zero out DC bin (frequency bin 0) before comparing.
            # DC captures windowed mean — the pointwise loss already handles
            # mean calibration with the correct sign. Leaving DC in creates
            # a conflicting gradient: when ŷ < y in mean, |DC_pred| < |DC_true|,
            # STFT gradient pushes ŷ DOWN (wrong). Masking makes spectral
            # purely an AC temporal-structure term with no overlap with
            # the pointwise loss.
            mag_pred = mag_pred.clone()
            mag_true = mag_true.clone()
            mag_pred[:, 0, :] = 0.0
            mag_true[:, 0, :] = 0.0

            total = total + self._log_cosh(mag_pred - mag_true).mean()
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

        # ── Pointwise loss in log1p space ─────────────────────────────
        # Model receives log1p-transformed targets (LogTransform scaler).
        # log_cosh(ŷ - y) in log1p space directly optimises MSLE —
        # no transformation chain needed.
        #
        # Gradient = tanh(ŷ - y) ∈ (−1, 1), always bounded.
        #
        # Upward/downward pressure:
        #   Event underprediction (ŷ < y): e < 0, tanh(e) < 0 → pushes ŷ up ✓
        #   Event overprediction  (ŷ > y): e > 0, tanh(e) > 0 → pushes ŷ down ✓
        #   Peace correct  (ŷ = 0, y = 0): e = 0, tanh(0) = 0 → no pressure ✓
        #   Peace overfit  (ŷ > 0, y = 0): e > 0, tanh(e) > 0 → pushes ŷ down ✓
        #
        # The moment ŷ drifts positive on peace cells, downward correction
        # activates automatically. No DC/AC decomposition needed — the
        # log1p space geometry self-corrects by construction.
        e = y_pred - y_true
        cell_loss = self._log_cosh(e)

        # ── Flat event boost ──────────────────────────────────────────
        # Events get (1 + alpha)× weight — flat across all event sizes.
        # The log1p transform already provides correct magnitude-aware
        # gradients (a 1-death miss and a 50k-death miss of the same
        # log-ratio get equal gradient). The flat boost addresses only
        # class imbalance: 90% peace cells outnumber events 9:1.
        abs_y = torch.abs(y_true)
        is_event = abs_y > self.non_zero_threshold
        w = torch.where(is_event, 1.0 + self.alpha, torch.ones_like(y_true))
        per_sample = w * cell_loss

        # ── Budget allocation ─────────────────────────────────────────
        if self.dual_mean:
            loss_main = self._balanced_mean(per_sample, is_event)
        else:
            loss_main = per_sample.mean()

        # ── Spectral — AC bins only ───────────────────────────────────
        # DC bin is masked; spectral is purely temporal structure.
        # Flat forecasts cannot be locally optimal when δ > 0: a flat
        # ŷ = c has zero energy in all AC bins. Any truth with temporal
        # variation has non-zero |S_true[k]| for k > 0, so gradient is
        # non-zero and pushes ŷ away from flat toward the frequencies
        # present in truth. Seasonal patterns (n_fft=12, bin 1 = annual
        # cycle) must be learned — the loss penalises their absence.
        loss_spectral = y_pred.new_tensor(0.0)
        if self.delta > 0.0 and y_pred.size(1) >= 6:
            loss_spectral = self._spectral_loss(y_pred, y_true)

        total_loss = loss_main + self.delta * loss_spectral

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLoss: main={loss_main.item():.6f} "
                f"spectral={loss_spectral.item():.6f}"
            )

        logger.debug(
            "SpotlightLoss | main=%.6f spec=%.6f total=%.6f",
            loss_main.item(),
            loss_spectral.item(),
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLoss(alpha={self.alpha}, delta={self.delta}, dual_mean={self.dual_mean}, "
            f"event_weight={self.event_weight}, non_zero_threshold={self.non_zero_threshold})"
        )
