import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PrismLoss(torch.nn.Module):
    """
    PrismLoss v33.

    Built for UCDP GED conflict fatality forecasting where ~90% of data
    is zeros (peace) and the remaining 10% spans four orders of magnitude
    in raw deaths.

    Two stages:

    ── Stage 1: MSE in log1p space (= MSLE) + class-balanced mean ────

    The model operates in log1p space (LogTransform target scaler).
    The base cell loss is plain MSE::

        L_cell = 0.5 × (ŷ − y)²

    In log1p space this is literally MSLE — direct optimisation of the
    target metric, no proxy. Previous versions used log_cosh(e) which
    approximates MSE near zero but caps gradient at tanh(e) → ±1 for
    |e| > 2. This created a 10–16× gradient deficit on the hardest
    cells (the ones MSLE penalises most), plateauing MSLE at ~0.5.

    MSE gradient = (ŷ − y), proportional to error. A cell that's off
    by 5 in log1p space gets 5× the correction of a cell off by 1.
    MSLE penalises it 25× more (e²). The gradients are aligned.

    Class balance comes from dual_mean: event and peace cells are
    averaged separately, then combined with event_weight ratio. This
    controls what fraction of gradient budget goes to events vs peace.
    No per-cell alpha boost needed — MSE's quadratic scaling naturally
    upweights large errors (which are events).

    In log1p space, values range from 0 (peace) to ~11 (50k deaths).
    Max MSE gradient ≈ 11. Safe with gradient_clip_val=10.

    ── Stage 2: multi-resolution spectral (temporal coherence) ───────

    Problem: 36 output steps produced simultaneously from shared MLP
    weights. No autoregressive coupling. Model has zero incentive to make
    adjacent months look coherent. Result: hockey sticks, wild oscillation,
    completely flat forecasts sometimes when there is clear seasonality.

    Solution: compare STFT magnitude spectra at 3 resolutions::

        L_spec = (1/K) Σ_k mean[ log_cosh(|S_k(ŷ)| − |S_k(y)|) ]

    Three resolutions capture different temporal scales:
        n_fft=6,  hop=3  → spike/cessation sharpness    (4 bins × 11 frames)
        n_fft=12, hop=6  → 12-month annual cycle         (7 bins × 5 frames)
        n_fft=24, hop=12 → multi-year trend + fine detail (13 bins × 2 frames)

    Phase-insensitive (magnitude only): predicting something 1 month
    early → ~zero spectral penalty. DC bin masked — pointwise handles
    level, spectral handles AC temporal structure only.

    log_cosh is used for spectral (not MSE) because bounded gradients
    prevent any single frequency bin from dominating the regulariser.

    ─────────────────────────────────────────────────────────────────────

    Args:
        delta: Spectral loss weight. 0.0 = disable (pointwise only).
            0.10 = spectral is ~15% of gradient. Range: 0.05–0.20.
            Above 0.20: spectral dominates, pointwise accuracy suffers.
        dual_mean: Whether to use the event/peace balanced mean.
            True = split into event/peace buckets, combine with event_weight
            ratio. False = plain per-cell mean (event_weight ignored).
        event_weight: Fraction of gradient budget allocated to event cells
            in the balanced mean. Only used when dual_mean=True.
            0.50 = 50/50 split (5× per-cell boost over natural prevalence).
            0.25 = moderate boost (2.85× per-cell). Range: 0.10–0.50.
        non_zero_threshold: log1p-space cutoff for event vs peace.
            0.693 = log1p(1) = "at least 1 battle death."

    Example:
        >>> loss_fn = PrismLoss(delta=0.10, dual_mean=True, event_weight=0.25, non_zero_threshold=0.693)
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
        delta: float,
        dual_mean: bool,
        event_weight: float,
        non_zero_threshold: float,
    ):
        if delta < 0.0:
            raise ValueError(f"PrismLoss: delta must be non-negative, got {delta}")
        if dual_mean and not (0.0 < event_weight <= 0.5):
            raise ValueError(
                f"PrismLoss: event_weight must be in (0, 0.5], got {event_weight}"
            )
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"PrismLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
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
            "PrismLoss v33 (MSE-native) | delta=%.4f dual_mean=%s event_weight=%.4f threshold=%.4f",
            delta, dual_mean, event_weight, non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers — the building blocks I've rewritten 20 times
    # ------------------------------------------------------------------

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

        # ── Pointwise MSE in log1p space = MSLE ──────────────────────
        # Model receives log1p-transformed targets (LogTransform scaler).
        # 0.5 × (ŷ - y)² in log1p space IS MSLE — direct optimisation
        # of the target metric, no proxy needed.
        #
        # Previous versions used log_cosh(e) which approximates MSE
        # near zero but caps gradient at tanh(e) → ±1 for |e| > 2.
        # This created a 10–16× gradient deficit on the hardest cells
        # (the ones MSLE penalises most), causing MSLE to plateau.
        #
        # MSE gradient = (ŷ - y), proportional to error magnitude.
        # In log1p space, max |e| ≈ 11 (log1p(50000)), so max gradient
        # ≈ 11. Safe with gradient_clip_val=10.
        e = y_pred - y_true
        cell_loss = 0.5 * e * e

        # ── Budget allocation ─────────────────────────────────────────
        # is_event uses max(|y|, |ŷ_sg|) > τ so false alarms (y=0, ŷ>τ)
        # share the event bucket with misses (y>τ, ŷ≈0). Without the
        # pred-side check, false alarms fall in the peace bucket and get
        # ~3× less per-cell gradient budget than misses of equal magnitude
        # (at event_weight=0.25, n_peace/n_event=9).
        #
        # Each cell's gradient direction is still correct regardless of bucket
        # (miss → negative e → pushes ŷ up; false alarm → positive e → pushes
        # ŷ down). Bucket membership only controls the scale factor 1/n_bucket.
        # Both misses and false alarms are high-priority cells — they share the
        # event_weight budget, ensuring equal per-cell correction scale.
        #
        # Detach pred: classification must not create a gradient path through w.
        abs_y = torch.abs(y_true)
        abs_y_hat = torch.abs(y_pred.detach())
        is_event = (abs_y > self.non_zero_threshold) | (abs_y_hat > self.non_zero_threshold)
        if self.dual_mean:
            loss_main = self._balanced_mean(cell_loss, is_event)
        else:
            loss_main = cell_loss.mean()

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
                f"NaN in PrismLoss: main={loss_main.item():.6f} "
                f"spectral={loss_spectral.item():.6f}"
            )

        logger.debug(
            "PrismLoss | main=%.6f spec=%.6f total=%.6f",
            loss_main.item(),
            loss_spectral.item(),
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"PrismLoss(delta={self.delta}, dual_mean={self.dual_mean}, "
            f"event_weight={self.event_weight}, non_zero_threshold={self.non_zero_threshold})"
        )
