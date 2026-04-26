import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PrismLoss(torch.nn.Module):
    """
    PrismLoss v35 — MSLE with adaptive difficulty weighting.

    Built for UCDP GED conflict fatality forecasting where ~90% of data
    is zeros (peace) and the remaining 10% spans four orders of magnitude
    in raw deaths.

    Three stages:

    ── Stage 1: MSE in transformed space (= MSLE) ────────────────────

    The model operates in transformed space (log1p / asinh / LSE scaler).
    The base cell loss is plain MSE::

        L_cell = 0.5 × (ŷ − y)²

    In log1p space this is literally MSLE — direct optimisation of the
    target metric, no proxy.

    MSE gradient = (ŷ − y), proportional to error. A cell that's off
    by 5 gets 5× the correction of a cell off by 1.
    MSLE penalises it 25× more (e²). The gradients are aligned.

    ── Stage 2: adaptive compound weighting ──────────────────────────

    Per-cell weight is the product of two bounded, parameter-free
    signals — difficulty (how wrong) and importance (how consequential)::

        difficulty = 1 − exp(−|e|)                ∈ [0, 1)
        importance = 1 − exp(−max(|y|, |ŷ_sg|))  ∈ [0, 1)
        w = 1 + difficulty × importance           ∈ [1, 2)

    Both signals must be present for high weight. A cell must be
    **both hard AND important** to receive full emphasis.

    Properties:
    - **Curriculum**: cells the model predicts well (|e|→0)
      get difficulty→0 → w→1 regardless of magnitude. Gradient
      budget automatically migrates to remaining hard cells.
    - **Magnitude-aware**: at equal error, high-magnitude cells
      (importance≈1) get more weight than low-magnitude cells
      (importance<1). A 50k-death miss at |e|=1 gets w≈1.63;
      a 10-death miss at |e|=1 gets w≈1.57. Not a large gap,
      but MSE quadratic (0.5×e²) provides the primary within-
      event separation — the weight is a steering signal on top.
    - **Bounded**: w ∈ [1, 2) always. No extrapolation risk.
    - **No hyperparameter**: both difficulty and importance use
      1−exp(−x), the maximum-entropy CDF for non-negative input.
    - **Symmetric**: false alarms (y=0, ŷ=8) get high difficulty
      AND high importance (via detached |ŷ|). Same weight as
      misses (y=8, ŷ=0).
    - **Normalised**: weights are divided by their batch mean
      before application. This makes weights relative (mean=1),
      so total gradient magnitude is independent of batch
      composition. Only the gradient *direction* changes. Fixes
      effective-LR drift between hard and easy batches.
    - **Detached**: no gradient flows through the weight.

    Comparison to v34 (β-based):
        β=0.10, Ukraine |y|=11: w = 1 + 0.10×11 = 2.10 (always, even
        if the model predicts Ukraine perfectly).
        v35, Ukraine well-predicted (|e|=0.2): w ≈ 1.18. Poorly
        predicted (|e|=3): w ≈ 1.95. Budget goes where it's needed.
        At equal |e|=1: 50k-death cell gets w=1.63 vs 10-death
        cell gets w=1.57 — magnitude awareness preserved.

    ── Stage 3: multi-resolution spectral (temporal coherence) ───────

    Unchanged from v33. Compares STFT magnitude spectra at 3
    resolutions via log_cosh. DC bin masked — pointwise handles level,
    spectral handles AC temporal structure only.

    ─────────────────────────────────────────────────────────────────────

    Args:
        delta: Spectral loss weight. 0.0 = disable (pointwise only).
            0.10 = spectral is ~15% of gradient. Range: 0.05–0.20.
        event_weight: Class-balanced mean budget fraction for event cells.
            0.0 = plain per-cell mean (MSLE exactly, no class imbalance correction).
            0.5 = 50/50 split (5× per-cell boost over natural 10% prevalence).
            Range: 0.0–0.5.
        non_zero_threshold: Transformed-space cutoff for event vs peace.
            0.693 = log1p(1), 0.88 = asinh(1), 0.95 = lse(1, κ=10).

    Example:
        >>> loss_fn = PrismLoss(delta=0.10,
        ...                     event_weight=0.0, non_zero_threshold=0.693)
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
        delta: float,
        event_weight: float,
        non_zero_threshold: float,
    ):
        if delta < 0.0:
            raise ValueError(f"PrismLoss: delta must be non-negative, got {delta}")
        if not (0.0 <= event_weight <= 0.5):
            raise ValueError(
                f"PrismLoss: event_weight must be in [0, 0.5], got {event_weight}"
            )
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"PrismLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.delta = delta
        self.event_weight = event_weight
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "PrismLoss v35 (adaptive) | delta=%.4f event_weight=%.4f threshold=%.4f",
            delta, event_weight, non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers — the building blocks I've rewritten 20 times
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Plain log(cosh(x)). Numerically stable identity: |x| + softplus(-2|x|) - ln2.

        Used for spectral magnitude comparison only. Bounded gradient (tanh)
        prevents any single frequency bin from dominating the regulariser.
        """
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _balanced_mean(
        self, per_sample: torch.Tensor, is_active: torch.Tensor
    ) -> torch.Tensor:
        """Weighted split between active (events + false alarms) and peace cells.

        Without this, 90% peace gradient drowns 10% active gradient.

        Each group (active / peace) gets averaged separately, then
        combined with event_weight / (1−event_weight). If one group
        is empty in a batch (rare but happens), the other gets 100%.
        """
        # Handle multi-target: unsqueeze to (N, C) for per-channel balance
        if per_sample.dim() == 2:
            per_sample = per_sample.unsqueeze(-1)
            is_active = is_active.unsqueeze(-1)

        C = per_sample.size(-1)
        ps_flat = per_sample.reshape(-1, C)  # (batch×seq, channels)
        ia_flat = is_active.reshape(-1, C)

        n_active = ia_flat.sum(0)
        n_peace = (~ia_flat).sum(0)

        # clamp(min=1) prevents div-by-zero when a group is empty
        loss_active = (ps_flat * ia_flat).sum(0) / n_active.clamp(min=1)
        loss_peace = (ps_flat * ~ia_flat).sum(0) / n_peace.clamp(min=1)

        # event_weight for active cells, (1−event_weight) for peace.
        # If one group is absent, the other gets 100%.
        w_a = self.event_weight * (n_active > 0).float()
        w_p = (1.0 - self.event_weight) * (n_peace > 0).float()
        total_w = (w_a + w_p).clamp(min=1e-8)  # paranoia clamp

        return ((w_a / total_w) * loss_active + (w_p / total_w) * loss_peace).mean()

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

        # ── Pointwise MSE in transformed space ───────────────────────
        e = y_pred - y_true
        cell_loss = 0.5 * e * e

        # ── Adaptive compound weighting ────────────────────────────────
        # difficulty = 1 − exp(−|e|) : how wrong the model is (curriculum)
        # importance = 1 − exp(−mag) : how consequential the cell is
        # w = 1 + difficulty × importance : both must be present
        #
        # Normalised by w.mean() so weights are relative (mean=1).
        # Total gradient magnitude stays constant across batches;
        # only direction changes (toward hard + important cells).
        abs_e = torch.abs(e.detach())
        abs_y = torch.abs(y_true)
        abs_y_hat = torch.abs(y_pred.detach())
        magnitude = torch.max(abs_y, abs_y_hat)

        difficulty = 1.0 - torch.exp(-abs_e)
        importance = 1.0 - torch.exp(-magnitude)
        w = 1.0 + difficulty * importance
        w = w / w.mean()  # relative weights, mean=1
        cell_loss = w * cell_loss

        # ── Budget allocation ─────────────────────────────────────────
        is_active = (abs_y > self.non_zero_threshold) | (abs_y_hat > self.non_zero_threshold)
        if self.event_weight > 0.0:
            loss_main = self._balanced_mean(cell_loss, is_active)
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
            f"PrismLoss(delta={self.delta}, "
            f"event_weight={self.event_weight}, non_zero_threshold={self.non_zero_threshold})"
        )
