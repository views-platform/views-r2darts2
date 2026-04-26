import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PrismLoss(torch.nn.Module):
    """
    PrismLoss v36 — MSLE with adaptive weighting and DRO aggregation.

    Built for UCDP GED conflict fatality forecasting where ~90% of data
    is zeros (peace) and the remaining 10% spans four orders of magnitude
    in raw deaths.

    Four stages:

    ── Stage 1: MSE in transformed space (= MSLE) ────────────────────

    The model operates in transformed space (log1p / asinh / LSE scaler).
    The base cell loss is plain MSE::

        L_cell = 0.5 × (ŷ − y)²

    In log1p space this is literally MSLE — direct optimisation of the
    target metric, no proxy.

    ── Stage 2: adaptive compound weighting ──────────────────────────

    Per-cell weight is the product of two bounded, parameter-free
    signals — difficulty (how wrong) and importance (how consequential)::

        difficulty = 1 − exp(−|e|)                ∈ [0, 1)
        importance = 1 − exp(−max(|y|, |ŷ_sg|))  ∈ [0, 1)
        w = 1 + difficulty × importance           ∈ [1, 2)

    Normalised to mean=1. Steers gradient direction toward cells
    that are both hard AND important. Detached — no gradient coupling.

    ── Stage 3: χ²-DRO tail aggregation ────────────────────────────

    Replaces event_weight / balanced_mean. Parameter-free, label-free.

    Instead of a plain mean over weighted cell losses, compute the
    closed-form dual of distributionally robust optimisation over a
    unit-radius χ²-divergence ball (Duchi & Namkoong 2018):

        z = (l − mean(l)) / std(l)     # z-score of each cell loss
        dro_w = clamp(1 + z, min=0)     # χ²-DRO dual solution
        dro_w = dro_w / mean(dro_w)     # normalise to mean=1
        loss = mean(dro_w × l)

    Properties:
    - **Tail-focusing**: cells with above-average loss get w > 1;
      cells with below-average loss get w < 1 or w = 0. In the
      90/10 peace/event distribution, event cells dominate the tail
      and receive ~75-85% of gradient budget automatically.
    - **No labels needed**: operates purely on the loss distribution.
      No event_weight, no non_zero_threshold for budget allocation.
    - **No hyperparameter**: unit-radius χ² ball is the canonical
      default. The z-score is the natural parameterisation.
    - **Self-correcting**: as the model improves on events, their
      losses shrink and exit the tail. Budget redistributes to
      whatever is currently hardest — natural curriculum.
    - **Prevents zero-attractor**: a model predicting ≈0 everywhere
      has high loss on event cells, which dominate the DRO tail.
      The model can only reduce DRO loss by learning events.
    - **Detached**: z-scores computed from detached losses.

    Stage 2 (compound weight) and Stage 3 (DRO aggregation) address
    orthogonal problems: compound weight steers *which cells matter
    per-se* (hard + important); DRO steers *how losses are aggregated*
    (tail-focused, not count-dominated).

    ── Stage 4: multi-resolution spectral (temporal coherence) ───────

    Unchanged from v33. Compares STFT magnitude spectra at 3
    resolutions via log_cosh. DC bin masked — pointwise handles level,
    spectral handles AC temporal structure only.

    ─────────────────────────────────────────────────────────────────────

    Args:
        delta: Spectral loss weight. 0.0 = disable (pointwise only).
            0.10 = spectral is ~15% of gradient. Range: 0.05–0.20.
        non_zero_threshold: Transformed-space cutoff for spectral
            signal filtering. 0.693 = log1p(1). Not used in pointwise
            loss — DRO handles budget allocation without labels.

    Example:
        >>> loss_fn = PrismLoss(delta=0.10, non_zero_threshold=0.693)
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
        non_zero_threshold: float,
        event_weight: float = 0.0,  # deprecated — ignored, kept for backward compat
    ):
        if event_weight != 0.0:
            logger.warning(
                "PrismLoss v36: event_weight is deprecated and ignored. "
                "χ²-DRO aggregation replaces balanced_mean. "
                "Remove event_weight from your config. (received event_weight=%.4f)",
                event_weight,
            )
        if delta < 0.0:
            raise ValueError(f"PrismLoss: delta must be non-negative, got {delta}")
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"PrismLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.delta = delta
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "PrismLoss v36 (DRO) | delta=%.4f threshold=%.4f",
            delta, non_zero_threshold,
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

        # ── χ²-DRO tail aggregation ───────────────────────────────────
        # Closed-form dual of DRO over a unit-radius χ²-divergence ball
        # (Duchi & Namkoong 2018). Replaces event_weight / balanced_mean.
        #
        # z = (l - mean(l)) / std(l): z-score of each cell loss
        # dro_w = clamp(1 + z, min=0): χ²-DRO dual solution
        # dro_w = dro_w / mean(dro_w): normalise to mean=1
        #
        # Cells with above-average loss get dro_w > 1 (more gradient).
        # Cells with below-average loss get dro_w < 1 or 0 (less/none).
        # With 90% peace (low loss) and 10% events (high loss), events
        # dominate the tail and receive ~75-85% of gradient budget.
        #
        # Prevents zero-attractor: a model predicting ≈0 everywhere has
        # high loss on events → events dominate DRO tail → model must
        # learn events to reduce DRO loss.
        loss_flat = cell_loss.detach().flatten()
        loss_std = loss_flat.std()
        if loss_std > 1e-8:
            z = (loss_flat - loss_flat.mean()) / loss_std
            dro_w = (1.0 + z).clamp(min=0.0)
            dro_w = dro_w / dro_w.mean()
            dro_w = dro_w.view_as(cell_loss)
            loss_main = (dro_w * cell_loss).mean()
        else:
            # All losses identical (degenerate batch) — plain mean
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
            f"non_zero_threshold={self.non_zero_threshold})"
        )
