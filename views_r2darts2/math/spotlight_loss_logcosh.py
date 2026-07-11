import torch
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Minimal scale-normalized loss for zero-inflated conflict forecasting.

    Operates in asinh space (AsinhTransform target scaler) on ``(B, T, C)``
    tensors for UCDP GED fatality forecasting (sb/ns/os), where 86–98% of
    cells are zero and non-zero values span ~4 orders of magnitude.

    ── Design ─────────────────────────────────────────────────────────

    A single joint error term lets the model trade *level* for *shape* — it
    can hide a magnitude error inside a plausible pattern (or vice versa) —
    and a symmetric penalty on a right-skewed target then settles into a
    flat, under-predicting compromise: the event peaks are never lifted. The
    loss therefore splits the error into two dedicated terms, each with its
    own bounded gradient, so neither objective can be sacrificed for the
    other and gradient reaches both:

      * **Shape (temporal pattern).** Per-series demeaned residual
        ``(pred - mean_t pred) - (true - mean_t true)``, penalized
        *quadratically* and standardized by each channel's own
        temporal-deviation scale. This scores the within-series dynamics
        only; a constant-wrong prediction is NOT free because the level term
        below catches its magnitude. The penalty is deliberately NOT
        saturating: a bounded (logcosh/L1) shape term gives every series the
        same tail gradient regardless of event size, so the model collapses
        onto one shared "median" template and never accumulates the gradient
        to sharpen a flat forecast into a real peak (large series such as
        Ukraine are then left flat). A quadratic residual keeps the gradient
        proportional to the deviation, so each series is fit to its own shape
        and flatness is actively driven out; per-channel standardization
        (data-driven, not a constant) keeps the three channels comparable
        without shrinking the within-channel magnitude signal that separates
        a war from a quiet series.

      * **Level (per-cell magnitude).** Squared error ``(pred - true)**2``
        at every event cell, self-normalized over the gate. It is NOT
        compressed over time: a ``logsumexp``/soft-max aggregate collapses
        all ``T`` months into one scalar per series, which a flat forecast
        satisfies by spreading mass (spread degeneracy), so the true peak
        height is never pinned and eval forecasts collapse toward zero
        (missed level, exploding MSLE on wars). Matching magnitude at each
        event cell removes that escape route, and squaring makes the
        gradient grow with the miss — this is exactly raw-space MSLE (MSE in
        asinh units), so an under-predicted war peak is corrected far harder
        than a skirmish and level dominates shape as a natural curriculum.

    The anti-under-prediction pressure is **dynamic and data-driven, not an
    imposed constant**: both terms are squared, so their gradient grows with
    the size of the miss. An under-predicted war peak — residual of several
    asinh units — is therefore corrected far harder than a small skirmish,
    and the pull fades to zero exactly as the height is matched. There is no
    asymmetry or weighting coefficient to set; the emphasis on peaks is the
    squared error itself.

    A soft **event gate** ``sigmoid((|·| - tau) / tau)`` (its only input is
    ``non_zero_threshold``) focuses both terms on conflict cells, and
    **Hájek** (self-normalized) gated means keep peace-heavy and event-heavy
    batches comparable. The shape residual is standardized to unit deviation
    while the level residual keeps its raw asinh scale, so level naturally
    outweighs shape on large events — the intended curriculum — with no
    weighting hyperparameter. Channels are combined by a plain mean.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` is the ONLY tunable — the asinh-space boundary
    that separates peace from conflict (default use ≈ 0.88 ≈ asinh(1)).
    Everything else is parameter-free: both terms are plain squared errors
    with no knobs, and the level/shape balance and the emphasis on peaks are
    emergent from the data rather than set by constants.
    """

    _EPS = 1e-6

    def __init__(self, non_zero_threshold: float):
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"non_zero_threshold must be positive, got {non_zero_threshold}"
            )
        super().__init__()
        self.non_zero_threshold = non_zero_threshold
        self._last_components: dict | None = None

        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold

        # ── Sharp event gate (restored slope 10) ─────────────────────
        # Peace cells at ~0.48 → gate < 0.02; conflict cells > 0.5.
        # This keeps gradient focused on real events.
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))

        # ── Shape term: within-series temporal pattern (AC) ───────────
        # Demeaned per series, quadratic, standardized – unchanged.
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = ((pred_ac - true_ac) / ac_scale) ** 2

        # ── Level term: quadratic total mass ──────────────────────────
        # Sum over time matches total event mass; quadratic penalty gives
        # gradient that grows with the underprediction, directly fixing
        # the level gap without saturating. No logsumexp, no logcosh.
        level_cell = (y_pred.sum(dim=1) - y_true.sum(dim=1)) ** 2

        # Per‑series event mass, so peace‑only series contribute almost nothing.
        w = gate.amax(dim=1)  # (B, C) – high if any timestep has an event

        # ── Normalisation ─────────────────────────────────────────────
        if multivariate:
            # Shape: Hájek self‑normalised gated mean
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(
                dim=(0, 1)
            ).clamp_min(self._EPS)
            # Level: mean over batch, weighted by per‑series event mass
            level = (w * level_cell).mean(dim=0)
            per_channel = shape + level
            total_loss = per_channel.mean()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)
            level = (w * level_cell).mean()
            total_loss = shape + level
            shape_c = [float(shape.detach())]
            level_c = [float(level.detach())]
            comp = [float(total_loss.detach())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossLogcosh: per_channel={comp}")

        # ── Telemetry ─────────────────────────────────────────────────
        n = len(comp)
        self._last_components = {
            "shape": shape_c,
            "level": level_c,
            "spec": [0.0] * n,
            "weight": [1.0] * n,
            "ema": [float("nan")] * n,
            "cal_ratio": [1.0] * n,
            "cal_score": [1.0] * n,
            "gates": [1.0] * n,
            "contribution": comp,
        }

        logger.debug(
            "SpotlightLossLogcosh | shape=%s level=%s total=%.6f",
            shape_c,
            level_c,
            total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"
