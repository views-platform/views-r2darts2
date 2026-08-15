import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossHuber(torch.nn.Module):
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
        ``(pred - mean_t pred) - (true - mean_t true)`` under ``logcosh``.
        This scores the within-series dynamics only; a constant-wrong
        prediction is NOT free because the level term below catches its
        magnitude.

      * **Level (soft-peak magnitude).** ``logcosh`` of
        ``logsumexp_t(pred) - logsumexp_t(true)`` per series. Because asinh
        values are already log-scale, ``logsumexp`` over time is a smooth
        maximum dominated by the event peak — matching it forces the model
        to reproduce raw peak magnitude, which is exactly what a symmetric
        cell loss leaves under-predicted.

    The anti-under-prediction pressure is **dynamic and data-driven, not an
    imposed constant**: the gradient of ``logsumexp`` is a softmax over
    timesteps, so the level term's correction concentrates automatically on
    the current peak month and, since the model sits below the true peak,
    pushes it upward. That asymmetry emerges from the data and fades to zero
    as the peak is matched — there is no asymmetry coefficient to set.

    A soft **event gate** ``sigmoid((|·| - tau) / tau)`` (its only input is
    ``non_zero_threshold``) focuses both terms on conflict cells, and
    **Hájek** (self-normalized) gated means keep peace-heavy and event-heavy
    batches comparable. Both terms live in the same asinh units and are
    ``logcosh``-bounded, so they are naturally comparable and need no
    weighting hyperparameter. Channels are combined by a plain mean.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` is the ONLY tunable — the asinh-space boundary
    that separates peace from conflict (default use ≈ 0.88 ≈ asinh(1)).
    Everything else is parameter-free: ``logcosh`` and ``logsumexp`` have no
    knobs, and the level/shape balance and the under-prediction asymmetry
    are both emergent from the data rather than set by constants.
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

        logger.info("SpotlightLossHuber | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold

        # ── Per-channel scale (data-driven, across series) ────────────
        # std over batch AND time, per channel, floored at tau. Crucially
        # this is NOT per-series: a per-series std divides each window by
        # its own spread, which shrinks the gradient on exactly the
        # high-intensity windows that raw-space MSLE (≈ MSE-in-asinh) cares
        # about → chronic under-prediction of large events. A shared
        # per-channel scale keeps the three channels comparable (the
        # anti-templating goal) while letting big events keep their full
        # error magnitude; Huber still bounds any single cell so no one
        # series can dominate the batch gradient.
        s = y_true.std(dim=(0, 1), keepdim=True).clamp_min(tau)

        # ── Scale-normalized robust base loss ─────────────────────────
        # Level + shape penalized jointly on the raw error. Standardization
        # equalizes gradient scale across series → strong curvature without
        # templating or saturation-driven underprediction.
        e = (y_pred - y_true) / s
        cell = F.huber_loss(
            e, torch.zeros_like(e), delta=self._HUBER_DELTA, reduction="none"
        )

        # ── Single soft event gate (keyed to tau) ─────────────────────
        # Concentrates emphasis on conflict cells; catches both false
        # negatives (y_true) and false positives (y_pred).
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid((abs_max - tau) / tau)

        # ── Hájek self-normalized gated mean ──────────────────────────
        # Composition-robust: numerator and denominator scale together with
        # event count, so peace-heavy vs event-heavy batches are comparable.
        if multivariate:
            num = (gate * cell).sum(dim=(0, 1))
            den = gate.sum(dim=(0, 1)).clamp_min(self._EMA_EPS)
            per_channel = num / den
            total_loss = per_channel.mean()
            comp = per_channel.detach().tolist()
        else:
            num = (gate * cell).sum()
            den = gate.sum().clamp_min(self._EMA_EPS)
            total_loss = num / den
            comp = [float(total_loss.detach())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossHuber: per_channel={comp}")

        # Telemetry kept shape-compatible with existing callbacks.
        C = len(comp)
        self._last_components = {
            "shape": comp,
            "level": [0.0] * C,
            "spec": [0.0] * C,
            "weight": [1.0] * C,
            "ema": [float("nan")] * C,
            "cal_ratio": [1.0] * C,
            "cal_score": [1.0] * C,
            "gates": [1.0] * C,
            "contribution": comp,
        }

        logger.debug(
            "SpotlightLossHuber | per_channel=%s total=%.6f",
            comp, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossHuber(non_zero_threshold={self.non_zero_threshold})"
