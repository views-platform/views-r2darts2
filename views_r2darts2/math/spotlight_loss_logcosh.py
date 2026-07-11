import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogCosh(torch.nn.Module):
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

        logger.info("SpotlightLossLogCosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _logcosh(z: torch.Tensor) -> torch.Tensor:
        # Numerically stable log(cosh(z)), up to an irrelevant additive
        # constant (log 2). Parameter-free robust penalty: quadratic near 0,
        # linear in the tails, no delta to tune. The 2 is the mathematical
        # identity cosh(z) = (e^z + e^-z)/2, not a hyperparameter.
        a = z.abs()
        return a + F.softplus(-2.0 * a)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold

        # ── Soft event gate (its only input is tau) ───────────────────
        # Focuses both terms on conflict cells; catches false negatives
        # (via y_true) and false positives (via y_pred).
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid((abs_max - tau) / tau)

        # ── Shape term: within-series temporal pattern (AC) ───────────
        # Demeaned per series over time, so this scores dynamics only and
        # cannot absorb a magnitude error — that is the level term's job.
        pred_bar = y_pred.mean(dim=1, keepdim=True)
        true_bar = y_true.mean(dim=1, keepdim=True)
        shape_cell = self._logcosh((y_pred - pred_bar) - (y_true - true_bar))

        # ── Level term: soft-peak magnitude (DC) ──────────────────────
        # logsumexp over time is a smooth max; in asinh (log-scale) units it
        # tracks the raw event peak. Its gradient is a softmax over months,
        # so the correction concentrates on the current peak and — since the
        # model sits below the true peak — pushes it up. Dynamic, data-driven
        # asymmetry with no coefficient.
        level_cell = self._logcosh(
            torch.logsumexp(y_pred, dim=1) - torch.logsumexp(y_true, dim=1)
        )
        # Per-series event mass, so peace-only series don't drown the level
        # signal (a whole flat series contributes almost nothing).
        w = gate.amax(dim=1)

        # ── Hájek self-normalized gated means, then sum level + shape ──
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = (w * level_cell).sum(dim=0) / w.sum(dim=0).clamp_min(self._EPS)
            per_channel = shape + level
            total_loss = per_channel.mean()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)
            level = (w * level_cell).sum() / w.sum().clamp_min(self._EPS)
            total_loss = shape + level
            shape_c = [float(shape.detach())]
            level_c = [float(level.detach())]
            comp = [float(total_loss.detach())]

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossLogCosh: per_channel={comp}")

        # Telemetry: shape/level now carry their real per-channel values.
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
            "SpotlightLossLogCosh | shape=%s level=%s total=%.6f",
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogCosh(non_zero_threshold={self.non_zero_threshold})"
