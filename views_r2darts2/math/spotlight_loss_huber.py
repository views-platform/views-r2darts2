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

    The loss scores the **raw error** ``e = y_pred - y_true`` — level and
    shape are penalized jointly, so a constant-correct prediction is
    correctly rewarded and bursty predictions are not "free" (no per-window
    demeaning). Channel magnitude imbalance (sb/ns/os differ in sparsity and
    scale) is removed by **per-channel standardization**: the error is
    divided by the channel's own asinh-space standard deviation (computed
    across series and time), floored at ``non_zero_threshold``. This is a
    *shared* per-channel scale, deliberately NOT per-series — a per-series
    scale divides each window by its own spread, shrinking the gradient on
    the high-intensity windows that raw-space MSLE (≈ MSE-in-asinh) most
    cares about, which causes chronic under-prediction of large events. The
    shared scale instead:

      * keeps the three channels contributing comparable gradient magnitude
        (no per-channel hyperparameters); while
      * preserving each series' true error magnitude, so large events are
        pushed up rather than equalized away.

    A robust **Huber** penalty on the standardized error keeps large events
    influential without letting any single series dominate the batch
    gradient (its bounded gradient is what prevents templating). A single
    soft **event gate** (keyed to ``non_zero_threshold``) concentrates
    emphasis on conflict cells, and a **Hájek** (self-normalized) gated mean
    keeps peace-heavy and event-heavy batches comparable. Channels are
    combined by a plain mean.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` is the ONLY tunable — the asinh-space boundary
    that separates peace from conflict (default use ≈ 0.88 ≈ asinh(1)). The
    Huber transition (delta=1.0) and the gate sharpness are fixed structural
    conventions, not tuning knobs.
    """

    _EMA_EPS = 1e-6
    _HUBER_DELTA = 1.0

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
