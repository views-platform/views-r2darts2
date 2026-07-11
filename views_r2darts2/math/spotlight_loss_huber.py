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
    demeaning). Magnitude imbalance across series (Ukraine-scale asinh≈13 vs
    low-intensity asinh≈1) is removed by **per-series, per-channel
    standardization**: the error is divided by the series' own asinh-space
    standard deviation, floored at ``non_zero_threshold``. This makes every
    series contribute comparable gradient magnitude, which:

      * prevents templating (no single high-magnitude series dominates the
        batch gradient), so a strong convex loss can be used without the
        gradient saturation that causes underprediction; and
      * scales automatically across the three channels despite very
        different sparsity, with no per-channel hyperparameters.

    A robust **Huber** penalty on the standardized error keeps large events
    influential without exploding on outliers. A single soft **event gate**
    (keyed to ``non_zero_threshold``) concentrates emphasis on conflict
    cells, and a **Hájek** (self-normalized) gated mean keeps peace-heavy
    and event-heavy batches comparable. Channels are combined by a plain
    mean.

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

        # ── Per-series, per-channel scale (data-driven) ───────────────
        # std over the time axis; floored at tau so peace-only series
        # (near-zero variance) do not blow up the normalized error.
        s = y_true.std(dim=1, keepdim=True).clamp_min(tau)

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
