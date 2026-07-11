import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Adaptive loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).
    The loss is designed for ~86‑98% zeros and heavy‑tailed non‑zero values.

    ── Design ─────────────────────────────────────────────────────────

    The error is split into two complementary terms:

    * **Shape (temporal pattern).** Demeaned per‑series residual,
      penalised with logcosh and standardised by the channel‑wise
      temporal standard deviation of the true AC component.
      logcosh is chosen over quadratic for stability: the training data
      contains noise and unexplainable spikes that destabilise unbounded
      gradients. The bounded tanh gradient (≤ ±1) keeps optimisation
      stable while still penalising flat forecasts through the
      standardised residual.

    * **Level (magnitude).** ``logcosh`` of ``sum_t(pred) - sum_t(true)``
      per series. The key insight: **aggregate magnitude BEFORE applying
      logcosh**, not after. Per‑cell logcosh gives each cell its own
      bounded gradient (tanh(e_t) ≤ 1), so a Ukraine‑scale error (e=9)
      gets the same gradient as a small error (e=2) — magnitude is
      invisible. By summing across timesteps first, the logcosh sees
      the TOTAL error: 36 months of underprediction by 2 → sum=72 →
      tanh(72)=1.0 → ALL cells get maximum gradient. 1 month of error
      by 2 → sum=2 → tanh(2)=0.96 → 1 cell gets gradient. The gradient
      is still bounded (≤ 1 per cell, stable) but now reflects TOTAL
      magnitude — sustained underprediction triggers a broad, strong
      correction rather than a weak per‑cell one.

    Both terms are gated by a sharp event mask
    ``sigmoid(10·(|·| − τ))`` (τ = ``non_zero_threshold``) and the shape
    term uses Hájek (self‑normalised weighted mean) normalisation. The
    level term uses a plain mean (not Hájek) to avoid gradient starvation
    at 90%+ peace series. Shape and level are added with equal weight –
    no extra scaling constant.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    All other quantities are data‑driven.
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
    # Helper: stable logcosh
    # ------------------------------------------------------------------
    @staticmethod
    def _Logcosh(z: torch.Tensor) -> torch.Tensor:
        """log(cosh(z)) without the additive constant log(2)."""
        a = z.abs()
        return a + F.softplus(-2.0 * a)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold

        # ── Sharp event gate ─────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))

        # ── Shape term: demeaned logcosh, standardised ─────────────
        # logcosh (not quadratic) for stability against noise/spikes.
        # The bounded tanh gradient (≤1) prevents destabilisation while
        # the standardised residual still penalises flat forecasts.
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # ── Level term: logcosh(SUM) — magnitude-aware ──────────────
        # Sum across timesteps BEFORE logcosh. This aggregates total
        # magnitude error: sustained underprediction → large sum →
        # tanh(sum) ≈ ±1 → ALL cells get max gradient (bounded, stable).
        # Per-cell logcosh cannot distinguish e=2 from e=9 (both tanh≈1);
        # sum-then-logcosh distinguishes "1 month wrong" from "36 months
        # wrong" by the number of cells that receive the gradient.
        level_cell = self._Logcosh(y_pred.sum(dim=1) - y_true.sum(dim=1))

        # Per-series event mass for level weighting
        w = gate.amax(dim=1)

        # ── Normalisation ────────────────────────────────────────────
        # Shape: Hájek (self-normalised) — composition-robust
        # Level: plain mean — avoids gradient starvation at 90%+ peace
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
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

        # ── Telemetry ────────────────────────────────────────────────
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
            shape_c, level_c, total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"
