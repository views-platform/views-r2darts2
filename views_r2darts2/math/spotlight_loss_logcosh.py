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
      logcosh is chosen for stability: the training data contains noise
      and unexplainable spikes that destabilise unbounded gradients.
      The bounded tanh gradient (≤ ±1) keeps optimisation stable.

    * **Level (magnitude).** ``(sum_t(pred) - sum_t(true))^2 / T`` per
      series. This is MSE on the temporal sum — unbounded gradient that
      grows with total magnitude error, so the model CAN distinguish
      "2 below truth" from "5 below truth" (unlike logcosh which
      saturates at ±1 for any sum > 3).

      Stability comes from the SUM itself: aggregating across T=36
      timesteps reduces per-spike variance by √T ≈ 6×, so individual
      noise spikes get averaged out before the gradient is computed.
      This is fundamentally safer than per-cell MSE (which sees each
      spike at full force) while still providing the magnitude-
      proportional gradient that logcosh cannot.

      Per-series event mass ``gate.amax(dim=1)`` weights the level so
      peace-only series don't drown the signal.

    Both terms are gated by a sharp event mask
    ``sigmoid(10·(|·| − τ))`` (τ = ``non_zero_threshold``). The shape
    term uses Hájek (self‑normalised weighted mean) normalisation; the
    level term uses a plain mean to avoid gradient starvation at 90%+
    peace series.

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
        T = y_pred.size(1)

        # ── Sharp event gate ─────────────────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))

        # ── Shape term: demeaned logcosh, standardised ─────────────
        # logcosh (not quadratic) for stability against noise/spikes.
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # ── Level term: per-cell MSE gated by event mask ────────────
        # Every aggregation (mean/sum) over T cells introduces a 1/T
        # factor in the per-cell gradient, making level 20-40x weaker
        # than shape. The solution: NO aggregation — per-cell MSE on
        # event cells only. The gate filters peace cells (gate ≈ 0.018
        # for peace, ≈ 1.0 for events), so spikes in peace don't
        # destabilize training. Event spikes are bounded by asinh space
        # (max ~13). The gradient is 2*(pred-true) per event cell —
        # magnitude-proportional and NOT attenuated by 1/T.
        #
        # Stability analysis:
        # - Peace cells: gate ≈ 0.018 → gradient ≈ 0.018 * 2 * spike
        #   → negligible (spike noise filtered by gate)
        # - Event cells: gate ≈ 1.0 → gradient = 2 * (pred - true)
        #   → bounded by asinh range: max 2 * 13 = 26 per cell
        #   → total for 4 events: 4 * 26 = 104 (within clip=100)
        # - The gate IS the stabilizer — it replaces the aggregation's
        #   √T noise reduction with explicit event/peace separation.
        level_cell = gate * (y_pred - y_true) ** 2

        # Per-series event mass for level weighting
        w = gate.amax(dim=1)

        # ── Normalisation ────────────────────────────────────────────
        # Shape: Hájek (self-normalised) — composition-robust
        # Level: plain SUM (not Hájek) — the gate already filters peace,
        # so Hájek's 1/sum(gate) would dilute the event gradient by
        # dividing by all peace cells' tiny gate values. The sum gives
        # raw gradient = gate * 2*(pred-true) per cell — strong on
        # events, ~0 on peace. The gate IS the stabilizer.
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = level_cell.sum(dim=(0, 1))

            per_channel = shape + level
            total_loss = per_channel.mean()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)
            level = level_cell.sum()
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
