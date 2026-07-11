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
      penalised with plain logcosh and standardised by the channel‑wise
      temporal standard deviation of the true AC component.
      This ensures each series is forced to learn its own dynamics
      and flat forecasts are actively penalised.

    * **Level (magnitude).** Per‑cell plain logcosh on raw error,
      weighted by event magnitude.  The gradient saturates at ±1,
      which is sufficient to correct level errors when combined with
      the magnitude weighting and shape term.

    Both terms are gated by a sharp event mask
    ``sigmoid(10·(|·| − τ))`` (τ = ``non_zero_threshold``) and
    normalised with Hájek (self‑normalised weighted means).
    The shape term is given an explicit relative weight of 5.0
    to maintain its influence alongside the level signal.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    All other quantities are data‑driven or fixed to robust values.
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
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # ── Level term: per‑cell plain logcosh on raw error ─────────
        raw_error = y_pred - y_true
        level_cell_raw = self._Logcosh(raw_error)

        # Weight level cells by event magnitude (gate × (1 + abs_max))
        mag_weight = gate * (1.0 + abs_max)                      # (B, T, C)
        level_cell = mag_weight * level_cell_raw

        # ── Normalisation (Hájek weighted means) ─────────────────────
        if multivariate:
            # Shape: gated mean
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            # Level: weighted by mag_weight
            level = level_cell.sum(dim=(0, 1)) / mag_weight.sum(dim=(0, 1)).clamp_min(self._EPS)

            # Boost shape influence to balance the level term
            w_shape = 5.0
            per_channel = w_shape * shape + level
            total_loss = per_channel.mean()
            shape_c = shape.detach().tolist()
            level_c = level.detach().tolist()
            comp = per_channel.detach().tolist()
        else:
            shape = (gate * shape_cell).sum() / gate.sum().clamp_min(self._EPS)
            level = level_cell.sum() / mag_weight.sum().clamp_min(self._EPS)
            w_shape = 5.0
            total_loss = w_shape * shape + level
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