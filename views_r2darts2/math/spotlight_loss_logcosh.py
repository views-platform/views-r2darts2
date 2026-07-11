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
      penalised quadratically and standardised by the channel‑wise
      temporal standard deviation of the true AC component.
      This ensures each series is forced to learn its own dynamics
      and flat forecasts are actively penalised.

    * **Level (magnitude).** Per‑cell Huber loss on raw error,
      with an adaptive threshold ``δ`` set per channel as
      ``δ = 2 * std(event‑cell true values)``, floored at the
      non‑zero threshold.  The gradient grows proportionally with
      moderate errors (up to ``δ``) and saturates gracefully for
      extreme outliers, providing far more lifting force than
      saturating losses while preventing gradient explosions.

    Both terms are gated by a sharp event mask
    ``sigmoid(10·(|·| − τ))`` (τ = ``non_zero_threshold``) and
    normalised with Hájek (self‑normalised weighted means).
    The shape term is given an explicit relative weight of 5.0
    to maintain its influence alongside the stronger level signal.

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    All other quantities (shapes, scales, Huber threshold) are
    data‑driven.
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
    # Helper: stable logcosh (used only if needed; kept for reference)
    # ------------------------------------------------------------------
    @staticmethod
    def _Logcosh(z: torch.Tensor) -> torch.Tensor:
        """log(cosh(z)) without the additive constant log(2)."""
        a = z.abs()
        return a + F.softplus(-2.0 * a)

    # ------------------------------------------------------------------
    # Helper: per‑channel std of event cells (for adaptive Huber delta)
    # ------------------------------------------------------------------
    @staticmethod
    def _event_std(y_true: torch.Tensor, tau: float) -> torch.Tensor:
        """Per‑channel standard deviation of y_true for |y_true| > tau."""
        mask = (y_true.abs() > tau).float()                     # (B, T, C)
        y_event = y_true * mask
        s1 = y_event.sum(dim=(0, 1))                            # (C,)
        s2 = (y_event ** 2).sum(dim=(0, 1))
        n = mask.sum(dim=(0, 1)).clamp_min(1e-6)                # (C,)
        mean = s1 / n
        var = (s2 / n) - (mean ** 2)
        return var.clamp_min(0.0).sqrt()

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

        # ── Shape term: demeaned quadratic, standardised ─────────────
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = ((pred_ac - true_ac) / ac_scale) ** 2

        # ── Level term: per‑cell Huber loss, adaptive delta ──────────
        # Adaptive δ = 2 × channel‑wise std of true event cells, floored at τ
        event_std = self._event_std(y_true, tau)                 # (C,) or scalar
        delta_huber = event_std.clamp_min(tau) * 2.0             # (C,) or scalar
        if multivariate:
            delta_huber = delta_huber.unsqueeze(0).unsqueeze(0)  # (1,1,C)
        else:
            # univariate: delta_huber is scalar or 1‑element tensor
            delta_huber = delta_huber.unsqueeze(0)               # make broadcastable

        raw_error = y_pred - y_true
        abs_error = raw_error.abs()
        quadratic = 0.5 * raw_error ** 2
        linear = delta_huber * (abs_error - 0.5 * delta_huber)
        level_cell_raw = torch.where(abs_error <= delta_huber, quadratic, linear)

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