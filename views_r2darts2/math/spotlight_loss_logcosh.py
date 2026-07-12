import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """Loss for zero‑inflated conflict fatality forecasting.

    Operates in asinh space on ``(B, T, C)`` tensors (sb/ns/os).

    ── Design ─────────────────────────────────────────────────────────

    * **Shape (per‑cell pattern).** Demeaned logcosh residual, gated
      and Hájek‑normalised. Scores temporal dynamics only. The gate
      filters peace (no pattern to learn in zeros).

    * **Level (per‑cell magnitude).** Per‑cell logcosh on the raw error
      (``pred − true``), weighted by ``log1p(abs_max)``, summed.

      No gate, no event‑count normalisation. The weight
      ``log1p(abs_max)`` is the natural and only filter:

        - **True peace** (``pred≈0, true=0``): ``abs_max≈0`` →
          ``log1p(0)=0`` → zero weight → correctly ignored.
        - **False positive** (``pred>0, true=0``): ``abs_max=pred`` →
          ``log1p(pred)>0`` → penalised → pushed DOWN toward 0.
        - **Underprediction** (``pred<true``): ``abs_max≥true`` →
          ``log1p(true)>0`` → penalised → pushed UP toward true.
        - **Correct prediction** (``pred=true``): ``raw_error=0`` →
          ``logcosh(0)=0`` → zero gradient → correctly ignored.

      This naturally focuses on spikes AND dips: false positives (dips
      in the forecast that should be 0) get strong downward gradient,
      and underpredicted events (spikes that should be higher) get
      strong upward gradient. The model learns to produce VARIABLE
      forecasts — high at events, zero at peace — instead of flat lines.

      Magnitude is proportional through EVENT COUNT: Ukraine (36 events
      × ``log1p(9)=2.3``) gets 25× more total gradient than a 1‑event
      series (1 × ``log1p(3)=1.4``). No normalisation is needed because
      ``log1p`` bounds the per‑cell weight at ``log1p(14)=2.7``, so the
      total for 36 cells is ``≈97`` (within ``gradient_clip_val=100``).

    ── Hyperparameters ────────────────────────────────────────────────

    ``non_zero_threshold`` — the only tunable, ≈ 0.88 (asinh(1)).
    Used only in the event gate for the shape term.
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

    @staticmethod
    def _Logcosh(z: torch.Tensor) -> torch.Tensor:
        a = z.abs()
        return a + F.softplus(-2.0 * a)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        multivariate = y_pred.dim() == 3
        tau = self.non_zero_threshold
        T = y_pred.size(1)

        # ── Event gate (for shape only) ──────────────────────────────
        abs_max = torch.max(y_true.abs(), y_pred.detach().abs())
        gate = torch.sigmoid(10.0 * (abs_max - tau))

        # ── Shape: demeaned logcosh, standardised, Hájek ────────────
        pred_ac = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_ac = y_true - y_true.mean(dim=1, keepdim=True)
        ac_scale = true_ac.std(dim=(0, 1), keepdim=True).clamp_min(tau)
        shape_cell = self._Logcosh((pred_ac - true_ac) / ac_scale)

        # ── Level: log1p(abs_max) × logcosh(raw_error), SUM ─────────
        # No gate, no /n_event. log1p(abs_max) is the natural filter:
        #   true peace (pred=0,true=0) → abs_max=0 → weight=0
        #   false positive (pred>0,true=0) → abs_max=pred → penalised
        #   underprediction (pred<true) → abs_max≥true → penalised
        #   correct (pred=true) → raw_error=0 → logcosh=0 → no gradient
        raw_error = y_pred - y_true
        level_cell = torch.log1p(abs_max) * self._Logcosh(raw_error)

        # ── Combine ──────────────────────────────────────────────────
        if multivariate:
            shape = (gate * shape_cell).sum(dim=(0, 1)) / gate.sum(dim=(0, 1)).clamp_min(self._EPS)
            level = level_cell.sum(dim=(0, 1))  # SUM, no gate, no /n_event

            per_channel = shape + level
            total_loss = per_channel.sum()  # SUM over channels
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
