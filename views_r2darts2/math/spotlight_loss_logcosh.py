import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v38 — DC/AC split with temporal-gradient shape loss.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros, 10% spanning
    four orders of magnitude in raw deaths.

    ── Architecture ─────────────────────────────────────────────────────

    Three components.  DC and AC are separated by construction (not by
    demeaning), eliminating the zero-sum gradient problem of v36.

    1. **Level loss (DC)** — `log_cosh_proportional(mean(ŷ − y))` per
       series.  Penalises mean prediction offset directly.  DRO-weighted
       so conflict countries get priority.

    2. **Temporal gradient loss (AC)** — the PRIMARY shape signal.
       `log_cosh_proportional(Δŷ − Δy)` where Δ is first difference.

       Why this works where e_shape failed:
       - DC-free by construction (first-difference annihilates constants)
       - NOT zero-sum: gradient telescopes to `loss'(d_1) − loss'(d_T)`,
         which is generally non-zero.  Model CAN increase output variance.
       - 13× stronger signal at onsets than pointwise cell_loss:
         flat prediction at onset (Δy=3, Δŷ=0) → proportional loss ≈ 9.5
         vs cell_loss on same cell with error ≈ 1 → loss ≈ 0.7
       - Zero gradient at plateaus: Δy=0, Δŷ≈0 → loss≈0.
         No wasted gradient on the 90% peace-cell desert.
       - Transition-event weighted: adjacent pairs near events get full
         weight; peace-peace pairs get near-zero weight.
       - Inverse-frequency rebalancing per series (same as v37).

    3. **Log-magnitude multi-resolution STFT** — secondary shape signal.
       Catches periodic/seasonal patterns that single-step gradient
       matching might miss (e.g. a 6-month conflict cycle where the
       model gets individual transitions right but misses the period).
       Same formulation as v37: `log_cosh(log1p(mag_pred) − log1p(mag_true))`
       with event-magnitude series weighting.

    ── Cross-series DRO (per component) ─────────────────────────────────

    Each component (level, shape) has its own additive-regret DRO:
    - Level DRO: upweights countries with large DC offset
    - Shape DRO: upweights countries with worst temporal-gradient fit
    Baselines are component-specific irreducible losses.

    ── Base cell loss: log_cosh × (1 + log(1+|x|²))  (proportional) ───

    Sub-linear proportional correction.  For |x| > 2, gradient grows as
    ≈ 1 + 2·ln|x| — fast enough to attend to large errors but
    sub-linear enough to prevent extreme-country errors from monopolising
    the aggregate gradient.

    ─────────────────────────────────────────────────────────────────────

    Args:
        non_zero_threshold: Transformed-space cutoff for event_magnitude.
            - AsinhTransform: 0.88 ≈ asinh(1)

    Example:
        >>> loss_fn = SpotlightLossLogcosh(non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = True

    def __init__(
        self,
        non_zero_threshold: float,
    ):
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.non_zero_threshold = non_zero_threshold
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    @staticmethod
    def _log_cosh_proportional(x: torch.Tensor) -> torch.Tensor:
        """log_cosh(x) × (1 + log(1 + |x|²)).

        Sub-linear proportional correction.  Gradient growth at large
        |x| is ≈ 1 + 2·ln|x|.  Reinforces MSLE sensitivity without
        letting extreme errors monopolise aggregate gradient.
        """
        abs_x = torch.abs(x)
        lc = abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)
        return lc * (1.0 + torch.log1p(abs_x * abs_x))

    @staticmethod
    def _dro_weights(scores: torch.Tensor) -> torch.Tensor:
        """Log-space DRO weights from non-negative scores.

        Returns normalised weights (mean ≈ 1) where above-mean
        log-scores get weight > 1.  `alpha = tanh(std)` blends toward
        uniform when spread is small.
        """
        s = scores.detach()
        s = s - s.min() + 1e-8
        log_s = torch.log(s)
        std = log_s.std()
        if not torch.isfinite(std) or std < 1e-8:
            return torch.ones_like(scores)
        alpha = torch.tanh(std)
        z = (log_s - log_s.mean()) / std.clamp(min=1e-3)
        w = 1.0 + torch.log1p(z.clamp(min=0.0))
        w = w / w.mean().clamp(min=1e-8)
        w = alpha * w + (1.0 - alpha)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _event_magnitude(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Continuous union eventness in transformed space, bounded [0, 1).

        ``event_mag = r / (1 + r)`` where ``r = log1p(|union| / τ)``.
        Union semantics: false positives also receive event weighting.
        Prediction branch detached — no second gradient path.
        """
        abs_union = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        log_ratio = torch.log1p(abs_union / self.non_zero_threshold)
        return log_ratio / (1.0 + log_ratio)

    # ------------------------------------------------------------------
    # Spectral loss
    # ------------------------------------------------------------------

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Log-magnitude multi-resolution STFT loss.

        `log_cosh(log1p(mag_pred) − log1p(mag_true))` — unbounded
        penalty for flat predictions, MSLE-aligned, no demean, no DC
        mask, no RMS normalization.  Gradient flows through identity
        Jacobian (no zero-sum constraint).  Event-magnitude weighting
        per series.
        """
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        # Per-series event relevance (max event_mag over time)
        abs_union = torch.max(torch.abs(true), torch.abs(pred.detach()))
        log_ratio = torch.log1p(abs_union / self.non_zero_threshold)
        series_event = (log_ratio / (1.0 + log_ratio)).amax(dim=1)
        if series_event.sum() < 1e-8:
            return pred.new_tensor(0.0)

        T = pred.size(1)
        total = pred.new_tensor(0.0)
        n_valid = 0

        for n_fft, hop in self._SPECTRAL_RESOLUTIONS:
            if T < n_fft:
                continue
            window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            S_pred = torch.stft(
                pred, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )
            S_true = torch.stft(
                true, n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=False, return_complex=True,
            )
            # Safe magnitude — bounded gradient at |z|→0
            mag_pred = torch.sqrt(S_pred.real ** 2 + S_pred.imag ** 2 + 1e-8)
            mag_true = torch.sqrt(S_true.real ** 2 + S_true.imag ** 2 + 1e-8)

            # Log-magnitude difference: MSLE-aligned, unbounded for
            # under-energy predictions (flat → log1p(ε) vs log1p(large))
            log_mag_err = torch.log1p(mag_pred) - torch.log1p(mag_true)
            cell_loss = self._log_cosh(log_mag_err)

            # Event-weighted series aggregation
            sw = series_event.view(-1, 1, 1)
            denom = (sw.sum() * cell_loss.size(1) * cell_loss.size(2)).clamp(min=1e-8)
            total = total + (sw * cell_loss).sum() / denom
            n_valid += 1

        return total / max(n_valid, 1)

    # ------------------------------------------------------------------
    # Temporal gradient (shape) loss
    # ------------------------------------------------------------------

    def _shape_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Temporal-gradient matching — the primary shape-learning signal.

        Computes first differences Δŷ and Δy and penalises their
        element-wise disagreement with `log_cosh_proportional`.

        Properties:
        • DC-free by construction — first-difference kills any constant
          offset, so level errors do NOT dilute shape learning.
        • NOT zero-sum — gradient at timestep t receives independent
          contributions from the (t−1,t) and (t,t+1) pairs.  The model
          IS free to increase output variance (unlike demeaned e_shape).
        • Sharp at transitions — for a flat prediction at an onset
          (Δy=3, Δŷ=0), the proportional loss gives ≈9.5.  For the same
          cell under pointwise cell_loss with error ≈1, loss ≈0.7.
          That's a 13× gradient amplification at the cells that matter.
        • Silent on plateaus — when Δy=0 and Δŷ≈0, loss≈0. No wasted
          gradient on peace-cell deserts.

        Weighting: transition-event curriculum.  Each adjacent pair gets
        weight proportional to the max(event_mag) of the two cells in the
        pair.  Event onsets (0→high) get full weight.  Peace-peace pairs
        (0→0) get near-zero weight.  Inverse-frequency rebalancing
        ensures conflict transitions aren't diluted by peace mass.
        """
        # First differences: (B, T-1)
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dy_true = y_true[:, 1:] - y_true[:, :-1]

        # Shape error at each transition
        shape_err = self._log_cosh_proportional(dy_pred - dy_true)

        # Transition-event weighting: max event_mag of each adjacent pair
        event_mag = self._event_magnitude(y_pred, y_true)
        pair_event = torch.max(event_mag[:, 1:], event_mag[:, :-1])  # (B, T-1)

        # Inverse-frequency rebalancing per series
        p_pair = pair_event.mean(dim=1, keepdim=True).clamp(min=1e-4, max=1.0 - 1e-4)
        w_shape = pair_event / p_pair + (1.0 - pair_event) / (1.0 - p_pair)
        w_shape = w_shape / w_shape.mean(dim=1, keepdim=True).clamp(min=1e-8)

        series_shape = (w_shape * shape_err).mean(dim=1)  # (B,)

        # Cross-series DRO: prioritise countries where shape is worst
        # Baseline: irreducible shape loss (target's own gradient variance)
        target_shape_err = self._log_cosh_proportional(dy_true)
        baseline = (w_shape.detach() * target_shape_err).mean(dim=1).detach()
        regret = torch.relu(series_shape.detach() - baseline)
        w_series = self._dro_weights(regret)
        w_series = w_series / w_series.mean().clamp(min=1e-8)

        return (w_series * series_shape).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        T = y_pred.size(1)

        # ── Level loss (DC) ───────────────────────────────────────────
        # Penalises per-series mean offset.  Simple and direct — the
        # model must predict the correct average level per country.
        # DRO-weighted so conflict countries (that matter) get priority.
        e_mean = (y_pred - y_true).mean(dim=1)                    # (B,)
        level_cell = self._log_cosh_proportional(e_mean)
        # DRO on level: countries with large DC offset get upweighted
        level_regret = torch.relu(
            level_cell.detach() - self._log_cosh_proportional(
                y_true.mean(dim=1)
            ).detach()
        )
        w_level = self._dro_weights(level_regret)
        w_level = w_level / w_level.mean().clamp(min=1e-8)
        loss_level = (w_level * level_cell).mean()

        # ── Shape loss (AC via temporal gradients) ────────────────────
        # Primary shape signal.  Gradient ≈13× stronger at transitions
        # than pointwise cell_loss.  DC-immune, not zero-sum.
        loss_shape = self._shape_loss(y_pred, y_true)

        # ── Log-magnitude multi-resolution STFT ───────────────────────
        # Secondary shape signal — penalises missing spectral energy at
        # multiple time-scales.  Catches periodic patterns that single-
        # step gradient matching might miss.
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        total_loss = loss_level + loss_shape + loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: level={loss_level.item():.6f} "
                f"shape={loss_shape.item():.6f} spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | level=%.6f shape=%.6f "
            "spec=%.6f total=%.6f",
            loss_level.item(), loss_shape.item(),
            loss_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"