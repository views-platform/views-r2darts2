import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v40 — plain log_cosh, level + AC + STFT (no temporal gradient).

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros, 10% spanning
    four orders of magnitude in raw deaths.

    ── Architecture ─────────────────────────────────────────────────────

    Three components.  All use plain `log_cosh` (saturating gradient at
    ±1).  Prioritization handled exclusively by inverse-frequency weights
    and DRO — no magnitude-dependent correction.

    1. **Multi-scale windowed level loss (DC)** — `log_cosh(mean(e))`
       at windows of 4, 12, and 36 months.  Catches level offsets at
       local, seasonal, and global scales.  W=4 windows provide the
       "break out of flat" signal: if prediction is flat but a quarter
       has events, this gives direct "raise your mean here" gradient.
       Event-magnitude weighted per window + inverse-frequency + DRO.

    2. **Demeaned error (AC shape) loss** — `log_cosh(ŷ_AC − y_AC)`
       where AC = signal − series_mean.
       Dense per-cell shape signal.  Pushes event cells UP and peace
       cells DOWN relative to series mean.  Together with level, fully
       specifies the target: level sets the mean per window, AC sets
       the relative heights within.  Event-magnitude weighted + DRO.

    3. **Log-magnitude multi-resolution STFT** — catches periodic
       patterns that pointwise losses miss.  `log_cosh(log1p(mag_pred) −
       log1p(mag_true))` with event-magnitude series weighting.

    ── Why no temporal gradient loss ────────────────────────────────────

    Temporal gradient (Δŷ − Δy) was removed because:
    - log_cosh saturates at ±1: model gets SAME gradient whether off by
      2 or 5 at a transition → can't learn to make BIG jumps for high-
      variance countries
    - Zero signal on plateaus (Δy=0): exactly where high-variance
      countries need gradient most (must sustain level, not just jump)
    - Creates competing smoothness pressure on plateaus where level+AC
      want the model to stay high
    - Redundant with level(W=4) + AC: level provides onset detection at
      quarterly scale, AC provides dense per-cell refinement

    ── Why no σ cap in RevIN ────────────────────────────────────────────

    Previous versions capped per-series σ at 5× batch-mean to prevent
    forecast runaway via proportional-loss amplification.  With plain
    log_cosh, that amplification loop is broken (saturating gradient).
    The cap PREVENTS high-variance countries from expressing full dynamic
    range: Syria (σ=200) capped to 75 → model can output max ≈ 6.3 when
    target needs 6.9.  Removed — the ±50 ẑ clamp provides sufficient
    float32 overflow protection.

    ── Why plain log_cosh ───────────────────────────────────────────────

    Previous `log_cosh_proportional` gave growing gradient (≈ 1+2ln|x|)
    for large errors.  This makes the model over-focus on unpredictable
    extremes (Syria) at the expense of learnable moderate conflicts
    (Nigeria).  Interferes with explicit inverse-frequency weights.

    Plain log_cosh: once wrong (|e|>2), all cells get equal gradient
    magnitude ≈1.  DRO + inverse-frequency cleanly control which cells/
    series matter without interference.

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
        """Sharp differentiable event/peace mask via shifted sigmoid.

        ``event_mag = ε + (1−ε) · σ(k·(|union|−τ)/τ)``
        Sharp transition around τ with small floor ε to prevent dead
        gradient zones.  Union semantics: false positives also receive
        event weighting.  Prediction branch detached.
        """
        abs_union = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        # Sigmoid with floor — k=5 gives 90% transition within ±1.2τ
        return 0.01 + 0.99 * torch.sigmoid(
            5.0 * (abs_union - self.non_zero_threshold) / self.non_zero_threshold
        )

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

    def _gradient_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Temporal-gradient matching — breaks the flat equilibrium.

        Computes first differences Δŷ and Δy and penalises their
        element-wise disagreement with `log_cosh`.

        Role: provides non-zero-sum gradient that BREAKS OUT of flat
        predictions.  Sparse (active only at transition boundaries)
        but with high per-cell magnitude at onsets/offsets.

        Weighting: transition-event curriculum.  Each adjacent pair gets
        weight proportional to the max(event_mag) of the two cells in the
        pair.  Inverse-frequency rebalancing per series.
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

        # Cross-series DRO
        target_shape_err = self._log_cosh_proportional(dy_true)
        baseline = (w_shape.detach() * target_shape_err).mean(dim=1).detach()
        regret = torch.relu(series_shape.detach() - baseline)
        w_series = self._dro_weights(regret)
        w_series = w_series / w_series.mean().clamp(min=1e-8)

        return (w_series * series_shape).mean()

    # ------------------------------------------------------------------
    # Pointwise event-cell loss
    # ------------------------------------------------------------------

    def _ac_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Direct pointwise error at event cells.

        Applies `log_cosh_proportional(ŷ − y)` per cell, weighted by
        event_magnitude (sharp sigmoid mask).  No decomposition, no
        zero-sum constraint — dense gradient at every event cell
        regardless of whether it's an onset, plateau interior, or
        offset.

        Role: ensures the model matches the target at every cell where
        conflict is present.  The level loss handles window-mean offsets;
        this loss handles per-cell accuracy within those windows.

        The sharp sigmoid event_magnitude (ε + (1−ε)·σ(k·(|u|−τ)/τ))
        concentrates gradient budget on clearly-active conflict cells,
        preventing peace cells from eating budget.  Bounded DRO
        prevents high-variance series from monopolizing.

        Weighting: per-cell event magnitude with inverse-frequency
        rebalancing + cross-series DRO.
        """
        # Direct pointwise error
        cell_err = self._log_cosh_proportional(y_pred - y_true)

        # Event-magnitude weighting per cell
        event_mag = self._event_magnitude(y_pred, y_true)
        p_cell = event_mag.mean(dim=1, keepdim=True).clamp(min=1e-4, max=1.0 - 1e-4)
        w_ac = event_mag / p_cell + (1.0 - event_mag) / (1.0 - p_cell)
        w_ac = w_ac / w_ac.mean(dim=1, keepdim=True).clamp(min=1e-8)

        series_ac = (w_ac * cell_err).mean(dim=1)  # (B,)

        # Cross-series DRO — bounded scoring (plain log_cosh) to
        # prevent feedback loop where hard-to-predict series (Sudan)
        # monopolize budget via proportional amplification.
        cell_err_bounded = self._log_cosh(y_pred - y_true)
        series_score = (w_ac.detach() * cell_err_bounded).mean(dim=1)
        target_err = self._log_cosh(y_true)
        baseline = (w_ac.detach() * target_err).mean(dim=1).detach()
        regret = torch.relu(series_score.detach() - baseline)
        w_series = self._dro_weights(regret)
        w_series = w_series / w_series.mean().clamp(min=1e-8)

        return (w_series * series_ac).mean()

    # ------------------------------------------------------------------
    # Multi-scale windowed level loss
    # ------------------------------------------------------------------

    _LEVEL_WINDOWS = (4, 12, 36)

    def _level_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-scale windowed level loss.

        Computes non-overlapping window means at 3 scales (4, 12, 36
        months) and penalises every (series, window) mean-error with
        plain `log_cosh` (saturating gradient — no magnitude interference
        with the explicit inverse-frequency weights).

        Multi-scale rationale:
        • W=4  — catches local level shifts (regime onset within a
          quarter).
        • W=12 — seasonal level.
        • W=36 — global DC.

        Event-magnitude weighting per window + inverse-frequency
        rebalancing + cross-series DRO.
        """
        T = y_pred.size(1)
        e = y_pred - y_true                                       # (B, T)
        event_mag = self._event_magnitude(y_pred, y_true)         # (B, T)

        all_window_losses = []
        all_window_weights = []

        for W in self._LEVEL_WINDOWS:
            if T < W:
                continue
            # Non-overlapping window means of error
            n_windows = T // W
            e_trunc = e[:, :n_windows * W].view(e.size(0), n_windows, W)
            window_mean_err = e_trunc.mean(dim=2)                 # (B, n_win)
            wloss = self._log_cosh_proportional(window_mean_err)  # (B, n_win)

            # Event weight per window: max event_mag within
            em_trunc = event_mag[:, :n_windows * W].view(e.size(0), n_windows, W)
            window_event = em_trunc.amax(dim=2)                   # (B, n_win)

            all_window_losses.append(wloss)
            all_window_weights.append(window_event)

        if not all_window_losses:
            return y_pred.new_tensor(0.0)

        # Concatenate across all scales: (B, total_windows)
        cat_loss = torch.cat(all_window_losses, dim=1)
        cat_event = torch.cat(all_window_weights, dim=1)

        # Inverse-frequency rebalancing per series
        p_win = cat_event.mean(dim=1, keepdim=True).clamp(min=1e-4, max=1.0 - 1e-4)
        w_level = cat_event / p_win + (1.0 - cat_event) / (1.0 - p_win)
        w_level = w_level / w_level.mean(dim=1, keepdim=True).clamp(min=1e-8)

        series_level = (w_level * cat_loss).mean(dim=1)           # (B,)

        # Cross-series DRO — bounded scoring (plain log_cosh) to
        # prevent proportional feedback loop.
        bounded_window_losses = []
        target_windows = []
        for W in self._LEVEL_WINDOWS:
            if T < W:
                continue
            n_windows = T // W
            e_trunc_b = e[:, :n_windows * W].view(e.size(0), n_windows, W)
            bounded_window_losses.append(self._log_cosh(e_trunc_b.mean(dim=2)))
            yt_trunc = y_true[:, :n_windows * W].view(y_true.size(0), n_windows, W)
            target_windows.append(self._log_cosh(yt_trunc.mean(dim=2)))
        cat_bounded = torch.cat(bounded_window_losses, dim=1)
        series_score = (w_level.detach() * cat_bounded).mean(dim=1)
        cat_baseline = torch.cat(target_windows, dim=1)
        baseline = (w_level.detach() * cat_baseline).mean(dim=1).detach()

        regret = torch.relu(series_score.detach() - baseline)
        w_series = self._dro_weights(regret)
        w_series = w_series / w_series.mean().clamp(min=1e-8)

        return (w_series * series_level).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        T = y_pred.size(1)

        # ── Multi-scale level loss (DC) ───────────────────────────────
        # Penalises local, seasonal, and global mean offset.  Event-
        # weighted so conflict epochs dominate.  Multiple scales catch
        # regime shifts that a single 36-month mean would wash out.
        # W=4 windows provide the "break out of flat" signal: if the
        # model is flat but a 4-month window has events, level loss at
        # that window gives direct "raise your mean here" gradient.
        loss_level = self._level_loss(y_pred, y_true)

        # ── Demeaned error (AC shape) loss ────────────────────────────
        # Dense per-cell shape signal.  Pushes every event cell up and
        # every peace cell down relative to series mean.  Together with
        # level loss, fully specifies the target: level sets the mean
        # per window, AC sets the relative heights within.
        #
        # No temporal gradient loss: it gives constant ±1 gradient
        # regardless of error magnitude (saturating log_cosh), so the
        # model gets no MORE incentive to make a 5-unit jump than a
        # 2-unit jump.  On sustained plateaus (Δy=0), it contributes
        # zero signal — exactly where high-variance countries need
        # gradient the most.  Redundant with level + AC and creates
        # competing smoothness pressure on plateaus.
        loss_ac = self._ac_loss(y_pred, y_true)

        # ── Log-magnitude multi-resolution STFT ───────────────────────
        # Penalises missing spectral energy at multiple time-scales.
        # Catches periodic patterns.
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        total_loss = loss_level + loss_ac + loss_spec

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: level={loss_level.item():.6f} "
                f"ac={loss_ac.item():.6f} spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | level=%.6f ac=%.6f "
            "spec=%.6f total=%.6f",
            loss_level.item(), loss_ac.item(),
            loss_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"