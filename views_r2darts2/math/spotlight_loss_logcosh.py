import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v47 — asinh + RevIN compatible, empty-batch invariant.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros at country-month and
    >99% zeros at pg-month, with the nonzero tail spanning four orders of
    magnitude in raw deaths.

    ── Components ───────────────────────────────────────────────────────

    1. **DC/AC decomposition** — per-window demeaning (same windows as level).
       e_shape = e − window_mean(e).  Shape and level are orthogonal:
       shape handles within-window patterns, level handles per-window DC.

    2. **Gated + magnitude-graded event weighting.**
       event_mag = gate × (1 + abs_max), gate = 0.005 + 0.995 × σ(10 × (abs_max − τ)).
       The gate suppresses peace (→ ~0.005) vs conflict (→ ~1); the (1 + abs_max)
       factor — bounded because abs_max is in asinh space — restores magnitude
       sensitivity across the 4-OOM tail so large wars outweigh small skirmishes
       instead of saturating flat. No model-state dependency (abs_max detached).

    3. **Per-series temporal DRO (event cells only)** — within-series shock
       therapy restricted to cells that clear τ in y_true OR y_pred, so it
       cannot amplify label noise on peaceful cells. Neutral (weight 1)
       elsewhere; renormalized to mean 1 among a series' event cells.

    4. **Windowed level anchor** — log_cosh on per-window means, peace/conflict
       *gated* (gating only, not magnitude-graded). The (short) window W sets
       its resolution for tracking non-stationary conflict on/off transitions.
       Its strength relative to the shape term is now *learned* (see 6), not a
       fixed T-scale.

    5. **Multi-resolution STFT loss** — DISABLED by default (_STFT=False).
       log_cosh on magnitude-spectrum differences, DC bin masked, gated to
       series with signal above τ.

    6. **Homoscedastic uncertainty weighting** (Kendall & Gal 2018; positive
       Liebel-Körner 2018 regularizer). ONE learned log-variance PER CHANNEL
       (not per term) balances the C channels, replacing a hand-tuned T-scale
       and an inverse-EMA channel equalizer:

           total = Σ_{c}  ½ e^{−s_c}·(L_shape,c + L_level,c)  +  softplus(s_c)

       s = log σ². The shape (magnitude-carrying) and level (DC) terms are two
       orthogonal projections of the SAME residual e (e_shape = e −
       window_mean(e); level = window_mean(e)), so they share one observation
       noise σ_c and combine 1:1 under a single per-channel precision. Giving
       them SEPARATE learned precisions was pathological: the level term (whose
       optimum is a flat within-window smear) is the easier to drive down, so it
       accrued weight and STARVED the shape term — the only anti-flat force —
       pushing w_sh/w_lv monotonically down and the forecasts toward a flat,
       under-predicting line. Coupling them removes that failure while keeping
       the per-channel budget (genuine sb/ns/os noise) fully data-driven.
       softplus(s) (≥0, = ln(1+σ²)) keeps the objective non-negative and stops
       any σ→0 collapse. The log-variances are real nn.Parameters (trained by
       the optimizer) but are excluded from state_dict, so Darts checkpoint load
       does not crash on unexpected keys.

    ── Normalization: every weighted term is a weight-MASS-weighted mean,
       loss = Σ(w·cell) / Σ(w).  This is stateless and composition-invariant:
       a near-empty batch has tiny weight mass and cannot dilute the event
       signal, removing the need for any cross-batch EMA on the term weights.

    ── Base cell loss: log_cosh (numerically stable). ───

    Args:
        non_zero_threshold: Event gate center τ (AsinhTransform: 0.88 ≈ asinh(1),
            i.e. 1 fatality).
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = False
    _EMA_EPS = 1e-6
    # Over-allocation for the per-channel learned log-variances. The loss is
    # constructed (via LossCatalog) with only non_zero_threshold — the channel
    # count C is unknown at __init__, yet the params must already exist before
    # configure_optimizers() reads self.parameters() (Darts builds the optimizer
    # before the first forward, so lazy creation would miss it). We allocate a
    # generous C_max and slice [:C] at forward; unused rows never receive
    # gradient. 8 ≫ the 3 UCDP targets.
    _MAX_CHANNELS = 8
    _SAFE_S = 10.0  # numerical guard on |log σ²| (non-binding: precision ∈ ~[2e-5, 1e4])

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

        # Homoscedastic uncertainty weighting (Kendall & Gal 2018 / positive
        # Liebel-Körner 2018 variant). ONE log-variance per CHANNEL (not per
        # term): the shape and level terms are two orthogonal projections of the
        # SAME residual e (e_shape = e − window_mean(e); level = window_mean(e)),
        # so they share one observation noise σ_c and combine 1:1 under a single
        # per-channel precision. Giving them SEPARATE learned precisions let the
        # easy-to-minimize level term (whose optimum is a flat within-window
        # smear) accrue weight and STARVE the shape term — the only anti-flat
        # force — driving w_sh/w_lv monotonically down and the forecasts toward
        # a flat, under-predicting line. Coupling them fixes that while keeping
        # the per-channel budget (genuine sb/ns/os noise) fully data-driven.
        # Real nn.Parameter so Darts' optimizer trains it (configure_optimizers
        # reads self.parameters()); EXCLUDED from state_dict via the _save/_load
        # overrides below so checkpoint load does not crash on unexpected keys.
        # Init 0 → σ²=1 → ½ precision.
        self._log_var = torch.nn.Parameter(torch.zeros(self._MAX_CHANNELS, 1))

        # Telemetry for callbacks
        self._last_components: dict | None = None
        self._last_weights: list[float] | None = None

        logger.info(
            "SpotlightLossLogcosh | threshold=%.4f | uncertainty-weighted "
            "(shape/level, per-channel)",
            non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Checkpoint safety: keep the learned log-variances OUT of state_dict.
    # They live in self.parameters() (so the optimizer trains them), but Darts
    # both pickles the whole criterion object into the checkpoint separately AND
    # calls load_state_dict(strict=...); emitting the params here would make the
    # freshly-built loss report unexpected/missing keys and crash model load.
    # The params only shape the training-time gradient balance and are
    # irrelevant at predict time, so skipping them is safe.
    # ------------------------------------------------------------------

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Intentionally emit nothing → state_dict() stays empty.
        return

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Intentionally a no-op: never claim missing keys, never consume any.
        return

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    @staticmethod
    def _dro_weights_2d(losses: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        """Per-series sqrt self-reweighting, restricted to event cells.

        w_it = sqrt(loss_it / mean_event(loss))   on event cells, else 1.

        Sublinear concentration: a cell 16× harder than the series' event
        average gets 4× the gradient (not 16×). Restricting to event cells
        (τ cleared in y_true OR y_pred) keeps DRO from amplifying label noise
        on the ~99% peaceful cells. Weights are renormalized to mean ≈ 1 among
        a series' event cells, so DRO only *redistributes* the event budget and
        does not change its total. Series with no event cells → all 1 (neutral).

        Returns weights shaped like losses: (B, T) or (B, T, C).
        """
        l = losses.detach()                                  # (B, T) or (B, T, C)
        m = event_mask.to(l.dtype)                           # 1 on event cells
        n_event = m.sum(dim=1, keepdim=True)                 # (B, 1) or (B, 1, C)
        mu = ((l * m).sum(dim=1, keepdim=True)
              / n_event.clamp(min=1.0)).clamp(min=1e-6)      # event-cell mean
        w = torch.sqrt(l / mu)                               # sublinear focus
        w_mean = ((w * m).sum(dim=1, keepdim=True)
                  / n_event.clamp(min=1.0)).clamp(min=1e-8)  # mean over events
        w = w / w_mean                                       # event weights → mean 1
        w = torch.where(event_mask & (n_event > 0), w, torch.ones_like(w))
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def _uncertainty_combine(
        self,
        loss_shape: torch.Tensor,
        loss_level: torch.Tensor,
        loss_spec: torch.Tensor,
    ) -> tuple[torch.Tensor, list[float], list[float], list[float]]:
        """Homoscedastic uncertainty weighting of the shape & level terms.

        Kendall & Gal (2018) with the positive Liebel-Körner (2018) regularizer,
        coupled per channel:

            total = Σ_c  ½ e^{−s_c}·(L_shape,c + L_level,c)
                       +  softplus(s_c)
                       +  Σ_c L_spec,c

        s = log σ² is learned per CHANNEL (one precision shared by both the
        shape and level terms, which are orthogonal projections of the same
        residual). A single shared precision keeps shape:level at a fixed 1:1
        balance so the easy flat-smear level term can no longer accrue weight
        and starve the anti-flat shape term (the failure that drove w_sh/w_lv
        monotonically down and the forecasts flat), while still budgeting
        gradient across channels data-drivenly. softplus(s) ≥ 0 keeps the
        objective non-negative and penalizes σ→0, so no channel can zero out
        its own weight.

        Returns (total, w_shape, w_level, contribution); the last three are
        per-channel Python lists for telemetry. w_shape == w_level now (the
        precision is shared) — the ratio is pinned at 1.0 by design.
        """
        univariate = loss_shape.dim() == 0
        L_shape = loss_shape.reshape(1) if univariate else loss_shape
        L_level = loss_level.reshape(1) if univariate else loss_level
        C = L_shape.shape[0]

        s = self._log_var[:C].clamp(-self._SAFE_S, self._SAFE_S)  # (C, 1)
        s_ch = s[:, 0]                                            # (C,)

        # ONE precision per channel, SHARED by the shape & level terms. They are
        # orthogonal projections of the same residual, so they take the same
        # observation noise and a fixed 1:1 balance. This removes the pathology
        # of independent (shape, level) precisions, under which the easy flat-
        # smear level term accrued weight and starved the anti-flat shape term.
        w_ch = 0.5 * torch.exp(-s_ch)                            # precision ½/σ²
        reg = F.softplus(s_ch)                                   # ≥0, = ln(1+σ²)

        weighted = w_ch * (L_shape + L_level) + reg             # (C,)
        # Spectral term (disabled by default) is carried UNWEIGHTED: it is 0 when
        # off and must not enter the uncertainty weighting (a zero L would drive
        # its s → −∞). loss_spec is a scalar 0 when off, or (C,) when enabled.
        spec = loss_spec.reshape(-1) if loss_spec.dim() else loss_spec.reshape(1).expand(C)
        total = weighted.sum() + spec.sum()

        contribution = (w_ch * (L_shape + L_level)).detach().tolist()
        w_list = w_ch.detach().tolist()
        return total, w_list, w_list, contribution

    def _event_calibration_ratio(
        self, y_pred_det: torch.Tensor, y_true: torch.Tensor
    ) -> list[float]:
        """Σ|y_pred| / Σ|y_true| over true-event cells, per channel.

        <1 → the model under-predicts event magnitude (the failure this
        rebalance targets); >1 → over-prediction. Detached diagnostic only.
        """
        with torch.no_grad():
            ev = (y_true.abs() > self.non_zero_threshold).to(y_true.dtype)
            if y_true.dim() == 3:
                num = (y_pred_det.abs() * ev).sum(dim=(0, 1))
                den = (y_true.abs() * ev).sum(dim=(0, 1)).clamp(min=self._EMA_EPS)
                return (num / den).tolist()
            num = (y_pred_det.abs() * ev).sum()
            den = (y_true.abs() * ev).sum().clamp(min=self._EMA_EPS)
            return [float(num / den)]

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
        y_pred_det: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Event-magnitude-weighted windowed level anchor.

        Splits the T-length error into non-overlapping windows, computes
        log_cosh on per-window means, then weights each series by its
        event magnitude (gate × (1 + series_mag)). No DRO, no CMW. Returns the
        raw weight-mass-normalized level loss; its weight relative to the shape
        term is learned downstream (see _uncertainty_combine).

        y_pred_det: detached prediction tensor — when supplied, series weighting
            uses max(|y_true|, |y_pred_det|) so that predicted false positives
            on peaceful series also attract level loss gradient. Must be same
            shape as y_true.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )  # (B, n_windows) or (B, n_windows, C)
        level_losses = self._log_cosh(window_means)

        # Per-series event magnitude: max(|y_true|, |y_pred|) across time → sigmoid
        # Using max of both ensures false-positive series attract full level gradient,
        # symmetric with the shape loss abs_max gating.
        if y_pred_det is not None:
            abs_max_series = torch.max(y_true.abs(), y_pred_det.abs())
        else:
            abs_max_series = y_true.abs()
        series_mag = abs_max_series.max(dim=1).values  # (B,) or (B, C)
        # GATE × MAGNITUDE on the level anchor. The gate suppresses peace
        # (→ ~0.005) vs conflict (→ ~1); the (1 + series_mag) factor makes the
        # level anchor prioritize HIGH-magnitude event series so the model is
        # pushed to raise its predicted LEVEL where it matters most — the direct
        # remedy for the systematic under-prediction on the biggest events
        # (e.g. ch2/os sitting at ~0.15x). series_mag is in asinh space, so the
        # factor is bounded (a 10k-death war ≈ 11x a 1-death cell) and adds NO
        # new constant. Level is the ONLY magnitude-carrying term — shape is
        # window-demeaned, hence DC/magnitude-blind — so this is where magnitude
        # grading BELONGS; putting it on shape (as (1+abs_max)) cannot lift the
        # predicted level at all.
        #
        # This was previously gate-only because the old FIXED ×T=36 level scale
        # turned the magnitude factor into a ~10x-amplified, T-scaled DC gradient
        # that drove ch0/sb to 2.3-2.5x over-prediction with grad-norm shocks.
        # That ×T is GONE: homoscedastic uncertainty weighting now sets the level
        # strength adaptively and PER CHANNEL, so (a) there is no T amplification
        # and (b) w_level self-lowers on any channel that begins to over-predict.
        # That makes the magnitude grading safe to restore, and it increases
        # per-country differentiation (big series get more level attention) —
        # which fights templating rather than causing it.
        series_gate = 0.005 + 0.995 * torch.sigmoid(
            10.0 * (series_mag - self.non_zero_threshold)
        )  # (B,) or (B, C) — peace-suppression gate
        series_w = series_gate * (1.0 + series_mag)  # gate × magnitude grading

        # ── Weighted-mean normalization (empty-batch invariant) ────────
        # Normalize the level loss by the series-weight MASS,
        # loss = Σ(series_w·level) / Σ(series_w), instead of the old cross-batch
        # EMA. An all-peace batch has tiny weight mass in both numerator and
        # denominator, so it neither snaps peaceful series back to weight ~1 nor
        # dilutes the level signal — stateless and composition-invariant.
        # No fixed scale on the level term: its strength relative to the shape
        # term is now learned via homoscedastic uncertainty weighting
        # (_uncertainty_combine). This returns the raw weight-mass-normalized
        # level loss; the old T-scale that dominated the budget (~95%) and drove
        # peak under-prediction is gone.
        n_windows = level_losses.shape[1]
        if level_losses.dim() == 3:
            num = (series_w.unsqueeze(1) * level_losses).sum(dim=(0, 1))   # (C,)
            den = (series_w.sum(dim=0) * n_windows).clamp(min=self._EMA_EPS)
            return num / den                                              # (C,)
        else:
            num = (series_w.unsqueeze(1) * level_losses).sum()
            den = (series_w.sum() * n_windows).clamp(min=self._EMA_EPS)
            return num / den                                              # scalar

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only).

        Safe magnitude sqrt(re² + im² + ε) avoids gradient blowup at
        |z|→0.  DC bin is masked — level anchor already handles DC.
        Only series with signal above threshold are included.
        """
        if y_pred.dim() == 3:
            C = y_pred.shape[-1]
            return torch.stack(
                [self._spectral_loss(y_pred[..., c], y_true[..., c]) for c in range(C)]
            )

        # 2D path continues here
        pred = y_pred
        true = y_true

        has_signal = (
            (torch.abs(true) > self.non_zero_threshold)
            | (torch.abs(pred.detach()) > self.non_zero_threshold)
        ).any(dim=1)
        if not has_signal.any():
            return pred.new_tensor(0.0)
            
        pred = pred[has_signal]
        true = true[has_signal]

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
            mag_true = S_true.abs()
            # Mask DC bin — level is handled by the level anchor
            mag_pred = mag_pred.clone()
            mag_true = mag_true.clone()
            mag_pred[:, 0, :] = 0.0
            mag_true[:, 0, :] = 0.0
            total = total + self._log_cosh(mag_pred - mag_true).mean()
            n_valid += 1

        return total / max(n_valid, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        T = y_pred.size(1)
        e = y_pred - y_true

        # ── Per-window DC/AC decomposition ────────────────────────────
        # Demean within each non-overlapping window (same W as level anchor).
        W = max(6, T // 3)
        windows = list(e.split(W, dim=1))  # list of (B, W_i) or (B, W_i, C)
        e_shape = torch.cat(
            [w - w.mean(dim=1, keepdim=True) for w in windows], dim=1
        )  # (B, T) or (B, T, C) — zero-mean within each window

        # ── Base cell loss ─────────────────────────────────────────────
        cell_loss = self._log_cosh(e_shape)

        # ── Gated + magnitude-graded event weighting ──────────────────
        # The sigmoid is a *peace-suppression gate* only: peace → ~0, conflict
        # → ~1. Above ~2 deaths it saturates, so on its own it weighted a
        # 2-death skirmish identically to a 10,000-death war and left the
        # entire 4-OOM tail flat (the source of peak under-prediction /
        # flattening). We restore magnitude sensitivity by multiplying the gate
        # by (1 + abs_max): abs_max is already in asinh space, which compresses
        # 4 OOM into ~[0,10], so the factor is bounded (Ukraine ~10x a 1-death
        # cell) and requires NO new constant — the asinh transform already in
        # the pipeline IS the data-driven scale. abs_max = max(|y_true|,
        # |y_pred.detach()|) keeps it feedback-loop-safe (under-predicting a
        # true event keeps |y_true| large; the detach prevents gaming).
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        event_gate = 0.005 + 0.995 * torch.sigmoid(10.0 * (abs_max - self.non_zero_threshold))
        event_mag = event_gate * (1.0 + abs_max)

        # ── Per-series temporal DRO (event cells only) ─────────────────
        # DRO re-focuses gradient onto the hardest timesteps within a series,
        # but on ~99%-zero data it would otherwise amplify label noise on
        # peaceful cells. Restrict it to cells that clear τ in y_true OR y_pred,
        # so peace cells stay neutral (weight 1).
        event_mask = (torch.abs(y_true) > self.non_zero_threshold) | (
            torch.abs(y_pred.detach()) > self.non_zero_threshold
        )
        w_dro = self._dro_weights_2d(cell_loss, event_mask)  # (B, T) or (B, T, C)
        w_total = torch.nan_to_num(event_mag * w_dro, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Weighted-mass normalization (empty-batch invariant) ────────
        # Normalize the shape loss by its own weight MASS rather than by cell
        # count or a cross-batch EMA: loss = Σ(w·cell) / Σ(w). Peace cells carry
        # weight ~0.005, so a near-empty batch contributes almost nothing to
        # both numerator and denominator and cannot dilute the accumulated event
        # signal, while within any batch the event cells dominate the average.
        # Stateless, composition-invariant, and the single most important
        # property for keeping the model sensitive to conflict on/off states.
        if w_total.dim() == 3:
            num = (w_total * cell_loss).sum(dim=(0, 1))          # (C,)
            den = w_total.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)
            loss_shape = num / den                              # (C,)
        else:
            num = (w_total * cell_loss).sum()
            den = w_total.sum().clamp(min=self._EMA_EPS)
            loss_shape = num / den                              # scalar

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, y_true, T, y_pred_det=y_pred.detach())

        # ── Multi-resolution spectral loss (always on) ──────────────
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        # ── Homoscedastic uncertainty weighting & telemetry ────────────
        total_loss, w_shape, w_level, contribution = self._uncertainty_combine(
            loss_shape, loss_level, loss_spec
        )

        # Real event calibration ratio Σ|y_pred|/Σ|y_true| over true-event cells,
        # per channel — <1 means under-prediction (the failure this rebalance
        # targets). Surfaced as cal_ratio for the LossComponents callback.
        cal_ratio = self._event_calibration_ratio(y_pred.detach(), y_true)

        if loss_shape.dim() == 0:
            # Univariate path
            self._last_weights = w_shape
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "spec": [float(loss_spec.detach()) if loss_spec.dim() == 0 else float(loss_spec)],
                "weight": w_shape,          # learned shape precision ½e^{−s}
                "cal_score": w_level,       # learned level precision ½e^{−s}
                "cal_ratio": cal_ratio,     # <1 → under-prediction
                "gates": contribution,
                "contribution": contribution,
            }
        else:
            # Multivariate path
            C = loss_shape.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            self._last_weights = w_shape
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "spec": spec_list,
                "weight": w_shape,          # learned shape precision ½e^{−s}
                "cal_score": w_level,       # learned level precision ½e^{−s}
                "cal_ratio": cal_ratio,     # <1 → under-prediction
                "ema": w_shape,
                "gates": contribution,      # per-channel weighted total
                "contribution": contribution,
            }

        if torch.isnan(total_loss):
            _s = float(loss_shape.sum()) if loss_shape.dim() else float(loss_shape)
            _l = float(loss_level.sum()) if loss_level.dim() else float(loss_level)
            _sp = float(loss_spec.sum()) if loss_spec.dim() else float(loss_spec)
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={_s:.6f} level={_l:.6f} spec={_sp:.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f "
            "spec=%.6f total=%.6f",
            loss_shape.item() if loss_shape.dim()==0 else loss_shape.sum().item(),
            loss_level.item() if loss_level.dim()==0 else loss_level.sum().item(),
            loss_spec.item() if loss_spec.dim()==0 else loss_spec.sum().item(),
            total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"