import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v46 — asinh + RevIN compatible, per-series DRO.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting: ~90% zeros, 10% spanning
    four orders of magnitude in raw deaths.

    ── Components ───────────────────────────────────────────────────────

    1. **DC/AC decomposition** — per-window demeaning (same windows as level).
       e_shape = e − window_mean(e).  Shape and level are orthogonal:
       shape handles within-window patterns, level handles per-window DC.

    2. **Occurrence-probability event weighting.**
    event_weight = z_event + (1 − z_event) × p_event,
    where z_event = 1{y_true > τ} and p_event = σ(y_pred − τ).
    True events get weight 1.0. Peace cells get the model's own predicted
    event probability — fully data-driven, zero hyperparameters. Self-
    correcting: hallucinated events raise p_event, increasing regression
    weight and pushing the prediction back down.

     3. **Pseudo-hurdle occurrence term** — zero vs non-zero supervision
         from the same scalar output. The model's asinh prediction is converted
         into an occurrence logit relative to ``non_zero_threshold`` and trained
         with positive-class-reweighted BCE. This gives explicit event-detection
         signal without adding a second head.

     4. **Per-series temporal DRO** — within-series shock therapy.
         Power-law reweighting (alpha in (0,1)) upweights harder timesteps
         *relative to that series* while remaining sublinear.

     5. **Windowed level anchor** — log_cosh on per-window means.
         Uses the exact same window partition as shape for strict orthogonality.

     6. **Multi-resolution STFT loss** — always on, ungated.
       log_cosh on magnitude-spectrum differences.  DC bin masked.

    ── Base cell loss: log_cosh × (1 + log(1+|x|³))  (proportional) ───

    Args:
        non_zero_threshold: Sigmoid center (AsinhTransform: 0.88 ≈ asinh(1))
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = False
    _EMA_BETA = 0.99
    _EMA_EPS = 1e-6

    # Adaptive DRO alpha bounds.
    # Event weighting uses no separate constants: event_weight is derived
    # entirely from the pseudo-hurdle occurrence probability p_event and the
    # already-present non_zero_threshold. See _event_weight() and
    # _occurrence_hurdle_terms().
    # alpha is computed each forward pass as:
    #   alpha = _DRO_ALPHA_MIN + (_DRO_ALPHA_MAX - _DRO_ALPHA_MIN) * (1 - f_event)
    # where f_event = fraction of cells with abs_max > tau_eff.
    # At 97% zeros (f_event ≈ 0.03): alpha ≈ _DRO_ALPHA_MAX (aggressive concentration).
    # At 50% events (f_event ≈ 0.50): alpha ≈ midpoint (relaxed).
    _DRO_ALPHA_MIN = 0.30
    _DRO_ALPHA_MAX = 0.80

    # Shared windowing for strict shape-level orthogonality.
    _WINDOW_DIVISOR = 3
    _MIN_WINDOW = 6
    _LEVEL_SCALE = 1.0  # Fine-tuning multiplier applied on top of the base T scaling.

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

        # Two-timescale self-referential loss tracking for progress routing.
        # Both EMAs reuse the single _EMA_BETA constant (slow is the EMA of
        # fast), so no extra timescale/hyperparameter is introduced.
        self._loss_ema: list[float] | None = None       # fast EMA (~1/(1-beta))
        self._loss_ema_slow: list[float] | None = None  # slow EMA (~2/(1-beta))

        # Shape and level terms are composition-robust WITHOUT cross-batch state:
        # each is a self-normalized (Hájek) ratio estimator loss = Σ(w·ℓ)/Σ(w)
        # over the current batch, so numerator and denominator scale together
        # with event composition and no running weight-scale EMA is needed.

        # Telemetry for callbacks
        self._last_components: dict | None = None
        self._last_weights: list[float] | None = None

        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _dro_weights_2d(
        self, losses: torch.Tensor, y_true: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Per-series power-law self-reweighting.

        w_it = (loss_it / mean_i(loss))^alpha, alpha in (0, 1)

        alpha is computed adaptively per-batch in forward() based on the
        fraction of active (conflict) cells in the batch.

        Returns weights with mean ≈ 1 per series, shape (B, T) or (B, T, C).
        """
        del y_true  # Kept for signature compatibility.
        l = losses.detach()                                  # (B, T) or (B, T, C)
        mu = l.mean(dim=1, keepdim=True).clamp(min=1e-6)     # (B, 1) or (B, 1, C)
        w = torch.pow(l / mu, alpha)                         # (B, T) or (B, T, C)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)  # renormalize mean=1
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _window_size(self, T: int) -> int:
        """Shared non-overlapping window size for shape and level terms."""
        return max(self._MIN_WINDOW, T // self._WINDOW_DIVISOR)

    def _event_weight(
        self, z_event: torch.Tensor, p_event: torch.Tensor
    ) -> torch.Tensor:
        """Data-driven event weight from occurrence probability.

        For true event cells (z_event=1): weight = 1.0 — full regression signal.
        For peace cells (z_event=0):      weight = p_event (detached) — the
          model's own predicted probability.

        Self-correcting property: if the model hallucinates an event on a
        peace cell, p_event is high, the regression loss gets high weight, and
        the gradient pushes the prediction back below the threshold. Once the
        model correctly predicts peace, p_event ≈ 0 and the regression weight
        vanishes — no pressure to overfit to exact zero values.

        Zero hyperparameters: uses only non_zero_threshold (via
        _occurrence_hurdle_terms) and no additional constants.
        """
        return z_event + (1.0 - z_event) * p_event.detach()

    def _occurrence_hurdle_terms(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return occurrence loss, true event indicator, and predicted event probability.

        Uses a unit-slope sigmoid centered at ``non_zero_threshold``.
        No slope multiplier or threshold shift constants are needed: the
        positive-class reweighting already calibrates the gradient magnitude,
        and the threshold is the semantically meaningful boundary (asinh(1) ≈
        one battle death).

        Returns:
            loss_occ: class-reweighted BCE, scalar (2D) or (C,) (3D).
            z_event:  true event indicator, same shape as ``y_true``.
            p_event:  predicted event probability, same shape as ``y_true``.
        """
        tau = self.non_zero_threshold
        z_event = (y_true > tau).to(dtype=y_pred.dtype)
        # Unit-slope logit: positive when y_pred > tau, negative otherwise.
        # The BCE gradient is already calibrated by the class-reweighted loss;
        # no additional slope multiplier is required.
        occ_logit = y_pred - tau
        p_event = torch.sigmoid(occ_logit)

        # Positive-class reweighting from the current batch event rate keeps
        # the occurrence loss informative even at 97% sparsity.
        if z_event.dim() == 3:
            event_rate = z_event.mean(dim=(0, 1)).detach().clamp(
                min=self._EMA_EPS, max=1.0 - self._EMA_EPS
            )
            pos_weight = ((1.0 - event_rate) / event_rate).view(1, 1, -1)
            loss_occ_raw = F.binary_cross_entropy_with_logits(
                occ_logit, z_event, reduction="none"
            )
            loss_occ_weighted = torch.where(z_event > 0.0, pos_weight * loss_occ_raw, loss_occ_raw)
            loss_occ = loss_occ_weighted.mean(dim=(0, 1))
        else:
            event_rate = z_event.mean().detach().clamp(
                min=self._EMA_EPS, max=1.0 - self._EMA_EPS
            )
            pos_weight = (1.0 - event_rate) / event_rate
            loss_occ_raw = F.binary_cross_entropy_with_logits(
                occ_logit, z_event, reduction="none"
            )
            loss_occ_weighted = torch.where(z_event > 0.0, pos_weight * loss_occ_raw, loss_occ_raw)
            loss_occ = loss_occ_weighted.mean()

        return loss_occ, z_event, p_event

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def _combine_channels(self, per_channel_loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Combine per-channel losses by *relative learning progress*.

        Two failure modes of magnitude-based routing are avoided:

        * Routing on a channel's absolute (scale-normalised) loss makes the
          router chase whichever target has the highest *irreducible* noise
          floor, permanently starving channels that could still improve.
        * Dividing the loss by the physical target scale (RMS) systematically
          down-weights the largest-signal channel — the primary target — and
          mixes units (a W-scaled level term over an asinh-RMS is not a clean
          relative error).

        Instead each channel is compared only to *its own* history via two
        cascaded EMAs that share the single existing smoothing constant
        (so no extra timescale is introduced):

            fast_c  = EMA_beta(loss_c)       # ~1/(1-beta) steps
            slow_c  = EMA_beta(fast_c)       # ~2/(1-beta) steps
            score_c = fast_c / slow_c        # dimensionless trend
            w_c     = C * score_c / Sum_k(score_k)

        score_c > 1 when channel c is regressing or lagging the others'
        progress, ~1 when it has plateaued (incl. at its noise floor), and
        < 1 when it is the fastest-improving channel.  Being a self-referential
        ratio, the score stays near 1 for any converged channel, so the weights
        cannot collapse to a winner-take-all regime (no target is starved)
        while gradient is still tilted toward the least-improving channel.
        """
        C = per_channel_loss.shape[0]
        batch_loss_det = per_channel_loss.detach()
        beta = self._EMA_BETA

        # ── Two-timescale self-referential loss tracking ─────────────
        if (
            self._loss_ema is None
            or self._loss_ema_slow is None
            or len(self._loss_ema) != C
        ):
            self._loss_ema = batch_loss_det.tolist()
            self._loss_ema_slow = batch_loss_det.tolist()
        else:
            for c in range(C):
                self._loss_ema[c] = beta * self._loss_ema[c] + (1.0 - beta) * float(batch_loss_det[c])
                self._loss_ema_slow[c] = beta * self._loss_ema_slow[c] + (1.0 - beta) * self._loss_ema[c]

        # ── Relative-progress routing ────────────────────────────────
        fast = per_channel_loss.new_tensor(self._loss_ema)
        slow = per_channel_loss.new_tensor(self._loss_ema_slow)
        scores = fast / slow.clamp(min=self._EMA_EPS)
        w_soft = C * scores / scores.sum().clamp(min=self._EMA_EPS)

        self._last_weights = w_soft.tolist()
        # Telemetry (keys preserved for the callback contract):
        self._last_cal_ratio = scores.tolist()       # progress ratio fast/slow
        self._last_cal_score = list(self._loss_ema)  # fast EMA
        self._last_gates = w_soft.tolist()

        return (w_soft * per_channel_loss).sum()

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
        y_pred_det: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Event-magnitude-weighted level anchor on shared shape windows.

        Strict orthogonality: shape is demeaned within exactly the same
        non-overlapping windows used by this level term.
        """
        W = self._window_size(T)

        # Series-level occurrence-probability weight — zero hyperparameters.
        # Series with any true event above threshold get weight 1.0.
        # Peaceful series get the model's predicted event probability for its
        # highest-valued timestep: self-correcting and no floor constant needed.
        tau = self.non_zero_threshold
        true_series_event = (y_true > tau).any(dim=1).to(dtype=y_true.dtype)
        if y_pred_det is not None:
            p_series = torch.sigmoid(y_pred_det.max(dim=1).values - tau)
            series_w = (
                true_series_event + (1.0 - true_series_event) * p_series.detach()
            )
        else:
            series_w = true_series_event

        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )  # (B, n_w) or (B, n_w, C)
        level_losses = self._log_cosh(window_means)
        n_windows = level_losses.shape[1]

        if level_losses.dim() == 3:
            num = (series_w.unsqueeze(1) * level_losses).sum(dim=(0, 1))
            den = (series_w.sum(dim=0) * n_windows).clamp(min=self._EMA_EPS)
        else:
            num = (series_w.unsqueeze(1) * level_losses).sum()
            den = (series_w.sum() * n_windows).clamp(min=self._EMA_EPS)

        level = num / den
        return T * self._LEVEL_SCALE * level

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
        # Strict orthogonality: use the exact same non-overlapping windows as
        # the level anchor.
        W = self._window_size(T)
        windows = list(e.split(W, dim=1))  # list of (B, W_i) or (B, W_i, C)
        e_shape = torch.cat(
            [w - w.mean(dim=1, keepdim=True) for w in windows], dim=1
        )  # (B, T) or (B, T, C)

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
        loss_occ, z_event, p_event = self._occurrence_hurdle_terms(y_pred, y_true)
        # Event weight: true events=1.0, peace cells=predicted event probability.
        # Self-correcting and zero hyperparameters — see _event_weight().
        event_weight = self._event_weight(z_event, p_event)

        # ── Adaptive DRO alpha ─────────────────────────────────────────
        # f_event: fraction of cells with abs_max above non_zero_threshold.
        # Low f_event (sparse batch, ~0.03 at pg-month) → alpha near _DRO_ALPHA_MAX
        # (concentrate harder on the few event cells).
        # High f_event → alpha near _DRO_ALPHA_MIN (relax; event weight carries
        # the differentiation).
        f_event = (abs_max.detach() > self.non_zero_threshold).float().mean().item()
        dro_alpha = self._DRO_ALPHA_MIN + (self._DRO_ALPHA_MAX - self._DRO_ALPHA_MIN) * (1.0 - f_event)

        # ── Per-series temporal DRO ────────────────────────────────────
        w_dro = self._dro_weights_2d(cell_loss, y_true, dro_alpha)  # (B, T) or (B, T, C)
        w_total = torch.nan_to_num(
            event_weight * w_dro, nan=1.0, posinf=1.0, neginf=0.0
        )

        # ── Hájek self-normalized shape (composition-robust) ──────────
        # Weight-mass-weighted mean of the per-cell log_cosh — the
        # self-normalized (Hájek) ratio estimator loss = Σ(w·ℓ)/Σ(w). Numerator
        # and denominator move together with the batch's event composition, so
        # the shape scale is invariant to how many event cells the batch happens
        # to contain. This replaces the cross-batch EMA rescale: no running
        # state, no lag, no composition memory (the EMA lag was implicated in the
        # flat-collapse oscillation).
        if w_total.dim() == 3:
            num = (w_total * cell_loss).sum(dim=(0, 1))              # (C,)
            den = w_total.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)   # (C,)
            loss_shape = num / den                                  # (C,)
        else:
            num = (w_total * cell_loss).sum()
            den = w_total.sum().clamp(min=self._EMA_EPS)
            loss_shape = num / den                                  # scalar

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, y_true, T, y_pred_det=y_pred.detach())

        # Occurrence BCE is automatically scale-matched to the existing shape
        # loss, avoiding a new tuning constant while still giving explicit
        # zero-vs-event supervision.
        occ_scale = loss_shape.detach() / loss_occ.detach().clamp(min=self._EMA_EPS)
        loss_occ_scaled = occ_scale * loss_occ

        # ── Multi-resolution spectral loss (always on) ──────────────
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        # ── Core objective assembly & telemetry ────────────────────
        if loss_shape.dim() == 0:
            # Univariate path
            total_loss = loss_shape + loss_level + loss_spec + loss_occ_scaled
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "occ": [float(loss_occ_scaled.detach())],
                "spec": [float(loss_spec.detach()) if loss_spec.dim()==0 else float(loss_spec)],
                "weight": [1.0],
            }
        else:
            # Multivariate path
            per_channel_total = loss_shape + loss_level + loss_spec + loss_occ_scaled
            total_loss = self._combine_channels(per_channel_total, y_pred, y_true)
            
            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "occ": loss_occ_scaled.detach().tolist(),
                "spec": spec_list,
                "weight": weights,
                "ema": self._loss_ema_slow or [float("nan")] * C,
                "cal_ratio": getattr(self, "_last_cal_ratio", [1.0] * C),
                "cal_score": getattr(self, "_last_cal_score", [1.0] * C),
                "gates": getattr(self, "_last_gates", [1.0] * C),
                "contribution": [
                    weights[c] * float(per_channel_total.detach()[c]) for c in range(C)
                ],
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