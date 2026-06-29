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

    2. **Sigmoid event-magnitude weighting** — ~50:1 contrast ratio.
       event_mag = 0.01 + 0.99 × σ(5 × (abs_max − τ)).  Peace → ~0.02,
       conflict → ~1.0.  No model-state dependency.

    3. **Per-series temporal DRO** — within-series shock therapy.
       Z-scores log(cell_loss) along time axis per series.  Upweights
       proportionally harder timesteps *relative to that series*.

    4. **Windowed level anchor** — T-scaled log_cosh on per-window means.

    5. **Multi-resolution STFT loss** — always on, ungated.
       log_cosh on magnitude-spectrum differences.  DC bin masked.

    ── Base cell loss: log_cosh × (1 + log(1+|x|³))  (proportional) ───

    Args:
        non_zero_threshold: Sigmoid center (AsinhTransform: 0.88 ≈ asinh(1))
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = True
    _EMA_BETA = 0.99
    _EMA_EPS = 1e-6

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

        # Running EMA of each channel's raw loss scale. Near convergence the
        # loss equals the task's irreducible noise variance, so this acts as a
        # parameter-free estimate of Kendall & Gal's sigma_c^2 for homoscedastic
        # channel balancing (see _combine_channels). No learnable parameters.
        self._loss_ema: list[float] | None = None

        # Running cross-batch scale of the shape-weight magnitude — keeps the
        # event spotlight composition-invariant across batches (see forward()).
        self._w_norm_ema: list[float] | None = None

        # Running EMA of each objective component's mean scale [shape, level, spec].
        # Three scalars — inter-channel balance is handled by _combine_channels.
        # Used by _assemble_objective for loss-ratio normalization.
        self._obj_ema: list[float] | None = None

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

    @staticmethod
    def _dro_weights_2d(losses: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Per-series sqrt self-reweighting

        w_it = sqrt(loss_it / mean_i(loss))

        Sublinear concentration: a cell 16× harder than average gets 4×
        the gradient (not 16×).  Redistributes enough signal to fix
        systematic bias while still focusing on spikes.

        Returns weights with mean ≈ 1 per series, shape (B, T) or (B, T, C).
        """
        l = losses.detach()                                  # (B, T) or (B, T, C)
        mu = l.mean(dim=1, keepdim=True).clamp(min=1e-6)     # (B, 1) or (B, 1, C)
        w = torch.sqrt(l / mu)                               # (B, T) or (B, T, C)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)  # renormalize mean=1
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def _combine_channels(self, per_channel_loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Combine per-channel losses by parameter-free homoscedastic-uncertainty
        weighting (inverse running-loss scale).

        Naive pooling of the per-target losses fails in two opposite ways:

        * An unweighted sum (Sum_c L_c) lets whichever channel carries the
          larger loss scale dominate the shared backbone, starving the minor
          targets.
        * A two-timescale "progress" ratio fast_c/slow_c collapses to 1 for
          every channel once training is steady — two EMAs that share the same
          smoothing constant beta converge together, so the ratio carries no
          dynamic range and silently degenerates back to the unweighted sum.

        Kendall & Gal (2018) show the Bayes-optimal weight for a homoscedastic
        task is 1/(2 sigma_c^2), where sigma_c^2 is the task's irreducible noise
        variance. Near convergence the running loss IS that variance, so we
        estimate sigma_c^2 by an EMA of the raw loss and weight each channel by
        its inverse — recovering uncertainty weighting WITHOUT the learnable
        log-variance parameters (which previously broke checkpoint loading) and
        WITHOUT any temperature / restoring-force constant:

            ema_c = EMA_beta(L_c)                      # estimate of sigma_c^2
            w_c   = C * (1/ema_c) / Sum_k (1/ema_k)    # mean(w) = 1, dimensionless

        Each channel's effective loss w_c * L_c then has comparable scale, so
        the per-channel gradients reaching the shared trunk are balanced and no
        target is starved. The +log sigma regulariser of Kendall & Gal is
        dropped because sigma_c^2 here is *estimated*, not a free parameter that
        could run to infinity. The EMA is a plain Python list and the weights
        are detached, so this adds no autograd path and no state_dict keys; its
        slow timescale (beta=0.99) keeps it DRO-safe — it tracks scale, not the
        non-monotonic per-step DRO spikes.
        """
        C = per_channel_loss.shape[0]
        batch_loss_det = per_channel_loss.detach()
        beta = self._EMA_BETA

        # ── Running per-channel loss-scale estimate (sigma_c^2) ──────
        if self._loss_ema is None or len(self._loss_ema) != C:
            self._loss_ema = batch_loss_det.clamp(min=self._EMA_EPS).tolist()
        else:
            for c in range(C):
                self._loss_ema[c] = beta * self._loss_ema[c] + (1.0 - beta) * float(batch_loss_det[c])

        # ── Inverse-scale (homoscedastic-uncertainty) weighting ──────
        ema = per_channel_loss.new_tensor(self._loss_ema).clamp(min=self._EMA_EPS)
        inv = 1.0 / ema
        w_soft = C * inv / inv.sum().clamp(min=self._EMA_EPS)

        self._last_weights = w_soft.tolist()
        # Telemetry (keys preserved for the callback contract):
        #   cal_ratio = scale-normalised relative loss L_c/EMA(L_c) (~1 at steady state),
        #   cal_score = running loss-scale EMA, gates = applied channel weights.
        self._last_cal_ratio = (batch_loss_det / ema).tolist()
        self._last_cal_score = list(self._loss_ema)
        self._last_gates = w_soft.tolist()

        return (w_soft * per_channel_loss).sum()

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
        y_pred_det: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Event-magnitude-weighted windowed level anchor.

        Splits the T-length error into non-overlapping windows, computes
        log_cosh on per-window means, then weights each series by its
        event magnitude. No DRO, no CMW, just the sigmoid weight.

        No manual scale factor is applied — the natural scale difference
        between this component and shape/spectral is resolved by
        _assemble_objective's loss-ratio normalisation.

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
        series_w = 0.01 + 0.99 * torch.sigmoid(
            5.0 * (series_mag - self.non_zero_threshold)
        )  # (B,) or (B, C)
        
        # FIX: Normalize per channel if 3D, else global
        if series_w.dim() == 2:
            series_w = series_w / series_w.mean(dim=0, keepdim=True).clamp(min=1e-8)
        else:
            series_w = series_w / series_w.mean().clamp(min=1e-8)

        # Weight each series' level loss (no manual scale — handled by _assemble_objective)
        scale_factor = 1.0
        if level_losses.dim() == 3:
            weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows, C)
            return scale_factor * weighted.mean(dim=(0, 1))  # (C,)
        else:
            weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows)
            return scale_factor * weighted.mean()  # scalar

    def _assemble_objective(
        self,
        loss_shape: torch.Tensor,
        loss_level: torch.Tensor,
        loss_spec: torch.Tensor,
    ) -> torch.Tensor:
        """Combine Shape + Level + Spectral via loss-ratio normalization.

        Each component is divided by a detached running EMA of its own magnitude
        so that each contributes ≈ 1.0 normalised units at steady state — no
        matter what their natural scale difference is (level is typically ~W×
        smaller than shape due to the windowed averaging operator):

            L_total = L_shape / EMA(L_shape) + L_level / EMA(L_level) + L_spec / EMA(L_spec)

        Contrast with inverse-EMA weighting (w = 1/EMA, as in Kendall & Gal):
        that multi-task heuristic down-weights a component when its loss is
        *large* — the opposite of what is wanted here. For sub-objectives of the
        same prediction target you want more gradient on whichever aspect the
        model is currently failing at, not less. Loss-ratio normalisation
        achieves this: L/EMA > 1 when the model regresses, > 1 gradient; ≈ 1
        when converged, equal contribution; < 1 if the component is over-fitted.

        Three scalar EMAs (not 3×C) — inter-channel balance is left to
        _combine_channels (Kendall & Gal). The denominators are plain Python
        floats (detached), so this adds no autograd path and no state_dict keys.
        """
        beta = self._EMA_BETA
        eps  = self._EMA_EPS

        # Scalar representative values per component (mean across channels)
        sh_val = float(loss_shape.detach().mean() if loss_shape.dim() > 0 else loss_shape.detach())
        lv_val = float(loss_level.detach().mean() if loss_level.dim() > 0 else loss_level.detach())
        sp_val = float(loss_spec.detach().mean()  if loss_spec.dim()  > 0 else loss_spec.detach())

        if self._obj_ema is None:
            self._obj_ema = [max(sh_val, eps), max(lv_val, eps), max(sp_val, eps)]
        else:
            self._obj_ema[0] = beta * self._obj_ema[0] + (1.0 - beta) * max(sh_val, eps)
            self._obj_ema[1] = beta * self._obj_ema[1] + (1.0 - beta) * max(lv_val, eps)
            self._obj_ema[2] = beta * self._obj_ema[2] + (1.0 - beta) * max(sp_val, eps)

        sh_denom = max(self._obj_ema[0], eps)
        lv_denom = max(self._obj_ema[1], eps)
        sp_denom = max(self._obj_ema[2], eps)

        # Ratios stored for telemetry (≈1 at steady state, >1 when regressing)
        self._last_obj_ratios = [sh_val / sh_denom, lv_val / lv_denom, sp_val / sp_denom]

        return loss_shape / sh_denom + loss_level / lv_denom + loss_spec / sp_denom

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

        # ── Sigmoid event-magnitude weighting ─────────────────────────
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        event_mag = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.non_zero_threshold))

        # ── Per-series temporal DRO ────────────────────────────────────
        w_dro = self._dro_weights_2d(cell_loss, y_true)  # (B, T) or (B, T, C)
        w_total = torch.nan_to_num(event_mag * w_dro, nan=1.0, posinf=1.0, neginf=0.0)

        # Normalize the spotlight weights by a running (cross-batch) average of
        # their own magnitude rather than by the *current* batch mean. The
        # per-batch mean was composition-dependent: an all-peace batch (mean
        # weight ~= 0.01) was divided by ~0.01 and snapped back to ~1, erasing
        # the peace/conflict contrast and handing peaceful cells full gradient.
        # Dividing by the running mean keeps the absolute peace-vs-conflict
        # contrast intact in every batch while holding the long-run loss_shape
        # scale stable. The denominator is detached, so it only rescales.
        beta = self._EMA_BETA
        if w_total.dim() == 3:
            batch_w_mean = w_total.detach().mean(dim=(0, 1))  # (C,)
            n = batch_w_mean.numel()
            if self._w_norm_ema is None or len(self._w_norm_ema) != n:
                self._w_norm_ema = batch_w_mean.clamp(min=self._EMA_EPS).tolist()
            else:
                for c in range(n):
                    self._w_norm_ema[c] = beta * self._w_norm_ema[c] + (1.0 - beta) * float(batch_w_mean[c])
            denom = w_total.new_tensor(self._w_norm_ema).clamp(min=self._EMA_EPS)
            loss_shape = (w_total / denom * cell_loss).mean(dim=(0, 1))  # (C,)
        else:
            batch_w_mean = float(w_total.detach().mean())
            if self._w_norm_ema is None or len(self._w_norm_ema) != 1:
                self._w_norm_ema = [max(batch_w_mean, self._EMA_EPS)]
            else:
                self._w_norm_ema[0] = beta * self._w_norm_ema[0] + (1.0 - beta) * batch_w_mean
            denom = max(self._w_norm_ema[0], self._EMA_EPS)
            loss_shape = (w_total / denom * cell_loss).mean()  # scalar

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, y_true, T, y_pred_det=y_pred.detach())

        # ── Multi-resolution spectral loss (always on) ──────────────
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        # ── Core objective assembly & telemetry ────────────────────
        # Loss-ratio normalisation: each component ≈ 1.0 normalised units
        # at steady state; large errors attract proportionally more gradient.
        if loss_shape.dim() == 0:
            # Univariate path
            total_loss = self._assemble_objective(loss_shape, loss_level, loss_spec)
            ratios = getattr(self, "_last_obj_ratios", [1.0, 1.0, 1.0])
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "spec": [float(loss_spec.detach()) if loss_spec.dim()==0 else float(loss_spec)],
                "weight": [1.0],
                "obj_ema": list(self._obj_ema) if self._obj_ema else [float("nan")] * 3,
                "obj_ratio": ratios,
            }
        else:
            # Multivariate path: normalise within component types, then balance channels
            per_channel_total = self._assemble_objective(loss_shape, loss_level, loss_spec)
            total_loss = self._combine_channels(per_channel_total, y_pred, y_true)

            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            weights = self._last_weights or [1.0] * C
            ratios = getattr(self, "_last_obj_ratios", [1.0, 1.0, 1.0])
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "spec": spec_list,
                "weight": weights,
                "ema": self._loss_ema or [float("nan")] * C,
                "cal_ratio": getattr(self, "_last_cal_ratio", [1.0] * C),
                "cal_score": getattr(self, "_last_cal_score", [1.0] * C),
                "gates": getattr(self, "_last_gates", [1.0] * C),
                "obj_ema": list(self._obj_ema) if self._obj_ema else [float("nan")] * 3,
                "obj_ratio": ratios,
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