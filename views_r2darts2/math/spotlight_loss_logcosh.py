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

    2. **Gated + magnitude-graded event weighting.**
       event_mag = gate × (1 + abs_max), gate = 0.005 + 0.995 × σ(10 × (abs_max − τ)).
       The gate suppresses peace (→ ~0.005) vs conflict (→ ~1); the (1 + abs_max)
       factor — bounded because abs_max is in asinh space — restores magnitude
       sensitivity across the 4-OOM tail so large wars outweigh small skirmishes
       instead of saturating flat. No model-state dependency (abs_max detached).

    3. **Per-series temporal DRO** — within-series shock therapy.
       Z-scores log(cell_loss) along time axis per series.  Upweights
       proportionally harder timesteps *relative to that series*.

    4. **Windowed level anchor** — T-scaled log_cosh on per-window means,
       peace/conflict *gated* (NOT magnitude-graded — gating only, so the DC
       anchor does not over-weight the rare highest-magnitude primary-channel
       series). Cross-batch EMA normalized for composition invariance.

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

        # Two-timescale self-referential loss tracking for progress routing.
        # Both EMAs reuse the single _EMA_BETA constant (slow is the EMA of
        # fast), so no extra timescale/hyperparameter is introduced.
        self._loss_ema: list[float] | None = None       # fast EMA (~1/(1-beta))
        self._loss_ema_slow: list[float] | None = None  # slow EMA (~2/(1-beta))

        # Running cross-batch scale of the shape-weight magnitude — keeps the
        # event spotlight composition-invariant across batches (see forward()).
        self._w_norm_ema: list[float] | None = None

        # Running cross-batch scale of the level-anchor series weight — same
        # role as _w_norm_ema but for _windowed_level_loss. Removes the level
        # term's per-batch-mean composition dependence (an all-peace batch
        # previously snapped every peaceful series back to weight ~1).
        self._series_w_norm_ema: list[float] | None = None

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
        """Event-magnitude-weighted windowed level anchor.

        Splits the T-length error into non-overlapping windows, computes
        log_cosh on per-window means, then weights each series by its 
        event magnitude. No DRO, no CMW, just the sigmoid weight.

        We scale by the window size W instead of sequence length T. Because the
        mean operator on a window of size W reduces gradient magnitude by a
        factor of 1/W, multiplying by W is the exact mathematical inverse of
        the operator's gradient attenuation, balancing level and shape losses
        naturally and stably.

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
        # GATE ONLY on the level anchor (deliberately NOT magnitude-graded).
        # The shape term gets gate x (1 + abs_max) to restore tail-peak
        # sensitivity, but the level anchor is DC/mean correction, x T-scaled.
        # Multiplying it by (1 + series_mag) too concentrated ~10x-amplified,
        # T-scaled DC gradient on the rare highest-magnitude series — which live
        # in the primary channel (sb/ch0) — and empirically drove ch0 to
        # 2.3-2.5x over-prediction while low-magnitude ch2 stayed under (0.75x),
        # plus fed the grad-norm shocks (max 2k-20k). The gate alone still
        # preserves per-country peace/conflict contrast (so no templating), it
        # just stops the level anchor from chasing magnitude on the primary
        # channel. It also lowers the per-batch variance of series_w, keeping the
        # cross-batch EMA denominator below cleaner.
        # series_w = 0.01 + 0.99 * torch.sigmoid(
        #     5.0 * (series_mag - self.non_zero_threshold)
        # )  # (B,) or (B, C)
        series_w = 0.005 + 0.995 * torch.sigmoid(
            10.0 * (series_mag - self.non_zero_threshold)
        )  # (B,) or (B, C) — gate only

        # Normalize by a running CROSS-BATCH EMA of the weight's own mean rather
        # than the current batch mean. The old per-batch-mean normalization was
        # composition-dependent: an all-peace batch (mean ~= 0.005) was divided
        # by ~0.005 and snapped every peaceful series back to weight ~1, erasing
        # the peace/conflict contrast and handing peaceful series full level
        # gradient. Dividing by the running mean keeps the peace-vs-conflict (and
        # now magnitude) contrast intact in every batch while holding the long-run
        # level scale stable. Detached denominator — rescale only. Mirrors the
        # shape term's _w_norm_ema. The magnitude factor is applied BEFORE this
        # normalization so the mean-1 renorm preserves (not cancels) the tilt.
        beta = self._EMA_BETA
        if series_w.dim() == 2:  # (B, C) — 3D input, per-channel
            batch_mean = series_w.detach().mean(dim=0)  # (C,)
            n = batch_mean.numel()
            if self._series_w_norm_ema is None or len(self._series_w_norm_ema) != n:
                self._series_w_norm_ema = batch_mean.clamp(min=self._EMA_EPS).tolist()
            else:
                for c in range(n):
                    self._series_w_norm_ema[c] = (
                        beta * self._series_w_norm_ema[c]
                        + (1.0 - beta) * float(batch_mean[c])
                    )
            denom = series_w.new_tensor(self._series_w_norm_ema).clamp(min=self._EMA_EPS)
            series_w = series_w / denom
        else:  # (B,) — 2D input, global
            batch_mean = float(series_w.detach().mean())
            if self._series_w_norm_ema is None or len(self._series_w_norm_ema) != 1:
                self._series_w_norm_ema = [max(batch_mean, self._EMA_EPS)]
            else:
                self._series_w_norm_ema[0] = (
                    beta * self._series_w_norm_ema[0] + (1.0 - beta) * batch_mean
                )
            denom = max(self._series_w_norm_ema[0], self._EMA_EPS)
            series_w = series_w / denom

        # Weight each series' level loss (scaled by natural log dampened sequence length)
        scale_factor = T # / math.log(T)
        if level_losses.dim() == 3:
            weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows, C)
            return scale_factor * weighted.mean(dim=(0, 1))  # (C,)
        else:
            weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows)
            return scale_factor * weighted.mean()  # scalar

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
        # event_mag = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.non_zero_threshold))
        event_gate = 0.005 + 0.995 * torch.sigmoid(10.0 * (abs_max - self.non_zero_threshold))
        event_mag = event_gate * (1.0 + abs_max)

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
        if loss_shape.dim() == 0:
            # Univariate path
            total_loss = loss_shape + loss_level + loss_spec
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "spec": [float(loss_spec.detach()) if loss_spec.dim()==0 else float(loss_spec)],
                "weight": [1.0],
            }
        else:
            # Multivariate path
            per_channel_total = loss_shape + loss_level + loss_spec
            total_loss = self._combine_channels(per_channel_total, y_pred, y_true)
            
            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
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