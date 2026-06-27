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

    6. **Inverse-EMA term balancing (always on)** — the level anchor is
       T-scaled (×36), which would otherwise make it ~80% of the loss and
       starve the within-window shape learner (~7%).  Each of the three terms
       (shape / level / spectral) is divided by a slow, detached EMA of its own
       magnitude and renormalised to mean 1, so they contribute on comparable
       footing regardless of the fixed T multiplier.  Parameter-free and
       DRO-safe (tracks term scale, not the loss trend); applies to both the
       single-target and multivariate paths.

    7. **Per-channel multi-task balancing (multivariate only)** — on the
       joint sb/ns/os objective each channel's shape+level+spectral loss is
       reduced within its own channel (per-channel DRO normalisation) and the
       channels are combined with inverse-EMA scale normalisation so the
       high-magnitude sb channel cannot dominate the summed loss by raw
       scale.  The spectral term is computed per channel and balanced through
       the *same* combine as shape+level, so no part of the objective bypasses
       the balancing.  Each channel is divided by a slow exponential moving average
       of its own loss magnitude (detached, parameter-free), which neutralises
       cross-channel scale without reading the loss *trend* — so it stays
       compatible with DRO's non-monotonic, hard-example-switching dynamics.
       The single-target path is unchanged.

    ── Base cell loss: log_cosh × (1 + log(1+|x|³))  (proportional) ───

    Args:
        non_zero_threshold: Sigmoid center (AsinhTransform: 0.88 ≈ asinh(1))

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
    # Cross-channel scale-balancing EMA (see _combine_channels).  beta close
    # to 1 makes the per-channel scale estimate slow relative to DRO's
    # hard-example switching, so a transient loss spike is not mistaken for a
    # scale change.  eps guards the division for a channel whose loss → 0.
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

        # Per-channel running scale estimate for the multivariate (C>1) path.
        # A plain Python list (NOT an nn.Parameter or registered buffer): it is
        # detached training state, so it must stay out of state_dict — Darts
        # deep-copies the loss into criterion/train_criterion/val_criterion and
        # strict checkpoint loading fails on any extra serialised key.  None
        # until the first multivariate forward, then length-C.
        self._loss_ema: list[float] | None = None
        # Per-term running scale estimate (shape / level / spectral) for the
        # inverse-EMA *term* balancing in _combine_terms.  Same parameter-free,
        # DRO-safe slow-EMA mechanism as _loss_ema but applied across the three
        # loss terms instead of across channels, so the T-scaled level anchor
        # cannot dominate the within-window shape learner.  Plain Python list of
        # length 3 ([shape, level, spec]); None until the first forward.
        self._term_ema: list[float] | None = None
        # Detached per-forward telemetry for LossComponentCallback (plain
        # Python, never Parameter/buffer → stays out of state_dict).  Holds the
        # per-channel shape/level/spectral split, the inverse-EMA channel
        # weights, and each channel's weighted contribution to total_loss, so
        # the cross-channel balance and the within-channel term mix are both
        # visible in wandb.  None until the first forward.
        self._last_components: dict | None = None
        self._last_weights: list[float] | None = None
        self._last_term_weights: list[float] | None = None
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    # @staticmethod
    # def _log_cosh_proportional(x: torch.Tensor) -> torch.Tensor:
    #     """log_cosh with proportional sensitivity correction.

    #     log_cosh(x) × (1 + log(1 + |x|³)).

    #     For |x| < 1: ≈ 0.5x² (cubic interior shrinks faster toward zero,
    #         so noise cells are quieter than the old x² variant).
    #     For |x| > 2: ≈ |x| × 3·ln|x| (~50% steeper than x² formula).

    #     Gradient = tanh(x)·(1 + log1p(|x|³))
    #                + log_cosh(x)·3x²·sign(x)/(1+|x|³).
    #     """
    #     abs_x = torch.abs(x)
    #     lc = abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)
    #     return lc * (1.0 + torch.log1p(abs_x * abs_x * abs_x))

    @staticmethod
    def _dro_weights_2d(losses: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Per-series sqrt self-reweighting

        w_it = sqrt(loss_it / mean_i(loss))

        Sublinear concentration: a cell 16× harder than average gets 4×
        the gradient (not 16×).  Redistributes enough signal to fix
        systematic bias while still focusing on spikes.

        Returns weights with mean ≈ 1 per series, shape (B, T).
        """
        l = losses.detach()                                  # (B, T)
        mu = l.mean(dim=1, keepdim=True).clamp(min=1e-6)     # (B, 1)
        w = torch.sqrt(l / mu)                               # (B, T)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)  # renormalize mean=1
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _shape_loss(
        self, e_shape: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Event-magnitude- and DRO-weighted shape loss.

        Returns a scalar for 2D (B, T) input (single target) and a
        per-channel vector (C,) for 3D (B, T, C) input.  The per-series sqrt
        DRO reduces along the time axis and is therefore already
        channel-independent; for 3D the final weight normalisation and
        reduction are done per channel so the result is each channel's own
        shape loss, ready for cross-channel uncertainty weighting.  The 2D
        branch is identical to the original inlined computation.
        """
        cell_loss = self._log_cosh(e_shape)
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        event_mag = 0.01 + 0.99 * torch.sigmoid(
            5.0 * (abs_max - self.non_zero_threshold)
        )
        w_dro = self._dro_weights_2d(cell_loss, y_true)
        w_total = event_mag * w_dro

        if cell_loss.dim() == 3:
            w_total = w_total / w_total.mean(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
            return (w_total * cell_loss).mean(dim=(0, 1))         # (C,)

        w_total = w_total / w_total.mean()
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
        return (w_total * cell_loss).mean()                      # scalar

    def _combine_channels(self, per_channel_loss: torch.Tensor) -> torch.Tensor:
        """Combine per-channel losses with inverse-EMA scale normalisation.

        Each channel is divided by a slow exponential moving average of its
        own loss magnitude and the result is summed:

            w_c = (1 / (EMA(L_c) + eps)),  normalised to mean 1
            combined = Σ_c  w_c · L_c

        so the high-magnitude sb channel cannot dominate the joint objective
        by raw scale.  The EMA is detached and parameter-free (a Python list,
        not an nn.Parameter/buffer) so it adds no gradient pathway and nothing
        to state_dict.

        Crucially this reads only the channel *scale*, never the loss *trend*,
        with a horizon (beta → 1) much slower than DRO's hard-example
        switching.  A DRO spike raises L_c instantaneously but the EMA lags,
        so the channel's contribution rises briefly above 1 and self-corrects
        as the EMA catches up — no runaway, fully DRO-compatible.
        """
        C = per_channel_loss.shape[0]
        losses_det = per_channel_loss.detach()

        # Update the per-channel running scale (plain Python floats).  Re-init
        # if the channel count changed (e.g. switching target configs).
        if self._loss_ema is None or len(self._loss_ema) != C:
            self._loss_ema = losses_det.clamp(min=self._EMA_EPS).tolist()
        else:
            beta = self._EMA_BETA
            for c in range(C):
                self._loss_ema[c] = (
                    beta * self._loss_ema[c] + (1.0 - beta) * float(losses_det[c])
                )

        ema = per_channel_loss.new_tensor(self._loss_ema)
        w = 1.0 / (ema + self._EMA_EPS)
        w = (w / w.mean()).detach()                 # mean 1 → preserve loss scale
        self._last_weights = w.tolist()             # logging only (plain floats)
        return (w * per_channel_loss).sum()

    def _combine_terms(
        self,
        shape_l: torch.Tensor,
        level_l: torch.Tensor,
        spec_l: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale-normalise the three loss terms (shape / level / spectral).

        The windowed level anchor is multiplied by T (≈36) which makes its raw
        magnitude ~80% of every channel's loss, drowning the within-window
        shape learner (~7%) and starving the temporal signal.  This applies the
        *same* inverse-EMA scale normalisation used across channels, but across
        the three terms:

            s_k  = mean|L_k|                         (detached per-term scale)
            w_k  = (1 / (EMA(s_k) + eps)),  normalised to mean 1 over k
            L_k' = w_k · L_k

        so shape, level and spectral contribute on comparable footing
        regardless of the fixed T multiplier — no new constant, no learnable
        parameter, fully adaptive to the realised term scales.

        The EMA is detached and parameter-free (a Python list, not an
        nn.Parameter/buffer → nothing in state_dict) and slow (beta → 1) so it
        tracks term *scale*, never the loss *trend*, staying compatible with
        DRO's hard-example switching.  Works for both scalar (single target)
        and (C,) per-channel term tensors: the scale reduces with mean over all
        elements, so there is one shared weight per term across channels and
        the cross-channel balancing still happens afterwards in
        _combine_channels.
        """
        terms = (shape_l, level_l, spec_l)
        scales = [float(t.detach().abs().mean()) for t in terms]

        # Update the per-term running scale (plain Python floats).
        if self._term_ema is None:
            self._term_ema = [max(s, self._EMA_EPS) for s in scales]
        else:
            beta = self._EMA_BETA
            for k in range(3):
                self._term_ema[k] = (
                    beta * self._term_ema[k] + (1.0 - beta) * scales[k]
                )

        ema = shape_l.new_tensor(self._term_ema)
        w = 1.0 / (ema + self._EMA_EPS)
        w = (w / w.mean()).detach()                 # mean 1 → preserve loss scale
        self._last_term_weights = w.tolist()        # logging only (plain floats)
        return w[0] * shape_l, w[1] * level_l, w[2] * spec_l

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
    ) -> torch.Tensor:
        """Event-magnitude-weighted windowed level anchor.

        Splits the T-length error into non-overlapping windows, computes
        log_cosh_proportional on per-window means, then weights each
        series by its event magnitude.  Without this weighting, the 76%
        peace series (with near-zero DC error) dilute the level gradient
        that event series need to correct their systematic underprediction.

        Uses proportional loss to avoid gradient saturation: plain
        log_cosh has gradient tanh(x) → 1 for |x| > 2, meaning a 2×
        underprediction gets the same gradient as a 10× underprediction.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )  # (B, n_windows)
        level_losses = self._log_cosh(window_means)

        # Per-series event magnitude: max |y_true| across time → sigmoid
        series_mag = y_true.abs().max(dim=1).values  # (B,) | (B, C)
        series_w = 0.01 + 0.99 * torch.sigmoid(
            5.0 * (series_mag - self.non_zero_threshold)
        )  # (B,) | (B, C)

        if level_losses.dim() == 3:
            # Per-channel level anchor: normalise event weights within each
            # channel and reduce per channel for cross-channel combination.
            series_w = series_w / series_w.mean(dim=0, keepdim=True).clamp(min=1e-8)
            weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows, C)
            return T * weighted.mean(dim=(0, 1))             # (C,)

        series_w = series_w / series_w.mean().clamp(min=1e-8)
        # Weight each series' level loss
        weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows)
        return T * weighted.mean()

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only).

        Safe magnitude sqrt(re² + im² + ε) avoids gradient blowup at
        |z|→0.  DC bin is masked — level anchor already handles DC.
        Only series with signal above threshold are included.
        """
        if y_pred.dim() == 3:
            # Per-channel spectral loss so it folds into the same inverse-EMA
            # channel balancing as shape+level.  Pooling all channels into one
            # flat mean (the previous behaviour) is dominated by the
            # high-magnitude sb channel and bypasses the cross-channel
            # balancing entirely, letting sb re-dominate the joint objective
            # through the frequency term.
            C = y_pred.shape[-1]
            return torch.stack(
                [self._spectral_loss(y_pred[..., c], y_true[..., c]) for c in range(C)]
            )                                                        # (C,)

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
        # This makes shape and level orthogonal: shape handles within-window
        # patterns, level handles per-window DC.  No shared frequencies.
        W = max(6, T // 3)
        windows = list(e.split(W, dim=1))  # list of (B, W_i)
        e_shape = torch.cat(
            [w - w.mean(dim=1, keepdim=True) for w in windows], dim=1
        )  # (B, T) — zero-mean within each window

        # ── Shape + level losses (per-channel when multivariate) ──────
        loss_shape_pc = self._shape_loss(e_shape, y_true, y_pred)   # scalar | (C,)
        loss_level_pc = self._windowed_level_loss(e, y_true, T)     # scalar | (C,)

        # ── Multi-resolution spectral loss (always on) ──────────────
        # scalar (single target) | (C,) per-channel (multivariate)
        loss_spec_pc = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec_pc = self._spectral_loss(y_pred, y_true)

        # ── Inverse-EMA term balancing (shape / level / spectral) ─────
        # The level anchor's ×T scaling otherwise makes it ~80% of the loss,
        # starving the shape learner (~7%).  Equalise the three term scales
        # with the same parameter-free slow-EMA mechanism used across channels
        # so shape/level/spec contribute comparably.  Applies to both the
        # single-target and multivariate paths (the ×T domination is identical
        # in both).
        loss_shape_pc, loss_level_pc, loss_spec_pc = self._combine_terms(
            loss_shape_pc, loss_level_pc, loss_spec_pc
        )
        term_weights = self._last_term_weights or [1.0, 1.0, 1.0]
        term_ema = list(self._term_ema) if self._term_ema else [float("nan")] * 3

        # ── Core objective ────────────────────────────────────────────
        # Univariate: sum of the term-balanced shape+level+spectral.
        # Multivariate: combine the per-channel term-balanced objectives with
        # inverse-EMA scale normalisation so the sb channel's magnitude cannot
        # dominate ns/os.  The spectral term is balanced through the *same*
        # combine as shape+level so no part of the objective bypasses the
        # balancing.
        if loss_shape_pc.dim() == 0:
            loss_shape = loss_shape_pc
            loss_level = loss_level_pc
            loss_spec = loss_spec_pc
            total_loss = loss_shape + loss_level + loss_spec
            # ── Telemetry (single target): one "channel" ──────────────
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "spec": [float(loss_spec.detach())],
                "ema": [float("nan")],     # no cross-channel balance here
                "weight": [1.0],
                "term_weight": list(term_weights),
                "term_ema": term_ema,
            }
        else:
            per_channel_total = loss_shape_pc + loss_level_pc + loss_spec_pc
            total_loss = self._combine_channels(per_channel_total)
            loss_shape = loss_shape_pc.sum().detach()  # logging only
            loss_level = loss_level_pc.sum().detach()  # logging only
            loss_spec = (
                loss_spec_pc.sum().detach()
                if loss_spec_pc.dim() else loss_spec_pc
            )
            # ── Telemetry (multivariate): per-channel term split (after ──
            # term balancing) + the inverse-EMA channel balance.  spec may be a
            # shared scalar if STFT is off (T<6); broadcast it across channels
            # for a uniform schema.
            C = per_channel_total.shape[0]
            spec_list = (
                loss_spec_pc.detach().tolist()
                if loss_spec_pc.dim() else [float(loss_spec_pc)] * C
            )
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape_pc.detach().tolist(),
                "level": loss_level_pc.detach().tolist(),
                "spec": spec_list,
                "ema": list(self._loss_ema) if self._loss_ema else [float("nan")] * C,
                "weight": weights,
                "term_weight": list(term_weights),
                "term_ema": term_ema,
                # weighted contribution of each channel to total_loss
                "contribution": [
                    weights[c] * float(per_channel_total.detach()[c])
                    for c in range(C)
                ],
            }

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f "
            "spec=%.6f total=%.6f",
            loss_shape.item(), loss_level.item(),
            loss_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"