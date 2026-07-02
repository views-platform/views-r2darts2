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

        # Stateless batch-local weighting: no cross-batch EMA state is used for
        # shape/level normalization or channel routing.

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

    def _combine_channels(
        self,
        per_channel_loss: torch.Tensor,
        event_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-local event-mass routing across channels.

        Option 2: route channel gradient by event mass and current difficulty,
        without any cross-batch memory:

            score_c = event_mass_c * loss_c
            w_c     = C * score_c / Sum_k(score_k)

        event_mass comes from the same event-magnitude weighting used inside the
        shape term, so routing is data-driven and aligned with sparse conflict
        exposure. This removes batch-composition dependency from EMA-based
        progress routing while still prioritizing high-mass, hard channels.
        """
        C = per_channel_loss.shape[0]
        loss_det = per_channel_loss.detach().clamp(min=0.0)
        mass_det = event_mass.detach().clamp(min=0.0)
        score = mass_det * loss_det

        if float(score.sum()) <= self._EMA_EPS:
            w_soft = per_channel_loss.new_full((C,), 1.0)
        else:
            w_soft = C * score / score.sum().clamp(min=self._EMA_EPS)

        self._last_weights = w_soft.tolist()
        # Telemetry (keys preserved for callback contract):
        self._last_cal_ratio = mass_det.tolist()   # event mass per channel
        self._last_cal_score = loss_det.tolist()   # per-channel raw loss
        self._last_gates = score.tolist()          # routing score before renorm

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
        # Gate (peace suppression) x magnitude factor (1 + series_mag), mirroring
        # the shape-term event_mag: the bare sigmoid saturates above ~2 deaths and
        # is magnitude-blind across the tail; (1 + series_mag) restores bounded
        # asinh-space magnitude sensitivity so large wars pull more DC gradient
        # than small skirmishes. No new constant (asinh IS the scale).
        # series_w = 0.01 + 0.99 * torch.sigmoid(
        #     5.0 * (series_mag - self.non_zero_threshold)
        # )  # (B,) or (B, C)
        series_gate = 0.005 + 0.995 * torch.sigmoid(
            10.0 * (series_mag - self.non_zero_threshold)
        )  # (B,) or (B, C)
        series_w = series_gate * (1.0 + series_mag)  # magnitude-graded

        # Option 1: stateless weight-mass normalization. This removes all
        # cross-batch composition dependency from the level term.
        n_windows = level_losses.shape[1]
        if level_losses.dim() == 3:
            num = (series_w.unsqueeze(1) * level_losses).sum(dim=(0, 1))
            den = (series_w.sum(dim=0) * n_windows).clamp(min=self._EMA_EPS)
            return num / den
        else:
            num = (series_w.unsqueeze(1) * level_losses).sum()
            den = (series_w.sum() * n_windows).clamp(min=self._EMA_EPS)
            return num / den

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

        # Option 1: stateless weight-mass normalization for shape, removing any
        # cross-batch composition dependency from EMA-based rescaling.
        if w_total.dim() == 3:
            num = (w_total * cell_loss).sum(dim=(0, 1))
            den = w_total.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)
            loss_shape = num / den
            event_mass = w_total.detach().sum(dim=(0, 1))
        else:
            num = (w_total * cell_loss).sum()
            den = w_total.sum().clamp(min=self._EMA_EPS)
            loss_shape = num / den

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
            total_loss = self._combine_channels(per_channel_total, event_mass)
            
            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "spec": spec_list,
                "weight": weights,
                "ema": [float("nan")] * C,
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