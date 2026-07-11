import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v48 — calibration through level redesign, no new terms.

    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting at country-month level:
    ~92% zeros for sb, ~97% for ns, ~98% for os.

    ── Design: calibration baked into existing components ─────────────

    1. **Level loss redesigned as z-score calibration.** The DC/level
       component is now z² of the event-cell mean bias:
           z = mean(e[event]) / std(y_true[event])
           level = z²
       This IS the calibration signal — no separate calibration term.
       Uses event_mag weighting (not soft sigmoid) to avoid peace-cell
       dilution. Computed at batch level for std stability.

       Properties:
       - Dimensionless (no scale hyperparameter)
       - Self-normalizing (divides by truth std per channel)
       - Per-channel (sb, ns, os each get independent calibration)
       - Quadratic gradient (2z/std grows with bias → strong push)
       - Converges to 0 when mean matches, then shape takes over

    2. **Shape loss unchanged (Hájek mean of log_cosh).** The AC
       component stays composition-robust for pattern learning.

    3. **Channel router redesigned as relative-loss routing.** Replaces
       the inert EMA-ratio (fast/slow ≈ 1.0) with sqrt-concentrated
       relative loss. Routes gradient to the worst-performing channel.

    4. **Gate floor removed.** Sigmoid alone provides sufficient peace
       suppression (σ(10×(0.48−0.88)) ≈ 0.018).

    ── AC-DC split maintained ─────────────────────────────────────────

    - DC (level) = batch-level event-cell mean bias as z²
    - AC (shape) = window-demeaned per-cell error (Hájek log_cosh)

    The window-mean demeaning for the shape loss is unchanged. The level
    loss operates on the same error tensor but aggregates it as an
    event-weighted z-score rather than a window mean, because window
    means are diluted by peace cells (1 event in 12 months → mean = 1/12).

    ── Components ───────────────────────────────────────────────────────

    1. DC/AC decomposition — per-window demeaning.
    2. Gated + magnitude-graded event weighting.
    3. Per-series temporal DRO (event-gated).
    4. Z-score level anchor — calibration as DC component (z²).
    5. Hájek shape — composition-robust AC component.
    6. Multi-resolution STFT loss (disabled by default).
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = False
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

        self._loss_ema: list[float] | None = None
        self._loss_ema_slow: list[float] | None = None

        self._last_components: dict | None = None
        self._last_weights: list[float] | None = None

        logger.info("SpotlightLossLogcosh v48 | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _dro_weights_2d(
        self,
        losses: torch.Tensor,
        soft_event_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Event-aware per-series DRO with robust denominator."""
        l = losses.detach()
        m = soft_event_mask.detach().to(dtype=l.dtype).clamp(min=0.0, max=1.0)

        def _wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            den = w.sum(dim=1, keepdim=True).clamp(min=self._EMA_EPS)
            return (x * w).sum(dim=1, keepdim=True) / den

        T = int(l.shape[1])
        W = max(6, T // 3)
        n_blocks = max(1, (T + W - 1) // W)

        means = []
        for lb, mb in zip(torch.tensor_split(l, n_blocks, dim=1), torch.tensor_split(m, n_blocks, dim=1)):
            means.append(_wmean(lb, mb))
        mom = torch.cat(means, dim=1)
        mu = mom.median(dim=1, keepdim=True).values.clamp(min=self._EMA_EPS)

        w = torch.sqrt(l / mu)

        w_active_mean = _wmean(w, m).clamp(min=self._EMA_EPS)
        w_normalized_active = w / w_active_mean

        w = 1.0 + m * (w_normalized_active - 1.0)

        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def _combine_channels(self, per_channel_loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Relative-loss channel routing with sqrt concentration.

        Replaces the inert EMA-ratio router (fast/slow ≈ 1.0). Routes
        gradient toward the worst-performing channel via:

            w_c = C · sqrt(ema_c / min_ema) / Σ_k sqrt(ema_k / min_ema)

        The sqrt gives sublinear concentration: a channel 4x worse gets
        2x the weight (not 4x). This prevents winner-take-all while
        still tilting toward the channel that needs the most help.
        """
        C = per_channel_loss.shape[0]
        batch_loss_det = per_channel_loss.detach()
        beta = self._EMA_BETA

        if (
            self._loss_ema is None
            or len(self._loss_ema) != C
        ):
            self._loss_ema = batch_loss_det.tolist()
        else:
            if self.training:
                for c in range(C):
                    self._loss_ema[c] = beta * self._loss_ema[c] + (1.0 - beta) * float(batch_loss_det[c])

        ema_tensor = per_channel_loss.new_tensor(self._loss_ema)
        min_ema = ema_tensor.min().clamp(min=self._EMA_EPS)
        scores = torch.sqrt(ema_tensor / min_ema)
        w_soft = C * scores / scores.sum().clamp(min=self._EMA_EPS)

        self._last_weights = w_soft.tolist()
        self._last_cal_ratio = (ema_tensor / min_ema).tolist()
        self._last_cal_score = list(self._loss_ema)
        self._last_gates = w_soft.tolist()

        return (w_soft * per_channel_loss).sum()

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, T: int,
        y_pred_det: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Relative-error level anchor (per-cell calibration in DC component).

        The DC/level component: Hájek ratio of weighted error² to weighted
        truth². This calibrates EVERY event cell toward truth, not just
        the aggregate mean:

            level = Σ(w · e²) / Σ(w · y_true²)

        Properties:
        - Model predicts 0  → level ≈ 1.0 (100% relative error per cell)
        - Model predicts 50% → level ≈ 0.25
        - Model matches      → level = 0.0
        - Gradient: 2e·w / Σ(w·y²) — grows with per-cell error

        This is dimensionless, self-normalizing (denominator is the true
        signal energy), and per-channel. Unlike z² (which only calibrates
        the mean), this calibrates the full distribution because it
        penalizes each cell's squared relative error.

        The AC-DC split is maintained:
        - DC (level) = per-cell magnitude calibration (this function)
        - AC (shape) = window-demeaned per-cell pattern (in forward)

        The window-mean demeaning for the shape loss is unchanged.
        """
        # Event mask and weighting (same as shape loss)
        if y_pred_det is not None:
            abs_max = torch.max(y_true.abs(), y_pred_det.abs())
        else:
            abs_max = y_true.abs()
        event_gate = torch.sigmoid(
            10.0 * (abs_max - self.non_zero_threshold)
        )
        event_mag = event_gate * (1.0 + abs_max)

        # True signal energy (denominator) — detached
        true_energy = (event_mag.detach() * y_true ** 2)

        # Error energy (numerator)
        error_energy = (event_mag * e ** 2)

        if e.dim() == 3:
            # (B, T, C) — per-channel
            num = error_energy.sum(dim=(0, 1))  # (C,)
            den = true_energy.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)  # (C,)
            return num / den  # (C,)
        else:
            # (B, T) — univariate
            num = error_energy.sum()
            den = true_energy.sum().clamp(min=self._EMA_EPS)
            return num / den  # scalar

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only)."""
        if y_pred.dim() == 3:
            C = y_pred.shape[-1]
            return torch.stack(
                [self._spectral_loss(y_pred[..., c], y_true[..., c]) for c in range(C)]
            )

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
            mag_pred = torch.sqrt(S_pred.real ** 2 + S_pred.imag ** 2 + 1e-8)
            mag_true = S_true.abs()
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
        W = max(6, T // 3)
        windows = list(e.split(W, dim=1))
        e_shape = torch.cat(
            [w - w.mean(dim=1, keepdim=True) for w in windows], dim=1
        )

        # ── Base cell loss ─────────────────────────────────────────────
        cell_loss = self._log_cosh(e_shape)

        # ── Gated + magnitude-graded event weighting ──────────────────
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        event_gate = torch.sigmoid(10.0 * (abs_max - self.non_zero_threshold))
        event_mag = event_gate * (1.0 + abs_max)

        soft_event_mask = torch.sigmoid(
            10.0 * (abs_max - self.non_zero_threshold)
        )
        w_dro = self._dro_weights_2d(cell_loss, soft_event_mask)
        w_total = torch.nan_to_num(event_mag * w_dro, nan=1.0, posinf=1.0, neginf=0.0)

        # ── Hájek self-normalized shape (AC component) ────────────────
        # Composition-robust pattern matching. The AC component captures
        # within-window timing/shape. Hájek mean makes it invariant to
        # event count, which is correct for pattern learning.
        if w_total.dim() == 3:
            num = (w_total * cell_loss).sum(dim=(0, 1))
            den = w_total.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)
            loss_shape = num / den  # (C,)
        else:
            num = (w_total * cell_loss).sum()
            den = w_total.sum().clamp(min=self._EMA_EPS)
            loss_shape = num / den  # scalar

        # ── Level (DC component — z-score calibration) ────────────────
        # The level loss IS the calibration signal: z² of the event-cell
        # mean bias. Uses event_mag weighting (not soft sigmoid) to avoid
        # peace-cell dilution. Computed at batch level for std stability.
        #
        # AC-DC split maintained:
        # - DC (level) = batch-level event-cell mean bias as z²
        # - AC (shape) = window-demeaned per-cell error (above)
        #
        # z² provides collective calibration: every event cell gets the
        # same gradient push (2z/std/n_event), effective for sparse channels.
        # Once the mean matches, z² → 0 and the shape loss takes over for
        # per-cell pattern correction.
        if e.dim() == 3:
            w_mag = event_mag.detach()  # (B, T, C) — ~0 for peace
            n_ev = w_mag.sum(dim=(0, 1)).clamp(min=1.0)  # (C,)
            ev_mean_e = (e * w_mag).sum(dim=(0, 1)) / n_ev  # (C,)
            ev_mean_true = (y_true * w_mag).sum(dim=(0, 1)) / n_ev  # (C,)
            ev_var_true = ((y_true - ev_mean_true.unsqueeze(0).unsqueeze(0)) ** 2 * w_mag).sum(dim=(0, 1)) / n_ev  # (C,)
            ev_std_true = (ev_var_true + self._EMA_EPS).sqrt()  # (C,)
            z_score = ev_mean_e / ev_std_true  # (C,)
            loss_level = z_score ** 2  # (C,)
        else:
            w_mag = event_mag.detach()
            n_ev = w_mag.sum().clamp(min=1.0)
            ev_mean_e = (e * w_mag).sum() / n_ev
            ev_mean_true = (y_true * w_mag).sum() / n_ev
            ev_var_true = ((y_true - ev_mean_true) ** 2 * w_mag).sum() / n_ev
            ev_std_true = (ev_var_true + self._EMA_EPS).sqrt()
            z_score = ev_mean_e / ev_std_true
            loss_level = z_score ** 2

        # ── Spectral loss ──────────────────────────────────────────────
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        # ── Assemble total ────────────────────────────────────────────
        if loss_shape.dim() == 0:
            total_loss = loss_shape + loss_level + loss_spec
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "spec": [float(loss_spec.detach()) if loss_spec.dim() == 0 else float(loss_spec)],
                "weight": [1.0],
            }
        else:
            per_channel_total = loss_shape + loss_level
            if loss_spec.dim() == 0:
                per_channel_total = per_channel_total + float(loss_spec)
            else:
                per_channel_total = per_channel_total + loss_spec

            total_loss = self._combine_channels(per_channel_total, y_pred, y_true)

            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "spec": spec_list,
                "weight": weights,
                "ema": self._loss_ema or [float("nan")] * C,
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
            "SpotlightLossLogcosh v48 | shape=%.6f level=%.6f spec=%.6f total=%.6f",
            loss_shape.item() if loss_shape.dim() == 0 else loss_shape.sum().item(),
            loss_level.item() if loss_level.dim() == 0 else loss_level.sum().item(),
            loss_spec.item() if loss_spec.dim() == 0 else loss_spec.sum().item(),
            total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"
