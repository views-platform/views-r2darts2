import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    Operates in asinh space (AsinhTransform target scaler). Designed for
    UCDP GED conflict fatality forecasting at country-month level:
    ~92% zeros for sb, ~97% for ns, ~98% for os.

    ── Changes from v46 ────────────────────────────────────────────────

    1. **Level scale_factor = W** (was 2*T). The 2*T overcompensated by 6x,
       making level 96% of total loss. W is the mathematically correct
       compensation for the 1/W gradient attenuation from the mean operator.

    2. **Relative z-score calibration term (z²).** Penalizes the squared
       z-score of the bias between predicted and true event-cell means,
       normalized by the true event-cell standard deviation. This is a
       dimensionless effect size — no weight hyperparameter needed.
       Per-channel: each target (sb, ns, os) gets its own calibration
       signal scaled to its own variability.

    3. **Event gate floor removed.** The sigmoid alone provides sufficient
       peace suppression (σ(10×(0.48−0.88)) ≈ 0.018 at asinh baseline).
       The 0.0125 floor was a hardcoded constant that caused Hájek dilution
       at high sparsity.

    ── Components ───────────────────────────────────────────────────────

    1. DC/AC decomposition — per-window demeaning.
    2. Gated + magnitude-graded event weighting.
    3. Per-series temporal DRO (event-gated).
    4. Windowed level anchor — W-scaled log_cosh on per-window means.
    5. Relative z-score calibration — per-channel mean-matching (z²).
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

        logger.info("SpotlightLossLogcosh v47 | threshold=%.4f", non_zero_threshold)

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
        """Combine per-channel losses by relative learning progress."""
        C = per_channel_loss.shape[0]
        batch_loss_det = per_channel_loss.detach()
        beta = self._EMA_BETA

        if (
            self._loss_ema is None
            or self._loss_ema_slow is None
            or len(self._loss_ema) != C
        ):
            self._loss_ema = batch_loss_det.tolist()
            self._loss_ema_slow = batch_loss_det.tolist()
        else:
            if self.training:
                for c in range(C):
                    self._loss_ema[c] = beta * self._loss_ema[c] + (1.0 - beta) * float(batch_loss_det[c])
                    self._loss_ema_slow[c] = beta * self._loss_ema_slow[c] + (1.0 - beta) * self._loss_ema[c]

        fast = per_channel_loss.new_tensor(self._loss_ema)
        slow = per_channel_loss.new_tensor(self._loss_ema_slow)
        scores = fast / slow.clamp(min=self._EMA_EPS)
        w_soft = C * scores / scores.sum().clamp(min=self._EMA_EPS)

        self._last_weights = w_soft.tolist()
        self._last_cal_ratio = scores.tolist()
        self._last_cal_score = list(self._loss_ema)
        self._last_gates = w_soft.tolist()

        return (w_soft * per_channel_loss).sum()

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
        y_pred_det: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Event-gated windowed level anchor with W-scaled Hájek normalization.

        scale_factor = W compensates the 1/W gradient attenuation from the
        mean operator. This is the mathematically exact inverse — no
        hardcoded constant, derived from the sequence length.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )
        level_losses = self._log_cosh(window_means)

        if y_pred_det is not None:
            abs_max_series = torch.max(y_true.abs(), y_pred_det.abs())
        else:
            abs_max_series = y_true.abs()
        series_mag = abs_max_series.max(dim=1).values
        series_gate = torch.sigmoid(
            10.0 * (series_mag - self.non_zero_threshold)
        )
        series_w = series_gate

        scale_factor = W  # FIX: was 2*T, now W (exact gradient compensation)
        n_windows = level_losses.shape[1]
        if level_losses.dim() == 3:
            num = (series_w.unsqueeze(1) * level_losses).sum(dim=(0, 1))
            den = (series_w.sum(dim=0) * n_windows).clamp(min=self._EMA_EPS)
            return scale_factor * num / den
        else:
            num = (series_w.unsqueeze(1) * level_losses).sum()
            den = (series_w.sum() * n_windows).clamp(min=self._EMA_EPS)
            return scale_factor * num / den

    def _calibration_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Relative z-score calibration on event cells. Per-channel.

        Computes the z-score of the prediction bias:
            z = (mean(y_pred[event]) - mean(y_true[event])) / std(y_true[event])

        Returns z² (squared z-score). This is:
        - Dimensionless (no scale hyperparameter)
        - Self-normalizing (divides by truth std, adapting to each channel's range)
        - Per-channel (sb, ns, os each get independent calibration)
        - Quadratic growth (z²) provides strong gradient for large biases

        The squared z-score is the standard effect-size measure in statistics.
        z²=1 means the bias is 1 standard deviation — a large, easily detectable
        error. z²=4 means 2 sigma — the model is severely miscalibrated.
        This naturally scales the calibration force to each channel's variability.
        """
        event_mask = (y_true.abs() > self.non_zero_threshold).float()

        if y_pred.dim() == 3:
            # (B, T, C) — per-channel stats
            n_event = event_mask.sum(dim=(0, 1)).clamp(min=1.0)  # (C,)
            pred_mean = (y_pred * event_mask).sum(dim=(0, 1)) / n_event  # (C,)
            true_mean = (y_true * event_mask).sum(dim=(0, 1)) / n_event  # (C,)
            # Variance of truth on event cells
            true_centered = (y_true - true_mean) * event_mask
            true_var = (true_centered ** 2).sum(dim=(0, 1)) / n_event  # (C,)
            true_std = (true_var + self._EMA_EPS).sqrt()  # (C,)

            z_score = (pred_mean - true_mean) / true_std  # (C,)
            cal = z_score ** 2  # (C,) — squared z-score
            return cal
        else:
            # (B, T) — univariate
            n_event = event_mask.sum().clamp(min=1.0)
            pred_mean = (y_pred * event_mask).sum() / n_event
            true_mean = (y_true * event_mask).sum() / n_event
            true_centered = (y_true - true_mean) * event_mask
            true_var = (true_centered ** 2).sum() / n_event
            true_std = (true_var + self._EMA_EPS).sqrt()

            z_score = (pred_mean - true_mean) / true_std
            cal = z_score ** 2
            return cal

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

        # ── Hájek self-normalized shape ───────────────────────────────
        if w_total.dim() == 3:
            num = (w_total * cell_loss).sum(dim=(0, 1))
            den = w_total.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)
            loss_shape = num / den  # (C,)
        else:
            num = (w_total * cell_loss).sum()
            den = w_total.sum().clamp(min=self._EMA_EPS)
            loss_shape = num / den  # scalar

        # ── Windowed level anchor (W-scaled) ──────────────────────────
        loss_level = self._windowed_level_loss(e, y_true, T, y_pred_det=y_pred.detach())

        # ── Calibration loss (relative z-score, z²) ───────────────────
        loss_cal = self._calibration_loss(y_pred, y_true)

        # ── Spectral loss ──────────────────────────────────────────────
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        # ── Assemble total ────────────────────────────────────────────
        # scale_factor = W gives exact gradient compensation for level.
        # z² calibration provides explicit mean-matching per channel.
        if loss_shape.dim() == 0:
            total_loss = loss_shape + loss_level + loss_cal + loss_spec
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "cal": [float(loss_cal.detach())],
                "spec": [float(loss_spec.detach()) if loss_spec.dim() == 0 else float(loss_spec)],
                "weight": [1.0],
            }
        else:
            per_channel_total = loss_shape + loss_level + loss_cal
            if loss_spec.dim() == 0:
                per_channel_total = per_channel_total + float(loss_spec)
            else:
                per_channel_total = per_channel_total + loss_spec

            total_loss = self._combine_channels(per_channel_total, y_pred, y_true)

            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            cal_list = loss_cal.detach().tolist() if loss_cal.dim() else [float(loss_cal)] * C
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "cal": cal_list,
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
            _c = float(loss_cal.sum()) if loss_cal.dim() else float(loss_cal)
            _sp = float(loss_spec.sum()) if loss_spec.dim() else float(loss_spec)
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={_s:.6f} level={_l:.6f} "
                f"cal={_c:.6f} spec={_sp:.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh v47 | shape=%.6f level=%.6f cal=%.6f spec=%.6f total=%.6f",
            loss_shape.item() if loss_shape.dim() == 0 else loss_shape.sum().item(),
            loss_level.item() if loss_level.dim() == 0 else loss_level.sum().item(),
            loss_cal.item() if loss_cal.dim() == 0 else loss_cal.sum().item(),
            loss_spec.item() if loss_spec.dim() == 0 else loss_spec.sum().item(),
            total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"
