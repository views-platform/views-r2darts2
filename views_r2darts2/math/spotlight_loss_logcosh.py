import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    v50 — MSE level + MSE shape (fixes log_cosh saturation).

    ROOT CAUSE OF ALL FAILURES: log_cosh gradient = tanh(e) saturates at ±1.
    No scale_factor fixes this — the per-cell gradient is bounded.
    Large/sustained errors get the same gradient as small errors.

    FIX: Replace log_cosh with MSE (squared error) in BOTH level and shape.
    MSE gradient = 2e → grows linearly with error magnitude.
    - e=5: 10x stronger than log_cosh (10 vs 1)
    - e=9: 18x stronger (18 vs 1)

    This fixes:
    1. Underprediction: large errors get proportionally stronger correction
    2. Persistence: sustained conflict (large window-mean error) gets strong
       gradient instead of saturated tanh≈-1
    3. Large magnitude capture: MSE grows quadratically with error

    PERSISTENCE FIX: Shape loss demeans (e_shape = e - window_mean), so
    constant wrong predictions get ZERO shape loss. Only level can fix
    persistence. With log_cosh, level gradient saturates at tanh≈-1 for
    sustained conflict. With MSE, gradient = 2*window_mean_e → grows with
    the severity of sustained underprediction.

    Components:
    1. DC/AC decomposition — per-window demeaning (unchanged)
    2. Gated event weighting with linear mag (unchanged)
    3. Per-series DRO (unchanged)
    4. Level anchor — MSE on window means, T-scaled
    5. Shape — MSE on demeaned errors, Hájek-normalized
    6. z² calibration with std floor (unchanged)
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = False
    _EMA_BETA = 0.99
    _EMA_EPS = 1e-6

    def __init__(self, non_zero_threshold: float):
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
        super().__init__()
        self.non_zero_threshold = non_zero_threshold
        self._loss_ema: list[float] | None = None
        self._loss_ema_slow: list[float] | None = None
        self._last_components: dict | None = None
        self._last_weights: list[float] | None = None
        logger.info("SpotlightLossLogcosh v50 | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _dro_weights_2d(self, losses: torch.Tensor, soft_event_mask: torch.Tensor) -> torch.Tensor:
        l = losses.detach()
        m = soft_event_mask.detach().to(dtype=l.dtype).clamp(min=0.0, max=1.0)

        def _wmean(x, w):
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

    def _combine_channels(self, per_channel_loss, y_pred, y_true):
        C = per_channel_loss.shape[0]
        batch_loss_det = per_channel_loss.detach()
        beta = self._EMA_BETA
        if self._loss_ema is None or self._loss_ema_slow is None or len(self._loss_ema) != C:
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

    def _windowed_level_loss(self, e, y_true, T, y_pred_det=None):
        """MSE level anchor — unbounded gradient fixes persistence + underprediction."""
        W = max(6, T // 3)
        window_means = torch.stack([ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1)
        level_losses = window_means ** 2  # MSE instead of log_cosh

        if y_pred_det is not None:
            abs_max_series = torch.max(y_true.abs(), y_pred_det.abs())
        else:
            abs_max_series = y_true.abs()
        series_mag = abs_max_series.max(dim=1).values
        series_gate = torch.sigmoid(10.0 * (series_mag - self.non_zero_threshold))
        series_w = series_gate

        scale_factor = T
        n_windows = level_losses.shape[1]
        if level_losses.dim() == 3:
            num = (series_w.unsqueeze(1) * level_losses).sum(dim=(0, 1))
            den = (series_w.sum(dim=0) * n_windows).clamp(min=self._EMA_EPS)
            return scale_factor * num / den
        else:
            num = (series_w.unsqueeze(1) * level_losses).sum()
            den = (series_w.sum() * n_windows).clamp(min=self._EMA_EPS)
            return scale_factor * num / den

    def _calibration_loss(self, y_pred, y_true):
        event_mask = (y_true.abs() > self.non_zero_threshold).float()
        if y_pred.dim() == 3:
            n_event = event_mask.sum(dim=(0, 1)).clamp(min=1.0)
            pred_mean = (y_pred * event_mask).sum(dim=(0, 1)) / n_event
            true_mean = (y_true * event_mask).sum(dim=(0, 1)) / n_event
            true_centered = (y_true - true_mean) * event_mask
            true_var = (true_centered ** 2).sum(dim=(0, 1)) / n_event
            true_std = (true_var + self._EMA_EPS).sqrt().clamp(min=self.non_zero_threshold)
            z_score = (pred_mean - true_mean) / true_std
            return z_score ** 2
        else:
            n_event = event_mask.sum().clamp(min=1.0)
            pred_mean = (y_pred * event_mask).sum() / n_event
            true_mean = (y_true * event_mask).sum() / n_event
            true_centered = (y_true - true_mean) * event_mask
            true_var = (true_centered ** 2).sum() / n_event
            true_std = (true_var + self._EMA_EPS).sqrt().clamp(min=self.non_zero_threshold)
            z_score = (pred_mean - true_mean) / true_std
            return z_score ** 2

    def _spectral_loss(self, y_pred, y_true):
        if y_pred.dim() == 3:
            C = y_pred.shape[-1]
            return torch.stack([self._spectral_loss(y_pred[..., c], y_true[..., c]) for c in range(C)])
        pred, true = y_pred, y_true
        has_signal = ((torch.abs(true) > self.non_zero_threshold) | (torch.abs(pred.detach()) > self.non_zero_threshold)).any(dim=1)
        if not has_signal.any():
            return pred.new_tensor(0.0)
        pred, true = pred[has_signal], true[has_signal]
        T = pred.size(1)
        total = pred.new_tensor(0.0)
        n_valid = 0
        for n_fft, hop in self._SPECTRAL_RESOLUTIONS:
            if T < n_fft:
                continue
            window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            S_pred = torch.stft(pred, n_fft, hop_length=hop, win_length=n_fft, window=window, center=False, return_complex=True)
            S_true = torch.stft(true, n_fft, hop_length=hop, win_length=n_fft, window=window, center=False, return_complex=True)
            mag_pred = torch.sqrt(S_pred.real ** 2 + S_pred.imag ** 2 + 1e-8)
            mag_true = S_true.abs()
            mag_pred[:, 0, :] = 0.0
            mag_true[:, 0, :] = 0.0
            total = total + self._log_cosh(mag_pred - mag_true).mean()
            n_valid += 1
        return total / max(n_valid, 1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        T = y_pred.size(1)
        e = y_pred - y_true

        # DC/AC decomposition
        W = max(6, T // 3)
        windows = list(e.split(W, dim=1))
        e_shape = torch.cat([w - w.mean(dim=1, keepdim=True) for w in windows], dim=1)

        # MSE shape loss (replaces log_cosh)
        cell_loss = e_shape ** 2  # MSE instead of log_cosh

        # Event weighting (linear mag)
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        event_gate = torch.sigmoid(10.0 * (abs_max - self.non_zero_threshold))
        event_mag = event_gate * (1.0 + abs_max)
        soft_event_mask = torch.sigmoid(10.0 * (abs_max - self.non_zero_threshold))
        w_dro = self._dro_weights_2d(cell_loss, soft_event_mask)
        w_total = torch.nan_to_num(event_mag * w_dro, nan=1.0, posinf=1.0, neginf=0.0)

        # Hájek shape
        if w_total.dim() == 3:
            num = (w_total * cell_loss).sum(dim=(0, 1))
            den = w_total.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)
            loss_shape = num / den
        else:
            num = (w_total * cell_loss).sum()
            den = w_total.sum().clamp(min=self._EMA_EPS)
            loss_shape = num / den

        # MSE level
        loss_level = self._windowed_level_loss(e, y_true, T, y_pred_det=y_pred.detach())

        # Calibration
        loss_cal = self._calibration_loss(y_pred, y_true)

        # Spectral
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        # Assemble
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
                "contribution": [weights[c] * float(per_channel_total.detach()[c]) for c in range(C)],
            }

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossLogcosh: shape={float(loss_shape)}, level={float(loss_level)}")
        return total_loss

    def __repr__(self):
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"
