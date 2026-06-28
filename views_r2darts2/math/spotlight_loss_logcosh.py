import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v47 — Robust EMA-DRO and Chain-Rule Level Scaling.
    
    Designed for highly sparse, zero-inflated conflict forecasting (e.g., UCDP GED).
    Operates in transformed space (e.g., Asinh).

    Components:
    1. DC/AC Decomposition: Orthogonal shape (within-window) and level (per-window DC) losses.
    2. Sigmoid Event-Magnitude Weighting: ~50:1 contrast for conflict vs. peace.
    3. EMA-Based Temporal DRO: Robust shape loss weighting using a global Exponential Moving 
       Average of the loss, making it immune to volatile batch compositions.
    4. Chain-Rule Level Scaling: Level loss scaled by window size (W) to perfectly compensate 
       for the 1/W gradient attenuation caused by window averaging.
    5. Multi-Resolution STFT: Frequency-domain magnitude matching (AC bins only).
    6. Inverse-EMA Channel Balancing: Multivariate objective balanced by slow-moving channel 
       scale estimates, preventing high-magnitude channels from dominating.
    """
    
    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = True
    
    # EMA hyperparameters
    _EMA_BETA = 0.99      # Slow-moving EMA for channel balancing and DRO baseline
    _EMA_EPS = 1e-6       # Guard against division by zero
    _DRO_MAX_WEIGHT = 10.0 # Cap for EMA-DRO weights to prevent outlier explosion

    def __init__(self, non_zero_threshold: float):
        super().__init__()
        if non_zero_threshold <= 0.0:
            raise ValueError(f"non_zero_threshold must be positive, got {non_zero_threshold}")
            
        self.non_zero_threshold = non_zero_threshold
        
        # Detached state variables
        self._loss_ema: list[float] | None = None     # Per-channel scale EMA (multivariate)
        self._shape_loss_ema: float = 1.0             # Global shape loss EMA (for robust DRO)
        
        # Telemetry for callbacks
        self._last_components: dict | None = None
        self._last_weights: list[float] | None = None
        
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable log(cosh(x)): |x| + softplus(-2|x|) - ln(2)."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _ema_dro_weights(self, cell_loss: torch.Tensor) -> torch.Tensor:
        """
        Computes robust DRO weights using a global EMA of the shape loss.
        Normalized per-series to act strictly as a temporal spotlight.
        """
        # 1. Update global historical EMA
        batch_mean = cell_loss.detach().mean().item()
        self._shape_loss_ema = (
            self._EMA_BETA * self._shape_loss_ema + 
            (1.0 - self._EMA_BETA) * batch_mean
        )
        
        # 2. Compare cell loss to historical baseline (provides stable denominator for peace series)
        w_dro = torch.sqrt(cell_loss / max(self._shape_loss_ema, self._EMA_EPS))
        
        # 3. Clamp to prevent extreme outliers from generating massive weights
        w_dro = w_dro.clamp(max=self._DRO_MAX_WEIGHT)
        
        # 4. Normalize per series (along time dimension) to mean 1.
        # This ensures DRO only reweights timesteps *within* a series (temporal spotlight)
        # and does not reweight series against each other (which causes high-variance 
        # countries to dominate and overpredict).
        w_dro = w_dro / w_dro.mean(dim=1, keepdim=True).clamp(min=1e-8)
            
        return w_dro

    def _shape_loss(
        self, e_shape: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Event-magnitude and EMA-DRO weighted shape loss (AC component)."""
        cell_loss = self._log_cosh(e_shape)
        
        # Sigmoid spotlight: upweight cells with high absolute magnitude
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        event_mag = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.non_zero_threshold))
        
        # Robust DRO weights
        w_dro = self._ema_dro_weights(cell_loss)
        w_total = event_mag * w_dro

        if cell_loss.dim() == 3:
            # Multivariate: normalize and reduce per channel
            w_total = w_total / w_total.mean(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
            return (w_total * cell_loss).mean(dim=(0, 1))
            
        # Univariate
        w_total = w_total / w_total.mean()
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
        return (w_total * cell_loss).mean()

    def _combine_channels(self, per_channel_loss: torch.Tensor) -> torch.Tensor:
        """Inverse-EMA scale normalization for multivariate channel balancing."""
        C = per_channel_loss.shape[0]
        losses_det = per_channel_loss.detach()

        if self._loss_ema is None or len(self._loss_ema) != C:
            self._loss_ema = losses_det.clamp(min=self._EMA_EPS).tolist()
        else:
            for c in range(C):
                self._loss_ema[c] = (
                    self._EMA_BETA * self._loss_ema[c] + 
                    (1.0 - self._EMA_BETA) * float(losses_det[c])
                )

        ema = per_channel_loss.new_tensor(self._loss_ema)
        w = 1.0 / (ema + self._EMA_EPS)
        w = (w / w.mean()).detach()  # Normalize to mean 1
        
        self._last_weights = w.tolist()
        return (w * per_channel_loss).sum()

    def _windowed_level_loss(self, e: torch.Tensor, T: int) -> torch.Tensor:
        """
        Windowed level anchor (DC component).
        Scaled by window size (W) to perfectly compensate for the 1/W gradient 
        attenuation caused by the chain rule on the window mean operation.
        Batch-wise DRO is removed to prevent volatility from batch composition.
        """
        W = max(6, T // 3)
        window_means = torch.stack([ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1)
        level_losses = self._log_cosh(window_means)

        # Scale by W to restore gradient magnitude
        if level_losses.dim() == 3:
            return float(W) * level_losses.mean(dim=(0, 1))
            
        return float(W) * level_losses.mean()

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only, DC masked)."""
        if y_pred.dim() == 3:
            C = y_pred.shape[-1]
            return torch.stack(
                [self._spectral_loss(y_pred[..., c], y_true[..., c]) for c in range(C)]
            )

        # Only compute for series with signal above threshold
        has_signal = (
            (torch.abs(y_true) > self.non_zero_threshold) | 
            (torch.abs(y_pred.detach()) > self.non_zero_threshold)
        ).any(dim=1)
        
        if not has_signal.any():
            return y_pred.new_tensor(0.0)
            
        pred = y_pred[has_signal]
        true = y_true[has_signal]
        T = pred.size(1)
        
        total = pred.new_tensor(0.0)
        n_valid = 0

        for n_fft, hop in self._SPECTRAL_RESOLUTIONS:
            if T < n_fft:
                continue
                
            window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            S_pred = torch.stft(pred, n_fft, hop_length=hop, win_length=n_fft,
                                window=window, center=False, return_complex=True)
            S_true = torch.stft(true, n_fft, hop_length=hop, win_length=n_fft,
                                window=window, center=False, return_complex=True)
                                
            # Safe magnitude and DC masking
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

        # 1. DC/AC Decomposition
        W = max(6, T // 3)
        windows = list(e.split(W, dim=1))
        e_shape = torch.cat([w - w.mean(dim=1, keepdim=True) for w in windows], dim=1)

        # 2. Component Losses
        loss_shape_pc = self._shape_loss(e_shape, y_true, y_pred)
        loss_level_pc = self._windowed_level_loss(e, T)
        
        loss_spec_pc = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec_pc = self._spectral_loss(y_pred, y_true)

        # 3. Aggregation and Balancing
        if loss_shape_pc.dim() == 0:
            # Univariate
            total_loss = loss_shape_pc + loss_level_pc + loss_spec_pc
            self._last_components = {
                "shape": [float(loss_shape_pc.detach())],
                "level": [float(loss_level_pc.detach())],
                "spec": [float(loss_spec_pc.detach())],
                "ema": [float("nan")],
                "weight": [1.0],
            }
        else:
            # Multivariate
            per_channel_total = loss_shape_pc + loss_level_pc + loss_spec_pc
            total_loss = self._combine_channels(per_channel_total)
            
            C = per_channel_total.shape[0]
            spec_list = (
                loss_spec_pc.detach().tolist() if loss_spec_pc.dim() 
                else [float(loss_spec_pc)] * C
            )
            weights = self._last_weights or [1.0] * C
            
            self._last_components = {
                "shape": loss_shape_pc.detach().tolist(),
                "level": loss_level_pc.detach().tolist(),
                "spec": spec_list,
                "ema": list(self._loss_ema) if self._loss_ema else [float("nan")] * C,
                "weight": weights,
                "contribution": [
                    weights[c] * float(per_channel_total.detach()[c]) for c in range(C)
                ],
            }

        if torch.isnan(total_loss):
            raise RuntimeError(f"NaN in SpotlightLossLogcosh: total={total_loss.item():.6f}")

        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"