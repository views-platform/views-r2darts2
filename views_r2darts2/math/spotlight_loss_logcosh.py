import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v48 — A multi-component loss function for highly imbalanced,
    heavy-tailed time series forecasting (e.g., conflict fatalities).

    Designed to operate in `asinh` (arc-hyperbolic sine) space alongside
    Reversible Instance Normalization (RevIN). It addresses datasets with
    ~90% zeros and rare events spanning four orders of magnitude.

    ── Workflow & Components ───────────────────────────────────────────

    1. **Per-Window DC/AC Decomposition**:
       The error tensor `e = y_pred - y_true` is split into non-overlapping
       windows of length `W`. The shape (AC) and level (DC) components are
       separated by demeaning each window. This ensures orthogonality:
       shape handles within-window patterns, level handles per-window means.

    2. **Shape Loss (Temporal EMA-DRO)**:
       Uses `log_cosh` as the base loss. Applies Continuous Magnitude
       Weighting (CMW) with a Dynamic Hallucination Floor to prioritize
       high-magnitude events while ensuring peaceful periods receive a
       baseline gradient to prevent drift. Applies a per-series temporal
       Distributionally Robust Optimization (DRO) spotlight, normalized by
       a slow Exponential Moving Average (EMA) of the loss, to focus on
       poorly performing timesteps relative to the model's own history.

    3. **Windowed Level Loss**:
       Computes `log_cosh` on per-window means. Scaled by total sequence
       length `T` to heavily enforce baseline matching. Applies CMW based
       on the true target magnitude to prevent the model from collapsing
       to flat zero predictions (variance collapse).

    4. **Multi-Resolution Spectral Loss**:
       Computes a multi-resolution Short-Time Fourier Transform (STFT)
       magnitude loss. The DC bin is masked as it is handled by the level
       loss. Only series with signal above a threshold are included.

    5. **Per-Channel Multi-Task Balancing**:
       For multivariate targets, each channel's shape+level+spectral loss
       is normalized by an inverse-EMA scale to ensure high-variance
       channels do not dominate the joint objective.

    ── Citations ───────────────────────────────────────────────────────

    * **Multi-Resolution STFT Loss**: Yamamoto, R., et al. (2020).
      "Parallel WaveGAN: A fast waveform generation model based on
      generative adversarial networks with multi-resolution spectrogram."
    * **Distributionally Robust Optimization (DRO)**: Sagawa, S., et al.
      (2020). "Distributionally Robust Neural Networks."
    * **Reversible Instance Normalization (RevIN)**: Kim, T., et al. (2021).
      "Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift."
    * **Variance-Stabilizing Transform**: Freeman, M. F., & Tukey, J. W.
      (1950). "Transformations related to the angular and the square root."

    Args:
        non_zero_threshold (float): Sigmoid center and event mask threshold.
            In `asinh` space, 0.88 ≈ asinh(1), separating 0 fatalities
            from actual conflict.
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = True
    _EMA_BETA = 0.99
    _EMA_EPS = 1e-6

    def __init__(self, non_zero_threshold: float):
        """Initializes the loss module and state trackers."""
        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"non_zero_threshold must be positive, got {non_zero_threshold}"
            )

        super().__init__()
        self.non_zero_threshold = non_zero_threshold

        # State for cross-channel balancing
        self._loss_ema: list[float] | None = None
        
        # State for shape loss EMA-DRO
        self._shape_ema: dict | None = None
        
        # State for dynamic hallucination floor (channel-aware)
        self._peace_pred_ema: list[float] | float = 0.0
        
        # Telemetry for callbacks
        self._last_components: dict | None = None
        self._last_weights: list[float] | None = None
        
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static Math Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable log(cosh(x)): |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    @staticmethod
    def _compute_cmw(x: torch.Tensor) -> torch.Tensor:
        """Continuous Magnitude Weighting (CMW)."""
        abs_x = torch.abs(x)
        return torch.log1p(abs_x) / (torch.log1p(abs_x) + 1.0)

    @staticmethod
    def _sanitize_tensor(
        w: torch.Tensor, posinf_val: float = 1.0, max_val: float | None = None
    ) -> torch.Tensor:
        """Clamps, removes NaNs/Infs, and sanitizes weight tensors."""
        if max_val is not None:
            w = w.clamp(max=max_val)
        return torch.nan_to_num(w, nan=1.0, posinf=posinf_val, neginf=0.0)

    @staticmethod
    def _windowed_mean(x: torch.Tensor, W: int) -> torch.Tensor:
        """Splits tensor along time (dim=1) into windows of length W and means them."""
        return torch.stack([w.mean(dim=1) for w in x.split(W, dim=1)], dim=1)

    # ------------------------------------------------------------------
    # Stateful Helpers
    # ------------------------------------------------------------------

    def _get_dynamic_floor(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculates a dynamic floor based on model hallucinations during peace.
        
        Tracks the model's average prediction magnitude per-channel when `y_true` 
        is below the `non_zero_threshold`.
        """
        is_peace = (y_true.abs() < self.non_zero_threshold)
        beta = self._EMA_BETA
        
        if y_pred.dim() == 3:
            C = y_pred.shape[-1]
            if not isinstance(self._peace_pred_ema, list) or len(self._peace_pred_ema) != C:
                self._peace_pred_ema = [0.0] * C
                
            floors = []
            for c in range(C):
                if is_peace[..., c].any():
                    curr_mag = y_pred.detach()[..., c][is_peace[..., c]].abs().mean().item()
                else:
                    curr_mag = 0.0
                self._peace_pred_ema[c] = beta * self._peace_pred_ema[c] + (1.0 - beta) * curr_mag
                floors.append(max(0.1, min(0.5, self._peace_pred_ema[c] * 0.5)))
            return y_pred.new_tensor(floors) # (C,)
        else:
            if isinstance(self._peace_pred_ema, list):
                self._peace_pred_ema = 0.0
                
            if is_peace.any():
                curr_mag = y_pred.detach()[is_peace].abs().mean().item()
            else:
                curr_mag = 0.0
            self._peace_pred_ema = beta * self._peace_pred_ema + (1.0 - beta) * curr_mag
            return y_pred.new_tensor(max(0.1, min(0.5, self._peace_pred_ema * 0.5))) # scalar

    def _update_and_get_shape_ema(self, cell_loss: torch.Tensor) -> torch.Tensor:
        """Updates and retrieves the EMA of the shape loss for DRO normalization."""
        beta = self._EMA_BETA
        if cell_loss.dim() == 3:
            C = cell_loss.shape[-1]
            curr_loss_mean = cell_loss.detach().mean(dim=(0, 1)).clamp(min=1e-6)

            if self._shape_ema is None or len(self._shape_ema.get("loss", [])) != C:
                self._shape_ema = {"loss": curr_loss_mean.tolist()}
            else:
                for c in range(C):
                    self._shape_ema["loss"][c] = (
                        beta * self._shape_ema["loss"][c]
                        + (1.0 - beta) * curr_loss_mean[c].item()
                    )
            return cell_loss.new_tensor(self._shape_ema["loss"])
        
        # Univariate path
        curr_loss_mean = cell_loss.detach().mean().clamp(min=1e-6)
        if (
            self._shape_ema is None
            or "loss" not in self._shape_ema
            or isinstance(self._shape_ema["loss"], list)
        ):
            self._shape_ema = {"loss": curr_loss_mean.item()}
        else:
            self._shape_ema["loss"] = (
                beta * self._shape_ema["loss"] + (1.0 - beta) * curr_loss_mean.item()
            )
        return cell_loss.new_tensor(self._shape_ema["loss"])

    def _compute_dro_weights(self, cell_loss: torch.Tensor, mu_loss: torch.Tensor) -> torch.Tensor:
        """Calculates temporal DRO weights using sqrt self-reweighting."""
        w_dro = torch.sqrt(cell_loss.detach() / mu_loss.clamp(min=1e-6))
        return self._sanitize_tensor(w_dro, posinf_val=10.0, max_val=10.0)

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def _shape_loss(
        self, e_shape: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, floor: torch.Tensor
    ) -> torch.Tensor:
        """Computes the event-magnitude- and EMA-weighted shape loss."""
        cell_loss = self._log_cosh(e_shape)
        
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        cmw = self._compute_cmw(abs_max)
        
        # Broadcast floor for multivariate path
        if floor.dim() == 1:
            floor = floor.view(1, 1, -1)
            
        event_mag = floor + (1.0 - floor) * cmw

        mu_loss = self._update_and_get_shape_ema(cell_loss)
        w_dro = self._compute_dro_weights(cell_loss, mu_loss)

        w_total = event_mag * w_dro
        w_total = w_total / w_total.mean(dim=1, keepdim=True).clamp(min=1e-8)
        w_total = self._sanitize_tensor(w_total, posinf_val=1.0)

        if cell_loss.dim() == 3:
            return (w_total * cell_loss).mean(dim=(0, 1))  # (C,)
        return (w_total * cell_loss).mean()  # scalar

    def _combine_channels(self, per_channel_loss: torch.Tensor) -> torch.Tensor:
        """Combines per-channel losses with inverse-EMA scale normalisation."""
        C = per_channel_loss.shape[0]
        losses_det = per_channel_loss.detach()

        if self._loss_ema is None or len(self._loss_ema) != C:
            self._loss_ema = losses_det.clamp(min=self._EMA_EPS).tolist()
        else:
            beta = self._EMA_BETA
            for c in range(C):
                self._loss_ema[c] = beta * self._loss_ema[c] + (1.0 - beta) * float(
                    losses_det[c]
                )

        ema = per_channel_loss.new_tensor(self._loss_ema)
        w = 1.0 / (ema + self._EMA_EPS)
        w = (w / w.mean()).detach()  # mean 1 → preserve loss scale
        self._last_weights = w.tolist()
        return (w * per_channel_loss).sum()

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int, floor: torch.Tensor
    ) -> torch.Tensor:
        """Computes the windowed level anchor scaled by target magnitude and T."""
        W = max(6, T // 3)
        window_means = self._windowed_mean(e, W)
        level_losses = self._log_cosh(window_means)

        true_window_means = self._windowed_mean(y_true, W)
        cmw = self._compute_cmw(true_window_means)

        # Broadcast floor for multivariate path
        if floor.dim() == 1:
            floor = floor.view(1, 1, -1)
            
        mag = floor + (1.0 - floor) * cmw

        if level_losses.dim() == 3:
            mag = mag / mag.mean(dim=(0, 1), keepdim=True).clamp(min=1e-6)
            mag = self._sanitize_tensor(mag, posinf_val=1.0)
            # return T * (mag * level_losses).mean(dim=(0, 1))  # (C,)
            return (mag * level_losses).mean(dim=(0, 1))

        mag = mag / mag.mean().clamp(min=1e-6)
        mag = self._sanitize_tensor(mag, posinf_val=1.0)
        # return T * (mag * level_losses).mean()  # scalar
        return (mag * level_losses).mean()

    def _spectral_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only)."""
        if y_pred.dim() == 3:
            C = y_pred.shape[-1]
            return torch.stack(
                [self._spectral_loss(y_pred[..., c], y_true[..., c]) for c in range(C)]
            )  # (C,)

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
            
            mag_pred = torch.sqrt(S_pred.real**2 + S_pred.imag**2 + 1e-8)
            mag_true = S_true.abs()
            
            mag_pred = mag_pred.clone()
            mag_true = mag_true.clone()
            mag_pred[:, 0, :] = 0.0
            mag_true[:, 0, :] = 0.0
            
            total = total + self._log_cosh(mag_pred - mag_true).mean()
            n_valid += 1

        return total / max(n_valid, 1)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Executes the full multi-component loss calculation."""
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        T = y_pred.size(1)
        e = y_pred - y_true

        # ── 1. Per-window DC/AC decomposition ─────────────────────────
        W = max(6, T // 3)
        windows = list(e.split(W, dim=1))
        e_shape = torch.cat(
            [w - w.mean(dim=1, keepdim=True) for w in windows], dim=1
        )

        # Compute channel-aware dynamic floor ONCE here
        floor = self._get_dynamic_floor(y_true, y_pred)

        # ── 2. Compute component losses ───────────────────────────────
        loss_shape_pc = self._shape_loss(e_shape, y_true, y_pred, floor)
        loss_level_pc = self._windowed_level_loss(e, y_true, T, floor)
        
        loss_spec_pc = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec_pc = self._spectral_loss(y_pred, y_true)

        # ── 3. Core objective assembly & telemetry ────────────────────
        if loss_shape_pc.dim() == 0:
            # Univariate path
            loss_shape = loss_shape_pc
            loss_level = loss_level_pc
            loss_spec = loss_spec_pc
            total_loss = loss_shape + loss_level + loss_spec
            
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "spec": [float(loss_spec.detach())],
                "ema": [float("nan")],
                "weight": [1.0],
            }
        else:
            # Multivariate path
            per_channel_total = loss_shape_pc + loss_level_pc + loss_spec_pc
            total_loss = self._combine_channels(per_channel_total)
            
            loss_shape = loss_shape_pc.sum().detach()
            loss_level = loss_level_pc.sum().detach()
            loss_spec = (
                loss_spec_pc.sum().detach() if loss_spec_pc.dim() else loss_spec_pc
            )
            
            C = per_channel_total.shape[0]
            spec_list = (
                loss_spec_pc.detach().tolist()
                if loss_spec_pc.dim()
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
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f spec=%.6f total=%.6f",
            loss_shape.item(),
            loss_level.item(),
            loss_spec.item(),
            total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"