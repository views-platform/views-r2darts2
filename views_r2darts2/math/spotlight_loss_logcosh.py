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

        # State for cross-channel balancing (inverse-EMA)
        self._loss_ema: list[float] | None = None

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
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
    ) -> torch.Tensor:
        """Event-magnitude-weighted windowed level anchor.

        Splits the T-length error into non-overlapping windows, computes
        log_cosh on per-window means, then weights each series by its 
        event magnitude. No DRO, no CMW, just the sigmoid weight.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )  # (B, n_windows) or (B, n_windows, C)
        level_losses = self._log_cosh(window_means)

        # Per-series event magnitude: max |y_true| across time → sigmoid
        series_mag = y_true.abs().max(dim=1).values  # (B,) or (B, C)
        series_w = 0.01 + 0.99 * torch.sigmoid(
            5.0 * (series_mag - self.non_zero_threshold)
        )  # (B,) or (B, C)
        
        # FIX: Normalize per channel if 3D, else global
        if series_w.dim() == 2:
            series_w = series_w / series_w.mean(dim=0, keepdim=True).clamp(min=1e-8)
        else:
            series_w = series_w / series_w.mean().clamp(min=1e-8)

        # Weight each series' level loss
        if level_losses.dim() == 3:
            weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows, C)
            return W * weighted.mean(dim=(0, 1))  # (C,)
        else:
            weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows)
            return W * weighted.mean()  # scalar

    def _spectral_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only).

        Safe magnitude sqrt(re² + im² + ε) avoids gradient blowup at
        |z|→0.  DC bin is masked — level anchor already handles DC.
        Only series with signal above threshold are included.
        """
        if y_pred.dim() == 3:
            B, T, C = y_pred.shape
            pred = y_pred.permute(0, 2, 1).reshape(B * C, T)
            true = y_true.permute(0, 2, 1).reshape(B * C, T)
        else:
            pred = y_pred
            true = y_true

        has_signal = (
            (torch.abs(true) > self.non_zero_threshold)
            | (torch.abs(pred.detach()) > self.non_zero_threshold)
        ).any(dim=1)
        if not has_signal.any():
            # Return shape matching expected output
            if y_pred.dim() == 3:
                return pred.new_tensor(0.0).repeat(y_pred.shape[-1])
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

        spec_loss = total / max(n_valid, 1)
        
        # If 3D input, we processed B*C series. We need to return per-channel (C,)
        if y_pred.dim() == 3:
            # Because we flattened B*C, the mean is already across all channels.
            # To maintain channel isolation, we should technically loop, but 
            # flattening is mathematically equivalent to averaging across channels.
            # To keep it strictly per-channel:
            # (Reverting to loop to guarantee channel isolation)
            return torch.stack(
                [self._spectral_loss(y_pred[..., c], y_true[..., c]) for c in range(C)]
            )
            
        return spec_loss

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
        w_total = event_mag * w_dro
        
        # FIX: Per-channel global batch mean normalization for 3D
        if w_total.dim() == 3:
            w_total = w_total / w_total.mean(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            loss_shape = (w_total * cell_loss).mean(dim=(0, 1))  # (C,)
        else:
            w_total = w_total / w_total.mean()
            loss_shape = (w_total * cell_loss).mean()  # scalar

        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, y_true, T)

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
            total_loss = self._combine_channels(per_channel_total)
            
            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "spec": spec_list,
                "weight": weights,
                "contribution": [
                    weights[c] * float(per_channel_total.detach()[c]) for c in range(C)
                ],
            }

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} "
                f"spec={loss_spec.item() if loss_spec.dim()==0 else loss_spec.sum().item():.6f}"
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