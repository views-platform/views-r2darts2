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

        # State for cross-channel balancing (inverse-EMA of target RMS scale)
        self._target_ema: list[float] | None = None

        # State for raw-space running calibration EMAs
        self._cal_pred_ema: list[float] | None = None
        self._cal_true_ema: list[float] | None = None

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
        """Combines per-channel losses with inverse-EMA target RMS scale normalisation,

        augmented with an automated/curriculum gradient budget routing system.

        It monitors the raw-space calibration of each target by de-transforming the 
        predictions (torch.sinh in asinh space) and computing a running EMA of raw
        prediction over raw truth.

        It implements a curriculum sequential priority:
            Ch0 (sb) -> Ch1 (ns) -> Ch2 (os)
        Gating factors for lower-priority channels contract or expand dynamically
        based on the active raw-space calibration quality of higher-priority channels.
        If optimizing a sparse target (e.g. non-state) creates negative transfer that
        harms state-based calibration, the Ch0 calibration score plummets, causing
        the Ch1 and Ch2 curriculum gates to shut instantly. This redirects 100% of
        the gradient budget back to reconstructing Ch0 until calibration is recovered.
        """
        C = per_channel_loss.shape[0]
        # Calculate target physical scale: root-mean-square (RMS) of target series
        # y_true has shape (B, T, C)
        target_rms = torch.sqrt(torch.mean(y_true ** 2, dim=(0, 1)) + self._EMA_EPS)
        rms_det = target_rms.detach()
        rms_det = torch.nan_to_num(rms_det, nan=1.0, posinf=1.0, neginf=self._EMA_EPS)
        rms_det = rms_det.clamp(min=self._EMA_EPS)

        if self._target_ema is None or len(self._target_ema) != C:
            self._target_ema = rms_det.tolist()
        else:
            beta = self._EMA_BETA
            for c in range(C):
                self._target_ema[c] = beta * self._target_ema[c] + (1.0 - beta) * float(
                    rms_det[c]
                )

        # ── Raw-Space Calibration & Dynamic Gated Curriculum ──────
        # De-transform predictions and targets to raw space (asinh -> raw via sinh)
        raw_pred = torch.sinh(y_pred.detach())
        raw_true = torch.sinh(y_true)

        # Epoch-level simulation via running batch EMAs
        batch_pred_mean = torch.mean(raw_pred, dim=(0, 1))
        batch_true_mean = torch.mean(raw_true, dim=(0, 1))

        if self._cal_pred_ema is None or len(self._cal_pred_ema) != C:
            self._cal_pred_ema = batch_pred_mean.tolist()
            self._cal_true_ema = batch_true_mean.tolist()
        else:
            beta_cal = self._EMA_BETA
            for c in range(C):
                self._cal_pred_ema[c] = beta_cal * self._cal_pred_ema[c] + (1.0 - beta_cal) * float(batch_pred_mean[c])
                self._cal_true_ema[c] = beta_cal * self._cal_true_ema[c] + (1.0 - beta_cal) * float(batch_true_mean[c])

        # Compute calibration ratio theta = (pred + eps) / (true + eps)
        # We add 1e-4 raw fatalities to protect the denominator during clean/peace batches
        theta = []
        cal = []
        for c in range(C):
            p_val = max(0.0, self._cal_pred_ema[c])
            t_val = max(0.0, self._cal_true_ema[c])
            theta_c = (p_val + 1e-4) / (t_val + 1e-4)
            theta.append(theta_c)
            # Symmetric score cal_c ∈ (0, 1] representing distance from perfect calibration (1.0)
            log_ratio = math.log(max(1e-6, theta_c))
            cal_c = math.exp(-abs(log_ratio))
            cal.append(cal_c)

        # Sequentially cascade curriculum: current channel only unlocks if all previous channels are calibrated
        # G_0 = 1.0 (primary sb channel always trained directly)
        # G_c = product_{k=0..c-1} cal_k^2
        gates_list = [1.0]
        curr_prod = 1.0
        for c in range(1, C):
            curr_prod = curr_prod * (cal[c - 1] ** 2)
            gates_list.append(curr_prod)

        # Base AGC scale normalizing weights
        ema = per_channel_loss.new_tensor(self._target_ema)
        w_agc = 1.0 / (ema + self._EMA_EPS)

        # Combine AGC and Curriculum gates
        gates_tensor = per_channel_loss.new_tensor(gates_list)
        w_curr = w_agc * gates_tensor

        # Normalize back so mean(w) == 1.0, preserving loss magnitude/gradients
        w_final = (w_curr / w_curr.mean().clamp(min=1e-8)).detach()
        self._last_weights = w_final.tolist()

        # Telemetry storage
        self._last_cal_ratio = theta
        self._last_cal_score = cal
        self._last_gates = gates_list

        return (w_final * per_channel_loss).sum()

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
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
        w_total = event_mag * w_dro
        
        # FIX: Apply nan_to_num BEFORE computing loss_shape to prevent NaN propagation
        # Also perform per-channel global batch mean normalization for 3D
        if w_total.dim() == 3:
            w_total = w_total / w_total.mean(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
            loss_shape = (w_total * cell_loss).mean(dim=(0, 1))  # (C,)
        else:
            w_total = w_total / w_total.mean()
            w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
            loss_shape = (w_total * cell_loss).mean()  # scalar

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
            total_loss = self._combine_channels(per_channel_total, y_pred, y_true)
            
            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "spec": spec_list,
                "weight": weights,
                "ema": self._target_ema or [float("nan")] * C,
                "cal_ratio": getattr(self, "_last_cal_ratio", [1.0] * C),
                "cal_score": getattr(self, "_last_cal_score", [1.0] * C),
                "gates": getattr(self, "_last_gates", [1.0] * C),
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