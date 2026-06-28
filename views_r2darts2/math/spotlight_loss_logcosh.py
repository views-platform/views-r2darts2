import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLossLogcosh(torch.nn.Module):
    """
    SpotlightLoss v48 — asinh + RevIN compatible, EMA-stabilized shape loss,
    with a Dynamic Hallucination Floor.
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

        self._loss_ema: list[float] | None = None
        self._shape_ema: dict | None = None

        # EMA state for the dynamic hallucination floor.
        # Tracks the model's average prediction magnitude when y_true == 0.
        self._peace_pred_ema: float = 0.0

        self._last_components: dict | None = None
        self._last_weights: list[float] | None = None
        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _get_dynamic_floor(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculates a dynamic floor based on model hallucinations during peace."""
        # Find where true target is exactly zero (peace)
        is_peace = y_true == 0.0

        if is_peace.any():
            # What is the model predicting during these peaceful moments?
            peace_preds = y_pred.detach()[is_peace]
            curr_peace_mag = peace_preds.abs().mean().item()
        else:
            curr_peace_mag = 0.0

        # Update EMA of peaceful predictions
        beta = self._EMA_BETA
        self._peace_pred_ema = (
            beta * self._peace_pred_ema + (1.0 - beta) * curr_peace_mag
        )

        # Scale the hallucination to a floor between 0.01 and 0.5
        # If model predicts 0.0 during peace -> floor is 0.01 (1%)
        # If model predicts 0.5 during peace -> floor is 0.25 (25%)
        # If model predicts 1.0+ during peace -> floor caps at 0.5 (50%)
        floor = min(0.5, self._peace_pred_ema * 0.5)
        return max(0.01, floor)

    def _shape_loss(
        self,
        e_shape: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        cell_loss = self._log_cosh(e_shape)

        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        cmw = torch.log1p(abs_max) / (torch.log1p(abs_max) + 1.0)

        # DYNAMIC FLOOR: Replaces the hardcoded 0.1
        floor = self._get_dynamic_floor(y_true, y_pred)
        event_mag = floor + (1.0 - floor) * cmw

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

            mu_loss = cell_loss.new_tensor(self._shape_ema["loss"])

            w_dro = torch.sqrt(cell_loss.detach() / mu_loss.clamp(min=1e-6))
            w_dro = w_dro.clamp(max=10.0)
            w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=10.0, neginf=0.0)

            w_total = event_mag * w_dro
            w_total = w_total / w_total.mean(dim=1, keepdim=True).clamp(min=1e-8)
            w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)

            return (w_total * cell_loss).mean(dim=(0, 1))  # (C,)

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

        mu_loss = cell_loss.new_tensor(self._shape_ema["loss"])

        w_dro = torch.sqrt(cell_loss.detach() / mu_loss.clamp(min=1e-6))
        w_dro = w_dro.clamp(max=10.0)
        w_dro = torch.nan_to_num(w_dro, nan=1.0, posinf=10.0, neginf=0.0)

        w_total = event_mag * w_dro
        w_total = w_total / w_total.mean(dim=1, keepdim=True).clamp(min=1e-8)
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)

        return (w_total * cell_loss).mean()

    def _combine_channels(self, per_channel_loss: torch.Tensor) -> torch.Tensor:
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
        w = (w / w.mean()).detach()
        self._last_weights = w.tolist()
        return (w * per_channel_loss).sum()

    def _windowed_level_loss(
        self,
        e: torch.Tensor,
        y_true: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        W = max(6, T // 3)
        window_means = torch.stack([ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1)
        level_losses = self._log_cosh(window_means)

        true_window_means = torch.stack(
            [tw.mean(dim=1) for tw in y_true.split(W, dim=1)], dim=1
        )

        cmw = torch.log1p(torch.abs(true_window_means)) / (
            torch.log1p(torch.abs(true_window_means)) + 1.0
        )

        # DYNAMIC FLOOR: Reuse the hallucination EMA for the level loss as well
        floor = max(0.01, min(0.5, self._peace_pred_ema * 0.5))
        mag = floor + (1.0 - floor) * cmw

        if level_losses.dim() == 3:
            mag = mag / mag.mean(dim=(0, 1), keepdim=True).clamp(min=1e-6)
            mag = torch.nan_to_num(mag, nan=1.0, posinf=1.0, neginf=0.0)
            return (mag * level_losses).mean(dim=(0, 1))

        mag = mag / mag.mean().clamp(min=1e-6)
        mag = torch.nan_to_num(mag, nan=1.0, posinf=1.0, neginf=0.0)
        return (mag * level_losses).mean()

    def _spectral_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Multi-resolution STFT magnitude comparison (AC bins only).

        Safe magnitude sqrt(re² + im² + ε) avoids gradient blowup at
        |z|→0.  DC bin is masked — level anchor already handles DC.
        Only series with signal above threshold are included.
        """
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
                pred,
                n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=False,
                return_complex=True,
            )
            S_true = torch.stft(
                true,
                n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=False,
                return_complex=True,
            )
            # Safe magnitude — bounded gradient at |z|→0
            mag_pred = torch.sqrt(S_pred.real**2 + S_pred.imag**2 + 1e-8)
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
        loss_shape_pc = self._shape_loss(e_shape, y_true, y_pred)  # scalar | (C,)
        loss_level_pc = self._windowed_level_loss(e, y_true, T)  # scalar | (C,)

        # ── Multi-resolution spectral loss (always on) ──────────────
        # scalar (single target) | (C,) per-channel (multivariate)
        loss_spec_pc = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec_pc = self._spectral_loss(y_pred, y_true)

        # ── Core objective ────────────────────────────────────────────
        # Univariate: original sum (unchanged).  Multivariate: combine the
        # per-channel shape+level+spectral objectives with inverse-EMA scale
        # normalisation so the sb channel's magnitude cannot dominate ns/os.
        # The spectral term is balanced through the *same* combine as
        # shape+level so no part of the objective bypasses the balancing.
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
                "ema": [float("nan")],  # no cross-channel balance here
                "weight": [1.0],
            }
        else:
            per_channel_total = loss_shape_pc + loss_level_pc + loss_spec_pc
            total_loss = self._combine_channels(per_channel_total)
            loss_shape = loss_shape_pc.sum().detach()  # logging only
            loss_level = loss_level_pc.sum().detach()  # logging only
            loss_spec = (
                loss_spec_pc.sum().detach() if loss_spec_pc.dim() else loss_spec_pc
            )
            # ── Telemetry (multivariate): per-channel term split + the ──
            # inverse-EMA balance.  spec may be a shared scalar if STFT is
            # off (T<6); broadcast it across channels for a uniform schema.
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
                # weighted contribution of each channel to total_loss
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
            "SpotlightLossLogcosh | shape=%.6f level=%.6f " "spec=%.6f total=%.6f",
            loss_shape.item(),
            loss_level.item(),
            loss_spec.item(),
            total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"
