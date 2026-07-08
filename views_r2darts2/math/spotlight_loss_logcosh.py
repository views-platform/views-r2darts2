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

    2. **Parameter-Free Hurdle Event Weighting.**
       The event weight is derived directly from the pseudo-hurdle occurrence
       probability. True events get weight 1.0. True peace gets weight equal
       to the model's predicted event probability (p_event). This self-corrects
       hallucinations without any hardcoded gates, slopes, or gamma constants.

     3. **Pseudo-hurdle occurrence term** — zero vs non-zero supervision
         from the same scalar output. The model's asinh prediction is converted
         into an occurrence logit relative to ``non_zero_threshold`` and trained
         with positive-class-reweighted BCE. This gives explicit event-detection
         signal without adding a second head.

     4. **Per-series temporal DRO** — within-series shock therapy.
         Power-law reweighting (alpha in (0,1)) upweights harder timesteps
         *relative to that series* while remaining sublinear.

     5. **Windowed level anchor** — log_cosh on per-window means.
         Uses the exact same window partition as shape for strict orthogonality.

     6. **Multi-resolution STFT loss** — always on, ungated.
       log_cosh on magnitude-spectrum differences.  DC bin masked.

    ── Base cell loss: log_cosh × (1 + log(1+|x|³))  (proportional) ───

    Args:
        non_zero_threshold: Sigmoid center (AsinhTransform: 0.88 ≈ asinh(1))
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = False
    _EMA_BETA = 0.99
    _EMA_EPS = 1e-6

    # Adaptive DRO alpha bounds.
    _DRO_ALPHA_MIN = 0.30
    _DRO_ALPHA_MAX = 0.80

    # Shared windowing for strict shape-level orthogonality.
    _WINDOW_DIVISOR = 3
    _MIN_WINDOW = 6
    _LEVEL_SCALE = 1.0

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

        logger.info("SpotlightLossLogcosh | threshold=%.4f", non_zero_threshold)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """log(cosh(x)), numerically stable: |x| + softplus(−2|x|) − ln2."""
        abs_x = torch.abs(x)
        return abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

    def _dro_weights_2d(
        self, losses: torch.Tensor, y_true: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Per-series power-law self-reweighting."""
        del y_true
        l = losses.detach()
        mu = l.mean(dim=1, keepdim=True).clamp(min=1e-6)
        w = torch.pow(l / mu, alpha)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _window_size(self, T: int) -> int:
        """Shared non-overlapping window size for shape and level terms."""
        return max(self._MIN_WINDOW, T // self._WINDOW_DIVISOR)

    def _occurrence_hurdle_terms(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return occurrence loss and a soft magnitude mask.
        
        The mask is `max(z_event, p_event.detach())`. This acts as a 
        parameter-free event weight: true events get 1.0, true peace gets
        the model's predicted probability (hallucinations are penalized).
        """
        z_event = (y_true > self.non_zero_threshold).to(dtype=y_pred.dtype)
        occ_logit = y_pred - self.non_zero_threshold
        p_event = torch.sigmoid(occ_logit)

        if z_event.dim() == 3:
            event_rate = z_event.mean(dim=(0, 1)).detach().clamp(
                min=self._EMA_EPS, max=1.0 - self._EMA_EPS
            )
            pos_weight = torch.sqrt((1.0 - event_rate) / event_rate).view(1, 1, -1)
            loss_occ_raw = F.binary_cross_entropy_with_logits(
                occ_logit, z_event, reduction="none"
            )
            loss_occ_weighted = torch.where(z_event > 0.0, pos_weight * loss_occ_raw, loss_occ_raw)
            loss_occ = loss_occ_weighted.mean(dim=(0, 1))
        else:
            event_rate = z_event.mean().detach().clamp(
                min=self._EMA_EPS, max=1.0 - self._EMA_EPS
            )
            pos_weight = torch.sqrt((1.0 - event_rate) / event_rate)
            loss_occ_raw = F.binary_cross_entropy_with_logits(
                occ_logit, z_event, reduction="none"
            )
            loss_occ_weighted = torch.where(z_event > 0.0, pos_weight * loss_occ_raw, loss_occ_raw)
            loss_occ = loss_occ_weighted.mean()

        mag_mask = torch.maximum(z_event, p_event.detach())
        return loss_occ, mag_mask

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def _combine_channels(self, per_channel_loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Combine per-channel losses by *relative learning progress*."""
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
        """Event-mask-weighted level anchor on shared shape windows."""
        W = self._window_size(T)

        true_series_event = (y_true > self.non_zero_threshold).any(dim=1).to(dtype=y_true.dtype)
        if y_pred_det is not None:
            pred_series_event = torch.sigmoid(
                y_pred_det.max(dim=1).values - self.non_zero_threshold
            )
        else:
            pred_series_event = true_series_event
        series_mask = torch.maximum(true_series_event, pred_series_event.detach())
        series_w = series_mask

        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )
        level_losses = self._log_cosh(window_means)
        n_windows = level_losses.shape[1]

        if level_losses.dim() == 3:
            num = (series_w.unsqueeze(1) * level_losses).sum(dim=(0, 1))
            den = (series_w.sum(dim=0) * n_windows).clamp(min=self._EMA_EPS)
        else:
            num = (series_w.unsqueeze(1) * level_losses).sum()
            den = (series_w.sum() * n_windows).clamp(min=self._EMA_EPS)

        level = num / den
        return T * self._LEVEL_SCALE * level

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

        W = self._window_size(T)
        windows = list(e.split(W, dim=1))
        e_shape = torch.cat(
            [w - w.mean(dim=1, keepdim=True) for w in windows], dim=1
        )

        cell_loss = self._log_cosh(e_shape)

        # ── Parameter-Free Hurdle Event Weighting ─────────────────────
        loss_occ, mag_mask = self._occurrence_hurdle_terms(y_pred, y_true)

        # ── Adaptive DRO alpha ─────────────────────────────────────────
        f_event = (torch.abs(y_true).detach() > self.non_zero_threshold).float().mean().item()
        dro_alpha = self._DRO_ALPHA_MIN + (self._DRO_ALPHA_MAX - self._DRO_ALPHA_MIN) * (1.0 - f_event)

        # ── Per-series temporal DRO ────────────────────────────────────
        w_dro = self._dro_weights_2d(cell_loss, y_true, dro_alpha)
        w_total = torch.nan_to_num(
            mag_mask * w_dro, nan=1.0, posinf=1.0, neginf=0.0
        )

        # ── Hájek self-normalized shape ───────────────────────────────
        if w_total.dim() == 3:
            num = (w_total * cell_loss).sum(dim=(0, 1))
            den = w_total.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)
            loss_shape = num / den
        else:
            num = (w_total * cell_loss).sum()
            den = w_total.sum().clamp(min=self._EMA_EPS)
            loss_shape = num / den

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, y_true, T, y_pred_det=y_pred.detach())

        occ_scale = loss_shape.detach() / loss_occ.detach().clamp(min=self._EMA_EPS)
        loss_occ_scaled = occ_scale * loss_occ

        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        # ── Core objective assembly & telemetry ────────────────────
        if loss_shape.dim() == 0:
            total_loss = loss_shape + loss_level + loss_spec + loss_occ_scaled
            self._last_components = {
                "shape": [float(loss_shape.detach())],
                "level": [float(loss_level.detach())],
                "occ": [float(loss_occ_scaled.detach())],
                "spec": [float(loss_spec.detach()) if loss_spec.dim()==0 else float(loss_spec)],
                "weight": [1.0],
            }
        else:
            per_channel_total = loss_shape + loss_level + loss_spec + loss_occ_scaled
            total_loss = self._combine_channels(per_channel_total, y_pred, y_true)
            
            C = per_channel_total.shape[0]
            spec_list = loss_spec.detach().tolist() if loss_spec.dim() else [float(loss_spec)] * C
            weights = self._last_weights or [1.0] * C
            self._last_components = {
                "shape": loss_shape.detach().tolist(),
                "level": loss_level.detach().tolist(),
                "occ": loss_occ_scaled.detach().tolist(),
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