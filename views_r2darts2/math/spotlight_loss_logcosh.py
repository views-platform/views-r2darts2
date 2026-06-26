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

    Example:
        >>> loss_fn = SpotlightLossLogcosh(non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    _SPECTRAL_RESOLUTIONS = ((6, 3), (12, 6), (24, 12))
    _STFT = True

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

        # Learnable homoscedastic uncertainty weights (Kendall & Gal 2018).
        # total = Σ_i  0.5·exp(-s_i)·L_i + 0.5·s_i,   s_i = log σ_i².
        # The s_i are model parameters optimised by SGD over the whole
        # trajectory (darts registers the loss as a submodule, so these land
        # in self.parameters()). The balance therefore adapts to TRAINING
        # DYNAMICS — which term is reducible vs. irreducible noise — rather
        # than to any single sparse, unreliable batch's statistics. A term the
        # model can reduce shrinks its σ and is up-weighted; a noisy term
        # grows its σ and self-down-weights. The 0.5·s_i regularizer blocks the
        # trivial all-zero-weight solution; existing weight_decay supplies a
        # gentle pull toward equal weighting as a stabilising prior.
        # Init at 0 → all three terms start equally weighted.
        self.log_var_shape = torch.nn.Parameter(torch.zeros(()))
        self.log_var_level = torch.nn.Parameter(torch.zeros(()))
        self.log_var_spec = torch.nn.Parameter(torch.zeros(()))

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
    def _uncertainty_weight(loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Homoscedastic-uncertainty reweighting of one loss term.

        Returns 0.5·exp(-log_var)·loss + 0.5·log_var  (Kendall & Gal 2018,
        regression form).  The precision exp(-log_var) is the adaptive weight;
        the 0.5·log_var term regularises it away from the degenerate zero.
        """
        return 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var

    # @staticmethod
    # def _log_cosh_proportional(x: torch.Tensor) -> torch.Tensor:
    #     """log_cosh with proportional sensitivity correction.

    #     log_cosh(x) × (1 + log(1 + |x|³)).

    #     For |x| < 1: ≈ 0.5x² (cubic interior shrinks faster toward zero,
    #         so noise cells are quieter than the old x² variant).
    #     For |x| > 2: ≈ |x| × 3·ln|x| (~50% steeper than x² formula).

    #     Gradient = tanh(x)·(1 + log1p(|x|³))
    #                + log_cosh(x)·3x²·sign(x)/(1+|x|³).
    #     """
    #     abs_x = torch.abs(x)
    #     lc = abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)
    #     return lc * (1.0 + torch.log1p(abs_x * abs_x * abs_x))

    @staticmethod
    def _dro_weights_2d(losses: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Per-series sqrt self-reweighting

        w_it = sqrt(loss_it / mean_i(loss))

        Sublinear concentration: a cell 16× harder than average gets 4×
        the gradient (not 16×).  Redistributes enough signal to fix
        systematic bias while still focusing on spikes.

        Returns weights with mean ≈ 1 per series, shape (B, T).
        """
        l = losses.detach()                                  # (B, T)
        mu = l.mean(dim=1, keepdim=True).clamp(min=1e-6)     # (B, 1)
        w = torch.sqrt(l / mu)                               # (B, T)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)  # renormalize mean=1
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
    ) -> torch.Tensor:
        """Event-magnitude-weighted windowed level anchor.

        Splits the T-length error into non-overlapping windows, computes
        log_cosh_proportional on per-window means, then weights each
        series by its event magnitude.  Without this weighting, the 76%
        peace series (with near-zero DC error) dilute the level gradient
        that event series need to correct their systematic underprediction.

        Uses proportional loss to avoid gradient saturation: plain
        log_cosh has gradient tanh(x) → 1 for |x| > 2, meaning a 2×
        underprediction gets the same gradient as a 10× underprediction.
        """
        W = max(6, T // 3)
        window_means = torch.stack(
            [ew.mean(dim=1) for ew in e.split(W, dim=1)], dim=1
        )  # (B, n_windows)
        level_losses = self._log_cosh(window_means)

        # Per-series event magnitude: max |y_true| across time → sigmoid
        series_mag = y_true.abs().max(dim=1).values  # (B,)
        series_w = 0.01 + 0.99 * torch.sigmoid(
            5.0 * (series_mag - self.non_zero_threshold)
        )  # (B,)
        series_w = series_w / series_w.mean().clamp(min=1e-8)

        # Weight each series' level loss.  No explicit T scaling: inter-term
        # balance is now handled adaptively by the learnable uncertainty
        # weights in forward(), so a fixed T× multiplier would only bias the
        # learned σ_level and is redundant.
        weighted = series_w.unsqueeze(1) * level_losses  # (B, n_windows)
        return weighted.mean()

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
        # This makes shape and level orthogonal: shape handles within-window
        # patterns, level handles per-window DC.  No shared frequencies.
        W = max(6, T // 3)
        windows = list(e.split(W, dim=1))  # list of (B, W_i)
        e_shape = torch.cat(
            [w - w.mean(dim=1, keepdim=True) for w in windows], dim=1
        )  # (B, T) — zero-mean within each window

        # ── Base cell loss (proportional variant for MSLE sensitivity) ─
        cell_loss = self._log_cosh(e_shape)

        # ── Sigmoid event-magnitude weighting ─────────────────────────
        # Steep sigmoid: peace cells (abs_max ≈ 0) → ~0.01, moderate
        # events → ~0.5, high-conflict → ~1.0.  Gives ~50:1 contrast
        # ratio between Syria-class and zero cells.
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        event_mag = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.non_zero_threshold))

        # ── Per-series temporal DRO ────────────────────────────────────
        # Within each series, upweight the hardest timesteps relative to
        # that series' own loss distribution.  Between-series importance
        # is handled by event_mag above.
        w_dro = self._dro_weights_2d(cell_loss, y_true)  # (B, T)
        w_total = event_mag * w_dro
        w_total = w_total / w_total.mean()
        w_total = torch.nan_to_num(w_total, nan=1.0, posinf=1.0, neginf=0.0)
        loss_shape = (w_total * cell_loss).mean()

        # ── Windowed level anchor ─────────────────────────────────────
        loss_level = self._windowed_level_loss(e, y_true, T)

        # ── Multi-resolution spectral loss (always on) ──────────────
        loss_spec = y_pred.new_tensor(0.0)
        if self._STFT and T >= 6:
            loss_spec = self._spectral_loss(y_pred, y_true)

        # ── Adaptive inter-term balancing (learnable uncertainty weights) ──
        # Each term is rescaled by its own learned precision so none can
        # dominate by raw magnitude; the balance is driven by training
        # dynamics, not per-batch statistics.
        total_loss = (
            self._uncertainty_weight(loss_shape, self.log_var_shape)
            + self._uncertainty_weight(loss_level, self.log_var_level)
            + self._uncertainty_weight(loss_spec, self.log_var_spec)
        )

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN in SpotlightLossLogcosh: shape={loss_shape.item():.6f} "
                f"level={loss_level.item():.6f} "
                f"spec={loss_spec.item():.6f}"
            )

        logger.debug(
            "SpotlightLossLogcosh | shape=%.6f level=%.6f spec=%.6f "
            "| log_var(shape/level/spec)=%.3f/%.3f/%.3f total=%.6f",
            loss_shape.item(), loss_level.item(), loss_spec.item(),
            self.log_var_shape.item(), self.log_var_level.item(),
            self.log_var_spec.item(), total_loss.item(),
        )
        return total_loss

    def __repr__(self) -> str:
        return f"SpotlightLossLogcosh(non_zero_threshold={self.non_zero_threshold})"