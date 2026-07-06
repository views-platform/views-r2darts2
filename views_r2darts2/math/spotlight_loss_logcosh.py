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

     1. **DC/AC decomposition** — Haar local pairwise demeaning.
         Within each adjacent time pair (t, t+1), subtract the pair mean:
         the deviation is the Haar detail (local AC/shape), the pair mean is
         the Haar approximation (local DC/level). Strictly LOCAL — a spike at
         t only perturbs its own pair, so shape is not smeared across the
         horizon the way global demeaning smears it. Orthogonal to the pair
         mean by construction, preserving the AC/DC split.

    2. **Gated + magnitude-graded event weighting.**
    event_mag = gate × (1 + abs_max), gate = 0.0125 + 0.9875 × σ(10 × (abs_max − τ)).
    The gate suppresses peace (→ ~0.0125) vs conflict (→ ~1); the (1 + abs_max)
       factor — bounded because abs_max is in asinh space — restores magnitude
       sensitivity across the 4-OOM tail so large wars outweigh small skirmishes
       instead of saturating flat. No model-state dependency (abs_max detached).

    3. **Per-series temporal DRO** — within-series shock therapy.
       Z-scores log(cell_loss) along time axis per series.  Upweights
       proportionally harder timesteps *relative to that series*.

    4. **Multi-scale level anchor** — dyadic average-pool (Haar approximation)
       log_cosh matching. Strictly local, O(log T) gradient, DC-carrying;
       the orthogonal complement of the Haar shape detail (component 1).

    5. **Multi-resolution STFT loss** — always on, ungated.
       log_cosh on magnitude-spectrum differences.  DC bin masked.

    ── Base cell loss: log_cosh × (1 + log(1+|x|³))  (proportional) ───

    Args:
        non_zero_threshold: Sigmoid center (AsinhTransform: 0.88 ≈ asinh(1))
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

        # Two-timescale self-referential loss tracking for progress routing.
        # Both EMAs reuse the single _EMA_BETA constant (slow is the EMA of
        # fast), so no extra timescale/hyperparameter is introduced.
        self._loss_ema: list[float] | None = None       # fast EMA (~1/(1-beta))
        self._loss_ema_slow: list[float] | None = None  # slow EMA (~2/(1-beta))

        # Shape and level terms are composition-robust WITHOUT cross-batch state:
        # each is a self-normalized (Hájek) ratio estimator loss = Σ(w·ℓ)/Σ(w)
        # over the current batch, so numerator and denominator scale together
        # with event composition and no running weight-scale EMA is needed.

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

    @staticmethod
    def _haar_shape(e: torch.Tensor) -> torch.Tensor:
        """Haar detail (local AC) via adjacent-pair demeaning.

        Splits the time axis into non-overlapping pairs (t, t+1) and subtracts
        each pair's mean. The residual is the Haar detail coefficient carried
        in the original (B, T[, C]) layout (both members hold ±deviation), so
        every downstream per-cell op (event weighting, DRO, Hájek) is unchanged.

        Strictly local: a spike at t only affects its own pair, unlike global
        demeaning which contaminates every timestep's shape. Odd final timestep
        (T odd) carries zero deviation — its signal lives only in the level term.
        """
        T = e.shape[1]
        if T < 2:
            return e - e.mean(dim=1, keepdim=True)
        n_pairs = T // 2
        core = e[:, : 2 * n_pairs, ...]
        if e.dim() == 3:
            C = e.shape[2]
            pairs = core.reshape(e.shape[0], n_pairs, 2, C)
            detail = (pairs - pairs.mean(dim=2, keepdim=True)).reshape(
                e.shape[0], 2 * n_pairs, C
            )
        else:
            pairs = core.reshape(e.shape[0], n_pairs, 2)
            detail = (pairs - pairs.mean(dim=2, keepdim=True)).reshape(
                e.shape[0], 2 * n_pairs
            )
        if T - 2 * n_pairs:
            detail = torch.cat([detail, torch.zeros_like(e[:, 2 * n_pairs :, ...])], dim=1)
        return detail

    @staticmethod
    def _block_mean_broadcast(e: torch.Tensor, k: int) -> torch.Tensor:
        """Replace each timestep with the mean of its length-k dyadic block.

        Handles arbitrary T (a trailing partial block is averaged over its own
        length). The gradient of a block mean w.r.t. its members is 1/k and
        zero elsewhere, so the operation is strictly local.
        """
        B, T = e.shape[0], e.shape[1]
        n_full = T // k
        parts = []
        if n_full:
            core = e[:, : n_full * k, ...]
            if e.dim() == 3:
                C = e.shape[2]
                m = core.reshape(B, n_full, k, C).mean(dim=2, keepdim=True)
                parts.append(m.expand(B, n_full, k, C).reshape(B, n_full * k, C))
            else:
                m = core.reshape(B, n_full, k).mean(dim=2, keepdim=True)
                parts.append(m.expand(B, n_full, k).reshape(B, n_full * k))
        if T - n_full * k:
            tail = e[:, n_full * k :, ...]
            parts.append(tail.mean(dim=1, keepdim=True).expand_as(tail))
        return torch.cat(parts, dim=1)

    @classmethod
    def _haar_level(cls, e: torch.Tensor) -> torch.Tensor:
        """Multi-scale dyadic average-pool DC/level anchor (Haar approximation).

        For each dyadic block size k = 2, 4, 8, … up to the full horizon,
        compare the block-mean prediction error (the Haar approximation
        coefficient) against zero via log_cosh, broadcast that block value back
        over the timesteps it covers, and average across scales. The result is
        a strictly local, DC-carrying level signal in the original (B, T[, C])
        layout: the coarsest scale is the global mean (total-magnitude anchor);
        finer scales localize where the cumulative mass imbalance sits.

        Unlike cumsum, each e[t] contributes to exactly one block per scale, so
        the gradient is O(log T), position-symmetric, and non-smearing — it
        removes cumsum's O(T) early-step amplification while keeping DC
        sensitivity. Orthogonal complement of the Haar detail used for shape.
        """
        T = e.shape[1]
        if T < 2:
            return cls._log_cosh(e)
        scales = []
        k = 2
        while True:
            scales.append(min(k, T))
            if k >= T:
                break
            k *= 2
        acc = None
        for k in scales:
            l = cls._log_cosh(cls._block_mean_broadcast(e, k))
            acc = l if acc is None else acc + l
        return acc / len(scales)

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def _combine_channels(self, per_channel_loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Combine per-channel losses by *relative learning progress*.

        Two failure modes of magnitude-based routing are avoided:

        * Routing on a channel's absolute (scale-normalised) loss makes the
          router chase whichever target has the highest *irreducible* noise
          floor, permanently starving channels that could still improve.
        * Dividing the loss by the physical target scale (RMS) systematically
          down-weights the largest-signal channel — the primary target — and
          mixes units (a scaled level term over an asinh-RMS is not a clean
          relative error).

        Instead each channel is compared only to *its own* history via two
        cascaded EMAs that share the single existing smoothing constant
        (so no extra timescale is introduced):

            fast_c  = EMA_beta(loss_c)       # ~1/(1-beta) steps
            slow_c  = EMA_beta(fast_c)       # ~2/(1-beta) steps
            score_c = fast_c / slow_c        # dimensionless trend
            w_c     = C * score_c / Sum_k(score_k)

        score_c > 1 when channel c is regressing or lagging the others'
        progress, ~1 when it has plateaued (incl. at its noise floor), and
        < 1 when it is the fastest-improving channel.  Being a self-referential
        ratio, the score stays near 1 for any converged channel, so the weights
        cannot collapse to a winner-take-all regime (no target is starved)
        while gradient is still tilted toward the least-improving channel.
        """
        C = per_channel_loss.shape[0]
        batch_loss_det = per_channel_loss.detach()
        beta = self._EMA_BETA

        # ── Two-timescale self-referential loss tracking ─────────────
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

        # ── Relative-progress routing ────────────────────────────────
        fast = per_channel_loss.new_tensor(self._loss_ema)
        slow = per_channel_loss.new_tensor(self._loss_ema_slow)
        scores = fast / slow.clamp(min=self._EMA_EPS)
        w_soft = C * scores / scores.sum().clamp(min=self._EMA_EPS)

        self._last_weights = w_soft.tolist()
        # Telemetry (keys preserved for the callback contract):
        self._last_cal_ratio = scores.tolist()       # progress ratio fast/slow
        self._last_cal_score = list(self._loss_ema)  # fast EMA
        self._last_gates = w_soft.tolist()

        return (w_soft * per_channel_loss).sum()

    def _windowed_level_loss(
        self, e: torch.Tensor, y_true: torch.Tensor, T: int,
        y_pred_det: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Event-magnitude-weighted multi-scale DC/level anchor.

        Compare dyadic block-mean prediction errors (Haar approximation)
        against zero with log_cosh, averaged across scales.

        y_pred_det: detached prediction tensor — when supplied, series weighting
            uses max(|y_true|, |y_pred_det|) so that predicted false positives
            on peaceful series also attract level loss gradient. Must be same
            shape as y_true.
        """
        # Per-series event magnitude: max(|y_true|, |y_pred|) across time -> gate.
        if y_pred_det is not None:
            abs_max_series = torch.max(y_true.abs(), y_pred_det.abs())
        else:
            abs_max_series = y_true.abs()
        series_mag = abs_max_series.max(dim=1).values  # (B,) or (B, C)
        series_gate = 0.0125 + 0.9875 * torch.sigmoid(
            10.0 * (series_mag - self.non_zero_threshold)
        )
        series_w = series_gate * (1.0 + series_mag)  # magnitude-graded

        l_level = self._haar_level(e)

        if l_level.dim() == 3:
            num = (series_w.unsqueeze(1) * l_level).sum(dim=(0, 1))
            den = (series_w.sum(dim=0) * l_level.shape[1]).clamp(min=self._EMA_EPS)
        else:
            num = (series_w.unsqueeze(1) * l_level).sum()
            den = (series_w.sum() * l_level.shape[1]).clamp(min=self._EMA_EPS)

        level = num / den
        return level

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

        # ── Haar local DC/AC decomposition ────────────────────────────
        # Adjacent-pair demeaning: the deviation is the Haar detail (local AC/
        # shape), the pair mean is the Haar approximation (local DC/level).
        # Strictly local, so a spike only perturbs its own pair.
        e_shape = self._haar_shape(e)

        # ── Base cell loss ─────────────────────────────────────────────
        cell_loss = self._log_cosh(e_shape)

        # ── Gated + magnitude-graded event weighting ──────────────────
        # The sigmoid is a *peace-suppression gate* only: peace → ~0, conflict
        # → ~1. Above ~2 deaths it saturates, so on its own it weighted a
        # 2-death skirmish identically to a 10,000-death war and left the
        # entire 4-OOM tail flat (the source of peak under-prediction /
        # flattening). We restore magnitude sensitivity by multiplying the gate
        # by (1 + abs_max): abs_max is already in asinh space, which compresses
        # 4 OOM into ~[0,10], so the factor is bounded (Ukraine ~10x a 1-death
        # cell) and requires NO new constant — the asinh transform already in
        # the pipeline IS the data-driven scale. abs_max = max(|y_true|,
        # |y_pred.detach()|) keeps it feedback-loop-safe (under-predicting a
        # true event keeps |y_true| large; the detach prevents gaming).
        abs_max = torch.max(torch.abs(y_true), torch.abs(y_pred.detach()))
        # event_mag = 0.01 + 0.99 * torch.sigmoid(5.0 * (abs_max - self.non_zero_threshold))
        event_gate = 0.0125 + 0.9875 * torch.sigmoid(10.0 * (abs_max - self.non_zero_threshold))
        event_mag = event_gate * (1.0 + abs_max)

        # ── Per-series temporal DRO ────────────────────────────────────
        w_dro = self._dro_weights_2d(cell_loss, y_true)  # (B, T) or (B, T, C)
        w_total = torch.nan_to_num(event_mag * w_dro, nan=1.0, posinf=1.0, neginf=0.0)

        # ── Hájek self-normalized shape (composition-robust) ──────────
        # Weight-mass-weighted mean of the per-cell log_cosh — the
        # self-normalized (Hájek) ratio estimator loss = Σ(w·ℓ)/Σ(w). Numerator
        # and denominator move together with the batch's event composition, so
        # the shape scale is invariant to how many event cells the batch happens
        # to contain. This replaces the cross-batch EMA rescale: no running
        # state, no lag, no composition memory (the EMA lag was implicated in the
        # flat-collapse oscillation).
        if w_total.dim() == 3:
            num = (w_total * cell_loss).sum(dim=(0, 1))              # (C,)
            den = w_total.sum(dim=(0, 1)).clamp(min=self._EMA_EPS)   # (C,)
            loss_shape = num / den                                  # (C,)
        else:
            num = (w_total * cell_loss).sum()
            den = w_total.sum().clamp(min=self._EMA_EPS)
            loss_shape = num / den                                  # scalar

        # ── Multi-scale (Haar approximation) DC/level anchor ─────────────
        loss_level = self._windowed_level_loss(e, y_true, T, y_pred_det=y_pred.detach())

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