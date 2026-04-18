import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Magnitude-aware temporal loss for zero-inflated conflict time-series
    on asinh-transformed targets (UCDP GED sb).

    Prevents flat-line forecasts via cosh magnitude weighting and a temporal
    dynamics term. Uses balanced dual-mean averaging to guarantee conflict
    gradients are never diluted by 90%+ zero-inflation.

    Ensures equal upwards/downwards pressure via an Arithmetic Mean spotlight
    weight with a detached prediction (preventing sinh gradient explosions).

    Uses a global continuity score to distinguish isolated spikes (likely
    noise) from sustained onsets/cessations. Interpolates the spotlight
    weight between sqrt(w_mag) and w_mag based on continuity, providing
    a data-driven, sub-linear dampening floor for one-off events.

    Parameters
    ----------
    alpha : float
        Rate for cosh magnitude weight cosh(alpha * |y|).
        At alpha=0.4: y=0 → w=1, y=5 → w≈3.8, y=10 → w≈27.
        Recommended range 0.2–0.5.
    gamma : float
        Weight for the temporal dynamics term.  Set to 0 to disable.
        Default 1.0 gives dynamics equal footing with pointwise loss
        after both are balanced-mean normalised.  Reduce to 0.5 if
        dynamics tracking dominates pointwise accuracy during tuning.
    non_zero_threshold : float
        asinh threshold separating "event" from "peace" cells for
        balanced averaging.  0.88 ≈ asinh(1), i.e. ≥1 battle-related
        death.  Applied to |y_true| for pointwise and |Δy_true| for
        dynamics.
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        non_zero_threshold: float,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.non_zero_threshold = non_zero_threshold

        logger.info(
            "SpotlightLoss | alpha=%.4f gamma=%.4f threshold=%.4f",
            self.alpha,
            self.gamma,
            self.non_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_cosh_scaled(error: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Scaled log-cosh: s * log(cosh(e / s)).

        Quadratic for |e| << s, linear for |e| >> s, smooth transition.
        Gradient = tanh(e/s), hard-bounded in (-1, 1).

        Numerically stable via:
            log(cosh(x)) = |x| + softplus(-2|x|) - ln2
        """
        z = error / scale
        abs_z = torch.abs(z)
        return scale * (abs_z + F.softplus(-2.0 * abs_z) - math.log(2.0))

    def _balanced_mean(
        self, per_sample: torch.Tensor, is_event: torch.Tensor
    ) -> torch.Tensor:
        """Equal-weighted average over event and non-event cells.

        Prevents the majority class from diluting the minority class gradient.
        Each group contributes 50% if both are present. If one group is
        entirely absent, the present group contributes 100%, preventing
        silent gradient scaling issues.
        """
        # Standardize to 3D [B, T, C] for vectorized per-target logic
        if per_sample.dim() == 2:
            per_sample = per_sample.unsqueeze(-1)
            is_event = is_event.unsqueeze(-1)

        C = per_sample.size(-1)
        ps_flat = per_sample.reshape(-1, C)
        ie_flat = is_event.reshape(-1, C)

        n_event = ie_flat.sum(0)           # [C]
        n_peace = (~ie_flat).sum(0)        # [C]

        # Calculate average loss per class (clamp prevents div/0)
        loss_event = (ps_flat * ie_flat).sum(0) / n_event.clamp(min=1)   # [C]
        loss_peace = (ps_flat * ~ie_flat).sum(0) / n_peace.clamp(min=1)  # [C]

        # Dynamic weights: 50/50 if both present, 100/0 if one is absent
        w_e = 0.5 * (n_event > 0).float()
        w_p = 0.5 * (n_peace > 0).float()
        total_w = (w_e + w_p).clamp(min=1e-8)  # clamp prevents div/0 on empty batch
        w_e = w_e / total_w
        w_p = w_p / total_w

        balanced = w_e * loss_event + w_p * loss_peace  # [C]

        # Average across target channels
        return balanced.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        e = y_pred - y_true
        abs_y = torch.abs(y_true)
        abs_y_pred = torch.abs(y_pred)

        # ---- 1. Pointwise spotlight weight ----
        # Arithmetic Mean Spotlight with detached prediction.
        # Value = AM (perfect equal pressure). Gradient = detached (no sinh explosion).
        cosh_true = torch.cosh(self.alpha * abs_y)
        cosh_pred = torch.cosh(self.alpha * abs_y_pred.detach())
        w_mag = 0.5 * (cosh_true + cosh_pred)

        # ---- 2. Time-series continuity score (Local Conv1d) ----
        # Continuity is a local property. We use a 1D convolution with an
        # exponential decay kernel to measure local conflict support.

        T = y_true.size(1)
        conflict_mask = (abs_y > self.non_zero_threshold).float()  # [B, T] or [B, T, C]

        # Define kernel parameters
        # Half-life of T/4 means ~95% of weight is within 2 months for T=36
        tau = max(T / 4.0, 1.0)
        radius = int(math.ceil(3 * tau))  # Capture 95% of exponential decay
        K = 2 * radius + 1

        # Construct the 1D kernel
        t_idx_k = torch.arange(K, dtype=y_true.dtype, device=y_true.device) - radius
        kernel = torch.exp(-torch.abs(t_idx_k) / tau)
        kernel[radius] = 0.0  # CRITICAL: Zero out center so a spike can't support itself

        # Reshape kernel for conv1d: [out_channels, in_channels/groups, K]
        # We will use groups=C to compute continuity independently per target
        if conflict_mask.dim() == 3:
            C = conflict_mask.size(-1)
            weight = kernel.view(1, 1, K).repeat(C, 1, 1)  # [C, 1, K]
            mask_input = conflict_mask.transpose(1, 2)       # [B, C, T]
            groups = C
        else:
            weight = kernel.view(1, 1, K)                   # [1, 1, K]
            mask_input = conflict_mask.unsqueeze(1)          # [B, 1, T]
            groups = 1

        # Calculate local support (numerator)
        support = F.conv1d(mask_input, weight, padding=radius, groups=groups)

        # Calculate max possible local support for normalization (denominator)
        ones_input = torch.ones_like(mask_input)
        max_support = F.conv1d(ones_input, weight, padding=radius, groups=groups)

        # Normalize to [0, 1] and restore shape
        continuity = support / max_support.clamp(min=1e-8)

        if conflict_mask.dim() == 3:
            continuity = continuity.transpose(1, 2)  # Back to [B, T, C]
        else:
            continuity = continuity.squeeze(1)       # Back to [B, T]

        # ---- 3. Effective spotlight: sqrt(w_mag) floor ----
        # Interpolates between sqrt(w_mag) and w_mag along the continuity axis:
        #   continuity=0 → w_eff = sqrt(w_mag)   e.g. √27 ≈ 5.2× for a 50k one-off
        #   continuity=1 → w_eff = w_mag          full 27× for sustained conflict
        # Floor is data-derived (sub-linear of the same cosh weight, no constants).
        w_soft = torch.sqrt(w_mag)
        w_eff = w_soft + (w_mag - w_soft) * continuity

        # ---- 4. Pointwise loss ----
        scale = 1.0 + abs_y / (1.0 + abs_y)
        base_loss = self._log_cosh_scaled(e, scale)

        per_sample_pw = w_eff * base_loss

        is_event = abs_y > self.non_zero_threshold
        loss_pointwise = self._balanced_mean(per_sample_pw, is_event)

        # ---- 5. Temporal dynamics loss ----
        loss_dynamics = y_pred.new_tensor(0.0)
        if y_pred.size(1) > 1 and self.gamma > 0.0:
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            e_grad = diff_pred - diff_true

            abs_diff_true = torch.abs(diff_true)
            abs_diff_pred = torch.abs(diff_pred)

            # AM with detach for dynamics
            cosh_dyn_true = torch.cosh(self.alpha * abs_diff_true)
            cosh_dyn_pred = torch.cosh(self.alpha * abs_diff_pred.detach())
            w_dyn = 0.5 * (cosh_dyn_true + cosh_dyn_pred)
            
            scale_dyn = 1.0 + abs_diff_true / (1.0 + abs_diff_true)
            per_sample_dyn = w_dyn * self._log_cosh_scaled(e_grad, scale_dyn)

            is_dynamic = abs_diff_true > self.non_zero_threshold
            loss_dynamics = self._balanced_mean(per_sample_dyn, is_dynamic)

        total_loss = loss_pointwise + self.gamma * loss_dynamics

        if torch.isnan(total_loss):
            raise RuntimeError(
                "NaN in SpotlightLoss. "
                f"pointwise={loss_pointwise.item():.6f} "
                f"dynamics={loss_dynamics.item():.6f}"
            )

        logger.debug(
            "SpotlightLoss | pw=%.6f dyn=%.6f total=%.6f",
            loss_pointwise.item(),
            loss_dynamics.item(),
            total_loss.item(),
        )

        return total_loss

    def __repr__(self) -> str:
        return (
            f"SpotlightLoss(alpha={self.alpha}, gamma={self.gamma}, "
            f"non_zero_threshold={self.non_zero_threshold})"
        )
