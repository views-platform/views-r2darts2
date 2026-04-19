import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SpotlightLoss(torch.nn.Module):
    """
    Magnitude-aware temporal loss for zero-inflated conflict time-series.

    Designed for asinh-transformed targets (UCDP GED state-based conflict),
    where ~90% of cells are structural zeros and a small minority of high-
    magnitude conflict events carry the forecasting signal. Standard mean
    losses collapse to flat-line predictions; this loss prevents that via
    four coordinated mechanisms.

    Mechanism 1 — Asymmetric spotlight weight (cosh truth, log-cosh pred):
        Each cell is weighted by the arithmetic mean of two complementary terms:

            w_mag = 0.5 * (cosh(alpha * |y|) + (1 + log_cosh(alpha * |ŷ|.detach())))

        Truth side: cosh(alpha * |y|) — exponential growth, delivering strong
        intra-event contrast on real data which is bounded by the data distribution
        (asinh-space max ≈ 11.5 for UCDP GED). At alpha=0.3:
            y=0  → 1.0×  (peace)
            y=5  → 7.5×  (moderate conflict)
            y=10 → 47×   (high conflict)
        Contrast ratio high/moderate ≈ 6×, vs ~3× for quadratic at the same alpha.

        Pred side: 1 + log_cosh(alpha * |ŷ|) — linear for large |ŷ| (since
        log_cosh(x) → |x| - ln2 for |x| >> 1), minimum 1.0, overflow-safe.
        Acts as a shock absorber for OOD blowups: for |ŷ|=50 at alpha=0.3,
        pred weight ≈ 16 vs 226 for quadratic. This caps the gradient magnitude
        (w_mag × tanh(e/s)) at a controlled level, preventing violent AdamW
        momentum updates that cause oscillation and divergence.

        The prediction side is detached: no second-order gradient flows through
        w_mag. However, w_mag is a direct first-order scalar multiplier on the
        gradient (∂L/∂ŷ = w_eff × tanh(e/s)), so its magnitude matters. Linear
        growth on the pred side keeps OOD recovery gradients firm but bounded,
        while exponential growth on the truth side maximises contrast on real
        data where the distribution is known and bounded.

    Mechanism 2 — Temporal continuity score:
        An exponential-decay convolution kernel (tau = T/4, radius = 3*tau)
        is applied to the binary conflict mask to measure how much nearby
        conflict evidence surrounds each cell. The kernel self-excludes the
        current time step so a single isolated spike cannot support itself.
        Normalized against the maximum possible support (accounting for
        sequence boundaries), yielding continuity ∈ [0, 1] per cell.

    Mechanism 3 — Adaptive spotlight interpolation:
        The effective weight interpolates between a sub-linear floor and the
        full magnitude weight based on continuity::

            w_eff = sqrt(w_mag) + (w_mag - sqrt(w_mag)) * continuity

        continuity=0 (isolated spike) → w_eff = sqrt(w_mag)  (dampened)
        continuity=1 (sustained conflict) → w_eff = w_mag  (full weight)

        The floor is data-derived from the same cosh weight, requiring no
        manually tuned constants and automatically scaling with alpha.

    Mechanism 4 — Balanced dual-mean aggregation:
        Per-cell weighted losses are aggregated with equal 50/50 weight across
        event and peace groups, regardless of their relative frequency. If one
        group is absent from a batch, the present group receives 100% weight
        rather than 50%, preventing silent gradient halving. Balancing is
        computed per target channel so multi-target scenarios with different
        zero-inflation rates are handled independently.

    Base loss — Scaled log-cosh:
        The per-cell base loss uses a scaled log-cosh::

            L(e, s) = s * log(cosh(e / s)),
            s = 1 + |y| / (1 + |y|) + relu(|ŷ|.detach() - |y|)

        The scale has two components:
        - Truth-based floor: 1 + |y|/(1+|y|) ∈ [1.0, 2.0). Quadratic for |e| << s,
          linear for |e| >> s, smooth transition. The gradient tanh(e/s) is
          hard-bounded in (-1, 1) — natural gradient clipping without Huber.
        - OOD extension: relu(|ŷ| - |y|) adds the excess prediction beyond truth
          directly to the scale, without saturation. For in-distribution (ŷ ≈ y),
          relu = 0, no change. For OOD (ŷ=50, y=0): scale = 1 + 0 + 50 = 51,
          so tanh(50/51) = 0.75 — still in the linear (unsaturated) gradient
          regime. Without this, tanh(50/1.0) = 1.0 for all |e| > 3, meaning
          the model receives identical corrective gradient for ŷ=20 and ŷ=200.
          The prediction-aware scale ensures gradient grows with error magnitude,
          giving the optimizer increasing incentive to return from OOD.

    Optional dynamics term:
        When gamma > 0 and sequence length > 1, a second balanced-mean loss
        is computed on the first differences (Δy_pred - Δy_true), using
        level-based weighting and scaled log-cosh base loss. This encourages
        the model to track conflict onsets and cessations, and — critically —
        to maintain temporal coherence at high conflict levels.

        The dynamics weight uses the midpoint level between each adjacent step
        pair, not the velocity:

            abs_y_mid = 0.5 * (|y[t]| + |y[t+1]|)
            w_dyn_true = cosh(alpha * abs_y_mid)

        This gives plateau steps at y=11.5 weight ≈16× — identical to the
        pointwise term. Velocity-based weighting gave plateau steps weight
        ≈1×, allowing the model to produce slowly escalating output predictions
        without meaningful penalty. At inference on Ukraine-like contexts, this
        drift compounds across all 36 output steps, producing OOD extrapolation.

        The event/peace split is also level-based: abs_y_mid > threshold
        classifies a diff step as high-conflict (event), regardless of whether
        Δy_true is large or small. This ensures 50% of the dynamics budget goes
        to high-level outputs (plateaus included), not just onset/cessation steps.

        The scale is level-based (consistent with the pointwise scale):
        plateau steps at y_mid=11.5 get scale≈1.91 (wide quadratic region),
        giving finer gradient signal near the target velocity.

    Args:
        alpha (float): Scale for spotlight weights. Controls how aggressively
            high-magnitude conflict cells are up-weighted relative to peace.
            Recommended range: 0.2–0.5. Truth side uses cosh(alpha * |y|) for
            exponential contrast on bounded real data. Pred side uses
            1 + log_cosh(alpha * |ŷ|) for OOD-safe linear-growth damping.
        gamma (float): Scalar weight applied to the temporal dynamics loss
            before summing with the pointwise loss. Set to 0.0 to disable
            the dynamics term entirely. At gamma=1.0, dynamics and pointwise
            losses have equal footing after balanced-mean normalisation.
            Reduce to 0.5 if dynamics tracking dominates pointwise accuracy
            during early tuning.
        non_zero_threshold (float): asinh-space threshold separating "event"
            (conflict) cells from "peace" cells for balanced averaging.
            0.88 ≈ asinh(1), corresponding to ≥1 battle-related death in the
            raw UCDP count. Applied to |y_true| for the pointwise term and
            to |Δy_true| for the dynamics term.

    Raises:
        RuntimeError: If the computed total loss is NaN. Includes pointwise
            and dynamics component values in the message to aid diagnosis.

    Example:
        >>> loss_fn = SpotlightLoss(alpha=0.4, gamma=1.0, non_zero_threshold=0.88)
        >>> y_pred = torch.randn(8, 36)   # [batch, seq_len]
        >>> y_true = torch.zeros(8, 36)
        >>> y_true[:, 10:15] = 2.5        # sustained conflict window
        >>> loss = loss_fn(y_pred, y_true)
        >>> loss.backward()
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        non_zero_threshold: float,
    ):
        # ---- Parameter guardrails ----
        if alpha <= 0.0:
            raise ValueError(f"SpotlightLoss: alpha must be positive, got {alpha}")
        if alpha > 0.7:
            logger.warning(
                "SpotlightLoss: alpha=%.4f is above the recommended range (0.15–0.5). "
                "cosh truth weight at max UCDP (asinh≈11.5) = cosh(%.2f) ≈ %.0f×. "
                "Exponential contrast at this scale may cause training instability.",
                alpha, alpha * 11.5, math.cosh(min(alpha * 11.5, 88.0)),
            )
        elif alpha > 0.5:
            logger.warning(
                "SpotlightLoss: alpha=%.4f exceeds recommended range (0.15–0.5). "
                "cosh truth weight at max UCDP ≈ %.0f×. Monitor for instability.",
                alpha, math.cosh(alpha * 11.5),
            )

        if gamma < 0.0:
            raise ValueError(f"SpotlightLoss: gamma must be non-negative, got {gamma}")
        if gamma > 1.0:
            logger.warning(
                "SpotlightLoss: gamma=%.4f > 1.0. The dynamics term will dominate pointwise loss. "
                "This is known to cause OOD extrapolation and training instability. "
                "Strongly recommend gamma <= 0.7.",
                gamma,
            )
        elif gamma > 0.7:
            logger.warning(
                "SpotlightLoss: gamma=%.4f > 0.7 may amplify velocity trends. "
                "Monitor predictions for OOD extrapolation in early epochs.",
                gamma,
            )
        elif 0.0 < gamma < 0.2:
            logger.warning(
                "SpotlightLoss: gamma=%.4f < 0.2. The dynamics term provides critical "
                "temporal coherence for multi-step output. Below ~0.2, output steps are "
                "effectively unconstrained and OOD spikes are likely during inference. "
                "Recommended minimum: 0.2.",
                gamma,
            )

        if non_zero_threshold <= 0.0:
            raise ValueError(
                f"SpotlightLoss: non_zero_threshold must be positive, got {non_zero_threshold}"
            )
        if non_zero_threshold > 2.0:
            logger.warning(
                "SpotlightLoss: non_zero_threshold=%.4f corresponds to >%.1f raw deaths (asinh scale). "
                "Most conflict cells will be classified as peace, potentially starving the event group "
                "in the balanced mean. Recommended range: 0.5–1.5.",
                non_zero_threshold,
                math.sinh(non_zero_threshold),
            )

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

    @staticmethod
    def _safe_pred_weight(x: torch.Tensor) -> torch.Tensor:
        """1 + log_cosh(x). Linear-growth spotlight weight, minimum 1.0.

        For small |x|: ≈ 1 + x²/2 (quadratic-like, smooth at origin).
        For large |x|: ≈ 1 + |x| - ln2 (linear growth, shock absorber).
        Overflow-safe for any finite input (log_cosh is bounded by |x|).

        OOD comparison at alpha=0.3, |ŷ|=50:
            1 + log_cosh(15) ≈ 16   vs   1 + 15² = 226 (quadratic)
        The linear regime caps gradient magnitude during OOD recovery,
        preventing violent AdamW momentum updates.
        """
        abs_x = torch.abs(x)
        return 1.0 + abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)

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

        n_event = ie_flat.sum(0)  # [C]
        n_peace = (~ie_flat).sum(0)  # [C]

        # Calculate average loss per class (clamp prevents div/0)
        loss_event = (ps_flat * ie_flat).sum(0) / n_event.clamp(min=1)  # [C]
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

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        e = y_pred - y_true
        abs_y = torch.abs(y_true)
        abs_y_pred = torch.abs(y_pred)

        # ---- 1. Pointwise spotlight weight (asymmetric: cosh truth, log-cosh pred) ----
        # Truth side: cosh(alpha*|y|) — exponential contrast on bounded real data.
        #   At alpha=0.3: peace→1×, y=5→7.5×, y=10→47×. Strong intra-event contrast.
        # Pred side: 1+log_cosh(alpha*|ŷ|) — linear for large |ŷ|, OOD shock absorber.
        #   At alpha=0.3, |ŷ|=50: w_pred≈16 → gradient≈8.5 (vs 113.5 with quadratic).
        # Prediction side is detached: no second-order gradient through w_mag.
        w_true = torch.cosh(self.alpha * abs_y)
        w_pred = self._safe_pred_weight(self.alpha * abs_y_pred.detach())
        w_mag = 0.5 * (w_true + w_pred)

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
        kernel[radius] = 0.0  # Zero out center so a spike can't support itself

        # Reshape kernel for conv1d: [out_channels, in_channels/groups, K]
        # We will use groups=C to compute continuity independently per target
        if conflict_mask.dim() == 3:
            C = conflict_mask.size(-1)
            weight = kernel.view(1, 1, K).repeat(C, 1, 1)  # [C, 1, K]
            mask_input = conflict_mask.transpose(1, 2)  # [B, C, T]
            groups = C
        else:
            weight = kernel.view(1, 1, K)  # [1, 1, K]
            mask_input = conflict_mask.unsqueeze(1)  # [B, 1, T]
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
            continuity = continuity.squeeze(1)  # Back to [B, T]

        # ---- 3. Effective spotlight: sqrt(w_mag) floor ----
        # Interpolates between sqrt(w_mag) and w_mag along the continuity axis:
        #   continuity=0 → w_eff = sqrt(w_mag)   e.g. √27 ≈ 5.2× for a 50k one-off
        #   continuity=1 → w_eff = w_mag          full 27× for sustained conflict
        # Floor is data-derived (sub-linear of the same cosh weight, no constants).
        w_soft = torch.sqrt(w_mag)
        w_eff = w_soft + (w_mag - w_soft) * continuity

        # ---- 4. Pointwise loss ----
        # Prediction-aware scale: truth floor + OOD extension.
        # Floor: 1 + |y|/(1+|y|) ∈ [1, 2) — standard adaptive scale.
        # Extension: relu(|ŷ| - |y|) widens scale when pred overshoots truth,
        # keeping tanh(e/s) in its linear regime for OOD errors.
        # In-distribution (ŷ ≈ y): relu=0, no effect.
        # OOD (ŷ=50, y=0): scale=51, tanh(50/51)=0.75 (linear) vs tanh(50/1)=1.0 (saturated).
        ood_excess = F.relu(abs_y_pred.detach() - abs_y)
        scale = 1.0 + abs_y / (1.0 + abs_y) + ood_excess
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

            # Weight, scale, and event/peace split all use midpoint LEVEL, not velocity.
            # Plateau steps at y_mid=11.5 get cosh(alpha*11.5)≈16× weight — the model
            # cannot produce a slowly escalating output without paying the same cost as
            # a pointwise error of equal magnitude. Velocity weighting (cosh(alpha*|Dy|))
            # gave plateau steps weight≈1×, which allowed inference OOD drift.
            abs_y_mid = 0.5 * (abs_y[:, :-1] + abs_y[:, 1:])
            abs_y_pred_mid = 0.5 * (abs_y_pred[:, :-1] + abs_y_pred[:, 1:])

            w_dyn_true = torch.cosh(self.alpha * abs_y_mid)
            w_dyn_pred = self._safe_pred_weight(self.alpha * abs_y_pred_mid.detach())
            w_dyn = 0.5 * (w_dyn_true + w_dyn_pred)

            # Continuity dampening for dynamics: reuse pointwise continuity at the
            # midpoint of each adjacent pair. Isolated velocity spikes (e.g. Feb 2022
            # onset: 7k→50k in one step) get dampened to sqrt(w_dyn), while sustained
            # acceleration patterns keep full w_dyn. Same logic as pointwise mechanism 3.
            continuity_dyn = 0.5 * (continuity[:, :-1] + continuity[:, 1:])
            w_dyn_soft = torch.sqrt(w_dyn)
            w_dyn_eff = w_dyn_soft + (w_dyn - w_dyn_soft) * continuity_dyn

            # Level-based scale with OOD extension (same as pointwise).
            # Plateau at y_mid=11.5 → floor≈1.91. OOD excess from pred_mid widens
            # the scale to keep dynamics gradient unsaturated during blowups.
            ood_excess_dyn = F.relu(abs_y_pred_mid.detach() - abs_y_mid)
            scale_dyn = 1.0 + abs_y_mid / (1.0 + abs_y_mid) + ood_excess_dyn
            per_sample_dyn = w_dyn_eff * self._log_cosh_scaled(e_grad, scale_dyn)

            # Event/peace split by LEVEL: plateau at y_mid=11.5 counts as event,
            # sharing 50% of dynamics budget with onset/cessation steps.
            # Velocity split would classify plateaus as peace, burying them in
            # the 50% shared with uninformative peace-to-peace zero transitions.
            is_dynamic = abs_y_mid > self.non_zero_threshold
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
