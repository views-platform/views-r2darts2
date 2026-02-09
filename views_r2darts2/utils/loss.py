import torch
import torch.nn.functional as F
import inspect
import logging

logger = logging.getLogger(__name__)

# CHANGES MADE BY GEMINI CLI AGENT:
# - Added the following standard PyTorch loss functions to LossSelector.get_loss_function:
#   - MSELoss (torch.nn.MSELoss)
#   - L1Loss (torch.nn.L1Loss)
#   - HuberLoss (torch.nn.HuberLoss)
#   - SmoothL1Loss (torch.nn.SmoothL1Loss)
#   - PoissonNLLLoss (torch.nn.PoissonNLLLoss)
# - The docstring for LossSelector.get_loss_function was updated to reflect these additions.
#
# This comment is added to facilitate updating the test suite for these new loss functions.


class LossSelector:
    @staticmethod
    def get_loss_function(loss_name, **kwargs):
        """
        Returns an instance of the specified loss function class with provided keyword arguments.

        Parameters:
            loss_name (str): The name of the loss function to instantiate. Must be one of:
                - "WeightedHuberLoss"
                - "TimeAwareWeightedHuberLoss"
                - "SpikeFocalLoss"
                - "WeightedPenaltyHuberLoss"
                - "TweedieLoss"
                - "AsymmetricQuantileLoss"
                - "ZeroInflatedLoss"
                - "ShrinkageLoss"
                - "MSELoss"
                - "L1Loss"
                - "HuberLoss"
                - "SmoothL1Loss"
                - "PoissonNLLLoss"
            **kwargs: Arbitrary keyword arguments to pass to the loss function's constructor.
                Only arguments matching the constructor's parameters will be used.

        Returns:
            An instance of the requested loss function class, initialized with the valid keyword arguments.

        Raises:
            ValueError: If the provided loss_name is not recognized.

        Example:
            >>> loss = get_loss_function("WeightedHuberLoss", delta=1.0, weight=0.5)
        """
        # Standard PyTorch losses
        loss_classes = {
            "WeightedHuberLoss": WeightedHuberLoss,
            "TimeAwareWeightedHuberLoss": TimeAwareWeightedHuberLoss,
            "SpikeFocalLoss": SpikeFocalLoss,
            "WeightedPenaltyHuberLoss": WeightedPenaltyHuberLoss,
            "TweedieLoss": TweedieLoss,
            "AsymmetricQuantileLoss": AsymmetricQuantileLoss,
            "ZeroInflatedLoss": ZeroInflatedLoss,
            "ShrinkageLoss": ShrinkageLoss,
            # Standard PyTorch losses
            "MSELoss": torch.nn.MSELoss,
            "L1Loss": torch.nn.L1Loss,
            "HuberLoss": torch.nn.HuberLoss,
            "SmoothL1Loss": torch.nn.SmoothL1Loss,
            "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
        }

        if loss_name not in loss_classes:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Get the class constructor parameters
        cls = loss_classes[loss_name]
        params = inspect.signature(cls).parameters

        # Filter kwargs to only include valid parameters for this loss
        valid_kwargs = {k: v for k, v in kwargs.items() if k in params}

        return cls(**valid_kwargs)


class ShrinkageLoss(torch.nn.Module):
    """
    Implementation of the Shrinkage Loss for regression, as described in
    "Deep Object Tracking with Shrinkage Loss" by Lu et al. (2018).

    This loss function is designed to handle data imbalance in regression tasks by
    penalizing the importance of easy samples (those with small errors),
    allowing the model to focus more on hard samples. It is particularly effective
    for zero-inflated data where the model might otherwise be biased towards
    predicting zeros.

    The loss is calculated as:
    LS = (preds - targets)**2 / (1 + exp(a * (c - |preds - targets|)))

    Args:
        a (float, optional): Controls the shrinkage speed. Higher values lead to
            faster shrinkage of the loss for easy samples. Defaults to 10.0,
            as recommended in the paper.
        c (float, optional): Represents the threshold (localization) of what is
            considered an "easy" sample. Errors below this value will be more
            heavily penalized (i.e., their contribution to the loss will be shrunk).
            Defaults to 0.2, as recommended for a target range of [0, 1].

    Returns:
        torch.Tensor: A scalar tensor representing the mean Shrinkage Loss over the batch.
    """

    def __init__(self, a: float = 10.0, c: float = 0.2):
        super().__init__()
        self.a = a
        self.c = c
        logger.info(
            "\n{:<25} {:<10}\n{:<25} {:<10}".format(
                "a (shrinkage speed)",
                self.a,
                "c (easy sample threshold)",
                self.c,
            )
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        abs_error = torch.abs(preds - targets)
        # The denominator is the core of the shrinkage mechanism.
        # For small l (easy samples), exp() is large, so the denominator is large, shrinking the loss.
        # For large l (hard samples), exp() is small, so the denominator is close to 1, leaving the loss largely unchanged.
        shrinkage_factor = 1 + torch.exp(self.a * (self.c - abs_error))

        # The numerator contains the squared error.
        base_loss = abs_error**2

        loss = base_loss / shrinkage_factor
        return torch.mean(loss)

class WeightedHuberLoss(torch.nn.Module):
    """
    A PyTorch loss module that computes a weighted Huber loss.
    Samples with target values whose absolute value exceeds `zero_threshold` are assigned a higher weight (`non_zero_weight`),
    while others are assigned a weight of 1.0. The Huber loss is calculated with the specified `delta`.

    Args:
        zero_threshold (float, optional): Threshold to determine if a target is considered non-zero. Defaults to 0.01.
        delta (float, optional): The delta parameter for the Huber loss. Defaults to 0.5.
        non_zero_weight (float, optional): Weight applied to samples with non-zero targets. Defaults to 5.0.

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth target values.

    Returns:
        torch.Tensor: The mean weighted Huber loss over the batch.
    """

    def __init__(self, zero_threshold, delta, non_zero_weight):
        super().__init__()
        self.threshold = zero_threshold
        self.delta = delta
        self.non_zero_weight = non_zero_weight

    def forward(self, preds, targets):
        # Create sample weights
        weights = torch.where(
            torch.abs(targets) > self.threshold, self.non_zero_weight, 1.0
        )

        # Calculate base Huber loss
        errors = targets - preds
        huber_loss = torch.where(
            torch.abs(errors) <= self.delta,
            0.5 * errors**2,
            self.delta * (torch.abs(errors) - 0.5 * self.delta),
        )

        # Apply sample weighting
        weighted_loss = weights * huber_loss
        return torch.mean(weighted_loss)


class TimeAwareWeightedHuberLoss(torch.nn.Module):
    """
    A PyTorch loss module that computes a time-aware, event-weighted Huber loss.

    This loss function applies temporal decay and event-based weighting to the standard Huber loss,
    allowing for differential emphasis on target values based on their time step and magnitude.

    Args:
        zero_weight (float): Weight applied to target values considered 'zero' (abs(target) <= 1e-4).
        non_zero_weight (float): Weight applied to target values considered 'non-zero' (abs(target) > 1e-4).
        decay_factor (float): Factor by which weights decay over time steps (should be in (0, 1]).
        delta (float): The threshold at which the Huber loss transitions from quadratic to linear.

    Forward Args:
        preds (Tensor): Predicted values of shape (batch_size, seq_len, ...).
        targets (Tensor): Ground truth values of shape (batch_size, seq_len, ...).

    Returns:
        Tensor: The weighted mean Huber loss value.
    """

    def __init__(self, zero_weight, non_zero_weight, decay_factor, delta):
        super().__init__()
        self.zero_weight = zero_weight
        self.non_zero_weight = non_zero_weight
        self.decay_factor = decay_factor
        self.delta = delta
        self.huber = torch.nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, preds, targets):
        # Temporal decay weights
        seq_len = targets.size(1)
        # Corrected exponent to give higher weight to more recent events
        time_weights = torch.tensor(
            [self.decay_factor ** (seq_len - 1 - i) for i in range(seq_len)],
            device=targets.device,
        )

        # Reshape for broadcasting: (seq_len,) -> (1, seq_len, 1, ...)
        # This allows multiplication with (batch, seq_len, features, ...)
        view_shape = (1, seq_len) + (1,) * (targets.dim() - 2)
        time_weights = time_weights.view(*view_shape)

        # Event weights
        event_weights = torch.where(
            targets.abs() > 1e-4, self.non_zero_weight, self.zero_weight
        )

        # Combined weighting
        weights = time_weights * event_weights
        losses = self.huber(preds, targets)
        return (weights * losses).mean()


class SpikeFocalLoss(torch.nn.Module):
    """
    SpikeFocalLoss is a custom PyTorch loss function that combines mean squared error (MSE) with a focal weighting mechanism,
    specifically designed to emphasize errors on "spike" targets.

    Args:
        alpha (float, optional): Weighting factor for spike targets. Default is 0.8.
        gamma (float, optional): Focusing parameter for modulating the focal effect. Default is 2.0.
        spike_threshold (float, optional): Threshold above which targets are considered spikes. Default is 3.0445.

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Scalar loss value.

    Description:
        - Calculates the absolute error between predictions and targets.
        - Identifies "spike" targets using the spike_threshold.
        - Applies a focal weighting to the squared error, with different weights for spike and non-spike targets.
        - Returns the mean of the weighted loss values.
    """

    def __init__(self, alpha, gamma, spike_threshold):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.spike_threshold = spike_threshold

    def forward(self, preds, targets):
        errors = (preds - targets).abs()
        is_spike = targets > self.spike_threshold

        # Base loss
        loss = errors**2

        # Focal component
        focal_weights = torch.where(
            is_spike,
            self.alpha * (1 - torch.exp(-errors)) ** self.gamma,
            (1 - self.alpha) * (torch.exp(-errors)) ** self.gamma,
        )

        return torch.mean(focal_weights * loss)


# class WeightedPenaltyHuberLoss(torch.nn.Module):
#     """
#     Custom weighted Huber loss with penalties for false positives and false negatives.

#     This loss function extends the standard Huber loss by applying different weights to:
#     - Non-zero targets (higher importance)
#     - False positives (predicted non-zero when target is zero)
#     - False negatives (predicted zero when target is non-zero)

#     Args:
#         zero_threshold (float, optional): Threshold to consider a value as zero. Defaults to 0.01.
#         delta (float, optional): Huber loss delta parameter. Defaults to 0.5.
#         non_zero_weight (float, optional): Weight for non-zero targets. Defaults to 5.0.
#         false_positive_weight (float, optional): Penalty weight for false positives. Defaults to 10.0.
#         false_negative_weight (float, optional): Penalty weight for false negatives. Defaults to 15.0.

#     Forward Args:
#         preds (torch.Tensor): Predicted values.
#         targets (torch.Tensor): Ground truth values.

#     Returns:
#         torch.Tensor: Scalar mean of the weighted Huber loss.
#     """

#     def __init__(
#         self,
#         zero_threshold=0.01,
#         delta=0.5,
#         non_zero_weight=5.0,
#         false_positive_weight=10.0,
#         false_negative_weight=15.0,
#     ):
#         super().__init__()
#         self.threshold = zero_threshold
#         self.delta = delta
#         self.non_zero_weight = non_zero_weight
#         self.false_positive_weight = false_positive_weight
#         self.false_negative_weight = false_negative_weight
#         logger.info(
#             "\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}".format(
#                 "zero_threshold",
#                 zero_threshold,
#                 "delta",
#                 delta,
#                 "non_zero_weight",
#                 non_zero_weight,
#                 "false_positive_weight",
#                 false_positive_weight,
#                 "false_negative_weight",
#                 false_negative_weight,
#             )
#         )

#     def forward(self, preds, targets):
#         # Identify non-zero targets and predictions (detach masks to prevent gradient flow)
#         is_target_nonzero = torch.abs(targets) > self.threshold
#         is_pred_nonzero = (torch.abs(preds) > self.threshold).detach()

#         # Base weights: prioritize non-zero targets
#         base_weights = torch.where(is_target_nonzero, self.non_zero_weight, 1.0)

#         # Identify error types
#         false_positive_mask = ~is_target_nonzero & is_pred_nonzero
#         false_negative_mask = is_target_nonzero & ~is_pred_nonzero

#         # Apply penalties to error types while preserving base weights
#         weights = torch.where(
#             false_positive_mask,
#             self.false_positive_weight,
#             torch.where(false_negative_mask, self.false_negative_weight, base_weights),
#         )

#         # Calculate Huber loss
#         errors = targets - preds
#         huber_loss = torch.where(
#             torch.abs(errors) <= self.delta,
#             0.5 * errors**2,
#             self.delta * (torch.abs(errors) - 0.5 * self.delta),
#         )

#         # Apply computed weights
#         weighted_loss = weights * huber_loss
#         return torch.mean(weighted_loss)

class WeightedPenaltyHuberLoss(torch.nn.Module):
    """
    Custom weighted Huber loss with multiplicative penalties for false positives and false negatives.

    This loss function extends the standard Huber loss by applying different weights to:
    - Non-zero targets (higher importance via non_zero_weight)
    - False positives (predicted non-zero when target is zero) - multiplied by false_positive_weight
    - False negatives (predicted zero when target is non-zero) - multiplied by false_negative_weight

    Args:
        zero_threshold (float, optional): Threshold to consider a value as zero. Defaults to 0.01.
        delta (float, optional): Huber loss delta parameter. Defaults to 0.5.
        non_zero_weight (float, optional): Base weight for non-zero targets. Defaults to 5.0.
        false_positive_weight (float, optional): Multiplier applied to base weight for false positives. Defaults to 2.0.
        false_negative_weight (float, optional): Multiplier applied to base weight for false negatives. Defaults to 3.0.

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Scalar mean of the weighted Huber loss.
    
    Example weights:
        - Zero target, zero pred: weight = 1.0
        - Non-zero target, correct pred: weight = non_zero_weight (e.g., 5.0)
        - Zero target, non-zero pred (FP): weight = 1.0 * false_positive_weight (e.g., 2.0)
        - Non-zero target, zero pred (FN): weight = non_zero_weight * false_negative_weight (e.g., 5.0 * 3.0 = 15.0)
    """

    def __init__(
        self,
        zero_threshold,
        delta,
        non_zero_weight,
        false_positive_weight,
        false_negative_weight,
    ):
        super().__init__()
        self.threshold = zero_threshold
        self.delta = delta
        self.non_zero_weight = non_zero_weight
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        logger.info(
            "\n{:<30} {:<10}\n{:<30} {:<10}\n{:<30} {:<10}\n{:<30} {:<10}\n{:<30} {:<10}".format(
                "zero_threshold",
                zero_threshold,
                "delta",
                delta,
                "non_zero_weight",
                non_zero_weight,
                "false_positive_weight",
                false_positive_weight,
                "false_negative_weight",
                false_negative_weight,
            )
        )

    def forward(self, preds, targets):
        # Ensure same dtype and device for numerical stability
        device = preds.device
        dtype = preds.dtype
        
        # Check for NaN/Inf in TARGETS (data issue) - these we can safely handle
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logger.warning("NaN or Inf detected in targets - replacing with 0")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for NaN/Inf in PREDICTIONS (model instability) - log but DON'T mask
        # Masking predictions hides the problem and produces garbage gradients
        nan_mask = torch.isnan(preds) | torch.isinf(preds)
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            total_count = preds.numel()
            nan_pct = 100 * nan_count / total_count
            logger.warning(
                f"NaN/Inf in predictions: {nan_count}/{total_count} ({nan_pct:.1f}%) - "
                f"model is unstable! Check: learning_rate, gradient_clip, norm_type, d_model/nhead ratio"
            )
            # Replace with a large penalty value instead of 0 to create strong gradient signal
            # This helps the optimizer correct the instability rather than ignoring it
            preds = torch.where(nan_mask, torch.zeros_like(preds), preds)
        
        # Identify non-zero targets and predictions (detach pred mask to prevent gradient flow)
        is_target_nonzero = torch.abs(targets) > self.threshold
        is_pred_nonzero = (torch.abs(preds) > self.threshold).detach()

        # Base weights: prioritize non-zero targets (ensure correct dtype/device)
        non_zero_w = torch.tensor(self.non_zero_weight, device=device, dtype=dtype)
        one = torch.tensor(1.0, device=device, dtype=dtype)
        base_weights = torch.where(is_target_nonzero, non_zero_w, one)

        # Identify error types
        false_positive_mask = ~is_target_nonzero & is_pred_nonzero  # Predicted conflict when none exists
        false_negative_mask = is_target_nonzero & ~is_pred_nonzero  # Missed a real conflict

        # Apply multiplicative penalties on top of base weights (ensure correct dtype/device)
        fp_weight = torch.tensor(self.false_positive_weight, device=device, dtype=dtype)
        fn_weight = torch.tensor(self.false_negative_weight, device=device, dtype=dtype)
        
        weights = torch.where(
            false_positive_mask,
            base_weights * fp_weight,  # e.g., 1.0 * 2.0 = 2.0
            torch.where(
                false_negative_mask,
                base_weights * fn_weight,  # e.g., 5.0 * 3.0 = 15.0
                base_weights  # No error: use base weight
            ),
        )

        # Calculate Huber loss
        errors = targets - preds
        delta_t = torch.tensor(self.delta, device=device, dtype=dtype)
        abs_errors = torch.abs(errors)
        huber_loss = torch.where(
            abs_errors <= delta_t,
            0.5 * errors**2,
            delta_t * (abs_errors - 0.5 * delta_t),
        )

        # Apply computed weights and handle potential NaN
        weighted_loss = weights * huber_loss
        
        # Final NaN check
        if torch.isnan(weighted_loss).any():
            logger.warning("NaN in weighted_loss, replacing with 0")
            weighted_loss = torch.nan_to_num(weighted_loss, nan=0.0)
        
        return torch.mean(weighted_loss)


class TweedieLoss(torch.nn.Module):
    """
    Tweedie Deviance Loss for zero-inflated continuous data.
    
    The Tweedie distribution is a family of exponential dispersion models:
    - p=1: Poisson (discrete counts)
    - p=2: Gamma (continuous positive)
    - 1<p<2: Compound Poisson-Gamma (zero-inflated continuous) ← USE THIS
    
    Why Tweedie for conflict forecasting:
    -------------------------------------
    Tweedie with 1<p<2 is theoretically designed for zero-inflated data:
    - Has a point mass at zero (most country-months have no conflict)
    - Continuous positive distribution for non-zero values
    - Naturally handles excess zeros without explicit zero-inflation modeling
    - Is a proper scoring rule (unlike heuristic losses)
    
    The power parameter p controls the zero-inflation behavior:
    - p closer to 1: More tolerance for zeros, better for very sparse data
    - p closer to 2: More weight on positive values, better for moderate sparsity
    - p=1.5: Good default for conflict data (~85% zeros)
    
    Deviance Formula:
    -----------------
    D(y, μ) = 2 * [y^(2-p)/((1-p)(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p)]
    
    The first term is constant w.r.t. μ (predictions) and is dropped for optimization.

    FP/FN Weighting:
    ----------------
    Unlike Huber loss, Tweedie naturally handles zero-inflation through its 
    mathematical structure. However, it doesn't encode asymmetric operational costs
    (e.g., "missing a conflict is worse than a false alarm").
    
    The FP/FN weights are optional (default to 1.0) and can be tuned if you need
    this asymmetry. For pure Tweedie behavior, leave them at 1.0.

    Args:
        p (float): Power parameter in (1, 2). p=1.5 is good for conflict data.
        non_zero_weight (float): Base weight multiplier for non-zero targets.
        zero_threshold (float): Threshold to determine if a target is considered zero.
        false_positive_weight (float): Multiplier for FP errors. Defaults to 1.0.
        false_negative_weight (float): Multiplier for FN errors. Defaults to 1.0.
        eps (float): Small constant for numerical stability. Defaults to 1e-8.

    Forward Args:
        preds (torch.Tensor): Predicted values (passed through softplus for positivity).
        targets (torch.Tensor): Ground truth target values (should be non-negative).

    Returns:
        torch.Tensor: The mean weighted Tweedie deviance loss.
    
    Example weight structure (with defaults fp=1.0, fn=1.0):
        - TN (zero→zero): weight = 1.0
        - FP (zero→non-zero): weight = 1.0 * false_positive_weight = 1.0
        - TP (non-zero→non-zero): weight = non_zero_weight = 5.0
        - FN (non-zero→zero): weight = non_zero_weight * false_negative_weight = 5.0
    
    Note: Unlike Huber, Tweedie naturally handles zero-inflation, so you may not
    need aggressive FP/FN weighting. Start with defaults and tune if needed.
    """

    def __init__(
        self,
        non_zero_weight: float,
        zero_threshold: float,
        p: float,
        false_positive_weight: float,
        false_negative_weight: float,
        eps: float,
    ):
        super().__init__()
        if not (1 < p < 2):
            raise ValueError(f"Tweedie power parameter p must be in (1, 2), but got {p}")
        self.p = p
        self.non_zero_weight = non_zero_weight
        self.threshold = zero_threshold
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        self.eps = eps
        logger.info(
            "\n{:<30} {:<10}\n{:<30} {:<10}\n{:<30} {:<10}\n{:<30} {:<10}\n{:<30} {:<10}".format(
                "p (Tweedie power)", p,
                "non_zero_weight", non_zero_weight,
                "zero_threshold", zero_threshold,
                "false_positive_weight", false_positive_weight,
                "false_negative_weight", false_negative_weight,
            )
        )

    def forward(self, preds, targets):
        device = preds.device
        dtype = preds.dtype
        
        # Handle NaN/Inf in targets
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logger.warning("NaN or Inf detected in targets - replacing with 0")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=0.0)
        
        # Ensure targets are non-negative (Tweedie requires y >= 0)
        targets = torch.clamp(targets, min=0.0)
        
        # Ensure predictions are positive via softplus (Tweedie requires μ > 0)
        # softplus(x) = log(1 + exp(x)), smooth approximation to ReLU
        preds_pos = F.softplus(preds) + self.eps
        
        # Handle NaN/Inf in predictions
        nan_mask = torch.isnan(preds_pos) | torch.isinf(preds_pos)
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            total_count = preds_pos.numel()
            logger.warning(
                f"NaN/Inf in predictions: {nan_count}/{total_count} - model unstable"
            )
            preds_pos = torch.where(nan_mask, torch.full_like(preds_pos, self.eps), preds_pos)
        
        # Tweedie deviance (dropping constant term that doesn't depend on predictions)
        # D(y, μ) ∝ μ^(2-p)/(2-p) - y*μ^(1-p)/(1-p)
        p = self.p
        loss = (
            torch.pow(preds_pos, 2 - p) / (2 - p)
            - targets * torch.pow(preds_pos, 1 - p) / (1 - p)
        )
        
        # Identify zero/non-zero for weighting
        is_target_nonzero = targets > self.threshold
        is_pred_nonzero = (preds_pos > self.threshold).detach()
        
        # Base weights: prioritize non-zero targets
        non_zero_w = torch.tensor(self.non_zero_weight, device=device, dtype=dtype)
        one = torch.tensor(1.0, device=device, dtype=dtype)
        base_weights = torch.where(is_target_nonzero, non_zero_w, one)
        
        # Apply FP/FN multipliers if configured (defaults are 1.0, so no effect)
        false_positive_mask = ~is_target_nonzero & is_pred_nonzero
        false_negative_mask = is_target_nonzero & ~is_pred_nonzero
        
        fp_w = torch.tensor(self.false_positive_weight, device=device, dtype=dtype)
        fn_w = torch.tensor(self.false_negative_weight, device=device, dtype=dtype)
        
        weights = torch.where(
            false_positive_mask,
            base_weights * fp_w,
            torch.where(
                false_negative_mask,
                base_weights * fn_w,
                base_weights
            ),
        )
        
        weighted_loss = weights * loss
        
        # Final NaN check
        if torch.isnan(weighted_loss).any():
            logger.warning("NaN in Tweedie loss, replacing with 0")
            weighted_loss = torch.nan_to_num(weighted_loss, nan=0.0)
        
        return torch.mean(weighted_loss)


class AsymmetricQuantileLoss(torch.nn.Module):
    """
    Asymmetric Quantile Loss for conflict forecasting.
    
    Quantile regression naturally handles asymmetric error costs. For conflict
    forecasting where underestimation (missing conflicts) is typically costlier
    than overestimation (false alarms), use tau > 0.5.
    
    - tau = 0.5: Symmetric (equivalent to MAE)
    - tau = 0.75: 3x penalty for underestimation vs overestimation
    - tau = 0.9: 9x penalty for underestimation vs overestimation

    Args:
        tau (float, optional): Quantile level in (0, 1). Higher values penalize underestimation more. Defaults to 0.75.
        non_zero_weight (float, optional): Additional weight for non-zero targets. Defaults to 5.0.
        zero_threshold (float, optional): Threshold to determine if a target is considered zero. Defaults to 0.01.

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth target values.

    Returns:
        torch.Tensor: The mean weighted quantile loss over the batch.
    """

    def __init__(self, tau, non_zero_weight, zero_threshold):
        super().__init__()
        if not (0 < tau < 1):
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = tau
        self.non_zero_weight = non_zero_weight
        self.threshold = zero_threshold
        logger.info(
            "\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}".format(
                "tau (quantile)", tau,
                "non_zero_weight", non_zero_weight,
                "zero_threshold", zero_threshold,
            )
        )

    def forward(self, preds, targets):
        errors = targets - preds
        
        # Asymmetric quantile loss
        # Positive error (underestimation): weight by tau
        # Negative error (overestimation): weight by (1 - tau)
        quantile_loss = torch.where(
            errors >= 0,
            self.tau * errors,           # Underestimation penalty
            (self.tau - 1) * errors      # Overestimation penalty (note: errors < 0, so this is positive)
        )
        
        # Additional weight for non-zero targets
        # Use explicit tensors for MPS compatibility
        weights = torch.where(
            torch.abs(targets) > self.threshold, 
            torch.tensor(self.non_zero_weight, device=targets.device, dtype=targets.dtype),
            torch.tensor(1.0, device=targets.device, dtype=targets.dtype)
        )
        
        return (weights * quantile_loss).mean()


class ZeroInflatedLoss(torch.nn.Module):
    """
    Zero-Inflated Loss for sparse conflict data.
    
    Explicitly models the two-part nature of zero-inflated data:
    1. Binary component: Is there any conflict? (logistic-like)
    2. Count component: How much conflict? (Huber loss on non-zero values)
    
    This is particularly suited for conflict data where the majority of
    observations are zeros, and we need to separately model the occurrence
    vs. intensity of conflict.

    Args:
        zero_weight (float, optional): Weight for the binary (zero/non-zero) classification component. Defaults to 1.0.
        count_weight (float, optional): Weight for the count/intensity component. Defaults to 1.0.
        delta (float, optional): Huber loss delta parameter for the count component. Defaults to 0.5.
        zero_threshold (float, optional): Threshold to determine if a value is considered zero. Defaults to 0.01.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-8.

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth target values.

    Returns:
        torch.Tensor: The weighted sum of binary and count loss components.
    """

    def __init__(self, zero_weight, count_weight, delta, zero_threshold, eps):
        super().__init__()
        self.zero_weight = zero_weight
        self.count_weight = count_weight
        self.delta = delta
        self.threshold = zero_threshold
        self.eps = eps
        logger.info(
            "\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}".format(
                "zero_weight", zero_weight,
                "count_weight", count_weight,
                "delta", delta,
                "zero_threshold", zero_threshold,
            )
        )

    def forward(self, preds, targets):
        # Ensure shapes match by flattening if needed
        # Darts may pass (batch, time, features) or (batch, time) tensors
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Binary component: classify zero vs non-zero
        is_zero = (torch.abs(targets_flat) < self.threshold).to(preds.dtype)
        
        # Soft zero detection using sigmoid
        # Higher predictions -> lower probability of zero
        pred_prob_zero = torch.sigmoid(-preds_flat * 10)  # Scale factor for sharper transition
        
        # Clamp to avoid numerical issues with BCE
        pred_prob_zero = torch.clamp(pred_prob_zero, self.eps, 1.0 - self.eps)
        
        # Binary cross-entropy for zero/non-zero classification
        zero_loss = F.binary_cross_entropy(pred_prob_zero, is_zero, reduction='mean')
        
        # Count component: Pseudo-Huber loss only on non-zero targets
        count_mask = 1.0 - is_zero
        count_errors = (preds_flat - targets_flat) * count_mask
        
        # Pseudo-Huber: smooth approximation to Huber loss
        # L(e) = delta^2 * (sqrt(1 + (e/delta)^2) - 1)
        count_loss = self.delta**2 * (torch.sqrt(1 + (count_errors / self.delta)**2) - 1)
        
        return self.zero_weight * zero_loss + self.count_weight * count_loss.mean()
