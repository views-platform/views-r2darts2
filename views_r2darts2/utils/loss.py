import torch
import inspect
import logging

logger = logging.getLogger(__name__)


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
            **kwargs: Arbitrary keyword arguments to pass to the loss function's constructor.
                Only arguments matching the constructor's parameters will be used.

        Returns:
            An instance of the requested loss function class, initialized with the valid keyword arguments.

        Raises:
            ValueError: If the provided loss_name is not recognized.

        Example:
            >>> loss = get_loss_function("WeightedHuberLoss", delta=1.0, weight=0.5)
        """
        # Map loss names to their classes
        loss_classes = {
            "WeightedHuberLoss": WeightedHuberLoss,
            "TimeAwareWeightedHuberLoss": TimeAwareWeightedHuberLoss,
            "SpikeFocalLoss": SpikeFocalLoss,
            "WeightedPenaltyHuberLoss": WeightedPenaltyHuberLoss,
        }

        if loss_name not in loss_classes:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Get the class constructor parameters
        cls = loss_classes[loss_name]
        params = inspect.signature(cls).parameters

        # Filter kwargs to only include valid parameters for this loss
        valid_kwargs = {k: v for k, v in kwargs.items() if k in params}

        return cls(**valid_kwargs)


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

    def __init__(self, zero_threshold=0.01, delta=0.5, non_zero_weight=5.0):
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
        time_weights = torch.tensor(
            [self.decay_factor ** (seq_len - i) for i in range(seq_len)],
            device=targets.device,
        )

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

    def __init__(self, alpha=0.8, gamma=2.0, spike_threshold=3.0445):
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


class WeightedPenaltyHuberLoss(torch.nn.Module):
    """
    Custom weighted Huber loss with penalties for false positives and false negatives.

    This loss function extends the standard Huber loss by applying different weights to:
    - Non-zero targets (higher importance)
    - False positives (predicted non-zero when target is zero)
    - False negatives (predicted zero when target is non-zero)

    Args:
        zero_threshold (float, optional): Threshold to consider a value as zero. Defaults to 0.01.
        delta (float, optional): Huber loss delta parameter. Defaults to 0.5.
        non_zero_weight (float, optional): Weight for non-zero targets. Defaults to 5.0.
        false_positive_weight (float, optional): Penalty weight for false positives. Defaults to 10.0.
        false_negative_weight (float, optional): Penalty weight for false negatives. Defaults to 15.0.

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Scalar mean of the weighted Huber loss.
    """

    def __init__(
        self,
        zero_threshold=0.01,
        delta=0.5,
        non_zero_weight=5.0,
        false_positive_weight=10.0,
        false_negative_weight=15.0,
    ):
        super().__init__()
        self.threshold = zero_threshold
        self.delta = delta
        self.non_zero_weight = non_zero_weight
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        logger.info(
            "\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}".format(
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
        # Identify non-zero targets and predictions (detach masks to prevent gradient flow)
        is_target_nonzero = torch.abs(targets) > self.threshold
        is_pred_nonzero = (torch.abs(preds) > self.threshold).detach()

        # Base weights: prioritize non-zero targets
        base_weights = torch.where(is_target_nonzero, self.non_zero_weight, 1.0)

        # Identify error types
        false_positive_mask = ~is_target_nonzero & is_pred_nonzero
        false_negative_mask = is_target_nonzero & ~is_pred_nonzero

        # Apply penalties to error types while preserving base weights
        weights = torch.where(
            false_positive_mask,
            self.false_positive_weight,
            torch.where(false_negative_mask, self.false_negative_weight, base_weights),
        )

        # Calculate Huber loss
        errors = targets - preds
        huber_loss = torch.where(
            torch.abs(errors) <= self.delta,
            0.5 * errors**2,
            self.delta * (torch.abs(errors) - 0.5 * self.delta),
        )

        # Apply computed weights
        weighted_loss = weights * huber_loss
        return torch.mean(weighted_loss)
