import torch
import logging

logger = logging.getLogger(__name__)

class WeightedHuberLoss(torch.nn.Module):
    """
    A PyTorch loss module that computes a weighted Huber loss.
    Samples with target values whose absolute value exceeds `zero_threshold` are assigned a higher weight (`non_zero_weight`),
    while others are assigned a weight of 1.0. The Huber loss is calculated with the specified `delta`.

    Args:
        zero_threshold (float): Threshold to determine if a target is considered non-zero.
        delta (float): The delta parameter for the Huber loss.
        non_zero_weight (float): Weight applied to samples with non-zero targets.

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth target values.

    Returns:
        torch.Tensor: The mean weighted Huber loss over the batch.
    """

    def __init__(self, zero_threshold: float, delta: float, non_zero_weight: float):
        super().__init__()
        self.threshold = zero_threshold
        self.delta = delta
        self.non_zero_weight = non_zero_weight

    def forward(self, preds, targets):
        from views_r2darts2.infrastructure.exceptions import NumericalSanityError
        
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in predictions.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in targets.")

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
        
        if torch.isnan(weighted_loss).any():
             raise NumericalSanityError("Numerical Sanity Violation: NaN detected in calculated loss.")
             
        return torch.mean(weighted_loss)
