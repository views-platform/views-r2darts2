import torch
import logging

logger = logging.getLogger(__name__)

class SpikeFocalLoss(torch.nn.Module):
    """
    SpikeFocalLoss is a custom PyTorch loss function that combines mean squared error (MSE) with a focal weighting mechanism,
    specifically designed to emphasize errors on "spike" targets.

    Args:
        alpha (float): Weighting factor for spike targets. 
        gamma (float): Focusing parameter for modulating the focal effect. 
        spike_threshold (float): Threshold above which targets are considered spikes. 

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Scalar loss value.
    """

    def __init__(self, alpha: float, gamma: float, spike_threshold: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.spike_threshold = spike_threshold

    def forward(self, preds, targets):
        from views_r2darts2.utils.gates import NumericalSanityError
        
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in predictions.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in targets.")

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

        weighted_loss = focal_weights * loss
        if torch.isnan(weighted_loss).any():
             raise NumericalSanityError("Numerical Sanity Violation: NaN detected in calculated loss.")
             
        return torch.mean(weighted_loss)
