import torch
import logging

logger = logging.getLogger(__name__)

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

    def __init__(self, zero_weight: float, non_zero_weight: float, decay_factor: float, delta: float):
        super().__init__()
        self.zero_weight = zero_weight
        self.non_zero_weight = non_zero_weight
        self.decay_factor = decay_factor
        self.delta = delta
        self.huber = torch.nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, preds, targets):
        from views_r2darts2.infrastructure.exceptions import NumericalSanityError
        
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in predictions.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in targets.")

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
        
        weighted_loss = weights * losses
        if torch.isnan(weighted_loss).any():
             raise NumericalSanityError("Numerical Sanity Violation: NaN detected in calculated loss.")
             
        return weighted_loss.mean()
