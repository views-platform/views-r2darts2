import torch
import logging

logger = logging.getLogger(__name__)

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
        a (float): Controls the shrinkage speed. Higher values lead to
            faster shrinkage of the loss for easy samples. 
        c (float): Represents the threshold (localization) of what is
            considered an "easy" sample. Errors below this value will be more
            heavily penalized (i.e., their contribution to the loss will be shrunk).

    Returns:
        torch.Tensor: A scalar tensor representing the mean Shrinkage Loss over the batch.
    """

    def __init__(self, a: float, c: float):
        super().__init__()
        self.a = a
        self.c = c
        logger.info(
            f"\na (shrinkage speed): {self.a}\n"
            f"c (easy sample threshold): {self.c}"
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        from views_r2darts2.infrastructure.exceptions import NumericalSanityError
        
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in predictions.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in targets.")

        abs_error = torch.abs(preds - targets)
        # The denominator is the core of the shrinkage mechanism.
        shrinkage_factor = 1 + torch.exp(self.a * (self.c - abs_error))

        # The numerator contains the squared error.
        base_loss = abs_error**2

        loss = base_loss / shrinkage_factor
        
        if torch.isnan(loss).any():
             raise NumericalSanityError("Numerical Sanity Violation: NaN detected in calculated loss.")
             
        return torch.mean(loss)
