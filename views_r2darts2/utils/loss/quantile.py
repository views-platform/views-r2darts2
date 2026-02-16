import torch
import logging

logger = logging.getLogger(__name__)

class AsymmetricQuantileLoss(torch.nn.Module):
    """
    Asymmetric Quantile Loss for conflict forecasting.

    Args:
        tau (float): Quantile level in (0, 1). Higher values penalize underestimation more. 
        non_zero_weight (float): Additional weight for non-zero targets. 
        zero_threshold (float): Threshold to determine if a target is considered zero. 

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth target values.

    Returns:
        torch.Tensor: The mean weighted quantile loss over the batch.
    """

    def __init__(self, tau: float, non_zero_weight: float, zero_threshold: float):
        super().__init__()
        if not (0 < tau < 1):
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = tau
        self.non_zero_weight = non_zero_weight
        self.threshold = zero_threshold
        logger.info(
            f"\ntau (quantile): {tau}\n"
            f"non_zero_weight: {non_zero_weight}\n"
            f"zero_threshold: {zero_threshold}"
        )

    def forward(self, preds, targets):
        from views_r2darts2.utils.gates import NumericalSanityError
        
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in predictions.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in targets.")

        errors = targets - preds

        # Asymmetric quantile loss
        quantile_loss = torch.where(
            errors >= 0,
            self.tau * errors,  # Underestimation penalty
            (self.tau - 1)
            * errors,  # Overestimation penalty
        )

        # Additional weight for non-zero targets
        weights = torch.where(
            torch.abs(targets) > self.threshold,
            torch.tensor(
                self.non_zero_weight, device=targets.device, dtype=targets.dtype
            ),
            torch.tensor(1.0, device=targets.device, dtype=targets.dtype),
        )

        weighted_loss = weights * quantile_loss
        if torch.isnan(weighted_loss).any():
             raise NumericalSanityError("Numerical Sanity Violation: NaN detected in calculated loss.")
             
        return weighted_loss.mean()
