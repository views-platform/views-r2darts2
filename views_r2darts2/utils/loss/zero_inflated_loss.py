import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ZeroInflatedLoss(torch.nn.Module):
    """
    Zero-Inflated Loss for sparse conflict data.

    Args:
        zero_weight (float): Weight for the binary (zero/non-zero) classification component. 
        count_weight (float): Weight for the count/intensity component. 
        delta (float): Huber loss delta parameter for the count component. 
        zero_threshold (float): Threshold to determine if a value is considered zero. 
        eps (float): Small constant for numerical stability. 

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth target values.

    Returns:
        torch.Tensor: The weighted sum of binary and count loss components.
    """

    def __init__(self, zero_weight: float, count_weight: float, delta: float, zero_threshold: float, eps: float):
        super().__init__()
        self.zero_weight = zero_weight
        self.count_weight = count_weight
        self.delta = delta
        self.threshold = zero_threshold
        self.eps = eps
        logger.info(
            f"\nzero_weight: {zero_weight}\n"
            f"count_weight: {count_weight}\n"
            f"delta: {delta}\n"
            f"zero_threshold: {zero_threshold}"
        )

    def forward(self, preds, targets):
        from views_r2darts2.utils.exceptions import NumericalSanityError
        
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in predictions.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise NumericalSanityError("Numerical Sanity Violation: NaN or Inf detected in targets.")

        # Ensure shapes match by flattening if needed
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)

        # Binary component: classify zero vs non-zero
        is_zero = (torch.abs(targets_flat) < self.threshold).to(preds.dtype)

        # Soft zero detection using sigmoid
        pred_prob_zero = torch.sigmoid(-preds_flat * 10) 

        # Clamp to avoid numerical issues with BCE
        pred_prob_zero = torch.clamp(pred_prob_zero, self.eps, 1.0 - self.eps)

        # Binary cross-entropy for zero/non-zero classification
        zero_loss = F.binary_cross_entropy(pred_prob_zero, is_zero, reduction="mean")

        # Count component: Pseudo-Huber loss only on non-zero targets
        count_mask = 1.0 - is_zero
        count_errors = (preds_flat - targets_flat) * count_mask

        # Pseudo-Huber: smooth approximation to Huber loss
        count_loss = self.delta**2 * (
            torch.sqrt(1 + (count_errors / self.delta) ** 2) - 1
        )

        final_loss = self.zero_weight * zero_loss + self.count_weight * count_loss.mean()
        
        if torch.isnan(final_loss).any():
             raise NumericalSanityError("Numerical Sanity Violation: NaN detected in calculated loss.")
             
        return final_loss
