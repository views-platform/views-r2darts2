import torch
import logging
from views_r2darts2.infrastructure.exceptions import NumericalSanityError

logger = logging.getLogger(__name__)

class WeightedPenaltyHuberLoss(torch.nn.Module):
    """
    Custom weighted Huber loss with multiplicative penalties for false positives and false negatives.

    This loss function extends the standard Huber loss by applying different weights to:
    - Non-zero targets (higher importance via non_zero_weight)
    - False positives (predicted non-zero when target is zero) - multiplied by false_positive_weight
    - False negatives (predicted zero when target is non-zero) - multiplied by false_negative_weight

    Args:
        zero_threshold (float): Threshold to consider a value as zero. 
        delta (float): Huber loss delta parameter. 
        non_zero_weight (float): Base weight for non-zero targets. 
        false_positive_weight (float): Multiplier applied to base weight for false positives. 
        false_negative_weight (float): Multiplier applied to base weight for false negatives. 

    Forward Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Scalar mean of the weighted Huber loss.
    """

    def __init__(
        self,
        zero_threshold: float,
        delta: float,
        non_zero_weight: float,
        false_positive_weight: float,
        false_negative_weight: float,
    ):
        super().__init__()
        self.threshold = zero_threshold
        self.delta = delta
        self.non_zero_weight = non_zero_weight
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        logger.info(
            f"\nzero_threshold: {zero_threshold}\n"
            f"delta: {delta}\n"
            f"non_zero_weight: {non_zero_weight}\n"
            f"false_positive_weight: {false_positive_weight}\n"
            f"false_negative_weight: {false_negative_weight}"
        )

    def forward(self, preds, targets):
        # Ensure same dtype and device for numerical stability
        device = preds.device
        dtype = preds.dtype

        # Check for NaN/Inf in TARGETS (data issue)
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            error_msg = "Numerical Sanity Violation: NaN or Inf detected in targets."
            logger.critical(error_msg)
            raise NumericalSanityError(error_msg)

        # Check for NaN/Inf in PREDICTIONS (model instability)
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            error_msg = "Numerical Sanity Violation: NaN or Inf detected in predictions (model instability)."
            logger.critical(error_msg)
            raise NumericalSanityError(error_msg)

        # Identify non-zero targets and predictions (detach pred mask to prevent gradient flow)
        is_target_nonzero = torch.abs(targets) > self.threshold
        is_pred_nonzero = (torch.abs(preds) > self.threshold).detach()

        # Base weights: prioritize non-zero targets
        non_zero_w = torch.tensor(self.non_zero_weight, device=device, dtype=dtype)
        one = torch.tensor(1.0, device=device, dtype=dtype)
        base_weights = torch.where(is_target_nonzero, non_zero_w, one)

        # Identify error types
        false_positive_mask = (
            ~is_target_nonzero & is_pred_nonzero
        ) 
        false_negative_mask = (
            is_target_nonzero & ~is_pred_nonzero
        ) 

        # Apply multiplicative penalties on top of base weights
        fp_weight = torch.tensor(self.false_positive_weight, device=device, dtype=dtype)
        fn_weight = torch.tensor(self.false_negative_weight, device=device, dtype=dtype)

        weights = torch.where(
            false_positive_mask,
            base_weights * fp_weight,
            torch.where(
                false_negative_mask,
                base_weights * fn_weight,
                base_weights,
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

        # Apply computed weights
        weighted_loss = weights * huber_loss

        # Final NaN check
        if torch.isnan(weighted_loss).any():
            error_msg = "Numerical Sanity Violation: NaN in weighted_loss."
            logger.critical(error_msg)
            raise NumericalSanityError(error_msg)

        return torch.mean(weighted_loss)
