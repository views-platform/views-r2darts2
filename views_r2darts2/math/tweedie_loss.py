import torch
import torch.nn.functional as F
import logging
from views_r2darts2.infrastructure.exceptions import NumericalSanityError

logger = logging.getLogger(__name__)

class TweedieLoss(torch.nn.Module):
    """
    Tweedie Deviance Loss for zero-inflated continuous data.

    Args:
        p (float): Power parameter in (1, 2). p=1.5 is good for conflict data.
        non_zero_weight (float): Base weight multiplier for non-zero targets.
        zero_threshold (float): Threshold to determine if a target is considered zero.
        false_positive_weight (float): Multiplier for FP errors. 
        false_negative_weight (float): Multiplier for FN errors. 
        eps (float): Small constant for numerical stability. 

    Forward Args:
        preds (torch.Tensor): Predicted values (passed through softplus for positivity).
        targets (torch.Tensor): Ground truth target values (should be non-negative).

    Returns:
        torch.Tensor: The mean weighted Tweedie deviance loss.
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
            raise ValueError(
                f"Tweedie power parameter p must be in (1, 2), but got {p}"
            )
        self.p = p
        self.non_zero_weight = non_zero_weight
        self.threshold = zero_threshold
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        self.eps = eps
        logger.info(
            f"\np (Tweedie power): {p}\n"
            f"non_zero_weight: {non_zero_weight}\n"
            f"zero_threshold: {zero_threshold}\n"
            f"false_positive_weight: {false_positive_weight}\n"
            f"false_negative_weight: {false_negative_weight}"
        )

    def forward(self, preds, targets):
        device = preds.device
        dtype = preds.dtype

        # Handle NaN/Inf in targets
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            error_msg = "Numerical Sanity Violation: NaN or Inf detected in targets."
            logger.critical(error_msg)
            raise NumericalSanityError(error_msg)

        # Ensure targets are non-negative (Tweedie requires y >= 0)
        targets = torch.clamp(targets, min=0.0)

        # Ensure predictions are positive via softplus (Tweedie requires μ > 0)
        preds_pos = F.softplus(preds) + self.eps

        # Handle NaN/Inf in predictions
        if torch.isnan(preds_pos).any() or torch.isinf(preds_pos).any():
            error_msg = "Numerical Sanity Violation: NaN or Inf detected in predictions (model instability)."
            logger.critical(error_msg)
            raise NumericalSanityError(error_msg)

        # Tweedie deviance
        p = self.p
        loss = torch.pow(preds_pos, 2 - p) / (2 - p) - targets * torch.pow(
            preds_pos, 1 - p
        ) / (1 - p)

        # Identify zero/non-zero for weighting
        is_target_nonzero = targets > self.threshold
        is_pred_nonzero = (preds_pos > self.threshold).detach()

        # Base weights: prioritize non-zero targets
        non_zero_w = torch.tensor(self.non_zero_weight, device=device, dtype=dtype)
        one = torch.tensor(1.0, device=device, dtype=dtype)
        base_weights = torch.where(is_target_nonzero, non_zero_w, one)

        # Apply FP/FN multipliers
        false_positive_mask = ~is_target_nonzero & is_pred_nonzero
        false_negative_mask = is_target_nonzero & ~is_pred_nonzero

        fp_w = torch.tensor(self.false_positive_weight, device=device, dtype=dtype)
        fn_w = torch.tensor(self.false_negative_weight, device=device, dtype=dtype)

        weights = torch.where(
            false_positive_mask,
            base_weights * fp_w,
            torch.where(false_negative_mask, base_weights * fn_w, base_weights),
        )

        weighted_loss = weights * loss

        # Final NaN check
        if torch.isnan(weighted_loss).any():
            error_msg = "Numerical Sanity Violation: NaN in Tweedie loss."
            logger.critical(error_msg)
            raise NumericalSanityError(error_msg)

        return torch.mean(weighted_loss)
