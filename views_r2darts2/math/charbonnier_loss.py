import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss for globally z-scored data (AsinhTransform -> StandardScaler).

    L(x) = mean( sqrt( (y_pred - y_true)^2 + eps^2 ) - eps )

    Behaves as L2 (MSE) for small errors and L1 (MAE) for large errors,
    with a C^inf smooth transition everywhere. No piecewise breakpoints,
    no hyperparameters. The subtracted eps term ensures L(0) = 0 exactly.

    When targets are globally standardized, the extreme exponential tails
    are fully neutralized by the preprocessing pipeline. The network
    operates in a clean ~N(0,1) space where Charbonnier provides
    mathematically perfect gradient flow at all magnitudes.
    """

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs:
            logger.info(
                "CharbonnierLoss | ignoring legacy kwargs: %s",
                list(kwargs.keys()),
            )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

        eps = 1e-4
        diff = y_pred - y_true
        return torch.mean(torch.sqrt(diff * diff + eps * eps) - eps)
