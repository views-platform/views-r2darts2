"""Huber loss — smooth approximation between MSE and MAE."""

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """Huber loss — robust to outliers, smooth transition between L2 and L1.
    
    For |residual| <= delta: loss = 0.5 * residual^2
    For |residual| > delta:  loss = delta * (|residual| - 0.5 * delta)
    
    Args:
        delta: Threshold above which loss switches from quadratic to linear.
               Typically 1.0 for regression. Smaller values (0.1-0.5) emphasize
               outlier robustness; larger values (2.0+) behave closer to MSE.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        """Initialize Huber loss.

        Args:
            delta: Threshold for switch between MSE and MAE behavior.
            reduction: 'mean', 'sum', or 'none' — how to aggregate.
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self._huber = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss.

        Args:
            y_pred: Predicted values (batch, time, features).
            y_true: Ground truth values (batch, time, features).

        Returns:
            Scalar loss or per-element loss depending on reduction.
        """
        return self._huber(y_pred, y_true)
