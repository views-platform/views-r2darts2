"""Mean Squared Error (MSE) loss."""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Mean Squared Error loss — standard L2 regression loss.
    
    Computes: mean((y_pred - y_true)^2)
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize MSE loss.

        Args:
            reduction: 'mean', 'sum', or 'none' — how to aggregate.
        """
        super().__init__()
        self.reduction = reduction
        self._mse = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            y_pred: Predicted values (batch, time, features).
            y_true: Ground truth values (batch, time, features).

        Returns:
            Scalar loss or per-element loss depending on reduction.
        """
        return self._mse(y_pred, y_true)
