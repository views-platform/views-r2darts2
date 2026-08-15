"""Log-Cosh loss — smooth approximation of L1 loss."""

import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    """Log-Cosh loss — smooth version of MAE, more numerically stable.
    
    loss = mean(log(cosh(y_pred - y_true)))
    
    Approximates L1 (MAE) at large errors but is smooth and twice-differentiable.
    Behaves like MSE for small errors, MAE for large errors.
    
    Benefits:
    - Smooth gradient everywhere (unlike L1).
    - Outlier-robust like L1 (unlike L2/MSE).
    - Numerically stable: uses log-cosh formulation to avoid overflow.
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize Log-Cosh loss.

        Args:
            reduction: 'mean', 'sum', or 'none' — how to aggregate.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Log-Cosh loss.

        Args:
            y_pred: Predicted values (batch, time, features).
            y_true: Ground truth values (batch, time, features).

        Returns:
            Scalar loss or per-element loss depending on reduction.
        """
        residual = y_pred - y_true
        # log(cosh(x)) = log((exp(x) + exp(-x)) / 2)
        #              = log(exp(|x|) * (1 + exp(-2|x|)) / 2)
        #              = |x| + log((1 + exp(-2|x|)) / 2)
        # For numerical stability, use:
        # log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
        # But simplest stable form:
        log_cosh = residual + torch.nn.functional.softplus(-2.0 * residual) - torch.log(
            torch.tensor(2.0, device=residual.device, dtype=residual.dtype)
        )

        if self.reduction == "mean":
            return log_cosh.mean()
        elif self.reduction == "sum":
            return log_cosh.sum()
        else:  # 'none'
            return log_cosh
