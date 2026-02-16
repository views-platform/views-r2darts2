import torch
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from views_r2darts2.utils.scaling import ScalerSelector

@dataclass
class LossSpec:
    """Standardized metadata for a loss function under test."""
    name: str
    params: Dict[str, Any]
    requires_positivity: bool = False
    requires_counts: bool = False
    is_scale_aware: bool = True
    uses_thresholds: bool = False

def make_fake_views_batch(
    batch_size: int = 32,
    seq_len: int = 36,
    num_targets: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    sparsity: float = 0.8,
    max_val: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a realistic (B, T, K) batch matching Darts/Fortress model outputs.
    Sparsity parameter simulates the zero-inflation typical of VIEWS conflict data.
    """
    # Targets: Zero-inflated counts
    targets = torch.rand(batch_size, seq_len, num_targets, device=device, dtype=dtype) * max_val
    mask = torch.rand(batch_size, seq_len, num_targets, device=device, dtype=dtype) > sparsity
    targets = targets * mask.to(dtype)
    
    # Predictions: targets + Gaussian noise
    noise = torch.randn_like(targets) * (max_val * 0.1)
    preds = targets + noise
    
    return preds, targets

class LossIntegrityHarness:
    """
    Unified environment for testing losses under realistic modeling conditions.
    Enforces the 'Fail Loud' mandate during diagnostics.
    """
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

    def simulate_pipeline(self, targets: torch.Tensor, scaler_name: str) -> torch.Tensor:
        """Applies a simulated scaler transform to raw targets."""
        arr = targets.detach().cpu().numpy().reshape(-1, 1)
        scaler = ScalerSelector.get_scaler(scaler_name)
        transformed = scaler.fit_transform(arr)
        return torch.tensor(transformed.reshape(targets.shape), device=self.device, dtype=self.dtype)

    def run_gradcheck(self, loss_fn: torch.nn.Module, preds: torch.Tensor, targets: torch.Tensor) -> bool:
        """Performs high-precision gradient check on float64."""
        from torch.autograd import gradcheck
        p = preds.detach().clone().to(torch.float64).requires_grad_(True)
        t = targets.detach().clone().to(torch.float64)
        return gradcheck(lambda x: loss_fn(x, t), (p,), eps=1e-6, atol=1e-4)
