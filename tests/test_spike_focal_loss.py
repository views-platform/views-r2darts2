# file: tests/test_spike_focal_loss.py

import torch
import pytest
from views_r2darts2.utils.loss import SpikeFocalLoss

# --- Unit Tests for SpikeFocalLoss ---

def test_spike_focal_loss_init():
    """Tests the initialization of SpikeFocalLoss."""
    loss_fn = SpikeFocalLoss(alpha=0.7, gamma=1.5, spike_threshold=5.0)
    assert loss_fn.alpha == 0.7
    assert loss_fn.gamma == 1.5
    assert loss_fn.spike_threshold == 5.0

@pytest.mark.parametrize(
    "preds, targets, alpha, gamma, spike_threshold, expected_loss",
    [
        # Case 1: Non-spike, small error
        # error=0.1. is_spike=False. focal_w = (1-0.8)*(exp(-0.1))^2 = 0.2*0.8187 = 0.1637.
        # base_loss=0.01. final_loss = 0.1637 * 0.01 = 0.001637
        (torch.tensor([0.6]), torch.tensor([0.5]), 0.8, 2.0, 3.0, 0.001637),

        # Case 2: Spike, small error
        # error=0.1. is_spike=True. focal_w = 0.8*(1-exp(-0.1))^2 = 0.8*0.00905 = 0.00724.
        # base_loss=0.01. final_loss = 0.00724 * 0.01 = 0.0000724
        (torch.tensor([3.6]), torch.tensor([3.5]), 0.8, 2.0, 3.0, 0.0000724),

        # Case 3: Spike, large error
        # error=1.0. is_spike=True. focal_w = 0.8*(1-exp(-1.0))^2 = 0.8*0.3995 = 0.3196.
        # base_loss=1.0. final_loss = 0.3196
        (torch.tensor([4.5]), torch.tensor([3.5]), 0.8, 2.0, 3.0, 0.3196),

        # Case 4: Non-spike, large error
        # error=1.0. is_spike=False. focal_w = (1-0.8)*(exp(-1.0))^2 = 0.2*0.1353 = 0.02706.
        # base_loss=1.0. final_loss = 0.02706
        (torch.tensor([1.5]), torch.tensor([0.5]), 0.8, 2.0, 3.0, 0.02706),
        
        # Case 5: Zero error
        (torch.tensor([1.0]), torch.tensor([1.0]), 0.8, 2.0, 3.0, 0.0),
    ]
)
def test_spike_focal_loss_golden_values(preds, targets, alpha, gamma, spike_threshold, expected_loss):
    """Tests the forward pass of SpikeFocalLoss against manually calculated golden values."""
    loss_fn = SpikeFocalLoss(alpha=alpha, gamma=gamma, spike_threshold=spike_threshold)
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)


def test_spike_focal_loss_invariant_gamma_zero():
    """
    Tests that with gamma=0, the loss becomes a simple weighted MSE,
    where weight is `alpha` for spikes and `1-alpha` for non-spikes.
    """
    alpha = 0.8
    spike_threshold = 3.0
    loss_fn = SpikeFocalLoss(alpha=alpha, gamma=0.0, spike_threshold=spike_threshold)

    # Spike case
    preds_spike = torch.tensor([4.0])
    targets_spike = torch.tensor([3.5]) # is_spike = True
    loss_spike = loss_fn(preds_spike, targets_spike)
    expected_spike_loss = alpha * (0.5**2)
    assert torch.isclose(loss_spike, torch.tensor(expected_spike_loss))

    # Non-spike case
    preds_non_spike = torch.tensor([1.0])
    targets_non_spike = torch.tensor([0.5]) # is_spike = False
    loss_non_spike = loss_fn(preds_non_spike, targets_non_spike)
    expected_non_spike_loss = (1 - alpha) * (0.5**2)
    assert torch.isclose(loss_non_spike, torch.tensor(expected_non_spike_loss))


def test_spike_focal_loss_gradient_check():
    """
    Performs a gradient check for SpikeFocalLoss.
    """
    from torch.autograd import gradcheck

    loss_fn = SpikeFocalLoss(alpha=0.8, gamma=2.0, spike_threshold=3.0)
    
    # Use double precision for gradcheck
    preds = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    targets = torch.randn(2, 2, dtype=torch.double)

    assert gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)
