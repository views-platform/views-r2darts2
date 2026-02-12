# file: tests/test_time_aware_weighted_huber_loss.py

import torch
from views_r2darts2.utils.loss import TimeAwareWeightedHuberLoss

# --- Unit Tests for TimeAwareWeightedHuberLoss ---


def test_time_aware_weighted_huber_loss_init():
    """Tests the initialization of TimeAwareWeightedHuberLoss."""
    loss_fn = TimeAwareWeightedHuberLoss(
        zero_weight=1.0, non_zero_weight=10.0, decay_factor=0.9, delta=0.5
    )
    assert loss_fn.zero_weight == 1.0
    assert loss_fn.non_zero_weight == 10.0
    assert loss_fn.decay_factor == 0.9
    assert loss_fn.delta == 0.5


def test_time_aware_weighted_huber_loss_golden_value():
    """
    Tests the forward pass of TimeAwareWeightedHuberLoss against a manually calculated golden value.
    """
    loss_fn = TimeAwareWeightedHuberLoss(
        zero_weight=1.0, non_zero_weight=10.0, decay_factor=0.9, delta=1.0
    )

    # Shape: (batch_size=1, seq_len=3)
    preds = torch.tensor([[0.1, 0.6, 2.0]])
    targets = torch.tensor([[0.0, 0.5, 0.0]])

    # Manual calculation based on LSS and implementation:
    # Implementation: time_weights = [decay**(seq-i) for i in range(seq)]
    # seq_len = 3. i = 0, 1, 2.
    # Exponents will be 3, 2, 1. Not 2, 1, 0. This seems wrong.
    # Let's verify the implementation's logic.
    # time_weights should be [0.9^3, 0.9^2, 0.9^1] = [0.729, 0.81, 0.9]
    # Let's assume the LSS had the right INTENT and the implementation has a bug.
    # The LSS states: `decay_factor ** (seq_len - 1 - i)` seems more correct.
    # This would give weights [0.9^2, 0.9^1, 0.9^0] = [0.81, 0.9, 1.0].
    # Let's calculate with the implementation's logic first to see if it matches a failure.

    # Calculation per implementation: time_weights = [0.729, 0.81, 0.9]
    # Step 1: pred=0.1, target=0.0. error=0.1. huber=0.005. event_w=1.0. time_w=0.729. loss=0.003645
    # Step 2: pred=0.6, target=0.5. error=0.1. huber=0.005. event_w=10.0. time_w=0.81. loss=0.0405
    # Step 3: pred=2.0, target=0.0. error=2.0. huber=1.5. event_w=1.0. time_w=0.9. loss=1.35
    # mean = (0.003645 + 0.0405 + 1.35) / 3 = 1.394145 / 3 = 0.464715

    # Let's assume the implementation is buggy and fix it mentally.
    # The time weights should give more weight to recent events.
    # The current implementation `decay_factor ** (seq_len - i)` gives LESS weight to recent events.
    # Let's assume it should be `decay_factor ** i` (simplest form, weight increases over time).
    # time_weights = [0.9^0, 0.9^1, 0.9^2] = [1.0, 0.9, 0.81]
    # Let's re-calculate:
    # Step 1: loss = 1.0 * 1.0 * 0.005 = 0.005
    # Step 2: loss = 10.0 * 0.9 * 0.005 = 0.045
    # Step 3: loss = 1.0 * 0.81 * 1.5 = 1.215
    # mean = (0.005 + 0.045 + 1.215) / 3 = 1.265 / 3 = 0.42166...
    # This is also probably not right. The most recent should have the highest weight.

    # Let's stick to the LSS proposal: `decay_factor ** (seq_len - 1 - i)`
    # Weights: [0.81, 0.9, 1.0]
    # Step 1: loss = 1.0 * 0.81 * 0.005 = 0.00405
    # Step 2: loss = 10.0 * 0.9 * 0.005 = 0.045
    # Step 3: loss = 1.0 * 1.0 * 1.5 = 1.5
    # mean = (0.00405 + 0.045 + 1.5) / 3 = 1.54905 / 3 = 0.51635

    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(0.51635), atol=1e-5), (
        "Golden value test failed. This may indicate the time decay is implemented incorrectly."
    )


def test_time_aware_huber_invariant_vs_torch():
    """
    Tests that TimeAwareWeightedHuberLoss with all weights=1.0 is identical
    to the standard torch.nn.HuberLoss.
    """
    preds = torch.randn(10, 5, 2)
    targets = torch.randn(10, 5, 2)
    delta = 0.75

    # Our implementation with neutral weights
    loss_fn_custom = TimeAwareWeightedHuberLoss(
        delta=delta, decay_factor=1.0, zero_weight=1.0, non_zero_weight=1.0
    )
    loss_custom = loss_fn_custom(preds, targets)

    # PyTorch reference implementation
    loss_fn_torch = torch.nn.HuberLoss(delta=delta, reduction="mean")
    loss_torch = loss_fn_torch(preds, targets)

    assert torch.isclose(loss_custom, loss_torch, atol=1e-6)


def test_time_aware_huber_gradient_check():
    """
    Performs a gradient check for TimeAwareWeightedHuberLoss.
    """
    from torch.autograd import gradcheck

    # Use double precision for gradcheck
    loss_fn = TimeAwareWeightedHuberLoss(
        delta=0.5, decay_factor=0.9, zero_weight=1.0, non_zero_weight=10.0
    )

    # Test with typical inputs
    preds = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    targets = torch.randn(2, 3, dtype=torch.double)

    assert gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)
