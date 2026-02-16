# file: tests/test_weighted_huber_loss.py

import torch
import pytest
from views_r2darts2.utils.loss.weighted_huber_loss import WeightedHuberLoss

# --- Unit Tests for WeightedHuberLoss ---


def test_weighted_huber_loss_init():
    """Tests the initialization of WeightedHuberLoss with default and custom parameters."""
    # Test custom initialization
    loss_fn_custom = WeightedHuberLoss(
        zero_threshold=0.1, delta=1.5, non_zero_weight=20.0
    )
    assert loss_fn_custom.threshold == 0.1
    assert loss_fn_custom.delta == 1.5
    assert loss_fn_custom.non_zero_weight == 20.0


@pytest.mark.parametrize(
    "preds, targets, zero_threshold, delta, non_zero_weight, expected_loss",
    [
        # Case 1: Zero target, small error (quadratic part of Huber)
        # error=0.1, huber_loss = 0.5 * 0.1^2 = 0.005. weight = 1.0. final_loss = 0.005
        (torch.tensor([0.1]), torch.tensor([0.0]), 0.01, 1.0, 10.0, 0.005),
        # Case 2: Non-zero target, small error (quadratic part of Huber)
        # error=0.1, huber_loss = 0.5 * 0.1^2 = 0.005. weight = 10.0. final_loss = 0.05
        (torch.tensor([0.6]), torch.tensor([0.5]), 0.01, 1.0, 10.0, 0.05),
        # Case 3: Zero target, large error (linear part of Huber)
        # error=1.5, huber_loss = 1.0 * (|1.5| - 0.5 * 1.0) = 1.0. weight = 1.0. final_loss = 1.0
        (
            torch.tensor([2.0]),
            torch.tensor([0.5]),
            1.0,
            1.0,
            10.0,
            1.0,
        ),  # target < threshold
        # Case 4: Non-zero target, large error (linear part of Huber)
        # error=1.5, huber_loss = 1.0 * (|1.5| - 0.5 * 1.0) = 1.0. weight = 10.0. final_loss = 10.0
        (
            torch.tensor([2.0]),
            torch.tensor([0.5]),
            0.01,
            1.0,
            10.0,
            10.0,
        ),  # target > threshold
        # Case 5: Batch with mixed conditions
        # Item 1: non-zero target, small error. loss = 10.0 * (0.5 * 0.1^2) = 0.05
        # Item 2: zero target, large error. loss = 1.0 * (1.0 * (|2.0| - 0.5 * 1.0)) = 1.5
        # Expected mean loss = (0.05 + 1.5) / 2 = 0.775
        (torch.tensor([0.6, 2.0]), torch.tensor([0.5, 0.0]), 0.01, 1.0, 10.0, 0.775),
    ],
)
def test_weighted_huber_loss_golden_values(
    preds, targets, zero_threshold, delta, non_zero_weight, expected_loss
):
    """Tests the forward pass of WeightedHuberLoss against manually calculated golden values."""
    loss_fn = WeightedHuberLoss(
        zero_threshold=zero_threshold, delta=delta, non_zero_weight=non_zero_weight
    )
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-6)


def test_weighted_huber_loss_invariant_vs_torch():
    """
    Tests that WeightedHuberLoss with non_zero_weight=1.0 is identical to
    the standard torch.nn.HuberLoss.
    """
    preds = torch.randn(10, 5)
    targets = torch.randn(10, 5)
    delta = 0.75

    # Our implementation
    loss_fn_custom = WeightedHuberLoss(
        zero_threshold=0.01, delta=delta, non_zero_weight=1.0
    )
    loss_custom = loss_fn_custom(preds, targets)

    # PyTorch reference implementation
    loss_fn_torch = torch.nn.HuberLoss(delta=delta, reduction="mean")
    loss_torch = loss_fn_torch(preds, targets)

    assert torch.isclose(loss_custom, loss_torch, atol=1e-6)


def test_weighted_huber_loss_gradient_check():
    """
    Performs a gradient check for WeightedHuberLoss using torch.autograd.gradcheck.
    """
    from torch.autograd import gradcheck

    # Use double precision for gradcheck
    delta = 0.5
    loss_fn = WeightedHuberLoss(zero_threshold=0.01, delta=delta, non_zero_weight=10.0)

    # Test away from the delta threshold
    preds = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    targets = torch.randn(2, 2, dtype=torch.double)
    assert gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)

    # The gradient is not defined at the boundary (|error|=delta).
    # `gradcheck` will fail here as expected. We need to test on either side.
    # We will test only points where the gradient is defined.

    # Test points strictly within the quadratic part
    preds_quad = torch.tensor([0.1, -0.2], dtype=torch.double, requires_grad=True)
    targets_quad = torch.zeros(2, dtype=torch.double)
    assert gradcheck(loss_fn, (preds_quad, targets_quad), eps=1e-6, atol=1e-4)

    # Test points strictly within the linear part
    preds_linear = torch.tensor([1.0, -1.5], dtype=torch.double, requires_grad=True)
    targets_linear = torch.zeros(2, dtype=torch.double)
    assert gradcheck(loss_fn, (preds_linear, targets_linear), eps=1e-6, atol=1e-4)
