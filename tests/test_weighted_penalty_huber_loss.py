# file: tests/test_weighted_penalty_huber_loss.py

import torch
import pytest
from views_r2darts2.utils.loss import WeightedPenaltyHuberLoss, WeightedHuberLoss

# --- Unit Tests for WeightedPenaltyHuberLoss ---


@pytest.fixture
def loss_params():
    """Default parameters for the loss function."""
    return {
        "delta": 1.0,
        "zero_threshold": 0.1,
        "non_zero_weight": 5.0,
        "false_positive_weight": 2.0,
        "false_negative_weight": 3.0,
    }


@pytest.mark.parametrize(
    "case, pred_val, target_val, expected_loss",
    [
        # Case 1: True Negative (TN) - loss should be 0
        ("TN", 0.05, 0.05, 0.0),
        # Case 2: True Positive (TP)
        # base_w=5. final_w=5. error=-0.1. huber=0.005. loss=0.025
        ("TP", 0.6, 0.5, 0.025),
        # Case 3: False Positive (FP)
        # base_w=1. final_w=2. error=-0.5. huber=0.125. loss=0.25
        ("FP", 0.5, 0.0, 0.25),
        # Case 4: False Negative (FN)
        # base_w=5. final_w=15. error=0.5. huber=0.125. loss=1.875
        ("FN", 0.0, 0.5, 1.875),
    ],
)
def test_weighted_penalty_huber_golden_values(
    case, pred_val, target_val, expected_loss, loss_params
):
    """Tests the forward pass of WeightedPenaltyHuberLoss against manually calculated golden values for each case."""
    loss_fn = WeightedPenaltyHuberLoss(**loss_params)
    preds = torch.tensor([pred_val])
    targets = torch.tensor([target_val])
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-6), (
        f"Failed on case: {case}"
    )


def test_invariant_reduction_to_weighted_huber(loss_params):
    """
    Tests that with penalty weights=1.0, the loss is identical to WeightedHuberLoss.
    """
    preds = torch.randn(10, 5)
    targets = torch.randn(10, 5)

    # Our implementation with neutral penalty weights
    loss_params["false_positive_weight"] = 1.0
    loss_params["false_negative_weight"] = 1.0
    loss_fn_penalty = WeightedPenaltyHuberLoss(**loss_params)
    loss_penalty = loss_fn_penalty(preds, targets)

    # WeightedHuberLoss reference implementation
    loss_fn_weighted = WeightedHuberLoss(
        delta=loss_params["delta"],
        zero_threshold=loss_params["zero_threshold"],
        non_zero_weight=loss_params["non_zero_weight"],
    )
    loss_weighted = loss_fn_weighted(preds, targets)

    assert torch.isclose(loss_penalty, loss_weighted, atol=1e-6)


def test_weighted_penalty_huber_gradient_check(loss_params):
    """
    Performs a gradient check for WeightedPenaltyHuberLoss.
    This is critical to ensure the .detach() call is working as intended.
    """
    from torch.autograd import gradcheck

    loss_fn = WeightedPenaltyHuberLoss(**loss_params)

    # Use double precision for gradcheck
    # These inputs will cover all 4 cases (TN, TP, FP, FN)
    preds = torch.tensor([0.05, 0.6, 0.5, 0.0], dtype=torch.double, requires_grad=True)
    targets = torch.tensor([0.05, 0.5, 0.0, 0.5], dtype=torch.double)

    assert gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)
