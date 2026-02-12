# file: tests/test_zero_inflated_loss.py

import torch
import pytest
from views_r2darts2.utils.loss import ZeroInflatedLoss

# --- Unit Tests for ZeroInflatedLoss ---


@pytest.fixture
def loss_params():
    """Default parameters for the loss function."""
    return {
        "zero_weight": 1.0,
        "count_weight": 1.0,
        "delta": 0.5,
        "zero_threshold": 0.01,
        "eps": 1e-8,
    }


def test_zero_inflated_loss_init(loss_params):
    """Tests the initialization of ZeroInflatedLoss."""
    loss_fn = ZeroInflatedLoss(**loss_params)
    assert loss_fn.zero_weight == loss_params["zero_weight"]
    assert loss_fn.count_weight == loss_params["count_weight"]
    assert loss_fn.delta == loss_params["delta"]
    assert loss_fn.threshold == loss_params["zero_threshold"]


@pytest.mark.parametrize(
    "pred_val, target_val, expected_loss",
    [
        # Case 1: Target is zero
        # p_zero = sig(-0.1)=0.475. is_zero=1. zero_loss = -log(0.475)=0.744. count_loss=0.
        # final = 1*0.744 + 1*0 = 0.744
        (0.01, 0.0, 0.7444),
        # Case 2: Target is non-zero, small error
        # p_zero = sig(-6)=0.00247. is_zero=0. zero_loss=-log(1-p_zero)=0.00247.
        # error=0.1. count_loss = 0.5^2*(sqrt(1+(0.1/0.5)^2)-1) = 0.00495
        # final = 1*0.00247 + 1*0.00495 = 0.00742
        (0.6, 0.5, 0.00742),
        # Case 3: Perfect prediction for non-zero target
        # p_zero = sig(-5)=0.00669. is_zero=0. zero_loss=-log(1-p_zero)=0.0067.
        # error=0. count_loss=0. final = 0.0067
        (0.5, 0.5, 0.0067),
    ],
)
def test_zero_inflated_loss_golden_values(
    pred_val, target_val, expected_loss, loss_params
):
    """Tests the forward pass of ZeroInflatedLoss against manually calculated golden values."""
    loss_fn = ZeroInflatedLoss(**loss_params)
    preds = torch.tensor([pred_val])
    targets = torch.tensor([target_val])
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)


def test_zero_inflated_loss_invariants(loss_params):
    """Tests the invariant properties of the loss function by isolating its components."""
    preds = torch.tensor([0.01, 0.6])
    targets = torch.tensor([0.0, 0.5])

    # --- Test zero component isolation ---
    loss_params["count_weight"] = 0.0
    loss_params["zero_weight"] = 1.0
    loss_fn_zero_only = ZeroInflatedLoss(**loss_params)

    # Manually calculate only the zero loss part
    p_zero = torch.sigmoid(-preds * 10).clamp(1e-8, 1.0 - 1e-8)
    is_zero = (torch.abs(targets) < 0.01).float()
    expected_zero_loss = torch.nn.functional.binary_cross_entropy(
        p_zero, is_zero, reduction="mean"
    )

    loss = loss_fn_zero_only(preds, targets)
    assert torch.isclose(loss, expected_zero_loss)

    # --- Test count component isolation ---
    loss_params["count_weight"] = 1.0
    loss_params["zero_weight"] = 0.0
    loss_fn_count_only = ZeroInflatedLoss(**loss_params)

    # Manually calculate only the count loss part
    is_zero = (torch.abs(targets) < 0.01).float()
    count_mask = 1.0 - is_zero
    errors = (preds - targets) * count_mask
    delta = loss_params["delta"]
    expected_count_loss = torch.mean(
        delta**2 * (torch.sqrt(1 + (errors / delta) ** 2) - 1)
    )

    loss = loss_fn_count_only(preds, targets)
    assert torch.isclose(loss, expected_count_loss)


def test_zero_inflated_loss_gradient_check(loss_params):
    """Performs a gradient check for ZeroInflatedLoss."""
    from torch.autograd import gradcheck

    loss_fn = ZeroInflatedLoss(**loss_params)

    # Use double precision for gradcheck
    preds = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    targets = torch.rand(2, 2, dtype=torch.double)  # Use rand to get non-zero targets

    assert gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)
