# file: tests/test_asymmetric_quantile_loss.py

import torch
import pytest
from views_r2darts2.utils.loss.quantile import AsymmetricQuantileLoss

# --- Unit Tests for AsymmetricQuantileLoss ---


def test_asymmetric_quantile_loss_init():
    """Tests the initialization of AsymmetricQuantileLoss, including the tau constraint."""
    # Test valid initialization
    loss_fn = AsymmetricQuantileLoss(tau=0.8, non_zero_weight=10.0, zero_threshold=0.01)
    assert loss_fn.tau == 0.8
    assert loss_fn.non_zero_weight == 10.0

    # Test invalid tau parameter
    with pytest.raises(ValueError, match="tau must be in"):
        AsymmetricQuantileLoss(tau=0.0, non_zero_weight=5.0, zero_threshold=0.01)
    with pytest.raises(ValueError, match="tau must be in"):
        AsymmetricQuantileLoss(tau=1.0, non_zero_weight=5.0, zero_threshold=0.01)
    with pytest.raises(ValueError, match="tau must be in"):
        AsymmetricQuantileLoss(tau=-0.1, non_zero_weight=5.0, zero_threshold=0.01)


@pytest.mark.parametrize(
    "pred_val, target_val, tau, non_zero_weight, expected_loss",
    [
        # Case 1: Underestimation (error > 0), non-zero target
        # error=0.5. loss_unw=0.75*0.5=0.375. weight=5.0. final=1.875
        (0.5, 1.0, 0.75, 5.0, 1.875),
        # Case 2: Overestimation (error < 0), non-zero target
        # error=-0.5. loss_unw=(0.75-1)*-0.5=0.125. weight=5.0. final=0.625
        (1.5, 1.0, 0.75, 5.0, 0.625),
        # Case 3: Overestimation, zero target
        # error=-0.5. loss_unw=0.125. weight=1.0. final=0.125
        (0.5, 0.0, 0.75, 5.0, 0.125),
        # Case 4: Perfect prediction
        (1.0, 1.0, 0.75, 5.0, 0.0),
    ],
)
def test_asymmetric_quantile_loss_golden_values(
    pred_val, target_val, tau, non_zero_weight, expected_loss
):
    """Tests the forward pass of AsymmetricQuantileLoss against manually calculated golden values."""
    loss_fn = AsymmetricQuantileLoss(
        tau=tau, non_zero_weight=non_zero_weight, zero_threshold=0.1
    )
    preds = torch.tensor([pred_val])
    targets = torch.tensor([target_val])
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-6)


def test_asymmetric_quantile_loss_invariant_mae():
    """
    Tests that with tau=0.5, the loss is 0.5 * MAE.
    """
    preds = torch.randn(10, 5)
    targets = torch.randn(10, 5)

    # Our implementation with tau=0.5 and no extra weighting
    loss_fn_quantile = AsymmetricQuantileLoss(
        tau=0.5, non_zero_weight=1.0, zero_threshold=0.01
    )
    loss_quantile = loss_fn_quantile(preds, targets)

    # PyTorch L1Loss (MAE)
    loss_fn_mae = torch.nn.L1Loss(reduction="mean")
    loss_mae = loss_fn_mae(preds, targets)

    assert torch.isclose(loss_quantile, 0.5 * loss_mae, atol=1e-6)


def test_asymmetric_quantile_loss_gradient_check():
    """
    Performs a gradient check for AsymmetricQuantileLoss.
    The gradient is discontinuous at error=0, so we test on either side.
    """
    from torch.autograd import gradcheck

    loss_fn = AsymmetricQuantileLoss(tau=0.8, non_zero_weight=5.0, zero_threshold=0.01)

    # Use double precision for gradcheck
    # Test with errors > 0
    preds_under = torch.tensor([0.1, 0.2], dtype=torch.double, requires_grad=True)
    targets_under = torch.tensor([0.5, 0.8], dtype=torch.double)
    assert gradcheck(loss_fn, (preds_under, targets_under), eps=1e-6, atol=1e-4)

    # Test with errors < 0
    preds_over = torch.tensor([0.9, 1.2], dtype=torch.double, requires_grad=True)
    targets_over = torch.tensor([0.5, 0.8], dtype=torch.double)
    assert gradcheck(loss_fn, (preds_over, targets_over), eps=1e-6, atol=1e-4)
