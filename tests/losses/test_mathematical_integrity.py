import pytest
import torch
from tests.losses.harness import LossSpec, LossIntegrityHarness, make_fake_views_batch
from views_r2darts2.utils.loss import LossCatalog
from views_r2darts2.utils.gates import NumericalSanityError

# 1. Decision: Deciding on Canonical Test Configuration
HARNESS = LossIntegrityHarness(device="cpu", dtype=torch.float32)

# 2. Section 1 Deliverable: Assumption & Failure Modes Specs
LOSS_SPECS = [
    LossSpec(
        name="WeightedPenaltyHuberLoss",
        params={
            "zero_threshold": 0.05, "delta": 0.5, "non_zero_weight": 5.0, 
            "false_positive_weight": 2.0, "false_negative_weight": 3.0
        },
        uses_thresholds=True
    ),
    LossSpec(
        name="TweedieLoss",
        params={
            "p": 1.5, "non_zero_weight": 5.0, "zero_threshold": 0.01, 
            "false_positive_weight": 1.0, "false_negative_weight": 1.0, "eps": 1e-8
        },
        requires_positivity=True,
        requires_counts=True # Ideally count-scale or similar
    ),
    LossSpec(
        name="ShrinkageLoss",
        params={"a": 10.0, "c": 0.2}
    ),
    LossSpec(
        name="WeightedHuberLoss",
        params={"zero_threshold": 0.01, "delta": 0.5, "non_zero_weight": 5.0},
        uses_thresholds=True
    ),
    LossSpec(
        name="TimeAwareWeightedHuberLoss",
        params={"zero_weight": 1.0, "non_zero_weight": 5.0, "decay_factor": 0.9, "delta": 0.5}
    ),
    LossSpec(
        name="SpikeFocalLoss",
        params={"alpha": 0.8, "gamma": 2.0, "spike_threshold": 3.0}
    ),
    LossSpec(
        name="AsymmetricQuantileLoss",
        params={"tau": 0.75, "non_zero_weight": 5.0, "zero_threshold": 0.01},
        uses_thresholds=True
    ),
    LossSpec(
        name="ZeroInflatedLoss",
        params={"zero_weight": 1.0, "count_weight": 1.0, "delta": 0.5, "zero_threshold": 0.01, "eps": 1e-8},
        uses_thresholds=True
    )
]

@pytest.mark.parametrize("spec", LOSS_SPECS)
def test_loss_forward_pass_integrity(spec):
    """Section 2: Forward pass on typical shapes and edge cases."""
    config = {**spec.params, "loss_function": spec.name}
    loss_fn = LossCatalog(config).get_loss()
    
    # Test typical VIEWS shape (B, T, 1)
    preds, targets = make_fake_views_batch(batch_size=8, seq_len=12)
    
    # Tweedie needs strictly non-negative targets
    if spec.requires_positivity:
        targets = torch.clamp(targets, min=0.0)
        
    loss = loss_fn(preds, targets)
    
    assert torch.is_tensor(loss)
    assert loss.dim() == 0 # Must be scalar
    assert torch.isfinite(loss)
    assert loss >= 0

@pytest.mark.parametrize("spec", LOSS_SPECS)
def test_loss_gradient_flow(spec):
    """Section 3: Gradient correctness and existence."""
    config = {**spec.params, "loss_function": spec.name}
    loss_fn = LossCatalog(config).get_loss()
    
    # Small batch for gradcheck
    preds, targets = make_fake_views_batch(batch_size=2, seq_len=4)
    if spec.requires_positivity:
        targets = torch.clamp(targets, min=0.0)
        
    # Standard gradcheck
    assert HARNESS.run_gradcheck(loss_fn, preds, targets)

@pytest.mark.parametrize("spec", LOSS_SPECS)
def test_numerical_stability_stress(spec):
    """Section 5: Stress test on extreme values and spikes."""
    config = {**spec.params, "loss_function": spec.name}
    loss_fn = LossCatalog(config).get_loss()
    
    # Test Case: Extreme Spikes (Mass Atrocity simulation)
    # 10,000 fatalities in transformed space can be large depending on scaler
    huge_val = 1e6
    preds = torch.tensor([[[huge_val]]], dtype=torch.float32)
    targets = torch.tensor([[[huge_val]]], dtype=torch.float32)
    
    loss = loss_fn(preds, targets)
    assert torch.isfinite(loss)
    
    # Test Case: Negative Inputs (Model Instability simulation)
    # Pre-softplus/transform values can be negative
    neg_preds = torch.tensor([[[-huge_val]]], dtype=torch.float32)
    loss_neg = loss_fn(neg_preds, targets)
    assert torch.isfinite(loss_neg)

@pytest.mark.parametrize("spec", LOSS_SPECS)
def test_fail_loud_on_nans(spec):
    """Red Team: Verify explicit failure on NaN injection."""
    config = {**spec.params, "loss_function": spec.name}
    loss_fn = LossCatalog(config).get_loss()
    
    preds, targets = make_fake_views_batch(batch_size=1, seq_len=1)
    preds[0, 0, 0] = float('nan')
    
    with pytest.raises(NumericalSanityError):
        loss_fn(preds, targets)
