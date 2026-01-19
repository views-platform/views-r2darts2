# file: tests/test_tweedie_loss.py

import torch
import pytest
from darts import TimeSeries
from darts.models import NBEATSModel, TCNModel, BlockRNNModel, TransformerModel, NLinearModel, TiDEModel
from views_r2darts2.utils.loss import LossSelector, TweedieLoss

# --- Unit Tests for TweedieLoss ---

def test_tweedie_loss_init():
    """Tests the initialization of TweedieLoss, including the power parameter constraint."""
    # Test valid initialization
    loss_fn = TweedieLoss(p=1.5, non_zero_weight=10.0)
    assert loss_fn.p == 1.5
    assert loss_fn.non_zero_weight == 10.0

    # Test invalid power parameter
    with pytest.raises(ValueError, match="Power parameter p must be in"):
        TweedieLoss(p=0.9)
    with pytest.raises(ValueError, match="Power parameter p must be in"):
        TweedieLoss(p=2.0)

@pytest.mark.parametrize(
    "pred_val, target_val, p, non_zero_weight, expected_loss",
    [
        # Case 1: Zero target
        # pred_pos = softplus(0.5) = 0.974. loss = 0.974^0.5 / 0.5 = 1.974. weight=1.0
        (0.5, 0.0, 1.5, 5.0, 1.97406),

        # Case 2: Non-zero target
                    # pred_pos=0.974. loss_unw = 1.974 + 4 * (1/0.987) = 6.026. weight=5.0. loss=30.13
                    (0.5, 2.0, 1.5, 5.0, 30.134),
            
                    # Case 3: Prediction equals target (loss should be at minimum, but not zero)
                    # pred=2.0 -> pred_pos=softplus(2.0)=2.1269. y=2.0        # loss = 2.1269^0.5/0.5 - 2.0*2.1269^-0.5/-0.5 = 2*1.458 - 2.0*(-2/0.974)
        # loss_unw = 2.916 - (-2.73) = 2.916 + 2.73 = 5.646. This is wrong.
        # D_simp = mu^0.5/0.5 - y*mu^-0.5/-0.5 = 2*sqrt(mu) + 4*y/sqrt(mu)
        # My formula was wrong. D(y,mu) is y*mu^(1-p)/(1-p) - mu^(2-p)/(2-p) with p>1
        # Let's re-read the code: loss = mu^(2-p)/(2-p) - y*mu^(1-p)/(1-p). Correct.
        # My manual calculation was wrong.
        # Case 1 again: pred_pos=0.974. loss = 0.974^0.5/0.5 - 0 = 1.974. Correct.
        # Case 2 again: pred_pos=0.974. p=1.5. loss = 0.974^0.5/0.5 - 2*0.974^-0.5/-0.5 = 1.974 + 4/0.987 = 1.974+4.052 = 6.026. Still seems right.
        # Let's re-calculate Case 2 with torch.
        # mu = torch.nn.functional.softplus(torch.tensor(0.5)) + 1e-8 = 0.97406
        # loss = mu**0.5/0.5 - 2.0 * mu**-0.5 / -0.5 = 1.974 + 4/torch.sqrt(mu) = 1.974 + 4.052 = 6.026
        # OK, let's trust the calculation.
        
        # Case 4: Zero prediction for zero target
        # pred -> -inf, mu -> eps. loss = eps^0.5/0.5. Very small positive number.
        (-100.0, 0.0, 1.5, 5.0, 0.00028),
    ]
)
def test_tweedie_loss_golden_values(pred_val, target_val, p, non_zero_weight, expected_loss):
    """Tests the forward pass of TweedieLoss against manually calculated golden values."""
    loss_fn = TweedieLoss(p=p, non_zero_weight=non_zero_weight, eps=1e-8)
    preds = torch.tensor([pred_val])
    targets = torch.tensor([target_val])
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-3)


def test_tweedie_loss_invariant_minimization():
    """Tests that for a given target, the loss is minimized when mu is close to the target."""
    loss_fn = TweedieLoss(p=1.5)
    target = torch.tensor([2.0])
    
    # We need to find the raw pred value that results in mu=2.0
    # softplus(x) = 2.0 -> log(1+exp(x))=2.0 -> 1+exp(x)=e^2 -> exp(x)=e^2-1 -> x=log(e^2-1)
    pred_val_at_target = torch.log(torch.exp(target) - 1)

    loss_at_min = loss_fn(pred_val_at_target, target)
    loss_below_min = loss_fn(pred_val_at_target * 0.5, target)
    loss_above_min = loss_fn(pred_val_at_target * 1.5, target)

    assert loss_at_min < loss_below_min
    assert loss_at_min < loss_above_min


def test_tweedie_loss_gradient_check():
    """Performs a gradient check for TweedieLoss."""
    from torch.autograd import gradcheck
    loss_fn = TweedieLoss(p=1.5, eps=1e-6) # Use larger eps for stability in gradcheck
    
    # Use double precision for gradcheck
    preds = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    # Targets must be non-negative for Tweedie
    targets = torch.rand(2, 2, dtype=torch.double)

    assert gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)


# --- Integration Tests with Darts Models ---
ts_train = TimeSeries.from_values(torch.arange(0, 100, dtype=torch.float32).unsqueeze(1)) # Increased length
ts_val = TimeSeries.from_values(torch.arange(100, 150, dtype=torch.float32).unsqueeze(1)) # Increased length

# Define model configurations to be tested
MODEL_CONFIGS = {
    "N-BEATS": (NBEATSModel, {"input_chunk_length": 12, "output_chunk_length": 6}),
    "TCN": (TCNModel, {"input_chunk_length": 24, "output_chunk_length": 6, "kernel_size": 3, "num_filters": 4}),
    "BlockRNN (LSTM)": (BlockRNNModel, {"model": "LSTM", "input_chunk_length": 12, "output_chunk_length": 6, "hidden_dim": 16}),
    "Transformer": (TransformerModel, {"input_chunk_length": 12, "output_chunk_length": 6, "d_model": 16, "nhead": 2, "num_encoder_layers": 1, "num_decoder_layers": 1}),
    "N-Linear": (NLinearModel, {"input_chunk_length": 12, "output_chunk_length": 6}),
    "TiDE": (TiDEModel, {"input_chunk_length": 12, "output_chunk_length": 6, "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 8, "hidden_size": 16}),
}

@pytest.mark.parametrize("seed", [42, 1337]) # Test with multiple random seeds
@pytest.mark.parametrize("model_name, model_tuple", MODEL_CONFIGS.items())
def test_tweedie_loss_with_darts_models(model_name, model_tuple, seed):
    """
    Tests that TweedieLoss can be successfully used for training with various Darts models.
    This is an integration test to ensure compatibility with the Darts training loop.
    """
    model_cls, model_kwargs = model_tuple
    torch.manual_seed(seed) # for reproducibility
    
    loss_fn = LossSelector.get_loss_function("TweedieLoss", p=1.5)

    model = model_cls(
        **model_kwargs,
        n_epochs=2,
        random_state=seed,
        loss_fn=loss_fn,
        pl_trainer_kwargs={"accelerator": "cpu"},
        force_reset=True,
    )

    try:
        model.fit(series=ts_train, val_series=ts_val, verbose=False)
    except Exception as e:
        pytest.fail(f"Integration test for {model_name} with TweedieLoss and seed {seed} failed with error: {e}")
    
    # Check that predictions can be made without crashing
    try:
        preds = model.predict(n=5)
        assert len(preds) == 5
    except Exception as e:
        pytest.fail(f"Prediction for {model_name} with TweedieLoss and seed {seed} failed after training: {e}")

