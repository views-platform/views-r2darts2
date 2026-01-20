# file: tests/test_shrinkage_loss.py

import torch
import numpy as np
import pytest
from views_r2darts2.utils.loss import ShrinkageLoss

# --- Unit Tests for ShrinkageLoss ---

def test_shrinkage_loss_init():
    """Tests the initialization of ShrinkageLoss with default and custom parameters."""
    # Test default initialization
    loss_fn_default = ShrinkageLoss()
    assert loss_fn_default.a == 10.0
    assert loss_fn_default.c == 0.2

    # Test custom initialization
    loss_fn_custom = ShrinkageLoss(a=5.0, c=0.5)
    assert loss_fn_custom.a == 5.0
    assert loss_fn_custom.c == 0.5


@pytest.mark.parametrize(
    "preds, targets, a, c, expected_loss",
    [
        # Test case 1: Zero error, loss should be zero
        (torch.tensor([0.5]), torch.tensor([0.5]), 10.0, 0.2, 0.0),

        # Test case 2: Small error (l < c), loss should be heavily shrunk
        # l = 0.1, shrinkage_factor = 3.718, base_loss = 0.01
        # expected_loss = 0.01 / 3.718 = 0.002689
        (torch.tensor([0.5]), torch.tensor([0.4]), 10.0, 0.2, 0.002689),

        # Test case 3: Error equals threshold (l = c), shrinkage should be moderate
        # l = 0.2, shrinkage_factor = 2.0, base_loss = 0.04
        # expected_loss = 0.04 / 2.0 = 0.02
        (torch.tensor([0.7]), torch.tensor([0.5]), 10.0, 0.2, 0.02),

        # Test case 4: Large error (l > c), shrinkage should be minimal
        # l = 0.4, shrinkage_factor = 1.135, base_loss = 0.16
        # expected_loss = 0.16 / 1.135 = 0.1409
        (torch.tensor([0.9]), torch.tensor([0.5]), 10.0, 0.2, 0.1409),

        # Test case 5: High 'a' value, very strong shrinkage
        # l = 0.1, shrinkage_factor = 149.41, base_loss = 0.01
        # expected_loss = 0.01 / 149.41 = 0.000067
        (torch.tensor([0.5]), torch.tensor([0.4]), 50.0, 0.2, 0.000067),
        
        # Test case 6: Edge case with zero target
        # l = 0.1, shrinkage_factor = 3.718, base_loss = 0.01
        # expected_loss = 0.01 / 3.718 = 0.002689
        (torch.tensor([0.1]), torch.tensor([0.0]), 10.0, 0.2, 0.002689),

        # Golden value: Easy sample (error l=0.01 is much smaller than c=0.2)
        # l = 0.01, shrinkage_factor = 6.686, base_loss = 0.0001
        # expected_loss = 0.0001 / 6.686 = 0.0000149
        (torch.tensor([0.1]), torch.tensor([0.11]), 10.0, 0.2, 0.0000149),

        # Golden value: Hard sample (error l=0.4 is much larger than c=0.2)
        # l = 0.4, shrinkage_factor = 1.135, base_loss = 0.16
        # expected_loss = 0.16 / 1.135 = 0.1409
        (torch.tensor([0.1]), torch.tensor([0.5]), 10.0, 0.2, 0.1409),
    ],
)
def test_shrinkage_loss_forward_calculation(preds, targets, a, c, expected_loss):
    """
    Tests the forward pass of ShrinkageLoss against manually calculated values.
    """
    loss_fn = ShrinkageLoss(a=a, c=c)
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)





@pytest.mark.skip(reason="Plotting is a manual verification step, not for automated testing.")
def test_plot_shrinkage_loss_landscape():
    """
    Generates a plot to visualize the behavior of the ShrinkageLoss function
    compared to MSE. This is a visual aid for understanding the loss landscape.
    """
    import matplotlib.pyplot as plt

    a = 10.0
    c = 0.2
    loss_fn = ShrinkageLoss(a=a, c=c)
    mse_fn = torch.nn.MSELoss()

    # Define a range of prediction errors
    errors = torch.linspace(-1.0, 1.0, 400)
    target_val = 0.5 # Assume a constant target for visualization
    
    # Create dummy tensors for preds and targets
    targets = torch.full_like(errors, fill_value=target_val)
    preds = targets + errors

    # Calculate losses
    shrinkage_losses = loss_fn(preds, targets)
    
    # Recalculate manually to isolate components for plotting
    l = torch.abs(preds - targets)
    shrinkage_factor = 1 + torch.exp(a * (c - l))
    base_loss = torch.exp(targets) * l**2
    manual_shrinkage_losses = base_loss / shrinkage_factor

    mse_losses = mse_fn(preds, targets)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(errors.numpy(), manual_shrinkage_losses.numpy(), label=f'Shrinkage Loss (a={a}, c={c})', color='blue', linewidth=2)
    plt.plot(errors.numpy(), mse_losses.numpy(), label='MSE Loss', color='red', linestyle='--', linewidth=2)
    
    plt.title('Shrinkage Loss vs. MSE Loss Landscape')
    plt.xlabel('Prediction Error (preds - targets)')
    plt.ylabel('Loss Value')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.axvline(x=c, color='green', linestyle=':', label=f'Shrinkage Threshold c={c}')
    plt.axvline(x=-c, color='green', linestyle=':')
    plt.show()

# --- Integration Tests with Darts Models ---

# A simple TimeSeries for testing
from darts import TimeSeries
from darts.models import NBEATSModel, TCNModel, BlockRNNModel, TransformerModel, NLinearModel, TiDEModel

# Define a small, consistent dataset for all integration tests
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
def test_shrinkage_loss_with_darts_models(model_name, model_tuple, seed):
    """
    Tests that ShrinkageLoss can be successfully used for training with various Darts models.
    This is an integration test to ensure compatibility with the Darts training loop.
    """
    model_cls, model_kwargs = model_tuple
    torch.manual_seed(seed) # for reproducibility
    
    loss_fn = ShrinkageLoss()

    model = model_cls(
        **model_kwargs,
        n_epochs=2,
        random_state=seed,
        loss_fn=loss_fn,
        pl_trainer_kwargs={"accelerator": "cpu", "devices": 1},
        force_reset=True,
    )

    try:
        model.fit(series=ts_train, val_series=ts_val, verbose=False)
    except Exception as e:
        pytest.fail(f"Integration test for {model_name} with ShrinkageLoss and seed {seed} failed with error: {e}")
    
    # Check that predictions can be made without crashing
    try:
        preds = model.predict(n=5)
        assert len(preds) == 5
    except Exception as e:
        pytest.fail(f"Prediction for {model_name} with ShrinkageLoss and seed {seed} failed after training: {e}")


def test_shrinkage_loss_gradient_check():
    """
    Performs a gradient check for ShrinkageLoss using torch.autograd.gradcheck.
    This is critical for ensuring the loss is implemented correctly for backpropagation.
    """
    from torch.autograd import gradcheck
    # gradcheck needs double precision and a requires_grad=True input
    loss_fn = ShrinkageLoss(a=10.0, c=0.2)
    
    # Test with typical inputs
    preds = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    targets = torch.randn(2, 2, dtype=torch.double)
    
    # The check will be True if the analytical and numerical gradients match
    assert gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)

    # Test near the 'c' threshold, which can be unstable
    preds_near_c = torch.tensor([[0.3], [0.5]], dtype=torch.double, requires_grad=True)
    targets_near_c = torch.tensor([[0.1], [0.8]], dtype=torch.double) # errors are 0.2 and 0.3
    
    assert gradcheck(loss_fn, (preds_near_c, targets_near_c), eps=1e-6, atol=1e-4)
