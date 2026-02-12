# file: tests/test_mse_loss.py

import torch
import pytest
from darts import TimeSeries
from darts.models import (
    NBEATSModel,
    TCNModel,
    BlockRNNModel,
    TransformerModel,
    NLinearModel,
    TiDEModel,
)
from views_r2darts2.utils.loss import LossSelector

# Define a small, consistent dataset for all integration tests
ts_train = TimeSeries.from_values(
    torch.arange(0, 100, dtype=torch.float32).unsqueeze(1)
)  # Increased length
ts_val = TimeSeries.from_values(
    torch.arange(100, 150, dtype=torch.float32).unsqueeze(1)
)  # Increased length

# Define model configurations to be tested
MODEL_CONFIGS = {
    "N-BEATS": (NBEATSModel, {"input_chunk_length": 12, "output_chunk_length": 6}),
    "TCN": (
        TCNModel,
        {
            "input_chunk_length": 24,
            "output_chunk_length": 6,
            "kernel_size": 3,
            "num_filters": 4,
        },
    ),
    "BlockRNN (LSTM)": (
        BlockRNNModel,
        {
            "model": "LSTM",
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "hidden_dim": 16,
        },
    ),
    "Transformer": (
        TransformerModel,
        {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "d_model": 16,
            "nhead": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
        },
    ),
    "N-Linear": (NLinearModel, {"input_chunk_length": 12, "output_chunk_length": 6}),
    "TiDE": (
        TiDEModel,
        {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 8,
            "hidden_size": 16,
        },
    ),
}


@pytest.mark.parametrize("seed", [42, 1337])  # Test with multiple random seeds
@pytest.mark.parametrize("model_name, model_tuple", MODEL_CONFIGS.items())
def test_mse_loss_with_darts_models(model_name, model_tuple, seed):
    """
    Tests that MSELoss can be successfully used for training with various Darts models.
    This is an integration test to ensure compatibility with the Darts training loop.
    """
    model_cls, model_kwargs = model_tuple
    torch.manual_seed(seed)  # for reproducibility

    loss_fn = LossSelector.get_loss_function("MSELoss")

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
        pytest.fail(
            f"Integration test for {model_name} with MSELoss and seed {seed} failed with error: {e}"
        )

    # Check that predictions can be made without crashing
    try:
        preds = model.predict(n=5)
        assert len(preds) == 5
    except Exception as e:
        pytest.fail(
            f"Prediction for {model_name} with MSELoss and seed {seed} failed after training: {e}"
        )
