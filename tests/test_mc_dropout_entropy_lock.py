import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from views_r2darts2.model.darts_forecaster import DartsForecaster
from views_r2darts2.utils.reproducibility_gate import ReproducibilityGate
from views_r2darts2.model.model_catalog import ModelCatalog

# --- Mocking Utilities ---

def mock_model_predict_behavior(self, n: int, series: TimeSeries, num_samples: int = 1, **kwargs) -> TimeSeries:
    """
    Simulates MC Dropout behavior.
    Uses global torch.randn, so it's sensitive to the global seed state.
    """
    outputs = []
    # Darts predict receives a single series or a list. Handle both.
    if isinstance(series, list):
        ref_series = series[0]
    else:
        ref_series = series

    for _ in range(num_samples):
        # Noise simulates dropout randomness
        output_value = ref_series.values()[-1, 0] + torch.randn(1).item() * 0.1 
        outputs.append(output_value)
    
    # Return a TimeSeries with integer index to match the input
    start_idx = ref_series.time_index[-1] + 1
    times = pd.RangeIndex(start=start_idx, stop=start_idx + n, step=1)
    values = np.array(outputs).reshape(n, 1, num_samples)
    
    return TimeSeries.from_times_and_values(
        times=times,
        values=values,
        columns=["prediction"],
        static_covariates=ref_series.static_covariates, # Critical: Preserve entity ID
    )

# --- Fixtures ---

@pytest.fixture
def dummy_timeseries():
    """A simple TimeSeries for mock model input."""
    # Length 50 to support slicing with integer index
    ts = TimeSeries.from_times_and_values(
        pd.RangeIndex(start=0, stop=50, step=1),
        (np.random.randn(50, 1) + 100.0).astype(np.float32), # Shift to positive to avoid clipping
        columns=["prediction"], # Explicit column name match
    )
    # Correctly attach static covariates (returns new object)
    return ts.with_static_covariates(pd.DataFrame({"entity": [1.0]}))

@pytest.fixture
def mock_darts_model_instance():
    """
    Creates a robust MagicMock of TorchForecastingModel.
    Handles the tricky 'parameters()' iterator requirement.
    """
    # 1. Mock the underlying PyTorch Module
    mock_torch_module = MagicMock(spec=torch.nn.Module)
    
    # 2. Setup 'parameters()' to return a FRESH iterator every time it's called.
    def fresh_parameters_iterator():
        param = MagicMock(spec=torch.Tensor)
        param.device = torch.device("cpu")
        yield param
        
    mock_torch_module.parameters.side_effect = fresh_parameters_iterator

    # 3. Mock the Darts Model Wrapper
    mock_darts_model = MagicMock(spec=TorchForecastingModel)
    mock_darts_model.model = mock_torch_module # DartsForecaster accesses .model.model
    mock_darts_model.input_chunk_length = 4 # Match dummy series
    mock_darts_model.output_chunk_length = 1
    mock_darts_model.device = torch.device("cpu")
    
    # 4. Attach prediction behavior
    mock_darts_model.predict.side_effect = mock_model_predict_behavior.__get__(
        mock_darts_model, type(mock_darts_model)
    )
    
    # 5. Mock save/load to avoid IO errors if called
    mock_darts_model.save = MagicMock()
    mock_darts_model.__class__.load = MagicMock(return_value=mock_darts_model)

    return mock_darts_model

@pytest.fixture
def basic_config():
    return {"random_state": 42}

# --- Tests ---

class TestMCDropoutEntropyLock:

    @patch("views_r2darts2.model.darts_forecaster.DartsForecaster.get_device", return_value="cpu")
    @patch.object(ModelCatalog, "get_model", autospec=True)
    def test_mc_dropout_sequence_reproducibility(
        self, mock_get_model, mock_get_device, dummy_timeseries, mock_darts_model_instance, basic_config
    ):
        """
        [Green Team] Proves that DartsForecaster.predict() produces bit-identical sequences
        of MC Dropout samples when run with the same seed.
        """
        mock_get_model.return_value = mock_darts_model_instance
        
        # Mock Dataset
        mock_dataset = MagicMock()
        mock_dataset.as_darts_timeseries.return_value = [dummy_timeseries]
        mock_dataset.targets = ["prediction"]
        mock_dataset.features = []
        mock_dataset._time_id = "time"
        mock_dataset._entity_id = "entity"
        
        forecaster = DartsForecaster(
            dataset=mock_dataset,
            model=mock_darts_model_instance,
            partition_dict={"train": (0, 20), "test": (21, 40)}, # Valid dummy partition
            random_state=basic_config["random_state"],
        )
        forecaster.device = "cpu" # Force CPU to bypass hardware self-healing in tests

        num_samples = 10

        # Run 1
        predictions_1 = forecaster.predict(
            sequence_number=0, output_length=1, num_samples=num_samples
        )
        # The DataFrame contains lists of samples in the prediction column. Extract the list from the first cell.
        samples_1 = predictions_1.filter(like="pred_").iloc[0, 0]

        # Run 2
        predictions_2 = forecaster.predict(
            sequence_number=0, output_length=1, num_samples=num_samples
        )
        samples_2 = predictions_2.filter(like="pred_").iloc[0, 0]

        # Assert: The entire sequence of 10 samples must be identical across runs
        np.testing.assert_array_equal(samples_1, samples_2)
        assert len(samples_1) == num_samples

    @patch("views_r2darts2.model.darts_forecaster.DartsForecaster.get_device", return_value="cpu")
    @patch.object(ModelCatalog, "get_model", autospec=True)
    def test_mc_dropout_samples_are_diverse(
        self, mock_get_model, mock_get_device, dummy_timeseries, mock_darts_model_instance, basic_config
    ):
        """
        [Beige Team] Proves that within a SINGLE run, the samples are diverse.
        This confirms that lock_entropy is NOT resetting the seed before every single sample generation.
        """
        mock_get_model.return_value = mock_darts_model_instance

        mock_dataset = MagicMock()
        mock_dataset.as_darts_timeseries.return_value = [dummy_timeseries]
        mock_dataset.targets = ["prediction"]
        mock_dataset.features = []
        mock_dataset._time_id = "time"
        mock_dataset._entity_id = "entity"

        forecaster = DartsForecaster(
            dataset=mock_dataset,
            model=mock_darts_model_instance,
            partition_dict={"train": (0, 20), "test": (21, 40)},
            random_state=basic_config["random_state"],
        )
        forecaster.device = "cpu"

        predictions = forecaster.predict(
            sequence_number=0, output_length=1, num_samples=10
        )
        
        # Extract samples (list of 10 values)
        individual_values = predictions.filter(like="pred_").iloc[0, 0]
        
        # Assert: There must be variance in the samples
        assert len(np.unique(individual_values)) > 1, "Entropy Lock erroneously collapsed the MC Dropout distribution to a point estimate."

    def test_mock_setup_sensitivity(self, dummy_timeseries, mock_darts_model_instance):
        """
        [Red Team Setup] Verifies our mock model actually responds to global seed changes.
        """
        # Scenario 1: Seed 123
        ReproducibilityGate.Data.lock_entropy(123)
        out1 = mock_darts_model_instance.predict(
            n=1, series=dummy_timeseries, num_samples=1
        ).all_values()

        # Scenario 2: Seed 123 (Should match)
        ReproducibilityGate.Data.lock_entropy(123)
        out2 = mock_darts_model_instance.predict(
            n=1, series=dummy_timeseries, num_samples=1
        ).all_values()

        # Scenario 3: Seed 999 (Should differ)
        ReproducibilityGate.Data.lock_entropy(999)
        out3 = mock_darts_model_instance.predict(
            n=1, series=dummy_timeseries, num_samples=1
        ).all_values()

        assert np.allclose(out1, out2)
        assert not np.allclose(out1, out3)
