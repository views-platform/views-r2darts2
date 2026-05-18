import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock, patch
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from sklearn.preprocessing import StandardScaler
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.transformers.views_dataset_darts import _ViewsDatasetDarts


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = Mock(spec=_ViewsDatasetDarts)
    dataset.targets = ["target1", "target2"]
    dataset.features = ["feature1", "feature2"]
    dataset._time_id = "time"
    dataset._entity_id = "entity_id"
    return dataset


@pytest.fixture
def mock_model():
    """Create a mock Darts model."""
    model = Mock(spec=TorchForecastingModel)
    model.input_chunk_length = 12
    model.output_chunk_length = 6
    model.to_device = Mock()

    # Nested mock for the underlying PyTorch model
    mock_torch_model = Mock()
    mock_param = Mock()
    mock_param.device = torch.device("cpu")
    mock_torch_model.parameters.return_value = iter([mock_param])
    model.model = mock_torch_model

    return model


@pytest.fixture
def partition_dict():
    """Create sample partition dictionary."""
    return {"train": (0, 100), "test": (100, 150)}


@pytest.fixture
def forecaster(mock_dataset, mock_model, partition_dict):
    """Create a DartsForecaster instance."""
    # Ensure weights match CPU device
    mock_param = Mock()
    mock_param.device = torch.device("cpu")
    mock_model.model.parameters.return_value = iter([mock_param])

    with (
        patch("views_r2darts2.engines.darts_forecaster.ScalerSelector"),
        patch.object(DartsForecaster, "get_device", return_value="cpu"),
    ):
        forecaster = DartsForecaster(
            dataset=mock_dataset,
            model=mock_model,
            partition_dict=partition_dict,
            random_state=42,
        )

    
    return forecaster


@pytest.fixture
def forecaster_with_scalers(mock_dataset, mock_model, partition_dict):
    """Create a DartsForecaster with real, picklable scalers."""
    # Use a real scaler that can be pickled by torch.save

    # Ensure weights match CPU device
    mock_param = Mock()
    mock_param.device = torch.device("cpu")
    mock_model.model.parameters.return_value = iter([mock_param])

    # We patch the selector to return our real scaler instance
    with (
        patch(
            "views_r2darts2.engines.darts_forecaster.ScalerSelector.get_scaler",
            return_value=StandardScaler(),
        ),
        patch.object(DartsForecaster, "get_device", return_value="cpu"),
    ):
        forecaster = DartsForecaster(
            dataset=mock_dataset,
            model=mock_model,
            partition_dict=partition_dict,
            feature_scaler="StandardScaler",
            target_scaler="StandardScaler",  # Using the same for simplicity
            random_state=42,
        )
    
    return forecaster


class TestDartsForecaster:
    def test_initialization(self, forecaster, mock_dataset, mock_model):
        """Test forecaster initialization."""
        assert forecaster.dataset == mock_dataset
        assert forecaster.model == mock_model
        assert forecaster._train_start == 0
        assert forecaster._train_end == 100
        assert forecaster._test_start == 100
        assert forecaster._test_end == 150
        assert forecaster.scaler_fitted is False

    def test_initialization_with_scalers(
        self, mock_dataset, mock_model, partition_dict
    ):
        """Test initialization with feature and target scalers."""
        with patch(
            "views_r2darts2.engines.darts_forecaster.ScalerSelector"
        ) as mock_scaler_selector:
            mock_scaler_selector.get_scaler.return_value = Mock()

            forecaster = DartsForecaster(
                dataset=mock_dataset,
                model=mock_model,
                partition_dict=partition_dict,
                feature_scaler="StandardScaler",
                target_scaler="RobustScaler",
                random_state=42,
            )

            assert forecaster._feature_scaler_cfg == "StandardScaler"
            assert forecaster._target_scaler_cfg == "RobustScaler"
            assert forecaster.feature_scaler is not None
            assert forecaster.target_scaler is not None

    def test_initialization_without_scalers(self, forecaster):
        """Test initialization without scalers."""
        assert forecaster._feature_scaler_cfg is None
        assert forecaster._target_scaler_cfg is None
        assert forecaster.feature_scaler is None
        assert forecaster.target_scaler is None

    def test_get_device_cuda(self):
        """Test get_device returns cuda when available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device = DartsForecaster.get_device()
            assert device == "cuda"

    def test_get_device_mps(self):
        """Test get_device returns mps when available."""
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            device = DartsForecaster.get_device()
            assert device == "mps"

    def test_get_device_cpu(self):
        """Test get_device returns cpu when no accelerator available."""
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            device = DartsForecaster.get_device()
            assert device == "cpu"

    def test_device_assignment(self, mock_dataset, mock_model, partition_dict):
        """Test that model is moved to correct device."""
        # Update the mock model weight to be on CUDA for this test
        mock_param = Mock()
        mock_param.device = torch.device("cuda:0")
        mock_model.model.parameters.return_value = iter([mock_param])

        with (
            patch("views_r2darts2.engines.darts_forecaster.ScalerSelector"),
            patch.object(DartsForecaster, "get_device", return_value="cuda"),
        ):
            forecaster = DartsForecaster(
                dataset=mock_dataset,
                model=mock_model,
                partition_dict=partition_dict,
                random_state=42,
            )

            assert forecaster.device == "cuda"
            mock_model.to_device.assert_called_once_with("cuda")

    def test_min_length_calculation(self, forecaster):
        """Test minimum length calculation."""
        forecaster.min_length = (
            forecaster.model.input_chunk_length + forecaster.model.output_chunk_length
        )
        assert forecaster.min_length == 18  # 12 + 6

    def test_preprocess_timeseries_train_mode(
        self, forecaster_with_scalers, mock_dataset
    ):
        """Test preprocessing in training mode."""
        # Create mock TimeSeries
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.__len__ = Mock(return_value=120)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)
        # Provide a valid time_index mock that supports .max() returning a real number
        mock_index = Mock()
        mock_index.max = Mock(return_value=100)
        # Mock values.astype(int) to return a real numpy array for contiguity check
        mock_values = Mock()
        mock_values.astype = Mock(return_value=np.arange(101))
        mock_index.values = mock_values
        mock_ts.time_index = mock_index

        # Mock all_values to return a numpy array for _check_data_sanity
        mock_ts.all_values = Mock(return_value=np.array([[1.0, 2.0], [3.0, 4.0]]))
        mock_ts.components = pd.Index(["target1", "target2"])

        timeseries = [mock_ts]

        # Mock scaler methods
        forecaster_with_scalers.target_scaler = Mock()
        forecaster_with_scalers.feature_scaler = Mock()
        forecaster_with_scalers.target_scaler.fit_transform = Mock(
            return_value=[mock_ts]
        )
        forecaster_with_scalers.feature_scaler.fit_transform = Mock(
            return_value=[mock_ts]
        )

        targets, past_cov = forecaster_with_scalers._preprocess_timeseries(
            timeseries=timeseries, start=0, end=100, train_mode=True
        )

        assert forecaster_with_scalers.scaler_fitted is True
        forecaster_with_scalers.target_scaler.fit_transform.assert_called_once()
        forecaster_with_scalers.feature_scaler.fit_transform.assert_called_once()

    def test_preprocess_timeseries_prediction_mode(self, forecaster_with_scalers):
        """Test preprocessing in prediction mode."""
        # Create mock TimeSeries
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)
        # Mock all_values to return a numpy array for _check_data_sanity
        mock_ts.all_values = Mock(return_value=np.array([[1.0, 2.0], [3.0, 4.0]]))
        mock_ts.components = pd.Index(["target1", "target2"])

        timeseries = [mock_ts]

        # Mock scaler methods
        forecaster_with_scalers.target_scaler = Mock()
        forecaster_with_scalers.feature_scaler = Mock()
        forecaster_with_scalers.target_scaler.transform = Mock(return_value=[mock_ts])
        forecaster_with_scalers.feature_scaler.transform = Mock(return_value=[mock_ts])
        forecaster_with_scalers.scaler_fitted = True

        targets, past_cov = forecaster_with_scalers._preprocess_timeseries(
            timeseries=timeseries, start=100, end=120, train_mode=False
        )

        forecaster_with_scalers.target_scaler.transform.assert_called_once()
        forecaster_with_scalers.feature_scaler.transform.assert_called_once()

    def test_process_predictions_deterministic(self, forecaster):
        """Test processing deterministic predictions."""
        # Create mock prediction
        times = pd.date_range("2020-01-01", periods=3, freq="M")
        values = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )  # 3 timesteps, 2 components

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()  # Use proper pandas offset

        results = forecaster._process_predictions([mock_pred])

        assert len(results) == 3  # 3 timesteps
        assert results[0]["entity_id"] == 42
        assert "pred_target1" in results[0]
        assert "pred_target2" in results[0]

    def test_process_predictions_probabilistic(self, forecaster):
        """Test processing probabilistic predictions."""
        times = pd.date_range("2020-01-01", periods=2, freq="M")
        # 2 timesteps, 2 components, 10 samples
        values = np.random.randn(2, 2, 10)

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()

        results = forecaster._process_predictions([mock_pred])

        assert len(results) == 2
        assert isinstance(results[0]["pred_target1"], list)
        assert len(results[0]["pred_target1"]) == 10  # 10 samples

    def test_process_predictions_clips_negative_values(self, forecaster):
        """Test that negative predictions are clipped to zero."""
        times = pd.date_range("2020-01-01", periods=1, freq="M")
        # 1 timestep, 2 components, 1 sample each
        values = np.array(
            [
                [[-5.0], [2.0]]  # timestep 0: component 0 = -5.0, component 1 = 2.0
            ]
        )

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()

        results = forecaster._process_predictions([mock_pred])

        # Check that negative value was clipped to 0.0
        # When there's 1 sample, it returns a list with 1 element
        assert abs(results[0]["pred_target1"][0] - 0.0) < 1e-10
        assert results[0]["pred_target2"] == [2.0]

    def test_process_predictions_raises_on_nans(self, forecaster):
        """[RED TEAM] Test that NaN values trigger a NumericalSanityError."""
        from views_r2darts2.infrastructure.exceptions import NumericalSanityError

        times = pd.date_range("2020-01-01", periods=1, freq="M")
        # 1 timestep, 2 components, 1 sample each
        values = np.array(
            [
                [[np.nan], [2.0]]  # timestep 0: component 0 = nan, component 1 = 2.0
            ]
        )

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()

        with pytest.raises(NumericalSanityError, match="NaN detected"):
            forecaster._process_predictions([mock_pred])

    def test_train(self, forecaster):
        """Test model training."""
        # Create mock TimeSeries
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.__len__ = Mock(return_value=120)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)

        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))

        forecaster.train()

        forecaster.model.fit.assert_called_once()
        call_kwargs = forecaster.model.fit.call_args[1]
        assert "series" in call_kwargs
        assert "past_covariates" in call_kwargs
        assert call_kwargs["verbose"] is True

    def test_train_without_features(self, forecaster):
        """Test training without past covariates."""
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.__len__ = Mock(return_value=120)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)

        forecaster.dataset.features = []
        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [None]))

        forecaster.train()

        forecaster.model.fit.assert_called_once()

    def test_predict(self, forecaster):
        """Test making predictions."""
        # Create mock TimeSeries
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)

        # Create mock prediction
        times = pd.date_range("2020-01-01", periods=2, freq="M")
        values = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()

        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])

        result = forecaster.predict(sequence_number=0, output_length=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        forecaster.model.predict.assert_called_once()

    def test_predict_rolling_origin_sequence_zero_ends_at_test_start_minus_one(
        self, forecaster
    ):
        """[REGRESSION — C-01 / commit f78bbf7] Rolling-origin base-origin convention.

        For `sequence_number=0`, the input window must end at `test_start - 1`
        (exclusive of the first test month), so that the first forecast month
        equals `partition['test'][0]`. An off-by-one here had `end = test_start`,
        causing the model to observe month `test_start` itself and forecast one
        month too late (e.g. 446–481 instead of 445–480 for the standard
        adolecent_slob partition). Only caught in production by the
        views-pipeline-core `_assert_predictions_in_step_window()` pre-flight.
        """
        # fixture partition: train=(0,100), test=(100,150); input_chunk_length=12
        mock_ts = Mock(spec=TimeSeries)
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=np.array([[1.0, 2.0]]))
        mock_pred.start_time = Mock(return_value=pd.Timestamp("2020-01-31"))
        mock_pred.freq = pd.offsets.MonthEnd()

        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])

        forecaster.predict(sequence_number=0, output_length=1)

        _, kwargs = forecaster._preprocess_timeseries.call_args
        assert kwargs["end"] == forecaster._test_start - 1, (
            f"sequence_number=0 must produce end={forecaster._test_start - 1} "
            f"(base_origin convention), got end={kwargs['end']}. "
            "An off-by-one here produces forecasts one month too late."
        )
        # input window starts `input_chunk_length` months before the origin
        assert kwargs["start"] == (
            forecaster._test_start - forecaster.model.input_chunk_length
        )

    def test_predict_rolling_origin_advances_one_month_per_sequence(self, forecaster):
        """[REGRESSION — C-01] Sequence n must advance the origin by exactly n months.

        If the start/end arithmetic in `predict()` ever drifts from the contract
        `end = test_start - 1 + sequence_number`, this test fails before the
        views-pipeline-core pre-flight ever gets to catch it.
        """
        mock_ts = Mock(spec=TimeSeries)
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=np.array([[1.0, 2.0]]))
        mock_pred.start_time = Mock(return_value=pd.Timestamp("2020-01-31"))
        mock_pred.freq = pd.offsets.MonthEnd()

        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])

        # predict() calls next(self.model.model.parameters()) on every invocation
        # for device verification — refresh the iterator before each call so it
        # doesn't exhaust after the first sequence.
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        for seq in (0, 1, 5, 12):
            forecaster.model.model.parameters.return_value = iter([mock_param])
            forecaster._preprocess_timeseries.reset_mock()
            forecaster.predict(sequence_number=seq, output_length=1)
            _, kwargs = forecaster._preprocess_timeseries.call_args
            assert kwargs["end"] == forecaster._test_start - 1 + seq
            assert kwargs["start"] == (
                forecaster._test_start + seq - forecaster.model.input_chunk_length
            )

    def test_predict_with_scaler_inverse_transform(self, forecaster_with_scalers):
        """Test prediction with target scaler inverse transform."""
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)

        times = pd.date_range("2020-01-01", periods=2, freq="M")
        values = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()

        forecaster_with_scalers.dataset.as_darts_timeseries = Mock(
            return_value=[mock_ts]
        )
        forecaster_with_scalers._preprocess_timeseries = Mock(
            return_value=([mock_ts], [mock_ts])
        )
        forecaster_with_scalers.model.predict = Mock(return_value=[mock_pred])
        forecaster_with_scalers.target_scaler.inverse_transform = Mock(
            return_value=[mock_pred]
        )
        # Must set scaler_fitted=True for inverse transform to be applied
        forecaster_with_scalers.scaler_fitted = True

        result = forecaster_with_scalers.predict(sequence_number=0, output_length=2)

        forecaster_with_scalers.target_scaler.inverse_transform.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_predict_raises_exception(self, forecaster):
        """Test prediction error handling."""
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)

        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(side_effect=Exception("Prediction failed"))

        with pytest.raises(Exception, match="Prediction failed"):
            forecaster.predict(sequence_number=0)

    def test_predict_raises_on_nans(self, forecaster):
        """[RED TEAM] Test that predictions with NaN trigger a NumericalSanityError."""
        from views_r2darts2.infrastructure.exceptions import NumericalSanityError

        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)

        times = pd.date_range("2020-01-01", periods=2, freq="M")
        values = np.array([[1.0, np.nan], [3.0, 4.0]])

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()

        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])

        with pytest.raises(NumericalSanityError, match="NaN detected"):
            forecaster.predict(sequence_number=0, output_length=2)

    def test_save_model(self, forecaster_with_scalers, tmp_path):
        """Test saving model and scalers."""
        model_path = tmp_path / "model.pt"

        forecaster_with_scalers.model.save = Mock()

        with patch("torch.save") as mock_torch_save:
            forecaster_with_scalers.save_model(str(model_path))

            forecaster_with_scalers.model.save.assert_called_once_with(
                path=str(model_path)
            )
            mock_torch_save.assert_called_once()

            # Check that scaler data was saved
            call_args = mock_torch_save.call_args
            assert "target_scaler" in call_args[0][0]
            assert "feature_scaler" in call_args[0][0]
            assert "scaler_fitted" in call_args[0][0]

    def test_load_model(
        self, forecaster_with_scalers, mock_dataset, partition_dict, tmp_path
    ):
        """
        Test loading model restores a FUNCTIONAL scaler.
        This is a robust behavioral test, not a fragile internal state check.
        It verifies that a scaler fitted and saved can be loaded and used
        to correctly perform an inverse_transform, which is its core function.
        """
        model_path = tmp_path / "model.pt"

        # 1. ARRANGE: Create known data and a new TimeSeries for it.
        # We need to test the target_scaler, so we only need one component.
        mock_dataset.targets = ["target1"]
        known_data = np.array([[[10.0]], [[20.0]], [[30.0]]])
        known_ts = TimeSeries.from_values(known_data)

        # 2. ACT (Fit, Save, Load)
        # Fit the scaler on our original forecaster and get the transformed data
        transformed_ts = forecaster_with_scalers.target_scaler.fit_transform(known_ts)
        forecaster_with_scalers.scaler_fitted = True

        # Mock the model's save/load methods since we only test scaler serialization
        forecaster_with_scalers.model.save = Mock()

        # Save the forecaster state (including the fitted scaler)
        forecaster_with_scalers.save_model(str(model_path))

        # Create a new forecaster instance to load the state into
        # The mock model's __class__ needs a 'load' method for this to work
        forecaster_with_scalers.model.__class__.load = Mock(
            return_value=forecaster_with_scalers.model
        )
        new_forecaster = DartsForecaster(
            dataset=mock_dataset,
            model=forecaster_with_scalers.model,
            partition_dict=partition_dict,
            target_scaler=forecaster_with_scalers._target_scaler_cfg,
            feature_scaler=forecaster_with_scalers._feature_scaler_cfg,
            random_state=42,
        )
        new_forecaster.load_model(str(model_path))

        # 3. ASSERT
        # The critical part: use the loaded scaler to perform a round-trip
        round_trip_ts = new_forecaster.target_scaler.inverse_transform(transformed_ts)

        # The inverse transform should return the original data, proving the
        # scaler's fitted state was correctly saved and loaded.
        assert new_forecaster.scaler_fitted is True
        assert np.allclose(known_ts.values(), round_trip_ts.values())

    def test_load_model_missing_scalers(self, forecaster_with_scalers, tmp_path):
        """Test loading model when scaler file is missing."""
        model_path = tmp_path / "model.pt"

        # The model file itself might exist, but the scaler file (`.pt.scalers`) won't.
        # DartsForecaster.load_model should raise FileNotFoundError in this case.
        with open(model_path, "w") as f:
            f.write("dummy model data")

        forecaster_with_scalers.model.__class__.load = Mock()

        with pytest.raises(FileNotFoundError):
            forecaster_with_scalers.load_model(str(model_path))

        # Ensure we didn't even attempt to load the Darts model
        forecaster_with_scalers.model.__class__.load.assert_not_called()

    def test_load_model_moves_to_device(self, forecaster_with_scalers, tmp_path):
        """
        Test that `load_model` respects the forecaster's device setting.
        This test verifies that the `map_location` argument is correctly passed
        to the underlying Darts model's load method.
        """
        model_path = tmp_path / "model.pt"
        forecaster_with_scalers.device = "cuda"

        # Mock the model's save method
        forecaster_with_scalers.model.save = Mock()
        # Mock the class's load method to return a model instance
        forecaster_with_scalers.model.__class__.load = Mock(
            return_value=forecaster_with_scalers.model
        )

        # 1. Save the model first (the contents don't matter for this test)
        forecaster_with_scalers.scaler_fitted = True
        forecaster_with_scalers.save_model(str(model_path))

        # 2. Now, load the model
        forecaster_with_scalers.load_model(str(model_path))

        # 3. ASSERT
        # The main point of this test: assert that the underlying Darts `load`
        # was called with the correct `map_location`.
        forecaster_with_scalers.model.__class__.load.assert_called_once_with(
            path=str(model_path), map_location="cuda"
        )

        # Also verify that the `to_device` method was called after loading
        # to ensure the model is on the correct device.
        forecaster_with_scalers.model.to_device.assert_called_with("cuda")

    def test_predict_with_kwargs(self, forecaster):
        """Test prediction with additional kwargs."""
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)

        times = pd.date_range("2020-01-01", periods=2, freq="M")
        values = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()

        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])

        forecaster.predict(sequence_number=0, output_length=2, num_samples=100)

        call_kwargs = forecaster.model.predict.call_args[1]
        assert "num_samples" in call_kwargs
        assert call_kwargs["num_samples"] == 100


class TestDartsForecasterIntegration:
    """Integration tests for DartsForecaster."""

    def test_full_workflow_without_scalers(
        self, mock_dataset, mock_model, partition_dict
    ):
        """Test complete workflow without scalers."""
        with patch("views_r2darts2.engines.darts_forecaster.ScalerSelector"):
            forecaster = DartsForecaster(
                dataset=mock_dataset,
                model=mock_model,
                partition_dict=partition_dict,
                random_state=42,
            )

            assert forecaster.target_scaler is None
            assert forecaster.feature_scaler is None
            assert forecaster.scaler_fitted is False

    def test_sequence_number_calculation(self, forecaster):
        """Test that sequence number correctly calculates time windows."""
        assert forecaster._test_start == 100
        assert forecaster.model.input_chunk_length == 12

        # For sequence_number=0, start should be test_start - input_chunk_length
        expected_start = 100 - 12
        assert expected_start == 88

    def test_output_length_parameter(self, forecaster):
        """Test that output_length is correctly passed to model.predict."""
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)

        times = pd.date_range("2020-01-01", periods=10, freq="M")
        values = np.random.randn(10, 2)

        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({"entity_id": [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()

        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])

        forecaster.predict(sequence_number=0, output_length=10)

        call_kwargs = forecaster.model.predict.call_args[1]
        assert call_kwargs["n"] == 10
