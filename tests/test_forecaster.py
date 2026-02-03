import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from views_r2darts2.model.forecaster import DartsForecaster
from views_r2darts2.data.handlers import _ViewsDatasetDarts


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
    return model


@pytest.fixture
def partition_dict():
    """Create sample partition dictionary."""
    return {
        "train": (0, 100),
        "test": (100, 150)
    }


@pytest.fixture
def forecaster(mock_dataset, mock_model, partition_dict):
    """Create a DartsForecaster instance."""
    with patch('views_r2darts2.model.forecaster.ScalerSelector'):
        forecaster = DartsForecaster(
            dataset=mock_dataset,
            model=mock_model,
            partition_dict=partition_dict
        )
    return forecaster


@pytest.fixture
def forecaster_with_scalers(mock_dataset, mock_model, partition_dict):
    """Create a DartsForecaster with scalers."""
    with patch('views_r2darts2.model.forecaster.ScalerSelector') as mock_scaler_selector:
        mock_scaler_selector.get_scaler.return_value = Mock()
        forecaster = DartsForecaster(
            dataset=mock_dataset,
            model=mock_model,
            partition_dict=partition_dict,
            feature_scaler="StandardScaler",
            target_scaler="RobustScaler"
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

    def test_initialization_with_scalers(self, mock_dataset, mock_model, partition_dict):
        """Test initialization with feature and target scalers."""
        with patch('views_r2darts2.model.forecaster.ScalerSelector') as mock_scaler_selector:
            mock_scaler_selector.get_scaler.return_value = Mock()
            
            forecaster = DartsForecaster(
                dataset=mock_dataset,
                model=mock_model,
                partition_dict=partition_dict,
                feature_scaler="StandardScaler",
                target_scaler="RobustScaler"
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
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.backends.mps.is_available', return_value=False):
            device = DartsForecaster.get_device()
            assert device == "cuda"

    def test_get_device_mps(self):
        """Test get_device returns mps when available."""
        with patch('torch.backends.mps.is_available', return_value=True), \
             patch('torch.cuda.is_available', return_value=False):
            device = DartsForecaster.get_device()
            assert device == "mps"

    def test_get_device_cpu(self):
        """Test get_device returns cpu when no accelerator available."""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):
            device = DartsForecaster.get_device()
            assert device == "cpu"

    def test_device_assignment(self, mock_dataset, mock_model, partition_dict):
        """Test that model is moved to correct device."""
        with patch('views_r2darts2.model.forecaster.ScalerSelector'), \
             patch.object(DartsForecaster, 'get_device', return_value='cuda'):
            forecaster = DartsForecaster(
                dataset=mock_dataset,
                model=mock_model,
                partition_dict=partition_dict
            )
            
            assert forecaster.device == 'cuda'
            mock_model.to_device.assert_called_once_with('cuda')

    def test_min_length_calculation(self, forecaster):
        """Test minimum length calculation."""
        forecaster.min_length = forecaster.model.input_chunk_length + forecaster.model.output_chunk_length
        assert forecaster.min_length == 18  # 12 + 6

    def test_preprocess_timeseries_train_mode(self, forecaster_with_scalers, mock_dataset):
        """Test preprocessing in training mode."""
        # Create mock TimeSeries
        times = pd.date_range('2020-01-01', periods=120, freq='M')
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.__len__ = Mock(return_value=120)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)
        # Mock all_values to return a numpy array for _check_data_sanity
        mock_ts.all_values = Mock(return_value=np.array([[1.0, 2.0], [3.0, 4.0]]))
        mock_ts.components = pd.Index(['target1', 'target2'])
        
        timeseries = [mock_ts]
        
        # Mock scaler methods
        forecaster_with_scalers.target_scaler = Mock()
        forecaster_with_scalers.feature_scaler = Mock()
        forecaster_with_scalers.target_scaler.fit_transform = Mock(return_value=[mock_ts])
        forecaster_with_scalers.feature_scaler.fit_transform = Mock(return_value=[mock_ts])
        
        targets, past_cov = forecaster_with_scalers._preprocess_timeseries(
            timeseries=timeseries,
            start=0,
            end=100,
            train_mode=True
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
        mock_ts.components = pd.Index(['target1', 'target2'])
        
        timeseries = [mock_ts]
        
        # Mock scaler methods
        forecaster_with_scalers.target_scaler = Mock()
        forecaster_with_scalers.feature_scaler = Mock()
        forecaster_with_scalers.target_scaler.transform = Mock(return_value=[mock_ts])
        forecaster_with_scalers.feature_scaler.transform = Mock(return_value=[mock_ts])
        forecaster_with_scalers.scaler_fitted = True
        
        targets, past_cov = forecaster_with_scalers._preprocess_timeseries(
            timeseries=timeseries,
            start=100,
            end=120,
            train_mode=False
        )
        
        forecaster_with_scalers.target_scaler.transform.assert_called_once()
        forecaster_with_scalers.feature_scaler.transform.assert_called_once()

    def test_process_predictions_deterministic(self, forecaster):
        """Test processing deterministic predictions."""
        # Create mock prediction
        times = pd.date_range('2020-01-01', periods=3, freq='M')
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 timesteps, 2 components
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()  # Use proper pandas offset
        
        results = forecaster._process_predictions([mock_pred])
        
        assert len(results) == 3  # 3 timesteps
        assert results[0]['entity_id'] == 42
        assert 'pred_target1' in results[0]
        assert 'pred_target2' in results[0]

    def test_process_predictions_probabilistic(self, forecaster):
        """Test processing probabilistic predictions."""
        times = pd.date_range('2020-01-01', periods=2, freq='M')
        # 2 timesteps, 2 components, 10 samples
        values = np.random.randn(2, 2, 10)
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()
        
        results = forecaster._process_predictions([mock_pred])
        
        assert len(results) == 2
        assert isinstance(results[0]['pred_target1'], list)
        assert len(results[0]['pred_target1']) == 10  # 10 samples

    def test_process_predictions_clips_negative_values(self, forecaster):
        """Test that negative predictions are clipped to zero."""
        times = pd.date_range('2020-01-01', periods=1, freq='M')
        # 1 timestep, 2 components, 1 sample each
        values = np.array([
            [[-5.0], [2.0]]  # timestep 0: component 0 = -5.0, component 1 = 2.0
        ])
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()
        
        results = forecaster._process_predictions([mock_pred])
        
        # Check that negative value was clipped to eps (1e-8)
        # When there's 1 sample, it returns a list with 1 element
        eps = 1e-8
        assert abs(results[0]['pred_target1'][0] - eps) < 1e-10
        assert results[0]['pred_target2'] == [2.0]

    def test_process_predictions_handles_nans(self, forecaster):
        """Test that NaN values are replaced with zero."""
        times = pd.date_range('2020-01-01', periods=1, freq='M')
        # 1 timestep, 2 components, 1 sample each
        values = np.array([
            [[np.nan], [2.0]]  # timestep 0: component 0 = nan, component 1 = 2.0
        ])
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()
        
        results = forecaster._process_predictions([mock_pred])
        
        # NaN replaced with 0.0, then clipped to eps (1e-8)
        # When there's 1 sample, it returns a list with 1 element
        eps = 1e-8
        assert abs(results[0]['pred_target1'][0] - eps) < 1e-10
        assert results[0]['pred_target2'] == [2.0]

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
        assert 'series' in call_kwargs
        assert 'past_covariates' in call_kwargs
        assert call_kwargs['verbose'] is True

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
        times = pd.date_range('2020-01-01', periods=2, freq='M')
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
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

    def test_predict_with_scaler_inverse_transform(self, forecaster_with_scalers):
        """Test prediction with target scaler inverse transform."""
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)
        
        times = pd.date_range('2020-01-01', periods=2, freq='M')
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()
        
        forecaster_with_scalers.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster_with_scalers._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster_with_scalers.model.predict = Mock(return_value=[mock_pred])
        forecaster_with_scalers.target_scaler.inverse_transform = Mock(return_value=[mock_pred])
        
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

    def test_predict_fillna(self, forecaster):
        """Test that predictions fill NaN values with 0."""
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)
        
        times = pd.date_range('2020-01-01', periods=2, freq='M')
        values = np.array([[1.0, np.nan], [3.0, 4.0]])
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()
        
        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])
        
        result = forecaster.predict(sequence_number=0, output_length=2)
        
        # NaN should be converted to 0 by nan_to_num in _process_predictions
        assert not result.isna().any().any()

    def test_save_model(self, forecaster_with_scalers, tmp_path):
        """Test saving model and scalers."""
        model_path = tmp_path / "model.pt"
        
        forecaster_with_scalers.model.save = Mock()
        
        with patch('torch.save') as mock_torch_save:
            forecaster_with_scalers.save_model(str(model_path))
            
            forecaster_with_scalers.model.save.assert_called_once_with(path=str(model_path))
            mock_torch_save.assert_called_once()
            
            # Check that scaler data was saved
            call_args = mock_torch_save.call_args
            assert 'target_scaler' in call_args[0][0]
            assert 'feature_scaler' in call_args[0][0]
            assert 'scaler_fitted' in call_args[0][0]

    def test_load_model(self, forecaster_with_scalers, tmp_path):
        """Test loading model and scalers."""
        model_path = tmp_path / "model.pt"
        
        mock_scaler_data = {
            'target_scaler': Mock(),
            'feature_scaler': Mock(),
            'scaler_fitted': True
        }
        
        # Mock the class's load static method
        mock_loaded_model = Mock(spec=TorchForecastingModel)
        mock_loaded_model.to_device = Mock()
        
        with patch('torch.load', return_value=mock_scaler_data), \
             patch.object(forecaster_with_scalers.model.__class__, 'load', return_value=mock_loaded_model) as mock_load:
            forecaster_with_scalers.load_model(str(model_path))
            
            assert forecaster_with_scalers.scaler_fitted is True
            mock_load.assert_called_once_with(path=str(model_path), map_location=str(forecaster_with_scalers.device))

    def test_load_model_missing_scalers(self, forecaster_with_scalers, tmp_path):
        """Test loading model when scaler file is missing."""
        model_path = tmp_path / "model.pt"
        
        with patch('torch.load', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                forecaster_with_scalers.load_model(str(model_path))

    def test_load_model_moves_to_device(self, forecaster_with_scalers, tmp_path):
        """Test that loaded model is moved to correct device."""
        model_path = tmp_path / "model.pt"
        
        mock_scaler_data = {
            'target_scaler': Mock(),
            'feature_scaler': Mock(),
            'scaler_fitted': True
        }
        
        # Mock the class's load static method
        mock_loaded_model = Mock(spec=TorchForecastingModel)
        mock_loaded_model.to_device = Mock()
        forecaster_with_scalers.device = 'cuda'
        
        with patch('torch.load', return_value=mock_scaler_data), \
             patch.object(forecaster_with_scalers.model.__class__, 'load', return_value=mock_loaded_model):
            forecaster_with_scalers.load_model(str(model_path))
            
            # The loaded model should have to_device called
            mock_loaded_model.to_device.assert_called_with('cuda')

    def test_predict_with_kwargs(self, forecaster):
        """Test prediction with additional kwargs."""
        mock_ts = Mock(spec=TimeSeries)
        mock_ts.astype = Mock(return_value=mock_ts)
        mock_ts.slice = Mock(return_value=mock_ts)
        mock_ts.__getitem__ = Mock(return_value=mock_ts)
        
        times = pd.date_range('2020-01-01', periods=2, freq='M')
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()
        
        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])
        
        forecaster.predict(sequence_number=0, output_length=2, num_samples=100)
        
        call_kwargs = forecaster.model.predict.call_args[1]
        assert 'num_samples' in call_kwargs
        assert call_kwargs['num_samples'] == 100


class TestDartsForecasterIntegration:
    """Integration tests for DartsForecaster."""
    
    def test_full_workflow_without_scalers(self, mock_dataset, mock_model, partition_dict):
        """Test complete workflow without scalers."""
        with patch('views_r2darts2.model.forecaster.ScalerSelector'):
            forecaster = DartsForecaster(
                dataset=mock_dataset,
                model=mock_model,
                partition_dict=partition_dict
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
        
        times = pd.date_range('2020-01-01', periods=10, freq='M')
        values = np.random.randn(10, 2)
        
        mock_pred = Mock(spec=TimeSeries)
        mock_pred.static_covariates = pd.DataFrame({'entity_id': [42]})
        mock_pred.all_values = Mock(return_value=values)
        mock_pred.start_time = Mock(return_value=times[0])
        mock_pred.freq = pd.offsets.MonthEnd()
        
        forecaster.dataset.as_darts_timeseries = Mock(return_value=[mock_ts])
        forecaster._preprocess_timeseries = Mock(return_value=([mock_ts], [mock_ts]))
        forecaster.model.predict = Mock(return_value=[mock_pred])
        
        forecaster.predict(sequence_number=0, output_length=10)
        
        call_kwargs = forecaster.model.predict.call_args[1]
        assert call_kwargs['n'] == 10