# import pytest
# import pandas as pd
# import torch
# from pathlib import Path
# from unittest.mock import Mock, MagicMock, patch, call, mock_open
# from views_r2darts2.manager.model import DartsForecastingModelManager, custom_torch_load
# from views_pipeline_core.managers.model import ModelPathManager
# from views_r2darts2.model.forecaster import DartsForecaster
# from views_r2darts2.data.handlers import _ViewsDatasetDarts
# from views_r2darts2.model.catalog import ModelCatalog


# @pytest.fixture
# def mock_model_path():
#     """Create a mock ModelPathManager."""
#     mock_path = Mock(spec=ModelPathManager)
#     mock_path.data_raw = Path("/fake/data/raw")
#     mock_path.artifacts = Path("/fake/artifacts")
#     mock_path.get_latest_model_artifact_path = Mock(
#         return_value=Path("/fake/artifacts/model_20231215_120000.pt")
#     )
#     return mock_path


# @pytest.fixture
# def mock_config():
#     """Create a mock configuration dictionary."""
#     return {
#         "run_type": "calibration",
#         "algorithm": "TFTModel",
#         "targets": ["target1", "target2"],
#         "steps": [1, 2, 3, 6, 12, 24, 36],
#         "sweep": False,
#         "feature_scaler": "StandardScaler",
#         "target_scaler": "RobustScaler",
#         "num_samples": 100,
#         "n_jobs": 4,
#         "mc_dropout": True,
#     }


# @pytest.fixture
# def mock_data_loader():
#     """Create a mock data loader."""
#     loader = Mock()
#     loader.partition_dict = {
#         "train": (0, 100),
#         "test": (100, 150)
#     }
#     return loader


# @pytest.fixture
# def model_manager(mock_model_path, mock_config, mock_data_loader):
#     """Create a DartsForecastingModelManager instance."""
#     with patch('views_r2darts2.manager.model.ForecastingModelManager.__init__', return_value=None) as mock_init, \
#          patch('views_r2darts2.manager.model.torch.load'), \
#          patch('views_r2darts2.manager.model.logger'):
#         # Manually set configs after mocking __init__
#         manager = DartsForecastingModelManager.__new__(DartsForecastingModelManager)
#         # Set the private attribute that backs the configs property
#         manager._configs = mock_config
#         manager._model_path = mock_model_path
#         manager.config = mock_config
#         manager._data_loader = mock_data_loader
        
#         # Call the real __init__ but with mocked parent
#         DartsForecastingModelManager.__init__(
#             manager,
#             model_path=mock_model_path,
#             wandb_notifications=True,
#             use_prediction_store=True
#         )
#     return manager


# class TestCustomTorchLoad:
#     """Tests for custom_torch_load function."""
    
#     def test_custom_torch_load_sets_weights_only_false(self):
#         """Test that custom_torch_load sets weights_only=False by default."""
#         mock_original = Mock()
        
#         with patch('views_r2darts2.manager.model._original_torch_load', mock_original):
#             custom_torch_load("model.pt")
            
#             mock_original.assert_called_once()
#             call_kwargs = mock_original.call_args[1]
#             assert call_kwargs['weights_only'] is False

#     def test_custom_torch_load_respects_explicit_weights_only(self):
#         """Test that custom_torch_load respects explicitly set weights_only parameter."""
#         mock_original = Mock()
        
#         with patch('views_r2darts2.manager.model._original_torch_load', mock_original):
#             custom_torch_load("model.pt", weights_only=True)
            
#             call_kwargs = mock_original.call_args[1]
#             assert call_kwargs['weights_only'] is True

#     def test_custom_torch_load_passes_all_args(self):
#         """Test that custom_torch_load passes all arguments correctly."""
#         mock_original = Mock()
        
#         with patch('views_r2darts2.manager.model._original_torch_load', mock_original):
#             custom_torch_load("model.pt", map_location="cpu", pickle_module=None)
            
#             mock_original.assert_called_once_with(
#                 "model.pt",
#                 map_location="cpu",
#                 pickle_module=None,
#                 weights_only=False
#             )

#     def test_custom_torch_load_returns_result(self):
#         """Test that custom_torch_load returns the result from original function."""
#         mock_result = {"state": "data"}
#         mock_original = Mock(return_value=mock_result)
        
#         with patch('views_r2darts2.manager.model._original_torch_load', mock_original):
#             result = custom_torch_load("model.pt")
            
#             assert result == mock_result


# class TestDartsForecastingModelManager:
#     """Tests for DartsForecastingModelManager class."""
    
#     def test_initialization(self, mock_model_path):
#         """Test model manager initialization."""
#         with patch('views_r2darts2.manager.model.ForecastingModelManager.__init__', return_value=None) as mock_super_init, \
#              patch('views_r2darts2.manager.model.torch') as mock_torch, \
#              patch('views_r2darts2.manager.model.logger') as mock_logger:
            
#             # Create instance and manually set configs before __init__
#             manager = DartsForecastingModelManager.__new__(DartsForecastingModelManager)
#             manager.configs = {"algorithm": "TFTModel"}
            
#             # Now call __init__
#             DartsForecastingModelManager.__init__(
#                 manager,
#                 model_path=mock_model_path,
#                 wandb_notifications=True,
#                 use_prediction_store=True
#             )
            
#             mock_super_init.assert_called_once_with(
#                 model_path=mock_model_path,
#                 wandb_notifications=True,
#                 use_prediction_store=True
#             )
            
#             # Verify torch.load was overridden
#             assert mock_torch.load == custom_torch_load
            
#             # Verify logger was called
#             mock_logger.info.assert_called_once()

#     def test_initialization_logs_algorithm(self, mock_model_path):
#         """Test that initialization logs the algorithm."""
#         with patch('views_r2darts2.manager.model.ForecastingModelManager.__init__', return_value=None), \
#              patch('views_r2darts2.manager.model.torch'), \
#              patch('views_r2darts2.manager.model.logger') as mock_logger:
            
#             # Create instance and manually set configs
#             manager = DartsForecastingModelManager.__new__(DartsForecastingModelManager)
#             manager.configs = {"algorithm": "TFTModel"}
            
#             DartsForecastingModelManager.__init__(manager, model_path=mock_model_path)
            
#             # Verify the log message was called with the algorithm
#             expected_msg = f"Current model architecture: \033[92mTFTModel\033[0m"
#             mock_logger.info.assert_called_once_with(expected_msg)

#     def test_initialization_with_default_parameters(self, mock_model_path):
#         """Test initialization with default parameters."""
#         with patch('views_r2darts2.manager.model.ForecastingModelManager.__init__', return_value=None) as mock_super_init, \
#              patch('views_r2darts2.manager.model.torch'), \
#              patch('views_r2darts2.manager.model.logger'):
            
#             # Create instance and manually set configs
#             manager = DartsForecastingModelManager.__new__(DartsForecastingModelManager)
#             manager.configs = {"algorithm": "TFTModel"}
            
#             DartsForecastingModelManager.__init__(manager, model_path=mock_model_path)
            
#             # Verify defaults are passed to parent
#             call_kwargs = mock_super_init.call_args[1]
#             assert call_kwargs['wandb_notifications'] is True
#             assert call_kwargs['use_prediction_store'] is True

#     def test_train_model_artifact(self, model_manager, mock_config):
#         """Test training a model artifact."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts') as mock_dataset, \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster) as mock_forecaster_class, \
#              patch('views_r2darts2.manager.model.generate_model_file_name', return_value='model_test.pt'):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             result = model_manager._train_model_artifact()
            
#             # Verify dataset was created
#             mock_dataset.assert_called_once()
#             dataset_kwargs = mock_dataset.call_args[1]
#             assert dataset_kwargs['targets'] == mock_config['targets']
#             assert dataset_kwargs['broadcast_features'] is True
            
#             # Verify forecaster was created
#             mock_forecaster_class.assert_called_once()
#             forecaster_kwargs = mock_forecaster_class.call_args[1]
#             assert forecaster_kwargs['model'] == mock_model
#             assert forecaster_kwargs['feature_scaler'] == 'StandardScaler'
#             assert forecaster_kwargs['target_scaler'] == 'RobustScaler'
            
#             # Verify training was called
#             mock_forecaster.train.assert_called_once()
            
#             # Verify model was saved
#             mock_forecaster.save_model.assert_called_once()
            
#             assert result == mock_forecaster

#     def test_train_model_artifact_with_sweep(self, model_manager):
#         """Test training with sweep enabled (no save)."""
#         model_manager.config['sweep'] = True
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             result = model_manager._train_model_artifact()
            
#             # Verify model was NOT saved
#             mock_forecaster.save_model.assert_not_called()

#     def test_train_model_artifact_without_scalers(self, model_manager):
#         """Test training without feature and target scalers."""
#         model_manager.config['feature_scaler'] = None
#         model_manager.config['target_scaler'] = None
        
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster) as mock_forecaster_class:
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             model_manager._train_model_artifact()
            
#             forecaster_kwargs = mock_forecaster_class.call_args[1]
#             assert forecaster_kwargs['feature_scaler'] is None
#             assert forecaster_kwargs['target_scaler'] is None

#     def test_evaluate_model_artifact_with_artifact_name(self, model_manager):
#         """Test evaluating a specific model artifact."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
#         mock_prediction = pd.DataFrame({'pred': [1, 2, 3]})
#         mock_forecaster.predict.return_value = mock_prediction
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster), \
#              patch.object(
#                  DartsForecastingModelManager, 
#                  '_resolve_evaluation_sequence_number', 
#                  return_value=3
#              ):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             result = model_manager._evaluate_model_artifact(
#                 eval_type="standard",
#                 artifact_name="custom_model.pt"
#             )
            
#             # Verify artifact was loaded
#             expected_path = model_manager._model_path.artifacts / "custom_model.pt"
#             mock_forecaster.load_model.assert_called_once_with(path=expected_path)
            
#             # Verify predictions were made
#             assert mock_forecaster.predict.call_count == 3
#             assert len(result) == 3

#     def test_evaluate_model_artifact_default_artifact(self, model_manager):
#         """Test evaluating with default (latest) artifact."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
#         mock_prediction = pd.DataFrame({'pred': [1, 2, 3]})
#         mock_forecaster.predict.return_value = mock_prediction
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster), \
#              patch.object(
#                  DartsForecastingModelManager, 
#                  '_resolve_evaluation_sequence_number', 
#                  return_value=2
#              ):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             result = model_manager._evaluate_model_artifact(eval_type="standard")
            
#             # Verify latest artifact path was retrieved
#             model_manager._model_path.get_latest_model_artifact_path.assert_called_once_with(
#                 "calibration"
#             )
            
#             # Verify predictions were made
#             assert len(result) == 2

#     def test_evaluate_model_artifact_with_mc_params(self, model_manager):
#         """Test evaluation with Monte Carlo parameters."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
#         mock_prediction = pd.DataFrame({'pred': [1, 2, 3]})
#         mock_forecaster.predict.return_value = mock_prediction
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster), \
#              patch.object(
#                  DartsForecastingModelManager, 
#                  '_resolve_evaluation_sequence_number', 
#                  return_value=1
#              ):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             model_manager._evaluate_model_artifact(eval_type="standard")
            
#             # Verify predict was called with MC parameters
#             call_kwargs = mock_forecaster.predict.call_args[1]
#             assert call_kwargs['num_samples'] == 100
#             assert call_kwargs['n_jobs'] == 4
#             assert call_kwargs['mc_dropout'] is True

#     def test_evaluate_model_artifact_extracts_timestamp(self, model_manager):
#         """Test that timestamp is extracted from artifact name."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster), \
#              patch.object(
#                  DartsForecastingModelManager, 
#                  '_resolve_evaluation_sequence_number', 
#                  return_value=1
#              ):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = Mock()
#             mock_catalog.return_value = mock_catalog_instance
            
#             model_manager._evaluate_model_artifact(
#                 eval_type="standard",
#                 artifact_name="model_20231215_120000.pt"
#             )
            
#             # Verify timestamp was extracted (last 15 characters of stem)
#             assert model_manager.config['timestamp'] == '15_120000.pt'[:15]

#     def test_forecast_model_artifact(self, model_manager):
#         """Test forecasting with a model artifact."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
#         mock_prediction = pd.DataFrame({'pred': [1, 2, 3]})
#         mock_forecaster.predict.return_value = mock_prediction
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             result = model_manager._forecast_model_artifact(artifact_name="model.pt")
            
#             # Verify predict was called with sequence_number=0
#             mock_forecaster.predict.assert_called_once()
#             call_args = mock_forecaster.predict.call_args
#             assert call_args[0][0] == 0  # sequence_number
#             assert call_args[0][1] == max(model_manager.config['steps'])  # output_length
            
#             assert result.equals(mock_prediction)

#     def test_forecast_model_artifact_default_artifact(self, model_manager):
#         """Test forecasting with default artifact."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             model_manager._forecast_model_artifact(artifact_name=None)
            
#             # Verify latest artifact was used
#             model_manager._model_path.get_latest_model_artifact_path.assert_called_once()

#     def test_evaluate_sweep(self, model_manager):
#         """Test sweep evaluation."""
#         mock_model = Mock()
#         mock_prediction = pd.DataFrame({'pred': [1, 2, 3]})
#         mock_model.predict.return_value = mock_prediction
        
#         with patch.object(
#             DartsForecastingModelManager, 
#             '_resolve_evaluation_sequence_number', 
#             return_value=5
#         ):
#             result = model_manager._evaluate_sweep(eval_type="standard", model=mock_model)
            
#             # Verify predictions were made for all sequences
#             assert mock_model.predict.call_count == 5
#             assert len(result) == 5
            
#             # Verify correct sequence numbers were used
#             for i in range(5):
#                 call_args = mock_model.predict.call_args_list[i]
#                 assert call_args[0][0] == i

#     def test_evaluate_sweep_uses_max_steps(self, model_manager):
#         """Test that sweep evaluation uses max steps."""
#         mock_model = Mock()
#         model_manager.config['steps'] = [1, 3, 6, 12, 36]
        
#         with patch.object(
#             DartsForecastingModelManager, 
#             '_resolve_evaluation_sequence_number', 
#             return_value=1
#         ):
#             model_manager._evaluate_sweep(eval_type="standard", model=mock_model)
            
#             call_args = mock_model.predict.call_args
#             assert call_args[0][1] == 36  # max(steps)


# class TestDartsForecastingModelManagerIntegration:
#     """Integration tests for DartsForecastingModelManager."""
    
#     def test_train_and_evaluate_workflow(self, model_manager):
#         """Test complete train and evaluate workflow."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
#         mock_prediction = pd.DataFrame({'pred': [1, 2, 3]})
#         mock_forecaster.predict.return_value = mock_prediction
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster), \
#              patch('views_r2darts2.manager.model.generate_model_file_name', return_value='model.pt'), \
#              patch.object(
#                  DartsForecastingModelManager, 
#                  '_resolve_evaluation_sequence_number', 
#                  return_value=2
#              ):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             # Train
#             trained_forecaster = model_manager._train_model_artifact()
#             assert trained_forecaster == mock_forecaster
#             mock_forecaster.train.assert_called_once()
            
#             # Evaluate
#             predictions = model_manager._evaluate_model_artifact(eval_type="standard")
#             assert len(predictions) == 2

#     def test_different_run_types(self, model_manager):
#         """Test manager works with different run types."""
#         run_types = ["calibration", "testing", "forecasting"]
        
#         for run_type in run_types:
#             model_manager.config['run_type'] = run_type
#             mock_df = pd.DataFrame({'col1': [1, 2, 3]})
            
#             with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df) as mock_read:
#                 with patch('views_r2darts2.manager.model.ModelCatalog'), \
#                      patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#                      patch('views_r2darts2.manager.model.DartsForecaster'), \
#                      patch('views_r2darts2.manager.model.generate_model_file_name', return_value='model.pt'):
                    
#                     model_manager._train_model_artifact()
                    
#                     # Verify correct file was read
#                     expected_filename = f"{run_type}_viewser_df.parquet"
#                     assert expected_filename in str(mock_read.call_args[0][0])

#     def test_evaluation_types(self, model_manager):
#         """Test different evaluation types."""
#         eval_types = ["standard", "long", "complete", "live"]
        
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster), \
#              patch.object(
#                  DartsForecastingModelManager, 
#                  '_resolve_evaluation_sequence_number', 
#                  return_value=1
#              ) as mock_resolve:
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             for eval_type in eval_types:
#                 model_manager._evaluate_model_artifact(eval_type=eval_type)
                
#                 # Verify resolve was called with correct eval_type
#                 assert mock_resolve.call_args[0][0] == eval_type

#     def test_model_path_management(self, model_manager):
#         """Test model path management across operations."""
#         mock_df = pd.DataFrame({'col1': [1, 2, 3]})
#         mock_model = Mock()
#         mock_forecaster = Mock(spec=DartsForecaster)
        
#         with patch('views_r2darts2.manager.model.read_dataframe', return_value=mock_df), \
#              patch('views_r2darts2.manager.model.ModelCatalog') as mock_catalog, \
#              patch('views_r2darts2.manager.model._ViewsDatasetDarts'), \
#              patch('views_r2darts2.manager.model.DartsForecaster', return_value=mock_forecaster), \
#              patch('views_r2darts2.manager.model.generate_model_file_name', return_value='model.pt'):
            
#             mock_catalog_instance = Mock()
#             mock_catalog_instance.get_model.return_value = mock_model
#             mock_catalog.return_value = mock_catalog_instance
            
#             # Train saves to artifacts path
#             model_manager._train_model_artifact()
#             save_path = mock_forecaster.save_model.call_args[1]['path']
#             assert str(model_manager._model_path.artifacts) in save_path
            
#             # Evaluate loads from artifacts path
#             with patch.object(
#                 DartsForecastingModelManager, 
#                 '_resolve_evaluation_sequence_number', 
#                 return_value=1
#             ):
#                 model_manager._evaluate_model_artifact(eval_type="standard")
#                 load_path = mock_forecaster.load_model.call_args[1]['path']
#                 assert isinstance(load_path, Path)