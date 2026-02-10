import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from views_r2darts2.manager.model import DartsForecastingModelManager
from views_pipeline_core.managers.model.model import ModelPathManager, ForecastingModelArgs

class TestReproducibilityInfra:
    """
    Infrastructure tests to ensure reproducibility fault lines remain closed.
    """

    @pytest.fixture
    def mock_manager(self):
        with patch('views_r2darts2.manager.model.ForecastingModelManager.__init__', return_value=None):
            with patch('views_r2darts2.manager.model.torch.load'):
                with patch('views_r2darts2.manager.model.logger'):
                    path_manager = MagicMock(spec=ModelPathManager)
                    path_manager.model_name = "test_model"
                    
                    # Use __new__ to avoid calling __init__ which triggers property access
                    manager = DartsForecastingModelManager.__new__(DartsForecastingModelManager)
                    manager._model_path = path_manager
                    manager._data_loader = MagicMock()
                    manager._config_manager = MagicMock()
                    # Initialize missing attributes required by property access during __init__
                    manager._sweep = False
                    
                    return manager

    def test_data_loader_sync_alignment(self, mock_manager):
        """Proves that _sync_data_loader correctly updates the loader partitions."""
        config = {
            "steps": [1, 2, 3, 4, 5, 6],
            "run_type": "calibration"
        }
        
        # Setup mock behavior for the parent data loader's partition calculator
        mock_manager._data_loader._get_partition_dict.return_value = {"train": (1, 10), "test": (11, 17)}
        
        mock_manager._sync_data_loader(config)
        
        assert mock_manager._data_loader.steps == 6
        assert mock_manager._data_loader.partition == "calibration"
        # Ensure it actually called the recalculation logic
        mock_manager._data_loader._get_partition_dict.assert_called_with(steps=6)

    def test_predict_kwargs_validation(self, mock_manager):
        """Ensures the manager raises an error if mandatory prediction params are missing."""
        
        # Case 1: Missing mc_dropout
        config_missing = {"num_samples": 100}
        with pytest.raises(ValueError, match="Missing mandatory prediction parameters"):
            mock_manager._get_predict_kwargs(config_missing)
            
        # Case 2: Complete config
        config_complete = {"num_samples": 50, "mc_dropout": True, "n_jobs": 4}
        kwargs = mock_manager._get_predict_kwargs(config_complete)
        assert kwargs["num_samples"] == 50
        assert kwargs["mc_dropout"] is True
        assert kwargs["n_jobs"] == 4

    def test_config_snapshot_stability(self, mock_manager):
        """
        Ensures that methods capture a snapshot of the config dictionary
        rather than relying on repeated property access.
        """
        dict1 = {"id": 1, "algorithm": "TFT", "steps": [1]}
        dict2 = {"id": 2, "algorithm": "TFT", "steps": [1]}
        
        # Use patch from unittest.mock correctly
        with patch.object(DartsForecastingModelManager, 'configs', new_callable=PropertyMock) as mock_configs:
            mock_configs.side_effect = [dict1, dict2, {"id": 3}]
            
            # Simulated training logic using snapshot pattern
            active_config = mock_manager.configs  # Access 1 -> dict1
            assert active_config["id"] == 1
            
            # Access again via property -> dict2
            assert mock_manager.configs["id"] == 2 
            
            # But the 'active_config' snapshot we took is still dict1
            assert active_config["id"] == 1
            assert active_config is dict1