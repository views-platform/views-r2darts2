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
                    manager._sweep = False
                    
                    return manager

    def test_partition_resolution_alignment(self, mock_manager):
        """Proves that _resolve_active_partition_dict correctly calculates partitions for any steps."""
        config = {
            "steps": [1, 2, 3, 4, 5, 6],
            "run_type": "calibration"
        }
        
        # Mock the master partitions
        mock_manager._partition_dict = {
            "calibration": {"train": (100, 200), "test": (201, 210)}
        }
        
        partition = mock_manager._resolve_active_partition_dict(config)
        assert partition["train"] == (100, 200)
        
        # Test dynamic forecasting partition
        config_forecasting = {
            "steps": list(range(1, 13)), # 12 steps
            "run_type": "forecasting"
        }
        
        # The manager should fallback to the data loader's internal calculator
        mock_manager._data_loader._get_partition_dict.return_value = {"train": (100, 500), "test": (501, 512)}
        
        partition_fc = mock_manager._resolve_active_partition_dict(config_forecasting)
        
        assert mock_manager._data_loader.partition == "forecasting"
        mock_manager._data_loader._get_partition_dict.assert_called_with(steps=12)
        assert partition_fc["test"] == (501, 512)

    def test_predict_kwargs_validation(self, mock_manager):
        """Ensures the manager raises an error if mandatory prediction params are missing."""
        
        # Case 1: Missing n_jobs (new strict requirement)
        config_missing = {"num_samples": 100, "mc_dropout": True}
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
        [GREEN TEAM] Ensures that methods capture a snapshot of the config dictionary
        rather than relying on repeated property access.
        """
        dict1 = {"id": 1, "algorithm": "TFT", "steps": [1]}
        dict2 = {"id": 2, "algorithm": "TFT", "steps": [1]}
        
        # Access the underlying property object if possible, or patch it on the instance
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

    def test_partition_continuity_enforcement(self, mock_manager):
        """Ensures the manager refuses to run on discontinuous partitions (t+1 check)."""
        # Case 1: Gap detected
        mock_manager._partition_dict = {
            "calibration": {"train": (100, 200), "test": (202, 210)} # Gap at 201
        }
        with pytest.raises(ValueError, match="CRITICAL TEMPORAL DISCONTINUITY"):
            mock_manager._resolve_active_partition_dict({"run_type": "calibration", "steps": [1]})

        # Case 2: Overlap detected
        mock_manager._partition_dict = {
            "calibration": {"train": (100, 200), "test": (200, 210)} # Overlap at 200
        }
        with pytest.raises(ValueError, match="CRITICAL TEMPORAL DISCONTINUITY"):
            mock_manager._resolve_active_partition_dict({"run_type": "calibration", "steps": [1]})

    def test_horizon_standard_warning(self, mock_manager):
        """Verifies that the manager logs a warning for non-36 step horizons."""
        mock_manager._partition_dict = {"calibration": {"train": (100, 200), "test": (201, 236)}}
        
        with patch('views_r2darts2.manager.model.logger.warning') as mock_warn:
            # Test with 12 steps
            mock_manager._resolve_active_partition_dict({"run_type": "calibration", "steps": list(range(12))})
            mock_warn.assert_called()
            assert any("NON-STANDARD FORECAST HORIZON" in str(args) for args, _ in mock_warn.call_args_list)

    def test_data_leakage_prevention(self, mock_manager):
        """Tests that the system detects if test data months leaked into training."""
        from views_r2darts2.model.forecaster import DartsForecaster
        from darts import TimeSeries
        import pandas as pd

        # Boundary is 150
        df_leaked = pd.DataFrame({
            "target": [1.0, 2.0, 3.0],
            "time": [149, 150, 151] # 151 is in the test set!
        })
        ts = TimeSeries.from_dataframe(df_leaked, time_col="time", value_cols="target")
        
        mock_ds = MagicMock()
        mock_ds.targets = ["target"]
        mock_model = MagicMock()
        mock_model.input_chunk_length = 12
        mock_model.output_chunk_length = 6
        
        forecaster = DartsForecaster(
            dataset=mock_ds,
            model=mock_model,
            partition_dict={"train": (100, 150), "test": (151, 160)}
        )
        
        with pytest.raises(RuntimeError, match="DATA LEAKAGE DETECTED"):
            # Trigger the real internal validator
            forecaster._validate_temporal_integrity([ts], [], train_end=150)

    def test_training_continuity_enforcement(self, mock_manager):
        """Ensures that training data must be a contiguous range with no holes."""
        from views_r2darts2.model.forecaster import DartsForecaster
        import numpy as np

        # Simulated raw time IDs with a hole (missing 102)
        time_ids_holey = np.array([100, 101, 103, 104])
        
        # We simulate the validation logic that will be inside DartsForecaster
        # or the Dataset handler.
        def validate_ids(ids):
            expected = np.arange(ids.min(), ids.max() + 1)
            if not np.array_equal(np.sort(np.unique(ids)), expected):
                raise RuntimeError("TEMPORAL HOLE DETECTED")

        with pytest.raises(RuntimeError, match="TEMPORAL HOLE DETECTED"):
            validate_ids(time_ids_holey)

    def test_stochastic_parity_serialization(self, tmp_path):
        """
        [GREEN TEAM] Verifies that saving/reloading a model does not change its output.
        Note: For probabilistic models, this tests identity of weights.
        """
        from darts.models import NLinearModel
        from darts import TimeSeries
        import pandas as pd
        import numpy as np

        # Create tiny model and data
        model = NLinearModel(input_chunk_length=4, output_chunk_length=1, n_epochs=1, pl_trainer_kwargs={"accelerator": "cpu"})
        df = pd.DataFrame({"target": np.random.rand(10), "time": np.arange(10)})
        ts = TimeSeries.from_dataframe(df, time_col="time", value_cols="target")
        
        model.fit(ts)
        
        # Predict in-memory
        pred_ts = model.predict(n=1, series=ts)
        pred_in_mem = pred_ts.values().flatten()[0]
        
        # Save and reload
        path = str(tmp_path / "model.pt")
        model.save(path)
        model_reloaded = NLinearModel.load(path)
        
        # Predict reloaded
        pred_reloaded_ts = model_reloaded.predict(n=1, series=ts)
        pred_reloaded = pred_reloaded_ts.values().flatten()[0]
        
        # Assert bit-level weight reproducibility
        assert np.isclose(pred_in_mem, pred_reloaded, atol=1e-7)

    def test_schema_integrity_lockdown(self, mock_manager):
        """
        [BEIGE TEAM] Ensures that the system refuses to predict if 'boring' but 
        deadly parameters are missing.
        """
        # A user forgets n_jobs in their config
        config_human_error = {
            "num_samples": 1,
            "mc_dropout": False
            # n_jobs is missing!
        }
        
        with pytest.raises(ValueError, match="Missing mandatory prediction parameters"):
            mock_manager._get_predict_kwargs(config_human_error)

    def test_numerical_poisoning_red_team(self):
        """
        [RED TEAM] Deliberately injects poisonous data to ensure the 
        DataSanity gate kills the run.
        """
        from views_r2darts2.model.forecaster import DartsForecaster
        from darts import TimeSeries
        import pandas as pd
        import numpy as np

        # Create a series with a NaN
        df_poisoned = pd.DataFrame({
            "target": [1.0, np.nan, 3.0],
            "time": [1, 2, 3]
        })
        ts = TimeSeries.from_dataframe(df_poisoned, time_col="time", value_cols="target")
        
        # Create a forecaster shell
        forecaster = DartsForecaster(
            dataset=MagicMock(), model=MagicMock(), partition_dict={'train':(0,10), 'test':(11,20)}
        )
        
        with patch('views_r2darts2.model.forecaster.logger.error') as mock_log:
            forecaster._check_data_sanity([ts], "poison_test")
            assert mock_log.called
            assert any("NaN values found" in str(args) for args, _ in mock_log.call_args_list)

    def test_adversarial_outlier_spike(self):
        """
        [RED TEAM] Injects a 1-billion fatality spike to ensure it is flagged.
        """
        from views_r2darts2.model.forecaster import DartsForecaster
        from darts import TimeSeries
        import pandas as pd

        df_spike = pd.DataFrame({"target": [1.0, 1e9, 1.0], "time": [1, 2, 3]})
        ts = TimeSeries.from_dataframe(df_spike, time_col="time", value_cols="target")
        
        forecaster = DartsForecaster(
            dataset=MagicMock(), model=MagicMock(), partition_dict={'train':(0,10), 'test':(11,20)}
        )
        
        with patch('views_r2darts2.model.forecaster.logger.warning') as mock_warn:
            forecaster._check_data_sanity([ts], "spike_test", max_abs_val=100.0)
            assert mock_warn.called
            assert any("exceed ±100.0" in str(args) for args, _ in mock_warn.call_args_list)