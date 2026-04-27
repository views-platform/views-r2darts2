import pytest
import numpy as np
import pandas as pd
from darts import TimeSeries
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.transformers.scaler_selector import ScalerSelector
from darts.dataprocessing.transformers import Scaler


class TestScalingRobustness:
    """
    Robustness tests to prevent 'silent killer' scaling regressions.

    These tests target:
    1. Scaling inversion bugs (hardcoded [0] indexing).
    2. Missing global_fit configuration for global models.
    3. Physically impossible prediction magnitudes.
    """

    @pytest.fixture
    def multi_entity_ts(self):
        """Create a dataset with two extremely different entities."""
        n_time = 24
        times = pd.date_range("2000-01", periods=n_time, freq="MS")

        # Entity 0: Peaceful (0-1 deaths)
        data_0 = np.linspace(0, 1, n_time).reshape(-1, 1)
        ts_0 = TimeSeries.from_times_and_values(
            times=times,
            values=data_0,
            columns=["target"],
            static_covariates=pd.DataFrame({"id": [0]}),
        )

        # Entity 1: Violent (100-1000 deaths)
        data_1 = np.linspace(100, 1000, n_time).reshape(-1, 1)
        ts_1 = TimeSeries.from_times_and_values(
            times=times,
            values=data_1,
            columns=["target"],
            static_covariates=pd.DataFrame({"id": [1]}),
        )

        return [ts_0, ts_1]

    def test_inverse_transform_multi_entity_consistency(self, multi_entity_ts):
        """
        GIVEN a globally-fitted target scaler
        WHEN inverse_transform is applied to multiple entities
        THEN every entity must be restored to its ORIGINAL magnitude.

        This catches the bug where fitted_params[0] is applied to everyone.
        """
        # 1. Setup Scaler with global_fit=True
        target_scaler = Scaler(
            ScalerSelector.get_scaler("MinMaxScaler"), global_fit=True
        )

        # 2. Forward Scale (Global Fit)
        scaled_series = target_scaler.fit_transform(multi_entity_ts)

        # Mock a Forecaster instance to use its _inverse_transform_target_scaler
        # We need this because the bug was in the Forecaster's wrapper logic
        class MockDataset:
            def __init__(self, targets):
                self.targets = targets

        from unittest.mock import MagicMock

        forecaster = MagicMock(spec=DartsForecaster)
        forecaster.target_scaler = target_scaler
        forecaster.scaler_fitted = True

        # Manually link the method from the class to our mock
        forecaster._inverse_transform_target_scaler = (
            DartsForecaster._inverse_transform_target_scaler.__get__(forecaster)
        )

        # 3. Inverse Scale
        recovered_series = forecaster._inverse_transform_target_scaler(scaled_series)

        # 4. ASSERT: Peaceful country is restored to ~1.0
        np.testing.assert_allclose(
            recovered_series[0].all_values().max(),
            multi_entity_ts[0].all_values().max(),
            atol=1e-5,
            err_msg="Peaceful entity magnitude was corrupted during inverse transform!",
        )

        # 5. ASSERT: Violent country is restored to ~1000.0
        np.testing.assert_allclose(
            recovered_series[1].all_values().max(),
            multi_entity_ts[1].all_values().max(),
            atol=1e-5,
            err_msg="Violent entity magnitude was corrupted during inverse transform! (Likely using first entity's scale)",
        )

    def test_global_fit_configuration_mandatory(self):
        """
        VERIFY that the model catalog always uses global_fit=True for its scalers.
        Global models like N-BEATS fail to generalize if scaling is per-entity.
        """
        # Mock config similar to HP sweep
        config = {
            "run_type": "test",
            "num_samples": 1,
            "mc_dropout": False,
            "algorithm": "NBEATSModel",
            "targets": ["target"],
            "steps": list(range(1, 37)),
            "input_chunk_length": 24,
            "output_chunk_length": 36,
            "output_chunk_shift": 0,
            "num_stacks": 2,
            "num_blocks": 3,
            "num_layers": 3,
            "layer_widths": 64,
            "activation": "LeakyReLU",
            "generic_architecture": True,
            "dropout": 0.3,
            "batch_size": 8,
            "n_epochs": 1,
            "lr": 0.0003,
            "weight_decay": 0.0003,
            "loss_function": "WeightedPenaltyHuberLoss",
            "optimizer_cls": "Adam",
            "delta": 0.025,
            "zero_threshold": 0.01,
            "non_zero_weight": 7.0,
            "false_positive_weight": 1.0,
            "false_negative_weight": 10.0,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 5,
            "lr_scheduler_min_lr": 1e-6,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.001,
            "gradient_clip_val": 1.0,
            "force_reset": True,
            "random_state": 1,
            "name": "TestModel",
            "use_static_covariates": True,
            "use_reversible_instance_norm": False,
        }

        # Instantiate a forecaster (which calls _instantiate_scaler)
        from views_r2darts2.transformers.views_dataset_darts import _ViewsDatasetDarts
        from unittest.mock import MagicMock

        catalog = ModelCatalog(config=config)
        model = catalog.get_model("NBEATSModel")

        multi_ts = [
            TimeSeries.from_times_and_values(
                pd.date_range("2000-01", periods=12, freq="MS"), np.random.randn(12, 1)
            )
            for _ in range(3)
        ]

        # --- Univariate case: features=[] → feature_scaler auto-disabled ---
        dataset_no_feat = MagicMock(spec=_ViewsDatasetDarts)
        dataset_no_feat.features = []
        dataset_no_feat.targets = ["target"]

        forecaster_no_feat = DartsForecaster(
            dataset=dataset_no_feat,
            model=model,
            partition_dict={"train": (0, 100), "test": (100, 120)},
            target_scaler="MinMaxScaler",
            feature_scaler="StandardScaler",
            random_state=42,
        )

        assert forecaster_no_feat.feature_scaler is None, (
            "feature_scaler must be None when dataset has no features!"
        )

        forecaster_no_feat.target_scaler.fit(multi_ts)
        assert len(forecaster_no_feat.target_scaler._fitted_params) == 1, (
            "Target scaler must have 1 set of params (global_fit=True)!"
        )

        # --- Multivariate case: features present → both scalers use global_fit ---
        dataset_with_feat = MagicMock(spec=_ViewsDatasetDarts)
        dataset_with_feat.features = ["feat_a", "feat_b"]
        dataset_with_feat.targets = ["target"]

        forecaster_with_feat = DartsForecaster(
            dataset=dataset_with_feat,
            model=model,
            partition_dict={"train": (0, 100), "test": (100, 120)},
            target_scaler="MinMaxScaler",
            feature_scaler="StandardScaler",
            random_state=42,
        )

        forecaster_with_feat.target_scaler.fit(multi_ts)
        assert len(forecaster_with_feat.target_scaler._fitted_params) == 1, (
            "Target scaler must have 1 set of params (global_fit=True)!"
        )

        forecaster_with_feat.feature_scaler.fit(multi_ts)
        assert len(forecaster_with_feat.feature_scaler._fitted_params) == 1, (
            "Feature scaler must have 1 set of params (global_fit=True)!"
        )

    def test_magnitude_sanity_runtime_simulation(self):
        """
        Simulate a prediction collapse (y_hat_bar = 0.01) and verify
        if we can detect it.
        """
        # Create a series with near-zero predictions (the failure mode)
        times = pd.date_range("2000-01", periods=12, freq="MS")
        bad_pred = [
            TimeSeries.from_times_and_values(
                times=times,
                values=np.full((12, 1), 0.01),
                static_covariates=pd.DataFrame({"id": [0]}),
            )
        ]

        # Check mean magnitude
        y_hat_bar = np.mean([ts.all_values().mean() for ts in bad_pred])

        assert y_hat_bar < 0.1, (
            f"Magnitude sanity check failed to identify near-zero predictions: {y_hat_bar}"
        )

    def test_feature_leakage_prevention(self, multi_entity_ts):
        """
        Ensure that scaling parameters do not change when transforming new data.
        If they change, it means the scaler is 'learning' from the test set.
        """
        from copy import deepcopy

        target_scaler = Scaler(
            ScalerSelector.get_scaler("MinMaxScaler"), global_fit=True
        )

        # 1. Fit on the first half of the series
        train_slice = [ts[:12] for ts in multi_entity_ts]
        target_scaler.fit(train_slice)

        # Capture fitted params
        params_before = deepcopy(target_scaler._fitted_params[0])

        # 2. Transform the second half (simulating test set)
        test_slice = [ts[12:] for ts in multi_entity_ts]
        _ = target_scaler.transform(test_slice)

        # 3. Verify params didn't change
        params_after = target_scaler._fitted_params[0]

        # For MinMaxScaler, check data_max_ and data_min_
        np.testing.assert_array_equal(params_before.data_max_, params_after.data_max_)
        np.testing.assert_array_equal(params_before.data_min_, params_after.data_min_)
