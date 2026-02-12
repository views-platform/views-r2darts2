import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from darts import TimeSeries
import pandas as pd
from views_r2darts2.utils.scaling import ScalerSelector, FeatureScalerManager


class TestScalerSelector:
    def test_get_standard_scaler(self):
        scaler = ScalerSelector.get_scaler("StandardScaler")
        assert scaler.__class__.__name__ == "StandardScaler"

    def test_get_robust_scaler(self):
        scaler = ScalerSelector.get_scaler("RobustScaler")
        assert scaler.__class__.__name__ == "RobustScaler"

    def test_get_minmax_scaler(self):
        scaler = ScalerSelector.get_scaler("MinMaxScaler")
        assert scaler.__class__.__name__ == "MinMaxScaler"

    def test_get_maxabs_scaler(self):
        scaler = ScalerSelector.get_scaler("MaxAbsScaler")
        assert scaler.__class__.__name__ == "MaxAbsScaler"

    def test_get_yeojohnson_transform(self):
        scaler = ScalerSelector.get_scaler("YeoJohnsonTransform")
        assert scaler.__class__.__name__ == "PowerTransformer"
        assert scaler.method == "yeo-johnson"

    def test_unknown_scaler_raises_error(self):
        with pytest.raises(
            ValueError, match="Scaler 'UnknownScaler' is not recognized"
        ):
            ScalerSelector.get_scaler("UnknownScaler")

    def test_scaler_with_kwargs(self):
        scaler = ScalerSelector.get_scaler("RobustScaler", quantile_range=(10, 90))
        assert scaler.quantile_range == (10, 90)

    def test_available_scalers_list(self):
        try:
            ScalerSelector.get_scaler("InvalidScaler")
        except ValueError as e:
            error_msg = str(e)
            assert "StandardScaler" in error_msg
            assert "RobustScaler" in error_msg
            assert "YeoJohnsonTransform" in error_msg

    def test_scaler_with_multiple_kwargs(self):
        scaler = ScalerSelector.get_scaler(
            "StandardScaler", with_mean=False, with_std=True
        )
        assert not scaler.with_mean
        assert scaler.with_std

    def test_minmax_scaler_with_feature_range(self):
        scaler = ScalerSelector.get_scaler("MinMaxScaler", feature_range=(-1, 1))
        assert scaler.feature_range == (-1, 1)

    def test_power_transformer_standardize_param(self):
        scaler = ScalerSelector.get_scaler("YeoJohnsonTransform", standardize=False)
        assert not scaler.standardize

    def test_scaler_is_fitted_false_initially(self):
        """Test that all scalers start unfitted."""
        scaler = ScalerSelector.get_scaler("StandardScaler")
        with pytest.raises(NotFittedError):
            scaler.transform(np.array([[1, 2, 3]]))

    def test_scaler_can_be_fitted_and_used(self):
        """Test that scalers can be fitted and used for transformation."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = ScalerSelector.get_scaler("StandardScaler")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape
        # StandardScaler should produce mean ~0 and std ~1
        assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_transformed.std(axis=0), 1, atol=1e-7)

    def test_robust_scaler_functionality(self):
        """Test RobustScaler removes median and scales by IQR."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        scaler = ScalerSelector.get_scaler("RobustScaler")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape
        # RobustScaler should produce median ~0
        assert np.allclose(np.median(X_transformed, axis=0), 0, atol=1e-7)

    def test_minmax_scaler_functionality(self):
        """Test MinMaxScaler scales to [0, 1] by default."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = ScalerSelector.get_scaler("MinMaxScaler")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape
        assert X_transformed.min() == 0.0
        assert X_transformed.max() == 1.0

    def test_maxabs_scaler_functionality(self):
        """Test MaxAbsScaler scales by maximum absolute value."""
        X = np.array([[-2, 3], [-1, 1], [0, 0], [1, -3]])
        scaler = ScalerSelector.get_scaler("MaxAbsScaler")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape
        assert np.max(np.abs(X_transformed)) == 1.0

    def test_yeojohnson_transform_with_positive_data(self):
        """Test YeoJohnson transform with positive data."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = ScalerSelector.get_scaler("YeoJohnsonTransform")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape
        # After power transform and standardization, mean should be ~0
        assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-7)

    def test_yeojohnson_transform_with_negative_data(self):
        """Test YeoJohnson transform handles negative values."""
        X = np.array([[-2, -1], [0, 1], [2, 3]])
        scaler = ScalerSelector.get_scaler("YeoJohnsonTransform")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape
        assert not np.any(np.isnan(X_transformed))

    def test_scaler_inverse_transform(self):
        """Test that inverse_transform restores original data."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = ScalerSelector.get_scaler("StandardScaler")
        scaler.fit(X)
        X_transformed = scaler.transform(X)
        X_restored = scaler.inverse_transform(X_transformed)

        np.testing.assert_array_almost_equal(X, X_restored, decimal=7)

    def test_partial_application_creates_callable(self):
        """Test that partial application for YeoJohnson works correctly."""
        from functools import partial
        from sklearn.preprocessing import PowerTransformer

        scaler_func = partial(PowerTransformer, method="yeo-johnson")
        scaler = scaler_func()

        assert isinstance(scaler, PowerTransformer)
        assert scaler.method == "yeo-johnson"

    def test_scaler_with_empty_kwargs(self):
        """Test that scalers work with no additional kwargs."""
        scaler = ScalerSelector.get_scaler("StandardScaler")
        assert scaler.with_mean
        assert scaler.with_std

    def test_invalid_kwargs_raises_error(self):
        """Test that invalid kwargs raise TypeError."""
        with pytest.raises(TypeError):
            ScalerSelector.get_scaler("StandardScaler", invalid_param=True)

    def test_case_sensitive_scaler_name(self):
        """Test that scaler name is case-sensitive."""
        with pytest.raises(ValueError):
            ScalerSelector.get_scaler("standardscaler")

    def test_all_scalers_can_be_instantiated(self):
        """Test that all available scalers can be instantiated."""
        scaler_names = [
            "StandardScaler",
            "RobustScaler",
            "MinMaxScaler",
            "MaxAbsScaler",
            "YeoJohnsonTransform",
        ]

        for name in scaler_names:
            scaler = ScalerSelector.get_scaler(name)
            assert scaler is not None
            assert hasattr(scaler, "fit")
            assert hasattr(scaler, "transform")

    def test_scaler_with_1d_array(self):
        """Test scaler handling of 1D arrays."""
        X = np.array([1, 2, 3, 4, 5])
        scaler = ScalerSelector.get_scaler("StandardScaler")
        scaler.fit(X.reshape(-1, 1))
        X_transformed = scaler.transform(X.reshape(-1, 1))

        assert X_transformed.shape == (5, 1)

    def test_scaler_with_single_feature(self):
        """Test scaler with single feature."""
        X = np.array([[1], [2], [3], [4], [5]])
        scaler = ScalerSelector.get_scaler("RobustScaler")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape

    def test_scaler_with_many_features(self):
        """Test scaler with many features."""
        X = np.random.randn(100, 50)
        scaler = ScalerSelector.get_scaler("StandardScaler")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape

    def test_robust_scaler_custom_quantile_range(self):
        """Test RobustScaler with custom quantile range."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        scaler = ScalerSelector.get_scaler("RobustScaler", quantile_range=(10, 90))
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert scaler.quantile_range == (10, 90)
        assert X_transformed.shape == X.shape

    def test_standard_scaler_with_mean_false(self):
        """Test StandardScaler without centering."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = ScalerSelector.get_scaler("StandardScaler", with_mean=False)
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        # Mean should not be zero when with_mean=False
        assert not np.allclose(X_transformed.mean(axis=0), 0)

    def test_scaler_selector_returns_new_instances(self):
        """Test that each call returns a new instance."""
        scaler1 = ScalerSelector.get_scaler("StandardScaler")
        scaler2 = ScalerSelector.get_scaler("StandardScaler")

        assert scaler1 is not scaler2

    def test_error_message_contains_available_scalers(self):
        """Test that error message lists all available scalers."""
        try:
            ScalerSelector.get_scaler("NonExistent")
        except ValueError as e:
            error_msg = str(e)
            assert "Available scalers:" in error_msg
            assert "StandardScaler" in error_msg
            assert "RobustScaler" in error_msg
            assert "MinMaxScaler" in error_msg
            assert "MaxAbsScaler" in error_msg
            assert "YeoJohnsonTransform" in error_msg


class TestScalerSelectorIntegration:
    """Integration tests for ScalerSelector."""

    def test_scaler_in_pipeline(self):
        """Test using scaler from selector in sklearn pipeline."""
        from sklearn.pipeline import Pipeline

        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = ScalerSelector.get_scaler("StandardScaler")

        pipeline = Pipeline([("scaler", scaler)])
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)

        assert X_transformed.shape == X.shape

    def test_multiple_scalers_different_data(self):
        """Test different scalers on the same data."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        scalers = {
            "StandardScaler": ScalerSelector.get_scaler("StandardScaler"),
            "RobustScaler": ScalerSelector.get_scaler("RobustScaler"),
            "MinMaxScaler": ScalerSelector.get_scaler("MinMaxScaler"),
            "MaxAbsScaler": ScalerSelector.get_scaler("MaxAbsScaler"),
        }

        results = {}
        for name, scaler in scalers.items():
            scaler.fit(X)
            results[name] = scaler.transform(X)

        # All results should have the same shape
        for result in results.values():
            assert result.shape == X.shape

        # Results should be different for different scalers
        assert not np.allclose(results["StandardScaler"], results["MinMaxScaler"])

    def test_scaler_with_train_test_split(self):
        """Test scaler fitted on train and applied to test."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[7, 8], [9, 10]])

        scaler = ScalerSelector.get_scaler("StandardScaler")
        scaler.fit(X_train)

        X_train_transformed = scaler.transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        assert X_train_transformed.shape == X_train.shape
        assert X_test_transformed.shape == X_test.shape

    def test_scaler_persistence(self):
        """Test that scaler state can be saved and loaded."""
        import pickle

        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = ScalerSelector.get_scaler("StandardScaler")
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        # Pickle and unpickle
        pickled = pickle.dumps(scaler)
        scaler_loaded = pickle.loads(pickled)

        X_transformed_loaded = scaler_loaded.transform(X)

        np.testing.assert_array_almost_equal(X_transformed, X_transformed_loaded)


class TestFeatureScalerManager:
    """Tests for FeatureScalerManager - multi-scaler feature handling."""

    @pytest.fixture
    def sample_timeseries_list(self):
        """Create sample TimeSeries for testing."""
        # Create a sample DataFrame with different feature types
        np.random.seed(42)
        n_time = 50

        # Simulate different data types
        data = {
            # Zero-inflated conflict data (lots of zeros, occasional spikes)
            "ged_sb": np.concatenate([np.zeros(30), np.random.exponential(10, 20)]),
            "ged_ns": np.concatenate([np.zeros(35), np.random.exponential(5, 15)]),
            "acled_sb": np.concatenate([np.zeros(25), np.random.exponential(15, 25)]),
            # WDI-style continuous data
            "wdi_gdp": np.random.normal(1000, 200, n_time),
            "wdi_pop": np.random.normal(1e6, 1e5, n_time),
            # V-Dem bounded indices (0-1)
            "vdem_polyarchy": np.random.beta(2, 5, n_time),
            "vdem_liberal": np.random.beta(2, 5, n_time),
        }

        times = pd.date_range("2000-01", periods=n_time, freq="MS")

        ts1 = TimeSeries.from_times_and_values(
            times=times,
            values=np.column_stack([data[k] for k in data]),
            columns=list(data.keys()),
        )

        # Create a second series with different values
        data2 = {k: v * 1.1 + np.random.normal(0, 0.1, n_time) for k, v in data.items()}
        ts2 = TimeSeries.from_times_and_values(
            times=times,
            values=np.column_stack([data2[k] for k in data2]),
            columns=list(data2.keys()),
        )

        return [ts1, ts2]

    def test_simple_format_parsing(self):
        """Test parsing of simple format: {'ScalerName': ['feat1', 'feat2']}"""
        config = {
            "RobustScaler": ["ged_sb", "ged_ns"],
            "MinMaxScaler": ["vdem_polyarchy"],
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            default_scaler="StandardScaler",
            all_features=["ged_sb", "ged_ns", "vdem_polyarchy", "wdi_gdp"],
        )

        mapping = manager.get_feature_scaler_mapping()

        assert mapping["ged_sb"] == "scaler_RobustScaler"
        assert mapping["ged_ns"] == "scaler_RobustScaler"
        assert mapping["vdem_polyarchy"] == "scaler_MinMaxScaler"
        assert mapping["wdi_gdp"] == "default"  # Not in map, uses default

    def test_named_group_format_parsing(self):
        """Test parsing of named group format."""
        config = {
            "conflict": {
                "scaler": "RobustScaler",
                "features": ["ged_sb", "ged_ns", "acled_sb"],
            },
            "democracy": {
                "scaler": "MinMaxScaler",
                "features": ["vdem_polyarchy", "vdem_liberal"],
            },
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            default_scaler="StandardScaler",
            all_features=[
                "ged_sb",
                "ged_ns",
                "acled_sb",
                "vdem_polyarchy",
                "vdem_liberal",
                "wdi_gdp",
            ],
        )

        mapping = manager.get_feature_scaler_mapping()

        assert mapping["ged_sb"] == "group_conflict"
        assert mapping["acled_sb"] == "group_conflict"
        assert mapping["vdem_polyarchy"] == "group_democracy"
        assert mapping["wdi_gdp"] == "default"

    def test_fit_transform(self, sample_timeseries_list):
        """Test fit_transform applies different scalers to different features."""
        config = {
            "RobustScaler": ["ged_sb", "ged_ns", "acled_sb"],
            "StandardScaler": ["wdi_gdp", "wdi_pop"],
            "MinMaxScaler": ["vdem_polyarchy", "vdem_liberal"],
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            all_features=list(sample_timeseries_list[0].components),
        )

        assert not manager.is_fitted

        transformed = manager.fit_transform(sample_timeseries_list)

        assert manager.is_fitted
        assert len(transformed) == len(sample_timeseries_list)

        # Check that transformed series have same shape and components
        for orig, trans in zip(sample_timeseries_list, transformed):
            assert list(trans.components) == list(orig.components)
            assert len(trans) == len(orig)

    def test_transform_requires_fit(self, sample_timeseries_list):
        """Test that transform raises error if not fitted."""
        config = {"RobustScaler": ["ged_sb"]}
        manager = FeatureScalerManager(feature_scaler_map=config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            manager.transform(sample_timeseries_list)

    def test_inverse_transform(self, sample_timeseries_list):
        """Test inverse transform recovers original values."""
        config = {
            "StandardScaler": ["wdi_gdp", "wdi_pop"],
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            all_features=list(sample_timeseries_list[0].components),
        )

        # Transform and inverse transform
        transformed = manager.fit_transform(sample_timeseries_list)
        recovered = manager.inverse_transform(transformed)

        # Check recovery for features that were scaled
        wdi_gdp_idx = list(sample_timeseries_list[0].components).index("wdi_gdp")

        for orig_ts, recovered_ts in zip(sample_timeseries_list, recovered):
            orig_arr = orig_ts.all_values()
            rec_arr = recovered_ts.all_values()

            # WDI features should be recovered (with some floating point tolerance)
            # Use decimal=4 for float32 precision (allows ~1e-4 absolute error)
            np.testing.assert_array_almost_equal(
                orig_arr[:, wdi_gdp_idx], rec_arr[:, wdi_gdp_idx], decimal=4
            )

    def test_duplicate_feature_raises_error(self):
        """Test that assigning a feature to multiple scalers raises an error."""
        config = {
            "RobustScaler": ["ged_sb", "ged_ns"],
            "StandardScaler": ["ged_sb", "wdi_gdp"],  # ged_sb is duplicate!
        }

        with pytest.raises(ValueError, match="assigned to multiple"):
            FeatureScalerManager(feature_scaler_map=config)

    def test_empty_config(self, sample_timeseries_list):
        """Test that empty config returns data unchanged."""
        manager = FeatureScalerManager(feature_scaler_map={})

        result = manager.fit_transform(sample_timeseries_list)

        for orig, res in zip(sample_timeseries_list, result):
            np.testing.assert_array_equal(orig.all_values(), res.all_values())

    def test_scaler_with_kwargs(self):
        """Test using scaler with custom kwargs via dict format."""
        config = {
            "robust_custom": {
                "scaler": {
                    "name": "RobustScaler",
                    "kwargs": {"quantile_range": (10, 90)},
                },
                "features": ["ged_sb"],
            }
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            all_features=["ged_sb", "other"],
        )

        # Should not raise
        assert "group_robust_custom" in manager._scalers

    def test_repr(self):
        """Test string representation."""
        config = {
            "RobustScaler": ["feat1", "feat2"],
            "MinMaxScaler": ["feat3"],
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            all_features=["feat1", "feat2", "feat3"],
        )

        repr_str = repr(manager)
        assert "FeatureScalerManager" in repr_str
        assert "fitted=False" in repr_str

    def test_no_default_scaler_when_all_mapped(self, sample_timeseries_list):
        """Test that no default scaler is created when all features are mapped."""
        all_features = list(sample_timeseries_list[0].components)

        config = {
            "RobustScaler": ["ged_sb", "ged_ns", "acled_sb"],
            "StandardScaler": ["wdi_gdp", "wdi_pop"],
            "MinMaxScaler": ["vdem_polyarchy", "vdem_liberal"],
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            default_scaler="StandardScaler",
            all_features=all_features,
        )

        # Should not have a "default" key since all features are mapped
        assert "default" not in manager._scalers

    def test_different_scalers_produce_different_results(self, sample_timeseries_list):
        """Test that different scalers actually produce different transformations."""
        # Create managers with different scalers for the same features
        config1 = {"StandardScaler": ["ged_sb"]}
        config2 = {"MinMaxScaler": ["ged_sb"]}

        manager1 = FeatureScalerManager(feature_scaler_map=config1)
        manager2 = FeatureScalerManager(feature_scaler_map=config2)

        transformed1 = manager1.fit_transform(sample_timeseries_list)
        manager2.fit_transform(sample_timeseries_list)
        transformed2 = manager2.transform(sample_timeseries_list)

        ged_sb_idx = list(sample_timeseries_list[0].components).index("ged_sb")

        # The transformations should be different
        vals1 = transformed1[0].all_values()[:, ged_sb_idx]
        vals2 = transformed2[0].all_values()[:, ged_sb_idx]

        # They shouldn't be exactly equal (different scalers)
        assert not np.allclose(vals1, vals2)


class TestScalerSelectorChaining:
    """Tests for ScalerSelector chain-related methods."""

    def test_is_chain_spec(self):
        """Test detection of chain specifications."""
        assert ScalerSelector.is_chain_spec("AsinhTransform->StandardScaler")
        assert ScalerSelector.is_chain_spec("A->B->C")
        assert not ScalerSelector.is_chain_spec("StandardScaler")
        assert not ScalerSelector.is_chain_spec("RobustScaler")

    def test_get_chained_scaler(self):
        """Test creating chained scaler from string."""
        from darts.dataprocessing import Pipeline

        chained = ScalerSelector.get_chained_scaler("AsinhTransform->StandardScaler")
        assert isinstance(chained, Pipeline)
        assert len(chained) == 2

    def test_get_chained_scaler_with_spaces(self):
        """Test that spaces around -> are handled."""
        from darts.dataprocessing import Pipeline

        chained = ScalerSelector.get_chained_scaler("AsinhTransform -> StandardScaler")
        assert isinstance(chained, Pipeline)
        assert len(chained) == 2

    def test_get_chained_scaler_single_raises_error(self):
        """Test that single scaler in chain format raises error."""
        with pytest.raises(ValueError, match="at least 2 scalers"):
            ScalerSelector.get_chained_scaler("StandardScaler")

    def test_get_scaler_or_chain_single(self):
        """Test get_scaler_or_chain returns single scaler."""
        scaler = ScalerSelector.get_scaler_or_chain("StandardScaler")
        assert scaler.__class__.__name__ == "StandardScaler"

    def test_get_scaler_or_chain_chained(self):
        """Test get_scaler_or_chain returns chained scaler."""
        from darts.dataprocessing import Pipeline

        scaler = ScalerSelector.get_scaler_or_chain("AsinhTransform->StandardScaler")
        assert isinstance(scaler, Pipeline)

    def test_three_scaler_chain(self):
        """Test chaining three scalers."""
        from darts.dataprocessing import Pipeline

        chained = ScalerSelector.get_chained_scaler(
            "AsinhTransform->StandardScaler->MinMaxScaler"
        )
        assert isinstance(chained, Pipeline)
        assert len(chained) == 3

        # Test functionality
        X = np.array([[0, 1], [10, 100], [1000, 10000]]).astype(np.float64)
        ts = TimeSeries.from_values(X.astype(np.float32))

        transformed = chained.fit_transform([ts])[0]
        recovered = chained.inverse_transform([transformed])[0]

        np.testing.assert_array_almost_equal(
            X, recovered.all_values().squeeze(), decimal=2
        )


class TestFeatureScalerManagerChaining:
    """Tests for FeatureScalerManager with chained scalers."""

    @pytest.fixture
    def sample_timeseries_list(self):
        """Create sample TimeSeries for testing."""
        np.random.seed(42)
        n_time = 50

        data = {
            "ged_sb": np.concatenate([np.zeros(30), np.random.exponential(10, 20)]),
            "ged_ns": np.concatenate([np.zeros(35), np.random.exponential(5, 15)]),
            "wdi_gdp": np.random.normal(1000, 200, n_time),
            "vdem_polyarchy": np.random.beta(2, 5, n_time),
        }

        times = pd.date_range("2000-01", periods=n_time, freq="MS")

        ts1 = TimeSeries.from_times_and_values(
            times=times,
            values=np.column_stack([data[k] for k in data]),
            columns=list(data.keys()),
        )

        return [ts1]

    def test_simple_format_chained_scaler(self, sample_timeseries_list):
        """Test simple format with chained scaler using -> syntax."""
        config = {
            "AsinhTransform->StandardScaler": ["ged_sb", "ged_ns"],
            "MinMaxScaler": ["vdem_polyarchy"],
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            default_scaler="StandardScaler",
            all_features=list(sample_timeseries_list[0].components),
        )

        # Should have created the chained scaler
        assert "scaler_AsinhTransform->StandardScaler" in manager._scalers

        # Test that fit_transform works
        transformed = manager.fit_transform(sample_timeseries_list)
        assert len(transformed) == 1
        assert transformed[0].n_components == 4

    def test_named_group_format_chained_list(self, sample_timeseries_list):
        """Test named group format with list-based chain."""
        config = {
            "conflict": {
                "scaler": ["AsinhTransform", "StandardScaler"],
                "features": ["ged_sb", "ged_ns"],
            },
            "democracy": {"scaler": "MinMaxScaler", "features": ["vdem_polyarchy"]},
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            default_scaler="RobustScaler",
            all_features=list(sample_timeseries_list[0].components),
        )

        transformed = manager.fit_transform(sample_timeseries_list)
        assert manager.is_fitted
        assert len(transformed) == 1

    def test_named_group_format_chain_dict(self, sample_timeseries_list):
        """Test named group format with chain dict config."""
        config = {
            "conflict": {
                "scaler": {"chain": ["AsinhTransform", "RobustScaler"]},
                "features": ["ged_sb", "ged_ns"],
            },
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            default_scaler="StandardScaler",
            all_features=list(sample_timeseries_list[0].components),
        )

        manager.fit_transform(sample_timeseries_list)
        assert manager.is_fitted

    def test_chained_inverse_transform(self, sample_timeseries_list):
        """Test that inverse_transform works with chained scalers."""
        config = {
            "AsinhTransform->StandardScaler": ["ged_sb", "ged_ns"],
        }

        manager = FeatureScalerManager(
            feature_scaler_map=config,
            all_features=["ged_sb", "ged_ns"],
        )

        # Get original values
        orig_values = sample_timeseries_list[0].all_values(copy=True)

        # Transform
        transformed = manager.fit_transform(sample_timeseries_list)

        # Inverse transform
        recovered = manager.inverse_transform(transformed)

        # Check recovery (with tolerance for floating point)
        ged_sb_idx = list(sample_timeseries_list[0].components).index("ged_sb")
        np.testing.assert_array_almost_equal(
            orig_values[:, ged_sb_idx],
            recovered[0].all_values()[:, ged_sb_idx],
            decimal=4,
        )
