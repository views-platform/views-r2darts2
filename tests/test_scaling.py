import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from views_r2darts2.utils.scaling import ScalerSelector


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
        with pytest.raises(ValueError, match="Scaler 'UnknownScaler' is not recognized"):
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
            "StandardScaler", 
            with_mean=False, 
            with_std=True
        )
        assert not scaler.with_mean
        assert scaler.with_std

    def test_minmax_scaler_with_feature_range(self):
        scaler = ScalerSelector.get_scaler("MinMaxScaler", feature_range=(-1, 1))
        assert scaler.feature_range == (-1, 1)

    def test_power_transformer_standardize_param(self):
        scaler = ScalerSelector.get_scaler(
            "YeoJohnsonTransform", 
            standardize=False
        )
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
            "YeoJohnsonTransform"
        ]
        
        for name in scaler_names:
            scaler = ScalerSelector.get_scaler(name)
            assert scaler is not None
            assert hasattr(scaler, 'fit')
            assert hasattr(scaler, 'transform')

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
        
        pipeline = Pipeline([('scaler', scaler)])
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
        assert not np.allclose(
            results["StandardScaler"], 
            results["MinMaxScaler"]
        )

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
        
        np.testing.assert_array_almost_equal(
            X_transformed, 
            X_transformed_loaded
        )