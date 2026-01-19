import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted
from typing import Dict, List, Optional, Union, Any
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

# from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    PowerTransformer,
    FunctionTransformer,
    QuantileTransformer,
)
from functools import partial


def _log_transform(x):
    """Log transform function: log(1 + x)"""
    return np.log1p(x)


def _inverse_log_transform(x):
    """Inverse log transform function: exp(x) - 1"""
    return np.expm1(x)


def _sqrt_transform(x):
    """
    Square root transform - gentler than log for zero-inflated data.
    
    Benefits for conflict data:
    - Defined at zero (unlike log)
    - Less aggressive compression than log
    - Variance stabilization for count-like data
    """
    return np.sqrt(np.maximum(x, 0))


def _inverse_sqrt_transform(x):
    """Inverse square root transform: x^2"""
    return np.square(x)


def _asinh_transform(x):
    """
    Inverse hyperbolic sine transform - best for zero-inflated continuous data.
    
    Benefits for conflict data:
    - Handles zeros gracefully (asinh(0) = 0)
    - Handles negative values if any exist
    - Similar to log for large values: asinh(x) ≈ log(2x) for large x
    - Smooth and differentiable everywhere
    - Less aggressive than log near zero
    """
    return np.arcsinh(x)


def _inverse_asinh_transform(x):
    """Inverse of asinh transform: sinh(x)"""
    return np.sinh(x)


class ChainedScaler(BaseEstimator, TransformerMixin):
    """
    A scaler that chains multiple sklearn scalers/transformers in sequence.
    
    Applies transforms in order during fit_transform/transform, and applies
    inverse transforms in reverse order during inverse_transform.
    
    Example:
        # First apply asinh, then standardize
        chained = ChainedScaler([
            ScalerSelector.get_scaler("AsinhTransform"),
            ScalerSelector.get_scaler("StandardScaler"),
        ])
        X_transformed = chained.fit_transform(X)
        X_original = chained.inverse_transform(X_transformed)
    
    Config usage:
        "feature_scaler_map": {
            "AsinhTransform->StandardScaler": ["ged_sb", "ged_ns"],
            "LogTransform->RobustScaler": ["wdi_ny_gdp_mktp_kd"],
        }
    """
    
    def __init__(self, scalers: List[BaseEstimator]):
        """
        Initialize the chained scaler.
        
        Args:
            scalers: List of sklearn-compatible scalers/transformers to chain.
                     Applied in order during transform, reversed during inverse.
        """
        if not scalers:
            raise ValueError("ChainedScaler requires at least one scaler.")
        self.scalers = scalers
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """Fit all scalers in sequence."""
        X_current = X
        for scaler in self.scalers:
            X_current = scaler.fit_transform(X_current)
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Apply all transforms in order."""
        check_is_fitted(self, '_is_fitted')
        X_current = X
        for scaler in self.scalers:
            X_current = scaler.transform(X_current)
        return X_current
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        X_current = X
        for scaler in self.scalers:
            X_current = scaler.fit_transform(X_current)
        self._is_fitted = True
        return X_current
    
    def inverse_transform(self, X):
        """Apply inverse transforms in reverse order."""
        check_is_fitted(self, '_is_fitted')
        X_current = X
        # Apply inverse transforms in reverse order
        for scaler in reversed(self.scalers):
            if hasattr(scaler, 'inverse_transform'):
                X_current = scaler.inverse_transform(X_current)
            else:
                raise ValueError(
                    f"Scaler {scaler.__class__.__name__} does not support inverse_transform. "
                    f"Cannot use it in a chain that requires inversion."
                )
        return X_current
    
    def __repr__(self) -> str:
        scaler_names = [s.__class__.__name__ for s in self.scalers]
        return f"ChainedScaler({' -> '.join(scaler_names)})"


class ScalerSelector:
    @staticmethod
    def get_scaler(scaler_name: str, **kwargs) -> BaseEstimator:
        """
        Returns a scaler instance based on the provided scaler name.

        Parameters
        ----------
        scaler_name : str
            Name of the scaler to instantiate.
            
            Available scalers:
            - StandardScaler: Zero mean, unit variance. Best for normally distributed data.
            - RobustScaler: Uses median/IQR, resistant to outliers. Good for conflict data.
            - MinMaxScaler: Scales to [0,1]. Sensitive to outliers.
            - MaxAbsScaler: Scales by max absolute value. Preserves sparsity.
            - YeoJohnsonTransform: Power transform to make data more Gaussian-like.
            - LogTransform: log(1+x). Good for positive skewed data, but undefined at 0.
            - SqrtTransform: sqrt(x). Gentler than log, defined at 0.
            - AsinhTransform: arcsinh(x). Best for zero-inflated data, handles zeros/negatives.
            - QuantileUniform: Maps to uniform distribution. Forces uniform output.
            - QuantileNormal: Maps to normal distribution. Forces Gaussian output.

        Returns
        -------
        BaseEstimator
            An instance of the specified scaler.

        Raises
        ------
        ValueError
            If the scaler name is not recognized.
        """
        scalers = {
            # Standard statistical scalers
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "MinMaxScaler": MinMaxScaler,
            "MaxAbsScaler": MaxAbsScaler,
            
            # Power transforms
            "YeoJohnsonTransform": partial(PowerTransformer, method="yeo-johnson"),
            
            # Logarithmic transform (for strictly positive data)
            "LogTransform": partial(
                FunctionTransformer,
                func=_log_transform,
                inverse_func=_inverse_log_transform,
                validate=True,
            ),
            
            # Square root transform (gentler than log, handles zeros)
            "SqrtTransform": partial(
                FunctionTransformer,
                func=_sqrt_transform,
                inverse_func=_inverse_sqrt_transform,
                validate=True,
            ),
            
            # Inverse hyperbolic sine (best for zero-inflated data)
            "AsinhTransform": partial(
                FunctionTransformer,
                func=_asinh_transform,
                inverse_func=_inverse_asinh_transform,
                validate=True,
            ),
            
            # Quantile transforms (force specific output distributions)
            "QuantileUniform": partial(
                QuantileTransformer,
                output_distribution="uniform",
                n_quantiles=1000,
                random_state=42,
            ),
            "QuantileNormal": partial(
                QuantileTransformer,
                output_distribution="normal",
                n_quantiles=1000,
                random_state=42,
            ),
        }

        if scaler_name not in scalers:
            raise ValueError(
                f"Scaler '{scaler_name}' is not recognized. Available scalers: {list(scalers.keys())}"
            )

        return scalers[scaler_name](**kwargs)

    @staticmethod
    def get_chained_scaler(scaler_chain: str) -> ChainedScaler:
        """
        Create a ChainedScaler from a chain specification string.
        
        Parameters
        ----------
        scaler_chain : str
            Chain specification using '->' separator.
            Example: "AsinhTransform->StandardScaler"
            
        Returns
        -------
        ChainedScaler
            A ChainedScaler instance with the specified scalers.
            
        Example
        -------
        >>> chained = ScalerSelector.get_chained_scaler("AsinhTransform->StandardScaler")
        >>> # Equivalent to: ChainedScaler([AsinhTransform(), StandardScaler()])
        """
        scaler_names = [s.strip() for s in scaler_chain.split("->")]
        if len(scaler_names) < 2:
            raise ValueError(
                f"Chain specification '{scaler_chain}' must contain at least 2 scalers "
                f"separated by '->'. Got {len(scaler_names)} scaler(s)."
            )
        
        scalers = [ScalerSelector.get_scaler(name) for name in scaler_names]
        return ChainedScaler(scalers)
    
    @staticmethod
    def is_chain_spec(scaler_name: str) -> bool:
        """Check if a scaler name is a chain specification."""
        return "->" in scaler_name
    
    @staticmethod
    def get_scaler_or_chain(scaler_spec: str, **kwargs) -> BaseEstimator:
        """
        Get either a single scaler or a chained scaler based on the specification.
        
        Parameters
        ----------
        scaler_spec : str
            Either a single scaler name or a chain specification.
            Examples: "StandardScaler", "AsinhTransform->StandardScaler"
            
        Returns
        -------
        BaseEstimator
            Either a single scaler or a ChainedScaler instance.
        """
        if ScalerSelector.is_chain_spec(scaler_spec):
            if kwargs:
                raise ValueError(
                    "kwargs are not supported for chained scalers. "
                    "Use the named group format with individual scaler configs instead."
                )
            return ScalerSelector.get_chained_scaler(scaler_spec)
        return ScalerSelector.get_scaler(scaler_spec, **kwargs)


class FeatureScalerManager:
    """
    Manages multiple scalers for different feature groups.
    
    This allows applying different scalers to different features based on their
    data characteristics (e.g., RobustScaler for zero-inflated conflict data,
    MinMaxScaler for bounded V-Dem indices, StandardScaler for WDI indicators).
    
    Supports chained scalers using '->' syntax or list format. Chained scalers
    apply transforms in order and inverse transforms in reverse order.
    
    Configuration format in config_hyperparameters.py:
    
    Named group format:
    ```python
    "feature_scaler_map": {
        "conflict": {
            "scaler": "RobustScaler",  # or {"name": "RobustScaler", "kwargs": {...}}
            "features": ["ged_sb", "ged_ns", "ged_os", "acled_sb", "acled_os"]
        },
        "wdi": {
            "scaler": "StandardScaler",
            "features": ["wdi_sm_pop_netm", "wdi_ny_gdp_mktp_kd"]
        },
        "vdem": {
            "scaler": "MinMaxScaler",
            "features": ["vdem_v2x_polyarchy", "vdem_v2x_libdem"]
        },
        # Chained scalers using list format
        "conflict_chained": {
            "scaler": ["AsinhTransform", "StandardScaler"],  # Applied in order
            "features": ["ged_sb", "ged_ns"]
        },
        # Or using chain key
        "wdi_chained": {
            "scaler": {"chain": ["LogTransform", "RobustScaler"]},
            "features": ["wdi_ny_gdp_mktp_kd"]
        },
    }
    ```
    
    Simple format (recommended):
    ```python
    "feature_scaler_map": {
        # Single scalers
        "RobustScaler": ["topic_tokens_t1", "topic_tokens_t1_splag"],
        "MinMaxScaler": ["vdem_v2x_polyarchy", "vdem_v2x_libdem"],
        
        # Chained scalers using '->' syntax
        "AsinhTransform->StandardScaler": ["ged_sb", "ged_ns", "acled_sb"],
        "LogTransform->RobustScaler": ["wdi_ny_gdp_mktp_kd", "wdi_sm_pop_netm"],
    }
    ```
    
    Chained scalers apply transforms in sequence:
    - Forward: AsinhTransform -> StandardScaler (first asinh, then standardize)
    - Inverse: StandardScaler.inverse -> AsinhTransform.inverse (reverse order)
    """
    
    def __init__(
        self,
        feature_scaler_map: Dict[str, Any],
        default_scaler: Optional[str] = "RobustScaler",
        all_features: Optional[List[str]] = None,
    ):
        """
        Initialize the feature scaler manager.
        
        Args:
            feature_scaler_map: Mapping of scaler names/groups to feature lists.
            default_scaler: Default scaler for features not in the map.
            all_features: Complete list of features (used to identify unmapped features).
        """
        self.feature_scaler_map = feature_scaler_map
        self.default_scaler_name = default_scaler
        self.all_features = set(all_features or [])
        
        # Parse the config and build internal structures
        self._scalers: Dict[str, Scaler] = {}  # scaler_key -> Darts Scaler
        self._feature_to_scaler: Dict[str, str] = {}  # feature_name -> scaler_key
        self._scaler_to_features: Dict[str, List[str]] = {}  # scaler_key -> [features]
        self._fitted = False
        
        self._parse_config()
    
    def _parse_config(self):
        """Parse the feature_scaler_map configuration into internal structures."""
        if not self.feature_scaler_map:
            return
        
        # Check which format is being used
        first_value = next(iter(self.feature_scaler_map.values()), None)
        
        if isinstance(first_value, dict) and "features" in first_value:
            # Named group format: {"conflict": {"scaler": "RobustScaler", "features": [...]}}
            self._parse_named_group_format()
        elif isinstance(first_value, list):
            # Simple format: {"RobustScaler": ["feat1", "feat2"], ...}
            self._parse_simple_format()
        else:
            raise ValueError(
                f"Unrecognized feature_scaler_map format. Expected either:\n"
                f"  1. Named groups: {{'group_name': {{'scaler': 'ScalerName', 'features': [...]}}}}\n"
                f"  2. Simple mapping: {{'ScalerName': ['feat1', 'feat2']}}"
            )
        
        # Assign default scaler to unmapped features
        self._assign_default_scaler()
    
    def _parse_named_group_format(self):
        """Parse format: {"conflict": {"scaler": "RobustScaler", "features": [...]}}"""
        for group_name, group_config in self.feature_scaler_map.items():
            scaler_cfg = group_config.get("scaler", self.default_scaler_name)
            features = group_config.get("features", [])
            
            if not features:
                continue
            
            # Create scaler key based on group name
            scaler_key = f"group_{group_name}"
            
            # Instantiate the scaler
            self._scalers[scaler_key] = self._instantiate_scaler(scaler_cfg)
            self._scaler_to_features[scaler_key] = list(features)
            
            for feat in features:
                if feat in self._feature_to_scaler:
                    raise ValueError(
                        f"Feature '{feat}' is assigned to multiple scaler groups!"
                    )
                self._feature_to_scaler[feat] = scaler_key
    
    def _parse_simple_format(self):
        """Parse format: {"RobustScaler": ["feat1", "feat2"], ...}"""
        for scaler_name, features in self.feature_scaler_map.items():
            if not features:
                continue
            
            scaler_key = f"scaler_{scaler_name}"
            
            # Instantiate the scaler
            self._scalers[scaler_key] = self._instantiate_scaler(scaler_name)
            self._scaler_to_features[scaler_key] = list(features)
            
            for feat in features:
                if feat in self._feature_to_scaler:
                    raise ValueError(
                        f"Feature '{feat}' is assigned to multiple scalers!"
                    )
                self._feature_to_scaler[feat] = scaler_key
    
    def _assign_default_scaler(self):
        """Assign the default scaler to any features not explicitly mapped."""
        if not self.all_features or not self.default_scaler_name:
            return
        
        unmapped_features = [
            f for f in self.all_features 
            if f not in self._feature_to_scaler
        ]
        
        if unmapped_features:
            scaler_key = "default"
            self._scalers[scaler_key] = self._instantiate_scaler(self.default_scaler_name)
            self._scaler_to_features[scaler_key] = unmapped_features
            
            for feat in unmapped_features:
                self._feature_to_scaler[feat] = scaler_key
    
    def _instantiate_scaler(self, scaler_cfg) -> Scaler:
        """
        Create a Darts Scaler wrapper from config.
        
        Supports:
        - String: "StandardScaler" or "AsinhTransform->StandardScaler" (chained)
        - List: ["AsinhTransform", "StandardScaler"] (chained)
        - Dict: {"name": "RobustScaler", "kwargs": {...}}
        - Dict with chain: {"chain": ["AsinhTransform", "StandardScaler"]}
        """
        if isinstance(scaler_cfg, str):
            # Check if it's a chain specification
            estimator = ScalerSelector.get_scaler_or_chain(scaler_cfg)
            return Scaler(estimator)
        
        if isinstance(scaler_cfg, list):
            # List of scalers to chain: ["AsinhTransform", "StandardScaler"]
            if len(scaler_cfg) == 1:
                estimator = ScalerSelector.get_scaler(scaler_cfg[0])
            else:
                scalers = [ScalerSelector.get_scaler(name) for name in scaler_cfg]
                estimator = ChainedScaler(scalers)
            return Scaler(estimator)
        
        if isinstance(scaler_cfg, dict):
            # Check if it's a chain config
            if "chain" in scaler_cfg:
                chain_list = scaler_cfg["chain"]
                if isinstance(chain_list, str):
                    # "chain": "AsinhTransform->StandardScaler"
                    estimator = ScalerSelector.get_chained_scaler(chain_list)
                elif isinstance(chain_list, list):
                    # "chain": ["AsinhTransform", "StandardScaler"]
                    scalers = [ScalerSelector.get_scaler(name) for name in chain_list]
                    estimator = ChainedScaler(scalers)
                else:
                    raise TypeError(
                        f"'chain' must be a string or list, got {type(chain_list).__name__}"
                    )
                return Scaler(estimator)
            
            # Standard dict format with name and optional kwargs
            name = scaler_cfg.get("name")
            kwargs = scaler_cfg.get("kwargs", {})
            if name is None:
                raise ValueError(
                    "Scaler config dict must have a 'name' key or a 'chain' key."
                )
            estimator = ScalerSelector.get_scaler_or_chain(name, **kwargs)
            return Scaler(estimator)
        
        raise TypeError(
            f"Scaler config must be str, list, or dict. Got {type(scaler_cfg).__name__}."
        )
    
    def fit_transform(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        """
        Fit scalers on the data and transform.
        
        Each scaler is fit on ALL time series combined (not per-series) to ensure
        consistent statistics across the entire dataset.
        
        Args:
            series_list: List of TimeSeries with features as components.
            
        Returns:
            List of transformed TimeSeries.
        """
        if not self._scalers:
            return series_list
        
        # First, fit all scalers on the combined data from all series
        self._fit_scalers_on_all_series(series_list)
        
        # Then transform each series using the fitted scalers
        result = []
        for ts in series_list:
            transformed_ts = self._transform_single_series(ts, fit=False)
            result.append(transformed_ts)
        
        self._fitted = True
        return result
    
    def _fit_scalers_on_all_series(self, series_list: List[TimeSeries]):
        """
        Fit all scalers using combined data from all time series.
        
        This ensures that scalers like StandardScaler compute statistics
        (mean, std) over the entire dataset, not per-series.
        """
        if not series_list:
            return
        
        components = list(series_list[0].components)
        
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            
            # Find indices of features belonging to this scaler
            feature_indices = [
                i for i, comp in enumerate(components) 
                if comp in feature_names
            ]
            
            if not feature_indices:
                continue
            
            # Collect data from all series for these features
            all_subsets = []
            for ts in series_list:
                arr = ts.all_values(copy=False)
                if arr.ndim == 2:
                    subset = arr[:, feature_indices]
                else:  # ndim == 3 (probabilistic)
                    subset = arr[:, feature_indices, :]
                all_subsets.append(subset)
            
            # Stack all data vertically (along time dimension)
            combined_data = np.concatenate(all_subsets, axis=0)
            
            # Create a single TimeSeries from combined data for fitting
            subset_names = [components[i] for i in feature_indices]
            # Use a dummy time index for fitting
            dummy_times = pd.date_range('2000-01-01', periods=len(combined_data), freq='MS')
            combined_ts = TimeSeries.from_times_and_values(
                times=dummy_times,
                values=combined_data.astype(np.float32),
                columns=subset_names,
            )
            
            # Fit the scaler on combined data
            scaler.fit([combined_ts])
    
    def transform(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        """
        Transform data using fitted scalers.
        
        Args:
            series_list: List of TimeSeries to transform.
            
        Returns:
            List of transformed TimeSeries.
        """
        if not self._fitted:
            raise RuntimeError("Scalers have not been fitted. Call fit_transform first.")
        
        if not self._scalers:
            return series_list
        
        result = []
        for ts in series_list:
            transformed_ts = self._transform_single_series(ts, fit=False)
            result.append(transformed_ts)
        
        return result
    
    def _transform_single_series(self, ts: TimeSeries, fit: bool = False) -> TimeSeries:
        """
        Transform a single TimeSeries by applying appropriate scalers to each feature.
        
        Args:
            ts: Input TimeSeries.
            fit: Whether to fit the scalers (True for fit_transform, False for transform).
            
        Returns:
            Transformed TimeSeries.
        """
        components = list(ts.components)
        arr = ts.all_values(copy=True)  # Shape: (time, features) or (time, features, samples)
        
        # Track which features have been transformed
        transformed_indices = set()
        
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            
            # Find indices of features belonging to this scaler
            feature_indices = [
                i for i, comp in enumerate(components) 
                if comp in feature_names
            ]
            
            if not feature_indices:
                continue
            
            # Extract the subset of features for this scaler
            if arr.ndim == 2:
                subset = arr[:, feature_indices]
            else:  # ndim == 3 (probabilistic)
                subset = arr[:, feature_indices, :]
            
            # Create a temporary TimeSeries for this subset
            subset_names = [components[i] for i in feature_indices]
            temp_ts = TimeSeries.from_times_and_values(
                times=ts.time_index,
                values=subset.astype(np.float32),
                columns=subset_names,
                freq=ts.freq,
            )
            
            # Fit and/or transform
            if fit:
                transformed_subset = scaler.fit_transform([temp_ts])[0]
            else:
                transformed_subset = scaler.transform([temp_ts])[0]
            
            # Put the transformed values back
            transformed_values = transformed_subset.all_values(copy=False)
            if arr.ndim == 2:
                arr[:, feature_indices] = transformed_values
            else:
                arr[:, feature_indices, :] = transformed_values
            
            transformed_indices.update(feature_indices)
        
        # Rebuild the TimeSeries
        new_ts = TimeSeries.from_times_and_values(
            times=ts.time_index,
            values=arr.astype(np.float32),
            columns=components,
            freq=ts.freq,
            static_covariates=ts.static_covariates,
        )
        
        return new_ts
    
    def inverse_transform(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        """
        Inverse transform data using fitted scalers.
        
        Note: This is typically only needed for target scalers, not feature scalers.
        
        Args:
            series_list: List of TimeSeries to inverse transform.
            
        Returns:
            List of inverse transformed TimeSeries.
        """
        if not self._fitted:
            raise RuntimeError("Scalers have not been fitted.")
        
        if not self._scalers:
            return series_list
        
        result = []
        for ts in series_list:
            transformed_ts = self._inverse_transform_single_series(ts)
            result.append(transformed_ts)
        
        return result
    
    def _inverse_transform_single_series(self, ts: TimeSeries) -> TimeSeries:
        """Inverse transform a single TimeSeries."""
        components = list(ts.components)
        arr = ts.all_values(copy=True)
        
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            
            feature_indices = [
                i for i, comp in enumerate(components) 
                if comp in feature_names
            ]
            
            if not feature_indices:
                continue
            
            if arr.ndim == 2:
                subset = arr[:, feature_indices]
            else:
                subset = arr[:, feature_indices, :]
            
            subset_names = [components[i] for i in feature_indices]
            temp_ts = TimeSeries.from_times_and_values(
                times=ts.time_index,
                values=subset.astype(np.float32),
                columns=subset_names,
                freq=ts.freq,
            )
            
            inv_subset = scaler.inverse_transform([temp_ts])[0]
            inv_values = inv_subset.all_values(copy=False)
            
            if arr.ndim == 2:
                arr[:, feature_indices] = inv_values
            else:
                arr[:, feature_indices, :] = inv_values
        
        new_ts = TimeSeries.from_times_and_values(
            times=ts.time_index,
            values=arr.astype(np.float32),
            columns=components,
            freq=ts.freq,
            static_covariates=ts.static_covariates,
        )
        
        return new_ts
    
    @property
    def is_fitted(self) -> bool:
        """Return whether the scalers have been fitted."""
        return self._fitted
    
    def get_feature_scaler_mapping(self) -> Dict[str, str]:
        """Return the mapping of features to their scaler keys."""
        return self._feature_to_scaler.copy()
    
    def __repr__(self) -> str:
        mapping_str = ", ".join(
            f"{k}: {len(v)} features" 
            for k, v in self._scaler_to_features.items()
        )
        return f"FeatureScalerManager({mapping_str}, fitted={self._fitted})"
