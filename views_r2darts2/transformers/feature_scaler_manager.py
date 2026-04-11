import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, List, Optional, Any
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from .scaler_selector import ScalerSelector

class FeatureScalerManager:
    """
    Manages multiple scalers for different feature groups.
    """

    def __init__(
        self,
        feature_scaler_map: Dict[str, Any],
        default_scaler: Optional[str] = "RobustScaler",
        all_features: Optional[List[str]] = None,
    ):
        self.feature_scaler_map = feature_scaler_map
        self.default_scaler_name = default_scaler
        self.all_features = set(all_features or [])

        self._scalers: Dict[str, Scaler] = {}
        self._feature_to_scaler: Dict[str, str] = {}
        self._scaler_to_features: Dict[str, List[str]] = {}
        self._fitted = False

        self._parse_config()

    def _parse_config(self):
        if not self.feature_scaler_map:
            return

        first_value = next(iter(self.feature_scaler_map.values()), None)

        if isinstance(first_value, dict) and "features" in first_value:
            self._parse_named_group_format()
        elif isinstance(first_value, list):
            self._parse_simple_format()
        else:
            raise ValueError("Unrecognized feature_scaler_map format.")

        self._assign_default_scaler()

    def _parse_named_group_format(self):
        for group_name, group_config in self.feature_scaler_map.items():
            scaler_cfg = group_config.get("scaler", self.default_scaler_name)
            features = group_config.get("features", [])
            if not features:
                continue
            scaler_key = f"group_{group_name}"
            self._scalers[scaler_key] = self._instantiate_scaler(scaler_cfg)
            self._scaler_to_features[scaler_key] = list(features)
            for feat in features:
                if feat in self._feature_to_scaler:
                    raise ValueError(f"Feature '{feat}' is assigned to multiple groups!")
                self._feature_to_scaler[feat] = scaler_key

    def _parse_simple_format(self):
        for scaler_name, features in self.feature_scaler_map.items():
            if not features:
                continue
            scaler_key = f"scaler_{scaler_name}"
            self._scalers[scaler_key] = self._instantiate_scaler(scaler_name)
            self._scaler_to_features[scaler_key] = list(features)
            for feat in features:
                if feat in self._feature_to_scaler:
                    raise ValueError(f"Feature '{feat}' is assigned to multiple scalers!")
                self._feature_to_scaler[feat] = scaler_key

    def _assign_default_scaler(self):
        if not self.all_features or not self.default_scaler_name:
            return
        unmapped_features = [f for f in self.all_features if f not in self._feature_to_scaler]
        if unmapped_features:
            scaler_key = "default"
            self._scalers[scaler_key] = self._instantiate_scaler(self.default_scaler_name)
            self._scaler_to_features[scaler_key] = unmapped_features
            for feat in unmapped_features:
                self._feature_to_scaler[feat] = scaler_key

    @staticmethod
    def _instantiate_scaler(scaler_cfg):
        """
        Delegate to ScalerSelector.instantiate_darts_scaler(), rejecting None.

        `ScalerSelector.instantiate_darts_scaler(None)` legitimately returns
        `None` for the forecaster-level target/feature scaler paths, where a
        missing scaler is a valid configuration. Inside `FeatureScalerManager`,
        however, a `None` entry gets stored in `self._scalers` and later
        propagated into `fit()` / `transform()` / `inverse_transform()` calls
        that assume every entry is a Darts Scaler or Pipeline, producing an
        `AttributeError` at fit time that is hard to trace back to the
        misconfigured group. Fail loudly at parse time instead.
        """
        if scaler_cfg is None:
            raise ValueError(
                "Scaler configuration cannot be None in FeatureScalerManager. "
                "Provide a valid scaler configuration (str, list, or dict) for "
                "each group, or set `default_scaler` on the manager so groups "
                "without an explicit `scaler` key have a fallback."
            )
        return ScalerSelector.instantiate_darts_scaler(scaler_cfg)

    def fit_transform(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        if not self._scalers:
            return series_list
        self._fit_scalers_on_all_series(series_list)
        result = [self._transform_single_series(ts, fit=False) for ts in series_list]
        self._fitted = True
        return result

    def _fit_scalers_on_all_series(self, series_list: List[TimeSeries]):
        from darts.dataprocessing import Pipeline
        if not series_list:
            return
        components = list(series_list[0].components)
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            feature_indices = [i for i, comp in enumerate(components) if comp in feature_names]
            if not feature_indices:
                continue

            if isinstance(scaler, Pipeline):
                all_subsets = []
                for ts in series_list:
                    arr = ts.all_values(copy=False)
                    subset = arr[:, feature_indices] if arr.ndim == 2 else arr[:, feature_indices, :]
                    all_subsets.append(subset)
                combined_data = np.concatenate(all_subsets, axis=0)
                subset_names = [components[i] for i in feature_indices]
                dummy_index = pd.RangeIndex(start=0, stop=combined_data.shape[0], step=1)
                combined_2d = combined_data if combined_data.ndim == 2 else combined_data[:, :, 0]
                dummy_ts = TimeSeries.from_times_and_values(
                    times=dummy_index, values=combined_2d.astype(np.float32), columns=subset_names
                )
                scaler.fit([dummy_ts])
                continue

            all_subsets = []
            for ts in series_list:
                arr = ts.all_values(copy=False)
                subset = arr[:, feature_indices] if arr.ndim == 2 else arr[:, feature_indices, :]
                all_subsets.append(subset)
            combined_data = np.concatenate(all_subsets, axis=0)
            if combined_data.ndim == 3:
                n_time, n_features, n_samples = combined_data.shape
                combined_data_2d = combined_data.transpose(0, 2, 1).reshape(-1, n_features)
            else:
                combined_data_2d = combined_data
            underlying_scaler = deepcopy(scaler.transformer)
            fitted_scaler = underlying_scaler.fit(combined_data_2d.astype(np.float64))
            scaler._fitted_params = (fitted_scaler,)
            scaler._fit_called = True

    def transform(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        if not self._fitted:
            raise RuntimeError("Scalers not fitted. Call fit_transform first.")
        if not self._scalers:
            return series_list
        return [self._transform_single_series(ts, fit=False) for ts in series_list]

    def _transform_single_series(self, ts: TimeSeries, fit: bool = False) -> TimeSeries:
        components = list(ts.components)
        arr = ts.all_values(copy=True)
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            feature_indices = [i for i, comp in enumerate(components) if comp in feature_names]
            if not feature_indices:
                continue
            subset = arr[:, feature_indices] if arr.ndim == 2 else arr[:, feature_indices, :]
            subset_names = [components[i] for i in feature_indices]
            temp_ts = TimeSeries.from_times_and_values(
                times=ts.time_index, values=subset.astype(np.float32), columns=subset_names, freq=ts.freq
            )
            transformed_subset = scaler.fit_transform([temp_ts])[0] if fit else scaler.transform([temp_ts])[0]
            transformed_values = transformed_subset.all_values(copy=False)
            if arr.ndim == 2:
                arr[:, feature_indices] = transformed_values
            else:
                arr[:, feature_indices, :] = transformed_values
        return TimeSeries.from_times_and_values(
            times=ts.time_index, values=arr.astype(np.float32), columns=components,
            freq=ts.freq, static_covariates=ts.static_covariates
        )

    def inverse_transform(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        if not self._fitted:
            raise RuntimeError("Scalers not fitted.")
        if not self._scalers:
            return series_list
        return [self._inverse_transform_single_series(ts) for ts in series_list]

    def _inverse_transform_single_series(self, ts: TimeSeries) -> TimeSeries:
        from darts.dataprocessing import Pipeline
        components = list(ts.components)
        arr = ts.all_values(copy=True)
        is_probabilistic = arr.ndim == 3
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            feature_indices = [i for i, comp in enumerate(components) if comp in feature_names]
            if not feature_indices:
                continue
            subset_names = [components[i] for i in feature_indices]
            subset = arr[:, feature_indices, :] if is_probabilistic else arr[:, feature_indices]
            temp_ts = TimeSeries.from_times_and_values(
                times=ts.time_index, values=subset.astype(np.float32), columns=subset_names, freq=ts.freq
            )
            if isinstance(scaler, Pipeline):
                inv_subset = scaler.inverse_transform([temp_ts])[0]
                inv_values = inv_subset.all_values(copy=False)
                if is_probabilistic:
                    arr[:, feature_indices, :] = inv_values.astype(np.float32)
                else:
                    arr[:, feature_indices] = inv_values.astype(np.float32)
            elif is_probabilistic:
                n_time, n_features, n_samples = subset.shape
                subset_2d = subset.transpose(0, 2, 1).reshape(-1, n_features)
                sklearn_scaler = None
                if hasattr(scaler, "_fitted_params") and scaler._fitted_params:
                    fp = scaler._fitted_params[0]
                    sklearn_scaler = fp["fitted"] if isinstance(fp, dict) and "fitted" in fp else fp
                inv_2d = sklearn_scaler.inverse_transform(subset_2d.astype(np.float64)) if sklearn_scaler else subset_2d
                inv_values = inv_2d.reshape(n_time, n_samples, n_features).transpose(0, 2, 1)
                arr[:, feature_indices, :] = inv_values.astype(np.float32)
            else:
                inv_subset = scaler.inverse_transform([temp_ts])[0]
                arr[:, feature_indices] = inv_subset.all_values(copy=False)
        return TimeSeries.from_times_and_values(
            times=ts.time_index, values=arr.astype(np.float32), columns=components,
            freq=ts.freq, static_covariates=ts.static_covariates
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_feature_scaler_mapping(self) -> Dict[str, str]:
        return self._feature_to_scaler.copy()

    def __repr__(self) -> str:
        mapping_str = ", ".join(f"{k}: {len(v)} features" for k, v in self._scaler_to_features.items())
        return f"FeatureScalerManager({mapping_str}, fitted={self._fitted})"
