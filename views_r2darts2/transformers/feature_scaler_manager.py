"""Per-feature scaler manager (pandas-free).

Manages multiple Darts :class:`Scaler` / :class:`Pipeline` objects, each
operating on a disjoint subset of feature components. Replaces the legacy
``pd.RangeIndex`` usage with Darts' own ``TimeSeries.from_values`` (which
auto-creates a RangeIndex internally) so the manager has zero direct pandas
imports.

The fit / transform / inverse_transform paths use the shared helpers in
:mod:`views_r2darts2.transformers.inverse` to confine all Darts private-attribute
access to a single location and to avoid the duplicated probabilistic reshape
logic that lived in this class and in :class:`DartsForecaster` previously.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing import Pipeline

from views_r2darts2.transformers.inverse import (
    extract_fitted_sklearn_scaler,
    fit_scaler_on_concatenated_subset,
    inverse_transform_probabilistic_subset,
    inverse_transform_subset_via_darts,
    transform_subset_via_darts,
)
from views_r2darts2.transformers.scaler_selector import ScalerSelector


class FeatureScalerManager:
    """Manages multiple scalers for different feature groups.

    Each feature column is assigned to exactly one scaler; the assignment is
    derived from ``feature_scaler_map`` at construction time. Unassigned
    features fall back to ``default_scaler`` when ``all_features`` is provided.
    """

    def __init__(
        self,
        feature_scaler_map: Mapping[str, Any],
        default_scaler: str | None = "RobustScaler",
        all_features: list[str] | None = None,
    ) -> None:
        self.feature_scaler_map = dict(feature_scaler_map)
        self.default_scaler_name = default_scaler
        self.all_features = set(all_features or [])

        self._scalers: dict[str, Scaler | Pipeline] = {}
        self._feature_to_scaler: dict[str, str] = {}
        self._scaler_to_features: dict[str, list[str]] = {}
        self._fitted = False

        self._parse_config()

    # ------------------------------------------------------------------ parsing

    def _parse_config(self) -> None:
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

    def _parse_named_group_format(self) -> None:
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
                    raise ValueError(
                        f"Feature '{feat}' is assigned to multiple groups!"
                    )
                self._feature_to_scaler[feat] = scaler_key

    def _parse_simple_format(self) -> None:
        for scaler_name, features in self.feature_scaler_map.items():
            if not features:
                continue
            scaler_key = f"scaler_{scaler_name}"
            self._scalers[scaler_key] = self._instantiate_scaler(scaler_name)
            self._scaler_to_features[scaler_key] = list(features)
            for feat in features:
                if feat in self._feature_to_scaler:
                    raise ValueError(
                        f"Feature '{feat}' is assigned to multiple scalers!"
                    )
                self._feature_to_scaler[feat] = scaler_key

    def _assign_default_scaler(self) -> None:
        if not self.all_features or not self.default_scaler_name:
            return
        unmapped = [
            f for f in self.all_features if f not in self._feature_to_scaler
        ]
        if unmapped:
            scaler_key = "default"
            self._scalers[scaler_key] = self._instantiate_scaler(
                self.default_scaler_name
            )
            self._scaler_to_features[scaler_key] = unmapped
            for feat in unmapped:
                self._feature_to_scaler[feat] = scaler_key

    @staticmethod
    def _instantiate_scaler(scaler_cfg: Any) -> Scaler | Pipeline:
        """Delegate to :meth:`ScalerSelector.instantiate_darts_scaler`, rejecting ``None``.

        A ``None`` entry inside this manager would propagate into ``fit`` /
        ``transform`` / ``inverse_transform`` calls that assume every entry is a
        Darts ``Scaler`` or ``Pipeline``, producing a hard-to-trace
        ``AttributeError`` at fit time. Fail loudly at parse time instead.
        """
        if scaler_cfg is None:
            raise ValueError(
                "Scaler configuration cannot be None in FeatureScalerManager. "
                "Provide a valid scaler configuration (str, list, or dict) for "
                "each group, or set `default_scaler` on the manager so groups "
                "without an explicit `scaler` key have a fallback."
            )
        return ScalerSelector.instantiate_darts_scaler(scaler_cfg)

    # ------------------------------------------------------------------ fit / transform

    def fit_transform(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Fit all scalers on the concatenated series, then transform each."""
        if not self._scalers:
            return series_list
        self._fit_scalers_on_all_series(series_list)
        result = [self._transform_single_series(ts) for ts in series_list]
        self._fitted = True
        return result

    def _fit_scalers_on_all_series(self, series_list: list[TimeSeries]) -> None:
        if not series_list:
            return
        components = list(series_list[0].components)
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            feature_indices = [
                i for i, comp in enumerate(components) if comp in feature_names
            ]
            if not feature_indices:
                continue
            fit_scaler_on_concatenated_subset(
                scaler=scaler,
                series_list=series_list,
                feature_indices=feature_indices,
            )

    def transform(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Transform with already-fitted scalers; raises if not fitted."""
        if not self._fitted:
            raise RuntimeError("Scalers not fitted. Call fit_transform first.")
        if not self._scalers:
            return series_list
        return [self._transform_single_series(ts) for ts in series_list]

    def _transform_single_series(self, ts: TimeSeries) -> TimeSeries:
        components = list(ts.components)
        arr = ts.all_values(copy=True)
        is_probabilistic = arr.ndim == 3
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            feature_indices = [
                i for i, comp in enumerate(components) if comp in feature_names
            ]
            if not feature_indices:
                continue
            subset_names = [components[i] for i in feature_indices]
            if is_probabilistic:
                subset = arr[:, feature_indices, :]
                # For the transform path, take the first sample (deterministic
                # transform — same as legacy behavior).
                subset_2d = subset[:, :, 0]
            else:
                subset = arr[:, feature_indices]
                subset_2d = subset

            transformed = transform_subset_via_darts(
                subset_2d=subset_2d.astype(np.float32),
                columns=subset_names,
                time_index=ts.time_index,
                freq=ts.freq,
                static_covariates=ts.static_covariates,
                scaler=scaler,
            )
            if transformed.ndim == 3 and is_probabilistic:
                # Broadcast the transformed first sample back to all samples
                # (legacy behavior — the scaler is deterministic).
                for s in range(arr.shape[-1]):
                    arr[:, feature_indices, s] = transformed[:, :, 0]
            elif transformed.ndim == 2 and is_probabilistic:
                for s in range(arr.shape[-1]):
                    arr[:, feature_indices, s] = transformed
            elif transformed.ndim == 3 and not is_probabilistic:
                arr[:, feature_indices] = transformed[:, :, 0]
            else:
                arr[:, feature_indices] = transformed


        return TimeSeries.from_times_and_values(
            times=ts.time_index,
            values=arr.astype(np.float32),
            columns=components,
            freq=ts.freq,
            static_covariates=ts.static_covariates,
        )

    # ------------------------------------------------------------------ inverse

    def inverse_transform(self, series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Inverse-transform with already-fitted scalers; raises if not fitted."""
        if not self._fitted:
            raise RuntimeError("Scalers not fitted.")
        if not self._scalers:
            return series_list
        return [self._inverse_transform_single_series(ts) for ts in series_list]

    def _inverse_transform_single_series(self, ts: TimeSeries) -> TimeSeries:

        components = list(ts.components)
        arr = ts.all_values(copy=True)
        is_probabilistic = arr.ndim == 3
        for scaler_key, scaler in self._scalers.items():
            feature_names = self._scaler_to_features.get(scaler_key, [])
            feature_indices = [
                i for i, comp in enumerate(components) if comp in feature_names
            ]
            if not feature_indices:
                continue
            subset_names = [components[i] for i in feature_indices]

            if isinstance(scaler, Pipeline):
                # Pipeline path: delegate to Darts. Build a temporary series,
                # inverse-transform, and write back. Handle both 2-D and 3-D.
                if is_probabilistic:
                    subset = arr[:, feature_indices, :]
                    n_time, n_features, n_samples = subset.shape
                    # Pipeline.inverse_transform handles 3-D when the inner
                    # scalers support it; otherwise we reshape to 2-D, transform
                    # per sample, and reshape back. Use the per-sample 2-D path
                    # for parity with the legacy code.
                    for s in range(n_samples):
                        sample_2d = subset[:, :, s]
                        inv = inverse_transform_subset_via_darts(
                            subset=sample_2d.astype(np.float32),
                            columns=subset_names,
                            time_index=ts.time_index,
                            freq=ts.freq,
                            static_covariates=ts.static_covariates,
                            scaler=scaler,
                        )
                        if inv.ndim == 3:
                            inv = inv[:, :, 0]
                        arr[:, feature_indices, s] = inv.astype(np.float32)
                else:
                    subset = arr[:, feature_indices]
                    inv = inverse_transform_subset_via_darts(
                        subset=subset.astype(np.float32),
                        columns=subset_names,
                        time_index=ts.time_index,
                        freq=ts.freq,
                        static_covariates=ts.static_covariates,
                        scaler=scaler,
                    )
                    if inv.ndim == 3:
                        inv = inv[:, :, 0]
                    arr[:, feature_indices] = inv.astype(np.float32)
                continue

            # Single-Scaler path.
            if is_probabilistic:
                subset = arr[:, feature_indices, :]
                inv_values = inverse_transform_probabilistic_subset(
                    subset_3d=subset.astype(np.float32),
                    scaler=scaler,
                )
                arr[:, feature_indices, :] = inv_values
            else:
                subset = arr[:, feature_indices]
                sklearn_scaler = extract_fitted_sklearn_scaler(scaler)
                if sklearn_scaler is not None:
                    inv = sklearn_scaler.inverse_transform(
                        subset.astype(np.float64)
                    )
                    arr[:, feature_indices] = inv.astype(np.float32)
                else:
                    # Unfitted — passthrough.
                    pass

        return TimeSeries.from_times_and_values(
            times=ts.time_index,
            values=arr.astype(np.float32),
            columns=components,
            freq=ts.freq,
            static_covariates=ts.static_covariates,
        )

    # ------------------------------------------------------------------ public

    @property
    def is_fitted(self) -> bool:
        """Whether the scalers have been fitted."""
        return self._fitted

    def get_feature_scaler_mapping(self) -> dict[str, str]:
        """Return a copy of the ``{feature: scaler_key}`` mapping."""
        return dict(self._feature_to_scaler)

    def __repr__(self) -> str:
        mapping_str = ", ".join(
            f"{k}: {len(v)} features" for k, v in self._scaler_to_features.items()
        )
        return f"FeatureScalerManager({mapping_str}, fitted={self._fitted})"
