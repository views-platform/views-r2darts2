"""Shared inverse-transform utilities for Darts ``Scaler`` / ``Pipeline``.

The legacy code had two near-identical inverse-transform paths — one in
``FeatureScalerManager._inverse_transform_single_series`` and one in
``DartsForecaster._inverse_transform_target_scaler``. Both reached into the
Darts ``Scaler`` private attributes (``_fitted_params``, ``_fit_called``) to
manually reconstruct the sklearn object for the probabilistic 3-D case.

This module unifies the two paths into a single set of helpers that:

    * Preserve the EXACT numerical semantics of the legacy implementation —
      probabilistic 3-D series are reshaped to 2-D, inverse-transformed, and
      reshaped back; deterministic 2-D series are inverse-transformed directly.
    * Confine all Darts private-attribute access to one place.
    * Drop the dead ``fit=True`` branch (the legacy ``_transform_single_series``
      ``fit=True`` path was never called by any production code path).

Google Python Style.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler


def extract_fitted_sklearn_scaler(scaler: Scaler) -> object | None:
    """Extract the fitted sklearn object from a Darts :class:`Scaler`.

    The Darts ``Scaler`` stores its fitted sklearn transformer inside
    ``_fitted_params``. The exact layout depends on the Darts version — older
    versions used a tuple, newer versions may use a dict. This helper handles
    both shapes.

    Returns ``None`` when the scaler has not been fitted or the layout is
    unrecognized (the caller should treat this as a passthrough).
    """
    fitted_params = getattr(scaler, "_fitted_params", None)
    if not fitted_params:
        return None
    if isinstance(fitted_params, tuple) and len(fitted_params) >= 1:
        candidate = fitted_params[0]
        if isinstance(candidate, dict) and "fitted" in candidate:
            return candidate["fitted"]
        return candidate
    if isinstance(fitted_params, dict) and "fitted" in fitted_params:
        return fitted_params["fitted"]
    return fitted_params


def inverse_transform_probabilistic_subset(
    *,
    subset_3d: NDArray[np.float32],
    scaler: Scaler,
) -> NDArray[np.float32]:
    """Inverse-transform a 3-D probabilistic subset through a Darts ``Scaler``.

    Args:
        subset_3d: Shape ``(n_time, n_features, n_samples)`` float32.
        scaler: A fitted Darts :class:`Scaler` (single-step scaler; for
            ``Pipeline`` scalers use
            :func:`inverse_transform_subset_via_darts` instead).

    Returns:
        Shape ``(n_time, n_features, n_samples)`` float32, inverse-transformed.
    """
    n_time, n_features, n_samples = subset_3d.shape
    subset_2d = np.ascontiguousarray(
        subset_3d.transpose(0, 2, 1).reshape(-1, n_features)
    )
    sklearn_scaler = extract_fitted_sklearn_scaler(scaler)
    if sklearn_scaler is not None:
        inv_2d = sklearn_scaler.inverse_transform(
            subset_2d.astype(np.float64)
        )
    else:
        # Unfitted scaler — passthrough (preserves legacy behavior).
        inv_2d = subset_2d.astype(np.float64)
    inv_values = inv_2d.reshape(n_time, n_samples, n_features).transpose(0, 2, 1)
    return inv_values.astype(np.float32)


def inverse_transform_deterministic_subset(
    *,
    subset_2d: NDArray[np.float32],
    scaler: Scaler,
) -> NDArray[np.float32]:
    """Inverse-transform a 2-D deterministic subset through a Darts ``Scaler``.

    Args:
        subset_2d: Shape ``(n_time, n_features)`` float32.
        scaler: A fitted Darts :class:`Scaler`.

    Returns:
        Shape ``(n_time, n_features)`` float32, inverse-transformed.
    """
    sklearn_scaler = extract_fitted_sklearn_scaler(scaler)
    if sklearn_scaler is None:
        return subset_2d.astype(np.float32, copy=True)
    inv_2d = sklearn_scaler.inverse_transform(subset_2d.astype(np.float64))
    return inv_2d.astype(np.float32)


def fit_scaler_on_concatenated_subset(
    *,
    scaler: Scaler,
    series_list: Sequence[TimeSeries],
    feature_indices: list[int],
) -> None:
    """Fit a Darts :class:`Scaler` on the concatenated feature subset.

    Replaces the legacy ``FeatureScalerManager._fit_scalers_on_all_series``
    inner block. The scaler is fit on the concatenation of every series's
    feature subset, in 2-D (the sample axis is collapsed for fitting — this
    matches the legacy behavior).

    For a :class:`Pipeline` scaler, the fit goes through Darts' own
    ``Pipeline.fit`` on a dummy TimeSeries built from the concatenated values.
    For a single :class:`Scaler`, the underlying sklearn object is fit directly
    on the 2-D array and the Darts ``Scaler`` private attributes are wired up
    to match the post-fit state.
    """
    if not series_list:
        return

    all_subsets: list[NDArray[np.float32]] = []
    for ts in series_list:
        arr = ts.all_values(copy=False)
        subset = (
            arr[:, feature_indices, :] if arr.ndim == 3 else arr[:, feature_indices]
        )
        all_subsets.append(subset)
    combined = np.concatenate(all_subsets, axis=0)

    if isinstance(scaler, Pipeline):
        # Pipeline path: build a dummy TimeSeries from the 2-D concatenated
        # data and let Darts fit the pipeline. We use ``TimeSeries.from_values``
        # so no time index is needed (Darts auto-creates a RangeIndex).
        combined_2d = (
            combined if combined.ndim == 2 else combined[:, :, 0]
        )
        components = [series_list[0].components[i] for i in feature_indices]
        dummy_ts = TimeSeries.from_values(
            combined_2d.astype(np.float32), columns=components
        )
        scaler.fit([dummy_ts])
        return

    # Single-Scaler path: fit the underlying sklearn object directly on the
    # 2-D concatenated data, then wire the Darts Scaler private attributes to
    # mark it as fitted. This mirrors the legacy behavior exactly.
    if combined.ndim == 3:
        n_time, n_features, n_samples = combined.shape
        combined_2d = combined.transpose(0, 2, 1).reshape(-1, n_features)
    else:
        combined_2d = combined

    underlying = copy.deepcopy(scaler.transformer)
    fitted = underlying.fit(combined_2d.astype(np.float64))
    scaler._fitted_params = (fitted,)
    scaler._fit_called = True


def transform_subset_via_darts(
    *,
    subset_2d: NDArray[np.float32],
    columns: list[str],
    time_index: object,
    freq: object,
    static_covariates: object,
    scaler: Scaler | Pipeline,
) -> NDArray[np.float32]:
    """Transform a 2-D subset through a Darts ``Scaler`` / ``Pipeline``.

    Builds a temporary :class:`TimeSeries` from the subset, runs
    ``scaler.transform``, and returns the transformed values. Used by
    :class:`FeatureScalerManager` for the per-series transform step.
    """
    temp_ts = TimeSeries.from_times_and_values(
        times=time_index,
        values=subset_2d.astype(np.float32),
        columns=columns,
        freq=freq,
        static_covariates=static_covariates,
    )
    transformed = scaler.transform([temp_ts])[0]
    return transformed.all_values(copy=False)


def inverse_transform_subset_via_darts(
    *,
    subset: NDArray[np.float32],
    columns: list[str],
    time_index: object,
    freq: object,
    static_covariates: object,
    scaler: Scaler | Pipeline,
) -> NDArray[np.float32]:
    """Inverse-transform a subset through a Darts ``Scaler`` / ``Pipeline``.

    Builds a temporary :class:`TimeSeries`, runs ``scaler.inverse_transform``,
    and returns the inverse-transformed values. Used by
    :class:`FeatureScalerManager` for the per-series inverse step.
    """
    temp_ts = TimeSeries.from_times_and_values(
        times=time_index,
        values=subset.astype(np.float32),
        columns=columns,
        freq=freq,
        static_covariates=static_covariates,
    )
    inv = scaler.inverse_transform([temp_ts])[0]
    return inv.all_values(copy=False)
