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
    ``_fitted_params``. The layout varies across Darts versions and fit modes:

        * ``global_fit=True``  → ``list`` of length 1: ``[sklearn_scaler]``
        * ``global_fit=False`` → ``list`` of length N: one per fitted series
        * Older Darts versions → ``tuple`` of the same shapes
        * Some Darts versions   → ``dict`` with a ``"fitted"`` key
        * Edge case             → nested lists: ``[[sklearn_scaler]]``

    Recursively unwraps lists/tuples until it finds an object with an
    ``inverse_transform`` method (the sklearn transformer itself) or a dict
    with a ``"fitted"`` key.

    Returns ``None`` when the scaler has not been fitted or the layout is
    unrecognized.
    """
    fitted_params = getattr(scaler, "_fitted_params", None)
    if not fitted_params:
        return None
    return _unwrap_fitted_params(fitted_params)


def _unwrap_fitted_params(obj: object, depth: int = 0) -> object | None:
    """Recursively unwrap ``_fitted_params`` until we find the sklearn object.

    Args:
        obj: The current candidate (list, tuple, dict, or sklearn object).
        depth: Recursion depth guard (max 5 to prevent infinite loops).

    Returns:
        The sklearn transformer (an object with ``inverse_transform``), or
        ``None`` if not found.
    """
    if depth > 5:
        return None

    # Dict with "fitted" key — Darts' per-series layout.
    if isinstance(obj, dict) and "fitted" in obj:
        return _unwrap_fitted_params(obj["fitted"], depth + 1)

    # List or tuple — unwrap the first element (global_fit=True) or recurse.
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return None
        # Try the first element; if it unwraps to a transformer, return it.
        return _unwrap_fitted_params(obj[0], depth + 1)

    # Found an object with inverse_transform — this is the sklearn transformer.
    if hasattr(obj, "inverse_transform"):
        return obj

    return None


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

    Fallback: If the sklearn transformer cannot be extracted from
    ``_fitted_params`` (e.g., the layout is unrecognized), the function
    applies the scaler's underlying ``transformer.inverse_func`` elementwise
    on the 3-D array. This is correct for stateless transforms like
    ``AsinhTransform`` (``FunctionTransformer`` with ``func=arcsinh``,
    ``inverse_func=sinh``) which work elementwise and don't need 2-D input.
    """
    n_time, n_features, n_samples = subset_3d.shape
    sklearn_scaler = extract_fitted_sklearn_scaler(scaler)
    if sklearn_scaler is not None:
        # Standard path: reshape to 2-D, inverse-transform, reshape back.
        subset_2d = np.ascontiguousarray(
            subset_3d.transpose(0, 2, 1).reshape(-1, n_features)
        )
        inv_2d = sklearn_scaler.inverse_transform(
            subset_2d.astype(np.float64)
        )
        inv_values = inv_2d.reshape(n_time, n_samples, n_features).transpose(0, 2, 1)
        return inv_values.astype(np.float32)

    # Fallback: use the Darts Scaler's underlying transformer directly.
    # This works for stateless transforms (FunctionTransformer with
    # func/inverse_func) which apply elementwise and don't need 2-D input.
    underlying = getattr(scaler, "transformer", None)
    if underlying is not None and hasattr(underlying, "inverse_func"):
        # FunctionTransformer stores inverse_func; apply elementwise.
        inv_func = underlying.inverse_func
        if inv_func is not None:
            # Clamp to prevent float32 overflow on the inverse. For AsinhTransform,
            # the inverse is sinh — values > ~88 overflow float32. The RINorm
            # patch already clamps to [-88, 88] before sinh; we mirror that here
            # for the scaler inverse path.
            clamped = np.clip(subset_3d.astype(np.float64), -88.0, 88.0)
            inv_values = inv_func(clamped)
            return inv_values.astype(np.float32)

    # Last resort: passthrough (preserves legacy behavior for unfitted scalers).
    return subset_3d.astype(np.float32, copy=True)


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

    Fallback: If the sklearn transformer cannot be extracted, applies the
    scaler's underlying ``transformer.inverse_func`` elementwise (correct for
    stateless transforms like ``AsinhTransform``).
    """
    sklearn_scaler = extract_fitted_sklearn_scaler(scaler)
    if sklearn_scaler is not None:
        inv_2d = sklearn_scaler.inverse_transform(subset_2d.astype(np.float64))
        return inv_2d.astype(np.float32)

    # Fallback: use the underlying transformer's inverse_func elementwise.
    underlying = getattr(scaler, "transformer", None)
    if underlying is not None and hasattr(underlying, "inverse_func"):
        inv_func = underlying.inverse_func
        if inv_func is not None:
            # Clamp to prevent float32 overflow on the inverse (sinh overflow).
            clamped = np.clip(subset_2d.astype(np.float64), -88.0, 88.0)
            return inv_func(clamped).astype(np.float32)

    # Last resort: passthrough.
    return subset_2d.astype(np.float32, copy=True)


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
