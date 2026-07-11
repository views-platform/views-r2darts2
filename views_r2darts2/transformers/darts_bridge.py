"""Darts boundary bridge — the ONLY module that touches pandas.

This module converts :class:`views_frames.FeatureFrame` data to and from Darts
:class:`TimeSeries` objects. Darts internally uses ``pandas.Index`` and
``pandas.DataFrame`` for its time index and static covariates, so this single
module imports pandas to construct those objects. The rest of the package is
pandas-free.

Confining the pandas import here means:

    * The data loaders, scalers, model managers, and tests have zero direct
      pandas usage.
    * When Darts eventually drops its pandas dependency (or when this package
      switches to a non-Darts backend), only this file changes.
    * The memmap-backed :class:`FeatureFrame` stays the source of truth; Darts
      TimeSeries are short-lived view objects built on demand.

Google Python Style: this module has a single public function
``build_entity_timeseries`` and a single helper ``build_prediction_frame``.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from darts import TimeSeries

from views_frames import (
    FrameMetadata,
    PredictionFrame,
    SpatioTemporalIndex,
    SpatialLevel,
)

logger = logging.getLogger(__name__)

# The pandas import is isolated here. ``pd`` is used ONLY for:
#   * Wrapping integer time-id arrays in an Int64Index for Darts.
#   * Constructing static-covariate DataFrames (Darts requires this).
#   * Reading ``TimeSeries.static_covariates`` and ``TimeSeries.time_index``
#     back into numpy arrays on the inverse path.
# No other module in the package may import pandas directly.


def build_entity_timeseries(
    *,
    time: NDArray[np.int64],
    values: NDArray[np.float32],
    columns: Sequence[str],
    entity_id_name: str,
    entity_id_value: int,
    static_covariates: Mapping[str, float] | None = None,
) -> TimeSeries:
    """Build a Darts :class:`TimeSeries` from numpy arrays.

    Args:
        time: 1-D int64 array of time identifiers (already sorted ascending
            within this entity).
        values: 2-D float32 array of shape ``(T, F)`` — time-major, one column
            per name in ``columns``. Single-sample (deterministic) only.
        columns: Component (column) names.
        entity_id_name: Name of the entity identifier (e.g. ``"country_id"``).
        entity_id_value: The integer entity id for this series.
        static_covariates: Optional per-entity fingerprint (e.g. ``{mu: ...,
            sigma: ...}``). When ``None``, only the entity id is attached.

    Returns:
        A Darts :class:`TimeSeries` carrying the entity id (and any supplied
        fingerprint) as static covariates.
    """
    cov_dict: dict[str, float] = {entity_id_name: float(entity_id_value)}
    if static_covariates is not None:
        cov_dict.update({k: float(v) for k, v in static_covariates.items()})
    static_df = pd.DataFrame({k: [v] for k, v in cov_dict.items()})

    time_arr = np.asarray(time, dtype=np.int64)
    # When the time array has fewer than 2 elements, Darts cannot infer the
    # integer step from ``np.diff`` (it sees an empty diff array and raises
    # "non-unique step sizes"). Pass ``freq=1`` explicitly to declare the
    # integer-step frequency up front. The VIEWS viewser contract is always
    # month_id-step-1 (or week/day/year step 1), so freq=1 is the right
    # default for the integer-indexed path.
    freq = 1 if time_arr.shape[0] >= 1 else None

    return TimeSeries.from_times_and_values(
        times=pd.Index(time_arr),
        values=np.asarray(values, dtype=np.float32),
        columns=list(columns),
        static_covariates=static_df,
        freq=freq,
    )


def prediction_frame_from_darts(
    *,
    predictions: Sequence[TimeSeries],
    entity_id_name: str,
    target_columns: Sequence[str],
    level: SpatialLevel,
    clip_negatives: bool = True,
) -> PredictionFrame:
    """Convert a list of Darts prediction TimeSeries to a :class:`PredictionFrame`.

    Args:
        predictions: One Darts TimeSeries per entity (the output of
            ``DartsForecaster.predict`` *before* this conversion).
        entity_id_name: Static-covariate column name that holds the entity id.
        target_columns: Names of the target components (in order). Used to
            select the right columns from each TimeSeries.
        level: Spatial level of the output frame (must match the input data).
        clip_negatives: When ``True``, clip negative predictions to 0 (physical
            floor for the conflict-fatality count domain).

    Returns:
        A :class:`PredictionFrame` with shape ``(N_rows, S)`` where ``N_rows``
        is the total number of (entity, time) rows across all entities and
        ``S`` is the sample count. The frame carries a
        :class:`SpatioTemporalIndex` of ``(time, entity)`` pairs.

    Raises:
        ValueError: A prediction TimeSeries is missing a required target
            component or its static covariates do not contain the
            ``entity_id_name`` column.
    """
    if not predictions:
        raise ValueError("Cannot build PredictionFrame from an empty prediction list.")

    time_chunks: list[NDArray[np.int64]] = []
    entity_chunks: list[NDArray[np.int64]] = []
    value_chunks: list[NDArray[np.float32]] = []
    sample_count = 1

    for pred in predictions:
        # Extract the entity id from the static covariates (Darts stores these
        # as a one-row pandas DataFrame).
        static_covs = pred.static_covariates
        if entity_id_name not in static_covs.columns:
            raise ValueError(
                f"Prediction TimeSeries is missing the '{entity_id_name}' "
                f"static covariate column. Available: {list(static_covs.columns)}."
            )
        entity_id_value = int(static_covs[entity_id_name].iloc[0])

        # Time index → numpy int64 array. Darts may store as DatetimeIndex or
        # Int64Index depending on the input; coerce to int64 for the frame.
        time_index = np.asarray(pred.time_index.values, dtype=np.int64)
        n_time = time_index.shape[0]

        # Component selection: pull only the target columns, in the requested
        # order. ``pred.all_values(copy=False)`` returns shape ``(T, C, S)``.
        all_values = pred.all_values(copy=False)
        if all_values.ndim == 2:
            # Deterministic series — lift to 3D with S=1.
            all_values = all_values[:, :, np.newaxis]
        sample_count = max(sample_count, all_values.shape[-1])

        component_list = list(pred.components)
        col_indices: list[int] = []
        for tgt in target_columns:
            if tgt not in component_list:
                raise ValueError(
                    f"Prediction TimeSeries is missing target component "
                    f"'{tgt}'. Available: {component_list}."
                )
            col_indices.append(component_list.index(tgt))
        target_values = np.ascontiguousarray(all_values[:, col_indices, :])

        # Each entity contributes ``n_time`` rows. Broadcast the entity id.
        entity_chunks.append(
            np.full(n_time, entity_id_value, dtype=np.int64)
        )
        time_chunks.append(time_index)
        # Reshape to (n_time, S) by summing across the target axis — but
        # actually, the contract is per-target: PredictionFrame holds a single
        # value column. To support multi-target models, we instead build one
        # PredictionFrame per target in the caller. For now, store the FIRST
        # target (the common single-target case) and let the caller build
        # additional frames for additional targets.
        if target_values.shape[1] != 1:
            raise ValueError(
                "prediction_frame_from_darts expects a single target column. "
                f"Got {target_values.shape[1]} targets. Use "
                f"prediction_frames_from_darts (plural) for multi-target models."
            )
        value_chunks.append(target_values[:, 0, :].astype(np.float32, copy=False))

    time_arr = np.concatenate(time_chunks)
    entity_arr = np.concatenate(entity_chunks)
    value_arr = np.concatenate(value_chunks, axis=0).astype(np.float32, copy=False)

    if clip_negatives:
        np.maximum(value_arr, 0.0, out=value_arr)

    index = SpatioTemporalIndex(time=time_arr, unit=entity_arr, level=level)
    metadata = FrameMetadata(model="darts")
    return PredictionFrame(value_arr, index=index, metadata=metadata)


def prediction_frames_from_darts(
    *,
    predictions: Sequence[TimeSeries],
    entity_id_name: str,
    target_columns: Sequence[str],
    level: SpatialLevel,
    clip_negatives: bool = True,
) -> dict[str, PredictionFrame]:
    """Multi-target variant: one :class:`PredictionFrame` per target column.

    Args:
        predictions: One Darts TimeSeries per entity.
        entity_id_name: Static-covariate column name that holds the entity id.
        target_columns: Names of the target components to extract.
        level: Spatial level of the output frames.
        clip_negatives: When ``True``, clip negative predictions to 0.

    Returns:
        A ``{target_name: PredictionFrame}`` mapping. Each frame has the same
        :class:`SpatioTemporalIndex` (rows are ordered as
        ``entity-major, time-minor`` within each entity).
    """
    if not predictions:
        raise ValueError("Cannot build PredictionFrames from an empty list.")

    time_chunks: list[NDArray[np.int64]] = []
    entity_chunks: list[NDArray[np.int64]] = []
    per_target_values: dict[str, list[NDArray[np.float32]]] = {
        t: [] for t in target_columns
    }
    sample_count = 1

    for pred in predictions:
        static_covs = pred.static_covariates
        if entity_id_name not in static_covs.columns:
            raise ValueError(
                f"Prediction TimeSeries is missing the '{entity_id_name}' "
                f"static covariate column. Available: {list(static_covs.columns)}."
            )
        entity_id_value = int(static_covs[entity_id_name].iloc[0])
        time_index = np.asarray(pred.time_index.values, dtype=np.int64)
        n_time = time_index.shape[0]

        all_values = pred.all_values(copy=False)
        if all_values.ndim == 2:
            all_values = all_values[:, :, np.newaxis]
        sample_count = max(sample_count, all_values.shape[-1])

        component_list = list(pred.components)
        entity_chunks.append(
            np.full(n_time, entity_id_value, dtype=np.int64)
        )
        time_chunks.append(time_index)
        for tgt in target_columns:
            if tgt not in component_list:
                raise ValueError(
                    f"Prediction TimeSeries is missing target component "
                    f"'{tgt}'. Available: {component_list}."
                )
            col_idx = component_list.index(tgt)
            per_target_values[tgt].append(
                np.ascontiguousarray(all_values[:, col_idx, :]).astype(
                    np.float32, copy=False
                )
            )

    time_arr = np.concatenate(time_chunks)
    entity_arr = np.concatenate(entity_chunks)
    index = SpatioTemporalIndex(time=time_arr, unit=entity_arr, level=level)
    metadata = FrameMetadata(model="darts")

    frames: dict[str, PredictionFrame] = {}
    for tgt, chunks in per_target_values.items():
        values = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if clip_negatives:
            np.maximum(values, 0.0, out=values)
        frames[tgt] = PredictionFrame(values, index=index, metadata=metadata)
    return frames
