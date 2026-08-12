"""Batch prediction utilities

The core idea: instead of building 259k Darts ``TimeSeries`` objects for all
entities simultaneously (which causes OOM), predict in **batches** of entities.

Each batch:
    1. Extracts a small numpy slice from the xarray dataset (e.g. 1000 entities).
    2. Applies log + scaler transforms in numpy (vectorized).
    3. Builds a **small** list of Darts TimeSeries (1000, not 259k).
    4. Calls ``model.predict_from_dataset(values_only=True)`` → raw numpy.
    5. Inverse-transforms the predictions in numpy.
    6. Writes the batch directly to the zarr-backed prediction scaffold via
       ``add_batch``.

Peak memory is bounded by the batch size, not the entity count.

Google Python Style.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def batch_predict(
    dataset: Any,
    model: Any,
    *,
    input_start: int,
    input_end: int,
    output_length: int,
    batch_size: int = 1000,
    num_samples: int = 1,
    mc_dropout: bool = False,
    use_cyclic_encoders: bool = False,
    apply_inverse: bool = True,
    clip_negatives: bool = True,
) -> dict[str, Any]:
    """Run prediction in entity batches, writing directly to a zarr scaffold.

    This function is the memory-safe prediction path for large entity counts
    (259k PRIO-GRID cells). It:

        1. Determines the full entity list from the dataset.
        2. Creates a zarr-backed prediction scaffold via
           ``dataset.create_prediction_scaffold``.
        3. Iterates over entities in batches of ``batch_size``:
            a. Extracts the input window for this batch as numpy.
            b. Applies log + scaler transforms (numpy-direct).
            c. Builds a small list of Darts TimeSeries (batch_size, not 259k).
            d. Calls ``model.predict_from_dataset(values_only=True)``.
            e. Inverse-transforms the predictions (numpy-direct).
            f. Writes the batch to the zarr scaffold via ``add_batch``.
        4. After all batches, converts the scaffold to PredictionFrames.

    Peak memory is ``O(batch_size * input_chunk_length * n_features)`` —
    typically a few hundred MB for batch_size=1000, vs OOM for 259k entities.

    Args:
        dataset: A :class:`ViewsDataset` with fitted scalers.
        model: A Darts :class:`TorchForecastingModel`.
        input_start: First time id of the input window (inclusive).
        input_end: Last time id of the input window (inclusive).
        output_length: Number of time steps to forecast.
        batch_size: Number of entities per batch (default 1000).
        num_samples: Number of probabilistic samples (1 for deterministic).
        mc_dropout: Enable MC dropout for probabilistic prediction.
        use_cyclic_encoders: Append sin/cos cyclic time encoders.
        apply_inverse: Apply inverse target scaler + log inverse.
        clip_negatives: Clip negative predictions to 0.

    Returns:
        A ``{target_name: PredictionFrame}`` dict.
    """
    from views_r2darts2.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )
    from views_r2darts2.infrastructure.exceptions import NumericalSanityError

    if not dataset.scalers_fitted:
        raise RuntimeError("Scalers not fitted. Call dataset.fit_scalers first.")

    # Get the full entity list from the dataset (cheap — just reading coords).
    entity_ids = dataset._ds[dataset._entity_id].values.astype("int64")
    n_entities = len(entity_ids)
    time_ids_input = list(range(input_start, input_end + 1))
    pred_time_start = input_end + 1
    pred_time_ids = np.arange(
        pred_time_start, pred_time_start + output_length, dtype="int64"
    )

    logger.info(
        "batch_predict: %d entities, batch_size=%d, input=[%d, %d], "
        "output_length=%d, n_samples=%d",
        n_entities, batch_size, input_start, input_end, output_length, num_samples,
    )

    # Create the prediction scaffold — zarr-backed, empty, ready for add_batch.
    scaffold = dataset.create_prediction_scaffold(
        entity_ids=entity_ids,
        time_ids=pred_time_ids,
        target_names=dataset.targets,
        sample_size=num_samples,
    )

    n_batches = (n_entities + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start_e = batch_idx * batch_size
        end_e = min(start_e + batch_size, n_entities)
        batch_entity_ids = entity_ids[start_e:end_e]
        batch_size_actual = len(batch_entity_ids)

        logger.info(
            "batch_predict: batch %d/%d, entities [%d:%d] (%d entities)",
            batch_idx + 1, n_batches, start_e, end_e, batch_size_actual,
        )

        # --- 1. Extract input window for this batch (numpy) ----------------
        target_arr, feature_arr, time_arr, ent_arr = dataset._extract_numpy_2d(
            target_names=dataset.targets,
            feature_names=dataset.features,
            time_ids=time_ids_input,
            entity_ids=batch_entity_ids.tolist(),
        )

        # --- 2. Apply log + scaler transforms (numpy-direct) ---------------
        target_arr, feature_arr = dataset._apply_transforms_numpy(
            target_arr, feature_arr, dataset.features, dataset.targets,
        )

        # --- 3. Build Darts TimeSeries for this batch only -----------------
        target_ts, past_cov_ts = dataset._build_batch_timeseries(
            target_arr=target_arr,
            feature_arr=feature_arr,
            target_names=dataset.targets,
            feature_names=dataset.features,
            time_arr=time_arr,
            entity_arr=ent_arr,
            use_cyclic_encoders=use_cyclic_encoders,
        )

        # Free numpy arrays — the TimeSeries hold copies now.
        del target_arr, feature_arr
        gc.collect()

        # --- 4. Predict with values_only=True (no output TimeSeries) -------
        try:
            predictions, _, _ = _predict_values_only(
                model=model,
                n=output_length,
                series=target_ts,
                past_covariates=past_cov_ts,
                num_samples=num_samples,
                mc_dropout=mc_dropout,
            )
        finally:
            del target_ts, past_cov_ts
            gc.collect()

        # --- 5. Inverse-transform (numpy-direct) ---------------------------
        if apply_inverse:
            predictions = dataset._inverse_transform_numpy_predictions(
                predictions
            )

        # Clip negatives.
        if clip_negatives:
            np.maximum(predictions, 0.0, out=predictions)

        # NaN/Inf guard.
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            raise NumericalSanityError(
                f"NaN/Inf in batch {batch_idx} predictions "
                f"(shape={predictions.shape})."
            )

        # --- 6. Write batch to the zarr scaffold ---------------------------
        # predictions shape: (batch_entities, n_time, n_components, n_samples)
        # We need to write per-target: extract target columns, reshape to
        # (batch_entities * n_time, n_samples) and call add_batch.
        n_targets = len(dataset.targets)
        target_indices = list(range(
            predictions.shape[2] - n_targets, predictions.shape[2]
        ))
        for t_idx, target_name in enumerate(dataset.targets):
            # (batch_entities, n_time, n_samples) → flatten to rows.
            vals = predictions[:, :, target_indices[t_idx], :]
            # Transpose to (n_time, batch_entities, n_samples) → flatten.
            vals = vals.transpose(1, 0, 2)  # (n_time, batch_entities, n_samples)
            vals = vals.reshape(-1, vals.shape[-1]).astype(np.float32)

            # Build time/entity arrays for this batch.
            t_grid, e_grid = np.meshgrid(
                pred_time_ids, batch_entity_ids, indexing="ij"
            )
            batch_times = t_grid.ravel()
            batch_entities = e_grid.ravel()

            scaffold.add_batch(
                times=batch_times,
                entities=batch_entities,
                values={f"pred_{target_name}": vals},
            )

        # Free the predictions array.
        del predictions
        gc.collect()

    logger.info("batch_predict: all %d batches complete", n_batches)

    # --- Convert the scaffold to PredictionFrames --------------------------
    frames = scaffold._to_prediction_frames()
    return frames


def _predict_values_only(
    model: Any,
    n: int,
    series: list,
    past_covariates: list | None = None,
    num_samples: int = 1,
    mc_dropout: bool = False,
) -> tuple:
    """Call ``model.predict_from_dataset(values_only=True)``.

    Returns ``(predictions, series_schemas, pred_starts)`` as raw numpy.
    """
    # Build the inference dataset from the TimeSeries list.
    dataset = model._build_inference_dataset(
        n=n,
        series=series,
        past_covariates=past_covariates,
        future_covariates=None,
        stride=0,
        bounds=None,
    )

    # Free the input TimeSeries — the dataset holds tensor data now.
    del series, past_covariates

    # Call predict_from_dataset with values_only=True.
    predictions, series_schemas, pred_starts = model.predict_from_dataset(
        n=n,
        dataset=dataset,
        num_samples=num_samples,
        mc_dropout=mc_dropout,
        values_only=True,
        verbose=False,
    )
    return predictions, series_schemas, pred_starts
