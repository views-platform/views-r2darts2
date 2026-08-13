"""Darts forecaster engine — slim, delegates data ops to :class:`ViewsDataset`.

The forecaster is now a thin orchestration layer that couples a Darts
:class:`TorchForecastingModel` with a :class:`ViewsDataset` (which owns all
data operations: loading, slicing, scaling, Darts TimeSeries construction,
inverse transforms, and PredictionFrame conversion).

Data flow:

    ViewsDataset (zarr-backed, owns scalers)
        ↓  dataset.get_scaled_darts_timeseries()
    List[TimeSeries] (scaled targets) + List[TimeSeries] (scaled past_cov)
        ↓  model.fit / model.predict
    List[TimeSeries] (predictions, scaled space)
        ↓  dataset.ingest_darts_predictions()
    dict[str, PredictionFrame]  (one per target column)

The forecaster itself holds no data manipulation logic — it only manages the
model lifecycle (device, fit, predict, save/load) and the partition windows.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Union

import numpy as np
import torch
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.models.forecasting.sklearn_model import SKLearnModel

from views_frames import PredictionFrame

from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.infrastructure.device import get_device as _get_device
from views_r2darts2.infrastructure.exceptions import NumericalSanityError
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate

logger = logging.getLogger(__name__)


class DartsForecaster:
    """Slim orchestration layer: model + partition + dataset.

    The dataset owns all data operations (loading, slicing, scaling, Darts
    TimeSeries construction, inverse transforms). The forecaster only manages
    the model lifecycle and the train/predict flow.

    Intent Contract:
        - Purpose: Couple a Darts model with a partition and a dataset, and
          orchestrate the train/predict flow.
        - Non-Goals: Does not own data manipulation logic (that's the
          dataset's job). Does not manage experiment orchestration (that's
          :class:`DartsForecastingModelManager`).
        - Guarantees:
            - Scalers are fitted ONLY on training data (via the dataset's
              ``fit_scalers`` with a time_ids filter).
            - Predictions are inverse-transformed and clipped to non-negative.
        - Failure Behavior: Raises ``RuntimeError`` if predict is called
          before scalers are fitted.
    """

    def __init__(
        self,
        dataset: ViewsDataset,
        model: Union[TorchForecastingModel, SKLearnModel],
        partition_dict: dict,
        feature_scaler: str | None = None,
        target_scaler: str | None = None,
        log_targets: bool = False,
        log_features: list[str] | None = None,
        feature_scaler_map: Mapping[str, Any] | None = None,
        random_state: int | None = None,
        static_covariate_stats: Mapping[str, Any] | None = None,
        checkpoint_mode: str = "best",
        use_cyclic_encoders: bool = False,
    ) -> None:
        """Initialize the forecaster.

        Args:
            dataset: The :class:`ViewsDataset` (zarr-backed, owns scalers).
            model: A Darts :class:`TorchForecastingModel` or
                :class:`SKLearnModel` instance. Sklearn models bypass the
                torch-specific train/predict path (no checkpoints, no device
                moves, no ``_build_inference_dataset``).
            partition_dict: ``{"train": (start, end), "test": (start, end)}``.
            feature_scaler: Scaler name for all features.
            target_scaler: Scaler name for targets.
            log_targets: Apply ``log1p`` to targets before scaling.
            log_features: Feature names to apply ``log1p`` to.
            feature_scaler_map: Per-feature scaler map (takes precedence).
            random_state: Random seed (mandatory).
            static_covariate_stats: Static covariate config (unused in slim
                version — static covariates are computed by the dataset).
            checkpoint_mode: ``"best"`` or ``"last"``. Ignored for sklearn
                models (no checkpoint machinery).
            use_cyclic_encoders: Append sin/cos cyclic time encoders.

        Raises:
            ValueError: ``random_state`` is ``None`` or ``checkpoint_mode``
                is invalid.
        """
        self.dataset = dataset
        self.model = model
        self._train_start, self._train_end = partition_dict["train"]
        self._test_start, self._test_end = partition_dict["test"]

        if random_state is None:
            raise ValueError(
                "MANDATORY PARAMETER MISSING: random_state must be provided."
            )
        self.random_state = random_state

        if checkpoint_mode not in ("best", "last"):
            raise ValueError(
                f"checkpoint_mode must be 'best' or 'last', got {checkpoint_mode!r}"
            )
        self._checkpoint_mode = checkpoint_mode
        self._use_cyclic_encoders = use_cyclic_encoders

        # Store scaler configs — the dataset's fit_scalers will use them.
        self._target_scaler_cfg = target_scaler
        self._feature_scaler_cfg = feature_scaler
        self._feature_scaler_map_cfg = (
            dict(feature_scaler_map) if feature_scaler_map else None
        )
        self._log_targets = bool(log_targets)
        self._log_features = list(log_features or [])

        # Warn about double log transform.
        if self._log_targets and target_scaler == "LogTransform":
            logger.warning(
                "Both log_targets=True and target_scaler='LogTransform' — "
                "disabling log_targets to avoid double transformation."
            )
            self._log_targets = False
        if self._log_features and feature_scaler == "LogTransform":
            raise ValueError(
                "Both log_features and feature_scaler='LogTransform' — "
                "use only one transformation method."
            )

        self.scaler_fitted = False
        self.device = _get_device()
        # Store the full partition dict for later use (e.g., re-attaching
        # the dataset to a loaded sklearn model).
        self._partition_dict = dict(partition_dict)
        # Detect sklearn-based models — these bypass the torch-specific
        # train/predict path. The MarkovModel is the canonical example.
        self._is_sklearn_model = isinstance(self.model, SKLearnModel)
        if self._is_sklearn_model:
            # Sklearn models have no device concept — force 'cpu' so the
            # downstream parallelism check in DartsForecastingModelManager
            # (``if forecaster.device == "cpu"``) takes the multi-worker path.
            self.device = "cpu"
            logger.info(
                "Using sklearn-based model %s — torch device checks bypassed.",
                type(self.model).__name__,
            )
            # Give the model access to the dataset so it can use the full
            # data infrastructure (FeatureFrame API, zarr-backed lazy
            # loading, index validation). The model's fit/predict methods
            # check for ``self._dataset`` and use it when available.
            if hasattr(self.model, "set_dataset"):
                self.model.set_dataset(self.dataset, partition_dict=self._partition_dict)
        else:
            logger.info("Using device: %s", self.device)
            self._move_model_to_device()

    # ------------------------------------------------------------------ device

    def _move_model_to_device(self) -> None:
        """Move the model to the configured device.

        No-op for sklearn-based models (no torch parameters to move).
        """
        if self._is_sklearn_model:
            return
        if hasattr(self.model, "to_device"):
            self.model.to_device(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(self.device)

    @staticmethod
    def get_device() -> str:
        """Return the device type (delegates to infrastructure.device)."""
        return _get_device()

    # ------------------------------------------------------------------ train

    def train(self) -> None:
        """Train the model.

        Delegates scaler fitting to the dataset, then calls ``model.fit`` with
        the scaled TimeSeries from the training partition. Handles the
        forecasting-mode carve (when the test partition is too short for a
        validation window, carves val from the train end).

        For sklearn-based models (e.g. MarkovModel), the scaler-fitting,
        validation-set carve, and torch-specific kwargs (``val_series``,
        ``val_past_cov``, ``dataloader_kwargs``) are all skipped — sklearn
        models do their own internal transforms and do not consume torch
        plumbing. The model is given access to the dataset directly (via
        ``set_dataset`` in ``__init__``) so it can use the FeatureFrame API.
        """
        # Sklearn models bypass the scaler-fitting and validation-set carve
        # entirely. The MarkovModel does its own log1p transform and does
        # not use external scalers — fitting scalers here would be wasted
        # work (and could cause double-transform issues if log_targets is
        # also set). The model uses the dataset directly via set_dataset.
        if self._is_sklearn_model:
            # Mark the scaler as "fitted" so predict() doesn't raise.
            # No actual scalers are fitted — the MarkovModel applies its
            # own transforms internally.
            self.scaler_fitted = True
            # Call model.fit with the raw (unscaled) TimeSeries. When a
            # dataset is attached, the model ignores these series and
            # uses the dataset's FeatureFrame API instead. The series are
            # passed only for darts interface compliance.
            train_time_ids = list(range(self._train_start, self._train_end + 1))
            target_series, past_covariates = self.dataset.to_darts_timeseries(
                time_ids=train_time_ids,
                use_cyclic_encoders=self._use_cyclic_encoders,
            ), None
            # Split into targets and past_covariates (the MarkovModel
            # expects the standard darts two-argument interface).
            target_series, past_covariates = self.dataset._split_targets_covariates(
                target_series
            )
            self.model.fit(
                series=target_series,
                past_covariates=past_covariates,
            )
            return

        # --- Torch model path ---
        # Fit scalers and return the already-transformed training series in one
        # zarr load — avoids a second full read for the same time range.
        train_time_ids = list(range(self._train_start, self._train_end + 1))
        target_series, past_covariates = self.dataset.fit_scalers(
            target_scaler=self._target_scaler_cfg,
            feature_scaler=self._feature_scaler_cfg,
            feature_scaler_map=self._feature_scaler_map_cfg,
            log_targets=self._log_targets,
            log_features=self._log_features,
            time_ids=train_time_ids,
            return_series=True,
            use_cyclic_encoders=self._use_cyclic_encoders,
        )
        self.scaler_fitted = True

        # Validation set: test partition (or carved from train end for
        # forecasting mode).
        val_targets, val_past_cov = self._build_validation_set()

        # Train.
        import os
        num_workers = min(max((os.cpu_count() or 1) // 2, 0), 8)
        dataloader_kwargs = (
            {"num_workers": num_workers, "persistent_workers": False}
            if num_workers > 0
            else {}
        )
        self.model.fit(
            series=target_series,
            past_covariates=past_covariates,
            val_series=val_targets,
            val_past_covariates=val_past_cov,
            dataloader_kwargs=dataloader_kwargs,
            verbose=True,
        )

        if self._checkpoint_mode == "last":
            try:
                self.model.load_weights_from_checkpoint(best=False)
                logger.info("checkpoint_mode='last': reloaded final epoch weights.")
            except Exception as exc:
                logger.warning("checkpoint_mode='last' reload failed: %s", exc)

    def _build_validation_set(self):
        """Build the validation set (test partition or carved from train end)."""
        icl = self.model.input_chunk_length
        ocl = self.model.output_chunk_length
        val_start = self._test_start - icl
        val_end = self._test_end

        val_targets, val_past_cov = self.dataset.get_scaled_darts_timeseries(
            time_ids=list(range(val_start, val_end + 1)),
            use_cyclic_encoders=self._use_cyclic_encoders,
        )

        # Check if val is too short (forecasting mode).
        min_val_len = icl + ocl
        max_val_len = max((len(ts) for ts in val_targets), default=0)
        if max_val_len < min_val_len:
            # Carve from train end.
            carved_start = self._train_end - ocl - icl + 1
            logger.info(
                "Forecasting mode: val too short (%d < %d). Carving [%d, %d].",
                max_val_len, min_val_len, carved_start, self._train_end,
            )
            # Refit scalers on trimmed train (no holdout leakage).
            trimmed_end = self._train_end - ocl
            self.dataset.fit_scalers(
                target_scaler=self._target_scaler_cfg,
                feature_scaler=self._feature_scaler_cfg,
                feature_scaler_map=self._feature_scaler_map_cfg,
                log_targets=self._log_targets,
                log_features=self._log_features,
                time_ids=list(range(self._train_start, trimmed_end + 1)),
            )
            val_targets, val_past_cov = self.dataset.get_scaled_darts_timeseries(
                time_ids=list(range(carved_start, self._train_end + 1)),
                use_cyclic_encoders=self._use_cyclic_encoders,
            )

        return val_targets, val_past_cov

    # ------------------------------------------------------------------ predict

    def predict(
        self,
        sequence_number: int,
        output_length: int = 36,
        **predict_kwargs: Any,
    ) -> dict[str, PredictionFrame]:
        """Generate forecasts and return them as a per-target dict of frames.

        Uses Darts' ``predict_from_dataset(values_only=True)`` to bypass
        output :class:`TimeSeries` construction entirely — the model returns
        raw numpy arrays, which are inverse-transformed and converted to
        :class:`PredictionFrame` objects without ever building per-entity
        Darts ``TimeSeries`` on the output side. This is critical for
        large entity counts (e.g. 259k PRIO-GRID cells) where building
        259k output TimeSeries objects causes OOM kills.

        For sklearn-based models (e.g. MarkovModel), the predict path is
        simpler: the model's ``predict(n, series, past_covariates)`` method
        is called directly (returning a list of :class:`TimeSeries`), and
        the resulting series are fed to ``ingest_darts_predictions`` (the
        TimeSeries-based ingestion path). Sklearn models do not support
        ``values_only=True``.

        Args:
            sequence_number: Rolling-origin sequence index (0 = first test step).
            output_length: Number of time steps to forecast.
            **predict_kwargs: Forwarded to ``model.predict``.

        Returns:
            ``{target_name: PredictionFrame}`` dict.

        Raises:
            RuntimeError: Scalers not fitted.
            NumericalSanityError: NaNs/Infs in predictions.
        """
        if not self.scaler_fitted:
            raise RuntimeError(
                "predict() called before scalers were fitted. "
                "Call train() or load_model() first."
            )

        import gc

        # Lock entropy for reproducible probabilistic samples.
        ReproducibilityGate.Data.lock_entropy(self.random_state)

        # Get input window for this sequence.
        # Sklearn models do not use external scalers — fetch raw
        # (unscaled) TimeSeries directly from the dataset. Torch models
        # use the scaler-fitted path (``get_scaled_darts_timeseries``).
        icl = self.model.input_chunk_length
        start = self._test_start + sequence_number - icl
        end = self._test_start - 1 + sequence_number
        if self._is_sklearn_model:
            series_list = self.dataset.to_darts_timeseries(
                time_ids=list(range(start, end + 1)),
                use_cyclic_encoders=self._use_cyclic_encoders,
            )
            target_series, past_covariates = self.dataset._split_targets_covariates(
                series_list
            )
        else:
            target_series, past_covariates = self.dataset.get_scaled_darts_timeseries(
                time_ids=list(range(start, end + 1)),
                use_cyclic_encoders=self._use_cyclic_encoders,
            )

        # Capture the entity/time index before freeing the input series.
        # All entities share the same time index (integer step 1).
        entity_ids = np.array([
            int(ts.static_covariates[self.dataset._entity_id].iloc[0])
            for ts in target_series
        ], dtype=np.int64)
        time_index_start = int(target_series[0].time_index[0])
        # The prediction time ids start one step after the input window ends.
        pred_time_start = int(target_series[0].time_index[-1]) + 1
        pred_time_ids = np.arange(
            pred_time_start, pred_time_start + output_length, dtype=np.int64
        )

        # --- Sklearn-model predict path -----------------------------------
        # SKLearnModel-based models (e.g. MarkovModel) do not support the
        # ``_build_inference_dataset`` / ``predict_from_dataset`` interface.
        # Call ``model.predict`` directly — it returns a list of TimeSeries
        # (one per entity) which we then ingest via the TimeSeries-based
        # path.
        if self._is_sklearn_model:
            try:
                pred_series = self.model.predict(
                    n=output_length,
                    series=target_series,
                    past_covariates=past_covariates,
                )
            except Exception as exc:
                logger.error("Error during sklearn-model prediction: %s", exc)
                raise
            finally:
                del target_series, past_covariates
                gc.collect()

            # Normalise to a list.
            if isinstance(pred_series, TimeSeries):
                pred_series = [pred_series]
            elif not isinstance(pred_series, list):
                pred_series = list(pred_series)

            # Audit for NaN/Inf.
            for i, ts in enumerate(pred_series):
                arr = ts.all_values(copy=False)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    raise NumericalSanityError(
                        f"NaN/Inf in sklearn-model predictions (series {i}, "
                        f"shape={arr.shape})."
                    )

            frames = self.dataset.ingest_darts_predictions(
                predictions=pred_series,
                apply_inverse=True,
                clip_negatives=True,
            )
            del pred_series
            gc.collect()

            for target, frame in frames.items():
                if np.isnan(frame.values).any():
                    raise NumericalSanityError(
                        f"NaNs in final PredictionFrame for target '{target}'."
                    )
            return frames

        # --- Torch predict path: values_only=True -------------------------
        # Device management.
        self._ensure_model_on_device()

        # Decide between the in-memory path (single predict_from_dataset
        # call, full predictions array in RAM) and the streaming path
        # (entity-batched predict_from_dataset, each batch written to a
        # zarr-backed scaffold). The streaming path keeps peak memory
        # bounded for probabilistic forecasts (num_samples large) and/or
        # huge entity counts (e.g. 259k PRIO-GRID cells).
        num_samples = predict_kwargs.get("num_samples")
        # use_streaming = self._should_stream_predictions(
        #     n_entities=len(entity_ids),
        #     n_time=output_length,
        #     num_samples=num_samples,
        # )
        use_streaming = True

        if use_streaming:
            frames = self._predict_streaming(
                target_series=target_series,
                past_covariates=past_covariates,
                entity_ids=entity_ids,
                pred_time_ids=pred_time_ids,
                output_length=output_length,
                **predict_kwargs,
            )
            del target_series, past_covariates
            gc.collect()
            for target, frame in frames.items():
                if np.isnan(frame.values).any():
                    raise NumericalSanityError(
                        f"NaNs in final PredictionFrame for target '{target}'."
                    )
            return frames

        # In-memory path: single predict_from_dataset call.
        # Bypasses Darts' output TimeSeries construction entirely. The model
        # returns (predictions, series_schemas, pred_starts) as raw numpy.
        # This avoids building 259k TimeSeries + 259k pandas DataFrames on
        # the output side — the single biggest memory consumer.
        try:
            # Use predict_from_dataset with values_only=True.
            # We need to build the inference dataset ourselves, then call
            # predict_from_dataset directly.
            predictions, series_schemas, pred_starts = (
                self._predict_values_only(
                    n=output_length,
                    series=target_series,
                    past_covariates=past_covariates,
                    **predict_kwargs,
                )
            )
        except Exception as exc:
            logger.error("Error during prediction: %s", exc)
            raise
        finally:
            # Free the input TimeSeries immediately — they're no longer needed.
            del target_series, past_covariates
            gc.collect()

        # Audit model output for numerical sanity (numpy-direct)..
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            raise NumericalSanityError(
                f"NaN/Inf in model predictions (shape={predictions.shape})."
            )

        # Inverse-transform + convert to PredictionFrames (numpy-direct).
        # This bypasses ingest_darts_predictions (which needs TimeSeries)
        # and uses ingest_numpy_predictions (which works on raw numpy).
        frames = self.dataset.ingest_numpy_predictions(
            predictions=predictions,
            entity_ids=entity_ids,
            time_ids=pred_time_ids,
            apply_inverse=True,
            clip_negatives=True,
        )

        # Free the raw predictions array.
        del predictions
        gc.collect()

        # Final NaN guard on the PredictionFrame values.
        for target, frame in frames.items():
            if np.isnan(frame.values).any():
                raise NumericalSanityError(
                    f"NaNs in final PredictionFrame for target '{target}'."
                )
        return frames

    def _predict_values_only(
        self,
        n: int,
        series: list,
        past_covariates: list | None = None,
        **predict_kwargs: Any,
    ) -> tuple:
        """Run prediction with ``values_only=True`` to bypass output TimeSeries.

        Returns:
            ``(predictions, series_schemas, pred_starts)`` where:
            * ``predictions``: shape ``(n_entities, n_time, n_components, n_samples)``
            * ``series_schemas``: list of schema dicts (one per entity)
            * ``pred_starts``: list of prediction start times (one per entity)
        """
        # Extract predict kwargs.
        num_samples = predict_kwargs.get("num_samples")
        mc_dropout = predict_kwargs.get("mc_dropout")
        batch_size = predict_kwargs.get("batch_size")
        verbose = predict_kwargs.get("verbose", True)
        # Any remaining kwargs are ignored (Darts doesn't accept them).

        # Build the inference dataset from the TimeSeries list.
        # This is where the input TimeSeries are consumed — after this, the
        # dataset holds torch tensors, not TimeSeries, so we can free the
        # original list.
        dataset = self.model._build_inference_dataset(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=None,
            stride=0,
            bounds=None,
        )

        # Free the input TimeSeries — the dataset has already extracted the
        # tensor data from them.
        del series, past_covariates

        # Call predict_from_dataset with values_only=True.
        predictions, series_schemas, pred_starts = (
            self.model.predict_from_dataset(
                n=n,
                dataset=dataset,
                batch_size=batch_size,
                verbose=verbose,
                num_samples=num_samples,
                mc_dropout=mc_dropout,
                values_only=True,
            )
        )
        return predictions, series_schemas, pred_starts

    # ------------------------------------------------------------------ #
    # Streaming prediction path (zarr-backed scaffold)
    # ------------------------------------------------------------------ #

    #: Threshold above which the streaming path is used. The full
    #: predictions array has shape
    #: ``(n_entities, n_time, n_components, n_samples)`` float32 — 4 bytes
    #: per cell. When the array exceeds this many cells, we switch to the
    #: streaming path (entity-batched predict_from_dataset + zarr scaffold).
    #: Default: 50M cells (~200 MB float32) — large enough that the
    #: in-memory path is still used for small deterministic forecasts,
    #: small enough that probabilistic PGM forecasts (259k × 36 × 3 × 500
    #: ≈ 14B cells) always stream.
    STREAMING_CELL_THRESHOLD: int = 1

    #: Default entity batch size for the streaming path. Each batch runs
    #: one ``predict_from_dataset`` call, so smaller batches mean lower
    #: peak memory but more calls. Tuned for PGM (259k entities → ~260
    #: batches of 1000).
    STREAMING_ENTITY_BATCH: int = 1000

    def _should_stream_predictions(
        self, *, n_entities: int, n_time: int, num_samples: int
    ) -> bool:
        """Return ``True`` when the full predictions array would be large.

        The threshold is :attr:`STREAMING_CELL_THRESHOLD` cells. The
        component count is approximated as ``len(self.dataset.targets)``
        (the in-memory path also includes feature components, but the
        target count is the dominant factor for the final PredictionFrame
        size).
        """
        n_components = max(1, len(self.dataset.targets))
        n_cells = n_entities * n_time * n_components * max(1, num_samples)
        return n_cells > self.STREAMING_CELL_THRESHOLD

    def _predict_streaming(
        self,
        *,
        target_series: list,
        past_covariates: list | None,
        entity_ids: np.ndarray,
        pred_time_ids: np.ndarray,
        output_length: int,
        **predict_kwargs: Any,
    ) -> dict[str, Any]:
        """Run prediction in entity batches, writing each batch to a scaffold.

        Reuses :meth:`ViewsDataset.create_prediction_scaffold`,
        :meth:`ViewsDataset.write_prediction_batch`, and
        :meth:`ViewsDataset.to_predictionframe_per_target`. The inverse
        transform + clip path is shared with
        :meth:`ViewsDataset.ingest_numpy_predictions` (both delegate to
        :meth:`ViewsDataset._inverse_transform_numpy_predictions`), so the
        streaming and in-memory paths produce bit-identical frames.
        """
        import gc

        num_samples = predict_kwargs.get("num_samples", 1)
        mc_dropout = predict_kwargs.get("mc_dropout", False)
        batch_size = predict_kwargs.get("batch_size", None)
        verbose = predict_kwargs.get("verbose", True)

        target_names = list(self.dataset.targets)
        n_targets = len(target_names)
        n_entities = len(target_series)
        # Use half of batch_size for entity batching (or default to 500), minimum 1
        entity_batch_size = max(1, (batch_size // 2) if batch_size else (self.STREAMING_ENTITY_BATCH // 2))

        # Create the zarr-backed scaffold sized for the full grid.
        scaffold = ViewsDataset.create_prediction_scaffold(
            entity_ids=entity_ids,
            time_ids=pred_time_ids,
            targets=target_names,
            sample_size=int(num_samples),
            level=self._dataset_level_code(),
            time_id=self.dataset._time_id,
            entity_id=self.dataset._entity_id,
        )
        # Share the fitted scalers with the scaffold so
        # write_prediction_batch can apply the inverse transform.
        scaffold._target_scaler = getattr(self.dataset, "_target_scaler", None)
        scaffold._scalers_fitted = getattr(self.dataset, "_scalers_fitted", False)
        scaffold._log_targets = getattr(self.dataset, "_log_targets", False)

        logger.info(
            "Streaming predictions: %d entities × %d steps × %d samples "
            "(entity_batch=%d, model_batch=%s) → zarr scaffold",
            n_entities, output_length, num_samples, entity_batch_size, batch_size,
        )

        # Run predict_from_dataset in entity batches.
        for start in range(0, n_entities, entity_batch_size):
            end = min(start + entity_batch_size, n_entities)
            batch_series = target_series[start:end]
            batch_cov = (
                past_covariates[start:end] if past_covariates else None
            )
            batch_entity_ids = entity_ids[start:end]

            inference_dataset = self.model._build_inference_dataset(
                n=output_length,
                series=batch_series,
                past_covariates=batch_cov,
                future_covariates=None,
                stride=0,
                bounds=None,
            )
            del batch_series, batch_cov

            batch_preds, _, _ = self.model.predict_from_dataset(
                n=output_length,
                dataset=inference_dataset,
                batch_size=batch_size,
                verbose=verbose,
                num_samples=num_samples,
                mc_dropout=mc_dropout,
                values_only=True,
            )
            del inference_dataset

            # Audit each batch for numerical sanity before writing.
            if np.isnan(batch_preds).any() or np.isinf(batch_preds).any():
                raise NumericalSanityError(
                    f"NaN/Inf in streaming batch predictions "
                    f"(entities {start}:{end}, shape={batch_preds.shape})."
                )

            # Extract target components from model output (which includes features + targets).
            # Darts returns all components in the order they were in the input series.
            # The last n_targets components are the targets (standard layout: features before targets).
            n_components = batch_preds.shape[2]
            target_indices = list(range(n_components - n_targets, n_components))
            batch_target_preds = batch_preds[:, :, target_indices, :]

            scaffold.write_prediction_batch(
                target_values=batch_target_preds,
                entity_ids_batch=batch_entity_ids,
                time_ids=pred_time_ids,
                target_names=target_names,
                apply_inverse=True,
                clip_negatives=True,
            )
            del batch_preds, batch_target_preds
            gc.collect()
            logger.info(
                "Streaming batch %d/%d written (entities %d:%d).",
                start // entity_batch_size + 1,
                (n_entities + entity_batch_size - 1) // entity_batch_size,
                start, end,
            )

        return scaffold.to_predictionframe_per_target()

    def _dataset_level_code(self) -> str:
        """Return the VIEWS LOA code for the dataset's entity level.

        Maps ``country_id`` → ``"cm"`` and ``priogrid_id`` → ``"pgm"``.
        """
        eid = self.dataset._entity_id
        if eid == "priogrid_id":
            return "pgm"
        if eid == "country_id":
            return "cm"
        # Fallback: derive from the first letter of the entity id.
        return f"{eid[0]}m" if eid else "cm"

    def _ensure_model_on_device(self) -> None:
        """Ensure the model is on the configured device (restore from CPU drift).

        No-op for sklearn-based models (no torch parameters).
        """
        if self._is_sklearn_model:
            return
        current_device = next(self.model.model.parameters()).device
        if self.device != "cpu" and current_device.type == "cpu":
            logger.info("Restoring model to %s...", self.device)
            self._move_model_to_device()
            current_device = next(self.model.model.parameters()).device
            if current_device.type == "cpu":
                logger.warning(
                    "Failed to move model from CPU to %s; continuing on CPU.",
                    self.device,
                )

    # ------------------------------------------------------------------ persistence

    def save_model(self, path: str) -> None:
        """Save the Darts model + scaler state to disk.

        For sklearn-based models, ``model.save(path)`` uses pickle (via
        darts' SKLearnModel.save). For torch models, ``model.save(path)``
        uses torch's checkpoint format. The scaler sidecar (``path.scalers``)
        is always written via ``torch.save`` (pickle underneath) — works for
        both.
        """
        path = str(path)
        self.model.save(path=path)
        torch.save(
            {
                "target_scaler": getattr(self.dataset, "_target_scaler", None),
                "feature_scaler": getattr(self.dataset, "_feature_scaler", None),
                "scaler_fitted": self.scaler_fitted,
                "log_targets": self._log_targets,
                "log_features": self._log_features,
                "target_scaler_cfg": self._target_scaler_cfg,
                "feature_scaler_cfg": self._feature_scaler_cfg,
                "feature_scaler_map_cfg": self._feature_scaler_map_cfg,
            },
            path + ".scalers",
        )

    def load_model(self, path: str) -> None:
        """Load the Darts model + scaler state from disk.

        For sklearn-based models, ``model.__class__.load(path)`` uses pickle
        (via darts' SKLearnModel.load). The ``map_location`` kwarg is only
        passed for torch models — SKLearnModel.load does not accept it.
        """
        path = str(path)
        scaler_data = torch.load(
            path + ".scalers", map_location="cpu", weights_only=False
        )
        # Restore scaler state onto the dataset.
        self.dataset._target_scaler = scaler_data["target_scaler"]
        self.dataset._feature_scaler = scaler_data["feature_scaler"]
        self.dataset._scalers_fitted = scaler_data["scaler_fitted"]
        self.dataset._log_targets = scaler_data.get("log_targets", False)
        self.dataset._log_features = set(scaler_data.get("log_features", []))
        self.scaler_fitted = scaler_data["scaler_fitted"]

        # Validate scaler config matches.
        saved_cfg = scaler_data.get("target_scaler_cfg")
        if saved_cfg is not None and saved_cfg != self._target_scaler_cfg:
            raise ValueError(
                f"SCALER CONFIG MISMATCH: artifact has target_scaler="
                f"'{saved_cfg}' but config has '{self._target_scaler_cfg}'."
            )

        # Load the model. Torch models accept ``map_location``; sklearn
        # models (SKLearnModel.load) do not.
        if self._is_sklearn_model:
            self.model = self.model.__class__.load(path=path)
        else:
            self.model = self.model.__class__.load(
                path=path, map_location=str(self.device)
            )
        self._move_model_to_device()

        # Re-attach the dataset to sklearn models (the loaded model
        # instance does not carry the dataset reference — it was set on
        # the pre-load instance by __init__).
        if self._is_sklearn_model and hasattr(self.model, "set_dataset"):
            self.model.set_dataset(self.dataset, partition_dict=self._partition_dict)

        logger.info("Model loaded on device %s.", self.device)
