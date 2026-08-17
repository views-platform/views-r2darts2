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
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from views_frames import PredictionFrame

from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.infrastructure.device import get_device as _get_device
from views_r2darts2.infrastructure.exceptions import NumericalSanityError
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate
from views_r2darts2.transformers.frame_builder import (
    build_prediction_frames_from_dataset,
)

logger = logging.getLogger(__name__)


class _FramesWithCleanup(dict):
    """A dict subclass that carries a ``_frames_dir`` for cleanup.

    ``_predict_streaming`` returns this instead of a plain dict so the
    manager can find and clean up the memmap temp dir after converting
    the frames to a DataFrame.
    """

    def __init__(self, frames: dict, frames_dir: Path) -> None:
        super().__init__(frames)
        self._frames_dir = frames_dir


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
        model: TorchForecastingModel,
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
            model: A Darts :class:`TorchForecastingModel` instance.
            partition_dict: ``{"train": (start, end), "test": (start, end)}``.
            feature_scaler: Scaler name for all features.
            target_scaler: Scaler name for targets.
            log_targets: Apply ``log1p`` to targets before scaling.
            log_features: Feature names to apply ``log1p`` to.
            feature_scaler_map: Per-feature scaler map (takes precedence).
            random_state: Random seed (mandatory).
            static_covariate_stats: Static covariate config (unused in slim
                version — static covariates are computed by the dataset).
            checkpoint_mode: ``"best"`` or ``"last"``.
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
        logger.info("Using device: %s", self.device)
        self._move_model_to_device()

    # ------------------------------------------------------------------ device

    def _move_model_to_device(self) -> None:
        """Move the model to the configured device."""
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
        """
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
        """Build the validation set (test partition or carved from train end).

        In forecasting mode, the test partition contains future months that
        don't exist in the data yet — we carve the validation set from the
        train end instead. This is detected by checking whether the test
        partition's time_ids are present in the dataset.
        """
        icl = self.model.input_chunk_length
        ocl = self.model.output_chunk_length

        # Check if the test partition exists in the dataset. In forecasting
        # mode, the test months (e.g. 560-596) are future months that haven't
        # been observed yet — they're not in the data. Fetching them would
        # trigger zero-fill warnings and produce useless all-zero validation
        # series. Detect this upfront and carve from the train end.
        available_times = set(
            int(t) for t in self.dataset._ds[self.dataset._time_id].values
        )
        test_times = set(range(self._test_start, self._test_end + 1))
        test_coverage = len(test_times & available_times) / max(len(test_times), 1)

        if test_coverage < 0.5:
            # Forecasting mode: test partition mostly absent from data.
            # Carve validation from the train end immediately.
            carved_start = self._train_end - ocl - icl + 1
            logger.info(
                "Forecasting mode: test partition [%d, %d] is %d%% present in "
                "data. Carving validation from train end [%d, %d].",
                self._test_start, self._test_end, int(test_coverage * 100),
                carved_start, self._train_end,
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

        # Calibration/validation mode: test partition exists in the data.
        val_start = self._test_start - icl
        val_end = self._test_end
        val_targets, val_past_cov = self.dataset.get_scaled_darts_timeseries(
            time_ids=list(range(val_start, val_end + 1)),
            use_cyclic_encoders=self._use_cyclic_encoders,
        )

        # Check if val is too short (edge case for very short test partitions).
        min_val_len = icl + ocl
        max_val_len = max((len(ts) for ts in val_targets), default=0)
        if max_val_len < min_val_len:
            carved_start = self._train_end - ocl - icl + 1
            logger.info(
                "Val too short (%d < %d). Carving [%d, %d].",
                max_val_len, min_val_len, carved_start, self._train_end,
            )
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
        output :class:`TimeSeries` construction entirely. The model runs in
        entity batches; each batch's predictions are written to a
        :class:`DatasetBuilder` scaffold on disk, so peak memory is one
        batch — never the full ``(n_entities, n_time, n_targets, n_samples)``
        grid. This is critical for large entity counts (e.g. 259k PRIO-GRID
        cells) and/or probabilistic forecasts (e.g. 500 samples) where
        materializing the full predictions array causes OOM kills.

        Args:
            sequence_number: Rolling-origin sequence index (0 = first test step).
            output_length: Number of time steps to forecast.
            **predict_kwargs: Forwarded to ``model.predict_from_dataset``.

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

        # Get scaled input window for this sequence.
        icl = self.model.input_chunk_length

        # In forecasting mode, the test partition's first month (test_start)
        # may be the month AFTER the last observed month — i.e. test_start - 1
        # is not in the dataset. The partition convention from
        # views_pipeline_core sets train_end = test_start - 1, but the actual
        # data may end one month earlier (train_end - 1). When this happens,
        # we shift the prediction window back by one so we don't skip a month.
        # Example: data ends at 558, partition is train=(121,559), test=(560,596).
        # Without the shift: input ends at 559 (zero-filled, wrong), predict 560+.
        # With the shift: input ends at 558 (real data), predict 559+.
        available_times = set(
            int(t) for t in self.dataset._ds[self.dataset._time_id].values
        )
        effective_test_start = self._test_start
        if (self._test_start - 1) not in available_times:
            # The month before test_start is missing — shift back by one.
            effective_test_start = self._test_start - 1
            logger.info(
                "Forecasting mode: test_start %d shifted to %d (month %d not "
                "in data, predicting from the first missing month).",
                self._test_start, effective_test_start, self._test_start - 1,
            )

        start = effective_test_start + sequence_number - icl
        end = effective_test_start - 1 + sequence_number
        target_series, past_covariates = self.dataset.get_scaled_darts_timeseries(
            time_ids=list(range(start, end + 1)),
            use_cyclic_encoders=self._use_cyclic_encoders,
        )

        # Capture the entity/time index before freeing the input series.
        entity_ids = np.array([
            int(ts.static_covariates[self.dataset._entity_id].iloc[0])
            for ts in target_series
        ], dtype=np.int64)
        pred_time_start = int(target_series[0].time_index[-1]) + 1
        pred_time_ids = np.arange(
            pred_time_start, pred_time_start + output_length, dtype=np.int64
        )

        # Device management.
        self._ensure_model_on_device()

        # --- Streaming prediction path via DatasetBuilder ---------------
        # The builder pre-allocates a NaN-filled Zarr skeleton (metadata
        # only — nothing in RAM) and scatter-writes each batch to disk.
        # Peak memory is one batch, never the grid.
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

        # No final NaN guard here — checking would materialize the full
        # memmap. Each batch was already NaN-audited in _predict_streaming
        # before writing to the scaffold.
        return frames

    #: Default entity batch size for the streaming path. Each batch runs
    #: one ``predict_from_dataset`` call. When the model's ``batch_size``
    #: is available, the actual entity batch size is set to
    #: ``batch_size // 2`` (half the torch batch size). This value is the
    #: fallback when ``batch_size`` is not set.
    STREAMING_ENTITY_BATCH: int = 1000

    def _predict_streaming(
        self,
        *,
        target_series: list,
        past_covariates: list | None,
        entity_ids: np.ndarray,
        pred_time_ids: np.ndarray,
        output_length: int,
        **predict_kwargs: Any,
    ) -> dict[str, PredictionFrame]:
        """Run prediction in entity batches, writing each batch to a builder.

        Uses :meth:`ViewsDataset.builder` to scaffold a disk-backed Zarr
        store, then runs ``predict_from_dataset`` per entity batch and
        writes the batch to the scaffold via ``write_batch``. The scaffold
        is converted to ``{target: PredictionFrame}`` at the end.
        """
        import gc

        num_samples = predict_kwargs.get("num_samples", 1)
        mc_dropout = predict_kwargs.get("mc_dropout", False)
        batch_size = predict_kwargs.get("batch_size", None)
        verbose = predict_kwargs.get("verbose", True)

        target_names = list(self.dataset.targets)
        n_entities = len(target_series)
        # Entity batch size = torch batch size (when available),
        # falling back to STREAMING_ENTITY_BATCH.
        if batch_size is not None and batch_size > 1:
            entity_batch_size = max(1, batch_size)
        else:
            entity_batch_size = self.STREAMING_ENTITY_BATCH

        # Determine the LOA code and variable specs for the builder.
        loa = self._dataset_level_code()
        pred_vars = {f"pred_{t}": "num3" for t in target_names}

        logger.info(
            "Streaming predictions: %d entities × %d steps × %d samples "
            "(batch=%d) → builder scaffold",
            n_entities, output_length, num_samples, entity_batch_size,
        )

        # Create the builder scaffold (disk-backed Zarr, metadata only).
        # Use the config's scratch_dir if set, else default to /tmp.
        # Use larger chunks aligned to the entity batch size to reduce
        # file count (prevents inode exhaustion on large grids).
        scratch_dir = predict_kwargs.get("scratch_dir")
        chunks = (output_length, max(1, entity_batch_size), int(num_samples))
        with ViewsDataset.builder(
            loa=loa,
            times=pred_time_ids,
            entities=entity_ids,
            variables=pred_vars,
            sample_size=int(num_samples),
            targets=pred_vars.keys(),
            base_dir=scratch_dir,
            chunks=chunks,
        ) as b:
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

                # Extract target components and write to the builder.
                # batch_preds shape: (n_batch, n_time, n_components, n_samples)
                n_batch = batch_preds.shape[0]
                n_time = batch_preds.shape[1]
                n_targets = len(target_names)
                # Assume the last n_targets components are the targets.
                target_indices = list(range(
                    batch_preds.shape[2] - n_targets, batch_preds.shape[2]
                ))

                # Apply inverse transforms + clip (numpy-direct).
                target_values = batch_preds[:, :, target_indices, :]
                if self.dataset.scalers_fitted:
                    target_values = self.dataset._inverse_transform_numpy_predictions(
                        target_values
                    )
                np.maximum(target_values, 0.0, out=target_values)

                # Write per-target to the builder. Each target is a separate
                # column. The builder expects (N, S) per (time, entity) row.
                # We expand the batch into (time * entity) rows per target.
                # time-major: time varies slowest, entity varies fastest.
                t_grid, e_grid = np.meshgrid(
                    pred_time_ids, batch_entity_ids, indexing="ij"
                )
                time_flat = t_grid.ravel()      # (n_time * n_batch,)
                entity_flat = e_grid.ravel()    # (n_time * n_batch,)

                for t_idx, target_name in enumerate(target_names):
                    pred_var = f"pred_{target_name}"
                    # Extract this target: (n_batch, n_time, n_samples).
                    vals = target_values[:, :, t_idx, :]
                    # Transpose to (n_time, n_batch, n_samples) then reshape
                    # to (n_time * n_batch, n_samples) — time-major.
                    vals = vals.transpose(1, 0, 2).reshape(
                        n_time * n_batch, -1
                    ).astype(np.float32)
                    b.write_batch(
                        times=time_flat,
                        entities=entity_flat,
                        columns={pred_var: vals},
                    )

                del batch_preds, target_values
                gc.collect()
                logger.info(
                    "Streaming batch %d/%d written (entities %d:%d).",
                    start // entity_batch_size + 1,
                    (n_entities + entity_batch_size - 1) // entity_batch_size,
                    start, end,
                )

            ds = b.build()

        # Stream the zarr-backed dataset into memmap-backed PredictionFrames.
        # This writes a row-major (N, S) float32 values.npy per target in
        # entity-aligned blocks, then wraps each in a PredictionFrame whose
        # values is a read-only np.memmap. Peak memory is one entity block
        # — never the full (n_entities, n_time, n_samples) grid.
        import tempfile

        # Use the config's scratch_dir if set, else default to /tmp.
        scratch_dir = predict_kwargs.get("scratch_dir")
        frames_dir = Path(tempfile.mkdtemp(
            prefix="pred_frames_",
            dir=scratch_dir,
        ))
        frames = build_prediction_frames_from_dataset(
            ds,
            target_names,
            frames_dir,
            entity_block=max(1, entity_batch_size),
        )
        # Tag the dict with the temp dir so the manager can clean it up
        # after converting to a DataFrame. We use a wrapper class since
        # plain dicts don't allow attribute assignment.
        frames = _FramesWithCleanup(frames, frames_dir)
        ds.close()
        return frames

    def _dataset_level_code(self) -> str:
        """Return the VIEWS LOA code for the dataset's entity level."""
        eid = self.dataset._entity_id
        if eid == "priogrid_id":
            return "pgm"
        if eid == "country_id":
            return "cm"
        return f"{eid[0]}m" if eid else "cm"

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

    def _ensure_model_on_device(self) -> None:
        """Ensure the model is on the configured device (restore from CPU drift)."""
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
        """Save the Darts model + scaler state to disk."""
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
        """Load the Darts model + scaler state from disk."""
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

        # Load the model.
        self.model = self.model.__class__.load(
            path=path, map_location=str(self.device)
        )
        self._move_model_to_device()
        logger.info("Model loaded on device %s.", self.device)
