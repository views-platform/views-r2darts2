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
from typing import Any, Mapping

import torch
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

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

        IMPORTANT: This method must NOT refit the scalers. The scalers were
        fitted in :meth:`train` on the full training partition. Refitting here
        would change the scaler parameters, causing a mismatch between the
        training data (transformed with the original scaler) and the
        validation/prediction data (transformed with the refit scaler).
        """
        icl = self.model.input_chunk_length
        ocl = self.model.output_chunk_length
        val_start = self._test_start - icl

        # For forecasting runs the entire test window is future data not yet in
        # the dataset. Cap val_end at train_end so the val window stays in
        # available data.
        max_dataset_time = int(
            self.dataset._ds[self.dataset._time_id].values.max()
        )
        if self._test_start > max_dataset_time:
            val_end = self._train_end
            logger.info(
                "Forecasting mode: test_start=%d > max_dataset_time=%d; "
                "validation will be carved from train end.",
                self._test_start, max_dataset_time,
            )
        else:
            val_end = self._test_end

        val_targets, val_past_cov = self.dataset.get_scaled_darts_timeseries(
            time_ids=list(range(val_start, val_end + 1)),
            use_cyclic_encoders=self._use_cyclic_encoders,
        )

        # Check if val is too short (forecasting mode).
        min_val_len = icl + ocl
        max_val_len = max((len(ts) for ts in val_targets), default=0)
        if max_val_len < min_val_len:
            # Carve from train end — NO scaler refit (preserves train-time scaler).
            carved_start = self._train_end - ocl - icl + 1
            logger.info(
                "Forecasting mode: val too short (%d < %d). Carving [%d, %d].",
                max_val_len, min_val_len, carved_start, self._train_end,
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
        import numpy as np

        # Lock entropy for reproducible probabilistic samples.
        ReproducibilityGate.Data.lock_entropy(self.random_state)

        # Get scaled input window for this sequence.
        icl = self.model.input_chunk_length
        start = self._test_start + sequence_number - icl
        end = self._test_start - 1 + sequence_number
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

        # Device management.
        self._ensure_model_on_device()

        # --- Fast prediction path: values_only=True ------------------------
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

        # Audit model output for numerical sanity (numpy-direct).
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
        num_samples = predict_kwargs.pop("num_samples", 1)
        mc_dropout = predict_kwargs.pop("mc_dropout", False)
        batch_size = predict_kwargs.pop("batch_size", None)
        verbose = predict_kwargs.pop("verbose", True)
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
