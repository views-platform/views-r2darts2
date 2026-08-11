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

        # Device management.
        self._ensure_model_on_device()

        # Predict.
        try:
            timeseries_pred = self.model.predict(
                n=output_length,
                series=target_series,
                past_covariates=past_covariates,
                verbose=True,
                **predict_kwargs,
            )
        except Exception as exc:
            logger.error("Error during prediction: %s", exc)
            raise

        # Audit model output for numerical sanity BEFORE inverse-transform.
        ReproducibilityGate.Data.audit_numerical_sanity(
            timeseries_pred, name="Model Predictions"
        )

        # Inverse-transform + convert to PredictionFrames via the dataset.
        frames = self.dataset.ingest_darts_predictions(
            timeseries_pred, apply_inverse=True, clip_negatives=True,
        )

        # Final NaN guard on the PredictionFrame values.
        import numpy as np
        for target, frame in frames.items():
            if np.isnan(frame.values).any():
                raise NumericalSanityError(
                    f"NaNs in final PredictionFrame for target '{target}'."
                )
        return frames

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
