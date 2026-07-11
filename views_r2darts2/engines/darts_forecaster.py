"""Darts forecaster engine — pandas-free, PredictionFrame-output.

Wraps a Darts :class:`TorchForecastingModel` with the preprocessing pipeline
(scalers, log transforms) needed for VIEWS conflict-fatality forecasting. The
stateful coupling between model and scalers is the class's reason for existing:
predictions are only meaningful once inverse-transformed back to raw space.

Data flow:

    FeatureFrame (memmap-backed)
        ↓  ``ViewsDatasetDarts.as_darts_timeseries``  (Darts boundary)
    List[TimeSeries]  (per-entity, with static covariates)
        ↓  ``_preprocess_timeseries`` (slice + log + scale + audit)
    List[TimeSeries] (scaled targets) + List[TimeSeries] (scaled past_cov)
        ↓  ``model.fit`` / ``model.predict``
    List[TimeSeries] (predictions, scaled space)
        ↓  ``_inverse_transform_target_scaler`` + ``_inverse_log_on_predictions``
    List[TimeSeries] (predictions, raw space)
        ↓  ``prediction_frames_from_darts`` (Darts boundary, inverse direction)
    dict[str, PredictionFrame]  (one per target column)

The class is pandas-free on its surface. The two Darts-boundary helpers in
:mod:`views_r2darts2.transformers.darts_bridge` are the only places pandas is
imported, and they are confined to that module.

Google Python Style.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

import numpy as np
import torch
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from views_frames import PredictionFrame

from views_r2darts2.data.views_dataset import ViewsDatasetDarts
from views_r2darts2.infrastructure.device import get_device as _get_device
from views_r2darts2.infrastructure.exceptions import NumericalSanityError
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate
from views_r2darts2.transformers.darts_bridge import prediction_frames_from_darts
from views_r2darts2.transformers.feature_scaler_manager import FeatureScalerManager
from views_r2darts2.transformers.inverse import (
    extract_fitted_sklearn_scaler,
    inverse_transform_probabilistic_subset,
)
from views_r2darts2.transformers.scaler_selector import ScalerSelector

logger = logging.getLogger(__name__)


class DartsForecaster:
    """Stateful wrapper coupling a Darts model with its preprocessing pipeline.

    Intent Contract:
        - Purpose: Maintain the stateful coupling between a deep learning model
          and its required preprocessing pipeline (scalers, log-transforms) so
          that predictions are on the correct scale.
        - Non-Goals: Does not manage Weights & Biases logging or experiment
          orchestration (that's :class:`DartsForecastingModelManager`).
        - Guarantees:
            - Data is downcast to float32 before entering the model (ADR-010).
            - Target scalers are fitted ONLY on training data and correctly
              inverse-applied during prediction (preserving sample dimensions
              for probabilistic forecasts).
            - Physical boundaries are respected during preprocessing via
              :class:`ReproducibilityGate`.
        - Failure Behavior: Raises ``RuntimeError`` if prediction is attempted
          before scalers are fitted or if numerical insanity is detected.
    """

    def __init__(
        self,
        dataset: ViewsDatasetDarts,
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
            dataset: The :class:`ViewsDatasetDarts` carrying the FeatureFrame.
            model: A Darts :class:`TorchForecastingModel` instance.
            partition_dict: ``{"train": (start, end), "test": (start, end)}``.
            feature_scaler: Name of the feature scaler for all features.
                Ignored when ``feature_scaler_map`` is provided.
            target_scaler: Name of the target scaler.
            log_targets: When ``True``, apply ``log1p`` to targets before
                scaling and ``expm1`` to inverse-transformed predictions.
            log_features: Feature names to apply ``log1p`` to (no inverse —
                past covariates are not reconstructed post-prediction).
            feature_scaler_map: Per-feature scaler map (takes precedence over
                ``feature_scaler``). See :class:`FeatureScalerManager`.
            random_state: Random seed for reproducibility. Mandatory.
            static_covariate_stats: Optional config for per-entity static
                covariate stats. Keys: ``transform`` (str|None), ``stats``
                (list[str]|None), ``inject`` (bool).
            checkpoint_mode: ``"best"`` (default) or ``"last"``.
            use_cyclic_encoders: When ``True``, append sin/cos cyclic time
                encoders to the feature axis.

        Raises:
            ValueError: ``random_state`` is ``None``, ``checkpoint_mode`` is
                invalid, or both ``log_features`` and
                ``feature_scaler='LogTransform'`` are set.
        """
        self.dataset = dataset
        self.model = model
        self._train_start, self._train_end = partition_dict["train"]
        self._test_start, self._test_end = partition_dict["test"]

        if random_state is None:
            raise ValueError(
                "MANDATORY PARAMETER MISSING: random_state must be provided "
                "to DartsForecaster."
            )
        self.random_state = random_state

        # Static covariate stats configuration.
        self._static_cov_transform = (
            static_covariate_stats.get("transform")
            if static_covariate_stats
            else None
        )
        self._static_cov_stats = (
            static_covariate_stats.get("stats") if static_covariate_stats else None
        )
        self._static_cov_inject = (
            static_covariate_stats.get("inject", False)
            if static_covariate_stats
            else False
        )
        logger.info(
            "static_covariate_stats: transform=%r stats=%r inject=%r",
            self._static_cov_transform,
            self._static_cov_stats,
            self._static_cov_inject,
        )

        if checkpoint_mode not in ("best", "last"):
            raise ValueError(
                f"checkpoint_mode must be 'best' or 'last', got {checkpoint_mode!r}"
            )
        self._checkpoint_mode = checkpoint_mode
        self._use_cyclic_encoders = use_cyclic_encoders

        self._feature_scaler_cfg = feature_scaler
        self._target_scaler_cfg = target_scaler
        self._feature_scaler_map_cfg = (
            dict(feature_scaler_map) if feature_scaler_map else None
        )
        self._log_targets = bool(log_targets)
        self._log_features = set(log_features or [])

        # Warn about double log transform on targets.
        if self._log_targets and target_scaler == "LogTransform":
            logger.warning(
                "Both log_targets=True and target_scaler='LogTransform' are "
                "set. This would apply log transform twice. Disabling "
                "log_targets to avoid double transformation."
            )
            self._log_targets = False
        # Raise on double log transform on features (asymmetric with the target
        # warning, but the legacy contract raised here — preserve it).
        if self._log_features and feature_scaler == "LogTransform":
            raise ValueError(
                "Both log_features and feature_scaler='LogTransform' are set. "
                "This would apply log transform twice on overlapping features. "
                "Use only one transformation method."
            )

        self.scaler_fitted = False

        # Initialize target scaler.
        self.target_scaler = self._instantiate_scaler(self._target_scaler_cfg)

        # Initialize feature scaler(s).
        if not self.dataset.features:
            if self._feature_scaler_cfg or self._feature_scaler_map_cfg:
                logger.info(
                    "Dataset has no feature columns — disabling feature_scaler."
                )
            self.feature_scaler = None
        elif self._feature_scaler_map_cfg:
            self.feature_scaler = FeatureScalerManager(
                feature_scaler_map=self._feature_scaler_map_cfg,
                default_scaler=self._feature_scaler_cfg,
                all_features=self.dataset.features,
            )
            logger.info("Using feature scaler map: %s", self.feature_scaler)
        else:
            self.feature_scaler = self._instantiate_scaler(self._feature_scaler_cfg)
            logger.info(
                "Using feature scaler: %s", self._feature_scaler_cfg
            )
        logger.info("Using target scaler: %s", self._target_scaler_cfg)

        self.device = self.get_device()
        logger.info("Using device: %s", self.device)
        if hasattr(self.model, "to_device"):
            self.model.to_device(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(self.device)

    # ------------------------------------------------------------------ scalers

    @staticmethod
    def _instantiate_scaler(scaler_cfg: Any) -> Scaler | Pipeline | None:
        """Delegate to :meth:`ScalerSelector.instantiate_darts_scaler`."""
        if scaler_cfg is None:
            return None
        return ScalerSelector.instantiate_darts_scaler(scaler_cfg)

    # ------------------------------------------------------------------ log transforms

    def _apply_log_to_targets(
        self, series_list: list[TimeSeries]
    ) -> list[TimeSeries]:
        """Vectorized ``log1p`` for target series (clip negatives first)."""
        if not self._log_targets:
            return series_list
        logger.info("Applying vectorized log1p transform to target series...")
        return [
            ts.map(lambda arr: np.log1p(np.maximum(arr, 0)).astype(np.float32))
            for ts in series_list
        ]

    def _inverse_log_on_predictions(
        self, series_list: list[TimeSeries]
    ) -> list[TimeSeries]:
        """Inverse of :meth:`_apply_log_to_targets`: ``expm1`` (clip negatives)."""
        if not self._log_targets:
            return series_list
        logger.info(
            "Applying vectorized expm1 inverse transform to predicted series..."
        )
        return [
            ts.map(lambda arr: np.expm1(np.maximum(arr, 0)).astype(np.float32))
            for ts in series_list
        ]

    def _apply_log_to_feature_series(self, ts: TimeSeries) -> TimeSeries:
        """Apply ``log1p`` to selected feature components in a single series."""
        if not self._log_features:
            return ts
        comps = ts.components
        if not any(c in self._log_features for c in comps):
            return ts
        arr = ts.all_values(copy=True)
        if arr.ndim == 2:
            for idx, name in enumerate(comps):
                if name in self._log_features:
                    arr[:, idx] = np.log1p(np.maximum(arr[:, idx], 0.0))
        elif arr.ndim == 3:
            for idx, name in enumerate(comps):
                if name in self._log_features:
                    arr[:, idx, :] = np.log1p(np.maximum(arr[:, idx, :], 0.0))
        return TimeSeries.from_times_and_values(
            times=ts.time_index,
            values=arr.astype(np.float32),
            columns=comps,
            freq=ts.freq,
            static_covariates=ts.static_covariates,
        )

    def _apply_log_to_features(
        self, series_list: list[TimeSeries]
    ) -> list[TimeSeries]:
        """Batch wrapper for :meth:`_apply_log_to_feature_series`."""
        if not self._log_features:
            return series_list
        logger.info(
            "Applying log1p transform to selected feature components: %s",
            sorted(self._log_features),
        )
        return [self._apply_log_to_feature_series(ts) for ts in series_list]

    # ------------------------------------------------------------------ inverse target scaler

    def _inverse_transform_target_scaler(
        self, timeseries_pred: list[TimeSeries]
    ) -> list[TimeSeries]:
        """Inverse-transform predictions, preserving samples for probabilistic forecasts.

        For Darts :class:`Pipeline` objects (chained scalers),
        ``Pipeline.inverse_transform`` handles probabilistic series natively.
        For single Darts :class:`Scaler` objects wrapping sklearn scalers, we
        manually reshape to 2-D, inverse-transform, and reshape back to preserve
        the sample dimension.
        """
        if not self.target_scaler or not self.scaler_fitted:
            return timeseries_pred

        if isinstance(self.target_scaler, Pipeline):
            return self.target_scaler.inverse_transform(timeseries_pred)

        result: list[TimeSeries] = []
        for ts in timeseries_pred:
            arr = ts.all_values(copy=True)
            is_probabilistic = arr.ndim == 3
            if is_probabilistic:
                inv_values = inverse_transform_probabilistic_subset(
                    subset_3d=arr.astype(np.float32),
                    scaler=self.target_scaler,
                )
                new_ts = TimeSeries.from_times_and_values(
                    times=ts.time_index,
                    values=inv_values.astype(np.float32),
                    columns=ts.components,
                    freq=ts.freq,
                    static_covariates=ts.static_covariates,
                )
            else:
                # Deterministic path — use Darts' own inverse_transform.
                new_ts = self.target_scaler.inverse_transform([ts])[0]
            result.append(new_ts)
        return result

    # ------------------------------------------------------------------ device

    @staticmethod
    def get_device() -> str:
        """Return the device type for model training (``mps``/``cuda``/``cpu``).

        Delegates to :func:`views_r2darts2.infrastructure.device.get_device` to
        avoid a circular import with :class:`ModelCatalog`.
        """
        return _get_device()

    # ------------------------------------------------------------------ preprocessing

    def _preprocess_timeseries(
        self,
        timeseries: list[TimeSeries],
        start: int,
        end: int,
        train_mode: bool = False,
    ) -> tuple[list[TimeSeries], list[TimeSeries] | None]:
        """Slice, log-transform, and scale the per-entity TimeSeries list.

        Args:
            timeseries: Per-entity TimeSeries collection (from
                :meth:`ViewsDatasetDarts.as_darts_timeseries`).
            start: Start time id (inclusive) for the slice.
            end: End time id (inclusive) for the slice — Darts ``slice`` is
                exclusive for integer indices, so we pass ``end + 1``.
            train_mode: When ``True``, fits scalers and enforces the temporal
                firewall gates (boundary integrity, sequence contiguity).

        Returns:
            ``(targets, past_covariates)``. ``past_covariates`` is ``None`` for
            univariate models (``dataset.features == []``).
        """
        timeseries_float = [s.astype(np.float32) for s in timeseries]
        min_length = self.model.input_chunk_length + self.model.output_chunk_length

        # Slice targets (end + 1 because Darts slice is exclusive for integer indices).
        if train_mode:
            # Build aligned (target, past_cov) pairs and filter together. The
            # paired filter is critical: ``model.fit(series=targets,
            # past_covariates=past_cov)`` pairs by list index, so any
            # entity-level mismatch silently trains the model on the wrong
            # covariate history.
            #
            # The filter has TWO conditions:
            #   1. ``len(sliced) >= min_length`` — enough rows for the
            #      input/output chunk window.
            #   2. ``sliced.time_index.max() == end`` — the series extends to
            #      the training boundary. Entities that stop earlier (e.g., a
            #      country that dissolved at month 379) are filtered out
            #      rather than triggering a hard DataStarvationError in the
            #      boundary audit below. This is the pragmatic choice: you
            #      can't train on data that doesn't exist, and excluding a
            #      handful of short-tail entities is preferable to aborting
            #      the entire training run.
            paired = []
            skipped_short = 0
            skipped_boundary = 0
            for s in timeseries_float:
                sliced = s.slice(start_ts=start, end_ts=end + 1)
                if len(sliced) < min_length:
                    skipped_short += 1
                    continue
                if int(sliced.time_index.max()) != end:
                    skipped_boundary += 1
                    continue
                paired.append(
                    (
                        sliced[self.dataset.targets],
                        sliced[self.dataset.features].astype(np.float32)
                        if self.dataset.features
                        else None,
                    )
                )
            targets = [p[0] for p in paired]
            past_cov = [p[1] for p in paired] if self.dataset.features else None
            logger.info(
                "Training filter: %d/%d entities passed (min_length >= %d AND "
                "extends to end=%d). Skipped: %d too-short, %d boundary-miss.",
                len(paired),
                len(timeseries_float),
                min_length,
                end,
                skipped_short,
                skipped_boundary,
            )
        else:
            # Prediction path: filter out entities whose slice is EMPTY.
            # Some entities in the validation parquet end before the val window
            # starts (e.g., entity 59 ends at month 379, but the val window is
            # [481, 553]). Slicing such an entity to the val window produces a
            # zero-length TimeSeries, which then crashes the scaler's
            # ``transform`` with ``Found array with 0 sample(s)``. Filter them
            # out here — the model cannot predict for an entity with no input
            # context anyway.
            #
            # Unlike train_mode, we do NOT enforce a min_length or boundary
            # check here: prediction must include every entity that has ANY
            # data in the window, even if its history is shorter than the
            # ideal input chunk length (Darts will pad/truncate as needed).
            targets = []
            past_cov = []
            skipped_empty = 0
            for s in timeseries_float:
                sliced_target = s.slice(start_ts=start, end_ts=end + 1)[
                    self.dataset.targets
                ]
                if len(sliced_target) == 0:
                    skipped_empty += 1
                    continue
                targets.append(sliced_target)
                if self.dataset.features:
                    sliced_cov = s.slice(start_ts=start, end_ts=end + 1)[
                        self.dataset.features
                    ].astype(np.float32)
                    past_cov.append(sliced_cov)
            if skipped_empty > 0:
                logger.info(
                    "Prediction filter: %d/%d entities had empty slices for "
                    "window [%d, %d] and were skipped.",
                    skipped_empty,
                    len(timeseries_float),
                    start,
                    end,
                )
            past_cov = past_cov if self.dataset.features else None

        # Apply log transforms.
        if not train_mode and past_cov is not None:
            past_cov = self._apply_log_to_features(past_cov)
        if train_mode and past_cov is not None:
            past_cov = self._apply_log_to_features(past_cov)

        targets = self._apply_log_to_targets(targets)

        if train_mode:
            # GATE 3, 4, 5: the Fortress Firewall.
            ReproducibilityGate.Temporal.audit_boundary_integrity(targets, end)
            for ts in targets:
                ReproducibilityGate.Temporal.audit_sequence_contiguity(
                    ts.time_index.values.astype(int)
                )

            logger.info("Fitting scalers for training data...")
            if self.target_scaler:
                targets = self.target_scaler.fit_transform(targets)
            if self.feature_scaler:
                past_cov = self.feature_scaler.fit_transform(past_cov)
            self.scaler_fitted = True
        else:
            logger.info("Transforming scalers for prediction data...")
            if self.target_scaler and self.scaler_fitted:
                targets = self.target_scaler.transform(targets)
            if self.feature_scaler and self.scaler_fitted:
                past_cov = self.feature_scaler.transform(past_cov)

        # Downcast after scaler/log (they yield float64).
        targets = [ts.astype(np.float32) for ts in targets]
        if past_cov is not None:
            past_cov = [pc.astype(np.float32) for pc in past_cov]

        ReproducibilityGate.Data.audit_numerical_sanity(targets, "targets")
        if past_cov is not None:
            ReproducibilityGate.Data.audit_numerical_sanity(
                past_cov, "past_covariates"
            )

        return targets, past_cov

    # ------------------------------------------------------------------ train

    def train(self) -> None:
        """Train the model on the train partition with a carved validation set.

        Preprocesses training data, fits scalers, then prepares a validation
        set from the test partition (transformed with train-fitted scalers —
        no leakage). For forecasting runs where the test partition is too
        short to form a validation window, carves the last ``output_chunk_length``
        steps from the training window as a holdout val set and refits scalers
        on the trimmed window to prevent holdout-target leakage into scaler
        statistics.
        """
        timeseries = self.dataset.as_darts_timeseries(
            stat_time_range=(self._train_start, self._train_end),
            static_cov_transform=self._static_cov_transform,
            static_cov_stats=self._static_cov_stats,
            inject_static_covariates=self._static_cov_inject,
            use_cyclic_encoders=self._use_cyclic_encoders,
        )

        target_series, past_covariates = self._preprocess_timeseries(
            timeseries=timeseries,
            start=self._train_start,
            end=self._train_end,
            train_mode=True,
        )
        target_series = [ts.astype(np.float32) for ts in target_series]
        if self.dataset.features:
            past_covariates = [
                pc.astype(np.float32) if pc is not None else None
                for pc in past_covariates
            ]

        # Validation set: test partition, transformed with train-fitted scalers.
        val_start = self._test_start - self.model.input_chunk_length
        val_end = self._test_end
        val_targets, val_past_cov = self._preprocess_timeseries(
            timeseries=timeseries,
            start=val_start,
            end=val_end,
            train_mode=False,
        )
        val_targets = [ts.astype(np.float32) for ts in val_targets]
        if self.dataset.features:
            val_past_cov = [
                pc.astype(np.float32) if pc is not None else None
                for pc in val_past_cov
            ]

        # Forecasting-mode carve: when the test partition has no ground-truth
        # output steps (run_type="forecasting"), val series are too short for
        # Darts to build even one sample. Carve the last ocl steps from the
        # training window as holdout val and refit scalers on the trimmed
        # window to prevent holdout leakage.
        min_val_len = (
            self.model.input_chunk_length + self.model.output_chunk_length
        )
        max_val_len = max((len(ts) for ts in val_targets), default=0)
        used_carved_val = max_val_len < min_val_len
        if used_carved_val:
            ocl = self.model.output_chunk_length
            icl = self.model.input_chunk_length
            trimmed_train_end = self._train_end - ocl
            carved_val_start = self._train_end - ocl - icl + 1
            logger.info(
                "Forecasting mode: val partition too short "
                "(%d < %d). Carving holdout val [%d, %d] (%d steps). "
                "Refitting scalers on trimmed train [%d, %d].",
                max_val_len,
                min_val_len,
                carved_val_start,
                self._train_end,
                icl + ocl,
                self._train_start,
                trimmed_train_end,
            )
            target_series, past_covariates = self._preprocess_timeseries(
                timeseries=timeseries,
                start=self._train_start,
                end=trimmed_train_end,
                train_mode=True,
            )
            target_series = [ts.astype(np.float32) for ts in target_series]
            if self.dataset.features:
                past_covariates = [
                    pc.astype(np.float32) if pc is not None else None
                    for pc in past_covariates
                ]
            val_targets, val_past_cov = self._preprocess_timeseries(
                timeseries=timeseries,
                start=carved_val_start,
                end=self._train_end,
                train_mode=False,
            )
            val_targets = [ts.astype(np.float32) for ts in val_targets]
            if self.dataset.features:
                val_past_cov = [
                    pc.astype(np.float32) if pc is not None else None
                    for pc in val_past_cov
                ]

        log_val_start = carved_val_start if used_carved_val else val_start
        log_val_end = self._train_end if used_carved_val else val_end
        logger.info(
            "Validation set: %d entities, range [%d, %d] (%s with %d steps of context).",
            len(val_targets) if val_targets is not None else 0,
            log_val_start,
            log_val_end,
            "carved from train end" if used_carved_val else "test partition",
            self.model.input_chunk_length,
        )

        # Auto-detect num_workers: half of available CPUs, capped at 8.
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

        # checkpoint_mode='last' overrides Darts' default best-val-loss reload.
        if self._checkpoint_mode == "last":
            try:
                self.model.load_weights_from_checkpoint(best=False)
                logger.info("checkpoint_mode='last': reloaded final epoch weights.")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "checkpoint_mode='last': failed to reload last checkpoint "
                    "(%s). Keeping best val_loss checkpoint.",
                    exc,
                )

    # ------------------------------------------------------------------ predict

    def predict(
        self,
        sequence_number: int,
        output_length: int = 36,
        **predict_kwargs: Any,
    ) -> dict[str, PredictionFrame]:
        """Generate forecasts and return them as a per-target dict of frames.

        Args:
            sequence_number: The index in the test set to start forecasting
                from (0 = first test step).
            output_length: Number of time steps to forecast (default 36).
            **predict_kwargs: Forwarded to ``model.predict`` (e.g.
                ``num_samples``, ``mc_dropout``).

        Returns:
            A ``{target_name: PredictionFrame}`` mapping. Each frame has a
            :class:`SpatioTemporalIndex` of ``(time, entity)`` pairs covering
            all entities × all forecast time steps. The frame's value array
            is ``(N, S)`` where ``S`` is the sample count (1 for deterministic,
            ``num_samples`` for probabilistic). Negative predictions are
            clipped to 0 (physical floor for fatality counts).

        Raises:
            RuntimeError: Scalers are not fitted (call :meth:`train` or
                :meth:`load_model` first).
            NumericalSanityError: NaNs or Infs leaked through the inverse
                pipeline into the final predictions.
        """
        if self.target_scaler and not self.scaler_fitted:
            raise RuntimeError(
                "predict() called before scalers were fitted. "
                "Call train() or load_model() first."
            )

        # LOCK ENTROPY: guarantee bit-perfect identity for probabilistic samples.
        ReproducibilityGate.Data.lock_entropy(self.random_state)

        logger.info(
            "predict() scaler state: target_scaler=%r (fitted=%s), "
            "feature_scaler=%r (fitted=%s), scaler_fitted=%s.",
            self._target_scaler_cfg,
            self.target_scaler is not None,
            self._feature_scaler_cfg,
            self.feature_scaler is not None,
            self.scaler_fitted,
        )

        timeseries = self.dataset.as_darts_timeseries(
            stat_time_range=(self._train_start, self._train_end),
            static_cov_transform=self._static_cov_transform,
            static_cov_stats=self._static_cov_stats,
            inject_static_covariates=self._static_cov_inject,
            use_cyclic_encoders=self._use_cyclic_encoders,
        )

        # Slice the input window for forecasting based on sequence_number.
        target_series, past_covariates = self._preprocess_timeseries(
            timeseries=timeseries,
            start=self._test_start + sequence_number - self.model.input_chunk_length,
            end=self._test_start - 1 + sequence_number,
        )

        # Resilient device management: Darts models can drift to CPU in
        # teardown(); restore them if needed before prediction.
        current_device = next(self.model.model.parameters()).device
        if self.device != "cpu" and current_device.type == "cpu":
            logger.info("Restoring model to %s before prediction...", self.device)
            if hasattr(self.model, "to_device"):
                self.model.to_device(self.device)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
                self.model.model.to(self.device)
            current_device = next(self.model.model.parameters()).device
            if current_device.type == "cpu":
                raise RuntimeError(
                    f"CRITICAL DEVICE FAILURE: failed to move model from CPU "
                    f"to {self.device}. Prediction aborted to prevent "
                    "inconsistent results."
                )

        # Generate forecasts.
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

        # Sample-preserving inverse transform.
        if self.target_scaler:
            timeseries_pred = self._inverse_transform_target_scaler(timeseries_pred)
        timeseries_pred = self._inverse_log_on_predictions(timeseries_pred)

        # Audit again after inverse-transform.
        ReproducibilityGate.Data.audit_numerical_sanity(
            timeseries_pred, name="Inverse-Transformed Predictions"
        )

        # Convert Darts predictions → {target: PredictionFrame}. Negative
        # predictions are clipped to 0 inside the bridge (physical floor).
        predictions = prediction_frames_from_darts(
            predictions=timeseries_pred,
            entity_id_name=self.dataset.entity_id,
            target_columns=self.dataset.targets,
            level=self.dataset.level,
            clip_negatives=True,
        )

        # Final NaN guard on the frame values (one of the per-target frames
        # might have picked up a NaN that the per-series audit missed).
        for target, frame in predictions.items():
            if np.isnan(frame.values).any():
                raise NumericalSanityError(
                    f"Numerical Sanity Violation: NaNs detected in final "
                    f"PredictionFrame for target '{target}'."
                )

        return predictions

    # ------------------------------------------------------------------ persistence

    def save_model(self, path: str) -> None:
        """Save the Darts model and the scaler state to disk.

        Writes two files:
            * ``path`` — the Darts model artifact (via ``model.save``).
            * ``path + ".scalers"`` — a torch-saved dict with the scalers,
              log-transform flags, and scaler configs.

        Args:
            path: Base path for the model artifact. The scaler state is
                written to ``path + ".scalers"``.
        """
        path = str(path)
        self.model.save(path=path)
        scaler_path = path + ".scalers"
        using_feature_scaler_map = isinstance(self.feature_scaler, FeatureScalerManager)
        torch.save(
            {
                "target_scaler": self.target_scaler,
                "feature_scaler": self.feature_scaler,
                "scaler_fitted": self.scaler_fitted,
                "log_targets": self._log_targets,
                "log_features": list(self._log_features),
                "using_feature_scaler_map": using_feature_scaler_map,
                "feature_scaler_map_cfg": self._feature_scaler_map_cfg,
                "feature_scaler_cfg": self._feature_scaler_cfg,
                "target_scaler_cfg": self._target_scaler_cfg,
            },
            scaler_path,
        )

    def load_model(self, path: str) -> None:
        """Load the Darts model and scaler state from disk.

        Args:
            path: Base path for the model artifact. The scaler state is read
                from ``path + ".scalers"``.

        Raises:
            FileNotFoundError: The scaler state file is missing.
            ValueError: The saved ``target_scaler_cfg`` does not match the
                current config (prevents silent scaler-mismatch bugs).
        """
        path = str(path)
        scaler_path = path + ".scalers"
        try:
            scaler_data = torch.load(
                scaler_path, map_location="cpu", weights_only=False
            )
            self.target_scaler = scaler_data["target_scaler"]
            self.feature_scaler = scaler_data["feature_scaler"]
            self.scaler_fitted = scaler_data["scaler_fitted"]
            self._log_targets = scaler_data.get("log_targets", False)
            self._log_features = set(scaler_data.get("log_features", []))
            self._feature_scaler_map_cfg = scaler_data.get("feature_scaler_map_cfg")
            self._feature_scaler_cfg = scaler_data.get("feature_scaler_cfg")
            saved_target_scaler_cfg = scaler_data.get("target_scaler_cfg")
            if saved_target_scaler_cfg is not None:
                if saved_target_scaler_cfg != self._target_scaler_cfg:
                    raise ValueError(
                        f"SCALER CONFIG MISMATCH: artifact was saved with "
                        f"target_scaler='{saved_target_scaler_cfg}' but "
                        f"current config has "
                        f"target_scaler='{self._target_scaler_cfg}'. "
                        "Retrain the model or align the config before loading."
                    )
                self._target_scaler_cfg = saved_target_scaler_cfg
            logger.info(
                "Scalers loaded from %s. target_scaler=%r, "
                "feature_scaler=%r, scaler_fitted=%s.",
                scaler_path,
                self._target_scaler_cfg,
                self._feature_scaler_cfg,
                self.scaler_fitted,
            )
        except FileNotFoundError:
            logger.error("Scaler state not found. Please retrain the model.")
            raise

        # Load the model (class method returns a new instance).
        self.model = self.model.__class__.load(
            path=path, map_location=str(self.device)
        )

        if hasattr(self.model, "to_device"):
            self.model.to_device(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(self.device)
        logger.info("Model loaded and moved to device: %s", self.device)
