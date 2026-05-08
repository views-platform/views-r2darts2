import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from views_r2darts2.transformers.scaler_selector import ScalerSelector
from views_r2darts2.transformers.feature_scaler_manager import FeatureScalerManager
from views_r2darts2.transformers.views_dataset_darts import _ViewsDatasetDarts
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate

import logging

logger = logging.getLogger(__name__)


class DartsForecaster:
    """
    DartsForecaster is a wrapper class for time series forecasting using Darts models.
    This class manages the workflow for training and predicting with a TorchForecastingModel,
    including preprocessing, scaling, device management, and result formatting.

    Intent Contract:
        - Purpose: Maintain the stateful coupling between a deep learning model and its required
          preprocessing pipeline (scalers, log-transforms) to ensure predictions are on the correct scale.
        - Non-Goals: Does not manage Weights & Biases logging or experiment orchestration.
        - Guarantees:
            - Ensures that data is downcast to float32 before entering the model.
            - Guarantees that target scalers are fitted ONLY on training data and correctly applied
              in inverse during prediction (preserving sample dimensions for probabilistic forecasts).
            - Ensures physical boundaries are respected during preprocessing via ReproducibilityGate.
        - Failure Behavior: Raises RuntimeError if prediction is attempted before scalers are fitted or
          if numerical insanity is detected in the input tensors.
    """

    def __init__(
        self,
        dataset: _ViewsDatasetDarts,
        model: TorchForecastingModel,
        partition_dict: dict,
        feature_scaler: str = None,
        target_scaler: str = None,
        log_targets: bool = False,
        log_features: list[str] | None = None,
        feature_scaler_map: Optional[Dict[str, Any]] = None,
        random_state: int = None,
        static_covariate_stats: Optional[Dict[str, Any]] = None,
        checkpoint_mode: str = "best",
    ):
        """
        Initializes the forecaster with dataset, model, partition information, and optional scalers.

        Args:
            dataset (_ViewsDatasetDarts): The dataset to be used for forecasting.
            model (TorchForecastingModel): The forecasting model instance.
            partition_dict (dict): Dictionary containing 'train' and 'test' partition indices.
            feature_scaler (str, optional): Name of the feature scaler to use for all features.
                Ignored if feature_scaler_map is provided. Defaults to None.
            target_scaler (str, optional): Name of the target scaler to use. Defaults to None.
            log_targets (bool, optional): Whether to apply log1p transform to targets. Defaults to False.
            log_features (list[str], optional): List of feature names to apply log1p transform. Defaults to None.
            feature_scaler_map (dict, optional): Mapping of scalers to specific feature groups.
                When provided, this takes precedence over feature_scaler.
            random_state (int): Random seed for reproducibility. Mandatory.
            static_covariate_stats (dict, optional): Configuration for per-entity
                static covariate statistics injection. When provided, must contain:
                  - 'transform' (str or None): Name of the element-wise transform to apply
                    to mu, sigma, max, trend stats before injection. Supported:
                    'AsinhTransform', 'LogTransform', 'SqrtTransform', 'FourthRootTransform'.
                    When None, stats are injected in raw space.
                When this parameter is None, static covariate stats are still injected
                (backward compatible) in raw space.
...
        Attributes:
            dataset (_ViewsDatasetDarts): The provided dataset.
            model (TorchForecastingModel): The forecasting model.
            _train_start (int): Start index for training partition.
            _train_end (int): End index for training partition.
            _test_start (int): Start index for testing partition.
            _test_end (int): End index for testing partition.
            _feature_scaler (str): Name of the feature scaler.
            _target_scaler (str): Name of the target scaler.
            scaler_fitted (bool): Indicates if scalers have been fitted.
            target_scaler (Scaler or None): Target scaler instance if provided.
            feature_scaler (Scaler, FeatureScalerManager, or None): Feature scaler instance.
            device (torch.device): Device used for model computation.
            random_state (int): Captured random seed for entropy locking.
            _static_cov_transform (str or None): Transform name for static covariate stats.

        Logs:
            Information about selected scalers and device.
        """
        self.dataset = dataset
        self.model = model
        self._train_start, self._train_end = partition_dict["train"]
        self._test_start, self._test_end = partition_dict["test"]

        if random_state is None:
            raise ValueError(
                "MANDATORY PARAMETER MISSING: random_state must be provided to DartsForecaster."
            )
        self.random_state = random_state

        # Static covariate stats transform (e.g. 'AsinhTransform' or None for raw)
        # and optional stat subset (e.g. ['trend', 'sparsity'] for lightweight injection).
        logger.info(
            f"static_covariate_stats received: {static_covariate_stats!r} "
            f"(type={type(static_covariate_stats).__name__})"
        )
        self._static_cov_transform = (
            static_covariate_stats.get("transform") if static_covariate_stats else None
        )
        self._static_cov_stats = (
            static_covariate_stats.get("stats") if static_covariate_stats else None
        )
        self._static_cov_inject = (
            static_covariate_stats.get("inject", True) if static_covariate_stats else True
        )
        logger.info(
            f"_static_cov_transform resolved to: {self._static_cov_transform!r}, "
            f"_static_cov_stats: {self._static_cov_stats!r}, "
            f"_static_cov_inject: {self._static_cov_inject!r}"
        )

        if checkpoint_mode not in ("best", "last"):
            raise ValueError(f"checkpoint_mode must be 'best' or 'last', got {checkpoint_mode!r}")
        self._checkpoint_mode = checkpoint_mode
        logger.info(f"checkpoint_mode: {self._checkpoint_mode!r}")

        self._feature_scaler_cfg = feature_scaler
        self._target_scaler_cfg = target_scaler
        self._feature_scaler_map_cfg = feature_scaler_map
        self._log_targets = bool(log_targets)
        self._log_features = set(log_features or [])

        # Warn about potential double log transform
        if self._log_targets and target_scaler == "LogTransform":
            logger.warning(
                "Both log_targets=True and target_scaler='LogTransform' are set. "
                "This will apply log transform twice! Consider using only one. "
                "Disabling manual log_targets to avoid double transformation."
            )
            self._log_targets = False

        if self._log_features and feature_scaler == "LogTransform":
            raise ValueError(
                "Both log_features and feature_scaler='LogTransform' are set. "
                "This will apply log transform twice on overlapping features. "
                "Use only one transformation method."
            )

        self.scaler_fitted = False

        # Initialize target scaler
        self.target_scaler = self._instantiate_scaler(self._target_scaler_cfg)

        # Initialize feature scaler(s)
        if not self.dataset.features:
            if self._feature_scaler_cfg or self._feature_scaler_map_cfg:
                logger.info(
                    "Dataset has no feature columns — disabling feature_scaler. "
                    "This is expected for univariate models using datafactory."
                )
            self.feature_scaler = None
        elif self._feature_scaler_map_cfg:
            self.feature_scaler = FeatureScalerManager(
                feature_scaler_map=self._feature_scaler_map_cfg,
                default_scaler=self._feature_scaler_cfg,  # fallback for unmapped features
                all_features=self.dataset.features,
            )
            logger.info(f"Using feature scaler map: {self.feature_scaler}")
        else:
            self.feature_scaler = self._instantiate_scaler(self._feature_scaler_cfg)
            logger.info(f"Using feature scaler: {self._feature_scaler_cfg}")

        logger.info(f"Using target scaler: {self._target_scaler_cfg}")

        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
        if hasattr(self.model, "to_device"):
            self.model.to_device(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(self.device)

    @staticmethod
    def _instantiate_scaler(scaler_cfg):
        """Delegate to ScalerSelector.instantiate_darts_scaler()."""
        if scaler_cfg is None:
            return None
        return ScalerSelector.instantiate_darts_scaler(scaler_cfg)

    def _apply_log_to_targets(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        """
        Vectorized log1p for target series.
        FIX: Darts TimeSeries.map passes the full ndarray, not scalars. Prior lambda assumed scalar.
        We now:
          - Clip negatives to 0.
          - Apply log1p.
          - Cast back to float32 (numpy log1p returns float64).
        """
        if not self._log_targets:
            return series_list
        logger.info("Applying vectorized log1p transform to target series...")
        out = []
        for ts in series_list:
            out.append(
                ts.map(lambda arr: np.log1p(np.maximum(arr, 0)).astype(np.float32))
            )
        return out

    def _inverse_log_on_predictions(
        self, series_list: List[TimeSeries]
    ) -> List[TimeSeries]:
        """
        Inverse of _apply_log_to_targets.
        - Ensure non-negative before expm1 (safety for any numerical drift).
        - Cast to float32.
        """
        if not self._log_targets:
            return series_list
        logger.info(
            "Applying vectorized expm1 inverse transform to predicted series..."
        )
        out = []
        for ts in series_list:
            out.append(
                ts.map(lambda arr: np.expm1(np.maximum(arr, 0)).astype(np.float32))
            )
        return out

    def _inverse_transform_target_scaler(
        self, timeseries_pred: List[TimeSeries]
    ) -> List[TimeSeries]:
        """
        Inverse transform predictions using target scaler, preserving samples for probabilistic forecasts.

        For Darts Pipeline objects (used for chained scalers), the inverse_transform
        properly handles probabilistic series. For single Darts Scaler wrapping sklearn
        scalers, we need manual reshaping to preserve the sample dimension.
        """
        if not self.target_scaler or not self.scaler_fitted:
            return timeseries_pred

        from darts.dataprocessing import Pipeline

        # If using Pipeline (for chained scalers), it handles samples correctly
        if isinstance(self.target_scaler, Pipeline):
            return self.target_scaler.inverse_transform(timeseries_pred)

        # For single Scaler, we need to manually handle probabilistic series
        result = []
        for i, ts in enumerate(timeseries_pred):
            arr = ts.all_values(copy=True)
            is_probabilistic = arr.ndim == 3

            if is_probabilistic:
                n_time, n_features, n_samples = arr.shape

                # Reshape to 2D for sklearn: (time * samples, features)
                arr_2d = arr.transpose(0, 2, 1).reshape(-1, n_features)

                # Get the fitted sklearn scaler from the Darts Scaler wrapper
                # Darts stores fitted params as a list/tuple of parameters.
                # If global_fit=True, the list has length 1.
                # If global_fit=False, the list length matches the number of series.
                sklearn_scaler = None
                if (
                    hasattr(self.target_scaler, "_fitted_params")
                    and self.target_scaler._fitted_params
                ):
                    fitted_params = self.target_scaler._fitted_params
                    # Use index 'i' if per-series, or index '0' if global
                    param_idx = i if len(fitted_params) > 1 else 0
                    if param_idx < len(fitted_params):
                        param = fitted_params[param_idx]
                        if isinstance(param, dict) and "fitted" in param:
                            sklearn_scaler = param["fitted"]
                        else:
                            sklearn_scaler = param

                if sklearn_scaler is not None and hasattr(
                    sklearn_scaler, "inverse_transform"
                ):
                    inv_2d = sklearn_scaler.inverse_transform(arr_2d.astype(np.float64))
                else:
                    logger.warning(
                        f"Target scaler fitted params not found for series {i}, skipping inverse transform"
                    )
                    inv_2d = arr_2d

                # Reshape back to 3D: (time, features, samples)
                inv_arr = inv_2d.reshape(n_time, n_samples, n_features).transpose(
                    0, 2, 1
                )

                new_ts = TimeSeries.from_times_and_values(
                    times=ts.time_index,
                    values=inv_arr.astype(np.float32),
                    columns=ts.components,
                    freq=ts.freq,
                    static_covariates=ts.static_covariates,
                )
            else:
                # For deterministic, use standard Darts inverse_transform
                # Standard Darts inverse_transform handles multiple series correctly
                # by internally matching the series index to the fitted params.
                new_ts = self.target_scaler.inverse_transform([ts])[0]

            result.append(new_ts)

        return result

    def _apply_log_to_feature_series(self, ts: TimeSeries) -> TimeSeries:
        """
        Applies log1p to selected feature components in a single TimeSeries.
        Only components whose names appear in self._log_features are transformed.
        Rationale:
          - Early weak signals (small positive feature counts) expanded.
          - Variance stabilized for heavy-tailed covariates.
        Note:
          - No inverse needed; past covariates not reconstructed post-prediction.
          - Negative values clipped to 0 before log1p.
        """
        if not self._log_features:
            return ts
        comps = ts.components
        if not any(c in self._log_features for c in comps):
            return ts
        arr = ts.all_values(copy=True)
        # Deterministic: (time, features); Probabilistic: (time, features, samples)
        if arr.ndim == 2:
            for idx, name in enumerate(comps):
                if name in self._log_features:
                    arr[:, idx] = np.log1p(np.maximum(arr[:, idx], 0.0))
        elif arr.ndim == 3:
            # Apply to each sample identically
            for idx, name in enumerate(comps):
                if name in self._log_features:
                    arr[:, idx, :] = np.log1p(np.maximum(arr[:, idx, :], 0.0))
        # Rebuild TimeSeries preserving metadata
        new_ts = TimeSeries.from_times_and_values(
            times=ts.time_index,
            values=arr.astype(np.float32),
            columns=comps,
            freq=ts.freq,
            static_covariates=ts.static_covariates,
        )
        return new_ts

    def _apply_log_to_features(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        """
        Batch wrapper to apply feature log transform.
        """
        if not self._log_features:
            return series_list
        logger.info(
            f"Applying vectorized log1p transform to selected feature components: {self._log_features}..."
        )
        out = [self._apply_log_to_feature_series(ts) for ts in series_list]
        return out

    @staticmethod
    def get_device() -> str:
        """
        Returns the device type for model training.
        """
        if torch.backends.mps.is_available():
            torch.set_default_dtype(torch.float32)
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _preprocess_timeseries(
        self,
        timeseries: List[TimeSeries],
        start: int,
        end: int,
        train_mode: bool = False,
    ) -> Tuple[List[TimeSeries], Optional[List[TimeSeries]]]:
        """
        Preprocesses time series for training or prediction.

        Args:
            timeseries: Time series collection to preprocess.
            start: Start timestamp for slicing.
            end: End timestamp for slicing.
            train_mode: If True, fits scalers and enforces reproducibility gates.

        Returns:
            Tuple of (targets, past_covariates). past_covariates is None for
            univariate models (features=[]).
        """
        timeseries_float = [s.astype(np.float32) for s in timeseries]

        self.min_length = self.model.input_chunk_length + self.model.output_chunk_length

        # Slice targets (end + 1 because Darts slice is exclusive for integer indices)
        if train_mode:
            # Build aligned (target, past_cov) pairs and filter together.
            #
            # BUG FIXED: Previously targets was filtered by len(s) >= min_length but
            # past_cov had no filter. This caused entity misalignment: targets[i] and
            # past_cov[i] came from DIFFERENT entities whenever any shorter series was
            # excluded. model.fit(series=targets, past_covariates=past_cov) pairs by
            # list index, so every entity after the first filtered-out one was trained
            # on the wrong covariate history.
            #
            # The filter now uses len(sliced_target) not len(s): the full series
            # length includes the test partition, so a series with 10 training months
            # and 74 test months would pass the old len(s)>=84 check while its
            # training slice is far too short for a 48+36 window.
            #
            # We MUST slice past_cov up to 'end + 1' to prevent feature leakage.
            paired = [
                (
                    s.slice(start_ts=start, end_ts=end + 1)[self.dataset.targets],
                    s.slice(start_ts=start, end_ts=end + 1)[self.dataset.features].astype(np.float32),
                )
                for s in timeseries_float
                if len(s.slice(start_ts=start, end_ts=end + 1)) >= self.min_length
            ]
            targets = [p[0] for p in paired]
            past_cov = [p[1] for p in paired]
            logger.info(
                f"Training filter: {len(paired)}/{len(timeseries_float)} entities "
                f"passed minimum length >= {self.min_length}."
            )
        else:
            targets = [
                s.slice(start_ts=start, end_ts=end + 1)[self.dataset.targets]
                for s in timeseries_float
            ]

        # Slice past covariates (end + 1 to prevent feature leakage)
        if self.dataset.features:
            past_cov = [
                s.slice(start_ts=start, end_ts=end + 1)[self.dataset.features].astype(
                    np.float32
                )
                for s in timeseries_float
            ]
            past_cov = self._apply_log_to_features(past_cov)
        else:
            past_cov = None

        targets = self._apply_log_to_targets(targets)

        if train_mode:
            # GATE 3, 4, 5: The Fortress Firewall
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

        # DOWNCAST after scaler/log (they yield float64)
        targets = [ts.astype(np.float32) for ts in targets]
        if past_cov is not None:
            past_cov = [pc.astype(np.float32) for pc in past_cov]

        ReproducibilityGate.Data.audit_numerical_sanity(targets, "targets")
        if past_cov is not None:
            ReproducibilityGate.Data.audit_numerical_sanity(past_cov, "past_covariates")

        return targets, past_cov

    def _process_predictions(self, timeseries_pred: List[TimeSeries]) -> list:
        """
        Processes a list of TimeSeries prediction objects into a structured list of dictionaries.

        Each dictionary in the returned list corresponds to a single time step and entity, containing:
            - The timestamp for the prediction.
            - The entity ID.
            - Predicted samples for each target component, clipped to non-negative values.

        Handles both deterministic and probabilistic forecasts by ensuring the prediction values are 3D arrays.
        Clips negative prediction values to zero.
        Converts prediction samples for each target and time step into lists.

        Args:
            timeseries_pred (List[TimeSeries]): List of TimeSeries prediction objects.

        Returns:
            list: List of dictionaries, each containing time, entity ID, and predicted samples for each target.
        """
        # HANDSHAKE: Audit model output for numerical sanity before processing
        ReproducibilityGate.Data.audit_numerical_sanity(
            timeseries_pred, name="Model Predictions"
        )

        # Process predictions into list format
        results = []
        for pred in timeseries_pred:
            if pred.static_covariates is None:
                raise ValueError(
                    "Prediction TimeSeries is missing static_covariates. "
                    "Ensure data was grouped by entity via TimeSeries.from_group_dataframe()."
                )
            entity_id = int(pred.static_covariates.iat[0, 0])
            pred_values = pred.all_values(copy=False)
            if pred_values.ndim == 2:
                pred_values = pred_values[..., np.newaxis]

            # Clamp: raw count predictions cannot be negative (deaths have a physical floor of 0).
            # Sub-unit fractional counts (0 < pred < 1) are kept as-is; zeroing them
            # collapses the [0, 0.88] asinh range and catastrophically inflates MSLE
            # for low-conflict countries (1–5 deaths/month).
            pred_values = np.maximum(pred_values, 0.0).astype(np.float32)
            for time_idx in range(pred_values.shape[0]):
                time_stamp = pred.start_time() + time_idx * pred.freq
                row_data = {
                    self.dataset._time_id: time_stamp,
                    self.dataset._entity_id: entity_id,
                }
                for comp_idx, target in enumerate(self.dataset.targets):
                    row_data[f"pred_{target}"] = pred_values[
                        time_idx, comp_idx, :
                    ].tolist()
                results.append(row_data)
        return results

    def train(self) -> None:
        """
        Trains the forecasting model using the dataset provided.

        Preprocesses training data, fits scalers, then prepares a validation set
        from the test partition (transformed with train-fitted scalers, no leakage).
        Val loss is computed every epoch for early stopping and monitoring.

        Returns:
            None
        """
        timeseries = self.dataset.as_darts_timeseries(
            stat_time_range=(self._train_start, self._train_end),
            static_cov_transform=self._static_cov_transform,
            static_cov_stats=self._static_cov_stats,
            inject_static_covariates=self._static_cov_inject,
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

        # --- Validation set: test partition, transformed with train-fitted scalers ---
        # No leakage: scalers were fit above on train only; val is .transform()'ed.
        # Static cov stats use train range (passed to as_darts_timeseries above).
        # Val needs icl steps of history before test_start for context window.
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

        # Guard: if the test partition has no ground-truth output steps (e.g.
        # run_type="forecasting" where test_start = train_end + 1), val series
        # will be exactly icl steps long — too short for Darts to build even one
        # sample (needs icl + ocl).
        #
        # Forecasting-mode fix: carve the last ocl steps from the training window
        # as a holdout val set. Scalers are refit on the trimmed window to prevent
        # leakage from holdout targets into the scaler statistics.
        #
        # The holdout months are still used at inference time: predict() slices up
        # to self._train_end as context, so the model sees them as encoder input
        # even though they never appeared in a gradient update.
        _min_val_len = self.model.input_chunk_length + self.model.output_chunk_length
        _max_val_len = max((len(ts) for ts in val_targets), default=0)
        if _max_val_len < _min_val_len:
            _ocl = self.model.output_chunk_length
            _icl = self.model.input_chunk_length
            trimmed_train_end = self._train_end - _ocl
            carved_val_start = self._train_end - _ocl - _icl + 1

            logger.info(
                f"Forecasting mode: val partition too short ({_max_val_len} < {_min_val_len}). "
                f"Carving holdout val [{carved_val_start}, {self._train_end}] ({_icl + _ocl} steps). "
                f"Refitting scalers on trimmed train [{self._train_start}, {trimmed_train_end}]."
            )

            # Refit on trimmed window — overwrites scaler fit from the full-range call above.
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

            # Build carved val (scalers already refitted above; transform only, no leakage).
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

        _used_carved_val = _max_val_len < _min_val_len
        _log_val_start = carved_val_start if _used_carved_val else val_start
        _log_val_end = self._train_end if _used_carved_val else val_end
        logger.info(
            f"Validation set: {len(val_targets) if val_targets is not None else 0} entities, "
            f"range [{_log_val_start}, {_log_val_end}] "
            f"({'carved from train end' if _used_carved_val else 'test partition'} "
            f"with {self.model.input_chunk_length} steps of context)."
        )

        # Train the model
        # Auto-detect num_workers: use half of available CPUs, capped at 8, minimum 0
        num_workers = min(max((os.cpu_count() or 1) // 2, 0), 8)
        # Note: persistent_workers=False to avoid file descriptor exhaustion in sweeps
        # Set persistent_workers=True only for single long runs where performance matters
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

        # After fit(), Darts automatically reloads the best val_loss checkpoint.
        # When checkpoint_mode='last', explicitly reload the final epoch weights
        # instead — useful when the training objective diverges from val_loss
        # (e.g. SpotlightLoss DRO shifts improve event_ratio late in training
        # but don't reduce val_loss).
        if self._checkpoint_mode == "last":
            try:
                self.model.load_weights_from_checkpoint(best=False)
                logger.info("checkpoint_mode='last': reloaded final epoch weights.")
            except Exception as e:
                logger.warning(
                    f"checkpoint_mode='last': failed to reload last checkpoint ({e}). "
                    "Keeping best val_loss checkpoint."
                )

    def predict(
        self,
        sequence_number: int,
        output_length: int = 36,
        **predict_kwargs,
    ) -> pd.DataFrame:
        """
        Generates forecasts for a given sequence number using the trained model.

        Args:
            sequence_number (int): The index in the test set to start forecasting from.
            output_length (int, optional): Number of time steps to forecast. Defaults to 36.
            **predict_kwargs: Additional keyword arguments to pass to the model's predict method.

        Returns:
            pd.DataFrame: A DataFrame containing the forecasted values, indexed by time and entity.

        Raises:
            RuntimeError: If scalers are not fitted (call train() or load_model() first).
            Exception: If an error occurs during prediction.
        """
        if self.target_scaler and not self.scaler_fitted:
            raise RuntimeError(
                "predict() called before scalers were fitted. "
                "Call train() or load_model() first."
            )

        # LOCK ENTROPY: Guarantee bit-perfect identity for probabilistic samples
        ReproducibilityGate.Data.lock_entropy(self.random_state)

        # Scaler provenance log: confirms whether scalers are in-memory (sweep path,
        # freshly fitted during train()) or disk-loaded (eval path, loaded via
        # load_model()). For stateless transforms (AsinhTransform), both are
        # functionally identical. For stateful scalers, divergence is silent if the
        # disk artifact was saved from a different training run.
        #
        # NOTE (structural): the sweep path never calls save_model() — _train_model_artifact()
        # skips save when config["sweep"]=True. Sweep eval always uses in-memory scalers.
        # Regular eval always loads from the latest on-disk artifact. These are DIFFERENT
        # scaler instances even if they produce identical outputs for stateless transforms.
        # Any time you switch to a stateful scaler (StandardScaler, MinMaxScaler, etc.),
        # verify that both paths load from the same artifact or retrain to refresh the disk scaler.
        logger.info(
            f"predict() scaler state: "
            f"target_scaler='{self._target_scaler_cfg}' (fitted={self.target_scaler is not None}), "
            f"feature_scaler='{self._feature_scaler_cfg}' (fitted={self.feature_scaler is not None}), "
            f"scaler_fitted={self.scaler_fitted}."
        )

        timeseries = self.dataset.as_darts_timeseries(
            stat_time_range=(self._train_start, self._train_end),
            static_cov_transform=self._static_cov_transform,
            static_cov_stats=self._static_cov_stats,
            inject_static_covariates=self._static_cov_inject,
        )

        # Get the input window for forecasting based on sequence_number
        target_series, past_covariates = self._preprocess_timeseries(
            timeseries=timeseries,
            start=self._test_start + sequence_number - self.model.input_chunk_length,
            end=self._test_start - 1 + sequence_number,  # origin = test_start - 1 + seq (base_origin convention)
        )

        # Resilient Device Management: Ensure model is on the correct device
        # Darts models often shift to CPU in teardown(); we restore them if needed.
        current_device = next(self.model.model.parameters()).device
        if self.device != "cpu" and current_device.type == "cpu":
            logger.info(f"Restoring model to {self.device} before prediction...")
            if hasattr(self.model, "to_device"):
                self.model.to_device(self.device)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
                self.model.model.to(self.device)

            # Final verification after restoration attempt
            current_device = next(self.model.model.parameters()).device
            if current_device.type == "cpu":
                error_msg = (
                    f"CRITICAL DEVICE FAILURE: Failed to move model from CPU to {self.device}. "
                    "Prediction aborted to prevent inconsistent results."
                )
                logger.critical(error_msg)
                raise RuntimeError(error_msg)

        # Generate forecasts
        try:
            timeseries_pred = self.model.predict(
                n=output_length,
                series=target_series,
                past_covariates=past_covariates,
                verbose=True,
                **predict_kwargs,
            )
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

        # Use sample-preserving inverse transform for probabilistic predictions
        if self.target_scaler:
            timeseries_pred = self._inverse_transform_target_scaler(timeseries_pred)

        timeseries_pred = self._inverse_log_on_predictions(timeseries_pred)

        # Process predictions into list format
        results = self._process_predictions(timeseries_pred)

        # Create final DataFrame
        df = pd.DataFrame(results)
        df = df.set_index([self.dataset._time_id, self.dataset._entity_id])

        # Numerical Sanity Check: Ensure no NaNs leaked through the inverse pipeline
        # (df.fillna(0) is forbidden by ADR-010 and ADR-008)
        if df.isna().any().any():
            from views_r2darts2.infrastructure.exceptions import NumericalSanityError

            error_msg = "Numerical Sanity Violation: NaNs detected in final prediction DataFrame."
            logger.critical(error_msg)
            raise NumericalSanityError(error_msg)

        return df.sort_index()

    def save_model(self, path: str) -> None:
        # Save scaler state along with model
        path = str(path)
        self.model.save(path=path)
        scaler_path = path + ".scalers"

        # Determine if using FeatureScalerManager
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
                # _target_scaler_cfg was previously missing from this dict, causing
                # self._target_scaler_cfg to reflect the current config rather than
                # the one used during training after a load_model() call.
                "target_scaler_cfg": self._target_scaler_cfg,
            },
            scaler_path,
        )

    def load_model(self, path: str) -> None:
        # Load scaler state
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
            # Restore target_scaler_cfg from the saved artifact so that
            # self._target_scaler_cfg reflects what was used during training,
            # not whatever the current config says.
            saved_target_scaler_cfg = scaler_data.get("target_scaler_cfg")
            if saved_target_scaler_cfg is not None:
                if saved_target_scaler_cfg != self._target_scaler_cfg:
                    raise ValueError(
                        f"SCALER CONFIG MISMATCH: artifact was saved with "
                        f"target_scaler='{saved_target_scaler_cfg}' but current "
                        f"config has target_scaler='{self._target_scaler_cfg}'. "
                        "Retrain the model or align the config before loading this artifact."
                    )
                self._target_scaler_cfg = saved_target_scaler_cfg
            logger.info(
                f"Scalers loaded from {scaler_path}. "
                f"target_scaler='{self._target_scaler_cfg}', "
                f"feature_scaler='{self._feature_scaler_cfg}', "
                f"scaler_fitted={self.scaler_fitted}."
            )
        except FileNotFoundError:
            logger.error("Scaler state not found. Please retrain the model.")
            raise

        # Load the model - use the class method to get a new instance
        self.model = self.model.__class__.load(path=path, map_location=str(self.device))

        # Ensure model is on the correct device
        if hasattr(self.model, "to_device"):
            self.model.to_device(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(self.device)

        logger.info(f"Model loaded and moved to device: {self.device}")
