import pandas as pd
import numpy as np
from typing import List
import torch
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from views_r2darts2.utils.scaling import ScalerSelector
from views_r2darts2.data.handlers import _ViewsDatasetDarts
from darts.dataprocessing.transformers import Scaler

import logging

logger = logging.getLogger(__name__)

class DartsForecaster:
    """
    DartsForecaster is a wrapper class for time series forecasting using Darts models.
    This class manages the workflow for training and predicting with a TorchForecastingModel,
    including preprocessing, scaling, device management, and result formatting.
        dataset (_ViewsDatasetDarts): The dataset containing time series, features, and targets.
        model (TorchForecastingModel): The Darts forecasting model instance.
        partition_dict (dict): Dictionary specifying 'train' and 'test' index ranges.
    Methods:
        get_device(): Returns the device type for model training (cpu, cuda, or mps).
        _preprocess_timeseries(timeseries, start, end, train_mode): Preprocesses time series for training or prediction.
        _process_predictions(timeseries_pred): Converts model predictions to structured list of dictionaries.
        train(): Trains the forecasting model using the dataset.
        predict(sequence_number, output_length, **predict_kwargs): Generates forecasts for a given sequence.
        save_model(path): Saves the current model to the specified file path.
        load_model(path): Loads a trained model from the specified file path.
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
    ):
        """
        Initializes the forecaster with dataset, model, partition information, and optional scalers.

        Args:
            dataset (_ViewsDatasetDarts): The dataset to be used for forecasting.
            model (TorchForecastingModel): The forecasting model instance.
            partition_dict (dict): Dictionary containing 'train' and 'test' partition indices.
            feature_scaler (str, optional): Name of the feature scaler to use. Defaults to None.
            target_scaler (str, optional): Name of the target scaler to use. Defaults to None.

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
            feature_scaler (Scaler or None): Feature scaler instance if provided.
            device (torch.device): Device used for model computation.

        Logs:
            Information about selected scalers and device.
        """
        self.dataset = dataset
        self.model = model
        self._train_start, self._train_end = partition_dict["train"]
        self._test_start, self._test_end = partition_dict["test"]

        self._feature_scaler_cfg = feature_scaler
        self._target_scaler_cfg = target_scaler
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
            logger.error(
                "Both log_features and feature_scaler='LogTransform' are set. "
                "This may apply log transform twice on overlapping features! "
                "Consider using only one transformation method."
            )
            raise

        self.scaler_fitted = False

        self.target_scaler = self._instantiate_scaler(self._target_scaler_cfg)
        self.feature_scaler = self._instantiate_scaler(self._feature_scaler_cfg)

        logger.info(f"Using feature scaler: {self._feature_scaler_cfg}")
        logger.info(f"Using target scaler: {self._target_scaler_cfg}")

        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
        if hasattr(self.model, "to_device"):
            self.model.to_device(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(self.device)

    def _instantiate_scaler(self, scaler_cfg):
        """
        Instantiate and wrap a scaler config.
        Accepts:
          - None
          - String (e.g. 'StandardScaler')
          - Dict: {'name': <str>, 'kwargs': <dict>}
        Returns:
          Darts Scaler wrapper or None.
        """
        if scaler_cfg is None:
            return None
        from darts.dataprocessing.transformers import Scaler

        if isinstance(scaler_cfg, str):
            estimator = ScalerSelector.get_scaler(scaler_cfg)
            return Scaler(estimator)
        if isinstance(scaler_cfg, dict):
            name = scaler_cfg.get("name")
            kwargs = scaler_cfg.get("kwargs", {})
            if name is None:
                raise ValueError("Scaler config dict must have a 'name' key.")
            estimator = ScalerSelector.get_scaler(name, **kwargs)
            return Scaler(estimator)
        raise TypeError("Scaler config must be None, str, or dict.")
    
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
            out.append(ts.map(lambda arr: np.log1p(np.maximum(arr, 0)).astype(np.float32)))
        return out

    def _inverse_log_on_predictions(self, series_list: List[TimeSeries]) -> List[TimeSeries]:
        """
        Inverse of _apply_log_to_targets.
        - Ensure non-negative before expm1 (safety for any numerical drift).
        - Cast to float32.
        """
        if not self._log_targets:
            return series_list
        logger.info("Applying vectorized expm1 inverse transform to predicted series...")
        out = []
        for ts in series_list:
            out.append(ts.map(lambda arr: np.expm1(np.maximum(arr, 0)).astype(np.float32)))
        return out
    
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
        logger.info(f"Applying vectorized log1p transform to selected feature components: {self._log_features}...")
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
        timeseries: TimeSeries,
        start: int,
        end: int,
        train_mode: bool = False,
    ) -> TimeSeries:
        """
        Preprocesses a list of time series for training or prediction.

        Converts each time series to float32, slices the series according to the specified
        start and end indices, and applies scaling if scalers are provided. Handles both
        training and prediction modes by adjusting the slicing logic for targets and features.

        Args:
            timeseries (TimeSeries): List of time series objects to preprocess.
            start (int): Start timestamp for slicing the time series.
            end (int): End timestamp for slicing the time series.
            train_mode (bool, optional): If True, preprocesses for training by ensuring
                full input and output window creation. If False, preprocesses for prediction.
                Defaults to False.

        Returns:
            Tuple[List[TimeSeries], List[TimeSeries]]: A tuple containing the preprocessed
            targets and past covariates (features).
        """
        timeseries_float = [s.astype(np.float32) for s in timeseries]

        self.min_length = self.model.input_chunk_length + self.model.output_chunk_length
        if train_mode:
            targets = [
                s.slice(start_ts=start, end_ts=end - self.model.output_chunk_length)[
                    self.dataset.targets
                ]
                for s in timeseries_float
                if len(s) >= self.min_length
            ]
            past_cov = [s[self.dataset.features].astype(np.float32) for s in timeseries_float]
        else:
            targets = [
                s.slice(start_ts=start, end_ts=end)[self.dataset.targets]
                for s in timeseries_float
            ]
            past_cov = [
                s.slice(start_ts=start, end_ts=end)[self.dataset.features].astype(np.float32)
                for s in timeseries_float
            ]


        # Log transform selected feature components before scaling
        past_cov = self._apply_log_to_features(past_cov)

        # Log transform before scaling (can create float64)
        targets = self._apply_log_to_targets(targets)

        if train_mode:
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
        past_cov = [pc.astype(np.float32) for pc in past_cov]

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
        # Process predictions into list format
        results = []
        for pred in timeseries_pred:
            entity_id = int(pred.static_covariates.iat[0, 0])
            pred_values = pred.all_values(copy=False)
            if pred_values.ndim == 2:
                pred_values = pred_values[..., np.newaxis]
            pred_values = np.nan_to_num(pred_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            pred_values = np.clip(pred_values, a_min=0, a_max=None).astype(np.float32)
            for time_idx in range(pred_values.shape[0]):
                time_stamp = pred.start_time() + time_idx * pred.freq
                row_data = {
                    self.dataset._time_id: time_stamp,
                    self.dataset._entity_id: entity_id,
                }
                for comp_idx, target in enumerate(self.dataset.targets):
                    row_data[f"pred_{target}"] = pred_values[time_idx, comp_idx, :].tolist()
                results.append(row_data)
        return results

    def train(self) -> None:
        """
        Trains the forecasting model using the dataset provided.

        This method preprocesses the time series data and any associated features,
        converts them to the appropriate data type, and fits the model using the
        processed target series and past covariates.

        Returns:
            None
        """
        timeseries = self.dataset.as_darts_timeseries()

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

        # Train the model
        self.model.fit(
            series=target_series,
            past_covariates=past_covariates,
            verbose=True,
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
            Exception: If an error occurs during prediction.
        """
        timeseries = self.dataset.as_darts_timeseries()

        # Get the input window for forecasting based on sequence_number
        target_series, past_covariates = self._preprocess_timeseries(
            timeseries=timeseries,
            # start=self._test_start + sequence_number - output_length,
            start=self._test_start + sequence_number - self.model.input_chunk_length,
            end=self._test_start
            + sequence_number,  # self._test_start + sequence_number is exclusive
        )

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
            print(f"Error during prediction: {e}")
            raise

        if self.target_scaler:
            timeseries_pred = self.target_scaler.inverse_transform(timeseries_pred)

        timeseries_pred = self._inverse_log_on_predictions(timeseries_pred)

        # Process predictions into list format
        results = self._process_predictions(timeseries_pred)

        # Create final DataFrame
        df = pd.DataFrame(results)
        df = df.set_index([self.dataset._time_id, self.dataset._entity_id])

        # Force nan to 0
        df = df.fillna(0)
        return df.sort_index()

    def save_model(self, path: str) -> None:
        # Save scaler state along with model
        path = str(path)
        self.model.save(path=path)
        scaler_path = path + ".scalers"
        torch.save({
            'target_scaler': self.target_scaler,
            'feature_scaler': self.feature_scaler,
            'scaler_fitted': self.scaler_fitted,
            'log_targets': self._log_targets,
            'log_features': list(self._log_features),
        }, scaler_path)

    def load_model(self, path: str) -> None:
        # Load scaler state
        path = str(path)
        scaler_path = path + ".scalers"
        try:
            scaler_data = torch.load(scaler_path, map_location='cpu', weights_only=False)
            self.target_scaler = scaler_data['target_scaler']
            self.feature_scaler = scaler_data['feature_scaler']
            self.scaler_fitted = scaler_data['scaler_fitted']
            self._log_targets = scaler_data.get('log_targets', False)
            self._log_features = set(scaler_data.get('log_features', []))
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
