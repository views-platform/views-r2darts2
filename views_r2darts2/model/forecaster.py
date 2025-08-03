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
        self._feature_scaler = feature_scaler
        self._target_scaler = target_scaler

        self.scaler_fitted = False  # Track scaler state

        if self._target_scaler:
            self.target_scaler = Scaler(ScalerSelector.get_scaler(self._target_scaler))
        else:
            self.target_scaler = None

        if self._feature_scaler:
            self.feature_scaler = Scaler(
                ScalerSelector.get_scaler(self._feature_scaler)
            )
        else:
            self.feature_scaler = None

        logger.info(f"Using feature scaler: {self._feature_scaler}")
        logger.info(f"Using target scaler: {self._target_scaler}")

        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
        if hasattr(self.model, "to_device"):
            self.model.to_device(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(self.device)

    @staticmethod
    def get_device() -> str:
        """
        Returns the device type for model training.
        """
        if torch.backends.mps.is_available():
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
        timeseries = [s.astype(np.float32) for s in timeseries]

        # Calculate valid prediction window boundaries

        self.min_length = self.model.input_chunk_length + self.model.output_chunk_length

        if train_mode:  # Training mode
            # Slice targets to allow full input+output window creation
            targets = [
                s.slice(start_ts=start, end_ts=end - self.model.output_chunk_length)[
                    self.dataset.targets
                ]
                for s in timeseries
                if len(s) >= self.min_length
            ]
            past_cov = [s[self.dataset.features] for s in timeseries]
        else:  # Prediction mode
            # Get last valid input window for forecasting
            targets = [
                s.slice(start_ts=start, end_ts=end)[self.dataset.targets]
                for s in timeseries
            ]
            past_cov = [
                s.slice(start_ts=start, end_ts=end)[self.dataset.features]
                for s in timeseries
            ]

        if train_mode:
            logger.info(f"Fitting scalers for training data...")
            if self.target_scaler:
                targets = self.target_scaler.fit_transform(targets)
            if self.feature_scaler:
                past_cov = self.feature_scaler.fit_transform(past_cov)
            self.scaler_fitted = True  # Mark scalers as fitted
        else:
            # Prediction mode: use fitted scalers
            logger.info(f"Transforming scalers for prediction data...")
            if self.target_scaler and self.scaler_fitted:
                targets = self.target_scaler.transform(targets)
            if self.feature_scaler and self.scaler_fitted:
                past_cov = self.feature_scaler.transform(past_cov)

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

            # Get all samples as numpy array (timesteps x components x samples)
            pred_values = pred.all_values(copy=False)

            # Ensure 3D array (time, components, samples) even for deterministic forecasts
            if pred_values.ndim == 2:
                pred_values = pred_values[..., np.newaxis]  # Add sample dimension

            # Replace NaNs and infs with 0, convert to float64, then clip negative values for all samples
            pred_values = np.nan_to_num(pred_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
            pred_values = np.clip(pred_values, a_min=0, a_max=None).astype(np.float64)

            # Convert to list format
            for time_idx in range(pred_values.shape[0]):
                time_stamp = pred.start_time() + time_idx * pred.freq
                row_data = {
                    self.dataset._time_id: time_stamp,
                    self.dataset._entity_id: entity_id,
                }

                for comp_idx, target in enumerate(self.dataset.targets):
                    # Extract all samples for this component and time step
                    samples = pred_values[time_idx, comp_idx, :].tolist()
                    row_data[f"pred_{target}"] = samples

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
            'scaler_fitted': self.scaler_fitted
        }, scaler_path)

    def load_model(self, path: str) -> None:
        # Load scaler state
        path = str(path)
        scaler_path = path + ".scalers"
        try:
            scaler_data = torch.load(scaler_path, map_location='cpu')
            self.target_scaler = scaler_data['target_scaler']
            self.feature_scaler = scaler_data['feature_scaler']
            self.scaler_fitted = scaler_data['scaler_fitted']
        except FileNotFoundError:
            logger.warning("Scaler state not found. Scaling disabled.")
            self.scaler_fitted = False
        
        # Load the model
        self.model = self.model.load(path=path)
        
        # Move model to appropriate device
        if hasattr(self.model, "to_device"):
            self.model.to_device(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(self.device)
