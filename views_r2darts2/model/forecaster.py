import pandas as pd
import numpy as np
from typing import List, Optional
import torch
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from views_r2darts2.utils.scaling import ScalerSelector
from views_r2darts2.data.handlers import _ViewsDatasetDarts
from darts.dataprocessing.transformers import Scaler

import logging
logger = logging.getLogger(__name__)

class DartsForecaster:
    def __init__(
        self,
        dataset: _ViewsDatasetDarts,
        model: TorchForecastingModel,
        partition_dict: dict,
        feature_scaler: str = None,
        target_scaler: str = None,
    ):
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
            self.feature_scaler = Scaler(ScalerSelector.get_scaler(self._feature_scaler))
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
        Preprocesses time series with proper alignment for NBEATS input/output windows.
        Ensures output_length=36 is maintained through correct temporal slicing.
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

        if self.target_scaler:
            targets = self.target_scaler.fit_transform(targets)
        if self.feature_scaler:
            past_cov = self.feature_scaler.fit_transform(past_cov)

        return targets, past_cov

    def _process_predictions(self, timeseries_pred: List[TimeSeries]) -> list:
        # Process predictions into list format
        results = []
        for pred in timeseries_pred:
            entity_id = int(pred.static_covariates.iat[0, 0])

            # Get all samples as numpy array (timesteps x components x samples)
            pred_values = pred.all_values(copy=False)

            # Ensure 3D array (time, components, samples) even for deterministic forecasts
            if pred_values.ndim == 2:
                pred_values = pred_values[..., np.newaxis]  # Add sample dimension

            # Clip negative values for all samples
            pred_values = np.clip(pred_values, a_min=0, a_max=None)

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
        timeseries = self.dataset.as_darts_timeseries()

        target_series, past_covariates = self._preprocess_timeseries(
            timeseries=timeseries, start=self._train_start, end=self._train_end, train_mode=True
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
        timeseries = self.dataset.as_darts_timeseries()

        # Get the input window for forecasting based on sequence_number 
        target_series, past_covariates = self._preprocess_timeseries(
            timeseries=timeseries,
            # start=self._test_start + sequence_number - output_length,
            start=self._test_start + sequence_number - self.model.input_chunk_length,
            end=self._test_start + sequence_number, #self._test_start + sequence_number is exclusive
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
        return df.sort_index()

    def save_model(self, path: str) -> None:
        path = str(path)
        self.model.save(path=path)

    def load_model(self, path: str) -> None:
        path = str(path)
        self.model = self.model.load(path=path)