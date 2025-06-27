import logging
from os import path
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from views_r2darts2.model import forecaster
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_r2darts2.data.handlers import _ViewsDatasetDarts
from views_r2darts2.model.forecaster import DartsForecaster
from views_pipeline_core.files.utils import generate_model_file_name
from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.files.utils import (
    read_dataframe,
)

from views_r2darts2.model.catalog import ModelCatalog

logger = logging.getLogger(__name__)

# https://github.com/suno-ai/bark/pull/619#issuecomment-2726747073
# Save the original torch.load function
_original_torch_load = torch.load

# Function that forces weights_only=False
def custom_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


class DartsForecastingModelManager(ForecastingModelManager):

    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = True,
        use_prediction_store: bool = True,
    ) -> None:
        super().__init__(
            model_path=model_path,
            wandb_notifications=wandb_notifications,
            use_prediction_store=use_prediction_store,
        )
        # Override torch.load globally
        torch.load = custom_torch_load
        logger.info(f"Current model architecture: \033[92m{self.configs['algorithm']}\033[0m")

    def _train_model_artifact(self):
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        run_type = self.config["run_type"]

        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )
        # Partitioner dict from ViewsDataLoader
        model_object = ModelCatalog(config=self.config).get_model(
            model_name=self.config["algorithm"]
        )

        forecaster = DartsForecaster(
            dataset=_ViewsDatasetDarts(
                source=df_viewser,
                targets=self.config.get("targets"),
                broadcast_features=True,
            ),
            model=model_object,
            partition_dict=self._data_loader.partition_dict,
            feature_scaler=self.config.get("feature_scaler", None),
            target_scaler=self.config.get("target_scaler", None),
        )
        forecaster.train()

        if not self.config["sweep"]:  # If not using wandb sweep
            model_filename = generate_model_file_name(
                run_type, file_extension=".pt"
            )  # Generate the model file name

            forecaster.save_model(path=f"{path_artifacts / model_filename}")
        # Save the model artifact to the artifacts directory "path_artifacts" or "self._model_path.artifacts"

        return forecaster

    def _evaluate_model_artifact(self, eval_type, artifact_name=None):
        # eval_type can be "standard", "long", "complete", "live"
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        run_type = self.config["run_type"]

        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")
            path_artifact = path_artifacts / artifact_name
        else:
            # Automatically use the latest model artifact based on the run type
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact"
            )
            path_artifact = self._model_path.get_latest_model_artifact_path(
                run_type
            )  # Path to the latest model artifact if it exists

        self.config["timestamp"] = path_artifact.stem[
            -15:
        ]  # Extract the timestamp from the artifact name

        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        model_object = ModelCatalog(config=self.config).get_model(
            model_name=self.config["algorithm"]
        )
        forecaster = DartsForecaster(
            dataset=_ViewsDatasetDarts(
                source=df_viewser,
                targets=self.config.get("targets"),
                broadcast_features=True,
            ),
            model=model_object,
            partition_dict=self._data_loader.partition_dict,
            scale=self.config.get("scale_data", True),
        )
        forecaster.load_model(path=path_artifact)

        total_sequence_number = (
            ForecastingModelManager._resolve_evaluation_sequence_number(eval_type)
        )

        df_predictions = [
            forecaster.predict(
                sequence_number,
                max(self.config["steps"]),
                num_samples=self.config.get("num_samples", 1),
                n_jobs=self.config.get("n_jobs", 1),
                mc_dropout=self.config.get("mc_dropout", False),
            )
            for sequence_number in range(total_sequence_number)
        ]

        return df_predictions

    def _forecast_model_artifact(self, artifact_name):
        # Commonly used paths
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        run_type = self.config["run_type"]

        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact"
            )
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        self.config["timestamp"] = path_artifact.stem[-15:]
        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        model_object = ModelCatalog(config=self.config).get_model(
            model_name=self.config["algorithm"]
        )
        forecaster = DartsForecaster(
            dataset=_ViewsDatasetDarts(
                source=df_viewser,
                targets=self.config.get("targets"),
                broadcast_features=True,
            ),
            model=model_object,
            partition_dict=self._data_loader.partition_dict,
            scale=self.config.get("scale_data", True),
        )
        forecaster.load_model(path=path_artifact)

        df_predictions = forecaster.predict(
            0,
            max(self.config["steps"]),
            num_samples=self.config.get("num_samples", 1),
            n_jobs=self.config.get("n_jobs", 1),
            mc_dropout=self.config.get("mc_dropout", False),
        )

        return df_predictions

    def _evaluate_sweep(self, eval_type: str, model: any):

        total_sequence_number = (
            ForecastingModelManager._resolve_evaluation_sequence_number(eval_type)
        )

        df_predictions = [
            model.predict(sequence_number, max(self.config["steps"]))
            for sequence_number in range(total_sequence_number)
        ]

        return df_predictions
