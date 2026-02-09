import logging
import torch
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
    """
    Loads a PyTorch model using the original torch load function, ensuring the 'weights_only' argument is set.

    Args:
        *args: Positional arguments to pass to the original torch load function.
        **kwargs: Keyword arguments to pass to the original torch load function. If 'weights_only' is not provided, it defaults to False.

    Returns:
        The result of the original torch load function with the specified arguments.

    """
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


class DartsForecastingModelManager(ForecastingModelManager):
    """
    DartsForecastingModelManager

    Manages the lifecycle of Darts-based forecasting models, including training, evaluation, and artifact management.

    This class extends ForecastingModelManager to provide specialized functionality for time series forecasting using the Darts library. It handles model initialization, training, evaluation, and prediction workflows, as well as artifact saving and loading. The manager integrates with external tools such as Weights & Biases for notifications and supports advanced features like Monte Carlo dropout inference and partitioned dataset handling.

    Attributes:
        model_path (ModelPathManager): Manager for model file paths and artifact directories.
        wandb_notifications (bool): Enables or disables Weights & Biases notifications.
        use_prediction_store (bool): Enables or disables the prediction store for caching predictions.

    Methods:
        __init__(model_path, wandb_notifications=True, use_prediction_store=True):
            Initializes the model manager, overrides torch.load globally, and logs the current model architecture.

        _train_model_artifact():
            Trains a forecasting model using the configured dataset and algorithm, saves the trained model artifact, and returns the trained forecaster.

        _evaluate_model_artifact(eval_type, artifact_name=None):
            Evaluates a model artifact for a specified evaluation type, optionally using a specific artifact. Returns a list of prediction DataFrames for each evaluation sequence.

        _forecast_model_artifact(artifact_name):
            Loads a model artifact and generates forecasts using the current configuration. Returns a DataFrame of forecasted predictions.

        _evaluate_sweep(eval_type, model):
            Evaluates the model over a sweep of sequence numbers and returns a list of predictions.

    Notes:
        - Supports automatic artifact selection based on run type and timestamp extraction.
        - Integrates with custom dataset and model catalog classes for flexible configuration.
        - Provides options for feature and target scaling, parallel prediction jobs, and Monte Carlo inference.
    """

    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        """
        Initializes the model manager with the specified configuration.

        Args:
            model_path (ModelPathManager): Manager for model file paths.
            wandb_notifications (bool, optional): Enable or disable Weights & Biases notifications on Slack. Defaults to False.
            use_prediction_store (bool, optional): Enable or disable the prediction store. Defaults to False.
        Side Effects:
            Overrides the global torch.load function with custom_torch_load.
            Logs the current model architecture.

        """
        super().__init__(
            model_path=model_path,
            wandb_notifications=wandb_notifications,
            use_prediction_store=use_prediction_store,
        )
        # Override torch.load globally
        torch.load = custom_torch_load
        logger.info(
            f"Current model architecture: \033[92m{self.configs['algorithm']}\033[0m"
        )

    def _train_model_artifact(self):
        """
        Trains a forecasting model using the specified configuration and dataset, and saves the trained model artifact.

        This method performs the following steps:
        1. Loads the raw dataset based on the configured run type.
        2. Initializes the model object from the model catalog using the specified algorithm.
        3. Constructs a DartsForecaster with the dataset, model, partitioning information, and optional scalers.
        4. Trains the forecaster.
        5. If not running a sweep, saves the trained model artifact to the designated artifacts directory.

        Returns:
            DartsForecaster: The trained forecaster object.
        """
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
            log_features=self.config.get("log_features", []),
            log_targets=self.config.get("log_targets", False),
            model=model_object,
            partition_dict=self._data_loader.partition_dict,
            feature_scaler=self.config.get("feature_scaler", None),
            target_scaler=self.config.get("target_scaler", None),
            feature_scaler_map=self.config.get("feature_scaler_map", None),
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
        """
        Evaluates a model artifact based on the specified evaluation type and artifact name.

        Parameters
        ----------
        eval_type : str
            The type of evaluation to perform. Can be one of "standard", "long", "complete", or "live".
        artifact_name : str, optional
            The name of the specific model artifact to use for evaluation. If not provided, the latest artifact
            corresponding to the current run type will be used.

        Returns
        -------
        list of pandas.DataFrame
            A list of prediction DataFrames, one for each evaluation sequence number.

        Notes
        -----
        - Updates the configuration with the timestamp extracted from the artifact name.
        - Loads the relevant dataset and model, and performs predictions for the specified evaluation type.
        - Supports additional configuration options such as number of samples, jobs, and dropout for Monte Carlo inference.
        - Predictions are generated in parallel while maintaining sequence order.
        """
        import concurrent.futures
        
        # eval_type can be "standard", "long", "complete", "live"
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        run_type = self.configs["run_type"]

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

        self._config_manager.add_config({"timestamp": path_artifact.stem[-15:]})

        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        model_object = ModelCatalog(config=self.configs).get_model(
            model_name=self.configs["algorithm"]
        )
        forecaster = DartsForecaster(
            dataset=_ViewsDatasetDarts(
                source=df_viewser,
                targets=self.configs.get("targets"),
                broadcast_features=True,
            ),
            model=model_object,
            partition_dict=self._data_loader.partition_dict,
            feature_scaler=self.configs.get("feature_scaler", None),
            target_scaler=self.configs.get("target_scaler", None),
            log_targets=self.configs.get("log_targets", False),
            log_features=self.configs.get("log_features", []),
            feature_scaler_map=self.configs.get("feature_scaler_map", None),
        )
        forecaster.load_model(path=path_artifact)

        total_sequence_number = 12

        # Parallel prediction with order preservation
        def predict_sequence(sequence_number):
            """Helper function to predict a single sequence."""
            logger.info(f"Starting prediction for sequence {sequence_number + 1}/{total_sequence_number}")
            result = forecaster.predict(
                sequence_number,
                max(self.configs["steps"]),
                num_samples=self.configs.get("num_samples"),
                n_jobs=self.configs.get("n_jobs", 1),
                mc_dropout=self.configs.get("mc_dropout"),
            )
            logger.info(f"✓ Completed prediction for sequence {sequence_number + 1}/{total_sequence_number}")
            return result

        # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound
        max_workers = self.configs.get("parallel_workers", None)
        
        logger.info(f"Starting parallel prediction with {max_workers} workers for {total_sequence_number} sequences")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Submit all tasks and maintain order
            futures = {
                executor.submit(predict_sequence, seq_num): seq_num
                for seq_num in range(total_sequence_number)
            }
            
            # Track completed futures and maintain order
            df_predictions = [None] * total_sequence_number
            completed = 0
            
            for future in concurrent.futures.as_completed(futures):
                seq_num = futures[future]
                try:
                    df_predictions[seq_num] = future.result()
                    completed += 1
                    logger.info(f"Progress: {completed}/{total_sequence_number} sequences completed")
                except Exception as e:
                    logger.error(f"Sequence {seq_num + 1} failed with error: {e}")
                    raise

        logger.info(f"All {total_sequence_number} predictions completed successfully")
        return df_predictions

    def _forecast_model_artifact(self, artifact_name):
        """
        Loads a model artifact and generates forecasts using the specified configuration.

        Args:
            artifact_name (str): The name of the model artifact to use. If None, uses the latest artifact based on run type.

        Returns:
            pd.DataFrame: DataFrame containing the forecasted predictions.

        Logs:
            Information about the artifact being used (default or specified).

        Side Effects:
            Updates self.config["timestamp"] with the timestamp extracted from the artifact path.
        """
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
            feature_scaler=self.config.get("feature_scaler", None),
            target_scaler=self.config.get("target_scaler", None),
            log_targets=self.config.get("log_targets", False),
            log_features=self.config.get("log_features", []),
            feature_scaler_map=self.config.get("feature_scaler_map", None),
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
        """
        Evaluates the model over a sweep of sequence numbers and returns predictions.

        Args:
            eval_type (str): The type of evaluation to perform, used to resolve the total number of sequences.
            model (any): The forecasting model instance with a `predict` method.

        Returns:
            list: A list of predictions generated by the model for each sequence number in the sweep.
        """

        # total_sequence_number = (
        #     ForecastingModelManager._resolve_evaluation_sequence_number(eval_type)
        # )
        logger.warning(
            "Using fixed total_sequence_number=12 for sweep evaluation eval_type will soon be deprecated."
        )
        total_sequence_number = 12

        df_predictions = [
            model.predict(sequence_number, max(self.config["steps"]))
            for sequence_number in range(total_sequence_number)
        ]

        return df_predictions
