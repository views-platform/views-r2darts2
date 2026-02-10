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
from views_r2darts2.utils.gates import ReproducibilityGate, ReproducibilityError

logger = logging.getLogger(__name__)

# Save the original torch.load function ONLY once to avoid recursion or mock-poisoning.
# If it's already patched (e.g. during tests), we don't want to re-save the patch.
if not hasattr(torch, "__original_load__"):
    # Priority 1: Check session-captured CLEAN_TORCH_LOAD (for tests)
    try:
        from tests.conftest import CLEAN_TORCH_LOAD
        torch.__original_load__ = CLEAN_TORCH_LOAD
    except (ImportError, ModuleNotFoundError):
        # Priority 2: Use current torch.load if it's not a Mock
        orig = torch.load
        if "Mock" not in str(type(orig)):
            torch.__original_load__ = orig
        else:
            # Last resort: we are in a mock-contaminated environment and conftest didn't help
            torch.__original_load__ = orig

# https://github.com/suno-ai/bark/pull/619#issuecomment-2726747073
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
    
    # Use the saved original
    return torch.__original_load__(*args, **kwargs)

custom_torch_load.monkeypatched = True


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
        if not hasattr(torch, "__original_load__"):
            torch.__original_load__ = torch.load
        torch.load = custom_torch_load
        logger.info(
            f"Current model architecture: \033[92m{self.configs['algorithm']}\033[0m"
        )

    def _resolve_active_partition_dict(self, config: dict) -> dict:
        """
        Explicitly resolves the partition dictionary for the current run.
        
        This avoids the 'Stale DataLoader' bug by re-calculating the 
        temporal windows based on the actual 'steps' in the active config.
        
        Args:
            config: Captured configuration snapshot.
            
        Returns:
            Dictionary containing 'train' and 'test' time ranges.
            
        Raises:
            KeyError: If run_type or steps are missing.
            ValueError: If the partition type is unsupported or discontinuous.
        """
        run_type = config.get("run_type")
        steps_list = config.get("steps")
        
        if not run_type or steps_list is None:
            raise KeyError(f"Cannot resolve partition: Missing 'run_type' or 'steps' in config.")
            
        if not isinstance(steps_list, list):
            raise TypeError(f"Config parameter 'steps' must be a list, got {type(steps_list).__name__}.")

        # SIREN: Horizon and Shift Checks
        ReproducibilityGate.Config.audit_architecture(config)
        
        # Get the master partition dict (defined in config_partitions.py)
        master_partitions = getattr(self, "_partition_dict", {})
        
        if run_type in master_partitions:
            partition = master_partitions[run_type]
        else:
            # Fallback to parent logic for dynamic partitions (like 'forecasting')
            if hasattr(self._data_loader, "_get_partition_dict"):
                self._data_loader.partition = run_type
                partition = self._data_loader._get_partition_dict(steps=len(steps_list))
            else:
                raise ValueError(f"Unsupported run_type for partition resolution: {run_type}")

        # GUARDIAN: The Continuity Check (t+1)
        ReproducibilityGate.Temporal.audit_continuity(partition)

        return partition

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
        # Capture stable config snapshot
        active_config = self.configs
        
        # DNA AUDIT: Verify mandatory hyperparameters
        ReproducibilityGate.Config.audit_manifest(active_config)
        ReproducibilityGate.Config.audit_architecture(active_config)
        
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        run_type = active_config["run_type"]

        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )
        # Partitioner dict from ViewsDataLoader
        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )

        # Explicitly resolve the partition for THIS specific run
        current_partition = self._resolve_active_partition_dict(active_config)
        logger.info(f"Training on partition [{run_type}]: {current_partition}")

        forecaster = DartsForecaster(
            dataset=_ViewsDatasetDarts(
                source=df_viewser,
                targets=active_config.get("targets"),
                broadcast_features=True,
            ),
            log_features=active_config.get("log_features", []),
            log_targets=active_config.get("log_targets", False),
            model=model_object,
            partition_dict=current_partition,
            feature_scaler=active_config.get("feature_scaler", None),
            target_scaler=active_config.get("target_scaler", None),
            feature_scaler_map=active_config.get("feature_scaler_map", None),
        )
        forecaster.train()

        if not active_config["sweep"]:  # If not using wandb sweep
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
        
        # Capture stable config snapshot
        active_config = self.configs
        
        # DNA AUDIT: Verify mandatory hyperparameters
        ReproducibilityGate.Config.audit_manifest(active_config)
        
        run_type = active_config["run_type"]

        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts

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

        # Persist timestamp to the underlying manager
        timestamp = path_artifact.stem[-15:]
        self._config_manager.add_config({"timestamp": timestamp})
        # Refresh snapshot to include the timestamp
        active_config = self.configs

        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        forecaster = DartsForecaster(
            dataset=_ViewsDatasetDarts(
                source=df_viewser,
                targets=active_config.get("targets"),
                broadcast_features=True,
            ),
            model=model_object,
            partition_dict=self._resolve_active_partition_dict(active_config),
            feature_scaler=active_config.get("feature_scaler", None),
            target_scaler=active_config.get("target_scaler", None),
            log_targets=active_config.get("log_targets", False),
            log_features=active_config.get("log_features", []),
            feature_scaler_map=active_config.get("feature_scaler_map", None),
        )
        forecaster.load_model(path=path_artifact)

        total_sequence_number = 12
        
        # HORIZON LOCKDOWN: Prevent forecasting beyond ground truth
        partition = self._resolve_active_partition_dict(active_config)
        ReproducibilityGate.Temporal.audit_prediction_horizon(
            run_type=run_type,
            train_end=partition["train"][1],
            test_end=partition["test"][1],
            max_steps=max(active_config["steps"]),
            total_sequences=total_sequence_number
        )

        predict_kwargs = self._get_predict_kwargs(active_config)

        # Parallel prediction with order preservation
        def predict_sequence(sequence_number):
            """Helper function to predict a single sequence."""
            logger.info(f"Starting prediction for sequence {sequence_number + 1}/{total_sequence_number}")
            result = forecaster.predict(
                sequence_number,
                max(active_config["steps"]),
                **predict_kwargs
            )
            logger.info(f"✓ Completed prediction for sequence {sequence_number + 1}/{total_sequence_number}")
            return result

        # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound
        max_workers = active_config.get("parallel_workers", None)
        
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
        """
        # Capture stable config snapshot
        active_config = self.configs
        
        # DNA AUDIT: Verify mandatory hyperparameters
        ReproducibilityGate.Config.audit_manifest(active_config)
        ReproducibilityGate.Config.audit_architecture(active_config)
        
        run_type = active_config["run_type"]

        # Commonly used paths
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts

        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact"
            )
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        # Update the underlying manager's state
        timestamp = path_artifact.stem[-15:]
        self._config_manager.add_config({"timestamp": timestamp})
        # Refresh snapshot
        active_config = self.configs

        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        forecaster = DartsForecaster(
            dataset=_ViewsDatasetDarts(
                source=df_viewser,
                targets=active_config.get("targets"),
                broadcast_features=True,
            ),
            model=model_object,
            partition_dict=self._resolve_active_partition_dict(active_config),
            feature_scaler=active_config.get("feature_scaler", None),
            target_scaler=active_config.get("target_scaler", None),
            log_targets=active_config.get("log_targets", False),
            log_features=active_config.get("log_features", []),
            feature_scaler_map=active_config.get("feature_scaler_map", None),
        )
        forecaster.load_model(path=path_artifact)

        predict_kwargs = self._get_predict_kwargs(active_config)

        df_predictions = forecaster.predict(
            0,
            max(active_config["steps"]),
            **predict_kwargs
        )

        return df_predictions

    def _execute_model_sweeping(self) -> None:
        """
        Execute single sweep iteration.
        """
        import wandb
        from views_pipeline_core.exceptions.exceptions import PipelineException

        with self._wandb_module.initialize_run(
            project=self._project,
            config=None,  # Will be set by wandb.config
            job_type="sweep",
        ):
            try:
                # Update config for sweep run using config_manager
                self._config_manager.update_for_sweep_run(
                    wandb.config,
                    self.args,
                    wandb_module=self._wandb_module,
                )
                
                active_config = self.configs
                
                # DNA AUDIT: Verify mandatory hyperparameters
                ReproducibilityGate.Config.audit_manifest(active_config)
                ReproducibilityGate.Config.audit_architecture(active_config)

                logger.info(f"Sweeping {self._model_path.target} {active_config['name']}...")
                model = self._train_model_artifact()

                self._wandb_module.send_alert(
                    title=f"Training for {self._model_path.target} {active_config['name']} completed successfully.",
                    text=f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n{wandb.config}\n```",
                    notifications_enabled=self._wandb_notifications,
                )

                logger.info(
                    f"Evaluating {self._model_path.target} {active_config['name']}..."
                )
                
                # HORIZON LOCKDOWN: Prevent forecasting beyond ground truth
                partition = self._resolve_active_partition_dict(active_config)
                ReproducibilityGate.Temporal.audit_prediction_horizon(
                    run_type=active_config["run_type"],
                    train_end=partition["train"][1],
                    test_end=partition["test"][1],
                    max_steps=max(active_config["steps"]),
                    total_sequences=12
                )
                
                df_predictions = self._evaluate_sweep(self._eval_type, model)

                for i, df in enumerate(df_predictions):
                    print(
                        f"\nValidating evaluation dataframe of sequence {i+1}/{len(df_predictions)}"
                    )
                    from views_pipeline_core.modules.validation.model import (
                        validate_prediction_dataframe,
                    )

                    validate_prediction_dataframe(
                        dataframe=df, target=active_config["targets"]
                    )

                if active_config.get("metrics"):
                    self._evaluate_prediction_dataframe(df_predictions, self._eval_type)
                else:
                    raise PipelineException("No evaluation metrics specified in config_meta.py")
            finally:
                self._wandb_module.finish_run()

    def _evaluate_sweep(self, eval_type: str, model: any):
        """
        Evaluates the model over a sweep of sequence numbers and returns predictions.

        Args:
            eval_type (str): The type of evaluation to perform, used to resolve the total number of sequences.
            model (any): The forecasting model instance with a `predict` method.

        Returns:
            list: A list of predictions generated by the model for each sequence number in the sweep.
        """
        # Snapshot the config once for the duration of evaluation
        active_config = self.configs
        
        logger.warning(
            "Using fixed total_sequence_number=12 for sweep evaluation eval_type will soon be deprecated."
        )
        total_sequence_number = 12
        
        # Explicitly extract kwargs to ensure reproducibility
        predict_kwargs = self._get_predict_kwargs(active_config)

        df_predictions = [
            model.predict(
                sequence_number, 
                max(active_config["steps"]),
                **predict_kwargs
            )
            for sequence_number in range(total_sequence_number)
        ]

        return df_predictions

    def _get_predict_kwargs(self, config: dict) -> dict:
        """
        Extracts and validates keyword arguments for the predict() method.
        
        Args:
            config: Configuration dictionary snapshot.
            
        Returns:
            Dictionary of keyword arguments for Darts predict().
            
        Raises:
            ValueError: If mandatory parameters are missing.
        """
        mandatory = ["num_samples", "mc_dropout", "n_jobs"]
        missing = [k for k in mandatory if k not in config]
        if missing:
            raise ValueError(
                f"Missing mandatory prediction parameters in config: {missing}. "
                "Explicit configuration is required for reproducibility."
            )
            
        return {
            "num_samples": config["num_samples"],
            "mc_dropout": config["mc_dropout"],
            "n_jobs": config["n_jobs"],
        }
