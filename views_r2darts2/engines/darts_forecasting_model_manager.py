import logging
import torch  # noqa: F401
from views_r2darts2.transformers.views_dataset_darts import _ViewsDatasetDarts
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_pipeline_core.files.utils import generate_model_file_name
from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager

from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate
from views_r2darts2.infrastructure.patches import apply_all_patches

logger = logging.getLogger(__name__)


class DartsForecastingModelManager(ForecastingModelManager):
    """
    Manages the lifecycle of Darts-based forecasting models, including training, evaluation, and artifact management.

    Intent Contract:
        - Purpose: Orchestrate the transition from raw VIEWS dataframes to persistent model artifacts and
          validated evaluation results, acting as the primary entry point for experiment execution.
        - Non-Goals: Does not define model architectures or implement core tensor math.
        - Guarantees:
            - Ensures every execution context is audited against the DNA manifest before state mutation.
            - Guarantees that temporal boundaries (t+1) are strictly enforced across train/test splits.
            - Ensures model artifacts (weights + scalers) are saved coupled together.
        - Failure Behavior: Fails loudly during the "Handshake" phase if configurations are incomplete
          or if predictions are attempted into the void beyond known ground truth.
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
        # Initialize all required monkey-patches
        apply_all_patches()
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
            raise KeyError(
                "Cannot resolve partition: Missing 'run_type' or 'steps' in config."
            )

        if not isinstance(steps_list, list):
            raise TypeError(
                f"Config parameter 'steps' must be a list, got {type(steps_list).__name__}."
            )

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
                raise ValueError(
                    f"Unsupported run_type for partition resolution: {run_type}"
                )

        # GUARDIAN: The Continuity Check (t+1)
        ReproducibilityGate.Temporal.audit_continuity(partition)

        return partition

    @staticmethod
    def _resolve_total_sequence_number(partition: dict, max_steps: int) -> int:
        """
        Derive the total number of rolling-origin sequences from the test partition.

        The count equals `test_len - max_steps + 1`, the standard pipeline contract
        for rolling-origin evaluation. Guards against the silent-failure mode where
        `max_steps > test_len` yields zero or negative sequences, which Python would
        silently accept (`[None] * -1 == []`, `range(-1) == []`) and propagate as an
        empty prediction batch through downstream evaluation.

        Args:
            partition: Resolved partition dict containing a `test` key with a
                (test_start, test_end) tuple (inclusive, both ends).
            max_steps: Maximum forecast horizon — typically `max(config['steps'])`.

        Returns:
            Number of rolling-origin sequences (always >= 1).

        Raises:
            ValueError: If the test partition is shorter than `max_steps`, which
                would otherwise produce an empty evaluation batch with no error.
        """
        test_start, test_end = partition["test"]
        test_len = test_end - test_start + 1
        if test_len < max_steps:
            raise ValueError(
                f"Invalid evaluation configuration: test partition length "
                f"({test_len}) is smaller than the maximum forecast horizon "
                f"({max_steps}). Rolling-origin evaluation requires "
                f"test_len >= max(steps); otherwise no sequences can be produced."
            )
        return test_len - max_steps + 1

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

        dataset = _ViewsDatasetDarts.from_views_path(
            path_raw=path_raw, run_type=run_type, config=active_config,
            cached_path=None,
        )

        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )

        # Explicitly resolve the partition for THIS specific run
        current_partition = self._resolve_active_partition_dict(active_config)
        logger.info(f"Training on partition [{run_type}]: {current_partition}")

        forecaster = DartsForecaster(
            dataset=dataset,
            log_features=active_config.get("log_features", []),
            log_targets=active_config.get("log_targets", False),
            model=model_object,
            partition_dict=current_partition,
            feature_scaler=active_config.get("feature_scaler", None),
            target_scaler=active_config.get("target_scaler", None),
            feature_scaler_map=active_config.get("feature_scaler_map", None),
            random_state=active_config["random_state"],
            static_covariate_stats=(
                active_config.get("static_covariate_stats", None)
                if active_config.get("use_static_covariates", False)
                else None
            ),
            checkpoint_mode=active_config.get("checkpoint_mode", "best"),
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

        dataset = _ViewsDatasetDarts.from_views_path(
            path_raw=path_raw, run_type=run_type, config=active_config,
            cached_path=None,
        )

        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        forecaster = DartsForecaster(
            dataset=dataset,
            model=model_object,
            partition_dict=self._resolve_active_partition_dict(active_config),
            feature_scaler=active_config.get("feature_scaler", None),
            target_scaler=active_config.get("target_scaler", None),
            log_targets=active_config.get("log_targets", False),
            log_features=active_config.get("log_features", []),
            feature_scaler_map=active_config.get("feature_scaler_map", None),
            random_state=active_config["random_state"],
            static_covariate_stats=(
                active_config.get("static_covariate_stats", None)
                if active_config.get("use_static_covariates", False)
                else None
            ),
        )
        forecaster.load_model(path=path_artifact)

        partition = self._resolve_active_partition_dict(active_config)
        _time_steps = max(active_config["steps"])
        total_sequence_number = self._resolve_total_sequence_number(
            partition, _time_steps
        )

        # HORIZON LOCKDOWN: Prevent forecasting beyond ground truth
        ReproducibilityGate.Temporal.audit_prediction_horizon(
            run_type=run_type,
            train_end=partition["train"][1],
            test_end=partition["test"][1],
            max_steps=_time_steps,
            total_sequences=total_sequence_number,
        )

        predict_kwargs = self._get_predict_kwargs(active_config)

        # Parallel prediction with order preservation
        def predict_sequence(sequence_number):
            """Helper function to predict a single sequence."""
            logger.info(
                f"Starting prediction for sequence {sequence_number + 1}/{total_sequence_number}"
            )
            result = forecaster.predict(
                sequence_number, max(active_config["steps"]), **predict_kwargs
            )
            logger.info(
                f"✓ Completed prediction for sequence {sequence_number + 1}/{total_sequence_number}"
            )
            return result

        # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound
        # FORCE SEQUENTIAL FOR GPU: Darts moves models to CPU in teardown(), which causes
        # race conditions in multi-threaded GPU inference.
        if forecaster.device == "cpu":
            max_workers = active_config.get("parallel_workers", 1)
        else:
            logger.info(
                "GPU detected: forcing sequential prediction to avoid device-shifting race conditions."
            )
            max_workers = 1

        logger.info(
            f"Starting parallel prediction with {max_workers} workers for {total_sequence_number} sequences"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                    logger.info(
                        f"Progress: {completed}/{total_sequence_number} sequences completed"
                    )
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

        dataset = _ViewsDatasetDarts.from_views_path(
            path_raw=path_raw, run_type=run_type, config=active_config,
            cached_path=None,
        )

        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        forecaster = DartsForecaster(
            dataset=dataset,
            model=model_object,
            partition_dict=self._resolve_active_partition_dict(active_config),
            feature_scaler=active_config.get("feature_scaler", None),
            target_scaler=active_config.get("target_scaler", None),
            log_targets=active_config.get("log_targets", False),
            log_features=active_config.get("log_features", []),
            feature_scaler_map=active_config.get("feature_scaler_map", None),
            random_state=active_config["random_state"],
            static_covariate_stats=(
                active_config.get("static_covariate_stats", None)
                if active_config.get("use_static_covariates", False)
                else None
            ),
        )
        forecaster.load_model(path=path_artifact)

        predict_kwargs = self._get_predict_kwargs(active_config)

        df_predictions = forecaster.predict(
            0, max(active_config["steps"]), **predict_kwargs
        )

        return df_predictions

    def _execute_model_sweeping(self) -> None:
        """
        Execute single sweep iteration.
        """
        import wandb
        from views_pipeline_core.exceptions.exceptions import PipelineException
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            CorePredictionSniffer,
        )

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

                logger.info(
                    f"Sweeping {self._model_path.target} {active_config['name']}..."
                )
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
                _max_steps = max(active_config["steps"])
                ReproducibilityGate.Temporal.audit_prediction_horizon(
                    run_type=active_config["run_type"],
                    train_end=partition["train"][1],
                    test_end=partition["test"][1],
                    max_steps=_max_steps,
                    total_sequences=self._resolve_total_sequence_number(
                        partition, _max_steps
                    ),
                )

                df_predictions = self._evaluate_sweep(self._eval_type, model)

                sniffer = CorePredictionSniffer(level=active_config["level"])
                for i, df in enumerate(df_predictions):
                    logger.info(
                        f"Validating evaluation dataframe of sequence {i + 1}/{len(df_predictions)}"
                    )
                    sniffer.sniff_predictions(df, targets=active_config["targets"])

                if self._has_evaluation_metrics():
                    self._evaluate_prediction_dataframe(df_predictions, self._eval_type)
                else:
                    raise PipelineException(
                        "No evaluation metrics specified in config_meta.py"
                    )
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

        partition = self._resolve_active_partition_dict(active_config)
        _time_steps = max(active_config["steps"])
        total_sequence_number = self._resolve_total_sequence_number(
            partition, _time_steps
        )

        # Explicitly extract kwargs to ensure reproducibility
        predict_kwargs = self._get_predict_kwargs(active_config)

        df_predictions = [
            model.predict(
                sequence_number, max(active_config["steps"]), **predict_kwargs
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
        mandatory = ["num_samples", "mc_dropout"]
        missing = [k for k in mandatory if k not in config]
        if missing:
            raise ValueError(
                f"Missing mandatory prediction parameters in config: {missing}. "
                "Explicit configuration is required for reproducibility."
            )

        return {
            "num_samples": config["num_samples"],
            "mc_dropout": config["mc_dropout"],
        }
