"""Darts forecasting model manager (orchestrator, pandas-free).

Inherits from :class:`views_pipeline_core.managers.model.ForecastingModelManager`
and overrides the template methods that build / evaluate / forecast model
artifacts. The parent class is imported lazily so that the rest of the package
remains importable in environments without ``views_pipeline_core`` (e.g. the
unit-test environment).

This module is pandas-free. The manager consumes :class:`PredictionFrame`
outputs from :meth:`DartsForecaster.predict` and forwards them to the parent's
``_evaluate_prediction_dataframe`` for scoring.

Key cleanup vs. the legacy implementation:
    * The 17-argument ``DartsForecaster(...)`` construction is extracted into a
      single :meth:`_build_forecaster` factory (was duplicated 3×).
    * ``import wandb`` moved to a single lazy import inside
      :meth:`_execute_model_sweeping`.
    * The ``CorePredictionSniffer`` and ``PipelineException`` imports remain
      lazy (they live in ``views_pipeline_core``).


"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import torch  # noqa: F401 — kept alive for the monkey-patches in apply_all_patches

from views_frames import PredictionFrame

from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.data.views_dataset import ViewsDatasetDarts
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.infrastructure.patches import apply_all_patches
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate
from views_r2darts2.transformers.darts_bridge import prediction_frames_to_dataframe

# Lazy import: the parent class lives in views_pipeline_core, which is not
# importable in every environment (e.g. the unit-test sandbox). The lazy
# import keeps this module importable for tests that only need
# ``_resolve_total_sequence_number``.
def _import_parent_class():
    """Lazily import the parent class + helpers from views_pipeline_core."""
    from views_pipeline_core.files.utils import generate_model_file_name
    from views_pipeline_core.managers.model import (
        ForecastingModelManager,
        ModelPathManager,
    )
    return ForecastingModelManager, ModelPathManager, generate_model_file_name


logger = logging.getLogger(__name__)


def _get_parent_class():
    """Return the parent class, importing it lazily on first use."""
    return _import_parent_class()[0]


# At import time, attempt to bind the parent class. If views_pipeline_core is
# not installed, the class is set to ``object`` and the manager is importable
# but cannot be instantiated (calling ``__init__`` will raise ImportError).
try:
    _PARENT_CLASS, _MODEL_PATH_MANAGER, _GENERATE_FILE_NAME = _import_parent_class()
except ImportError:  # pragma: no cover - depends on env
    _PARENT_CLASS = object  # type: ignore[assignment]
    _MODEL_PATH_MANAGER = None  # type: ignore[assignment]
    _GENERATE_FILE_NAME = None  # type: ignore[assignment]
    logger.warning(
        "views_pipeline_core is not installed — DartsForecastingModelManager "
        "will inherit from object. Training/evaluation/forecasting entry "
        "points will raise ImportError at call time. The static helpers "
        "(_resolve_total_sequence_number, _get_predict_kwargs) remain usable."
    )


class DartsForecastingModelManager(_PARENT_CLASS):  # type: ignore[misc, valid-type]
    """Manages the lifecycle of Darts-based forecasting models.

    Intent Contract:
        - Purpose: Orchestrate the transition from raw VIEWS dataframes to
          persistent model artifacts and validated evaluation results, acting
          as the primary entry point for experiment execution.
        - Non-Goals: Does not define model architectures or implement core
          tensor math.
        - Guarantees:
            - Every execution context is audited against the DNA manifest
              before state mutation.
            - Temporal boundaries (t+1) are strictly enforced across
              train/test splits.
            - Model artifacts (weights + scalers) are saved coupled together.
        - Failure Behavior: Fails loudly during the "Handshake" phase if
            configurations are incomplete or if predictions are attempted
            beyond known ground truth.
    """

    def __init__(
        self,
        model_path: Any,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        """Initialize the model manager.

        Args:
            model_path: A :class:`ModelPathManager` (from views_pipeline_core).
            wandb_notifications: Enable Weights & Biases Slack notifications.
            use_prediction_store: Enable the prediction store.

        Raises:
            ImportError: ``views_pipeline_core`` is not installed.
        """
        if _PARENT_CLASS is object:
            raise ImportError(
                "views_pipeline_core is not installed — "
                "DartsForecastingModelManager cannot be instantiated. "
                "Install views_pipeline_core or use the static helpers directly."
            )
        super().__init__(
            model_path=model_path,
            wandb_notifications=wandb_notifications,
            use_prediction_store=use_prediction_store,
        )
        # Apply all Darts monkey-patches once per manager instance.
        apply_all_patches()
        logger.info(
            "Current model architecture: %s", self.configs["algorithm"]
        )

    # ------------------------------------------------------------------ partition resolution

    def _resolve_active_partition_dict(self, config: Mapping[str, Any]) -> dict:
        """Explicitly resolve the partition dict for the current run.

        Avoids the "Stale DataLoader" bug by re-calculating the temporal
        windows based on the actual ``steps`` in the active config.

        Args:
            config: Captured configuration snapshot.

        Returns:
            ``{"train": (start, end), "test": (start, end)}``.

        Raises:
            KeyError: ``run_type`` or ``steps`` is missing.
            TypeError: ``steps`` is not a list.
            ValueError: ``run_type`` is unsupported.
        """
        run_type = config.get("run_type")
        steps_list = config.get("steps")
        if not run_type or steps_list is None:
            raise KeyError(
                "Cannot resolve partition: Missing 'run_type' or 'steps' in config."
            )
        if not isinstance(steps_list, list):
            raise TypeError(
                f"Config parameter 'steps' must be a list, got "
                f"{type(steps_list).__name__}."
            )

        # SIREN: horizon and shift checks.
        ReproducibilityGate.Config.audit_architecture(config)

        master_partitions = getattr(self, "_partition_dict", {})
        if run_type in master_partitions:
            partition = master_partitions[run_type]
        else:
            # Fallback to parent logic for dynamic partitions (e.g. forecasting).
            if hasattr(self._data_loader, "_get_partition_dict"):
                self._data_loader.partition = run_type
                partition = self._data_loader._get_partition_dict(
                    steps=len(steps_list)
                )
            else:
                raise ValueError(
                    f"Unsupported run_type for partition resolution: {run_type}"
                )

        # GUARDIAN: the continuity check (t+1).
        ReproducibilityGate.Temporal.audit_continuity(partition)
        return partition

    @staticmethod
    def _resolve_total_sequence_number(partition: dict, max_steps: int) -> int:
        """Derive the total number of rolling-origin sequences from the test partition.

        Equals ``test_len - max_steps + 1`` (the standard rolling-origin
        contract). Guards against the silent-failure mode where
        ``max_steps > test_len`` would yield zero or negative sequences.

        Args:
            partition: Resolved partition dict with a ``"test"`` key.
            max_steps: Maximum forecast horizon (typically
                ``max(config['steps'])``).

        Returns:
            Number of rolling-origin sequences (always >= 1).

        Raises:
            ValueError: The test partition is shorter than ``max_steps``.
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

    # ------------------------------------------------------------------ forecaster factory

    def _build_forecaster(
        self,
        *,
        active_config: Mapping[str, Any],
        partition: Mapping[str, Any],
        dataset: ViewsDatasetDarts,
        model_object: Any,
        checkpoint_mode: str | None = None,
    ) -> DartsForecaster:
        kwargs: dict[str, Any] = dict(
            dataset=dataset,
            model=model_object,
            partition_dict=dict(partition),
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
            use_cyclic_encoders=active_config.get("use_cyclic_encoders", False),
        )
        if checkpoint_mode is not None:
            kwargs["checkpoint_mode"] = checkpoint_mode
        return DartsForecaster(**kwargs)

    # ------------------------------------------------------------------ train

    def _get_prediction_format(self) -> str:
        """Return the configured prediction output format.

        Reads ``self.configs["prediction_format"]``. Two values are supported:

            * ``"dataframe"`` (default): ``_evaluate_model_artifact`` returns a
              ``list[pd.DataFrame]`` (one DataFrame per rolling-origin
              sequence), compatible with the legacy
              ``views_pipeline_core`` evaluation pipeline.
            * ``"prediction_frame"``: ``_evaluate_model_artifact`` returns a
              ``dict[str, list[PredictionFrame]]`` (target → list of frames,
              one per sequence), compatible with the new streaming evaluation
              pipeline that consumes ``PredictionFrame`` objects directly.
        """
        return self.configs.get("prediction_format", "dataframe")

    def _predictions_to_dataframe(self, predictions: dict) -> Any:
        """Convert ``dict[str, PredictionFrame]`` → pandas DataFrame.

        ``DartsForecaster.predict`` returns ``dict[str, PredictionFrame]``
        (pandas-free), but the parent ``views_pipeline_core`` manager expects
        a list of pandas DataFrames with a ``(time_id, entity_id)`` MultiIndex
        when ``prediction_format == "dataframe"``. This helper bridges the gap
        using :func:`prediction_frames_to_dataframe` (the only pandas
        touchpoint, confined to ``transformers/darts_bridge.py``).
        """
        active_config = self.configs
        time_id = active_config.get("time_id", "month_id")

        # Entity-id resolution is level-driven by config.
        level = active_config.get("level")
        if not isinstance(level, str):
            raise ValueError(
                "Missing required config['level'] for dataframe conversion. "
                "Expected 'cm' or 'pgm'."
            )

        level_norm = level.lower()
        if level_norm == "pgm":
            entity_id = "priogrid_id"
        elif level_norm == "cm":
            entity_id = "country_id"
        else:
            raise ValueError(f"Unsupported level for entity_id resolution: {level}")
        
        return prediction_frames_to_dataframe(
            predictions=predictions,
            time_id=time_id,
            entity_id=entity_id,
        )

    @staticmethod
    def _transpose_predictions(
        per_sequence_preds: list[dict[str, PredictionFrame]],
    ) -> dict[str, list[PredictionFrame]]:
        """Transpose ``list[dict[str, PredictionFrame]]`` → ``dict[str, list[PredictionFrame]]``.

        ``DartsForecaster.predict`` returns one ``dict[str, PredictionFrame]``
        per rolling-origin sequence. The ``prediction_format="prediction_frame"``
        contract expects the inverse shape: a dict keyed by target name, where
        each value is the list of per-sequence frames for that target.
        """
        if not per_sequence_preds:
            return {}
        target_names = list(per_sequence_preds[0].keys())
        return {
            tgt: [seq_preds[tgt] for seq_preds in per_sequence_preds]
            for tgt in target_names
        }

    def _format_eval_predictions(
        self, per_sequence_preds: list[dict[str, PredictionFrame]]
    ) -> Any:
        """Format per-sequence predictions for the parent manager.

        Args:
            per_sequence_preds: A list of ``{target: PredictionFrame}`` dicts,
                one per rolling-origin sequence (the raw output of
                ``DartsForecaster.predict``).

        Returns:
            * ``list[pd.DataFrame]`` when ``prediction_format == "dataframe"``.
            * ``dict[str, list[PredictionFrame]]`` when
              ``prediction_format == "prediction_frame"``.
        """
        fmt = self._get_prediction_format()
        if fmt == "prediction_frame":
            return self._transpose_predictions(per_sequence_preds)
        # Default: dataframe mode.
        return [self._predictions_to_dataframe(preds) for preds in per_sequence_preds]

    def _format_forecast_predictions(
        self, predictions: dict[str, PredictionFrame]
    ) -> Any:
        """Format a single forecast's predictions for the parent manager.

        Args:
            predictions: A ``{target: PredictionFrame}`` dict (the raw output
                of a single ``DartsForecaster.predict`` call).

        Returns:
            * ``pd.DataFrame`` when ``prediction_format == "dataframe"``.
            * ``dict[str, PredictionFrame]`` when
              ``prediction_format == "prediction_frame"``.
        """
        fmt = self._get_prediction_format()
        if fmt == "prediction_frame":
            return predictions
        return self._predictions_to_dataframe(predictions)

    def _train_model_artifact(self) -> DartsForecaster:
        """Train a forecasting model and (optionally) save the artifact.

        Returns:
            The trained :class:`DartsForecaster`.
        """
        active_config = self.configs

        # DNA AUDIT: verify mandatory hyperparameters.
        ReproducibilityGate.Config.audit_manifest(active_config)
        ReproducibilityGate.Config.audit_architecture(active_config)

        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        run_type = active_config["run_type"]

        dataset = ViewsDatasetDarts.from_views_path(
            path_raw=path_raw,
            run_type=run_type,
            config=active_config,
            cached_path=None,
        )

        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )

        current_partition = self._resolve_active_partition_dict(active_config)
        logger.info("Training on partition [%s]: %s", run_type, current_partition)

        forecaster = self._build_forecaster(
            active_config=active_config,
            partition=current_partition,
            dataset=dataset,
            model_object=model_object,
            checkpoint_mode=active_config.get("checkpoint_mode", "best"),
        )
        forecaster.train()

        if not active_config["sweep"]:
            model_filename = _GENERATE_FILE_NAME(run_type, file_extension=".pt")
            forecaster.save_model(path=f"{path_artifacts / model_filename}")

        return forecaster

    # ------------------------------------------------------------------ evaluate

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str | None = None
    ) -> list[dict[str, PredictionFrame]]:
        """Evaluate a model artifact over all rolling-origin sequences.

        Args:
            eval_type: Evaluation type (``"standard"`` / ``"long"`` /
                ``"complete"`` / ``"live"``).
            artifact_name: Optional specific artifact name. When ``None``,
                the latest artifact for the current run type is used.

        Returns:
            A list of ``{target_name: PredictionFrame}`` dicts, one per
            rolling-origin sequence.
        """
        import concurrent.futures

        active_config = self.configs
        ReproducibilityGate.Config.audit_manifest(active_config)

        run_type = active_config["run_type"]
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts

        if artifact_name:
            logger.info("Using (non-default) artifact: %s", artifact_name)
            path_artifact = path_artifacts / artifact_name
        else:
            logger.info(
                "Using latest (default) run type (%s) specific artifact", run_type
            )
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        # Persist the artifact timestamp to the underlying manager.
        timestamp = path_artifact.stem[-15:]
        self._config_manager.add_config({"timestamp": timestamp})
        active_config = self.configs

        dataset = ViewsDatasetDarts.from_views_path(
            path_raw=path_raw,
            run_type=run_type,
            config=active_config,
            cached_path=None,
        )
        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        partition = self._resolve_active_partition_dict(active_config)
        forecaster = self._build_forecaster(
            active_config=active_config,
            partition=partition,
            dataset=dataset,
            model_object=model_object,
        )
        forecaster.load_model(path=path_artifact)

        time_steps = max(active_config["steps"])
        total_sequence_number = self._resolve_total_sequence_number(
            partition, time_steps
        )

        # HORIZON LOCKDOWN: prevent forecasting beyond ground truth.
        ReproducibilityGate.Temporal.audit_prediction_horizon(
            run_type=run_type,
            train_end=partition["train"][1],
            test_end=partition["test"][1],
            max_steps=time_steps,
            total_sequences=total_sequence_number,
        )

        predict_kwargs = self._get_predict_kwargs(active_config)

        def predict_sequence(sequence_number: int):
            """Predict a single rolling-origin sequence."""
            logger.info(
                "Starting prediction for sequence %d/%d",
                sequence_number + 1,
                total_sequence_number,
            )
            result = forecaster.predict(
                sequence_number, time_steps, **predict_kwargs
            )
            logger.info(
                "Completed prediction for sequence %d/%d",
                sequence_number + 1,
                total_sequence_number,
            )
            return result

        # FORCE SEQUENTIAL FOR GPU: Darts moves models to CPU in teardown(),
        # causing race conditions in multi-threaded GPU inference.
        if forecaster.device == "cpu":
            max_workers = active_config.get("parallel_workers", 1)
        else:
            logger.info(
                "GPU detected: forcing sequential prediction to avoid "
                "device-shifting race conditions."
            )
            max_workers = 1

        logger.info(
            "Starting parallel prediction with %d workers for %d sequences",
            max_workers,
            total_sequence_number,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(predict_sequence, seq_num): seq_num
                for seq_num in range(total_sequence_number)
            }
            df_predictions: list[dict[str, PredictionFrame] | None] = [
                None
            ] * total_sequence_number
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                seq_num = futures[future]
                try:
                    df_predictions[seq_num] = future.result()
                    completed += 1
                    logger.info(
                        "Progress: %d/%d sequences completed",
                        completed,
                        total_sequence_number,
                    )
                except Exception as exc:
                    logger.error(
                        "Sequence %d failed with error: %s", seq_num + 1, exc
                    )
                    raise

        logger.info(
            "All %d predictions completed successfully", total_sequence_number
        )
        # Format per the configured prediction_format:
        #   "dataframe"       → list[pd.DataFrame]
        #   "prediction_frame" → dict[str, list[PredictionFrame]]
        return self._format_eval_predictions(df_predictions)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ forecast

    def _forecast_model_artifact(
        self, artifact_name: str | None
    ) -> dict[str, PredictionFrame]:
        """Load a model artifact and generate a single forecast.

        Args:
            artifact_name: Optional artifact name. When ``None``, the latest
                artifact for the current run type is used.

        Returns:
            A ``{target_name: PredictionFrame}`` dict for the single forecast.
        """
        active_config = self.configs
        ReproducibilityGate.Config.audit_manifest(active_config)
        ReproducibilityGate.Config.audit_architecture(active_config)

        run_type = active_config["run_type"]
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts

        if artifact_name:
            logger.info("Using (non-default) artifact: %s", artifact_name)
            path_artifact = path_artifacts / artifact_name
        else:
            logger.info(
                "Using latest (default) run type (%s) specific artifact", run_type
            )
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        timestamp = path_artifact.stem[-15:]
        self._config_manager.add_config({"timestamp": timestamp})
        active_config = self.configs

        dataset = ViewsDatasetDarts.from_views_path(
            path_raw=path_raw,
            run_type=run_type,
            config=active_config,
            cached_path=None,
        )
        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        partition = self._resolve_active_partition_dict(active_config)
        forecaster = self._build_forecaster(
            active_config=active_config,
            partition=partition,
            dataset=dataset,
            model_object=model_object,
        )
        forecaster.load_model(path=path_artifact)

        predict_kwargs = self._get_predict_kwargs(active_config)
        predictions = forecaster.predict(0, max(active_config["steps"]), **predict_kwargs)
        return self._format_forecast_predictions(predictions)

    # ------------------------------------------------------------------ sweep

    def _execute_model_sweeping(self) -> None:
        """Execute a single wandb sweep iteration (train + evaluate + score)."""
        import wandb
        from views_pipeline_core.exceptions.exceptions import PipelineException
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            CorePredictionSniffer,
        )

        with self._wandb_module.initialize_run(
            project=self._project,
            config=None,  # set by wandb.config
            job_type="sweep",
        ):
            try:
                self._config_manager.update_for_sweep_run(
                    wandb.config,
                    self.args,
                    wandb_module=self._wandb_module,
                )
                active_config = self.configs
                ReproducibilityGate.Config.audit_manifest(active_config)
                ReproducibilityGate.Config.audit_architecture(active_config)

                logger.info(
                    "Sweeping %s %s...",
                    self._model_path.target,
                    active_config["name"],
                )
                model = self._train_model_artifact()

                self._wandb_module.send_alert(
                    title=(
                        f"Training for {self._model_path.target} "
                        f"{active_config['name']} completed successfully."
                    ),
                    text=(
                        f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n"
                        f"{wandb.config}\n```"
                    ),
                    notifications_enabled=self._wandb_notifications,
                )

                logger.info(
                    "Evaluating %s %s...",
                    self._model_path.target,
                    active_config["name"],
                )

                # HORIZON LOCKDOWN.
                partition = self._resolve_active_partition_dict(active_config)
                max_steps = max(active_config["steps"])
                ReproducibilityGate.Temporal.audit_prediction_horizon(
                    run_type=active_config["run_type"],
                    train_end=partition["train"][1],
                    test_end=partition["test"][1],
                    max_steps=max_steps,
                    total_sequences=self._resolve_total_sequence_number(
                        partition, max_steps
                    ),
                )

                df_predictions = self._evaluate_sweep(self._eval_type, model)
                # The sweep path always uses DataFrames — the sniffer and
                # ``_evaluate_prediction_dataframe`` consume pandas objects,
                # not PredictionFrames. Force dataframe mode here regardless
                # of the configured ``prediction_format``.
                if isinstance(df_predictions, dict):
                    # prediction_frame mode: transpose back to per-sequence
                    # dicts, then convert each to a DataFrame.
                    target_names = list(df_predictions.keys())
                    n_seqs = len(df_predictions[target_names[0]])
                    per_seq = [
                        {tgt: df_predictions[tgt][i] for tgt in target_names}
                        for i in range(n_seqs)
                    ]
                    df_predictions = [
                        self._predictions_to_dataframe(preds) for preds in per_seq
                    ]

                sniffer = CorePredictionSniffer(level=active_config["level"])
                for i, df in enumerate(df_predictions):
                    logger.info(
                        "Validating evaluation dataframe of sequence %d/%d",
                        i + 1,
                        len(df_predictions),
                    )
                    sniffer.sniff_predictions(df, targets=active_config["targets"])

                if self._has_evaluation_metrics():
                    self._evaluate_prediction_dataframe(
                        df_predictions, self._eval_type
                    )
                else:
                    raise PipelineException(
                        "No evaluation metrics specified in config_meta.py"
                    )
            finally:
                self._wandb_module.finish_run()

    def _evaluate_sweep(
        self, eval_type: str, model: DartsForecaster
    ) -> list[dict[str, PredictionFrame]]:
        """Sequential per-sequence prediction for sweep evaluation.

        Args:
            eval_type: Evaluation type (used to resolve the total sequence count).
            model: A trained :class:`DartsForecaster`.

        Returns:
            A list of ``{target_name: PredictionFrame}`` dicts, one per
            rolling-origin sequence.
        """
        active_config = self.configs
        partition = self._resolve_active_partition_dict(active_config)
        time_steps = max(active_config["steps"])
        total_sequence_number = self._resolve_total_sequence_number(
            partition, time_steps
        )
        predict_kwargs = self._get_predict_kwargs(active_config)
        raw_preds = [
            model.predict(seq, time_steps, **predict_kwargs)
            for seq in range(total_sequence_number)
        ]
        # Format per the configured prediction_format.
        return self._format_eval_predictions(raw_preds)

    # ------------------------------------------------------------------ predict kwargs

    def _get_predict_kwargs(self, config: Mapping[str, Any]) -> dict:
        """Extract and validate keyword arguments for ``predict()``.

        Args:
            config: Configuration snapshot.

        Returns:
            ``{"num_samples": ..., "mc_dropout": ...}``.

        Raises:
            ValueError: Mandatory parameters are missing.
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
