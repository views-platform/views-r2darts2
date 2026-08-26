"""Darts forecasting model manager (slim orchestrator).

Inherits from :class:`views_pipeline_core.managers.model.ForecastingModelManager`
and overrides the template methods. The parent class is imported lazily so the
package remains importable without ``views_pipeline_core``.

The manager is now slim — all data operations (loading, slicing, scaling,
Darts TimeSeries construction, inverse transforms) are delegated to
:class:`ViewsDataset`. The manager only orchestrates:
    * Config → partition resolution
    * Dataset + model + forecaster construction
    * Train / evaluate / forecast / sweep entry points
    * Prediction format switching (dataframe vs prediction_frame)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import torch  # noqa: F401 — kept alive for monkey-patches

from views_frames import PredictionFrame

from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.infrastructure.patches import apply_all_patches
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate
from views_r2darts2.transformers.darts_bridge import prediction_frames_to_dataframe

logger = logging.getLogger(__name__)


def _import_parent_class():
    """Lazily import the parent class + helpers from views_pipeline_core."""
    from views_pipeline_core.files.utils import generate_model_file_name
    from views_pipeline_core.managers.model import (
        ForecastingModelManager,
        ModelPathManager,
    )
    return ForecastingModelManager, ModelPathManager, generate_model_file_name


try:
    _PARENT_CLASS, _MODEL_PATH_MANAGER, _GENERATE_FILE_NAME = _import_parent_class()
except ImportError:  # pragma: no cover - depends on env
    _PARENT_CLASS = object  # type: ignore[assignment]
    _MODEL_PATH_MANAGER = None  # type: ignore[assignment]
    _GENERATE_FILE_NAME = None  # type: ignore[assignment]
    logger.warning(
        "views_pipeline_core not installed — DartsForecastingModelManager "
        "inherits from object. Use static helpers directly."
    )


class DartsForecastingModelManager(_PARENT_CLASS):  # type: ignore[misc, valid-type]
    """Slim orchestrator: config → dataset → forecaster → train/eval/forecast.

    All data operations are delegated to :class:`ViewsDataset`. The manager
    only manages the experiment lifecycle.
    """

    def __init__(
        self,
        model_path: Any,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        """Initialize the manager.

        Args:
            model_path: A :class:`ModelPathManager`.
            wandb_notifications: Enable W&B Slack notifications.
            use_prediction_store: Enable the prediction store.

        Raises:
            ImportError: ``views_pipeline_core`` not installed.
        """
        if _PARENT_CLASS is object:
            raise ImportError(
                "views_pipeline_core not installed — cannot instantiate "
                "DartsForecastingModelManager."
            )
        super().__init__(
            model_path=model_path,
            wandb_notifications=wandb_notifications,
            use_prediction_store=use_prediction_store,
        )
        apply_all_patches()
        logger.info("Model architecture: %s", self.configs["algorithm"])

    # ------------------------------------------------------------------ partition

    def _resolve_active_partition_dict(self, config: Mapping[str, Any]) -> dict:
        """Resolve the partition dict for the current run."""
        run_type = config.get("run_type")
        steps_list = config.get("steps")
        if not run_type or steps_list is None:
            raise KeyError("Missing 'run_type' or 'steps' in config.")
        if not isinstance(steps_list, list):
            raise TypeError(f"'steps' must be a list, got {type(steps_list).__name__}.")

        ReproducibilityGate.Config.audit_architecture(config)

        master_partitions = getattr(self, "_partition_dict", {})
        if run_type in master_partitions:
            partition = master_partitions[run_type]
        else:
            if hasattr(self._data_loader, "_get_partition_dict"):
                self._data_loader.partition = run_type
                partition = self._data_loader._get_partition_dict(steps=len(steps_list))
            else:
                raise ValueError(f"Unsupported run_type: {run_type}")

        ReproducibilityGate.Temporal.audit_continuity(partition)
        return partition

    @staticmethod
    def _resolve_total_sequence_number(partition: dict, max_steps: int) -> int:
        """Derive the number of rolling-origin sequences."""
        test_start, test_end = partition["test"]
        test_len = test_end - test_start + 1
        if test_len < max_steps:
            raise ValueError(
                f"Test partition length ({test_len}) < max_steps ({max_steps})."
            )
        return test_len - max_steps + 1

    # ------------------------------------------------------------------ prediction format

    def _get_prediction_format(self) -> str:
        """Return the configured prediction output format."""
        return self.configs.get("prediction_format", "dataframe")

    def _predictions_to_dataframe(self, predictions: Any) -> Any:
        """Convert predictions to a pandas DataFrame.

        Handles two return types:
        * ``dict[str, PredictionFrame]`` — the legacy path (small forecasts).
        * ``_FramesWithCleanup`` (dict subclass with memmap-backed values) —
          the streaming path. After conversion, the memmap temp dir is
          cleaned up to free disk space.
        """
        time_id = self.configs.get("time_id", "month_id")
        entity_id = self.configs.get("entity_id", "country_id")
        df = prediction_frames_to_dataframe(
            predictions=predictions, time_id=time_id, entity_id=entity_id,
        )
        # Clean up the memmap temp dir if the predictions carry one.
        frames_dir = getattr(predictions, "_frames_dir", None)
        if frames_dir is not None:
            import shutil
            shutil.rmtree(str(frames_dir), ignore_errors=True)
        return df

    @staticmethod
    def _transpose_predictions(
        per_sequence_preds: list[dict[str, PredictionFrame]],
    ) -> dict[str, list[PredictionFrame]]:
        """Transpose ``list[dict]`` → ``dict[list]``."""
        if not per_sequence_preds:
            return {}
        target_names = list(per_sequence_preds[0].keys())
        return {
            tgt: [seq[tgt] for seq in per_sequence_preds]
            for tgt in target_names
        }

    def _format_eval_predictions(
        self, per_sequence_preds: list[dict[str, PredictionFrame]]
    ) -> Any:
        """Format per-sequence predictions for the parent manager."""
        if self._get_prediction_format() == "prediction_frame":
            return self._transpose_predictions(per_sequence_preds)
        return [self._predictions_to_dataframe(p) for p in per_sequence_preds]

    def _format_forecast_predictions(
        self, predictions: dict[str, PredictionFrame]
    ) -> Any:
        """Format a single forecast's predictions."""
        if self._get_prediction_format() == "prediction_frame":
            return predictions
        return self._predictions_to_dataframe(predictions)

    # ------------------------------------------------------------------ factory

    def _infer_cache_source_label(self) -> str:
        """Infer cache source label used by dataloaders file naming."""
        queryset = self._model_path.get_queryset()
        if isinstance(queryset, dict):
            source = str(queryset.get("source", "")).strip().lower()
            if source == "views-datafactory":
                return "datafactory"
            if source == "synthetic":
                return "synthetic"
            if source == "viewser":
                return "viewser"
        return "viewser"

    def _resolve_raw_parquet_path(self, run_type: str) -> Path:
        """Resolve raw parquet path across source-aware cache spellings."""
        path_raw = Path(self._model_path.data_raw)
        preferred = self._infer_cache_source_label()

        candidate_labels = [preferred, "viewser", "datafactory", "synthetic"]
        seen: set[str] = set()
        candidates: list[Path] = []
        for label in candidate_labels:
            if label in seen:
                continue
            seen.add(label)
            candidates.append(path_raw / f"{run_type}_{label}_df.parquet")

        for candidate in candidates:
            if candidate.exists():
                if candidate.name != f"{run_type}_{preferred}_df.parquet":
                    logger.warning(
                        "Raw parquet fallback used: preferred=%s selected=%s",
                        f"{run_type}_{preferred}_df.parquet",
                        candidate.name,
                    )
                return candidate

        raise FileNotFoundError(
            "No raw parquet found for run_type='{}' in {}. Tried: {}".format(
                run_type,
                path_raw,
                [p.name for p in candidates],
            )
        )

    def _build_dataset(self, active_config: Mapping[str, Any]) -> ViewsDataset:
        """Build a :class:`ViewsDataset` from the config's parquet path."""
        run_type = active_config["run_type"]
        parquet_path = self._resolve_raw_parquet_path(run_type)
        targets = list(active_config.get("targets") or active_config.get("regression_targets") or [])
        level = active_config.get("level", "cm")
        ds = ViewsDataset(parquet_path, targets=targets, broadcast_features=True)
        logger.info("Dataset loaded: %s (%s) from %s", ds, level, parquet_path.name)
        return ds

    def _build_forecaster(
        self,
        *,
        active_config: Mapping[str, Any],
        partition: Mapping[str, Any],
        dataset: ViewsDataset,
        model_object: Any,
        checkpoint_mode: str | None = None,
    ) -> DartsForecaster:
        """Single factory for :class:`DartsForecaster` construction."""
        kwargs: dict[str, Any] = dict(
            dataset=dataset,
            model=model_object,
            partition_dict=dict(partition),
            feature_scaler=active_config.get("feature_scaler"),
            target_scaler=active_config.get("target_scaler"),
            log_targets=active_config.get("log_targets", False),
            log_features=active_config.get("log_features", []),
            feature_scaler_map=active_config.get("feature_scaler_map"),
            random_state=active_config["random_state"],
            static_covariate_stats=(
                active_config.get("static_covariate_stats")
                if active_config.get("use_static_covariates")
                else None
            ),
            use_cyclic_encoders=active_config.get("use_cyclic_encoders", False),
        )
        if checkpoint_mode is not None:
            kwargs["checkpoint_mode"] = checkpoint_mode
        return DartsForecaster(**kwargs)

    # ------------------------------------------------------------------ train

    def _train_model_artifact(self) -> DartsForecaster:
        """Train a model and optionally save the artifact."""
        active_config = self.configs
        ReproducibilityGate.Config.audit_manifest(active_config)
        ReproducibilityGate.Config.audit_architecture(active_config)

        dataset = self._build_dataset(active_config)
        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        partition = self._resolve_active_partition_dict(active_config)
        logger.info("Training on partition [%s]: %s", active_config["run_type"], partition)

        forecaster = self._build_forecaster(
            active_config=active_config, partition=partition,
            dataset=dataset, model_object=model_object,
            checkpoint_mode=active_config.get("checkpoint_mode", "best"),
        )
        forecaster.train()

        if not active_config.get("sweep"):
            model_filename = _GENERATE_FILE_NAME(active_config["run_type"], file_extension=".pt")
            forecaster.save_model(path=f"{self._model_path.artifacts / model_filename}")
        return forecaster

    # ------------------------------------------------------------------ evaluate

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str | None = None
    ) -> Any:
        """Evaluate a model artifact over all rolling-origin sequences."""
        import concurrent.futures

        active_config = self.configs
        ReproducibilityGate.Config.audit_manifest(active_config)

        path_artifacts = self._model_path.artifacts
        run_type = active_config["run_type"]
        if artifact_name:
            path_artifact = path_artifacts / artifact_name
        else:
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        timestamp = path_artifact.stem[-15:]
        self._config_manager.add_config({"timestamp": timestamp})
        active_config = self.configs

        dataset = self._build_dataset(active_config)
        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        partition = self._resolve_active_partition_dict(active_config)
        forecaster = self._build_forecaster(
            active_config=active_config, partition=partition,
            dataset=dataset, model_object=model_object,
        )
        forecaster.load_model(path=path_artifact)

        time_steps = max(active_config["steps"])
        total_seq = self._resolve_total_sequence_number(partition, time_steps)
        ReproducibilityGate.Temporal.audit_prediction_horizon(
            run_type=run_type, train_end=partition["train"][1],
            test_end=partition["test"][1], max_steps=time_steps,
            total_sequences=total_seq,
        )
        predict_kwargs = self._get_predict_kwargs(active_config)

        def predict_sequence(seq_num: int):
            logger.info("Predicting sequence %d/%d", seq_num + 1, total_seq)
            return forecaster.predict(seq_num, time_steps, **predict_kwargs)

        max_workers = (
            active_config.get("parallel_workers", 1)
            if forecaster.device == "cpu" else 1
        )
        # Process sequences in chunks of max_workers. Each prediction
        # returns a _FramesWithCleanup (dict subclass of PredictionFrames
        # with memmap-backed values). The frames stay as PredictionFrames —
        # no DataFrame conversion here. The memmap temp dirs persist until
        # _format_eval_predictions / _predictions_to_dataframe is called
        # (which cleans them up after conversion).
        results: list = [None] * total_seq
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for chunk_start in range(0, total_seq, max_workers):
                    chunk_end = min(chunk_start + max_workers, total_seq)
                    futures = {
                        executor.submit(predict_sequence, s): s
                        for s in range(chunk_start, chunk_end)
                    }
                    for future in concurrent.futures.as_completed(futures):
                        seq_num = futures[future]
                        results[seq_num] = future.result()
                        logger.info("Completed %d/%d", seq_num + 1, total_seq)
        else:
            for s in range(total_seq):
                results[s] = predict_sequence(s)
                logger.info("Completed %d/%d", s + 1, total_seq)

        return self._format_eval_predictions(results)

    # ------------------------------------------------------------------ forecast

    def _forecast_model_artifact(self, artifact_name: str | None) -> Any:
        """Load a model and generate a single forecast."""
        active_config = self.configs
        ReproducibilityGate.Config.audit_manifest(active_config)
        ReproducibilityGate.Config.audit_architecture(active_config)

        path_artifacts = self._model_path.artifacts
        run_type = active_config["run_type"]
        if artifact_name:
            path_artifact = path_artifacts / artifact_name
        else:
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        timestamp = path_artifact.stem[-15:]
        self._config_manager.add_config({"timestamp": timestamp})
        active_config = self.configs

        dataset = self._build_dataset(active_config)
        model_object = ModelCatalog(config=active_config).get_model(
            model_name=active_config["algorithm"]
        )
        partition = self._resolve_active_partition_dict(active_config)
        forecaster = self._build_forecaster(
            active_config=active_config, partition=partition,
            dataset=dataset, model_object=model_object,
        )
        forecaster.load_model(path=path_artifact)

        predict_kwargs = self._get_predict_kwargs(active_config)
        preds = forecaster.predict(0, max(active_config["steps"]), **predict_kwargs)
        return self._format_forecast_predictions(preds)

    # ------------------------------------------------------------------ sweep

    def _execute_model_sweeping(self) -> None:
        """Execute a single wandb sweep iteration."""
        import wandb
        from views_pipeline_core.exceptions.exceptions import PipelineException
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            CorePredictionSniffer,
        )

        with self._wandb_module.initialize_run(
            project=self._project, config=None, job_type="sweep",
        ):
            try:
                self._config_manager.update_for_sweep_run(
                    wandb.config, self.args, wandb_module=self._wandb_module,
                )
                active_config = self.configs
                ReproducibilityGate.Config.audit_manifest(active_config)
                ReproducibilityGate.Config.audit_architecture(active_config)

                model = self._train_model_artifact()
                df_predictions = self._evaluate_sweep(self._eval_type, model)

                # Sweep path always uses DataFrames (sniffer + eval consume pandas).
                if isinstance(df_predictions, dict):
                    target_names = list(df_predictions.keys())
                    n_seqs = len(df_predictions[target_names[0]])
                    per_seq = [
                        {tgt: df_predictions[tgt][i] for tgt in target_names}
                        for i in range(n_seqs)
                    ]
                    df_predictions = [
                        self._predictions_to_dataframe(p) for p in per_seq
                    ]

                sniffer = CorePredictionSniffer(level=active_config["level"])
                for i, df in enumerate(df_predictions):
                    sniffer.sniff_predictions(df, targets=active_config.get("targets") or active_config.get("regression_targets"))

                if self._has_evaluation_metrics():
                    self._evaluate_prediction_dataframe(df_predictions, self._eval_type)
                else:
                    raise PipelineException("No evaluation metrics specified.")
            finally:
                self._wandb_module.finish_run()

    def _evaluate_sweep(self, eval_type: str, model: DartsForecaster) -> Any:
        """Sequential per-sequence prediction for sweep evaluation."""
        active_config = self.configs
        partition = self._resolve_active_partition_dict(active_config)
        time_steps = max(active_config["steps"])
        total_seq = self._resolve_total_sequence_number(partition, time_steps)
        predict_kwargs = self._get_predict_kwargs(active_config)
        raw_preds = [
            model.predict(s, time_steps, **predict_kwargs)
            for s in range(total_seq)
        ]
        return self._format_eval_predictions(raw_preds)

    # ------------------------------------------------------------------ predict kwargs

    def _get_predict_kwargs(self, config: Mapping[str, Any]) -> dict:
        """Extract and validate kwargs for ``predict()``."""
        mandatory = ["num_samples", "mc_dropout"]
        missing = [k for k in mandatory if k not in config]
        if missing:
            raise ValueError(f"Missing mandatory prediction parameters: {missing}.")
        return {"num_samples": config["num_samples"], "mc_dropout": config["mc_dropout"]}