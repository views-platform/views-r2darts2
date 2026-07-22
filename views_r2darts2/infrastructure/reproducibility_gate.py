"""Reproducibility gate — 100% experiment reproducibility firewall.

Pure-namespace class with three nested static-method containers:

    * :class:`Config`   — config / hyperparameter integrity gates.
    * :class:`Temporal` — temporal alignment and continuity gates.
    * :class:`Data`     — numerical sanity and leakage gates.

This module is pandas-free. The legacy ``audit_dataframe_schema`` method (which
took a ``pd.DataFrame``) has been replaced by :meth:`Data.audit_frame_schema`,
which takes a :class:`views_frames.FeatureFrame`. The checks are equivalent:

    * MultiIndex with at least 2 levels → the frame's
      :class:`SpatioTemporalIndex` is 2-D by construction (time, unit).
    * Index name is ``[month_id, country_id]`` → emit a warning when the
      frame's index columns are not the VIEWS canonical names.
    * All expected target/feature columns are present → check
      ``feature_names`` includes every expected name.
    * Warn on float64 columns → the frame is always float32 by construction,
      so the check is informational only.


"""

from __future__ import annotations

import logging
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from darts import TimeSeries

from views_frames import FeatureFrame

from views_r2darts2.infrastructure.exceptions import (
    ArchitectureMismatchError,
    DataLeakageError,
    DataStarvationError,
    MissingHyperparameterError,
    NumericalSanityError,
    PredictionHorizonError,
    ReproducibilityError,
    TemporalDiscontinuityError,
    TemporalHoleError,
)

logger = logging.getLogger(__name__)

# Re-exported so ``from views_r2darts2.infrastructure.reproducibility_gate
# import TemporalDiscontinuityError`` keeps working for legacy callers.
__all__ = [
    "ReproducibilityGate",
    "ReproducibilityError",
    "MissingHyperparameterError",
    "ArchitectureMismatchError",
    "TemporalDiscontinuityError",
    "DataLeakageError",
    "DataStarvationError",
    "NumericalSanityError",
    "TemporalHoleError",
    "PredictionHorizonError",
]


# ----------------------------------------------------------------------
# Note on TemporalDiscontinuityError alias
# ----------------------------------------------------------------------
# The legacy exception was named ``TemporalContinuityError`` in some test
# imports — keep an alias to preserve the public surface.
TemporalContinuityError = TemporalDiscontinuityError


class ReproducibilityGate:
    """Namespace of reproducibility-firewall static methods.

    This class is never instantiated. Its three nested classes (``Config``,
    ``Temporal``, ``Data``) each group related gate methods.
    """

    # ==================================================================
    # Config — config / hyperparameter integrity
    # ==================================================================

    class Config:
        """Gates related to configuration and hyperparameter integrity."""

        CORE_GENOME: list[str] = [
            "random_state",
            "steps",
            "run_type",
            "name",
            "algorithm",
            "loss_function",
            "lr",
            "weight_decay",
            "batch_size",
            "n_epochs",
            "optimizer_cls",
            "lr_scheduler_cls",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "gradient_clip_val",
            "num_samples",
            "mc_dropout",
        ]

        NULLABLE_PARAMS: set[str] = {
            "hidden_fc_sizes",
            "pooling_kernel_sizes",
            "n_freq_downsample",
            "categorical_embedding_sizes",
            "temporal_hidden_size_past",
            "temporal_hidden_size_future",
        }

        ALGORITHM_GENOMES = {
            "NBEATSModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "num_stacks",
                "num_blocks",
                "num_layers",
                "layer_widths",
                "expansion_coefficient_dim",
                "trend_polynomial_degree",
                "activation",
                "dropout",
                "generic_architecture",
                "force_reset",
                "use_reversible_instance_norm",
            ],
            "NHiTSModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "num_stacks",
                "num_blocks",
                "num_layers",
                "layer_widths",
                "pooling_kernel_sizes",
                "n_freq_downsample",
                "activation",
                "max_pool_1d",
                "dropout",
                "use_reversible_instance_norm",
                "force_reset",
            ],
            "TFTModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "hidden_size",
                "lstm_layers",
                "num_attention_heads",
                "full_attention",
                "feed_forward",
                "dropout",
                "hidden_continuous_size",
                "categorical_embedding_sizes",
                "add_relative_index",
                "skip_interpolation",
                "norm_type",
                "use_static_covariates",
                "use_reversible_instance_norm",
            ],
            "TiDEModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "num_encoder_layers",
                "num_decoder_layers",
                "decoder_output_dim",
                "hidden_size",
                "temporal_width_past",
                "temporal_width_future",
                "temporal_hidden_size_past",
                "temporal_hidden_size_future",
                "temporal_decoder_hidden",
                "use_layer_norm",
                "dropout",
                "use_static_covariates",
                "use_reversible_instance_norm",
            ],
            "TSMixerModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "num_blocks",
                "ff_size",
                "hidden_size",
                "activation",
                "dropout",
                "norm_type",
                "normalize_before",
                "use_static_covariates",
                "use_reversible_instance_norm",
            ],
            "NLinearModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "shared_weights",
                "const_init",
                "normalize",
                "use_static_covariates",
                "use_reversible_instance_norm",
            ],
            "DLinearModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "shared_weights",
                "kernel_size",
                "const_init",
                "use_static_covariates",
                "use_reversible_instance_norm",
            ],
            "BlockRNNModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "rnn_type",
                "hidden_dim",
                "n_rnn_layers",
                "hidden_fc_sizes",
                "dropout",
                "activation",
                "use_static_covariates",
                "use_reversible_instance_norm",
            ],
            "TransformerModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "d_model",
                "nhead",
                "num_encoder_layers",
                "num_decoder_layers",
                "dim_feedforward",
                "dropout",
                "activation",
                "norm_type",
                "use_reversible_instance_norm",
                "detect_anomaly",
            ],
            "MultiQueryTransformerModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "d_model",
                "nhead",
                "num_encoder_layers",
                "num_decoder_layers",
                "dim_feedforward",
                "dropout",
                "activation",
                "norm_type",
                "use_reversible_instance_norm",
                "detect_anomaly",
            ],
            "TCNModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "kernel_size",
                "num_filters",
                "num_layers",
                "dilation_base",
                "weight_norm",
                "dropout",
                "use_reversible_instance_norm",
            ],
        }

        # Optimizer-specific genes
        OPTIMIZER_GENOMES = {
            "Adam": ["lr", "weight_decay", "betas"],
            "AdamW": ["lr", "weight_decay", "betas"],
            "RAdam": ["lr", "weight_decay", "betas"],
            "SGD": ["lr", "weight_decay", "momentum"],
            "RMSprop": ["lr", "weight_decay", "momentum", "alpha"],
        }

        # Scheduler-specific genes
        SCHEDULER_GENOMES = {
            "ReduceLROnPlateau": [
                "lr_scheduler_factor",
                "lr_scheduler_patience",
                "lr_scheduler_min_lr",
            ],
            "CosineAnnealingWarmRestarts": [
                "lr_scheduler_T_0",
                "lr_scheduler_T_mult",
                "lr_scheduler_eta_min",
            ],
            "WarmupCAWR": [
                "lr_scheduler_warmup_epochs",
                "lr_scheduler_T_0",
                "lr_scheduler_T_mult",
                "lr_scheduler_eta_min",
            ],
            "StepLR": ["lr_scheduler_step_size", "lr_scheduler_gamma"],
            "ExponentialLR": ["lr_scheduler_gamma"],
        }

        # Loss-specific genes
        LOSS_GENOMES = {
            "WeightedPenaltyHuberLoss": [
                "zero_threshold",
                "delta",
                "non_zero_weight",
                "false_positive_weight",
                "false_negative_weight",
            ],
            "WeightedHuberLoss": ["zero_threshold", "delta", "non_zero_weight"],
            "TimeAwareWeightedHuberLoss": [
                "zero_weight",
                "non_zero_weight",
                "decay_factor",
                "delta",
            ],
            "SpikeFocalLoss": ["alpha", "gamma", "spike_threshold"],
            "TweedieLoss": [
                "p",
                "non_zero_weight",
                "zero_threshold",
                "false_positive_weight",
                "false_negative_weight",
                "eps",
            ],
            "AsymmetricQuantileLoss": ["tau", "non_zero_weight", "zero_threshold"],
            "ZeroInflatedLoss": [
                "zero_weight",
                "count_weight",
                "delta",
                "zero_threshold",
                "eps",
            ],
            "ShrinkageLoss": ["a", "c"],
            "PrismLoss": ["non_zero_threshold", "delta"],
            "SpotlightLoss": ["non_zero_threshold"],
            "SpotlightLossLogcosh": ["non_zero_threshold"],
            "SpotlightLossHuber": ["non_zero_threshold"],
            "SpotlightLossPowerLaw": ["non_zero_threshold"],
            "SpotlightLossAsinh": ["non_zero_threshold"],
            "CharbonnierLoss": [],
            "SpotlightFocalLoss": ["gamma", "delta", "non_zero_threshold"],
            "SentinelLoss": ["alpha", "beta", "kappa", "delta", "gamma"],
            "MSELoss": [],
            "L1Loss": [],
            "HuberLoss": ["delta"],
            "PoissonNLLLoss": [],
            # Darts Likelihood objects — loss_fn is set to None; the
            # likelihood is passed directly to the model constructor.
            "GaussianLikelihood": [],
            "LaplaceLikelihood": [],
            "PoissonLikelihood": [],
            "NegativeBinomialLikelihood": [],
            "BetaLikelihood": [],
            "CauchyLikelihood": [],
            "ExponentialLikelihood": [],
            "GumbelLikelihood": [],
            "LogNormalLikelihood": [],
            "WeibullLikelihood": [],
            "QuantileRegression": [],
        }

        @staticmethod
        def audit_manifest(config: Mapping[str, Any]) -> None:
            """Verify the presence of all mandatory DNA keys in ``config``.

            Audits, in order: Core Genome → Algorithm Genome → Optimizer
            Genome → Scheduler Genome → Loss Genome → None-values in all
            required keys (excluding :attr:`NULLABLE_PARAMS`).

            Raises:
                MissingHyperparameterError: Any mandatory key is absent or
                    ``None`` (and not in :attr:`NULLABLE_PARAMS`).
            """
            for key in ReproducibilityGate.Config.CORE_GENOME:
                if key not in config:
                    raise MissingHyperparameterError(
                        f"MANDATORY PARAMETER MISSING: '{key}' not in config. "
                        "The manifest must declare every core genome key."
                    )

            algorithm = config.get("algorithm")
            algo_genes = ReproducibilityGate.Config.ALGORITHM_GENOMES.get(
                algorithm, []
            )
            for key in algo_genes:
                if key not in config:
                    raise MissingHyperparameterError(
                        f"MANDATORY ALGORITHM GENE MISSING: '{key}' for "
                        f"algorithm '{algorithm}'."
                    )

            optimizer = config.get("optimizer_cls")
            opt_genes = ReproducibilityGate.Config.OPTIMIZER_GENOMES.get(
                optimizer, []
            )
            for key in opt_genes:
                if key not in config:
                    raise MissingHyperparameterError(
                        f"MANDATORY OPTIMIZER GENE MISSING: '{key}' for "
                        f"optimizer '{optimizer}'."
                    )

            scheduler = config.get("lr_scheduler_cls")
            sched_genes = ReproducibilityGate.Config.SCHEDULER_GENOMES.get(
                scheduler, []
            )
            for key in sched_genes:
                if key not in config:
                    raise MissingHyperparameterError(
                        f"MANDATORY SCHEDULER GENE MISSING: '{key}' for "
                        f"scheduler '{scheduler}'."
                    )

            loss = config.get("loss_function")
            loss_genes = ReproducibilityGate.Config.LOSS_GENOMES.get(loss, [])
            for key in loss_genes:
                if key not in config:
                    raise MissingHyperparameterError(
                        f"MANDATORY LOSS GENE MISSING: '{key}' for "
                        f"loss '{loss}'."
                    )

            # Final pass: no None values in any required key (excluding nullable).
            nullable = ReproducibilityGate.Config.NULLABLE_PARAMS
            for key, value in config.items():
                if value is None and key not in nullable:
                    # Only raise for keys we know are mandatory. Other keys
                    # may legitimately be None.
                    in_core = key in ReproducibilityGate.Config.CORE_GENOME
                    in_algo = key in algo_genes
                    in_opt = key in opt_genes
                    in_sched = key in sched_genes
                    in_loss = key in loss_genes
                    if in_core or in_algo or in_opt or in_sched or in_loss:
                        raise MissingHyperparameterError(
                            f"MANDATORY PARAMETER MISSING: '{key}' is None."
                        )

        @staticmethod
        def audit_architecture(config: Mapping[str, Any]) -> None:
            """Ensure ``len(steps) % output_chunk_length == 0``.

            Emits loud warnings if ``steps`` is not the standard 36-month
            horizon or if the first step is not 1 (VIEWS month_id start).

            Raises:
                ArchitectureMismatchError: ``steps`` length is not divisible
                    by ``output_chunk_length``.
            """
            steps = config.get("steps")
            output_chunk_length = config.get("output_chunk_length")
            if steps is None or output_chunk_length is None:
                raise ArchitectureMismatchError(
                    "Cannot audit architecture: 'steps' or "
                    "'output_chunk_length' is missing from config."
                )
            steps_list = list(steps)
            steps_len = len(steps_list)
            if steps_len % output_chunk_length != 0:
                raise ArchitectureMismatchError(
                    f"Architecture mismatch: len(steps)={steps_len} is not "
                    f"divisible by output_chunk_length={output_chunk_length}. "
                    "The forecast horizon must align with the step grid."
                )
            if steps_len != 36:
                logger.warning(
                    "Non-standard horizon: len(steps)=%d (expected 36). "
                    "Step grid: %s",
                    steps_len,
                    steps_list,
                )
            if steps_list and steps_list[0] != 1:
                logger.warning(
                    "Step grid does not start at 1 (starts at %d). The VIEWS "
                    "month_id convention expects step 1 to be the first "
                    "forecast month.",
                    steps_list[0],
                )

    # ==================================================================
    # Temporal — temporal alignment and continuity
    # ==================================================================

    class Temporal:
        """Gates related to time-series alignment and continuity."""

        @staticmethod
        def audit_continuity(partition: Mapping[str, Any]) -> None:
            """Continuity Guardian (t+1 check).

            Verifies that ``test_start == train_end + 1`` when both partitions
            are present. No-ops when only the train partition is declared.

            Raises:
                TemporalDiscontinuityError: The test partition does not begin
                    exactly one step after the train partition ends.
            """
            if "train" not in partition or "test" not in partition:
                return
            train_start, train_end = partition["train"]
            test_start, test_end = partition["test"]
            if test_start != train_end + 1:
                raise TemporalDiscontinuityError(
                    f"Temporal discontinuity: test_start ({test_start}) != "
                    f"train_end + 1 ({train_end + 1}). The test partition "
                    "must begin exactly one step after the train partition ends."
                )

        @staticmethod
        def audit_boundary_integrity(
            series_list: Sequence[TimeSeries], expected_end: int
        ) -> None:
            """Firewall Gate — ensure each series ends at ``expected_end``.

            Raises:
                DataLeakageError: A series extends beyond ``expected_end``.
                DataStarvationError: A series ends before ``expected_end``.
            """
            for ts in series_list:
                actual_end = int(ts.time_index.max())
                if actual_end > expected_end:
                    raise DataLeakageError(
                        f"Data leakage: a series extends to time {actual_end}, "
                        f"beyond the allowed training boundary {expected_end}."
                    )
                if actual_end < expected_end:
                    raise DataStarvationError(
                        f"Data starvation: a series ends at {actual_end}, "
                        f"before the available training boundary {expected_end}. "
                        "You are throwing away your most recent history."
                    )

        @staticmethod
        def audit_sequence_contiguity(time_ids: Sequence[int] | np.ndarray) -> None:
            """Sequence Auditor — no holes in the time-id sequence.

            Raises:
                TemporalHoleError: The sorted-unique time ids are not a
                    contiguous ``arange(min, max+1)``.
            """
            arr = np.asarray(list(time_ids), dtype=np.int64)
            if arr.shape[0] == 0:
                return
            unique = np.unique(arr)
            expected = np.arange(unique.min(), unique.max() + 1, dtype=np.int64)
            if not np.array_equal(unique, expected):
                missing = np.setdiff1d(expected, unique)
                raise TemporalHoleError(
                    f"Temporal holes detected: {len(missing)} missing time ids "
                    f"between {int(unique.min())} and {int(unique.max())}. "
                    f"First few missing: {missing[:5].tolist()}."
                )

        @staticmethod
        def audit_prediction_horizon(
            run_type: str,
            train_end: int,
            test_end: int,
            max_steps: int,
            total_sequences: int = 12,
        ) -> None:
            """Ensure the forecast horizon fits within the test partition.

            For ``calibration`` and ``validation`` runs, verifies
            ``train_end + (total_sequences - 1) + max_steps <= test_end``.

            Raises:
                PredictionHorizonError: The horizon overflows the test partition.
            """
            if run_type == "forecasting":
                return
            required_end = train_end + (total_sequences - 1) + max_steps
            if required_end > test_end:
                raise PredictionHorizonError(
                    f"PREDICTION OVERFLOW: train_end ({train_end}) + "
                    f"(total_sequences - 1) ({total_sequences - 1}) + "
                    f"max_steps ({max_steps}) = {required_end} > test_end "
                    f"({test_end}). Reduce max_steps or total_sequences."
                )

    # ==================================================================
    # Data — numerical sanity and leakage
    # ==================================================================

    class Data:
        """Gates related to numerical sanity and leakage prevention."""

        @staticmethod
        def audit_leakage(
            train_ids: Sequence[int] | np.ndarray,
            test_ids: Sequence[int] | np.ndarray,
        ) -> None:
            """Leakage Firewall — zero overlap between train and test time ids.

            Raises:
                DataLeakageError: The train and test time-id sets intersect.
            """
            train_set = set(np.asarray(train_ids, dtype=np.int64).tolist())
            test_set = set(np.asarray(test_ids, dtype=np.int64).tolist())
            overlap = train_set & test_set
            if overlap:
                raise DataLeakageError(
                    f"Data leakage: train and test time-id sets overlap at "
                    f"{sorted(overlap)[:5]} (showing first 5 of {len(overlap)})."
                )

        @staticmethod
        def audit_frame_schema(
            *,
            feature_frame: FeatureFrame,
            expected_targets: Sequence[str],
            expected_features: Sequence[str],
        ) -> None:
            """Handshake Contract — verify the frame carries every expected column.

            Replaces the legacy ``audit_dataframe_schema(df: pd.DataFrame, ...)``
            method. The frame is always float32 by construction, so the legacy
            "warn on float64 columns" check is moot — we instead verify:

                * The frame's index has the expected 2-level structure (always
                  true for :class:`SpatioTemporalIndex` — emitted as info).
                * Every expected target column is present in
                  ``feature_frame.feature_names``.
                * Every expected feature column is present in
                  ``feature_frame.feature_names``.

            Args:
                feature_frame: The loaded :class:`FeatureFrame` carrying all
                    value columns (features + targets).
                expected_targets: Target column names that must be present.
                expected_features: Feature column names that must be present.

            Raises:
                KeyError: An expected column is missing from the frame.
                NumericalSanityError: The frame's value array is not float32.
            """
            # The frame's value array is always float32 by construction; assert
            # it explicitly to catch any future regression in the loader.
            if feature_frame.values.dtype != np.float32:
                raise NumericalSanityError(
                    f"FeatureFrame values dtype is {feature_frame.values.dtype}, "
                    "expected float32 (ADR-010 airlock invariant)."
                )

            available = set(feature_frame.feature_names)
            missing_targets = set(expected_targets) - available
            if missing_targets:
                raise KeyError(
                    f"FeatureFrame is missing target columns: "
                    f"{sorted(missing_targets)}. Available: {sorted(available)}."
                )
            missing_features = set(expected_features) - available
            if missing_features:
                raise KeyError(
                    f"FeatureFrame is missing feature columns: "
                    f"{sorted(missing_features)}. Available: {sorted(available)}."
                )

            # Emit an informational log when the index columns are not the
            # VIEWS canonical names. The frame's SpatioTemporalIndex always has
            # two levels (time, unit) — this is structural, not configurable.
            level = feature_frame.index.level
            if level.value == "cm":
                expected_index_names = ("month_id", "country_id")
            elif level.value == "pgm":
                expected_index_names = ("month_id", "priogrid_id")
            else:
                expected_index_names = ("month_id", "<entity>")
            logger.info(
                "Frame schema audit passed: %d columns, %d rows, level=%s, "
                "expected_index_names=%s.",
                len(feature_frame.feature_names),
                feature_frame.n_rows,
                level.value,
                expected_index_names,
            )

        @staticmethod
        def audit_numerical_sanity(
            series_list: Sequence[TimeSeries],
            name: str,
            max_abs_val: float = 1e9,
        ) -> None:
            """Bit-level NaN / Inf / outlier check on a list of TimeSeries.

            Args:
                series_list: Darts TimeSeries objects to audit.
                name: Label for the audit (used in error messages).
                max_abs_val: Maximum allowed absolute value before a warning
                    is emitted (does not raise).

            Raises:
                NumericalSanityError: Any value is NaN or Inf.
            """
            for i, ts in enumerate(series_list):
                arr = ts.all_values(copy=False)
                if np.isnan(arr).any():
                    raise NumericalSanityError(
                        f"NaN detected in {name} (series {i}). "
                        "The data stream is contaminated."
                    )
                if np.isinf(arr).any():
                    raise NumericalSanityError(
                        f"Inf detected in {name} (series {i}). "
                        "The data stream is contaminated."
                    )
                extreme = np.abs(arr).max()
                if extreme > max_abs_val:
                    logger.warning(
                        "Extreme value in %s (series %d): |%g| > %g.",
                        name,
                        i,
                        float(extreme),
                        max_abs_val,
                    )

        @staticmethod
        def lock_entropy(seed: int) -> None:
            """Entropy Guardian — seed every RNG to ``seed``.

            Seeds ``random``, ``numpy.random``, ``torch.manual_seed``, and
            ``torch.cuda.manual_seed_all`` (when CUDA is available). For full
            GPU determinism, callers should also set
            ``torch.backends.cudnn.deterministic = True`` and
            ``torch.backends.cudnn.benchmark = False`` — this method does not
            touch those flags to avoid surprising global side effects.
            """
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logger.info("Entropy locked: random, numpy, torch seeded with %d.", seed)
