import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Union
import logging
from darts import TimeSeries

logger = logging.getLogger(__name__)

# --- Custom Exceptions for Reproducibility Failures ---


class ReproducibilityError(Exception):
    """Base class for all reproducibility gate failures."""

    pass


class MissingHyperparameterError(ReproducibilityError):
    """Raised when a mandatory hyperparameter is missing from the config."""

    pass


class ArchitectureMismatchError(ReproducibilityError, ValueError):
    """Raised when model architecture and forecast horizon are misaligned."""

    pass


class TemporalDiscontinuityError(ReproducibilityError):
    """Raised when training and testing sets are not contiguous (t+1 failure)."""

    pass


class DataLeakageError(ReproducibilityError):
    """Raised when test data is detected within a training tensor."""

    pass


class DataStarvationError(ReproducibilityError):
    """Raised when training data fails to utilize the full available history."""

    pass


class NumericalSanityError(ReproducibilityError):
    """Raised when NaNs, Infs, or extreme outliers are detected in the data stream."""

    pass


class TemporalHoleError(ReproducibilityError):
    """Raised when missing months are detected in a historical sequence."""

    pass


class PredictionHorizonError(ReproducibilityError):
    """Raised when a forecast exceeds the ground-truth boundary of a test set."""

    pass


class ReproducibilityGate:
    """
    Unified gatekeeper for ensuring experiment reproducibility and integrity.

    This class centralizes all validation logic to ensure that models are built
    on a stable 'DNA' manifest and that temporal boundaries are strictly enforced.

    Intent Contract:
        - Purpose: Enforce physical, temporal, and configuration invariants to ensure 100% experiment reproducibility.
        - Non-Goals: Does not perform data cleaning, model training, or metric calculation.
        - Guarantees:
            - Ensures training data never leaks into the future relative to its partition.
            - Ensures all mandatory DNA parameters are present and non-null before initialization.
            - Ensures numerical stability (no NaNs/Infs) at the system boundaries.
        - Failure Behavior: Raises specific ReproducibilityError subclasses (e.g., DataLeakageError, MissingHyperparameterError)
          immediately upon invariant violation, halting execution.
    """

    class Config:
        """Gates related to configuration and hyperparameter integrity."""

        # Core genes required by ALL experiments regardless of model
        CORE_GENOME = [
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
            "lr_scheduler_factor",
            "lr_scheduler_patience",
            "lr_scheduler_min_lr",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "gradient_clip_val",
            "num_samples",
            "mc_dropout",
        ]

        # Algorithm-specific genes (Only audited if the algorithm matches)
        ALGORITHM_GENOMES = {
            "NBEATSModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "num_stacks",
                "num_blocks",
                "num_layers",
                "layer_widths",
                "activation",
                "dropout",
                "generic_architecture",
                "force_reset",
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
                "dropout",
                "feed_forward",
                "add_relative_index",
                "use_static_covariates",
                "full_attention",
                "use_reversible_instance_norm",
                "skip_interpolation",
                "hidden_continuous_size",
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
                "dropout",
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
            "TCNModel": [
                "input_chunk_length",
                "output_chunk_length",
                "output_chunk_shift",
                "kernel_size",
                "num_filters",
                "dilation_base",
                "dropout",
                "use_reversible_instance_norm",
            ],
        }

        # Optimizer-specific genes
        OPTIMIZER_GENOMES = {
            "Adam": ["lr", "weight_decay"],
            "AdamW": ["lr", "weight_decay"],
            "SGD": ["lr", "weight_decay", "momentum"],
            "RMSprop": ["lr", "weight_decay", "momentum", "alpha"],
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
            "MSELoss": [],
            "L1Loss": [],
            "HuberLoss": ["delta"],
            "PoissonNLLLoss": [],
        }

        @staticmethod
        def audit_manifest(config: Dict[str, Any]) -> None:
            """
            Verifies the presence of all mandatory DNA keys in the configuration.
            Dynamically determines the required genome based on algorithm, optimizer, and loss.
            """
            # 1. Audit Core Genome
            missing_core = [
                k for k in ReproducibilityGate.Config.CORE_GENOME if k not in config
            ]
            if missing_core:
                error_msg = f"REPRODUCIBILITY CONTRACT VIOLATED: Missing core parameters: {missing_core}"
                logger.error(error_msg)
                raise MissingHyperparameterError(error_msg)

            # 2. Identify Algorithm & Audit Architecture Genome
            algo = config.get("algorithm")
            if algo not in ReproducibilityGate.Config.ALGORITHM_GENOMES:
                error_msg = (
                    f"REPRODUCIBILITY CONTRACT VIOLATED: Unknown algorithm '{algo}'."
                )
                logger.error(error_msg)
                raise MissingHyperparameterError(error_msg)

            algo_genome = ReproducibilityGate.Config.ALGORITHM_GENOMES[algo]
            missing_algo = [k for k in algo_genome if k not in config]
            if missing_algo:
                error_msg = (
                    f"REPRODUCIBILITY CONTRACT VIOLATED: Algorithm '{algo}' "
                    f"requires missing parameters: {missing_algo}"
                )
                logger.error(error_msg)
                raise MissingHyperparameterError(error_msg)

            # 3. Identify Optimizer & Audit Optimizer Genome
            opt = config.get("optimizer_cls")
            if opt in ReproducibilityGate.Config.OPTIMIZER_GENOMES:
                opt_genome = ReproducibilityGate.Config.OPTIMIZER_GENOMES[opt]
                missing_opt = [k for k in opt_genome if k not in config]
                if missing_opt:
                    error_msg = (
                        f"REPRODUCIBILITY CONTRACT VIOLATED: Optimizer '{opt}' "
                        f"requires missing parameters: {missing_opt}"
                    )
                    logger.error(error_msg)
                    raise MissingHyperparameterError(error_msg)
            else:
                logger.warning(
                    f"Optimizer '{opt}' is not registered in OPTIMIZER_GENOMES. Skipping specific audit."
                )

            # 4. Identify Loss & Audit Loss Genome
            loss = config.get("loss_function")
            if loss in ReproducibilityGate.Config.LOSS_GENOMES:
                loss_genome = ReproducibilityGate.Config.LOSS_GENOMES[loss]
                missing_loss = [k for k in loss_genome if k not in config]
                if missing_loss:
                    error_msg = (
                        f"REPRODUCIBILITY CONTRACT VIOLATED: Loss '{loss}' "
                        f"requires missing parameters: {missing_loss}"
                    )
                    logger.error(error_msg)
                    raise MissingHyperparameterError(error_msg)
            else:
                logger.warning(
                    f"Loss '{loss}' is not registered in LOSS_GENOMES. Skipping specific audit."
                )

            # 5. Check for None values in ALL required keys (Core + Algo + Opt + Loss)
            all_required = (
                ReproducibilityGate.Config.CORE_GENOME
                + algo_genome
                + ReproducibilityGate.Config.OPTIMIZER_GENOMES.get(opt, [])
                + ReproducibilityGate.Config.LOSS_GENOMES.get(loss, [])
            )
            explicit_nones = [k for k in all_required if config.get(k) is None]
            if explicit_nones:
                error_msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: Mandatory parameters set to None: "
                    f"{explicit_nones}. Implicit defaults are forbidden."
                )
                logger.error(error_msg)
                raise MissingHyperparameterError(error_msg)

        @staticmethod
        def audit_architecture(config: Dict[str, Any]) -> None:
            """
            Ensures model architecture and forecast horizon are mathematically aligned.
            """
            steps_list = config["steps"]
            steps_len = len(steps_list)
            ocl = config["output_chunk_length"]

            if steps_len % ocl != 0:
                error_msg = (
                    "Architecture Mismatch\n"
                    f"Forecast horizon 'steps' ({steps_len}) must be a multiple of "
                    f"'output_chunk_length' ({ocl}).\n"
                    f"Current alignment: {steps_len} / {ocl} = {steps_len / ocl:.2f}"
                )
                logger.error(error_msg)
                raise ArchitectureMismatchError(error_msg)

            # LOUD WARNING: Non-standard Horizon
            if steps_len != 36:
                logger.warning("\n" + "!" * 60)
                logger.warning(
                    "! 🚨 ATTENTION: NON-STANDARD FORECAST HORIZON DETECTED     !"
                )
                logger.warning(f"! Current: {steps_len:<49} !")
                logger.warning(
                    "! Standard Production Horizon: 36                          !"
                )
                logger.warning("!" * 60 + "\n")

            # LOUD WARNING: Start-Month Shift
            if steps_list[0] != 1:
                logger.warning("\n" + "?" * 60)
                logger.warning(
                    "? 🚨 ATTENTION: FORECAST START-MONTH SHIFT DETECTED        ?"
                )
                logger.warning(f"? Current start offset: {steps_list[0]:<35} ?")
                logger.warning(
                    "? Standard production offset is 1 (t+1).                   ?"
                )
                logger.warning("?" * 60 + "\n")

    class Temporal:
        """Gates related to time-series alignment and continuity."""

        @staticmethod
        def audit_continuity(partition: Dict[str, Any]) -> None:
            """
            The Continuity Guardian (t+1 Check).
            Verifies that the test set starts exactly one month after the train set ends.
            """
            if "train" not in partition or "test" not in partition:
                # Forecasting runs may only have 'train'
                return

            train_end = partition["train"][1]
            test_start = partition["test"][0]

            if test_start != train_end + 1:
                error_msg = (
                    "CRITICAL TEMPORAL DISCONTINUITY\n"
                    f"Train end: {train_end} | Test start: {test_start}\n"
                    "Requirement: test_start == train_end + 1. Run Terminated."
                )
                logger.error(error_msg)
                raise TemporalDiscontinuityError(error_msg)

        @staticmethod
        def audit_boundary_integrity(
            series_list: List[TimeSeries], expected_end: int
        ) -> None:
            """
            The Firewall Gate.
            Ensures that training data ends EXACTLY at the partition boundary.
            No peeking, no starvation.
            """
            for i, ts in enumerate(series_list):
                actual_end = int(ts.time_index.max())

                if actual_end > expected_end:
                    error_msg = (
                        "🚨 CRITICAL DATA LEAKAGE DETECTED 🚨\n"
                        f"Series [{i}] ends at {actual_end}, which is beyond the "
                        f"allowed training boundary of {expected_end}."
                    )
                    logger.error(error_msg)
                    raise DataLeakageError(error_msg)

                if actual_end < expected_end:
                    error_msg = (
                        "🚨 CRITICAL DATA STARVATION DETECTED 🚨\n"
                        f"Series [{i}] ends at {actual_end}, but the partition "
                        f"allows training up to {expected_end}.\n"
                        "You are throwing away your most recent history!"
                    )
                    logger.error(error_msg)
                    raise DataStarvationError(error_msg)

        @staticmethod
        def audit_sequence_contiguity(time_ids: Union[np.ndarray, List[int]]) -> None:
            """
            The Sequence Auditor (No Holes Check).
            Verifies that the training range contains every month in its span.
            """
            ids = np.sort(np.unique(time_ids))
            if len(ids) == 0:
                return

            expected = np.arange(ids.min(), ids.max() + 1)
            if not np.array_equal(ids, expected):
                missing = sorted(list(set(expected) - set(ids)))
                error_msg = (
                    "TEMPORAL HOLE DETECTED\n"
                    "The data sequence is discontinuous.\n"
                    f"Missing months: {missing}"
                )
                logger.error(error_msg)
                raise TemporalHoleError(error_msg)

        @staticmethod
        def audit_prediction_horizon(
            run_type: str,
            train_end: int,
            test_end: int,
            max_steps: int,
            total_sequences: int = 12,
        ) -> None:
            """
            Ensures predictions do not exceed the known ground truth in evaluation phases.
            """
            if run_type not in ["calibration", "validation"]:
                return

            # Abs max ID reached: train_end + sequence_offset (max seq - 1) + forecast_horizon
            # Example: t=444, seq=12 (0-11 offset), steps=36 -> 444 + 11 + 36 = 491
            abs_max_pred = train_end + (total_sequences - 1) + max_steps

            if abs_max_pred > test_end:
                error_msg = (
                    f"🚨 CRITICAL PREDICTION OVERFLOW DETECTED 🚨\n"
                    f"Run Type: {run_type}\n"
                    f"Test Set Ends: {test_end}\n"
                    f"Forecast Ends: {abs_max_pred} (train_end={train_end} + max_seq_offset={total_sequences - 1} + steps={max_steps})\n"
                    "You are forecasting into the void beyond known ground truth. "
                    "This would produce a 'weird lie' in your metrics. Run Terminated."
                )
                logger.error(error_msg)
                raise PredictionHorizonError(error_msg)

    class Data:
        """Gates related to numerical sanity and leakage prevention."""

        @staticmethod
        def audit_leakage(
            train_ids: Union[np.ndarray, List[int]],
            test_ids: Union[np.ndarray, List[int]],
        ) -> None:
            """
            The Leakage Firewall.
            Guarantees zero overlap between training and test time IDs.
            """
            overlap = set(train_ids) & set(test_ids)
            if overlap:
                error_msg = (
                    "CRITICAL DATA LEAKAGE DETECTED\n"
                    f"Common time IDs found in both Train and Test sets: {sorted(list(overlap))}"
                )
                logger.error(error_msg)
                raise DataLeakageError(error_msg)

        @staticmethod
        def audit_dataframe_schema(
            df: pd.DataFrame, expected_targets: List[str], expected_features: List[str]
        ) -> None:
            """
            Ensures that the input dataframe complies with the Handshake Contract (ADR-009).
            Verifies presence of required multi-index levels and all declared columns.
            """
            if not isinstance(df.index, pd.MultiIndex):
                raise NumericalSanityError(
                    "Dataframe must have a MultiIndex (time, entity)."
                )

            # 1. Audit Levels (ADR-001)
            if len(df.index.levels) < 2:
                raise NumericalSanityError(
                    f"MultiIndex must have at least 2 levels, got {len(df.index.levels)}."
                )

            level_names = list(df.index.names)
            if "month_id" not in level_names or "country_id" not in level_names:
                logger.warning(
                    f"NON-STANDARD INDEX DETECTED: Expected [month_id, country_id], got {level_names}. "
                    "Proceeding but mapping may be unstable."
                )

            # 2. Audit Presence (Handshake)
            missing_cols = []
            all_required = expected_targets + expected_features
            for col in all_required:
                if col not in df.columns:
                    missing_cols.append(col)

            if missing_cols:
                error_msg = f"Boundary Handshake Failed: Dataframe missing required columns: {missing_cols}"
                logger.error(error_msg)
                raise KeyError(error_msg)

            # 3. Audit Precision Warning (ADR-010)
            # We warn here, but the Dataset Airlock will fix it during conversion.
            float64_cols = [
                col
                for col in all_required
                if df[col].dtype == np.float64 or df[col].dtype == float
            ]
            if float64_cols:
                logger.warning(
                    f"PRECISION WARNING: Upstream provided float64 for columns {float64_cols}. "
                    "Data Airlock will downcast to float32 to satisfy ADR-010."
                )

        @staticmethod
        def audit_numerical_sanity(
            series_list: List[TimeSeries], name: str, max_abs_val: float = 1e9
        ) -> None:
            """
            Validates bit-level numerical integrity of the data stream.
            """
            for i, ts in enumerate(series_list):
                arr = ts.all_values(copy=False)

                if np.isnan(arr).any():
                    raise NumericalSanityError(
                        f"NaN detected in {name} (Series index {i})"
                    )

                if np.isinf(arr).any():
                    raise NumericalSanityError(
                        f"Inf detected in {name} (Series index {i})"
                    )

                if arr.size > 0:
                    abs_max = np.abs(arr).max()
                    if abs_max > max_abs_val:
                        logger.warning(
                            f"ADVERSARIAL OUTLIER DETECTED in {name}: "
                            f"Magnitude {abs_max:.2e} exceeds threshold {max_abs_val:.2e}"
                        )

        @staticmethod
        def lock_entropy(seed: int) -> None:
            """
            The Entropy Guardian.
            Force-resets all RNG seeds to guarantee bit-perfect identity in
            probabilistic distributions (e.g., MC Dropout) across reloads.
            """
            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Note: torch.manual_seed also seeds all CUDA devices and MPS.
            logger.info(f"Entropy Locked: RNG seeds reset to {seed}")
