import numpy as np
import pandas as pd
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
        
        MANDATORY_MANIFEST = [
            "random_state",
            "steps",
            "run_type",
            "input_chunk_length",
            "output_chunk_length",
            "use_reversible_instance_norm",
            "optimizer_cls",
            "lr",
            "batch_size",
            "n_epochs",
            "loss_function",
            "num_samples",
            "mc_dropout",
            "n_jobs"
        ]

        @staticmethod
        def audit_manifest(config: Dict[str, Any]) -> None:
            """
            Verifies the presence of all mandatory DNA keys in the configuration.
            """
            missing = [k for k in ReproducibilityGate.Config.MANDATORY_MANIFEST if k not in config]
            if missing:
                error_msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED\n"
                    f"Missing mandatory parameters: {missing}\n"
                    "Every experiment must explicitly define its entire DNA."
                )
                logger.error(error_msg)
                raise MissingHyperparameterError(error_msg)
            
            # Check for None values in mandatory keys
            explicit_nones = [k for k in ReproducibilityGate.Config.MANDATORY_MANIFEST if config.get(k) is None]
            if explicit_nones:
                error_msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED\n"
                    f"Mandatory parameters set to None: {explicit_nones}\n"
                    "Implicit defaults are forbidden. Please provide explicit values."
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
                    f"Current alignment: {steps_len} / {ocl} = {steps_len/ocl:.2f}"
                )
                logger.error(error_msg)
                raise ArchitectureMismatchError(error_msg)

            # LOUD WARNING: Non-standard Horizon
            if steps_len != 36:
                logger.warning("\n" + "!" * 60)
                logger.warning("! 🚨 ATTENTION: NON-STANDARD FORECAST HORIZON DETECTED     !")
                logger.warning(f"! Current: {steps_len:<49} !")
                logger.warning("! Standard Production Horizon: 36                          !")
                logger.warning("!" * 60 + "\n")

            # LOUD WARNING: Start-Month Shift
            if steps_list[0] != 1:
                logger.warning("\n" + "?" * 60)
                logger.warning("? 🚨 ATTENTION: FORECAST START-MONTH SHIFT DETECTED        ?")
                logger.warning(f"? Current start offset: {steps_list[0]:<35} ?")
                logger.warning("? Standard production offset is 1 (t+1).                   ?")
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
        def audit_boundary_integrity(series_list: List[TimeSeries], expected_end: int) -> None:
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
            total_sequences: int = 12
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
                    f"Forecast Ends: {abs_max_pred} (train_end={train_end} + max_seq_offset={total_sequences-1} + steps={max_steps})\n"
                    "You are forecasting into the void beyond known ground truth. "
                    "This would produce a 'weird lie' in your metrics. Run Terminated."
                )
                logger.error(error_msg)
                raise PredictionHorizonError(error_msg)

    class Data:
        """Gates related to numerical sanity and leakage prevention."""

        @staticmethod
        def audit_leakage(train_ids: Union[np.ndarray, List[int]], test_ids: Union[np.ndarray, List[int]]) -> None:
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
        def audit_dataframe_schema(df: pd.DataFrame, expected_targets: List[str], expected_features: List[str]) -> None:
            """
            Ensures that the input dataframe complies with the Handshake Contract (ADR-009).
            Verifies presence of required multi-index levels and all declared columns.
            """
            import pandas as pd
            if not isinstance(df.index, pd.MultiIndex):
                raise NumericalSanityError("Dataframe must have a MultiIndex (time, entity).")
            
            if len(df.index.levels) < 2:
                raise NumericalSanityError(f"MultiIndex must have at least 2 levels, got {len(df.index.levels)}.")

            missing_cols = []
            for col in expected_targets + expected_features:
                if col not in df.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                error_msg = f"Boundary Handshake Failed: Dataframe missing required columns: {missing_cols}"
                logger.error(error_msg)
                raise KeyError(error_msg)

        @staticmethod
        def audit_numerical_sanity(series_list: List[TimeSeries], name: str, max_abs_val: float = 1e9) -> None:
            """
            Validates bit-level numerical integrity of the data stream.
            """
            for i, ts in enumerate(series_list):
                arr = ts.all_values(copy=False)
                
                if np.isnan(arr).any():
                    raise NumericalSanityError(f"NaN detected in {name} (Series index {i})")
                
                if np.isinf(arr).any():
                    raise NumericalSanityError(f"Inf detected in {name} (Series index {i})")
                
                if arr.size > 0:
                    abs_max = np.abs(arr).max()
                    if abs_max > max_abs_val:
                        logger.warning(
                            f"ADVERSARIAL OUTLIER DETECTED in {name}: "
                            f"Magnitude {abs_max:.2e} exceeds threshold {max_abs_val:.2e}"
                        )