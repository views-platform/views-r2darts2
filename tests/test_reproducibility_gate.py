"""Tests for :mod:`views_r2darts2.infrastructure.reproducibility_gate`.

Exercises the three nested gate classes (``Config``, ``Temporal``, ``Data``) on
the new pandas-free :class:`ReproducibilityGate`. All frame-based checks consume
:class:`views_frames.FeatureFrame` objects — never ``pd.DataFrame``.

Pandas-free (only at the Darts ``TimeSeries`` boundary for
the numerical-sanity gates).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd  # noqa: WPS433 — allowed at the Darts TimeSeries boundary
import pytest
import torch
from darts import TimeSeries

from views_frames import (
    FeatureFrame,
    SpatioTemporalIndex,
    SpatialLevel,
)
from views_r2darts2.infrastructure.exceptions import (
    ArchitectureMismatchError,
    DataLeakageError,
    DataStarvationError,
    MissingHyperparameterError,
    NumericalSanityError,
    PredictionHorizonError,
    TemporalDiscontinuityError,
    TemporalHoleError,
)
from views_r2darts2.infrastructure.reproducibility_gate import (
    ReproducibilityGate,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _full_nbeats_config() -> dict[str, Any]:
    """A complete NBEATS genome config (passes every audit_manifest check).

    Includes all core keys, every NBEATS algorithm key, the SGD optimizer
    keys, the ReduceLROnPlateau scheduler keys, and the
    WeightedPenaltyHuberLoss keys. ``hidden_fc_sizes`` is intentionally
    ``None`` (it is in :attr:`NULLABLE_PARAMS`).
    """
    return {
        # --- Core genome ---
        "random_state": 42,
        "steps": list(range(1, 37)),
        "run_type": "calibration",
        "name": "nbeats_smoke",
        "algorithm": "NBEATSModel",
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "n_epochs": 1,
        "optimizer_cls": "SGD",
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        "gradient_clip_val": 1.0,
        "num_samples": 1,
        "mc_dropout": False,
        # --- NBEATS algorithm genome ---
        "input_chunk_length": 12,
        "output_chunk_length": 6,
        "output_chunk_shift": 0,
        "num_stacks": 2,
        "num_blocks": 2,
        "num_layers": 2,
        "layer_widths": 16,
        "expansion_coefficient_dim": 4,
        "trend_polynomial_degree": 2,
        "activation": "ReLU",
        "dropout": 0.0,
        "generic_architecture": True,
        "force_reset": False,
        "use_reversible_instance_norm": False,
        # --- SGD optimizer genome ---
        "momentum": 0.9,
        # --- ReduceLROnPlateau scheduler genome ---
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 2,
        "lr_scheduler_min_lr": 1e-6,
        # --- WeightedPenaltyHuberLoss genome ---
        "zero_threshold": 0.5,
        "delta": 1.0,
        "non_zero_weight": 1.0,
        "false_positive_weight": 1.0,
        "false_negative_weight": 1.0,
        # --- Nullable params ---
        "hidden_fc_sizes": None,
    }


def _make_feature_frame(
    *,
    n_time: int = 6,
    n_entities: int = 2,
    feature_names: list[str] | None = None,
    values_fn: Any = None,
    level: SpatialLevel = SpatialLevel.CM,
) -> FeatureFrame:
    """Build a tiny :class:`FeatureFrame` for the schema audit tests.

    Args:
        n_time: Number of time steps per entity.
        n_entities: Number of entities.
        feature_names: Column names (defaults to ``["feat_a", "target_a"]``).
        values_fn: Optional callable ``(N, F) -> ndarray`` returning a 2-D
            float32 array of values. When ``None``, sequential integers are
            used.
        level: Spatial level of the index.

    Returns:
        A :class:`FeatureFrame` carrying ``n_time * n_entities`` rows and
        ``len(feature_names)`` columns (single-sample, ``S=1``).
    """
    if feature_names is None:
        feature_names = ["feat_a", "target_a"]
    n_rows = n_time * n_entities
    time = np.tile(np.arange(1, n_time + 1, dtype=np.int64), n_entities)
    unit = np.repeat(np.arange(1, n_entities + 1, dtype=np.int64), n_time)
    if values_fn is None:
        values = np.arange(
            n_rows * len(feature_names), dtype=np.float32
        ).reshape(n_rows, len(feature_names))
    else:
        values = np.asarray(values_fn(n_rows, len(feature_names)), dtype=np.float32)
    index = SpatioTemporalIndex(time=time, unit=unit, level=level)
    return FeatureFrame.from_2d(values, index=index, feature_names=feature_names)


def _make_entity_timeseries(
    *,
    entity_id: int = 1,
    time_ids: tuple[int, ...] = (100, 101, 102, 103, 104, 105),
    fill_value: float = 1.0,
) -> TimeSeries:
    """Build a single-entity deterministic Darts :class:`TimeSeries`.

    Used for the boundary-integrity and numerical-sanity gates.
    """
    time_arr = np.asarray(time_ids, dtype=np.int64)
    values = np.full(
        (len(time_arr), 1), fill_value, dtype=np.float32
    )
    return TimeSeries.from_times_and_values(
        times=pd.Index(time_arr),
        values=values,
        columns=["y"],
        static_covariates=pd.DataFrame({"country_id": [float(entity_id)]}),
        freq=1,
    )


# ----------------------------------------------------------------------
# Config audit
# ----------------------------------------------------------------------


class TestConfigAudit:
    """Tests for :meth:`ReproducibilityGate.Config.audit_manifest` and
    :meth:`audit_architecture`."""

    def test_audit_manifest_passes_with_full_config(self) -> None:
        """A complete NBEATS+SGD+ReduceLROnPlateau+WeightedPenaltyHuberLoss
        config passes the audit without raising."""
        ReproducibilityGate.Config.audit_manifest(_full_nbeats_config())  # no raise

    def test_audit_manifest_missing_core_key_raises(self) -> None:
        """Removing ``random_state`` raises :class:`MissingHyperparameterError`."""
        config = _full_nbeats_config()
        del config["random_state"]
        with pytest.raises(MissingHyperparameterError, match="random_state"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_missing_algorithm_key_raises(self) -> None:
        """Removing ``num_stacks`` (NBEATS algorithm gene) raises
        :class:`MissingHyperparameterError`."""
        config = _full_nbeats_config()
        del config["num_stacks"]
        with pytest.raises(MissingHyperparameterError, match="num_stacks"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_missing_optimizer_key_raises(self) -> None:
        """Removing ``momentum`` (SGD optimizer gene) raises
        :class:`MissingHyperparameterError`."""
        config = _full_nbeats_config()
        del config["momentum"]
        with pytest.raises(MissingHyperparameterError, match="momentum"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_missing_scheduler_key_raises(self) -> None:
        """Removing ``lr_scheduler_factor`` (ReduceLROnPlateau gene) raises
        :class:`MissingHyperparameterError`."""
        config = _full_nbeats_config()
        del config["lr_scheduler_factor"]
        with pytest.raises(
            MissingHyperparameterError, match="lr_scheduler_factor"
        ):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_missing_loss_key_raises(self) -> None:
        """Removing ``delta`` (WeightedPenaltyHuberLoss gene) raises
        :class:`MissingHyperparameterError`."""
        config = _full_nbeats_config()
        del config["delta"]
        with pytest.raises(MissingHyperparameterError, match="delta"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_nullable_param_allowed(self) -> None:
        """``hidden_fc_sizes=None`` is permitted (it is in NULLABLE_PARAMS)."""
        config = _full_nbeats_config()
        config["hidden_fc_sizes"] = None
        # Must not raise — the None pass skips keys in NULLABLE_PARAMS.
        ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_architecture_passes(self) -> None:
        """``steps=[1..36]`` (len 36) and ``output_chunk_length=12`` →
        36 % 12 == 0, audit passes."""
        config = _full_nbeats_config()
        config["steps"] = list(range(1, 37))
        config["output_chunk_length"] = 12
        ReproducibilityGate.Config.audit_architecture(config)  # no raise

    def test_audit_architecture_mismatch_raises(self) -> None:
        """``steps=[1..5]`` (len 5) and ``output_chunk_length=3`` →
        5 % 3 != 0, raises :class:`ArchitectureMismatchError`."""
        config = _full_nbeats_config()
        config["steps"] = list(range(1, 6))
        config["output_chunk_length"] = 3
        with pytest.raises(ArchitectureMismatchError, match="not.*divisible"):
            ReproducibilityGate.Config.audit_architecture(config)


# ----------------------------------------------------------------------
# Sklearn-model bypass (MarkovModel)
# ----------------------------------------------------------------------


class TestSklearnBypass:
    """Tests for the sklearn-based-model bypass in the reproducibility gate.

    Sklearn models (currently only ``MarkovModel``) skip the
    torch-specific genome checks (optimizer, scheduler, loss) and use
    :attr:`SKLEARN_CORE_GENOME` instead of :attr:`CORE_GENOME`.
    """

    @staticmethod
    def _minimal_markov_config() -> dict[str, Any]:
        """A minimal MarkovModel config — no optimizer / scheduler / loss
        keys (which the torch CORE_GENOME would require)."""
        return {
            "algorithm": "MarkovModel",
            "name": "markov_smoke",
            "run_type": "calibration",
            "random_state": 42,
            "steps": [1, 2, 3],
            "targets": ["lr_ged_sb"],
            "markov_target": "lr_ged_sb",
            "markov_method": "direct",
            "regression_method": "single",
            "markov_threshold": 0,
            "n_jobs": -1,
        }

    def test_audit_manifest_passes_with_minimal_markov_config(self) -> None:
        """A minimal Markov config (no loss_function / optimizer_cls /
        lr_scheduler_cls / batch_size / n_epochs) passes the audit."""
        ReproducibilityGate.Config.audit_manifest(
            self._minimal_markov_config()
        )  # no raise

    def test_audit_manifest_missing_markov_target_raises(self) -> None:
        """Removing ``markov_target`` (Markov algorithm gene) raises
        :class:`MissingHyperparameterError`."""
        config = self._minimal_markov_config()
        del config["markov_target"]
        with pytest.raises(MissingHyperparameterError, match="markov_target"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_missing_targets_raises(self) -> None:
        """Removing ``targets`` (Markov algorithm gene) raises
        :class:`MissingHyperparameterError`."""
        config = self._minimal_markov_config()
        del config["targets"]
        with pytest.raises(MissingHyperparameterError, match="targets"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_missing_random_state_raises(self) -> None:
        """Removing ``random_state`` (shared core gene) raises."""
        config = self._minimal_markov_config()
        del config["random_state"]
        with pytest.raises(MissingHyperparameterError, match="random_state"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_does_not_require_torch_keys(self) -> None:
        """The Markov config has no ``loss_function``, ``optimizer_cls``,
        ``lr_scheduler_cls``, ``batch_size``, ``n_epochs`` — the audit
        must not raise (sklearn models use SKLEARN_CORE_GENOME)."""
        config = self._minimal_markov_config()
        for key in (
            "loss_function", "optimizer_cls", "lr_scheduler_cls",
            "batch_size", "n_epochs", "lr", "weight_decay",
            "gradient_clip_val", "num_samples", "mc_dropout",
            "early_stopping_patience", "early_stopping_min_delta",
        ):
            assert key not in config, f"test setup: {key} unexpectedly in config"
        ReproducibilityGate.Config.audit_manifest(config)  # no raise

    def test_audit_architecture_skips_output_chunk_length_check(self) -> None:
        """MarkovModel configs do not have ``output_chunk_length`` — the
        architecture audit skips the divisibility check.

        A torch model with steps len=5 and ocl=3 would raise; MarkovModel
        with the same steps and no ocl must not raise.
        """
        config = self._minimal_markov_config()
        config["steps"] = [1, 2, 3, 4, 5]  # len 5, not divisible by anything
        assert "output_chunk_length" not in config
        ReproducibilityGate.Config.audit_architecture(config)  # no raise

    def test_audit_architecture_torch_model_still_checked(self) -> None:
        """Sanity: torch-model configs are still subject to the
        divisibility check (the bypass is Markov-specific)."""
        config = {
            "algorithm": "NBEATSModel",
            "steps": [1, 2, 3, 4, 5],  # len 5
            "output_chunk_length": 3,  # 5 % 3 != 0
        }
        with pytest.raises(ArchitectureMismatchError, match="not.*divisible"):
            ReproducibilityGate.Config.audit_architecture(config)

    def test_markov_algorithm_in_sklearn_algorithms_set(self) -> None:
        """``MarkovModel`` is registered in :attr:`SKLEARN_ALGORITHMS`."""
        assert "MarkovModel" in ReproducibilityGate.Config.SKLEARN_ALGORITHMS

    def test_sklearn_core_genome_subset_of_core_genome(self) -> None:
        """Every key in :attr:`SKLEARN_CORE_GENOME` must also appear in
        :attr:`CORE_GENOME` (the sklearn path is a strict subset of the
        torch path, not a separate genome)."""
        sklearn_keys = set(ReproducibilityGate.Config.SKLEARN_CORE_GENOME)
        torch_keys = set(ReproducibilityGate.Config.CORE_GENOME)
        assert sklearn_keys.issubset(torch_keys)

    def test_markov_model_in_algorithm_genomes(self) -> None:
        """``MarkovModel`` has its own entry in :attr:`ALGORITHM_GENOMES`."""
        assert "MarkovModel" in ReproducibilityGate.Config.ALGORITHM_GENOMES
        genes = ReproducibilityGate.Config.ALGORITHM_GENOMES["MarkovModel"]
        # Required genes.
        assert "targets" in genes
        assert "markov_target" in genes
        assert "markov_method" in genes
        assert "regression_method" in genes
        # NOT in the genome (because they're nullable / not used).
        assert "input_chunk_length" not in genes
        assert "output_chunk_length" not in genes
        assert "loss_function" not in genes


# ----------------------------------------------------------------------
# Temporal audit
# ----------------------------------------------------------------------


class TestTemporalAudit:
    """Tests for :class:`ReproducibilityGate.Temporal`."""

    def test_audit_continuity_passes(self) -> None:
        """Train ends at 200, test starts at 201 → passes."""
        ReproducibilityGate.Temporal.audit_continuity(
            {"train": (100, 200), "test": (201, 300)}
        )  # no raise

    def test_audit_continuity_gap_raises(self) -> None:
        """Train ends at 200, test starts at 202 →
        :class:`TemporalDiscontinuityError`."""
        with pytest.raises(TemporalDiscontinuityError, match="discontinuity"):
            ReproducibilityGate.Temporal.audit_continuity(
                {"train": (100, 200), "test": (202, 300)}
            )

    def test_audit_continuity_overlap_raises(self) -> None:
        """Train ends at 200, test starts at 200 → overlap → raises."""
        with pytest.raises(TemporalDiscontinuityError):
            ReproducibilityGate.Temporal.audit_continuity(
                {"train": (100, 200), "test": (200, 300)}
            )

    def test_audit_continuity_train_only_no_op(self) -> None:
        """When only the ``train`` key is present, audit is a no-op."""
        ReproducibilityGate.Temporal.audit_continuity({"train": (100, 200)})

    def test_audit_boundary_integrity_passes(self) -> None:
        """A TimeSeries ending exactly at ``expected_end`` passes."""
        ts = _make_entity_timeseries(time_ids=(100, 101, 102, 103, 104, 105))
        ReproducibilityGate.Temporal.audit_boundary_integrity([ts], expected_end=105)

    def test_audit_boundary_integrity_leakage_raises(self) -> None:
        """A TimeSeries extending beyond ``expected_end`` raises
        :class:`DataLeakageError`."""
        ts = _make_entity_timeseries(time_ids=(100, 101, 102, 103, 104, 105, 106))
        with pytest.raises(DataLeakageError, match="leakage"):
            ReproducibilityGate.Temporal.audit_boundary_integrity([ts], expected_end=105)

    def test_audit_boundary_integrity_starvation_raises(self) -> None:
        """A TimeSeries ending before ``expected_end`` raises
        :class:`DataStarvationError`."""
        ts = _make_entity_timeseries(time_ids=(100, 101, 102, 103))
        with pytest.raises(DataStarvationError, match="starvation"):
            ReproducibilityGate.Temporal.audit_boundary_integrity([ts], expected_end=105)

    def test_audit_sequence_contiguity_passes(self) -> None:
        """A contiguous ``[100, 101, 102, 103]`` sequence passes."""
        ReproducibilityGate.Temporal.audit_sequence_contiguity(
            np.array([100, 101, 102, 103])
        )

    def test_audit_sequence_contiguity_hole_raises(self) -> None:
        """A sequence with a hole (e.g. ``[100, 101, 103, 104]``) raises
        :class:`TemporalHoleError`."""
        with pytest.raises(TemporalHoleError, match="hole"):
            ReproducibilityGate.Temporal.audit_sequence_contiguity(
                np.array([100, 101, 103, 104])
            )

    def test_audit_sequence_contiguity_empty_no_op(self) -> None:
        """An empty array is a no-op."""
        ReproducibilityGate.Temporal.audit_sequence_contiguity(np.array([], dtype=np.int64))

    def test_audit_prediction_horizon_passes(self) -> None:
        """Calibration run with train_end=444, test_end=492, max_steps=36,
        total_sequences=12 → passes (444 + 11 + 36 = 491 ≤ 492)."""
        ReproducibilityGate.Temporal.audit_prediction_horizon(
            run_type="calibration",
            train_end=444,
            test_end=492,
            max_steps=36,
            total_sequences=12,
        )  # no raise

    def test_audit_prediction_horizon_forecasting_exempt(self) -> None:
        """``run_type='forecasting'`` skips the horizon check entirely."""
        # Would overflow if the check ran, but forecasting is exempt.
        ReproducibilityGate.Temporal.audit_prediction_horizon(
            run_type="forecasting",
            train_end=100,
            test_end=110,
            max_steps=60,
            total_sequences=12,
        )  # no raise

    def test_audit_prediction_horizon_overflow_raises(self) -> None:
        """Calibration with max_steps=60 (444 + 11 + 60 = 515 > 492) raises
        :class:`PredictionHorizonError`."""
        with pytest.raises(PredictionHorizonError, match="OVERFLOW"):
            ReproducibilityGate.Temporal.audit_prediction_horizon(
                run_type="calibration",
                train_end=444,
                test_end=492,
                max_steps=60,
                total_sequences=12,
            )


# ----------------------------------------------------------------------
# Data audit
# ----------------------------------------------------------------------


class TestDataAudit:
    """Tests for :class:`ReproducibilityGate.Data`."""

    def test_audit_leakage_passes(self) -> None:
        """Disjoint train/test time ids pass the leakage check."""
        ReproducibilityGate.Data.audit_leakage(
            train_ids=[100, 101, 102], test_ids=[103, 104, 105]
        )

    def test_audit_leakage_raises(self) -> None:
        """Overlapping ids raise :class:`DataLeakageError`."""
        with pytest.raises(DataLeakageError, match="overlap"):
            ReproducibilityGate.Data.audit_leakage(
                train_ids=[100, 101, 102], test_ids=[102, 103, 104]
            )

    def test_audit_frame_schema_passes(self) -> None:
        """A FeatureFrame carrying every expected target/feature passes."""
        frame = _make_feature_frame(feature_names=["feat_a", "target_a"])
        ReproducibilityGate.Data.audit_frame_schema(
            feature_frame=frame,
            expected_targets=["target_a"],
            expected_features=["feat_a"],
        )

    def test_audit_frame_schema_missing_target_raises(self) -> None:
        """An expected target not present in the frame raises ``KeyError``."""
        frame = _make_feature_frame(feature_names=["feat_a", "target_a"])
        with pytest.raises(KeyError, match="target"):
            ReproducibilityGate.Data.audit_frame_schema(
                feature_frame=frame,
                expected_targets=["nonexistent_target"],
                expected_features=["feat_a"],
            )

    def test_audit_frame_schema_missing_feature_raises(self) -> None:
        """An expected feature not present in the frame raises ``KeyError``."""
        frame = _make_feature_frame(feature_names=["feat_a", "target_a"])
        with pytest.raises(KeyError, match="feature"):
            ReproducibilityGate.Data.audit_frame_schema(
                feature_frame=frame,
                expected_targets=["target_a"],
                expected_features=["nonexistent_feature"],
            )

    def test_audit_frame_schema_wrong_dtype_raises(self) -> None:
        """A frame whose ``values.dtype`` is not ``float32`` raises
        :class:`NumericalSanityError`.

        Note:
            :class:`FeatureFrame` coerces its values to ``float32`` at
            construction (via :func:`coerce_values`). The production
            :meth:`audit_frame_schema` check is a defensive guard against
            future regressions in the loader. To exercise it, we construct a
            valid frame and then manually overwrite its private ``_values``
            attribute with a ``float64`` array — this simulates a regression
            where the loader hands the frame a non-float32 buffer.
        """
        frame = _make_feature_frame(feature_names=["feat_a", "target_a"])
        # Manually overwrite the internal float32 buffer with float64 to
        # simulate a future loader regression that breaks the ADR-010 airlock.
        frame._values = frame.values.astype(np.float64)  # type: ignore[attr-defined]
        assert frame.values.dtype == np.float64  # sanity
        with pytest.raises(NumericalSanityError, match="float32"):
            ReproducibilityGate.Data.audit_frame_schema(
                feature_frame=frame,
                expected_targets=["target_a"],
                expected_features=["feat_a"],
            )

    def test_audit_numerical_sanity_passes(self) -> None:
        """A list of clean TimeSeries passes the numerical sanity check."""
        series = [
            _make_entity_timeseries(entity_id=1, fill_value=1.0),
            _make_entity_timeseries(entity_id=2, fill_value=2.0),
        ]
        ReproducibilityGate.Data.audit_numerical_sanity(series, "targets")

    def test_audit_numerical_sanity_nan_raises(self) -> None:
        """A TimeSeries containing NaN raises
        :class:`NumericalSanityError`."""
        time_arr = np.array([100, 101, 102], dtype=np.int64)
        values = np.array([[1.0], [np.nan], [3.0]], dtype=np.float32)
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(time_arr),
            values=values,
            columns=["y"],
            static_covariates=pd.DataFrame({"country_id": [1.0]}),
            freq=1,
        )
        with pytest.raises(NumericalSanityError, match="NaN"):
            ReproducibilityGate.Data.audit_numerical_sanity([ts], "targets")

    def test_audit_numerical_sanity_inf_raises(self) -> None:
        """A TimeSeries containing Inf raises
        :class:`NumericalSanityError`."""
        time_arr = np.array([100, 101, 102], dtype=np.int64)
        values = np.array([[1.0], [np.inf], [3.0]], dtype=np.float32)
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(time_arr),
            values=values,
            columns=["y"],
            static_covariates=pd.DataFrame({"country_id": [1.0]}),
            freq=1,
        )
        with pytest.raises(NumericalSanityError, match="Inf"):
            ReproducibilityGate.Data.audit_numerical_sanity([ts], "targets")

    def test_audit_numerical_sanity_extreme_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A TimeSeries with |val|>1e9 logs a warning but does not raise."""
        time_arr = np.array([100, 101, 102], dtype=np.int64)
        values = np.array([[1.0], [2e9], [3.0]], dtype=np.float32)
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(time_arr),
            values=values,
            columns=["y"],
            static_covariates=pd.DataFrame({"country_id": [1.0]}),
            freq=1,
        )
        with caplog.at_level(logging.WARNING, logger="views_r2darts2.infrastructure.reproducibility_gate"):
            # Must not raise.
            ReproducibilityGate.Data.audit_numerical_sanity([ts], "targets")
        assert any(
            "Extreme value" in record.message for record in caplog.records
        ), f"Expected 'Extreme value' warning, got: {[r.message for r in caplog.records]}"

    def test_lock_entropy_seeds_torch(self) -> None:
        """After :meth:`lock_entropy`, two consecutive ``torch.randn`` calls
        produce bit-identical arrays."""
        ReproducibilityGate.Data.lock_entropy(42)
        first = torch.randn(5).clone()
        ReproducibilityGate.Data.lock_entropy(42)
        second = torch.randn(5).clone()
        np.testing.assert_array_equal(
            first.numpy(), second.numpy()
        )
