"""Tests for :class:`views_r2darts2.models.markov_model.MarkovModel`.

Exercises:
    * Construction: validation of markov_method / regression_method / steps.
    * Markov state computation (parity with the original R-derived logic).
    * Numpy-only fit/predict (single-target and multivariate).
    * Save/load round-trip — predictions are bit-identical after reload.
    * Integration with the catalog + reproducibility gate (sklearn bypass).
    * Integration with :class:`DartsForecaster` + :class:`ViewsDataset`
      (end-to-end pipeline test on a synthetic parquet).

The Markov model is the one sklearn-based exception in a torch-only
catalog. These tests verify that the model is faithful to the original
pandas-based implementation while removing all pandas dependencies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd  # noqa: WPS433 — allowed at the Darts TimeSeries boundary
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from darts import TimeSeries

from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.infrastructure.reproducibility_gate import (
    ReproducibilityGate,
    MissingHyperparameterError,
    ArchitectureMismatchError,
)
from views_r2darts2.models.markov_model import (
    MarkovFatalityModel,
    MarkovModel,
    MarkovState,
    MarkovStateModel,
)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

TARGETS: list[str] = ["lr_ged_sb"]
MULTI_TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns"]
FEATURES: list[str] = ["feat_a", "feat_b"]

PARTITION: dict[str, tuple[int, int]] = {
    "train": (121, 200),
    "test": (201, 220),
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_series(
    *,
    n_entities: int = 3,
    n_months: int = 60,
    start_month: int = 121,
    seed: int = 42,
    targets: list[str] | None = None,
    features: list[str] | None = None,
    zero_inflation: float = 0.7,
) -> list[TimeSeries]:
    """Build a list of Darts ``TimeSeries``, one per entity.

    Each series carries the target columns + feature columns. Fatality
    counts are zero-inflated log-normal — most rows are 0, the rest are
    log-normal noise. This gives the Markov-state classifier enough
    escalation / war samples to fit without being trivial.
    """
    if targets is None:
        targets = TARGETS
    if features is None:
        features = FEATURES
    rng = np.random.default_rng(seed)
    series_list: list[TimeSeries] = []
    for e in range(1, n_entities + 1):
        time_ids = np.arange(
            start_month, start_month + n_months, dtype=np.int64
        )
        cols: list[np.ndarray] = []
        for tname in targets:
            mask = rng.random(n_months) < (1.0 - zero_inflation)
            vals = np.zeros(n_months, dtype=np.float32)
            vals[mask] = rng.lognormal(
                mean=2.0, sigma=1.5, size=mask.sum()
            ).astype(np.float32)
            cols.append(vals)
        for fname in features:
            cols.append(
                rng.normal(0, 1, n_months).astype(np.float32)
                + 0.1 * cols[0]  # correlate feature with primary target
            )
        values = np.stack(cols, axis=1)  # (T, n_cols)
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(time_ids),
            values=values,
            columns=targets + features,
            static_covariates=pd.DataFrame({"country_id": [float(e)]}),
            freq=1,
        )
        series_list.append(ts)
    return series_list


def _make_markov_config(
    *,
    targets: list[str] | None = None,
    markov_target: str = "lr_ged_sb",
    steps: list[int] | None = None,
    rf_n_estimators: int = 10,
    markov_method: str = "direct",
    regression_method: str = "single",
) -> dict:
    """Build a minimal MarkovModel config that passes the reproducibility gate."""
    if targets is None:
        targets = TARGETS
    if steps is None:
        steps = [1, 2, 3]
    return {
        "algorithm": "MarkovModel",
        "name": "markov_test",
        "run_type": "test",
        "random_state": 42,
        "steps": steps,
        "regression_targets": list(targets),
        "markov_target": markov_target,
        "markov_method": markov_method,
        "regression_method": regression_method,
        "markov_threshold": 0,
        "n_jobs": 1,  # deterministic for tests
        "rf_class_params": {"n_estimators": rf_n_estimators},
        "rf_reg_params": {
            "n_estimators": rf_n_estimators,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
        },
    }


# ----------------------------------------------------------------------
# MarkovState enum
# ----------------------------------------------------------------------


class TestMarkovState:
    def test_states(self) -> None:
        assert MarkovState.PEACE.value == "peace"
        assert MarkovState.DESC.value == "desc"
        assert MarkovState.ESC.value == "esc"
        assert MarkovState.WAR.value == "war"

    def test_is_str_enum(self) -> None:
        assert isinstance(MarkovState.PEACE, str)
        assert MarkovState.PEACE == "peace"


# ----------------------------------------------------------------------
# MarkovModel construction / validation
# ----------------------------------------------------------------------


class TestMarkovModelConstruction:
    def test_basic_construction(self) -> None:
        m = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
        )
        assert m._steps == [1, 2, 3]
        assert m._targets == TARGETS
        assert m._markov_target == "lr_ged_sb"
        assert m._markov_method == "direct"
        assert m._regression_method == "single"
        assert m._markov_threshold == 0
        assert m.is_fitted_ is False
        # Darts interface plumbing
        assert m.input_chunk_length == 1
        assert m.output_chunk_length == 3

    def test_invalid_markov_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid markov_method"):
            MarkovModel(
                steps=[1],
                targets=TARGETS,
                markov_target="lr_ged_sb",
                markov_method="invalid",
            )

    def test_invalid_regression_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid regression_method"):
            MarkovModel(
                steps=[1],
                targets=TARGETS,
                markov_target="lr_ged_sb",
                regression_method="invalid",
            )

    def test_empty_targets_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            MarkovModel(steps=[1], targets=[], markov_target="lr_ged_sb")

    def test_non_list_targets_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            MarkovModel(steps=[1], targets="lr_ged_sb", markov_target="lr_ged_sb")

    def test_invalid_step_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Steps must be"):
            MarkovModel(
                steps="1,2,3",  # type: ignore[arg-type]
                targets=TARGETS,
                markov_target="lr_ged_sb",
            )

    def test_non_positive_step_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integers"):
            MarkovModel(
                steps=[0, 1, 2],
                targets=TARGETS,
                markov_target="lr_ged_sb",
            )

    def test_step_above_36_warns(self) -> None:
        with pytest.warns(UserWarning, match="higher than 36"):
            MarkovModel(
                steps=[1, 48],
                targets=TARGETS,
                markov_target="lr_ged_sb",
            )

    def test_default_rf_params_match_ranger(self) -> None:
        """Default RF params should match the Ranger R-package defaults
        (n_estimators=500, max_features='sqrt', min_samples_leaf=5 for
        regression; n_estimators=500 for classification)."""
        m = MarkovModel(steps=[1], targets=TARGETS, markov_target="lr_ged_sb")
        assert m._rf_class_params["n_estimators"] == 500
        assert m._rf_reg_params["n_estimators"] == 500
        assert m._rf_reg_params["max_features"] == "sqrt"
        assert m._rf_reg_params["min_samples_leaf"] == 5

    def test_custom_rf_params_override_defaults(self) -> None:
        m = MarkovModel(
            steps=[1],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            rf_class_params={"n_estimators": 50, "max_depth": 5},
            rf_reg_params={"n_estimators": 100, "min_samples_leaf": 10},
        )
        assert m._rf_class_params["n_estimators"] == 50
        assert m._rf_class_params["max_depth"] == 5
        assert m._rf_reg_params["n_estimators"] == 100
        assert m._rf_reg_params["min_samples_leaf"] == 10


# ----------------------------------------------------------------------
# Markov state computation parity
# ----------------------------------------------------------------------


class TestMarkovStateComputation:
    """Verify the numpy-based ``_add_markov_states`` matches the original
    R-derived logic from the fatalities002 pipeline."""

    def test_peace_state(self) -> None:
        """Both t and t-1 below threshold → PEACE."""
        m = MarkovModel(steps=[1], targets=TARGETS, markov_target="x")
        values = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        time_ids = np.array([1, 2], dtype=np.int64)
        entity_ids = np.array([1, 1], dtype=np.int64)
        states = m._add_markov_states(values, time_ids, entity_ids, target_idx=0)
        # First row has no t-1 → None; second row is PEACE.
        assert states[0] is None
        assert states[1] == MarkovState.PEACE

    def test_desc_state(self) -> None:
        """t below threshold, t-1 above → DESC (de-escalation)."""
        m = MarkovModel(steps=[1], targets=TARGETS, markov_target="x")
        values = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        time_ids = np.array([1, 2], dtype=np.int64)
        entity_ids = np.array([1, 1], dtype=np.int64)
        states = m._add_markov_states(values, time_ids, entity_ids, target_idx=0)
        assert states[0] is None
        assert states[1] == MarkovState.DESC

    def test_esc_state(self) -> None:
        """t above threshold, t-1 below → ESC (escalation)."""
        m = MarkovModel(steps=[1], targets=TARGETS, markov_target="x")
        values = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        time_ids = np.array([1, 2], dtype=np.int64)
        entity_ids = np.array([1, 1], dtype=np.int64)
        states = m._add_markov_states(values, time_ids, entity_ids, target_idx=0)
        assert states[0] is None
        assert states[1] == MarkovState.ESC

    def test_war_state(self) -> None:
        """Both t and t-1 above threshold → WAR."""
        m = MarkovModel(steps=[1], targets=TARGETS, markov_target="x")
        values = np.array([[5.0, 0.0], [7.0, 0.0]], dtype=np.float32)
        time_ids = np.array([1, 2], dtype=np.int64)
        entity_ids = np.array([1, 1], dtype=np.int64)
        states = m._add_markov_states(values, time_ids, entity_ids, target_idx=0)
        assert states[0] is None
        assert states[1] == MarkovState.WAR

    def test_first_row_per_entity_is_none(self) -> None:
        """The first row of each entity has no t-1 → state is None."""
        m = MarkovModel(steps=[1], targets=TARGETS, markov_target="x")
        values = np.array(
            [[0.0, 0.0], [5.0, 0.0], [0.0, 0.0], [5.0, 0.0]],
            dtype=np.float32,
        )
        time_ids = np.array([1, 2, 1, 2], dtype=np.int64)
        entity_ids = np.array([1, 1, 2, 2], dtype=np.int64)
        states = m._add_markov_states(values, time_ids, entity_ids, target_idx=0)
        # Entity 1: rows 0,1. Entity 2: rows 2,3.
        assert states[0] is None
        assert states[1] == MarkovState.ESC
        assert states[2] is None
        assert states[3] == MarkovState.ESC

    def test_nonzero_threshold_warns(self) -> None:
        """A non-zero threshold emits a UserWarning."""
        m = MarkovModel(
            steps=[1],
            targets=TARGETS,
            markov_target="x",
            markov_threshold=5,
        )
        values = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        time_ids = np.array([1, 2], dtype=np.int64)
        entity_ids = np.array([1, 1], dtype=np.int64)
        with pytest.warns(UserWarning, match="Non-zero threshold"):
            m._add_markov_states(values, time_ids, entity_ids, target_idx=0)


# ----------------------------------------------------------------------
# MarkovStateModel (state classifier) — numpy-only
# ----------------------------------------------------------------------


class TestMarkovStateModel:
    def test_fit_and_predict(self) -> None:
        """Fit on a small synthetic matrix, predict returns probabilities."""
        rng = np.random.default_rng(42)
        n = 50
        values = rng.normal(0, 1, (n, 3)).astype(np.float32)
        time_ids = np.arange(1, n + 1, dtype=np.int64)
        entity_ids = np.ones(n, dtype=np.int64)
        # Build markov states cyclically.
        states = np.array(
            [
                MarkovState([MarkovState.PEACE, MarkovState.ESC, MarkovState.WAR, MarkovState.DESC][i % 4])
                for i in range(n)
            ],
            dtype=object,
        )
        sm = MarkovStateModel(
            step=1,
            train_start=1,
            train_end=n,
            rf_class_params={"n_estimators": 5},
            n_jobs=1,
        )
        sm.fit(
            values=values,
            time_ids=time_ids,
            entity_ids=entity_ids,
            markov_state=states,
            markov_target=values[:, 0],
            feature_idx=np.array([0, 1, 2], dtype=np.int64),
        )
        assert sm.is_fitted_ is True
        assert len(sm.models) == 4  # one per state
        # Predict
        probs = sm.predict(values=values[:5], start_state=MarkovState.PEACE)
        assert probs.shape[0] == 5
        # Probabilities sum to 1 per row.
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ----------------------------------------------------------------------
# MarkovFatalityModel (fatality regressor) — numpy-only
# ----------------------------------------------------------------------


class TestMarkovFatalityModel:
    def test_fit_and_predict(self) -> None:
        """Fit on small synthetic matrix, predict returns scalar per row."""
        rng = np.random.default_rng(42)
        n = 50
        values = rng.normal(0, 1, (n, 3)).astype(np.float32)
        time_ids = np.arange(1, n + 1, dtype=np.int64)
        entity_ids = np.ones(n, dtype=np.int64)
        states = np.array(
            [
                MarkovState([MarkovState.PEACE, MarkovState.ESC, MarkovState.WAR, MarkovState.DESC][i % 4])
                for i in range(n)
            ],
            dtype=object,
        )
        fm = MarkovFatalityModel(
            step=1,
            train_start=1,
            train_end=n,
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        fm.fit(
            values=values,
            time_ids=time_ids,
            entity_ids=entity_ids,
            markov_state=states,
            fatalities_target=values[:, 0].astype(np.float64),
            feature_idx=np.array([0, 1, 2], dtype=np.int64),
        )
        assert fm.is_fitted_ is True
        assert len(fm.models) == 2  # ESC + WAR only
        # Predict for ESC and WAR
        preds_esc = fm.predict(values=values[:5], start_state=MarkovState.ESC)
        preds_war = fm.predict(values=values[:5], start_state=MarkovState.WAR)
        assert preds_esc.shape == (5,)
        assert preds_war.shape == (5,)


# ----------------------------------------------------------------------
# MarkovModel — fit / predict (single target)
# ----------------------------------------------------------------------


class TestMarkovModelFitPredict:
    def test_fit_sets_is_fitted(self) -> None:
        series = _make_series()
        m = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        assert m.is_fitted_ is False
        m.fit(series=series)
        assert m.is_fitted_ is True
        # State models: 3 (one per step, direct method)
        assert len(m._state_models) == 3
        # Fatality models: 1 (regression_method='single' → only step=1)
        assert len(m._fatality_models) == 1

    def test_predict_returns_list_of_timeseries(self) -> None:
        series = _make_series()
        m = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        m.fit(series=series)
        preds = m.predict(n=3, series=series)
        assert isinstance(preds, list)
        assert len(preds) == 3  # one per entity
        for ts in preds:
            assert isinstance(ts, TimeSeries)
            # 3 steps × 1 target
            assert ts.values().shape == (3, 1)

    def test_predict_before_fit_raises(self) -> None:
        m = MarkovModel(
            steps=[1],
            targets=TARGETS,
            markov_target="lr_ged_sb",
        )
        series = _make_series()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict(n=1, series=series)

    def test_predictions_are_non_negative(self) -> None:
        """Fatalities can't be negative — the predict path clips to 0."""
        series = _make_series(zero_inflation=0.4)  # more events
        m = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        m.fit(series=series)
        preds = m.predict(n=3, series=series)
        for ts in preds:
            assert (ts.values() >= 0).all()

    def test_predictions_are_finite(self) -> None:
        """No NaN / Inf in predictions."""
        series = _make_series()
        m = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        m.fit(series=series)
        preds = m.predict(n=3, series=series)
        for ts in preds:
            arr = ts.values()
            assert np.isfinite(arr).all()


# ----------------------------------------------------------------------
# MarkovModel — multivariate forecasting
# ----------------------------------------------------------------------


class TestMarkovModelMultivariate:
    def test_multivariate_fit_predict(self) -> None:
        """Multiple targets: separate fatality models per target, shared
        state classifier."""
        series = _make_series(targets=MULTI_TARGETS)
        m = MarkovModel(
            steps=[1, 2],
            targets=MULTI_TARGETS,
            markov_target="lr_ged_sb",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        m.fit(series=series)
        assert m.is_fitted_ is True
        # State models: 2 (one per step, direct method)
        assert len(m._state_models) == 2
        # Fatality models per target: 2 targets × 1 step (single) = 2
        assert len(m._fatality_models_per_target) == 2
        for tgt in MULTI_TARGETS:
            assert tgt in m._fatality_models_per_target
            assert len(m._fatality_models_per_target[tgt]) == 1

        preds = m.predict(n=2, series=series)
        assert isinstance(preds, list)
        assert len(preds) == 3  # 3 entities
        for ts in preds:
            # 2 steps × 2 targets
            assert ts.values().shape == (2, 2)
            assert list(ts.components) == MULTI_TARGETS

    def test_multivariate_with_past_covariates(self) -> None:
        """Targets in ``series``, features in ``past_covariates``."""
        full = _make_series(targets=MULTI_TARGETS)
        target_series = [ts[MULTI_TARGETS] for ts in full]
        cov_series = [ts[FEATURES] for ts in full]

        m = MarkovModel(
            steps=[1],
            targets=MULTI_TARGETS,
            markov_target="lr_ged_sb",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        m.fit(series=target_series, past_covariates=cov_series)
        preds = m.predict(n=1, series=target_series, past_covariates=cov_series)
        assert isinstance(preds, list)
        assert len(preds) == 3
        for ts in preds:
            assert ts.values().shape == (1, 2)


# ----------------------------------------------------------------------
# MarkovModel — markov_method and regression_method parity
# ----------------------------------------------------------------------


class TestMarkovMethods:
    def test_transition_method_fits_only_step_1_state_model(self) -> None:
        """``markov_method='transition'`` fits only the step-1 state model
        and applies the transition matrix power for higher steps."""
        series = _make_series()
        m = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            markov_method="transition",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        m.fit(series=series)
        # Only step 1 state model fitted.
        assert list(m._state_models.keys()) == [1]
        # Still produces a 3-step forecast.
        preds = m.predict(n=3, series=series)
        assert len(preds) == 3
        for ts in preds:
            assert ts.values().shape == (3, 1)

    def test_multi_regression_fits_one_model_per_step(self) -> None:
        """``regression_method='multi'`` fits a separate fatality model
        per step (vs. a single step-1 model for 'single')."""
        series = _make_series()
        m = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            regression_method="multi",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        m.fit(series=series)
        assert len(m._fatality_models) == 3  # one per step


# ----------------------------------------------------------------------
# MarkovModel — save / load
# ----------------------------------------------------------------------


class TestMarkovModelSaveLoad:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        series = _make_series()
        m = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        m.fit(series=series)
        preds_before = m.predict(n=3, series=series)
        assert isinstance(preds_before, list)

        path = tmp_path / "markov_model.pkl"
        m.save(path=str(path))
        assert path.exists()

        loaded = MarkovModel.load(str(path))
        assert loaded.is_fitted_ is True
        assert loaded._steps == [1, 2, 3]
        assert loaded._targets == TARGETS
        preds_after = loaded.predict(n=3, series=series)
        assert isinstance(preds_after, list)

        # Bit-identical predictions after save/load.
        for a, b in zip(preds_before, preds_after):
            np.testing.assert_array_equal(a.values(), b.values())


# ----------------------------------------------------------------------
# Reproducibility gate — sklearn bypass
# ----------------------------------------------------------------------


class TestReproducibilityGateSklearnBypass:
    """Tests for the sklearn-model bypass in
    :class:`ReproducibilityGate.Config`."""

    def test_audit_manifest_passes_with_minimal_markov_config(self) -> None:
        """A minimal MarkovModel config (no optimizer/scheduler/loss) passes
        the audit — the gate skips the torch-specific genome checks."""
        config = _make_markov_config()
        ReproducibilityGate.Config.audit_manifest(config)  # no raise

    def test_audit_manifest_missing_markov_target_raises(self) -> None:
        config = _make_markov_config()
        del config["markov_target"]
        with pytest.raises(MissingHyperparameterError, match="markov_target"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_missing_target_raises(self) -> None:
        config = _make_markov_config()
        del config["regression_targets"]
        with pytest.raises(MissingHyperparameterError, match="regression_targets"):
            ReproducibilityGate.Config.audit_manifest(config)

    def test_audit_manifest_does_not_require_loss_function(self) -> None:
        """MarkovModel config has no ``loss_function`` key — the audit must
        not raise (the torch CORE_GENOME includes loss_function, but
        SKLEARN_CORE_GENOME does not)."""
        config = _make_markov_config()
        assert "loss_function" not in config
        assert "optimizer_cls" not in config
        assert "lr_scheduler_cls" not in config
        assert "batch_size" not in config
        assert "n_epochs" not in config
        ReproducibilityGate.Config.audit_manifest(config)  # no raise

    def test_audit_architecture_skips_output_chunk_length_check(self) -> None:
        """MarkovModel configs do not have ``output_chunk_length`` — the
        architecture audit skips the divisibility check."""
        config = _make_markov_config(steps=[1, 2, 3, 4, 5])  # len=5
        # For a torch model, len(steps)=5 with output_chunk_length=3 would
        # raise ArchitectureMismatchError. For MarkovModel it must NOT raise
        # (no output_chunk_length required).
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


# ----------------------------------------------------------------------
# Catalog integration
# ----------------------------------------------------------------------


class TestMarkovModelCatalogIntegration:
    def test_catalog_creates_markov_model(self) -> None:
        """``ModelCatalog(config).get_model("MarkovModel")`` returns a
        MarkovModel instance with no torch plumbing attached."""
        config = _make_markov_config()
        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog = ModelCatalog(config)
        assert catalog._is_sklearn_model is True
        assert catalog.loss_fn is None
        assert catalog.likelihood is None
        assert catalog.opt_catalog is None
        assert catalog.sched_catalog is None
        model = catalog.get_model("MarkovModel")
        assert isinstance(model, MarkovModel)

    def test_catalog_markov_model_in_list(self) -> None:
        config = _make_markov_config()
        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog = ModelCatalog(config)
        models = catalog.list_models()
        assert "MarkovModel" in models

    def test_catalog_end_to_end_fit_predict(self) -> None:
        """End-to-end: catalog → model → fit → predict."""
        config = _make_markov_config(rf_n_estimators=5)
        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog = ModelCatalog(config)
            model = catalog.get_model("MarkovModel")
        series = _make_series()
        model.fit(series=series)
        preds = model.predict(n=3, series=series)
        assert isinstance(preds, list)
        assert len(preds) == 3


# ----------------------------------------------------------------------
# End-to-end pipeline test (DartsForecaster + ViewsDataset)
# ----------------------------------------------------------------------


def _write_synthetic_parquet(path: Path) -> Path:
    """Write a tiny synthetic parquet file mirroring the VIEWS schema."""
    rng = np.random.default_rng(42)
    n_countries = 5
    n_months = 100
    n_rows = n_countries * n_months
    country_ids = np.repeat(
        np.arange(1, n_countries + 1, dtype=np.int64), n_months
    )
    month_ids = np.tile(
        np.arange(121, 121 + n_months, dtype=np.int64), n_countries
    )
    columns: dict[str, np.ndarray] = {
        "month_id": month_ids,
        "country_id": country_ids,
    }
    for col in TARGETS + FEATURES:
        mask = rng.random(n_rows) < 0.3
        values = np.zeros(n_rows, dtype=np.float64)
        values[mask] = rng.lognormal(mean=2.0, sigma=1.5, size=mask.sum())
        columns[col] = np.maximum(values, 0.0).astype(np.float32)
    table = pa.table(columns)
    pq.write_table(table, str(path))
    return path


class TestEndToEndPipeline:
    """End-to-end: ViewsDataset → ModelCatalog → DartsForecaster → predict."""

    def test_full_pipeline_train_predict(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)

        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )

        config = _make_markov_config(rf_n_estimators=5)
        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog = ModelCatalog(config)
            model = catalog.get_model("MarkovModel")

        forecaster = DartsForecaster(
            dataset=dataset,
            model=model,
            partition_dict=PARTITION,
            target_scaler=None,
            feature_scaler=None,
            random_state=42,
        )
        assert forecaster._is_sklearn_model is True
        assert forecaster.device == "cpu"

        forecaster.train()
        assert forecaster.scaler_fitted is True

        preds = forecaster.predict(sequence_number=0, output_length=3)
        assert isinstance(preds, dict)
        assert set(preds.keys()) == set(TARGETS)
        for tgt, frame in preds.items():
            # 3 steps × 5 entities = 15 rows × 1 sample
            assert frame.values.shape == (15, 1)
            assert np.isfinite(frame.values).all()
            assert (frame.values >= 0).all()

    def test_full_pipeline_save_load(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)

        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )
        config = _make_markov_config(rf_n_estimators=5)
        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog = ModelCatalog(config)
            model = catalog.get_model("MarkovModel")

        forecaster = DartsForecaster(
            dataset=dataset,
            model=model,
            partition_dict=PARTITION,
            target_scaler=None,
            feature_scaler=None,
            random_state=42,
        )
        forecaster.train()
        preds_before = forecaster.predict(sequence_number=0, output_length=3)

        save_path = tmp_path / "markov_model"
        forecaster.save_model(str(save_path))
        assert save_path.exists()
        assert Path(str(save_path) + ".scalers").exists()

        # Build a fresh forecaster and load.
        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog2 = ModelCatalog(config)
            model2 = catalog2.get_model("MarkovModel")
        forecaster2 = DartsForecaster(
            dataset=dataset,
            model=model2,
            partition_dict=PARTITION,
            target_scaler=None,
            feature_scaler=None,
            random_state=42,
        )
        forecaster2.load_model(str(save_path))
        preds_after = forecaster2.predict(sequence_number=0, output_length=3)

        # Bit-identical after save/load.
        for tgt in preds_before:
            np.testing.assert_array_equal(
                preds_before[tgt].values, preds_after[tgt].values
            )

    def test_full_pipeline_multiple_sequences(self, tmp_path: Path) -> None:
        """The forecaster must produce predictions for multiple rolling-origin
        sequences (sequence_number = 0, 1, 2)."""
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)

        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )
        config = _make_markov_config(rf_n_estimators=5)
        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog = ModelCatalog(config)
            model = catalog.get_model("MarkovModel")

        forecaster = DartsForecaster(
            dataset=dataset,
            model=model,
            partition_dict=PARTITION,
            target_scaler=None,
            feature_scaler=None,
            random_state=42,
        )
        forecaster.train()

        for seq in range(3):
            preds = forecaster.predict(sequence_number=seq, output_length=3)
            assert set(preds.keys()) == set(TARGETS)
            for tgt, frame in preds.items():
                assert np.isfinite(frame.values).all()
                assert (frame.values >= 0).all()
