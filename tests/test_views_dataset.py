"""Tests for :class:`views_r2darts2.data.views_dataset.ViewsDatasetDarts`.

Exercises the FeatureFrame-backed data boundary against the session-scoped
synthetic country-month parquet fixture (200 countries × 100 months = 20,000
rows, ``country_id`` entity column, month_id range 121..220). The synthetic
fixture mirrors the schema of the real ``validation_viewser_df.parquet`` but
with dummy data so the suite runs anywhere without the real parquet — see
``tests/conftest.py`` for the fixture API.

``pandas`` is used only at the Darts boundary (for
``pd.Index``/``pd.DataFrame`` construction in :class:`TimeSeries`), mirroring
the production package's confinement rule.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from darts import TimeSeries

from views_frames import (
    FeatureFrame,
    SpatialLevel,
    SpatioTemporalIndex,
)
from views_r2darts2.data.parquet_loader import load_views_parquet
from views_r2darts2.data.views_dataset import ViewsDatasetDarts

# Three targets + six features used throughout the suite. These match the
# synthetic column vocabulary in ``tests/conftest.py``.
TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
FEATURES: list[str] = [
    "lr_ged_sb_delta",
    "lr_ged_ns_delta",
    "lr_ged_os_delta",
    "lr_splag_1_ged_sb",
    "lr_splag_1_ged_ns",
    "lr_splag_1_ged_os",
]

# Synthetic-parquet geometry constants (mirrors conftest.py).
N_ROWS: int = 20_000  # 200 countries × 100 months
N_ENTITIES: int = 200
N_TIME_STEPS: int = 100  # month_id 121..220 inclusive
MONTH_ID_MIN: int = 121
MONTH_ID_MAX: int = 220


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(synthetic_cm_parquet_small: Path) -> ViewsDatasetDarts:
    """Load the synthetic cm parquet into a :class:`ViewsDatasetDarts`."""
    frame, feats, targs = load_views_parquet(
        synthetic_cm_parquet_small, targets=TARGETS, features=FEATURES
    )
    return ViewsDatasetDarts(
        feature_frame=frame,
        targets=targs,
        features=feats,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestViewsDatasetLoad:
    """Top-level frame-load sanity tests."""

    def test_load_full_dataset(self, dataset: ViewsDatasetDarts) -> None:
        """Frame must report 20000 rows / 200 entities / CM level."""
        assert dataset.n_rows == N_ROWS
        assert dataset.n_entities == N_ENTITIES
        assert dataset.targets == TARGETS
        assert dataset.features == FEATURES
        assert dataset.level == SpatialLevel.CM
        assert dataset.time_id == "month_id"
        assert dataset.entity_id == "country_id"
        # n_time_steps = 220 - 121 + 1 = 100
        assert dataset.n_time_steps == N_TIME_STEPS


class TestViewsDatasetSubset:
    """``get_subset_arrays`` slicing tests."""

    def test_get_subset_arrays_by_entity(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Subsetting by entity_ids yields the matching row count."""
        time_arr, entity_arr, values_2d = dataset.get_subset_arrays(
            entity_ids=[1, 2, 3]
        )
        # Each of entities 1, 2, 3 has all 100 months → 300 rows.
        assert time_arr.shape[0] == 300
        assert entity_arr.shape[0] == 300
        assert values_2d.shape == (300, len(FEATURES) + len(TARGETS))
        assert set(np.unique(entity_arr).tolist()) == {1, 2, 3}
        # Time column dtype is int64.
        assert time_arr.dtype == np.int64
        assert entity_arr.dtype == np.int64
        assert values_2d.dtype == np.float32

    def test_get_subset_arrays_by_time(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Subsetting by time_ids restricts to the requested window."""
        # Pick a contiguous 12-month window.
        time_arr, entity_arr, values_2d = dataset.get_subset_arrays(
            time_ids=list(range(121, 133))
        )
        # All 200 entities have all 100 months (no missing entities, no
        # single-row entities in the synthetic cm parquet) → 200 × 12 = 2400
        # rows. Verify each row is in [121, 132].
        assert time_arr.min() == 121
        assert time_arr.max() == 132
        assert time_arr.shape[0] == 2400
        # The values array must align with the time array.
        assert values_2d.shape[0] == time_arr.shape[0]
        assert values_2d.shape[1] == len(FEATURES) + len(TARGETS)

    def test_get_subset_arrays_by_entity_and_time(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Combined entity + time filter restricts both axes."""
        time_arr, entity_arr, values_2d = dataset.get_subset_arrays(
            entity_ids=[1, 2], time_ids=[121, 122, 123]
        )
        # 2 entities × 3 months = 6 rows.
        assert time_arr.shape[0] == 6
        assert set(np.unique(entity_arr).tolist()) == {1, 2}
        assert set(np.unique(time_arr).tolist()) == {121, 122, 123}


class TestViewsDatasetAsDarts:
    """``as_darts_timeseries`` end-to-end tests."""

    def test_as_darts_timeseries_basic(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Three known multi-row entities yield 3 TimeSeries of equal length."""
        series_list = dataset.as_darts_timeseries(entity_ids=[1, 2, 3])
        assert len(series_list) == 3
        for ts in series_list:
            assert isinstance(ts, TimeSeries)
            assert len(ts) == N_TIME_STEPS  # all 100 months present
            # Components = features + targets.
            assert list(ts.components) == [*FEATURES, *TARGETS]
            # Time index spans the full month_id range.
            assert int(ts.time_index.min()) == MONTH_ID_MIN
            assert int(ts.time_index.max()) == MONTH_ID_MAX
            # Static covariate carries the entity id.
            assert "country_id" in ts.static_covariates.columns

    def test_as_darts_timeseries_all_entities(
        self,
        dataset: ViewsDatasetDarts,
        synthetic_cm_parquet_small: Path,
    ) -> None:
        """All 200 entities — regression for single-row entity 248 bug."""
        series_list = dataset.as_darts_timeseries()
        assert len(series_list) == N_ENTITIES
        # Every series must be non-empty and carry the entity id.
        entity_ids: list[int] = []
        for ts in series_list:
            assert len(ts) >= 1
            assert "country_id" in ts.static_covariates.columns
            entity_ids.append(int(ts.static_covariates["country_id"].iloc[0]))
        # The set of entity ids must match the parquet's unique country_id set.
        parquet_entities = set(
            pq.read_table(synthetic_cm_parquet_small, columns=["country_id"])
            .column("country_id")
            .to_numpy()
            .tolist()
        )
        assert set(entity_ids) == parquet_entities

    def test_as_darts_timeseries_single_row_entity(self) -> None:
        """A single-row entity (1 row at month 140) must build a valid 1-step series.

        The synthetic cm parquet has every country with all 100 months (no
        single-row entities), so we build a synthetic :class:`FeatureFrame`
        inline with one entity (id 248) having a single row at month_id 140.
        This preserves the regression coverage for the Darts 1-step
        TimeSeries construction bug (``freq=1`` must be passed explicitly to
        bypass Darts' empty-diff step-inference failure on 1-element time
        arrays).
        """
        feat = "lr_ged_sb_delta"
        target = "lr_ged_sb"
        time = np.array([140], dtype=np.int64)
        entity = np.array([248], dtype=np.int64)
        values = np.zeros((1, 2, 1), dtype=np.float32)
        index = SpatioTemporalIndex(
            time=time, unit=entity, level=SpatialLevel.CM
        )
        frame = FeatureFrame(values, index=index, feature_names=[feat, target])
        ds = ViewsDatasetDarts(
            feature_frame=frame,
            targets=[target],
            features=[feat],
        )
        series_list = ds.as_darts_timeseries(entity_ids=[248])
        assert len(series_list) == 1
        ts = series_list[0]
        assert len(ts) == 1
        assert int(ts.time_index.values[0]) == 140
        assert list(ts.components) == [feat, target]
        assert int(ts.static_covariates["country_id"].iloc[0]) == 248

    def test_as_darts_timeseries_with_cyclic_encoders(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """``use_cyclic_encoders=True`` appends month_sin/month_cos columns."""
        series_list = dataset.as_darts_timeseries(
            entity_ids=[1], use_cyclic_encoders=True
        )
        assert len(series_list) == 1
        ts = series_list[0]
        components = list(ts.components)
        # Original columns plus two cyclic columns.
        assert components == [
            *FEATURES,
            *TARGETS,
            "month_sin",
            "month_cos",
        ]
        # Cyclic values must lie in [-1, 1].
        arr = ts.all_values(copy=False)
        sin_idx = components.index("month_sin")
        cos_idx = components.index("month_cos")
        assert np.all(np.abs(arr[:, sin_idx, :]) <= 1.0 + 1e-6)
        assert np.all(np.abs(arr[:, cos_idx, :]) <= 1.0 + 1e-6)

    def test_as_darts_timeseries_with_static_covariates(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """``inject_static_covariates=True`` attaches the 5-stat fingerprint."""
        series_list = dataset.as_darts_timeseries(
            entity_ids=[1, 2, 3],
            inject_static_covariates=True,
            stat_time_range=(MONTH_ID_MIN, MONTH_ID_MAX),
        )
        assert len(series_list) == 3
        for ts in series_list:
            static_cols = set(ts.static_covariates.columns)
            # Entity id + 5 stats × 3 targets = 16 fingerprint columns.
            assert "country_id" in static_cols
            for tgt in TARGETS:
                for stat in ("mu", "sigma", "max", "trend", "sparsity"):
                    assert f"{tgt}_{stat}" in static_cols
            # Sparsity is in [0, 1].
            for tgt in TARGETS:
                key = f"{tgt}_sparsity"
                val = float(ts.static_covariates[key].iloc[0])
                assert 0.0 <= val <= 1.0

    def test_as_darts_timeseries_static_covariate_transform(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """``AsinhTransform->MaxAbsScaler`` fingerprint: values bounded in [-1, 1]."""
        # Use the first 20 entities actually present in the parquet. The
        # synthetic cm parquet has all country_ids 1..200 with full 100-month
        # coverage, so the first 20 are simply [1, 2, ..., 20].
        all_entities = sorted(
            set(np.unique(dataset.feature_frame.index.unit).tolist())
        )
        entity_subset = all_entities[:20]
        series_list = dataset.as_darts_timeseries(
            entity_ids=entity_subset,
            inject_static_covariates=True,
            stat_time_range=(MONTH_ID_MIN, MONTH_ID_MAX),
            static_cov_transform="AsinhTransform->MaxAbsScaler",
        )
        assert len(series_list) == 20
        # After MaxAbsScaler, every transformed stat (mu/sigma/max/trend) for
        # every target must lie in [-1, 1]. Sparsity is NOT scaled.
        for ts in series_list:
            for tgt in TARGETS:
                for stat in ("mu", "sigma", "max", "trend"):
                    key = f"{tgt}_{stat}"
                    val = float(ts.static_covariates[key].iloc[0])
                    assert -1.0 - 1e-5 <= val <= 1.0 + 1e-5, (
                        f"{key}={val} outside [-1, 1]"
                    )
                # Sparsity stays in [0, 1] (never transformed).
                sparsity_val = float(ts.static_covariates[f"{tgt}_sparsity"].iloc[0])
                assert 0.0 <= sparsity_val <= 1.0

    def test_as_darts_timeseries_empty_subset_returns_empty_list(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """A time_id filter that matches no rows returns an empty list."""
        series_list = dataset.as_darts_timeseries(time_ids=[99999])
        assert series_list == []


class TestViewsDatasetFactory:
    """``from_views_path`` factory tests."""

    def test_dataset_from_views_path(
        self,
        synthetic_cm_parquet_small: Path,
        tmp_path: Path,
    ) -> None:
        """The factory loads via ``path_raw`` + ``run_type`` + config dict."""
        # The factory looks for ``<run_type>_viewser_df.parquet`` inside
        # ``path_raw`` — copy the synthetic cm parquet into the tmp dir under
        # the expected filename so the factory can find it.
        target_path = tmp_path / "validation_viewser_df.parquet"
        shutil.copy(synthetic_cm_parquet_small, target_path)

        config: dict = {
            "targets": TARGETS,
            "features": FEATURES,
            "time_id": "month_id",
            "entity_id": "country_id",
        }
        ds = ViewsDatasetDarts.from_views_path(
            path_raw=tmp_path,
            run_type="validation",
            config=config,
        )
        assert ds.n_rows == N_ROWS
        assert ds.n_entities == N_ENTITIES
        assert ds.targets == TARGETS
        assert ds.features == FEATURES
        assert ds.level == SpatialLevel.CM

    def test_dataset_from_views_path_uses_regression_targets_key(
        self,
        synthetic_cm_parquet_small: Path,
        tmp_path: Path,
    ) -> None:
        """Task-split target keys are accepted without legacy ``targets``."""
        target_path = tmp_path / "validation_viewser_df.parquet"
        shutil.copy(synthetic_cm_parquet_small, target_path)

        config: dict = {
            "regression_targets": TARGETS,
            "features": FEATURES,
            "time_id": "month_id",
            "entity_id": "country_id",
        }
        ds = ViewsDatasetDarts.from_views_path(
            path_raw=tmp_path,
            run_type="validation",
            config=config,
        )
        assert ds.targets == TARGETS


class TestViewsDatasetValidation:
    """Constructor-time validation errors."""

    @staticmethod
    def _load_frame(parquet_path: Path) -> tuple:
        """Load the parquet once for the validation tests."""
        return load_views_parquet(
            parquet_path, targets=TARGETS, features=FEATURES
        )

    def test_dataset_rejects_target_feature_overlap(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """A column in both ``targets`` and ``features`` must raise."""
        frame, _, _ = self._load_frame(synthetic_cm_parquet_small)
        with pytest.raises(ValueError, match="both target and feature"):
            ViewsDatasetDarts(
                feature_frame=frame,
                targets=["lr_ged_sb"],
                features=["lr_ged_sb"],
            )

    def test_dataset_rejects_missing_target_column(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """A target column absent from the frame must raise."""
        frame, _, _ = self._load_frame(synthetic_cm_parquet_small)
        with pytest.raises(ValueError, match="missing target columns"):
            ViewsDatasetDarts(
                feature_frame=frame,
                targets=["nonexistent_target"],
                features=FEATURES,
            )

    def test_dataset_rejects_missing_feature_column(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """A feature column absent from the frame must raise."""
        frame, _, _ = self._load_frame(synthetic_cm_parquet_small)
        with pytest.raises(ValueError, match="missing feature columns"):
            ViewsDatasetDarts(
                feature_frame=frame,
                targets=TARGETS,
                features=["nonexistent_feature"],
            )

    def test_dataset_rejects_empty_targets(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """An empty targets list must raise at construction."""
        frame, _, _ = self._load_frame(synthetic_cm_parquet_small)
        with pytest.raises(ValueError, match="non-empty"):
            ViewsDatasetDarts(feature_frame=frame, targets=[])


# ----------------------------------------------------------------------
# Helpers (re-exported for downstream tests if needed)
# ----------------------------------------------------------------------


def build_test_timeseries(
    *,
    time_ids: Sequence[int],
    values: np.ndarray,
    columns: Sequence[str],
    entity_id_name: str = "country_id",
    entity_id_value: int = 1,
) -> TimeSeries:
    """Construct a Darts :class:`TimeSeries` from numpy arrays (test helper).

    Mirrors :func:`views_r2darts2.transformers.darts_bridge.build_entity_timeseries`
    but is duplicated here so the test file is hermetic.

    Args:
        time_ids: Integer time identifiers (must be sorted ascending).
        values: 2-D float32 array of shape ``(T, F)``.
        columns: Component (column) names.
        entity_id_name: Static-covariate column name for the entity id.
        entity_id_value: Integer entity id for this series.

    Returns:
        A Darts :class:`TimeSeries` carrying the entity id as a static covariate.
    """
    time_arr = np.asarray(time_ids, dtype=np.int64)
    return TimeSeries.from_times_and_values(
        times=pd.Index(time_arr),
        values=np.asarray(values, dtype=np.float32),
        columns=list(columns),
        static_covariates=pd.DataFrame({entity_id_name: [float(entity_id_value)]}),
        freq=1,
    )
