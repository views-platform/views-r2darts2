"""Tests for :class:`views_r2darts2.dataset.base.ViewsDataset`.

Exercises the new zarr-backed dataset against the session-scoped synthetic
country-month parquet fixture (200 countries × 100 months = 20,000 rows,
``country_id`` entity column, month_id range 121..220). The synthetic
fixture mirrors the schema of the real ``validation_viewser_df.parquet`` but
with dummy data so the suite runs anywhere without the real parquet — see
``tests/conftest.py`` for the fixture API.

Covers the new API surface:
    * Construction from a parquet path (``broadcast_features=True``).
    * Subset operations: ``get_subset_tensor`` / ``get_subset_dataset``.
    * Darts bridge: ``to_darts_timeseries`` (with cyclic encoders, with
      entity/time filters, with empty subsets).
    * Frame conversions: ``to_featureframe`` (feature mode),
      ``to_predictionframe`` (prediction mode → raises on feature mode).
    * Empty creation + incremental concatenation: ``create_empty``,
      ``add_row``, ``add_batch``.
    * Scaler integration: ``fit_scalers``, ``scalers_fitted``,
      ``get_scaled_darts_timeseries``, ``ingest_darts_predictions``.
    * Persistence: ``save_parquet`` / ``save_zarr`` / ``save_zarrzip``.
    * Introspection: ``num_entities``, ``num_time_steps``, ``num_features``,
      ``targets``, ``features``, ``_time_id``, ``_entity_id``,
      ``_build_spatial_level``.

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
    PredictionFrame,
    SpatialLevel,
    SpatioTemporalIndex,
)
from views_r2darts2.dataset.base import ViewsDataset

# Three targets + nine features used throughout the suite. These match the
# synthetic column vocabulary in ``tests/conftest.py`` (SYNTHETIC_TARGETS /
# SYNTHETIC_FEATURES).
TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
FEATURES: list[str] = [
    "lr_ged_sb_delta",
    "lr_ged_ns_delta",
    "lr_ged_os_delta",
    "lr_splag_1_ged_sb",
    "lr_splag_1_ged_ns",
    "lr_splag_1_ged_os",
    "lr_decay_ged_sb_1",
    "lr_decay_ged_sb_5",
    "lr_decay_ged_sb_25",
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
def dataset(synthetic_cm_parquet_small: Path) -> ViewsDataset:
    """Load the synthetic cm parquet into a :class:`ViewsDataset`.

    The new API takes the parquet path + ``targets`` + ``broadcast_features``.
    Features are auto-derived from the parquet schema (every numeric column
    that is not a target becomes a feature).
    """
    return ViewsDataset(
        synthetic_cm_parquet_small,
        targets=TARGETS,
        broadcast_features=True,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestViewsDatasetLoad:
    """Top-level load sanity tests."""

    def test_load_full_dataset(self, dataset: ViewsDataset) -> None:
        """Dataset must report 200 entities / 100 time steps / CM level."""
        assert dataset.num_entities == N_ENTITIES
        assert dataset.num_time_steps == N_TIME_STEPS
        assert dataset.targets == TARGETS
        # Features are auto-derived: every numeric column that is not a
        # target. The synthetic parquet has 9 feature columns.
        assert FEATURES == dataset.features  # exact set membership
        assert dataset._build_spatial_level() == SpatialLevel.CM
        assert dataset._time_id == "month_id"
        assert dataset._entity_id == "country_id"
        assert dataset.sample_size == 1
        assert dataset.is_prediction is False

    def test_repr(self, dataset: ViewsDataset) -> None:
        """``__repr__`` summarises the dataset geometry."""
        text = repr(dataset)
        assert "ViewsDataset" in text
        assert f"time_steps={N_TIME_STEPS}" in text
        assert f"entities={N_ENTITIES}" in text
        assert "prediction_mode=False" in text

    def test_num_features(self, dataset: ViewsDataset) -> None:
        """``num_features`` reports the auto-derived feature count."""
        assert dataset.num_features == len(FEATURES)


class TestViewsDatasetSubset:
    """``get_subset_tensor`` / ``get_subset_dataset`` slicing tests."""

    def test_get_subset_tensor_by_entity(
        self, dataset: ViewsDataset
    ) -> None:
        """Subsetting by entity_ids yields the matching tensor shape."""
        tensor = dataset.get_subset_tensor(entity_ids=[1, 2, 3])
        # Shape: (T=100, E=3, S=1, V=9).
        assert tensor.shape == (N_TIME_STEPS, 3, 1, len(FEATURES) + len(TARGETS))
        # Entity coordinate set must match.
        entity_coord = set(int(v) for v in tensor[dataset._entity_id].values)
        assert entity_coord == {1, 2, 3}

    def test_get_subset_tensor_by_time(
        self, dataset: ViewsDataset
    ) -> None:
        """Subsetting by time_ids restricts to the requested window."""
        # Pick a contiguous 12-month window.
        tensor = dataset.get_subset_tensor(time_ids=list(range(121, 133)))
        time_arr = tensor[dataset._time_id].values
        assert int(time_arr.min()) == 121
        assert int(time_arr.max()) == 132
        assert tensor.shape[0] == 12  # 12 time steps

    def test_get_subset_tensor_by_entity_and_time(
        self, dataset: ViewsDataset
    ) -> None:
        """Combined entity + time filter restricts both axes."""
        tensor = dataset.get_subset_tensor(
            entity_ids=[1, 2], time_ids=[121, 122, 123]
        )
        assert tensor.shape == (3, 2, 1, len(FEATURES) + len(TARGETS))
        assert set(int(v) for v in tensor[dataset._entity_id].values) == {1, 2}
        assert set(int(v) for v in tensor[dataset._time_id].values) == {121, 122, 123}

    def test_get_subset_tensor_with_features_filter(
        self, dataset: ViewsDataset
    ) -> None:
        """The ``features`` argument restricts the variable axis."""
        tensor = dataset.get_subset_tensor(
            entity_ids=[1], features=[FEATURES[0], TARGETS[0]]
        )
        assert tensor.shape == (N_TIME_STEPS, 1, 1, 2)
        var_names = [str(v) for v in tensor["variable"].values]
        assert var_names == [FEATURES[0], TARGETS[0]]

    def test_get_subset_dataset_returns_new_dataset(
        self, dataset: ViewsDataset
    ) -> None:
        """``get_subset_dataset`` returns an independent :class:`ViewsDataset`."""
        sub = dataset.get_subset_dataset(entity_ids=[1, 2, 3])
        assert isinstance(sub, ViewsDataset)
        assert sub.num_entities == 3
        assert sub.num_time_steps == N_TIME_STEPS
        assert sub.targets == TARGETS
        assert sub.features == dataset.features
        # The subset's tensor round-trips.
        assert sub.check_integrity() is True


class TestViewsDatasetAsDarts:
    """``to_darts_timeseries`` end-to-end tests."""

    def test_to_darts_timeseries_basic(
        self, dataset: ViewsDataset
    ) -> None:
        """Three known multi-row entities yield 3 TimeSeries of equal length."""
        series_list = dataset.to_darts_timeseries(entity_ids=[1, 2, 3])
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

    def test_to_darts_timeseries_all_entities(
        self,
        dataset: ViewsDataset,
        synthetic_cm_parquet_small: Path,
    ) -> None:
        """All 200 entities — regression for single-row entity 248 bug."""
        series_list = dataset.to_darts_timeseries()
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

    def test_to_darts_timeseries_single_row_entity(self) -> None:
        """A single-row entity (1 row at month 140) must build a valid 1-step series.

        Build a tiny parquet inline with one entity (id 248) having a single
        row at month_id 140. This preserves the regression coverage for the
        Darts 1-step TimeSeries construction bug (``freq=1`` must be passed
        explicitly to bypass Darts' empty-diff step-inference failure on
        1-element time arrays).
        """
        import pyarrow as pa
        import tempfile

        feat = "lr_ged_sb_delta"
        target = "lr_ged_sb"
        time = np.array([140], dtype=np.int64)
        entity = np.array([248], dtype=np.int64)
        # Write a tiny parquet file.
        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path = Path(tmpdir) / "one_row.parquet"
            table = pa.table({
                "month_id": time,
                "country_id": entity,
                feat: np.array([0.5], dtype=np.float32),
                target: np.array([1.5], dtype=np.float32),
            })
            pq.write_table(table, str(pq_path))
            ds = ViewsDataset(pq_path, targets=[target], broadcast_features=True)
            series_list = ds.to_darts_timeseries(entity_ids=[248])
        assert len(series_list) == 1
        ts = series_list[0]
        assert len(ts) == 1
        assert int(ts.time_index.values[0]) == 140
        assert list(ts.components) == [feat, target]
        assert int(ts.static_covariates["country_id"].iloc[0]) == 248

    def test_to_darts_timeseries_with_cyclic_encoders(
        self, dataset: ViewsDataset
    ) -> None:
        """``use_cyclic_encoders=True`` appends month_sin/month_cos columns."""
        series_list = dataset.to_darts_timeseries(
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

    def test_to_darts_timeseries_empty_entity_subset_returns_empty_list(
        self, dataset: ViewsDataset
    ) -> None:
        """An empty ``entity_ids`` list returns an empty list."""
        series_list = dataset.to_darts_timeseries(entity_ids=[])
        assert series_list == []


class TestViewsDatasetFactory:
    """Parquet-path factory tests (replaces the old ``from_views_path``).

    The new API loads the parquet directly:
        ``ViewsDataset(Path(path_raw) / f"{run_type}_viewser_df.parquet",
                        targets=config["targets"], broadcast_features=True)``
    """

    def test_dataset_from_parquet_path(
        self,
        synthetic_cm_parquet_small: Path,
        tmp_path: Path,
    ) -> None:
        """The factory pattern loads via ``path_raw`` + ``run_type`` + config."""
        # The convention is ``<run_type>_viewser_df.parquet`` inside ``path_raw``.
        target_path = tmp_path / "validation_viewser_df.parquet"
        shutil.copy(synthetic_cm_parquet_small, target_path)

        config: dict = {
            "targets": TARGETS,
            # ``features`` is no longer a constructor argument — features are
            # auto-derived from the parquet schema. ``broadcast_features=True``
            # lifts scalar feature columns to a sample axis for the Darts path.
        }
        ds = ViewsDataset(
            tmp_path / f"{config.get('run_type', 'validation')}_viewser_df.parquet",
            targets=config["targets"],
            broadcast_features=True,
        )
        assert ds.num_entities == N_ENTITIES
        assert ds.num_time_steps == N_TIME_STEPS
        assert ds.targets == TARGETS
        assert ds.features == FEATURES
        assert ds._build_spatial_level() == SpatialLevel.CM

    def test_dataset_from_pgm_parquet_path(
        self,
        synthetic_pgm_parquet_small: Path,
    ) -> None:
        """Loading a pgm parquet infers the PGM spatial level."""
        ds = ViewsDataset(
            synthetic_pgm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        assert ds._entity_id == "priogrid_id"
        assert ds._build_spatial_level() == SpatialLevel.PGM


class TestViewsDatasetValidation:
    """Constructor-time validation errors."""

    def test_dataset_rejects_missing_target_column(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """A target column absent from the parquet must raise."""
        with pytest.raises(ValueError, match="Targets not found"):
            ViewsDataset(
                synthetic_cm_parquet_small,
                targets=["nonexistent_target"],
                broadcast_features=True,
            )

    def test_dataset_rejects_empty_targets(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """An empty targets list must raise at construction."""
        with pytest.raises(ValueError, match="targets must be specified"):
            ViewsDataset(
                synthetic_cm_parquet_small,
                targets=[],
                broadcast_features=True,
            )

    def test_dataset_rejects_missing_file(self, tmp_path: Path) -> None:
        """A nonexistent parquet path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ViewsDataset(
                tmp_path / "nonexistent.parquet",
                targets=TARGETS,
            )

    def test_dataset_rejects_unsupported_source_type(self) -> None:
        """An int source raises ``TypeError``."""
        with pytest.raises(TypeError, match="Unsupported source type"):
            ViewsDataset(12345, targets=TARGETS)


# ----------------------------------------------------------------------
# Empty creation + incremental concatenation
# ----------------------------------------------------------------------


class TestViewsDatasetCreateEmpty:
    """Tests for :meth:`ViewsDataset.create_empty` (static factory)."""

    def test_create_empty_cm(self) -> None:
        """``create_empty('cm', ...)`` builds a CMDataset with zero rows."""
        ds = ViewsDataset.create_empty(
            "cm", features=["feat1"], targets=["target1"], sample_size=1
        )
        assert ds.num_entities == 0
        assert ds.num_time_steps == 0
        assert ds.features == ["feat1"]
        assert ds.targets == ["target1"]
        assert ds._entity_id == "country_id"
        assert ds._time_id == "month_id"
        assert ds.sample_size == 1
        assert ds.is_prediction is False

    def test_create_empty_pgm_routes_to_subclass(self) -> None:
        """``create_empty('pgm', ...)`` returns a :class:`PGMDataset`."""
        from views_r2darts2.dataset.subclasses import PGMDataset

        ds = ViewsDataset.create_empty(
            "pgm", features=["feat1"], targets=["target1"]
        )
        assert isinstance(ds, PGMDataset)
        assert ds._entity_id == "priogrid_id"
        assert ds._time_id == "month_id"

    def test_create_empty_probabilistic_sample_size(self) -> None:
        """``sample_size=N`` produces an empty dataset with N samples per cell."""
        ds = ViewsDataset.create_empty(
            "cm", features=["feat1"], targets=["target1"], sample_size=5
        )
        assert ds.sample_size == 5


class TestViewsDatasetAddRow:
    """Tests for :meth:`ViewsDataset.add_row`."""

    def test_add_row_single(self) -> None:
        """``add_row`` appends a single (time, entity) cell."""
        ds = ViewsDataset.create_empty(
            "cm", features=["feat1"], targets=["target1"], sample_size=1
        )
        ds.add_row(time=100, entity=1, values={"feat1": 0.5, "target1": 1.5})
        assert ds.num_time_steps == 1
        assert ds.num_entities == 1
        # Verify the value round-trips through the tensor.
        tensor = ds.to_tensor().compute()
        # Shape: (T=1, E=1, S=1, V=2) where V = [feat1, target1].
        assert tensor.shape == (1, 1, 1, 2)
        # Variables are ordered [features..., targets...].
        var_names = [str(v) for v in tensor["variable"].values]
        assert var_names == ["feat1", "target1"]
        # The written values match.
        feat_idx = var_names.index("feat1")
        target_idx = var_names.index("target1")
        assert float(tensor.values[0, 0, 0, feat_idx]) == pytest.approx(0.5)
        assert float(tensor.values[0, 0, 0, target_idx]) == pytest.approx(1.5)

    def test_add_row_extends_coordinates(self) -> None:
        """Adding rows for new time/entity ids extends the store."""
        ds = ViewsDataset.create_empty(
            "cm", features=["feat1"], targets=["target1"]
        )
        ds.add_row(time=100, entity=1, values={"feat1": 0.1, "target1": 1.0})
        ds.add_row(time=101, entity=2, values={"feat1": 0.2, "target1": 2.0})
        assert ds.num_time_steps == 2
        assert ds.num_entities == 2


class TestViewsDatasetAddBatch:
    """Tests for :meth:`ViewsDataset.add_batch`."""

    def test_add_batch_multiple_rows(self) -> None:
        """``add_batch`` appends multiple rows in one call."""
        ds = ViewsDataset.create_empty(
            "cm", features=["feat1"], targets=["target1"]
        )
        times = np.array([100, 100, 101, 101], dtype=np.int64)
        entities = np.array([1, 2, 1, 2], dtype=np.int64)
        values = {
            "feat1": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "target1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        }
        ds.add_batch(times=times, entities=entities, values=values)
        assert ds.num_time_steps == 2
        assert ds.num_entities == 2
        # Verify the tensor values round-trip.
        tensor = ds.to_tensor().compute()
        var_names = [str(v) for v in tensor["variable"].values]
        feat_idx = var_names.index("feat1")
        # Entity 1, time 100 → feat1 = 0.1.
        assert float(tensor.values[0, 0, 0, feat_idx]) == pytest.approx(0.1)
        # Entity 2, time 101 → feat1 = 0.4.
        assert float(tensor.values[1, 1, 0, feat_idx]) == pytest.approx(0.4)

    def test_add_batch_probabilistic_values(self) -> None:
        """Probabilistic columns (shape ``(N, S)``) are written per-sample."""
        ds = ViewsDataset.create_empty(
            "cm", features=["feat1"], targets=["target1"], sample_size=3
        )
        times = np.array([100], dtype=np.int64)
        entities = np.array([1], dtype=np.int64)
        values = {
            "feat1": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            "target1": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        }
        ds.add_batch(times=times, entities=entities, values=values)
        assert ds.sample_size == 3
        tensor = ds.to_tensor().compute()
        # Shape: (T=1, E=1, S=3, V=2).
        assert tensor.shape == (1, 1, 3, 2)


# ----------------------------------------------------------------------
# Scaler integration
# ----------------------------------------------------------------------


class TestViewsDatasetScalers:
    """Tests for ``fit_scalers`` / ``get_scaled_darts_timeseries`` /
    ``ingest_darts_predictions``."""

    def test_scalers_fitted_default_false(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """``scalers_fitted`` is False on a fresh dataset."""
        # Use a fresh dataset (the module-scoped ``dataset`` fixture may
        # already have scalers fitted from another test in this class).
        fresh = ViewsDataset(
            synthetic_cm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        assert fresh.scalers_fitted is False

    def test_fit_scalers_sets_flag(
        self, dataset: ViewsDataset
    ) -> None:
        """``fit_scalers`` flips ``scalers_fitted`` to True."""
        dataset.fit_scalers(
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            time_ids=list(range(MONTH_ID_MIN, MONTH_ID_MAX + 1)),
        )
        assert dataset.scalers_fitted is True

    def test_fit_scalers_none_scalers(
        self, dataset: ViewsDataset
    ) -> None:
        """``fit_scalers(target_scaler=None, feature_scaler=None)`` still
        marks scalers as fitted (no-op fit)."""
        dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(MONTH_ID_MIN, MONTH_ID_MAX + 1)),
        )
        assert dataset.scalers_fitted is True

    def test_get_scaled_darts_timeseries_before_fit_raises(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """``get_scaled_darts_timeseries`` before ``fit_scalers`` raises."""
        # Use a fresh dataset (the module-scoped ``dataset`` fixture may
        # already have scalers fitted from an earlier test in this class).
        fresh = ViewsDataset(
            synthetic_cm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        with pytest.raises(RuntimeError, match="Scalers not fitted"):
            fresh.get_scaled_darts_timeseries()

    def test_get_scaled_darts_timeseries_after_fit(
        self, dataset: ViewsDataset
    ) -> None:
        """After fit, returns scaled ``(targets, past_covariates)`` tuple."""
        dataset.fit_scalers(
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            time_ids=list(range(MONTH_ID_MIN, MONTH_ID_MAX + 1)),
        )
        targets, past_cov = dataset.get_scaled_darts_timeseries(
            entity_ids=[1, 2, 3]
        )
        assert len(targets) == 3
        assert past_cov is not None
        assert len(past_cov) == 3
        for ts in targets:
            # MinMaxScaler maps to [0, 1].
            arr = ts.all_values(copy=False)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            assert float(arr.min()) >= -1e-5
            assert float(arr.max()) <= 1.0 + 1e-5

    def test_ingest_darts_predictions_returns_frames(
        self, dataset: ViewsDataset
    ) -> None:
        """``ingest_darts_predictions`` returns a ``{target: PredictionFrame}`` dict."""
        # Fit scalers (target=None so inverse is a no-op).
        dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(MONTH_ID_MIN, MONTH_ID_MAX + 1)),
        )
        # Build synthetic Darts prediction TimeSeries (one per entity).
        n_steps = 6
        preds = []
        for eid in [1, 2, 3]:
            time = np.arange(
                MONTH_ID_MAX + 1, MONTH_ID_MAX + 1 + n_steps, dtype=np.int64
            )
            values = np.full(
                (n_steps, len(TARGETS)), 0.5, dtype=np.float32
            )
            ts = TimeSeries.from_times_and_values(
                times=pd.Index(time),
                values=values,
                columns=TARGETS,
                static_covariates=pd.DataFrame({"country_id": [float(eid)]}),
                freq=1,
            )
            preds.append(ts)
        frames = dataset.ingest_darts_predictions(
            preds, apply_inverse=True, clip_negatives=True
        )
        assert isinstance(frames, dict)
        assert set(frames.keys()) == set(TARGETS)
        for tgt, frame in frames.items():
            assert isinstance(frame, PredictionFrame)
            # 3 entities × 6 time steps = 18 rows.
            assert frame.n_rows == 18

    def test_ingest_darts_predictions_clips_negatives(
        self, dataset: ViewsDataset
    ) -> None:
        """``clip_negatives=True`` clips negative predictions to 0."""
        dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(MONTH_ID_MIN, MONTH_ID_MAX + 1)),
        )
        n_steps = 6
        preds = []
        for eid in [1, 2, 3]:
            time = np.arange(
                MONTH_ID_MAX + 1, MONTH_ID_MAX + 1 + n_steps, dtype=np.int64
            )
            values = np.full(
                (n_steps, len(TARGETS)), -1.0, dtype=np.float32
            )
            ts = TimeSeries.from_times_and_values(
                times=pd.Index(time),
                values=values,
                columns=TARGETS,
                static_covariates=pd.DataFrame({"country_id": [float(eid)]}),
                freq=1,
            )
            preds.append(ts)
        frames = dataset.ingest_darts_predictions(preds, clip_negatives=True)
        for tgt, frame in frames.items():
            assert np.all(frame.values >= 0.0), (
                f"target '{tgt}' has negative values after clipping"
            )


# ----------------------------------------------------------------------
# Frame conversions
# ----------------------------------------------------------------------


class TestViewsDatasetFrames:
    """Tests for ``to_featureframe`` / ``to_predictionframe``."""

    def test_to_featureframe(self, dataset: ViewsDataset) -> None:
        """``to_featureframe`` returns a :class:`FeatureFrame` with the right shape."""
        ff = dataset.to_featureframe()
        assert isinstance(ff, FeatureFrame)
        # (N=20k rows, F=9 features, S=1 sample).
        assert ff.values.shape == (N_ROWS, len(FEATURES) + len(TARGETS), 1)
        assert ff.values.dtype == np.float32
        assert ff.feature_names == [*FEATURES, *TARGETS]

    def test_to_predictionframe_raises_on_feature_mode(
        self, dataset: ViewsDataset
    ) -> None:
        """``to_predictionframe`` on a feature-mode dataset raises ValueError."""
        with pytest.raises(ValueError, match="prediction mode"):
            dataset.to_predictionframe()

    def test_to_predictionframe_on_prediction_dataset(self) -> None:
        """A prediction-mode dataset can convert to a :class:`PredictionFrame`."""
        # Build a small prediction-mode dataset via ``create_empty`` +
        # ``add_row`` with a ``pred_`` column.
        ds = ViewsDataset.create_empty(
            "cm", features=[], targets=["sb"], sample_size=1
        )
        # Manually mark it as a prediction dataset by adding a pred_ column.
        # Easiest path: ingest a PredictionFrame.
        time = np.array([100, 101], dtype=np.int64)
        entity = np.array([1, 1], dtype=np.int64)
        index = SpatioTemporalIndex(time=time, unit=entity, level=SpatialLevel.CM)
        pf = PredictionFrame(
            np.array([[0.5], [1.5]], dtype=np.float32), index=index
        )
        pred_ds = ViewsDataset(pf, targets=["sb"])
        assert pred_ds.is_prediction is True
        out = pred_ds.to_predictionframe()
        assert isinstance(out, PredictionFrame)
        assert out.n_rows == 2


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


class TestViewsDatasetPersistence:
    """Tests for ``save_parquet`` / ``save_zarr`` / ``save_zarrzip``."""

    def test_save_parquet_round_trip(
        self, dataset: ViewsDataset, tmp_path: Path
    ) -> None:
        """``save_parquet`` writes a parquet that reloads as a valid dataset."""
        out = tmp_path / "out.parquet"
        dataset.save_parquet(out)
        assert out.exists()
        # Reload and verify geometry (bit-parity is exercised by the
        # dedicated parquet-loader parity tests).
        reloaded = ViewsDataset(out, targets=TARGETS, broadcast_features=True)
        assert reloaded.num_entities == N_ENTITIES
        assert reloaded.num_time_steps == N_TIME_STEPS
        assert reloaded.targets == TARGETS
        assert set(reloaded.features) == set(FEATURES)

    def test_save_zarr(self, dataset: ViewsDataset, tmp_path: Path) -> None:
        """``save_zarr`` writes a consolidated zarr directory."""
        out = tmp_path / "out.zarr"
        dataset.save_zarr(out)
        assert out.exists()
        assert (out / ".zgroup").exists() or (out / "zarr.json").exists()

    def test_save_zarrzip(self, dataset: ViewsDataset, tmp_path: Path) -> None:
        """``save_zarrzip`` writes a zip archive."""
        out = tmp_path / "out.zip"
        dataset.save_zarrzip(out)
        assert out.exists()
        import zipfile
        assert zipfile.is_zipfile(out)

    def test_save_npz_via_featureframe(
        self, dataset: ViewsDataset, tmp_path: Path
    ) -> None:
        """``save_npz`` writes the views-frames leaf format."""
        out = tmp_path / "out.npz"
        # ``save_npz`` calls ``to_featureframe`` for feature-mode datasets.
        dataset.save_npz(out)
        # The frame's ``save`` method writes ``<path>.npz`` plus identifiers.
        assert out.exists() or out.with_suffix(".npz").exists()


# ----------------------------------------------------------------------
# for_loa routing
# ----------------------------------------------------------------------


class TestViewsDatasetForLoa:
    """Tests for :meth:`ViewsDataset.for_loa`."""

    def test_for_loa_cm_routes_to_cmdataset(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """``for_loa('cm', source)`` returns a :class:`CMDataset`."""
        from views_r2darts2.dataset.subclasses import CMDataset

        ds = ViewsDataset.for_loa(
            "cm",
            synthetic_cm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        assert isinstance(ds, CMDataset)
        assert ds._entity_id == "country_id"

    def test_for_loa_pgm_routes_to_pgmdataset(
        self, synthetic_pgm_parquet_small: Path
    ) -> None:
        """``for_loa('pgm', source)`` returns a :class:`PGMDataset`."""
        from views_r2darts2.dataset.subclasses import PGMDataset

        ds = ViewsDataset.for_loa(
            "pgm",
            synthetic_pgm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        assert isinstance(ds, PGMDataset)
        assert ds._entity_id == "priogrid_id"


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
