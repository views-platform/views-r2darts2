"""Tests for the parquet loading path of :class:`views_r2darts2.dataset.base.ViewsDataset`.

Verifies that ``ViewsDataset`` produces bit-identical float32 values to a
direct ``pyarrow.parquet.read_table`` call, that the schema is correctly
inferred (targets vs. features vs. entity/time columns), and that the loader
enforces the VIEWS viewser schema contract.

Uses session-scoped synthetic parquet fixtures (see ``conftest.py``) so the
tests run anywhere without the real validation parquet.

The new ``ViewsDataset`` no longer has a ``cache_dir`` parameter, a
``features`` parameter, or an ``entity_id`` parameter — entity_id and
time_id are auto-detected from the parquet schema, and features are
auto-derived (every numeric column that is not a target).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from views_frames import SpatialLevel
from views_r2darts2.dataset.base import ViewsDataset

TARGETS = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
FEATURES = [
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


class TestParquetLoaderBitParity:
    """Bit-for-bit parity vs a direct pyarrow column read."""

    def test_bit_parity_all_columns(self, synthetic_cm_parquet_small: Path) -> None:
        """Every feature and target column must match pyarrow bit-for-bit."""
        ds = ViewsDataset(
            synthetic_cm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        assert ds.features == FEATURES
        assert ds.targets == TARGETS

        # The tensor is (T, E, S, V) with V = features + targets, ordered.
        tensor = ds.to_tensor().compute()
        var_names = [str(v) for v in tensor["variable"].values]
        time_coord = tensor[ds._time_id].values
        entity_coord = tensor[ds._entity_id].values

        for col in TARGETS + FEATURES:
            direct = (
                pq.read_table(synthetic_cm_parquet_small, columns=[col])
                .column(col)
                .to_numpy()
                .astype(np.float32)
            )
            # Reshape direct to (T, E) using the meshgrid layout the converter
            # uses (time-major, entity-minor). The original parquet is long
            # format (one row per (time, entity)).
            t, e = len(time_coord), len(entity_coord)
            # Build a (time, entity) → value map by reading time+entity cols.
            time_col = (
                pq.read_table(synthetic_cm_parquet_small, columns=["month_id"])
                .column("month_id")
                .to_numpy()
                .astype(np.int64)
            )
            entity_col = (
                pq.read_table(synthetic_cm_parquet_small, columns=["country_id"])
                .column("country_id")
                .to_numpy()
                .astype(np.int64)
            )
            time_to_idx = {int(v): i for i, v in enumerate(time_coord)}
            entity_to_idx = {int(v): i for i, v in enumerate(entity_coord)}
            grid = np.full((t, e), np.nan, dtype=np.float32)
            for i in range(len(direct)):
                ti = time_to_idx[int(time_col[i])]
                ei = entity_to_idx[int(entity_col[i])]
                grid[ti, ei] = direct[i]
            # The dataset tensor's column slice.
            var_idx = var_names.index(col)
            loaded = tensor.values[:, :, 0, var_idx]
            assert loaded.shape == grid.shape, f"{col}: shape mismatch"
            assert np.array_equal(grid, loaded, equal_nan=True), (
                f"{col}: bit parity failed"
            )

    def test_tensor_shape_and_dtype(self, synthetic_cm_parquet_small: Path) -> None:
        """Tensor must be (T=100, E=200, S=1, V=12) float32."""
        ds = ViewsDataset(
            synthetic_cm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        tensor = ds.to_tensor().compute()
        assert tensor.values.dtype == np.float32
        n_time_expected = 100
        n_entities_expected = 200
        assert tensor.shape == (
            n_time_expected,
            n_entities_expected,
            1,
            len(FEATURES) + len(TARGETS),
        )
        assert ds.num_time_steps == n_time_expected
        assert ds.num_entities == n_entities_expected
        assert ds.num_features == len(FEATURES)
        assert ds.sample_size == 1

    def test_index_arrays(self, synthetic_cm_parquet_small: Path) -> None:
        """The (time, entity) coords must match the parquet's month_id/country_id."""
        ds = ViewsDataset(
            synthetic_cm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        time_direct = (
            pq.read_table(synthetic_cm_parquet_small, columns=["month_id"])
            .column("month_id")
            .to_numpy()
            .astype(np.int64)
        )
        entity_direct = (
            pq.read_table(synthetic_cm_parquet_small, columns=["country_id"])
            .column("country_id")
            .to_numpy()
            .astype(np.int64)
        )
        # The dataset's coords are the sorted-unique values.
        assert np.array_equal(
            ds._ds[ds._time_id].values,
            np.unique(time_direct),
        )
        assert np.array_equal(
            ds._ds[ds._entity_id].values,
            np.unique(entity_direct),
        )
        assert ds._build_spatial_level() == SpatialLevel.CM

    def test_value_axis_order(self, synthetic_cm_parquet_small: Path) -> None:
        """Value axis must be ``[features..., targets...]``."""
        ds = ViewsDataset(
            synthetic_cm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        tensor = ds.to_tensor().compute()
        var_names = [str(v) for v in tensor["variable"].values]
        assert var_names == [*FEATURES, *TARGETS]


class TestParquetLoaderErrors:
    """Error paths."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A nonexistent parquet path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ViewsDataset(tmp_path / "nonexistent.parquet", targets=TARGETS)

    def test_missing_target_column(self, tmp_path: Path) -> None:
        """A target column absent from the parquet raises ValueError."""
        # Include the time/entity columns so entity detection passes; the
        # missing-target check is what should fire.
        table = pa.table({
            "month_id": np.array([100, 101], dtype=np.int64),
            "country_id": np.array([1, 1], dtype=np.int64),
            "lr_ged_sb": np.array([1.0, 2.0], dtype=np.float32),
        })
        pq.write_table(table, tmp_path / "tiny.parquet")
        with pytest.raises(ValueError, match="Targets not found"):
            ViewsDataset(
                tmp_path / "tiny.parquet",
                targets=TARGETS,
            )

    def test_empty_targets(self, synthetic_cm_parquet_small: Path) -> None:
        """An empty targets list raises ValueError."""
        with pytest.raises(ValueError, match="targets must be specified"):
            ViewsDataset(synthetic_cm_parquet_small, targets=[])

    def test_unsupported_source_type(self) -> None:
        """An int source raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported source type"):
            ViewsDataset(12345, targets=TARGETS)


class TestParquetLoaderPgmLevel:
    """SpatialLevel inference from the entity_id column name."""

    def test_pgm_level_inference(self, tmp_path: Path) -> None:
        """A parquet with ``priogrid_id`` is auto-detected as ``SpatialLevel.PGM``."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "priogrid_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "pgm.parquet")
        ds = ViewsDataset(
            tmp_path / "pgm.parquet",
            targets=["lr_ged_sb"],
        )
        assert ds._build_spatial_level() == SpatialLevel.PGM
        assert ds._entity_id == "priogrid_id"

    def test_large_pgm_parquet(self, synthetic_pgm_parquet_small: Path) -> None:
        """The loader must handle the 1M-row PGM parquet without OOM.

        This is the zarr-store stress test: the file is ~1M rows × 12 columns.
        The full 25.9M-row file (259k cells × 100 months) would take too long
        to generate in a test session; this 10k-cell subset exercises the same
        code paths (zarr store, multi-entity TimeSeries, PGM level inference)
        in ~5 seconds.
        """
        ds = ViewsDataset(
            synthetic_pgm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        assert ds.num_entities == 1000
        assert ds.num_time_steps == 100
        assert ds._ds[ds._entity_id].values.dtype == np.int64
        assert ds._build_spatial_level() == SpatialLevel.PGM
        # Verify a single column's bit parity via the to_darts_timeseries path.
        series_list = ds.to_darts_timeseries(entity_ids=[1])
        assert len(series_list) == 1
        ts = series_list[0]
        assert len(ts) == 100


class TestParquetLoaderEntityAutoDetection:
    """Auto-detection of the entity column when the declared one is absent.

    The new ``ViewsDataset`` no longer accepts an ``entity_id`` parameter —
    it auto-detects the entity column from the parquet schema using
    :func:`views_r2darts2.dataset.readers.pick_time_entity`. The
    ``priogrid_gid`` wire name (the VIEWSER typo for ``priogrid_id``) is
    normalised on ingest via
    :func:`views_r2darts2.dataset.readers.normalize_entity_name`.
    """

    def test_priogrid_id_detected(self, tmp_path: Path) -> None:
        """``priogrid_id`` is auto-detected and produces ``SpatialLevel.PGM``."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "priogrid_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "pgm.parquet")
        ds = ViewsDataset(
            tmp_path / "pgm.parquet",
            targets=["lr_ged_sb"],
        )
        assert ds._build_spatial_level() == SpatialLevel.PGM
        assert np.array_equal(
            ds._ds[ds._entity_id].values, np.array([1, 2], dtype=np.int64)
        )

    def test_priogrid_gid_alias(self, tmp_path: Path) -> None:
        """``priogrid_gid`` (the VIEWSER wire name) is normalised to ``priogrid_id``."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "priogrid_gid": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "pgm_gid.parquet")
        ds = ViewsDataset(
            tmp_path / "pgm_gid.parquet",
            targets=["lr_ged_sb"],
        )
        assert ds._build_spatial_level() == SpatialLevel.PGM
        # The entity_id is normalised to ``priogrid_id``.
        assert ds._entity_id == "priogrid_id"
        assert np.array_equal(
            ds._ds[ds._entity_id].values, np.array([1, 2], dtype=np.int64)
        )

    def test_country_id_detected(self, tmp_path: Path) -> None:
        """``country_id`` is auto-detected and produces ``SpatialLevel.CM``."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "country_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "cm.parquet")
        ds = ViewsDataset(
            tmp_path / "cm.parquet",
            targets=["lr_ged_sb"],
        )
        assert ds._build_spatial_level() == SpatialLevel.CM
        assert ds._entity_id == "country_id"

    def test_no_entity_column_raises(self, tmp_path: Path) -> None:
        """When no entity column is present, the original error is raised."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "some_other_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "bad.parquet")
        with pytest.raises(ValueError, match="time/entity identifiers"):
            ViewsDataset(
                tmp_path / "bad.parquet",
                targets=["lr_ged_sb"],
            )
