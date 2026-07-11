"""Tests for the pandas-free parquet loader (data boundary).

Verifies that :func:`load_views_parquet` produces bit-identical float32 values
to a direct ``pyarrow.parquet.read_table`` call, that the memmap cache works,
and that the loader enforces the VIEWS viewser schema contract.

Uses a session-scoped synthetic parquet fixture (see ``conftest.py``) so the
tests run anywhere without the real validation parquet.


"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from views_frames import SpatialLevel
from views_r2darts2.data.parquet_loader import (
    ParquetLoadError,
    load_views_parquet,
)

TARGETS = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
FEATURES = [
    "lr_ged_sb_delta",
    "lr_ged_ns_delta",
    "lr_ged_os_delta",
    "lr_splag_1_ged_sb",
    "lr_splag_1_ged_ns",
    "lr_splag_1_ged_os",
]


class TestParquetLoaderBitParity:
    """Bit-for-bit parity vs a direct pyarrow column read."""

    def test_bit_parity_all_columns(self, synthetic_cm_parquet_small: Path) -> None:
        """Every feature and target column must match pyarrow bit-for-bit."""
        frame, feats, targs = load_views_parquet(
            synthetic_cm_parquet_small, targets=TARGETS, features=FEATURES
        )
        assert feats == FEATURES
        assert targs == TARGETS

        for col in TARGETS + FEATURES:
            direct = (
                pq.read_table(synthetic_cm_parquet_small, columns=[col])
                .column(col)
                .to_numpy()
                .astype(np.float32)
            )
            idx = frame.feature_names.index(col)
            loaded = frame.values[:, idx, 0]
            assert direct.shape == loaded.shape
            assert np.array_equal(direct, loaded), f"Bit parity failed for {col}"

    def test_frame_shape_and_dtype(self, synthetic_cm_parquet_small: Path) -> None:
        """Frame must be (N, F, 1) float32 with F = n_features + n_targets."""
        frame, _, _ = load_views_parquet(
            synthetic_cm_parquet_small, targets=TARGETS, features=FEATURES
        )
        assert frame.values.dtype == np.float32
        n_rows_expected = 200 * 100  # 200 countries × 100 months
        assert frame.values.shape == (n_rows_expected, len(FEATURES) + len(TARGETS), 1)
        assert frame.n_rows == n_rows_expected
        assert frame.n_features == len(FEATURES) + len(TARGETS)
        assert frame.sample_count == 1

    def test_index_arrays(self, synthetic_cm_parquet_small: Path) -> None:
        """The (time, unit) arrays must match the parquet month_id/country_id."""
        frame, _, _ = load_views_parquet(
            synthetic_cm_parquet_small, targets=TARGETS, features=FEATURES
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
        assert np.array_equal(frame.index.time, time_direct)
        assert np.array_equal(frame.index.unit, entity_direct)
        assert frame.index.level == SpatialLevel.CM

    def test_value_axis_order(self, synthetic_cm_parquet_small: Path) -> None:
        """Value axis must be ``[features..., targets...]``."""
        frame, _, _ = load_views_parquet(
            synthetic_cm_parquet_small, targets=TARGETS, features=FEATURES
        )
        assert frame.feature_names == [*FEATURES, *TARGETS]


class TestParquetLoaderMemmapCache:
    """Memmap-backed cache must produce bit-identical values to in-memory load."""

    def test_cache_round_trip(self, synthetic_cm_parquet_small: Path) -> None:
        """First read decodes parquet; second read memmaps the cache."""
        with tempfile.TemporaryDirectory() as cache_dir:
            frame1, _, _ = load_views_parquet(
                synthetic_cm_parquet_small,
                targets=TARGETS,
                features=FEATURES,
                cache_dir=cache_dir,
            )
            assert not isinstance(
                frame1.values, np.memmap
            ), "First read should not be memmap"

            frame2, _, _ = load_views_parquet(
                synthetic_cm_parquet_small,
                targets=TARGETS,
                features=FEATURES,
                cache_dir=cache_dir,
            )
            assert isinstance(
                frame2.values, np.memmap
            ), "Second read should be memmap"

            assert np.array_equal(frame1.values, frame2.values)
            assert np.array_equal(frame1.index.time, frame2.index.time)
            assert np.array_equal(frame1.index.unit, frame2.index.unit)

    def test_cache_invalidation_on_manifest_change(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """Changing the features list must invalidate the cache."""
        with tempfile.TemporaryDirectory() as cache_dir:
            load_views_parquet(
                synthetic_cm_parquet_small,
                targets=TARGETS,
                features=FEATURES,
                cache_dir=cache_dir,
            )
            new_features = FEATURES[:2]
            frame, _, _ = load_views_parquet(
                synthetic_cm_parquet_small,
                targets=TARGETS,
                features=new_features,
                cache_dir=cache_dir,
            )
            assert frame.n_features == len(new_features) + len(TARGETS)


class TestParquetLoaderErrors:
    """Error paths."""

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ParquetLoadError, match="not found"):
            load_views_parquet(tmp_path / "nonexistent.parquet", targets=TARGETS)

    def test_missing_columns(self, tmp_path: Path) -> None:
        """Schema mismatch must raise ParquetLoadError."""
        table = pa.table({"lr_ged_sb": [1.0, 2.0]})
        pq.write_table(table, tmp_path / "tiny.parquet")
        with pytest.raises(ParquetLoadError, match="missing required columns"):
            load_views_parquet(
                tmp_path / "tiny.parquet",
                targets=TARGETS,
                features=FEATURES,
            )

    def test_target_feature_overlap(self, synthetic_cm_parquet_small: Path) -> None:
        """A column cannot be both target and feature."""
        with pytest.raises(ParquetLoadError, match="both target and feature"):
            load_views_parquet(
                synthetic_cm_parquet_small,
                targets=["lr_ged_sb"],
                features=["lr_ged_sb"],
            )

    def test_empty_targets(self, synthetic_cm_parquet_small: Path) -> None:
        with pytest.raises(ParquetLoadError, match="non-empty"):
            load_views_parquet(synthetic_cm_parquet_small, targets=[])


class TestParquetLoaderPgmLevel:
    """SpatialLevel inference from the entity_id column name."""

    def test_pgm_level_inference(self, tmp_path: Path) -> None:
        """Passing ``entity_id='priogrid_id'`` must produce ``SpatialLevel.PGM``."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "priogrid_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "pgm.parquet")
        frame, _, _ = load_views_parquet(
            tmp_path / "pgm.parquet",
            targets=["lr_ged_sb"],
            features=None,
            entity_id="priogrid_id",
        )
        assert frame.index.level == SpatialLevel.PGM

    def test_large_pgm_parquet(self, synthetic_pgm_parquet: Path) -> None:
        """The loader must handle the 1M-row PGM parquet without OOM.

        This is the memmap stress test: the file is ~1M rows × 12 columns.
        The full 25.9M-row file (259k cells × 100 months) would take too long
        to generate in a test session; this 10k-cell subset exercises the same
        code paths (memmap cache, multi-entity TimeSeries, PGM level inference)
        in ~5 seconds.
        """
        frame, feats, targs = load_views_parquet(
            synthetic_pgm_parquet,
            targets=TARGETS,
            features=FEATURES[:3],  # subset for speed
            entity_id="priogrid_id",
        )
        assert frame.n_rows == 10_000 * 100
        assert frame.values.dtype == np.float32
        assert frame.index.level == SpatialLevel.PGM
        # Verify a single column's bit parity.
        direct = (
            pq.read_table(synthetic_pgm_parquet, columns=["lr_ged_sb"])
            .column("lr_ged_sb")
            .to_numpy()
            .astype(np.float32)
        )
        idx = frame.feature_names.index("lr_ged_sb")
        assert np.array_equal(direct, frame.values[:, idx, 0])


class TestParquetLoaderEntityAutoDetection:
    """Auto-detection of the entity column when the declared one is absent.

    The loader should fall back from ``country_id`` → ``priogrid_id`` (or the
    ``priogrid_gid`` alias) when the parquet is pgm-level, and vice versa.
    """

    def test_country_id_fallback_to_priogrid_id(self, tmp_path: Path) -> None:
        """Declaring ``country_id`` on a pgm parquet falls back to ``priogrid_id``."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "priogrid_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "pgm.parquet")
        # Declare country_id (default) — the loader should auto-detect pgm.
        frame, _, _ = load_views_parquet(
            tmp_path / "pgm.parquet",
            targets=["lr_ged_sb"],
            features=None,
            entity_id="country_id",  # absent in schema
        )
        assert frame.index.level == SpatialLevel.PGM
        assert np.array_equal(frame.index.unit, np.array([1, 2], dtype=np.int64))

    def test_priogrid_gid_alias(self, tmp_path: Path) -> None:
        """``priogrid_gid`` (typo) is normalized to ``priogrid_id``."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "priogrid_gid": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "pgm_gid.parquet")
        # Declare country_id — neither country_id nor priogrid_id is present,
        # but the priogrid_gid alias is. The loader should normalize it.
        frame, _, _ = load_views_parquet(
            tmp_path / "pgm_gid.parquet",
            targets=["lr_ged_sb"],
            features=None,
            entity_id="country_id",
        )
        assert frame.index.level == SpatialLevel.PGM
        assert np.array_equal(frame.index.unit, np.array([1, 2], dtype=np.int64))

    def test_priogrid_id_fallback_to_country_id(self, tmp_path: Path) -> None:
        """Declaring ``priogrid_id`` on a cm parquet falls back to ``country_id``."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "country_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "cm.parquet")
        frame, _, _ = load_views_parquet(
            tmp_path / "cm.parquet",
            targets=["lr_ged_sb"],
            features=None,
            entity_id="priogrid_id",  # absent in schema
        )
        assert frame.index.level == SpatialLevel.CM
        assert np.array_equal(frame.index.unit, np.array([1, 2], dtype=np.int64))

    def test_declared_entity_present_no_fallback(self, tmp_path: Path) -> None:
        """When the declared entity column IS present, no fallback occurs."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "country_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "cm.parquet")
        frame, _, _ = load_views_parquet(
            tmp_path / "cm.parquet",
            targets=["lr_ged_sb"],
            features=None,
            entity_id="country_id",
        )
        assert frame.index.level == SpatialLevel.CM

    def test_no_entity_column_found_raises(self, tmp_path: Path) -> None:
        """When no entity column is present, the original error is raised."""
        table = pa.table(
            {
                "month_id": np.array([100, 100], dtype=np.int64),
                "some_other_id": np.array([1, 2], dtype=np.int64),
                "lr_ged_sb": np.array([0.0, 1.0], dtype=np.float32),
            }
        )
        pq.write_table(table, tmp_path / "bad.parquet")
        with pytest.raises(ParquetLoadError, match="missing required columns"):
            load_views_parquet(
                tmp_path / "bad.parquet",
                targets=["lr_ged_sb"],
                features=None,
                entity_id="country_id",
            )
