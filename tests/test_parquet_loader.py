"""Tests for the pandas-free parquet loader (data boundary).

Verifies that :func:`load_views_parquet` produces bit-identical float32 values
to a direct ``pyarrow.parquet.read_table`` call, that the memmap cache works,
and that the loader enforces the VIEWS viewser schema contract.

Google Python Style.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from views_frames import FeatureFrame, SpatialLevel
from views_r2darts2.data.parquet_loader import (
    ParquetLoadError,
    load_views_parquet,
)

# Path to the user-provided validation parquet (12 MB, 87 cols, 81192 rows).
PARQUET_PATH = Path("/home/z/my-project/upload/validation_viewser_df.parquet")

TARGETS = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
FEATURES = [
    "lr_ged_sb_delta",
    "lr_ged_ns_delta",
    "lr_ged_os_delta",
    "lr_splag_1_ged_sb",
    "lr_splag_1_ged_ns",
    "lr_splag_1_ged_os",
]


# ----------------------------------------------------------------------
# Skip the suite if the validation parquet is not present (e.g., CI without
# the upload dir).
# ----------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason=f"Validation parquet not found at {PARQUET_PATH}",
)


class TestParquetLoaderBitParity:
    """Bit-for-bit parity vs a direct pyarrow column read."""

    def test_bit_parity_all_columns(self) -> None:
        """Every feature and target column must match pyarrow bit-for-bit."""
        frame, feats, targs = load_views_parquet(
            PARQUET_PATH, targets=TARGETS, features=FEATURES
        )
        assert feats == FEATURES
        assert targs == TARGETS

        for col in TARGETS + FEATURES:
            direct = (
                pq.read_table(PARQUET_PATH, columns=[col])
                .column(col)
                .to_numpy()
                .astype(np.float32)
            )
            idx = frame.feature_names.index(col)
            loaded = frame.values[:, idx, 0]
            assert direct.shape == loaded.shape
            assert np.array_equal(direct, loaded), f"Bit parity failed for {col}"

    def test_frame_shape_and_dtype(self) -> None:
        """Frame must be (N, F, 1) float32 with F = n_features + n_targets."""
        frame, _, _ = load_views_parquet(
            PARQUET_PATH, targets=TARGETS, features=FEATURES
        )
        assert frame.values.dtype == np.float32
        assert frame.values.shape == (81192, len(FEATURES) + len(TARGETS), 1)
        assert frame.n_rows == 81192
        assert frame.n_features == len(FEATURES) + len(TARGETS)
        assert frame.sample_count == 1

    def test_index_arrays(self) -> None:
        """The (time, unit) arrays must match the parquet month_id/country_id."""
        frame, _, _ = load_views_parquet(
            PARQUET_PATH, targets=TARGETS, features=FEATURES
        )
        time_direct = (
            pq.read_table(PARQUET_PATH, columns=["month_id"])
            .column("month_id")
            .to_numpy()
            .astype(np.int64)
        )
        entity_direct = (
            pq.read_table(PARQUET_PATH, columns=["country_id"])
            .column("country_id")
            .to_numpy()
            .astype(np.int64)
        )
        assert np.array_equal(frame.index.time, time_direct)
        assert np.array_equal(frame.index.unit, entity_direct)
        assert frame.index.level == SpatialLevel.CM

    def test_value_axis_order(self) -> None:
        """Value axis must be ``[features..., targets...]``."""
        frame, _, _ = load_views_parquet(
            PARQUET_PATH, targets=TARGETS, features=FEATURES
        )
        assert frame.feature_names == [*FEATURES, *TARGETS]


class TestParquetLoaderMemmapCache:
    """Memmap-backed cache must produce bit-identical values to in-memory load."""

    def test_cache_round_trip(self) -> None:
        """First read decodes parquet; second read memmaps the cache."""
        with tempfile.TemporaryDirectory() as cache_dir:
            frame1, _, _ = load_views_parquet(
                PARQUET_PATH, targets=TARGETS, features=FEATURES, cache_dir=cache_dir
            )
            assert not isinstance(
                frame1.values, np.memmap
            ), "First read should not be memmap"

            frame2, _, _ = load_views_parquet(
                PARQUET_PATH, targets=TARGETS, features=FEATURES, cache_dir=cache_dir
            )
            assert isinstance(
                frame2.values, np.memmap
            ), "Second read should be memmap"

            assert np.array_equal(frame1.values, frame2.values)
            assert np.array_equal(frame1.index.time, frame2.index.time)
            assert np.array_equal(frame1.index.unit, frame2.index.unit)

    def test_cache_invalidation_on_manifest_change(self) -> None:
        """Changing the features list must invalidate the cache."""
        with tempfile.TemporaryDirectory() as cache_dir:
            # First load with FEATURES.
            load_views_parquet(
                PARQUET_PATH, targets=TARGETS, features=FEATURES, cache_dir=cache_dir
            )
            # Second load with a different feature set.
            new_features = FEATURES[:2]  # subset
            frame, _, _ = load_views_parquet(
                PARQUET_PATH,
                targets=TARGETS,
                features=new_features,
                cache_dir=cache_dir,
            )
            # The cache key is content-addressed by the manifest, so the new
            # feature set produces a fresh cache miss and a frame with the
            # smaller column count.
            assert frame.n_features == len(new_features) + len(TARGETS)


class TestParquetLoaderErrors:
    """Error paths."""

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ParquetLoadError, match="not found"):
            load_views_parquet(tmp_path / "nonexistent.parquet", targets=TARGETS)

    def test_missing_columns(self, tmp_path: Path) -> None:
        """Schema mismatch must raise ParquetLoadError."""
        # Build a small parquet with only one column.
        import pyarrow as pa

        table = pa.table({"lr_ged_sb": [1.0, 2.0]})
        pq.write_table(table, tmp_path / "tiny.parquet")
        with pytest.raises(ParquetLoadError, match="missing required columns"):
            load_views_parquet(
                tmp_path / "tiny.parquet",
                targets=TARGETS,
                features=FEATURES,
            )

    def test_target_feature_overlap(self, tmp_path: Path) -> None:
        """A column cannot be both target and feature."""
        # Use the real parquet — the manifest check happens before schema read.
        with pytest.raises(ParquetLoadError, match="both target and feature"):
            load_views_parquet(
                PARQUET_PATH,
                targets=["lr_ged_sb"],
                features=["lr_ged_sb"],  # same column
            )

    def test_empty_targets(self, tmp_path: Path) -> None:
        with pytest.raises(ParquetLoadError, match="non-empty"):
            load_views_parquet(PARQUET_PATH, targets=[])


class TestParquetLoaderPgmLevel:
    """SpatialLevel inference from the entity_id column name."""

    def test_pgm_level_inference(self, tmp_path: Path) -> None:
        """Passing ``entity_id='priogrid_id'`` must produce ``SpatialLevel.PGM``."""
        import pyarrow as pa

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
