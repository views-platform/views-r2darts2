"""Shared test fixtures for the views-r2darts2 test suite.

Provides a synthetic VIEWS-format parquet file (pandas-free, written via
pyarrow) that mirrors the schema of the real ``validation_viewser_df.parquet``
but with dummy data. The file is generated once per test session and cached in
a temporary directory.

The synthetic data uses PRIO-GRID-month rows (``priogrid_id`` entity column)
with ~259,000 unique pgm cells × ~100 months ≈ 25M rows — large enough to
exercise the memmap cache path and the multi-entity TimeSeries construction,
small enough to fit in a CI sandbox.

 Pandas-free.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# ----------------------------------------------------------------------
# Schema constants — mirror the real validation_viewser_df.parquet contract.
# ----------------------------------------------------------------------

# A representative subset of the VIEWS feature vocabulary. The real parquet has
# 87 columns; we use 12 here (3 targets + 9 features) to keep the synthetic
# file small while exercising every code path (multi-target, multi-feature,
# the user's AsinhTransform->MaxAbsScaler chain, cyclic encoders, static
# covariates).
SYNTHETIC_TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]

SYNTHETIC_FEATURES: list[str] = [
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

# The user's example feature_scaler_map (subset for the synthetic file).
SYNTHETIC_FEATURE_SCALER_MAP: dict[str, list[str]] = {
    "AsinhTransform->MaxAbsScaler": SYNTHETIC_FEATURES,
}


# ----------------------------------------------------------------------
# Session-scoped synthetic parquet fixture (pgm level, ~259k cells).
# ----------------------------------------------------------------------

# Number of unique priogrid cells for the large PGM fixture. The real PRIO-GRID
# has ~259,200 cells; we use 10,000 cells (× 100 months = 1M rows) for the test
# fixture — large enough to exercise the memmap path and multi-entity
# TimeSeries construction, small enough to generate in ~5 seconds.
# The full 259k-cell × 100-month = 25.9M-row file would take ~2 minutes to
# generate and ~1.2 GB of RAM — too slow for a test session.
_N_PGM_CELLS_LARGE = 10_000
# Number of months per cell. The real validation set spans ~100 months.
_N_MONTHS_PER_CELL = 100
# Month_id range: start at 121 (VIEWS convention) and go to 220.
_MONTH_ID_START = 121


@pytest.fixture(scope="session")
def synthetic_pgm_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a synthetic PRIO-GRID-month parquet file (~25.9M rows).

    The file is written once per test session via ``pyarrow.parquet.write_table``
    (no pandas). The schema matches the VIEWS viewser contract:

        * 3 target columns (``lr_ged_sb``, ``lr_ged_ns``, ``lr_ged_os``)
        * 9 feature columns (conflict deltas, splags, decays)
        * ``month_id`` (int64, 121..220)
        * ``priogrid_id`` (int64, 1..259000)

    The data is dummy (log-normal noise + zeros for ~80% sparsity) but
    structurally valid: float32-castable, no NaN, non-negative for fatality
    counts.

    Returns:
        Path to the generated ``.parquet`` file.
    """
    cache_dir = tmp_path_factory.mktemp("synthetic_parquet")
    parquet_path = cache_dir / "validation_viewser_df.parquet"

    # If a previous session wrote it (session-scoped reuse), skip regeneration.
    if parquet_path.exists():
        return parquet_path

    n_rows = _N_PGM_CELLS_LARGE * _N_MONTHS_PER_CELL
    rng = np.random.default_rng(seed=42)

    # Build the index arrays: every (pgm_cell, month) pair.
    # Shape: (n_rows,). Layout: cell-major (cell 1 months 1..100, cell 2 months
    # 1..100, ...). This matches the VIEWS viewser long-format contract.
    pgm_ids = np.repeat(
        np.arange(1, _N_PGM_CELLS_LARGE + 1, dtype=np.int64), _N_MONTHS_PER_CELL
    )
    month_ids = np.tile(
        np.arange(_MONTH_ID_START, _MONTH_ID_START + _N_MONTHS_PER_CELL, dtype=np.int64),
        _N_PGM_CELLS_LARGE,
    )

    # Build the value columns. Fatality counts are zero-inflated (~80% zeros)
    # with log-normal noise for the non-zero tail.
    columns: dict[str, np.ndarray] = {
        "month_id": month_ids,
        "priogrid_id": pgm_ids,
    }

    for col_name in SYNTHETIC_TARGETS + SYNTHETIC_FEATURES:
        # Zero-inflated log-normal: 80% zeros, 20% log-normal(2, 1.5).
        mask = rng.random(n_rows) < 0.20
        values = np.zeros(n_rows, dtype=np.float64)
        values[mask] = rng.lognormal(mean=2.0, sigma=1.5, size=mask.sum())
        # Cast to float32 (the airlock invariant) and clip negatives (physical
        # floor for fatality counts).
        values = np.maximum(values, 0.0).astype(np.float32)
        columns[col_name] = values

    # Write via pyarrow (no pandas). The column order matches the VIEWS
    # viewser convention: value columns first, index columns last.
    table = pa.table(columns)
    pq.write_table(table, str(parquet_path))
    return parquet_path


@pytest.fixture(scope="session")
def synthetic_pgm_parquet_small(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A smaller synthetic parquet (~1000 cells × 100 months = 100k rows).

    Used by tests that need speed (e.g., the full as_darts_timeseries flow
    over all entities). The large fixture above is for memmap + parity tests.
    """
    cache_dir = tmp_path_factory.mktemp("synthetic_parquet_small")
    parquet_path = cache_dir / "validation_viewser_df_small.parquet"

    if parquet_path.exists():
        return parquet_path

    n_cells = 1000
    n_months = 100
    n_rows = n_cells * n_months
    rng = np.random.default_rng(seed=123)

    pgm_ids = np.repeat(np.arange(1, n_cells + 1, dtype=np.int64), n_months)
    month_ids = np.tile(
        np.arange(_MONTH_ID_START, _MONTH_ID_START + n_months, dtype=np.int64),
        n_cells,
    )

    columns: dict[str, np.ndarray] = {
        "month_id": month_ids,
        "priogrid_id": pgm_ids,
    }
    for col_name in SYNTHETIC_TARGETS + SYNTHETIC_FEATURES:
        mask = rng.random(n_rows) < 0.20
        values = np.zeros(n_rows, dtype=np.float64)
        values[mask] = rng.lognormal(mean=2.0, sigma=1.5, size=mask.sum())
        values = np.maximum(values, 0.0).astype(np.float32)
        columns[col_name] = values

    table = pa.table(columns)
    pq.write_table(table, str(parquet_path))
    return parquet_path


@pytest.fixture(scope="session")
def synthetic_cm_parquet_small(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small country-month parquet (~200 countries × 100 months = 20k rows).

    Used by tests that need the cm level (country_id entity column) and fast
    iteration. Mirrors the real validation parquet's spatial level.
    """
    cache_dir = tmp_path_factory.mktemp("synthetic_cm_parquet")
    parquet_path = cache_dir / "validation_viewser_df_cm.parquet"

    if parquet_path.exists():
        return parquet_path

    n_countries = 200
    n_months = 100
    n_rows = n_countries * n_months
    rng = np.random.default_rng(seed=456)

    country_ids = np.repeat(np.arange(1, n_countries + 1, dtype=np.int64), n_months)
    month_ids = np.tile(
        np.arange(_MONTH_ID_START, _MONTH_ID_START + n_months, dtype=np.int64),
        n_countries,
    )

    columns: dict[str, np.ndarray] = {
        "month_id": month_ids,
        "country_id": country_ids,
    }
    for col_name in SYNTHETIC_TARGETS + SYNTHETIC_FEATURES:
        mask = rng.random(n_rows) < 0.20
        values = np.zeros(n_rows, dtype=np.float64)
        values[mask] = rng.lognormal(mean=2.0, sigma=1.5, size=mask.sum())
        values = np.maximum(values, 0.0).astype(np.float32)
        columns[col_name] = values

    table = pa.table(columns)
    pq.write_table(table, str(parquet_path))
    return parquet_path


# ----------------------------------------------------------------------
# Convenience fixtures that return (path, targets, features) tuples.
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def pgm_parquet_and_columns(
    synthetic_pgm_parquet: Path,
) -> tuple[Path, list[str], list[str]]:
    """Return ``(parquet_path, SYNTHETIC_TARGETS, SYNTHETIC_FEATURES)``."""
    return synthetic_pgm_parquet, list(SYNTHETIC_TARGETS), list(SYNTHETIC_FEATURES)


@pytest.fixture(scope="session")
def cm_parquet_and_columns(
    synthetic_cm_parquet_small: Path,
) -> tuple[Path, list[str], list[str]]:
    """Return ``(parquet_path, SYNTHETIC_TARGETS, SYNTHETIC_FEATURES)`` for cm."""
    return synthetic_cm_parquet_small, list(SYNTHETIC_TARGETS), list(SYNTHETIC_FEATURES)
