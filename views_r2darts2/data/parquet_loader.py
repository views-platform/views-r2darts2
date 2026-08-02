"""Parquet loader for VIEWS viewser dataframes (pandas-free, memmap-friendly).

Reads a VIEWS-format parquet file directly via ``pyarrow.parquet`` and produces
a :class:`views_frames.FeatureFrame` without ever materializing a
``pandas.DataFrame``.

The parquet schema is expected to be the long-format VIEWS viewser contract:

    * Two index columns (``month_id`` + ``country_id`` for cm, or ``month_id`` +
      ``priogrid_id`` for pgm). The index columns may appear at any position in
      the parquet file (the viewser writer historically appends them last).
    * One float column per feature/target. All such columns are coerced to
      ``float32`` exactly once at the airlock boundary (ADR-010).

The returned :class:`FeatureFrame` holds **all** value columns (features first,
then targets) in its ``(N, F, 1)`` value array, with ``feature_names`` carrying
the column-name order. The caller receives the ``targets`` and ``features``
name lists alongside the frame so it can split the columns back out without
re-reading the parquet.

When ``cache_dir`` is provided, the loader writes a native ``FeatureFrame`` save
directory (``values.npy`` + ``identifiers.npz`` + ``header.json``) on first read
and memmaps the values on subsequent reads — peak RSS stays the working set
(register C-07, README §7 of views-frames).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from views_frames import (
    FeatureFrame,
    FrameMetadata,
    SpatioTemporalIndex,
    SpatialLevel,
)

logger = logging.getLogger(__name__)

# Recognized index-column name pairs. The loader infers the spatial level from
# the entity column that is actually present in the source.
_LEVEL_BY_ENTITY_COLUMN: dict[str, SpatialLevel] = {
    "country_id": SpatialLevel.CM,
    "priogrid_id": SpatialLevel.PGM,
}

# Canonical column names for every supported VIEWS spatio-temporal resolution.
# The first element is the time column, the second is the entity column.
# Levels follow the VIEWS naming convention:
#   prefix  — c = country, pg = priogrid
#   suffix  — m = month, w = week, d = day, y = year
_LEVEL_COLUMNS: dict[str, tuple[str, str]] = {
    "cm":  ("month_id",  "country_id"),
    "pgm": ("month_id",  "priogrid_id"),
    "cw":  ("week_id",   "country_id"),
    "pgw": ("week_id",   "priogrid_id"),
    "cd":  ("day_id",    "country_id"),
    "pgd": ("day_id",    "priogrid_id"),
    "cy":  ("year_id",   "country_id"),
    "pgy": ("year_id",   "priogrid_id"),
}


def resolve_columns_for_level(level: str) -> tuple[str, str]:
    """Return ``(time_id, entity_id)`` for a VIEWS spatio-temporal level string.

    Args:
        level: A VIEWS level string (e.g. ``"cm"``, ``"pgm"``, ``"pgd"``).

    Returns:
        A ``(time_id, entity_id)`` pair suitable for passing to
        :func:`load_views_parquet`.

    Raises:
        ParquetLoadError: ``level`` is not a recognised VIEWS level.
    """
    if level not in _LEVEL_COLUMNS:
        raise ParquetLoadError(
            f"Unrecognised VIEWS level '{level}'. "
            f"Expected one of {sorted(_LEVEL_COLUMNS)}."
        )
    return _LEVEL_COLUMNS[level]


class ParquetLoadError(ValueError):
    """Raised when a parquet file does not match the VIEWS viewser contract."""


def _resolve_level(entity_id: str) -> SpatialLevel:
    """Map an entity column name to its :class:`SpatialLevel`.

    Recognizes both canonical names (``country_id``, ``priogrid_id``) and
    aliases (``priogrid_gid`` → ``priogrid_id`` → ``PGM``).
    """
    # Canonical names.
    if entity_id in _LEVEL_BY_ENTITY_COLUMN:
        return _LEVEL_BY_ENTITY_COLUMN[entity_id]
    # Aliases.
    if entity_id in _ENTITY_ALIASES:
        canonical = _ENTITY_ALIASES[entity_id]
        return _LEVEL_BY_ENTITY_COLUMN[canonical]
    raise ParquetLoadError(
        f"Unrecognized entity column '{entity_id}'. "
        f"Expected one of {sorted(_LEVEL_BY_ENTITY_COLUMN)} "
        f"or aliases {sorted(_ENTITY_ALIASES)}."
    )


# Aliases that the loader will silently normalize to the canonical name.
# ``priogrid_gid`` is a typo that appears in some older viewser datasets; we
# treat it as ``priogrid_id`` so callers don't need to know the exact spelling.
_ENTITY_ALIASES: dict[str, str] = {
    "priogrid_gid": "priogrid_id",
}


def _canonical_entity_column(name: str) -> str:
    """Return canonical entity column name (normalizes known aliases)."""
    return _ENTITY_ALIASES.get(name, name)


def _resolve_entity_column(declared: str, available: set[str]) -> str:
    """Validate that the declared entity column is present in the source schema.

    The entity and time columns are derived from ``config["level"]`` upstream
    (via :func:`resolve_columns_for_level`) and therefore must be present
    exactly as declared — cross-level fallback is explicitly not supported.
    The only normalization that still happens is the ``priogrid_gid`` typo
    alias so that legacy datasets written before the canonical name was fixed
    can still be read.

    Args:
        declared: The entity column name expected in the source.
        available: The set of column names in the source schema.

    Returns:
        The resolved entity column name (the declared name, or its canonical
        alias if a known typo is present).

    Raises:
        ParquetLoadError: The declared column (and any known typo alias) is
            absent from the source schema.
    """
    # Exact match — the common case.
    if declared in available:
        return declared

    # Typo-alias normalisation only (e.g. priogrid_gid → priogrid_id).
    # We do NOT cross the entity boundary (country_id ↔ priogrid_id) — the
    # correct entity column must come from config["level"].
    for alias, canonical in _ENTITY_ALIASES.items():
        if canonical == declared and alias in available:
            logger.info(
                "Entity column '%s' resolved via alias '%s'.",
                declared, alias,
            )
            return alias

    # Not found — return the declared name so the downstream missing-column
    # error carries the original column name.
    return declared


def _read_parquet_columns(
    path: Path | str, columns: list[str]
) -> dict[str, NDArray[Any]]:
    """Read selected columns from a parquet file as numpy arrays.

    Uses ``pyarrow.parquet.read_table`` with an explicit ``columns=`` projection
    so only the requested columns are decoded — peak memory stays proportional
    to the working set, not the full parquet row group size.
    """
    table = pq.read_table(str(path), columns=columns)
    return {name: table.column(name).to_numpy() for name in columns}


def load_views_parquet(
    path: Path | str,
    *,
    targets: list[str],
    features: list[str] | None = None,
    time_id: str = "month_id",
    entity_id: str = "country_id",
    cache_dir: Path | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[FeatureFrame, list[str], list[str]]:
    """Load a VIEWS parquet file into a :class:`FeatureFrame`.

    Args:
        path: Path to the ``*_viewser_df.parquet`` file.
        targets: Target column names (mandatory, non-empty).
        features: Feature column names. When ``None`` or empty, the returned
            ``FeatureFrame`` carries only target columns. Callers that want a
            no-feature dataset for inference can pass an empty list explicitly.
        time_id: Time index column name (default ``month_id``).
        entity_id: Entity index column name (default ``country_id``; pass
            ``priogrid_id`` for pgm data).
        cache_dir: When provided, the loader writes a native ``FeatureFrame``
            save directory on first read and memmaps the values on subsequent
            reads. When ``None``, the parquet is decoded in-memory every call.
        metadata: Optional provenance header attached to the frame.

    Returns:
        A ``(FeatureFrame, features, targets)`` triple. The frame's value axis
        is ordered ``[features..., targets...]`` and its ``feature_names``
        attribute carries the same order. The two name lists are returned
        alongside so the caller can split the columns without re-reading.

    Raises:
        ParquetLoadError: The parquet schema is missing a declared target,
            feature, or index column, or a target column is also declared as a
            feature.
    """
    if not targets:
        raise ParquetLoadError("`targets` must be a non-empty list.")

    # Normalize caller-provided aliases so downstream semantics always use
    # canonical VIEWS column names.
    entity_id = _canonical_entity_column(entity_id)

    features = list(features) if features else []
    overlap = sorted(set(targets).intersection(features))
    if overlap:
        raise ParquetLoadError(
            "A column cannot be both target and feature. "
            f"Overlapping columns: {overlap}."
        )

    path = Path(path)
    if not path.exists():
        raise ParquetLoadError(f"Parquet file not found: {path}")

    # When caching is requested, prefer the native on-disk frame and only fall
    # back to parquet decoding on a cache miss. The cache key is a stable hash
    # of the resolved path + column manifest so the cache is invalidated any
    # time the manifest changes.
    if cache_dir is not None:
        cache_path = _resolve_cache_path(
            cache_dir, path, targets, features, time_id, entity_id
        )
        if _cache_is_valid(cache_path):
            logger.info("Loading FeatureFrame from cache: %s", cache_path)
            frame = FeatureFrame.load(cache_path, mmap=True)
            if metadata is not None:
                md = FrameMetadata.from_dict(dict(metadata))
                frame = frame.with_metadata(md)
            return frame, list(features), list(targets)

    # Cache miss — decode parquet directly via pyarrow (no pandas).
    schema = pq.read_schema(str(path))
    available = set(schema.names)

    # Auto-detect the entity column when the declared one is absent.
    # The VIEWS viewser contract uses ``country_id`` for cm data and
    # ``priogrid_id`` for pgm data, but some older datasets use
    # ``priogrid_gid`` (a typo). Resolve the actual entity column here so
    # callers don't need to know the level upfront — they can pass the default
    # ``country_id`` and the loader will fall back to ``priogrid_id`` (or the
    # ``priogrid_gid`` alias) when the parquet is pgm-level.
    entity_column = _resolve_entity_column(entity_id, available)
    entity_id = _canonical_entity_column(entity_column)

    required = {time_id, entity_column, *targets, *features}
    missing = required - available
    if missing:
        raise ParquetLoadError(
            f"Parquet file {path} is missing required columns: {sorted(missing)}."
        )

    # Value-axis order: features first, then targets not already present.
    # This allows targets to also be listed as features without duplicate
    # parquet projections or duplicated component names.
    value_columns = list(features)
    for target in targets:
        if target not in value_columns:
            value_columns.append(target)
    column_order = [time_id, entity_column, *value_columns]
    raw = _read_parquet_columns(path, column_order)
    time_arr = np.ascontiguousarray(raw[time_id]).astype(np.int64, copy=False)
    entity_arr = np.ascontiguousarray(raw[entity_column]).astype(np.int64, copy=False)

    # Cast every value column to float32 at the airlock boundary (ADR-010).
    if value_columns:
        value_matrix = np.stack(
            [np.ascontiguousarray(raw[name], dtype=np.float32) for name in value_columns],
            axis=1,
        )
    else:
        value_matrix = np.empty((time_arr.shape[0], 0), dtype=np.float32)

    level = _resolve_level(entity_id)
    index = SpatioTemporalIndex(time=time_arr, unit=entity_arr, level=level)
    frame_metadata = FrameMetadata.from_dict(dict(metadata)) if metadata else None

    # FeatureFrame requires a trailing sample axis (ADR-012). Raw viewser data
    # is single-sample (S=1) — lift the 2D (N, F) array to (N, F, 1) via
    # ``from_2d`` so the frame is always 3D internally.
    frame = FeatureFrame.from_2d(
        value_matrix,
        index=index,
        feature_names=value_columns,
        metadata=frame_metadata,
    )

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.save(cache_path)
        logger.info("Wrote FeatureFrame cache: %s", cache_path)

    return frame, list(features), list(targets)


def _resolve_cache_path(
    cache_dir: Path | str,
    parquet_path: Path,
    targets: list[str],
    features: list[str],
    time_id: str,
    entity_id: str,
) -> Path:
    """Stable, content-addressed cache directory for a (parquet, manifest) pair."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = json.dumps(
        {
            "parquet": str(parquet_path.resolve()),
            "mtime": parquet_path.stat().st_mtime,
            "size": parquet_path.stat().st_size,
            "targets": targets,
            "features": features,
            "time_id": time_id,
            "entity_id": entity_id,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(manifest.encode("utf-8")).hexdigest()[:16]
    return cache_root / f"views_df_{digest}"


def _cache_is_valid(cache_path: Path) -> bool:
    """True iff the cache directory has all required files."""
    if not cache_path.exists():
        return False
    return (
        (cache_path / "values.npy").exists()
        and (cache_path / "identifiers.npz").exists()
        and (cache_path / "header.json").exists()
    )
