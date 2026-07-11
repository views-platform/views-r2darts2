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
# whichever entity column is present. ``time_id`` is always ``month_id`` for the
# VIEWS platform, but the loader is generic enough to accept any integer time
# column declared in the manifest.
_LEVEL_BY_ENTITY_COLUMN: dict[str, SpatialLevel] = {
    "country_id": SpatialLevel.CM,
    "priogrid_id": SpatialLevel.PGM,
}


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


def _resolve_entity_column(declared: str, available: set[str]) -> str:
    """Resolve the actual entity column name against the parquet schema.

    The caller declares an ``entity_id`` (default ``"country_id"``), but the
    parquet may use the other level's column (``"priogrid_id"`` for pgm data)
    or a typo alias (``"priogrid_gid"``). This helper:

        1. If the declared column is present, return it as-is.
        2. If the declared column is absent, check the other level's canonical
           column (``country_id`` ↔ ``priogrid_id``).
        3. Check aliases (``priogrid_gid`` → ``priogrid_id``).
        4. Log the resolution so the caller can see what happened.

    Args:
        declared: The entity column name the caller requested.
        available: The set of column names in the parquet schema.

    Returns:
        The resolved entity column name (one of the canonical names
        ``country_id`` or ``priogrid_id``).
    """
    # Step 1: declared column is present.
    if declared in available:
        return declared

    # Step 2: check the other level's canonical column.
    canonical_names = set(_LEVEL_BY_ENTITY_COLUMN.keys())
    for canonical in canonical_names:
        if canonical != declared and canonical in available:
            logger.info(
                "Entity column '%s' not found in parquet — falling back to "
                "'%s'.", declared, canonical
            )
            return canonical

    # Step 3: check aliases (e.g. priogrid_gid → priogrid_id).
    for alias, canonical in _ENTITY_ALIASES.items():
        if alias in available:
            logger.info(
                "Entity column '%s' not found in parquet — using alias '%s' "
                "(will be normalized to '%s' in the frame index).",
                declared, alias, canonical,
            )
            # Return the ALIAS name (the actual parquet column), not the
            # canonical name — the caller needs to read this column from the
            # parquet. The level is resolved from the canonical name via
            # ``_resolve_level`` after we know which alias mapped to which
            # canonical. To make that work, we return the alias here and
            # adjust ``_resolve_level`` to recognize aliases.
            return alias

    # Step 4: none found — return the declared name so the caller gets the
    # original "missing column" error message.
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

    features = list(features) if features else []
    overlap = set(targets).intersection(features)
    if overlap:
        raise ParquetLoadError(
            f"Columns cannot be both target and feature: {sorted(overlap)}."
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
    entity_id = _resolve_entity_column(entity_id, available)

    required = {time_id, entity_id, *targets, *features}
    missing = required - available
    if missing:
        raise ParquetLoadError(
            f"Parquet file {path} is missing required columns: {sorted(missing)}."
        )

    # Value-axis order: features first, then targets. This matches the Darts
    # ``value_cols`` order used downstream so the TimeSeries components line up
    # with the feature_names list one-to-one.
    value_columns = [*features, *targets]
    column_order = [time_id, entity_id, *value_columns]
    raw = _read_parquet_columns(path, column_order)
    time_arr = np.ascontiguousarray(raw[time_id]).astype(np.int64, copy=False)
    entity_arr = np.ascontiguousarray(raw[entity_id]).astype(np.int64, copy=False)

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
