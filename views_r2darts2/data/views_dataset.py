"""FeatureFrame-backed VIEWS dataset (the Darts data boundary).

Replaces the legacy ``_ViewsDatasetDarts`` which inherited from
``views_pipeline_core.data.handlers._ViewsDataset`` and held a pandas DataFrame.
This module holds a :class:`views_frames.FeatureFrame` directly and exposes the
same surface the rest of the package needs:

    * ``targets`` / ``features`` — column-name lists (matches the legacy API).
    * ``_time_id`` / ``_entity_id`` — index column names (matches the legacy
      API; used by ``DartsForecaster`` to label output rows).
    * ``as_darts_timeseries(...)`` — build a list of per-entity Darts
      ``TimeSeries`` from the frame, with optional cyclic encoders and
      per-entity static covariates.

The frame is memmap-friendly: when ``FeatureFrame.load(..., mmap=True)`` is
used at construction time, the values array is a ``np.memmap`` and the dataset
operates without materializing the full frame in RAM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from darts import TimeSeries

from views_frames import FeatureFrame, SpatioTemporalIndex, SpatialLevel

from views_r2darts2.data.parquet_loader import load_views_parquet
from views_r2darts2.infrastructure.encoders import CYCLIC_ENCODERS_BY_RESOLUTION
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate
from views_r2darts2.transformers.darts_bridge import build_entity_timeseries
from views_r2darts2.transformers.static_covariates import (
    StaticCovariateConfig,
    StaticCovariateStats,
    compute_static_covariates,
)

logger = logging.getLogger(__name__)

# All stat names supported by the per-entity static-covariate fingerprint.
ALL_STAT_NAMES: tuple[str, ...] = ("mu", "sigma", "max", "trend", "sparsity")


class ViewsDatasetDarts:
    """FeatureFrame-backed data boundary between VIEWS and Darts.

    Intent Contract:
        - Purpose: Hold the raw VIEWS viewser dataframe as a
          :class:`FeatureFrame` (memmap-friendly) and expose a single
          ``as_darts_timeseries`` method that builds per-entity Darts
          ``TimeSeries`` collections on demand.
        - Non-Goals: Does not perform data cleaning or temporal slicing
          (slicing is handled by the forecaster). Does not own the scalers.
        - Guarantees:
            - Feature and target columns are stored as float32 at the airlock
              boundary (ADR-010); the underlying buffer is the FeatureFrame's
              ``values`` array (which may be a ``np.memmap``).
            - The Darts TimeSeries collection preserves the multi-index
              semantic structure: one TimeSeries per entity, time-major.
        - Failure Behavior: Raises ``ParquetLoadError`` if the parquet schema
          does not match the declared manifest. Raises ``ValueError`` if the
          frame is empty or the index is malformed.
    """

    def __init__(
        self,
        feature_frame: FeatureFrame,
        *,
        targets: Sequence[str],
        features: Sequence[str] | None = None,
        time_id: str = "month_id",
        entity_id: str = "country_id",
    ) -> None:
        if not targets:
            raise ValueError("`targets` must be a non-empty list of column names.")
        features = list(features) if features else []

        # The frame's feature_names axis is the source of truth for which
        # columns exist; the targets/features lists are subsets of it.
        available = set(feature_frame.feature_names)
        missing_targets = set(targets) - available
        if missing_targets:
            raise ValueError(
                f"FeatureFrame is missing target columns: {sorted(missing_targets)}."
            )
        missing_features = set(features) - available
        if missing_features:
            raise ValueError(
                f"FeatureFrame is missing feature columns: {sorted(missing_features)}."
            )
        overlap = sorted(set(targets).intersection(features))
        if overlap:
            logger.info(
                "Using %d target columns as features as requested: %s",
                len(overlap),
                overlap,
            )

        self._frame = feature_frame
        self._targets = list(targets)
        self._features = list(features)
        self._time_id = time_id
        self._entity_id = entity_id

        # Audit the incoming frame schema immediately at construction (ADR-009).
        ReproducibilityGate.Data.audit_frame_schema(
            feature_frame=feature_frame,
            expected_targets=self._targets,
            expected_features=self._features,
        )

    # ------------------------------------------------------------------ factory

    @classmethod
    def from_views_path(
        cls,
        path_raw: str | Path,
        run_type: str,
        config: Mapping[str, Any],
        *,
        cached_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> "ViewsDatasetDarts":
        """Factory: load a VIEWS parquet file and build a dataset.

        Args:
            path_raw: Directory containing the ``<run_type>_viewser_df.parquet``
                file. Ignored when ``cached_path`` is provided.
            run_type: VIEWS run type (e.g. ``"validation"``, ``"calibration"``).
            config: Experiment manifest. ``config["targets"]`` is mandatory;
                ``config["features"]`` is optional (defaults to all non-target
                value columns when absent — see ``_resolve_features_from_config``).
            cached_path: When provided, used directly instead of constructing a
                path from ``path_raw`` and ``run_type``.
            cache_dir: When provided, the parquet is decoded once and a native
                ``FeatureFrame`` cache (memmap-friendly) is written/loaded from
                this directory on subsequent calls.

        Returns:
            A :class:`ViewsDatasetDarts` holding the loaded :class:`FeatureFrame`.
        """
        file_path = Path(cached_path) if cached_path is not None else (
            Path(path_raw) / f"{run_type}_viewser_df.parquet"
        )

        targets = list(config.get("targets") or [])
        features = _resolve_features_from_config(config, targets)

        # Strict allowlist mode: when feature_scaler_map is provided, only the
        # listed features are eligible as model covariates. Any extra value
        # columns present in parquet are intentionally ignored.
        if config.get("feature_scaler_map"):
            ignored_features = _enumerate_ignored_input_features(
                file_path=file_path,
                targets=targets,
                allowed_features=features,
            )
            if ignored_features:
                logger.info(
                    "Ignoring %d input feature columns not present in "
                    "feature_scaler_map: %s",
                    len(ignored_features),
                    ignored_features,
                )

        time_id = config.get("time_id", "month_id")
        declared_entity_id = config.get("entity_id", "country_id")

        frame, features, _ = load_views_parquet(
            file_path,
            targets=targets,
            features=features,
            time_id=time_id,
            entity_id=declared_entity_id,
            cache_dir=cache_dir,
        )

        # The loader may have auto-detected a different entity column (e.g.,
        # ``priogrid_id`` when ``country_id`` was declared). Resolve the actual
        # entity_id from the frame's spatial level so the dataset labels output
        # rows with the correct column name.
        actual_entity_id = frame.index.level.entity_column

        return cls(
            feature_frame=frame,
            targets=targets,
            features=features,
            time_id=time_id,
            entity_id=actual_entity_id,
        )

    # ------------------------------------------------------------------ accessors

    @property
    def feature_frame(self) -> FeatureFrame:
        """The underlying :class:`FeatureFrame` (may be memmap-backed)."""
        return self._frame

    @property
    def targets(self) -> list[str]:
        """Target column names (subset of ``feature_frame.feature_names``)."""
        return list(self._targets)

    @property
    def features(self) -> list[str]:
        """Feature column names (subset of ``feature_frame.feature_names``)."""
        return list(self._features)

    @property
    def time_id(self) -> str:
        """Time index column name (e.g. ``"month_id"``)."""
        return self._time_id

    @property
    def entity_id(self) -> str:
        """Entity index column name (e.g. ``"country_id"``)."""
        return self._entity_id

    @property
    def level(self) -> SpatialLevel:
        """Spatial level of the underlying frame (cm or pgm)."""
        return self._frame.index.level

    @property
    def n_rows(self) -> int:
        """Total row count (``N``)."""
        return self._frame.n_rows

    @property
    def n_entities(self) -> int:
        """Number of unique entities in the frame."""
        return int(np.unique(self._frame.index.unit).shape[0])

    @property
    def n_time_steps(self) -> int:
        """Number of unique time steps in the frame."""
        return int(np.unique(self._frame.index.time).shape[0])

    # ------------------------------------------------------------------ slicing

    def get_subset_arrays(
        self,
        *,
        time_ids: int | Sequence[int] | None = None,
        entity_ids: int | Sequence[int] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
        """Return ``(time, entity, values_2d)`` for the requested row subset.

        Args:
            time_ids: Optional time-id filter (single int or list of ints).
            entity_ids: Optional entity-id filter (single int or list of ints).

        Returns:
            A triple of numpy arrays — the time and entity identifier columns
            and a 2D ``(N_subset, F)`` view of the values (single-sample case
            only; the trailing sample axis is squeezed).
        """
        mask = np.ones(self._frame.n_rows, dtype=np.bool_)
        if time_ids is not None:
            wanted_times = np.atleast_1d(np.asarray(time_ids, dtype=np.int64))
            mask &= np.isin(self._frame.index.time, wanted_times)
        if entity_ids is not None:
            wanted_entities = np.atleast_1d(np.asarray(entity_ids, dtype=np.int64))
            mask &= np.isin(self._frame.index.unit, wanted_entities)

        time_subset = self._frame.index.time[mask]
        entity_subset = self._frame.index.unit[mask]
        # Squeeze the trailing sample axis (always 1 for raw viewser data).
        values_subset = np.ascontiguousarray(self._frame.values[mask, :, 0])
        return time_subset, entity_subset, values_subset

    # ------------------------------------------------------------------ darts bridge

    def as_darts_timeseries(
        self,
        *,
        time_ids: int | Sequence[int] | None = None,
        entity_ids: int | Sequence[int] | None = None,
        stat_time_range: tuple[int, int] | None = None,
        static_cov_transform: str | None = None,
        static_cov_stats: Sequence[str] | None = None,
        inject_static_covariates: bool = False,
        use_cyclic_encoders: bool = False,
    ) -> list[TimeSeries]:
        """Build a list of per-entity Darts ``TimeSeries`` from the frame.

        Args:
            time_ids: Optional time-id filter (single int or list).
            entity_ids: Optional entity-id filter (single int or list).
            stat_time_range: ``(start, end)`` inclusive time-id window for the
                per-entity static-covariate fingerprint. Required when
                ``inject_static_covariates=True`` to prevent test-period leakage;
                ignored otherwise.
            static_cov_transform: Optional transform chain for the fingerprint
                (e.g. ``"AsinhTransform->MaxAbsScaler"``). See
                :class:`StaticCovariateConfig` for the supported vocabulary.
            static_cov_stats: Subset of ``("mu", "sigma", "max", "trend",
                "sparsity")`` to inject. Defaults to all five.
            inject_static_covariates: When ``True``, compute the per-entity
                fingerprint from ``stat_time_range`` and inject as Darts static
                covariates. When ``False``, only the entity id is attached as a
                static covariate (preserves the legacy ``group_cols`` behavior).
            use_cyclic_encoders: When ``True``, append sin/cos cyclic time
                encoders to the feature axis (resolution inferred from
                ``self._time_id``).

        Returns:
            A list of Darts :class:`TimeSeries` objects, one per entity, ordered
            by the first appearance of each entity in the frame. Each series
            carries the full ``features + targets`` value axis (features first,
            then targets — matching the frame's column order).
        """
        time_subset, entity_subset, values_2d = self.get_subset_arrays(
            time_ids=time_ids, entity_ids=entity_ids
        )
        if time_subset.shape[0] == 0:
            return []

        # Keep features first; append only targets not already present.
        value_columns = list(self._features)
        for target in self._targets:
            if target not in value_columns:
                value_columns.append(target)
        # Sanity: the frame's column order must match what we promise to Darts.
        if value_columns != self._frame.feature_names:
            # The frame may carry extra columns; align to the declared order.
            col_indices = [self._frame.feature_names.index(c) for c in value_columns]
            values_2d = np.ascontiguousarray(values_2d[:, col_indices])

        # --- Cyclic time encoders (past covariates) ----------------------
        # Infer temporal resolution from the time-id name (``month_id`` → ``m``,
        # ``week_id`` → ``w``, ``day_id`` → ``d``, ``year_id`` → ``y``). The
        # encoders are pure calendar math (zero leakage) and produce values in
        # [-1, 1]; no scaling needed.
        feature_columns_ext = list(value_columns)
        appended_cyclic: list[NDArray[np.float32]] = []
        if use_cyclic_encoders:
            resolution = self._time_id.split("_")[0][0]
            cyclic_encoders = CYCLIC_ENCODERS_BY_RESOLUTION.get(resolution)
            if cyclic_encoders is not None:
                for enc_fn in cyclic_encoders:
                    col = enc_fn(time_subset).astype(np.float32)
                    appended_cyclic.append(col)
                    feature_columns_ext.append(enc_fn.__name__)
                logger.info(
                    "Cyclic encoders: injected %s for resolution '%s' (%s).",
                    [fn.__name__ for fn in cyclic_encoders],
                    resolution,
                    self._time_id,
                )

        # --- Per-entity split ---------------------------------------------
        # Sort by (entity, time) so each entity's rows are contiguous and
        # time-ascending. Then split at entity boundaries. This is the
        # numpy-native replacement for ``df.groupby(entity_id)``.
        sort_order = np.lexsort((time_subset, entity_subset))
        time_sorted = time_subset[sort_order]
        entity_sorted = entity_subset[sort_order]
        values_sorted = np.ascontiguousarray(values_2d[sort_order])

        if appended_cyclic:
            cyclic_block = np.stack([c[sort_order] for c in appended_cyclic], axis=1)
            values_sorted = np.concatenate(
                [values_sorted, cyclic_block], axis=1
            ).astype(np.float32, copy=False)

        # Boundary indices: the start of each entity's block. ``np.searchsorted``
        # on the sorted entity column gives O(log N) per lookup, but the simpler
        # ``np.flatnonzero(np.diff(entity_sorted) != 0) + 1`` is O(N) and faster
        # for the typical entity count (~200 countries, ~24k pgm cells).
        boundaries = _entity_boundaries(entity_sorted)
        entity_ids_unique = entity_sorted[boundaries]

        # --- Per-entity static covariates ---------------------------------
        if inject_static_covariates:
            stat_cfg = StaticCovariateConfig(
                transform=static_cov_transform,
                stats=tuple(static_cov_stats) if static_cov_stats else ALL_STAT_NAMES,
            )
            static_cov = compute_static_covariates(
                time=time_subset,
                entity=entity_subset,
                values=values_2d,
                target_columns=self._targets,
                column_order=value_columns,
                stat_time_range=stat_time_range,
                config=stat_cfg,
            )
        else:
            logger.info(
                "inject_static_covariates=False — skipping static covariate "
                "fingerprint injection."
            )
            static_cov = None

        # --- Build per-entity TimeSeries ----------------------------------
        # The Darts-boundary helper ``build_entity_timeseries`` is the ONLY
        # place pandas is imported in the package — confined to
        # ``transformers/darts_bridge.py``. The rest of this module operates on
        # numpy arrays + the FeatureFrame.
        series_list: list[TimeSeries] = []
        for i, entity_id_value in enumerate(entity_ids_unique):
            start = boundaries[i]
            stop = boundaries[i + 1] if i + 1 < len(boundaries) else len(entity_sorted)
            entity_time = time_sorted[start:stop]
            entity_values = values_sorted[start:stop, :]

            static_for_entity = (
                static_cov.row_for_entity(int(entity_id_value))
                if static_cov is not None
                else None
            )
            ts = build_entity_timeseries(
                time=entity_time,
                values=entity_values,
                columns=feature_columns_ext,
                entity_id_name=self._entity_id,
                entity_id_value=int(entity_id_value),
                static_covariates=static_for_entity,
            )
            series_list.append(ts)

        return series_list


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_features_from_config(
    config: Mapping[str, Any], targets: Sequence[str]
) -> list[str]:
    """Resolve the feature column list from the experiment manifest.

    The legacy contract is:
                * ``config["force_target_only"] == True``) — return an empty feature
                    list and run in target-only mode.
        * ``config["feature_scaler_map"]`` present — STRICT allowlist mode;
          derive features from the union of all column lists in the map values
          (order-preserving) and ignore all other potential input features.
        * ``config["features"]`` (explicit list) and no scaler map — use as-is.
        * ``config["broadcast_features"] == True`` and no explicit features —
          load all non-target value columns from the parquet.
        * Otherwise — empty feature list (targets only).
    """
    if bool(config.get("force_target_only")):
        logger.info(
            "Target-only mode enabled via config flag "
            "(force_target_only/disable_features). No features will be passed "
            "to the model."
        )
        return []

    feature_scaler_map = config.get("feature_scaler_map")
    if feature_scaler_map:
        seen: dict[str, None] = {}
        for cols in feature_scaler_map.values():
            for col in cols:
                seen[col] = None
        features_from_map = list(seen)
        explicit = config.get("features")
        if explicit:
            explicit_not_in_map = [f for f in explicit if f not in seen]
            if explicit_not_in_map:
                logger.info(
                    "Ignoring %d config['features'] entries not present in "
                    "feature_scaler_map: %s",
                    len(explicit_not_in_map),
                    explicit_not_in_map,
                )
        return features_from_map

    explicit = config.get("features")
    if explicit:
        return list(explicit)

    if config.get("broadcast_features"):
        # Load the parquet schema to enumerate non-target columns.
        path_raw = config.get("path_raw")
        run_type = config.get("run_type", "validation")
        cached_path = config.get("cached_path")
        if cached_path is not None:
            file_path = Path(cached_path)
        elif path_raw is not None:
            file_path = Path(path_raw) / f"{run_type}_viewser_df.parquet"
        else:
            return []
        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(str(file_path))
            targets_set = set(targets)
            return [
                name
                for name in schema.names
                if name not in targets_set
                and name not in ("month_id", "country_id", "priogrid_id")
            ]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not enumerate features from %s: %s", file_path, exc)
            return []
    return []


def _enumerate_ignored_input_features(
    *,
    file_path: Path,
    targets: Sequence[str],
    allowed_features: Sequence[str],
) -> list[str]:
    """Return non-target parquet value columns that are excluded by allowlist.

    This is used only for observability in strict ``feature_scaler_map`` mode.
    If schema introspection fails, we silently skip logging rather than fail
    dataset construction.
    """
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(str(file_path))
    except Exception as exc:  # pragma: no cover - defensive observability
        logger.warning(
            "Could not enumerate input features from %s for ignore logging: %s",
            file_path,
            exc,
        )
        return []

    targets_set = set(targets)
    allowed_set = set(allowed_features)
    reserved = {
        "month_id",
        "country_id",
        "priogrid_id",
        "priogrid_gid",
    }
    input_value_features = [
        name
        for name in schema.names
        if name not in targets_set and name not in reserved
    ]
    return sorted(name for name in input_value_features if name not in allowed_set)


def _entity_boundaries(entity_sorted: NDArray[np.int64]) -> NDArray[np.intp]:
    """Return the start index of each entity's contiguous block (sorted array).

    Equivalent to ``np.unique(entity_sorted, return_index=True)[1]`` but
    preserves first-appearance order (``np.unique`` returns sorted-by-value
    indices, which is the same here since ``entity_sorted`` is sorted ascending).
    """
    if entity_sorted.shape[0] == 0:
        return np.empty(0, dtype=np.intp)
    # First index of each new entity value.
    change_points = np.flatnonzero(np.diff(entity_sorted) != 0) + 1
    return np.concatenate(([0], change_points))


# ----------------------------------------------------------------------
# Backward-compat alias
# ----------------------------------------------------------------------

# The legacy class name was ``_ViewsDatasetDarts`` (private, with underscore
# prefix). Tests and downstream code import it by that name. Re-export under
# the old name to preserve the import surface without exposing the legacy
# implementation.
_ViewsDatasetDarts = ViewsDatasetDarts
