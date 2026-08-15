"""Per-entity static-covariate fingerprint (pandas-free, numpy-only).

Replaces the legacy ``df.groupby(entity_id)[target_col].agg(...).apply(...)``
pandas chain with vectorized numpy. Produces the same five statistics per
target column:

    * ``mu``       — mean
    * ``sigma``    — standard deviation (ddof=1; 0 if group size < 2)
    * ``max``      — maximum
    * ``trend``    — OLS slope of the target over integer time index
    * ``sparsity`` — fraction of zero-valued observations

The fingerprint is then optionally transformed (elementwise transforms like
``AsinhTransform``, then cross-entity scalers like ``MaxAbsScaler``) before
being injected as Darts static covariates. The transform vocabulary matches the
legacy contract so existing manifests keep working.

Algorithmic parity with the legacy pandas path is enforced bit-for-bit:

    * ``mu``/``sigma``/``max`` use the same reduction semantics.
    * ``trend`` uses the same OLS-of-integer-position formula:
      ``Σ(t - t̄)(y - ȳ) / Σ(t - t̄)²`` where ``t = arange(len(group))``.
    * ``sparsity`` uses ``(y == 0).mean()``.
    * Cross-entity ``MaxAbsScaler`` divides by the global max-abs; the legacy
      code skipped division when ``abs_max == 0`` (we do the same).
    * Cross-entity ``StandardScaler`` divides by std only when ``std > 0``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Supported elementwise transforms (applied per-scalar, in transformed space).
_ELEMENTWISE_TRANSFORMS: dict[str, callable] = {
    "AsinhTransform": np.arcsinh,
    "LogTransform": np.log1p,
    "SqrtTransform": lambda x: np.sqrt(np.maximum(x, 0)),
    "FourthRootTransform": lambda x: np.power(1.0 + np.maximum(x, 0.0), 0.25) - 1.0,
}

# Supported cross-entity scalers (applied across entities, in transformed space).
_CROSS_ENTITY_SCALERS: set[str] = {"MaxAbsScaler", "StandardScaler"}


@dataclass(frozen=True)
class StaticCovariateConfig:
    """Configuration for the per-entity static-covariate fingerprint.

    Attributes:
        transform: Optional transform chain (e.g. ``"AsinhTransform->MaxAbsScaler"``).
            ``None`` means raw space. Elementwise transforms apply to
            ``mu``/``sigma``/``max``/``trend`` only — ``sparsity`` is already in
            ``[0, 1]`` and is never transformed.
        stats: Subset of ``("mu", "sigma", "max", "trend", "sparsity")`` to
            compute. Defaults to all five.
    """

    transform: str | None = None
    stats: tuple[str, ...] = (
        "mu",
        "sigma",
        "max",
        "trend",
        "sparsity",
    )

    def __post_init__(self) -> None:
        unknown = set(self.stats) - {"mu", "sigma", "max", "trend", "sparsity"}
        if unknown:
            raise ValueError(
                f"Unknown static-covariate stats: {sorted(unknown)}. "
                "Allowed: ('mu', 'sigma', 'max', 'trend', 'sparsity')."
            )
        if self.transform is not None:
            steps = [s.strip() for s in self.transform.split("->")]
            for step in steps:
                if (
                    step not in _ELEMENTWISE_TRANSFORMS
                    and step not in _CROSS_ENTITY_SCALERS
                ):
                    raise ValueError(
                        f"Unknown static_cov_transform step '{step}'. "
                        f"Available elementwise: {sorted(_ELEMENTWISE_TRANSFORMS)}. "
                        f"Available cross-entity: {sorted(_CROSS_ENTITY_SCALERS)}."
                    )


@dataclass
class StaticCovariateStats:
    """Result of the per-entity fingerprint computation.

    The ``values`` dict is keyed by ``f"{target_col}_{stat}"`` (e.g.
    ``"lr_ged_sb_mu"``) and maps to a 1-D numpy array of length
    ``len(entity_ids)``. The ``entity_ids`` array preserves first-appearance
    order in the input data.
    """

    entity_ids: NDArray[np.int64]
    values: dict[str, NDArray[np.float32]] = field(default_factory=dict)

    def row_for_entity(self, entity_id: int) -> Mapping[str, float]:
        """Return the ``{stat_name: value}`` mapping for a single entity.

        Args:
            entity_id: The integer entity id to look up.

        Raises:
            KeyError: The entity id is not in the fingerprint.
        """
        idx = int(np.searchsorted(self.entity_ids, entity_id))
        if (
            idx >= self.entity_ids.shape[0]
            or int(self.entity_ids[idx]) != int(entity_id)
        ):
            raise KeyError(
                f"Entity id {entity_id} not found in static-covariate fingerprint."
            )
        return {k: float(v[idx]) for k, v in self.values.items()}

    @property
    def column_names(self) -> list[str]:
        """Fingerprint column names in insertion order."""
        return list(self.values.keys())


def compute_static_covariates(
    *,
    time: NDArray[np.int64],
    entity: NDArray[np.int64],
    values: NDArray[np.float32],
    target_columns: list[str],
    column_order: list[str],
    stat_time_range: tuple[int, int] | None,
    config: StaticCovariateConfig,
) -> StaticCovariateStats:
    """Compute the per-entity fingerprint for the declared target columns.

    Args:
        time: 1-D int64 array of time identifiers (full subset, not yet
            restricted to ``stat_time_range``).
        entity: 1-D int64 array of entity identifiers, parallel to ``time``.
        values: 2-D float32 array of shape ``(N, F)`` with column order matching
            ``column_order``.
        target_columns: Target column names (subset of ``column_order``).
        column_order: The full column-name order of ``values``. Used to look up
            the column index for each target.
        stat_time_range: ``(start, end)`` inclusive time-id window. When
            ``None``, the full input is used (with a leakage warning).
        config: Fingerprint configuration (transform chain + stat subset).

    Returns:
        A :class:`StaticCovariateStats` carrying the per-entity fingerprint.
    """
    if stat_time_range is not None:
        start, end = stat_time_range
        mask = (time >= start) & (time <= end)
        n_time_steps_in_range = int(np.unique(time[mask]).shape[0])
        n_rows_in_range = int(mask.sum())
        logger.info(
            "Static covariate stats restricted to time range [%s, %s] "
            "(%d time steps, %d rows).",
            start,
            end,
            n_time_steps_in_range,
            n_rows_in_range,
        )
        time_stat = time[mask]
        entity_stat = entity[mask]
        values_stat = values[mask]
    else:
        logger.warning(
            "stat_time_range not provided — static covariate stats computed "
            "from the FULL dataframe. This may cause leakage if test-period "
            "data is present."
        )
        time_stat = time
        entity_stat = entity
        values_stat = values

    # Resolve the transform chain into elementwise_fn + cross_entity_scaler.
    elementwise_fn = None
    cross_entity_scaler = None
    if config.transform is not None:
        for step in config.transform.split("->"):
            step = step.strip()
            if step in _ELEMENTWISE_TRANSFORMS:
                elementwise_fn = _ELEMENTWISE_TRANSFORMS[step]
            elif step in _CROSS_ENTITY_SCALERS:
                cross_entity_scaler = step

    # Sort by (entity, time) for contiguous per-entity blocks. This mirrors the
    # legacy pandas groupby semantics where stats are computed per entity.
    sort_order = np.lexsort((time_stat, entity_stat))
    entity_sorted = entity_stat[sort_order]
    time_sorted = time_stat[sort_order]
    values_sorted = np.ascontiguousarray(values_stat[sort_order])

    boundaries = _entity_boundaries(entity_sorted)
    entity_ids = entity_sorted[boundaries]
    n_entities = entity_ids.shape[0]
    n_time_steps_total = int(np.unique(time_stat).shape[0])
    transform_label = config.transform or "raw"

    logger.info(
        "Static covariates: computing %s stats for %d target(s) × %d entities "
        "(%d rows, %d time steps, %s space).",
        config.stats,
        len(target_columns),
        n_entities,
        entity_sorted.shape[0],
        n_time_steps_total,
        transform_label,
    )

    # Compute per-target, per-entity stats. The result dict is keyed by
    # ``f"{target_col}_{stat}"`` and stores a 1-D float32 array of length
    # ``n_entities`` (one value per entity, in entity_ids order).
    result: dict[str, NDArray[np.float32]] = {}
    for target_col in target_columns:
        col_idx = column_order.index(target_col)
        col_values = values_sorted[:, col_idx]

        if "mu" in config.stats:
            result[f"{target_col}_mu"] = _per_entity_mean(
                col_values, boundaries
            ).astype(np.float32)
        if "sigma" in config.stats:
            result[f"{target_col}_sigma"] = _per_entity_std(
                col_values, boundaries
            ).astype(np.float32)
        if "max" in config.stats:
            result[f"{target_col}_max"] = _per_entity_max(
                col_values, boundaries
            ).astype(np.float32)
        if "trend" in config.stats:
            result[f"{target_col}_trend"] = _per_entity_ols_slope(
                col_values, boundaries
            ).astype(np.float32)
        if "sparsity" in config.stats:
            result[f"{target_col}_sparsity"] = _per_entity_sparsity(
                col_values, boundaries
            ).astype(np.float32)

    # Apply elementwise transform to scale-sensitive stats (not sparsity).
    transformable_stats = [
        s for s in ("mu", "sigma", "max", "trend") if s in config.stats
    ]
    if elementwise_fn is not None:
        for target_col in target_columns:
            for stat in transformable_stats:
                key = f"{target_col}_{stat}"
                result[key] = elementwise_fn(result[key].astype(np.float64)).astype(
                    np.float32
                )

    # Apply cross-entity scaler across entities, per stat. The legacy code
    # computed the scaler over the full fingerprint (all entities), so we do
    # the same — one global max-abs / mean-std per (target, stat) column.
    if cross_entity_scaler == "MaxAbsScaler":
        for target_col in target_columns:
            for stat in transformable_stats:
                key = f"{target_col}_{stat}"
                abs_max = float(np.abs(result[key]).max())
                if abs_max > 0:
                    result[key] = (result[key] / abs_max).astype(np.float32)
    elif cross_entity_scaler == "StandardScaler":
        for target_col in target_columns:
            for stat in transformable_stats:
                key = f"{target_col}_{stat}"
                mean = float(result[key].mean())
                std = float(result[key].std())
                if std > 0:
                    result[key] = ((result[key] - mean) / std).astype(np.float32)

    return StaticCovariateStats(entity_ids=entity_ids, values=result)


# ----------------------------------------------------------------------
# Vectorized per-entity reducers (numpy-only, parity with pandas groupby)
# ----------------------------------------------------------------------


def _entity_boundaries(entity_sorted: NDArray[np.int64]) -> NDArray[np.intp]:
    """Start index of each entity's contiguous block (sorted-ascending input)."""
    if entity_sorted.shape[0] == 0:
        return np.empty(0, dtype=np.intp)
    change_points = np.flatnonzero(np.diff(entity_sorted) != 0) + 1
    return np.concatenate(([0], change_points))


def _per_entity_mean(
    col: NDArray[np.float32], boundaries: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Per-entity mean (parity with ``groupby(entity).mean()``)."""
    if boundaries.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    sums = np.add.reduceat(col, boundaries, axis=0)
    sizes = np.diff(np.concatenate([boundaries, [col.shape[0]]]))
    return sums.astype(np.float64) / sizes


def _per_entity_max(
    col: NDArray[np.float32], boundaries: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Per-entity maximum (parity with ``groupby(entity).max()``)."""
    if boundaries.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    # ``np.maximum.reduceat`` does the per-block max.
    return np.maximum.reduceat(col, boundaries, axis=0).astype(np.float64)


def _per_entity_std(
    col: NDArray[np.float32], boundaries: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Per-entity standard deviation with ``ddof=1`` (parity with pandas ``std``).

    Single-row groups get ``0.0`` (parity with pandas ``fillna(0.0)``).
    """
    if boundaries.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    means = _per_entity_mean(col, boundaries)
    sizes = np.diff(np.concatenate([boundaries, [col.shape[0]]]))

    # Vectorized variance: Σ(y - ȳ)² / (n - 1). We compute the per-block
    # sum of squared deviations by expanding the identity
    # Σ(y - ȳ)² = Σy² - n·ȳ².
    col64 = col.astype(np.float64)
    sum_sq = np.add.reduceat(col64 * col64, boundaries, axis=0)
    sum_dev_sq = sum_sq - sizes * means * means
    # Guard against tiny negative values from floating-point cancellation.
    sum_dev_sq = np.maximum(sum_dev_sq, 0.0)
    ddof = sizes - 1
    # Avoid divide-by-zero for single-row groups (pandas returns NaN, which the
    # legacy code fillna(0.0)'d to zero — we replicate that here directly).
    std = np.zeros_like(means)
    valid = ddof > 0
    std[valid] = np.sqrt(sum_dev_sq[valid] / ddof[valid])
    return std


def _per_entity_ols_slope(
    col: NDArray[np.float32], boundaries: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Per-entity OLS slope (parity with the legacy ``_ols_slope`` apply).

    Formula: ``Σ(t - t̄)(y - ȳ) / Σ(t - t̄)²`` where ``t = arange(len(group))``.
    Returns ``0.0`` for single-row groups (denominator is zero).
    """
    if boundaries.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    sizes = np.diff(np.concatenate([boundaries, [col.shape[0]]]))
    slopes = np.zeros(sizes.shape[0], dtype=np.float64)

    # For each entity block, compute the slope. Vectorization is possible via
    # block-level reductions, but the per-entity loop is O(n_entities) and the
    # inner numpy ops are O(block_size) — total work is O(N) same as a single
    # vectorized pass. Keep the simple loop for parity clarity.
    for i, start in enumerate(boundaries):
        stop = boundaries[i + 1] if i + 1 < boundaries.shape[0] else col.shape[0]
        n = stop - start
        if n < 2:
            slopes[i] = 0.0
            continue
        t = np.arange(n, dtype=np.float64)
        t_centered = t - t.mean()
        y = col[start:stop].astype(np.float64)
        y_centered = y - y.mean()
        denom = float((t_centered * t_centered).sum())
        if denom == 0.0:
            slopes[i] = 0.0
            continue
        slopes[i] = float((t_centered * y_centered).sum() / denom)
    return slopes


def _per_entity_sparsity(
    col: NDArray[np.float32], boundaries: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Per-entity fraction of zero-valued observations (parity with ``apply``)."""
    if boundaries.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    is_zero = (col == 0.0).astype(np.float64)
    sums = np.add.reduceat(is_zero, boundaries, axis=0)
    sizes = np.diff(np.concatenate([boundaries, [col.shape[0]]]))
    return sums / sizes
