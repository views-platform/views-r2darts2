"""Tests for :mod:`views_r2darts2.transformers.static_covariates`.

Verifies the per-entity static-covariate fingerprint (``mu``, ``sigma``,
``max``, ``trend``, ``sparsity``), the optional transform chain
(``AsinhTransform``/``MaxAbsScaler``/``StandardScaler``), and bit-for-bit
parity with the equivalent pandas ``groupby`` path.

``pandas`` is imported ONLY in
``test_parity_with_pandas_groupby`` as the reference oracle — the production
module is pandas-free.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pytest

from views_r2darts2.transformers.static_covariates import (
    StaticCovariateConfig,
    StaticCovariateStats,
    compute_static_covariates,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_known_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build a small hand-verified dataset for ``test_basic_stats``.

    Two entities, three months each. The first target column ``y`` has
    well-defined mu/sigma/max/trend/sparsity for entity 0:

        Entity 0: y = [0.0, 1.0, 2.0]
            mu       = 1.0
            sigma    = 1.0  (ddof=1)
            max      = 2.0
            trend    = 1.0  (slope of [0,1,2] over t=[0,1,2])
            sparsity = 1/3  (one zero out of three)

        Entity 1: y = [3.0, 3.0, 3.0]
            mu       = 3.0
            sigma    = 0.0  (constant series, ddof=1 → 0 after fillna)
            max      = 3.0
            trend    = 0.0
            sparsity = 0/3  = 0.0

    Returns:
        ``(time, entity, values_2d, column_order)`` where:
            * time: int64 array of length 6.
            * entity: int64 array of length 6.
            * values_2d: float32 array of shape (6, 1).
            * column_order: list of column names.
    """
    time = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
    entity = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    values = np.array(
        [[0.0], [1.0], [2.0], [3.0], [3.0], [3.0]], dtype=np.float32
    )
    column_order = ["y"]
    return time, entity, values, column_order


def _compute(
    *,
    time: np.ndarray,
    entity: np.ndarray,
    values: np.ndarray,
    column_order: list[str],
    target_columns: list[str],
    stat_time_range: tuple[int, int] | None = None,
    transform: str | None = None,
    stats: tuple[str, ...] = ("mu", "sigma", "max", "trend", "sparsity"),
) -> StaticCovariateStats:
    """Thin wrapper around :func:`compute_static_covariates` for tests."""
    cfg = StaticCovariateConfig(transform=transform, stats=stats)
    return compute_static_covariates(
        time=time,
        entity=entity,
        values=values,
        target_columns=target_columns,
        column_order=column_order,
        stat_time_range=stat_time_range,
        config=cfg,
    )


# ----------------------------------------------------------------------
# Basic stats
# ----------------------------------------------------------------------


class TestStaticCovariatesBasic:
    """Tests for the five-stat fingerprint computation."""

    def test_basic_stats(self) -> None:
        """Hand-verify mu/sigma/max/trend/sparsity for the known dataset."""
        time, entity, values, columns = _build_known_dataset()
        stats = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=columns,
            target_columns=["y"],
        )
        # Two entities.
        assert stats.entity_ids.tolist() == [0, 1]

        # Entity 0: y = [0, 1, 2]
        np.testing.assert_allclose(stats.values["y_mu"][0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_sigma"][0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_max"][0], 2.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_trend"][0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_sparsity"][0], 1.0 / 3.0, rtol=1e-6)

        # Entity 1: y = [3, 3, 3]
        np.testing.assert_allclose(stats.values["y_mu"][1], 3.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_sigma"][1], 0.0, atol=1e-6)
        np.testing.assert_allclose(stats.values["y_max"][1], 3.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_trend"][1], 0.0, atol=1e-6)
        np.testing.assert_allclose(stats.values["y_sparsity"][1], 0.0, atol=1e-6)

    def test_single_row_entity_stats(self) -> None:
        """Single-row entity: sigma=0, trend=0, mu=max=value."""
        time = np.array([1], dtype=np.int64)
        entity = np.array([42], dtype=np.int64)
        values = np.array([[5.0]], dtype=np.float32)
        stats = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=["y"],
            target_columns=["y"],
        )
        assert stats.entity_ids.tolist() == [42]
        np.testing.assert_allclose(stats.values["y_mu"][0], 5.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_sigma"][0], 0.0, atol=1e-6)
        np.testing.assert_allclose(stats.values["y_max"][0], 5.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_trend"][0], 0.0, atol=1e-6)
        np.testing.assert_allclose(stats.values["y_sparsity"][0], 0.0, atol=1e-6)


# ----------------------------------------------------------------------
# Time-range filter
# ----------------------------------------------------------------------


class TestStaticCovariatesTimeRange:
    """``stat_time_range`` filter tests."""

    def test_stat_time_range_filter(self) -> None:
        """Only rows within ``[start, end]`` contribute to the fingerprint."""
        # Build a 4-timestep dataset for one entity.
        time = np.array([1, 2, 3, 4], dtype=np.int64)
        entity = np.array([0, 0, 0, 0], dtype=np.int64)
        values = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

        # Without filter: mu = 2.5.
        stats_all = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=["y"],
            target_columns=["y"],
        )
        np.testing.assert_allclose(stats_all.values["y_mu"][0], 2.5, rtol=1e-6)

        # With filter [1, 2]: mu = 1.5 (only the first two rows).
        stats_filtered = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=["y"],
            target_columns=["y"],
            stat_time_range=(1, 2),
        )
        np.testing.assert_allclose(stats_filtered.values["y_mu"][0], 1.5, rtol=1e-6)
        np.testing.assert_allclose(stats_filtered.values["y_max"][0], 2.0, rtol=1e-6)


# ----------------------------------------------------------------------
# Transform chain
# ----------------------------------------------------------------------


class TestStaticCovariatesTransform:
    """Transform-chain tests."""

    def test_transform_asinh_only(self) -> None:
        """``AsinhTransform`` only: mu = arcsinh(original_mu)."""
        time, entity, values, columns = _build_known_dataset()
        stats = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=columns,
            target_columns=["y"],
            transform="AsinhTransform",
        )
        # Entity 0: original_mu=1.0 → arcsinh(1.0) ≈ 0.8813735870
        np.testing.assert_allclose(
            stats.values["y_mu"][0], np.arcsinh(1.0), rtol=1e-5
        )
        # Entity 1: original_mu=3.0 → arcsinh(3.0) ≈ 1.8184464592
        np.testing.assert_allclose(
            stats.values["y_mu"][1], np.arcsinh(3.0), rtol=1e-5
        )
        # Sparsity is NOT transformed — still 1/3 for entity 0.
        np.testing.assert_allclose(stats.values["y_sparsity"][0], 1.0 / 3.0, rtol=1e-6)

    def test_transform_asinh_maxabs_chain(self) -> None:
        """``AsinhTransform->MaxAbsScaler`` chain: values bounded in [-1, 1]."""
        time, entity, values, columns = _build_known_dataset()
        stats = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=columns,
            target_columns=["y"],
            transform="AsinhTransform->MaxAbsScaler",
        )
        # After MaxAbsScaler, every transformable stat is in [-1, 1].
        for stat in ("mu", "sigma", "max", "trend"):
            col = f"y_{stat}"
            assert np.all(np.abs(stats.values[col]) <= 1.0 + 1e-6), (
                f"{col} outside [-1, 1]: {stats.values[col]}"
            )
        # The maximum absolute value across entities (after asinh) should be 1.0.
        # Entity 1 has mu = 3.0 → arcsinh(3.0) ≈ 1.8184 → after MaxAbs → 1.0.
        np.testing.assert_allclose(stats.values["y_mu"][1], 1.0, rtol=1e-5)

    def test_transform_standard_scaler(self) -> None:
        """``StandardScaler`` cross-entity: mean=0, std=1 per stat."""
        time, entity, values, columns = _build_known_dataset()
        stats = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=columns,
            target_columns=["y"],
            transform="StandardScaler",
        )
        for stat in ("mu", "sigma", "max", "trend"):
            col = f"y_{stat}"
            arr = stats.values[col].astype(np.float64)
            np.testing.assert_allclose(arr.mean(), 0.0, atol=1e-5)
            np.testing.assert_allclose(arr.std(), 1.0, atol=1e-5)


# ----------------------------------------------------------------------
# Stat subset
# ----------------------------------------------------------------------


class TestStaticCovariatesSubset:
    """``stats`` subset tests."""

    def test_stats_subset(self) -> None:
        """Asking for only ``mu`` and ``sparsity`` returns only those keys."""
        time, entity, values, columns = _build_known_dataset()
        stats = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=columns,
            target_columns=["y"],
            stats=("mu", "sparsity"),
        )
        assert set(stats.values.keys()) == {"y_mu", "y_sparsity"}
        np.testing.assert_allclose(stats.values["y_mu"][0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(stats.values["y_sparsity"][0], 1.0 / 3.0, rtol=1e-6)
        # The column_names property reflects the subset.
        assert stats.column_names == ["y_mu", "y_sparsity"]


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


class TestStaticCovariatesErrors:
    """Validation error tests."""

    def test_unknown_stat_raises(self) -> None:
        """An unknown stat name raises ``ValueError``."""
        with pytest.raises(ValueError, match="Unknown static-covariate stats"):
            StaticCovariateConfig(stats=("mu", "bogus"))  # type: ignore[arg-type]

    def test_unknown_transform_step_raises(self) -> None:
        """An unknown transform step raises ``ValueError``."""
        with pytest.raises(ValueError, match="Unknown static_cov_transform step"):
            StaticCovariateConfig(transform="BogusTransform")


# ----------------------------------------------------------------------
# row_for_entity
# ----------------------------------------------------------------------


class TestStaticCovariatesRowForEntity:
    """``StaticCovariateStats.row_for_entity`` tests."""

    def test_row_for_entity(self) -> None:
        """``row_for_entity`` returns the per-entity stat dict."""
        time, entity, values, columns = _build_known_dataset()
        stats = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=columns,
            target_columns=["y"],
        )
        row = stats.row_for_entity(0)
        assert set(row.keys()) == {
            "y_mu", "y_sigma", "y_max", "y_trend", "y_sparsity",
        }
        np.testing.assert_allclose(row["y_mu"], 1.0, rtol=1e-6)
        np.testing.assert_allclose(row["y_max"], 2.0, rtol=1e-6)

    def test_row_for_entity_missing_raises(self) -> None:
        """``row_for_entity`` on an absent id raises ``KeyError``."""
        time, entity, values, columns = _build_known_dataset()
        stats = _compute(
            time=time,
            entity=entity,
            values=values,
            column_order=columns,
            target_columns=["y"],
        )
        with pytest.raises(KeyError, match="not found"):
            stats.row_for_entity(999)


# ----------------------------------------------------------------------
# Parity with pandas groupby (the ONE allowed pandas import in tests)
# ----------------------------------------------------------------------


class TestStaticCovariatesPandasParity:
    """Bit-for-bit parity vs pandas ``groupby`` reductions.

    NOTE: This is the ONE test class in the suite that imports pandas — the
    reference oracle. The production module is pandas-free; pandas is used
    here only to verify the numpy path produces the same float32 values.
    """

    def test_parity_with_pandas_groupby(self) -> None:
        """mu/sigma/max/sparsity must match pandas groupby to float32 precision."""
        # NOTE: pandas is the reference oracle here — the only allowed pandas
        # import in the test suite.
        import pandas as pd  # noqa: WPS433 — allowed oracle import

        rng = np.random.default_rng(42)
        n_entities = 10
        n_time = 30
        # Build a synthetic dataset with mixed zero-inflation and outliers.
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(1, n_time + 1), n_entities)
        # Mix of zeros, small values, and occasional large spikes.
        base = rng.standard_normal(n_entities * n_time).astype(np.float32)
        spikes = (rng.random(n_entities * n_time) < 0.1).astype(np.float32) * 100.0
        zeros = (rng.random(n_entities * n_time) < 0.3).astype(np.float32)
        values = (base * 5.0 + spikes) * (1.0 - zeros)
        values_2d = values[:, np.newaxis]
        column_order = ["y"]

        # Numpy path (production).
        numpy_stats = _compute(
            time=time_ids.astype(np.int64),
            entity=entity_ids.astype(np.int64),
            values=values_2d.astype(np.float32),
            column_order=column_order,
            target_columns=["y"],
        )

        # Pandas oracle: groupby(entity).agg(mean, std, max) + apply for sparsity.
        df = pd.DataFrame(
            {
                "entity": entity_ids,
                "time": time_ids,
                "y": values.astype(np.float64),
            }
        )
        grouped = df.groupby("entity")["y"]
        pandas_mu = grouped.mean().to_numpy()
        pandas_sigma = grouped.std(ddof=1).fillna(0.0).to_numpy()
        pandas_max = grouped.max().to_numpy()
        pandas_sparsity = grouped.apply(
            lambda s: float((s == 0.0).mean())
        ).to_numpy()

        # Order: pandas groupby sorts by entity id ascending; the numpy path
        # preserves first-appearance order. With arange entities, both match.
        np.testing.assert_allclose(
            numpy_stats.values["y_mu"].astype(np.float64),
            pandas_mu,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            numpy_stats.values["y_sigma"].astype(np.float64),
            pandas_sigma,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            numpy_stats.values["y_max"].astype(np.float64),
            pandas_max,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            numpy_stats.values["y_sparsity"].astype(np.float64),
            pandas_sparsity,
            rtol=1e-5,
            atol=1e-5,
        )
