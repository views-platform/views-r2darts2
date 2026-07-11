"""Tests for :mod:`views_r2darts2.transformers.darts_bridge`.

Verifies the two public functions:

    * :func:`build_entity_timeseries` — numpy → Darts ``TimeSeries`` builder.
      Includes the regression for single-row entities (entity 248).
    * :func:`prediction_frame_from_darts` — Darts ``TimeSeries`` →
      :class:`views_frames.PredictionFrame` converter (single-target and
      multi-target variants, plus clip-negatives and error paths).

Google Python Style. ``pandas`` is used only at the Darts boundary (mirroring
the production ``darts_bridge`` module).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

from views_frames import PredictionFrame, SpatialLevel
from views_r2darts2.transformers.darts_bridge import (
    build_entity_timeseries,
    prediction_frame_from_darts,
    prediction_frames_from_darts,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_prediction_series(
    *,
    entity_id: int = 1,
    time_ids: Sequence[int] = (1, 2, 3),
    target_columns: Sequence[str] = ("y",),
    values: np.ndarray | None = None,
    n_samples: int = 1,
    entity_id_name: str = "country_id",
    extra_static_covs: dict[str, float] | None = None,
    omit_entity_id: bool = False,
) -> TimeSeries:
    """Build a Darts prediction :class:`TimeSeries` for tests.

    Args:
        entity_id: Entity id to attach as a static covariate.
        time_ids: Time identifiers (must be ascending).
        target_columns: Component (column) names.
        values: Optional values array of shape ``(T, F)`` or ``(T, F, S)``.
            When ``None``, deterministic values are synthesized.
        n_samples: When ``values`` is None and ``n_samples > 1``, build a 3-D
            probabilistic series with this many samples.
        entity_id_name: Static-covariate column name for the entity id.
        extra_static_covs: Additional static-covariate columns.
        omit_entity_id: When ``True``, do NOT include ``entity_id_name`` in
            the static covariates (used for the missing-entity-id test).

    Returns:
        A Darts :class:`TimeSeries`.
    """
    time_arr = np.asarray(time_ids, dtype=np.int64)
    n_time = time_arr.shape[0]
    n_feat = len(target_columns)
    if values is None:
        if n_samples == 1:
            values_arr = np.arange(
                1, n_time * n_feat + 1, dtype=np.float32
            ).reshape(n_time, n_feat)
        else:
            rng = np.random.default_rng(entity_id)
            values_arr = rng.standard_normal(
                (n_time, n_feat, n_samples)
            ).astype(np.float32)
    else:
        values_arr = np.asarray(values, dtype=np.float32)

    cov_dict: dict[str, float] = {}
    if not omit_entity_id:
        cov_dict[entity_id_name] = float(entity_id)
    if extra_static_covs:
        cov_dict.update(extra_static_covs)
    static_df = pd.DataFrame({k: [v] for k, v in cov_dict.items()})

    return TimeSeries.from_times_and_values(
        times=pd.Index(time_arr),
        values=values_arr,
        columns=list(target_columns),
        static_covariates=static_df,
        freq=1,
    )


# ----------------------------------------------------------------------
# build_entity_timeseries
# ----------------------------------------------------------------------


class TestBuildEntityTimeseries:
    """Tests for :func:`build_entity_timeseries`."""

    def test_build_entity_timeseries_basic(self) -> None:
        """Basic 3-timestep, 2-column series carries time/values/entity id."""
        time = np.array([1, 2, 3], dtype=np.int64)
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
        ts = build_entity_timeseries(
            time=time,
            values=values,
            columns=["a", "b"],
            entity_id_name="country_id",
            entity_id_value=42,
        )
        assert isinstance(ts, TimeSeries)
        assert len(ts) == 3
        assert list(ts.components) == ["a", "b"]
        # Time index preserved.
        assert ts.time_index.values.tolist() == [1, 2, 3]
        # Static covariate carries the entity id.
        assert "country_id" in ts.static_covariates.columns
        assert int(ts.static_covariates["country_id"].iloc[0]) == 42
        # Values preserved.
        np.testing.assert_array_equal(
            ts.all_values(copy=False)[:, :, 0], values
        )

    def test_build_entity_timeseries_single_row(self) -> None:
        """Regression: a 1-row entity (e.g. entity 248) builds a valid series."""
        time = np.array([140], dtype=np.int64)
        values = np.array([[7.5, 8.5]], dtype=np.float32)
        ts = build_entity_timeseries(
            time=time,
            values=values,
            columns=["a", "b"],
            entity_id_name="country_id",
            entity_id_value=248,
        )
        assert isinstance(ts, TimeSeries)
        assert len(ts) == 1
        assert list(ts.components) == ["a", "b"]
        assert int(ts.time_index.values[0]) == 140
        assert int(ts.static_covariates["country_id"].iloc[0]) == 248
        np.testing.assert_array_equal(
            ts.all_values(copy=False)[:, :, 0], values
        )

    def test_build_entity_timeseries_with_static_covariates(self) -> None:
        """Per-entity fingerprint is attached alongside the entity id."""
        time = np.array([1, 2], dtype=np.int64)
        values = np.array([[1.0], [2.0]], dtype=np.float32)
        ts = build_entity_timeseries(
            time=time,
            values=values,
            columns=["y"],
            entity_id_name="country_id",
            entity_id_value=7,
            static_covariates={"y_mu": 1.5, "y_sparsity": 0.0},
        )
        cols = set(ts.static_covariates.columns)
        assert "country_id" in cols
        assert "y_mu" in cols
        assert "y_sparsity" in cols
        assert float(ts.static_covariates["y_mu"].iloc[0]) == 1.5
        assert float(ts.static_covariates["y_sparsity"].iloc[0]) == 0.0


# ----------------------------------------------------------------------
# prediction_frame_from_darts (single-target)
# ----------------------------------------------------------------------


class TestPredictionFrameFromDartsSingle:
    """Single-target :func:`prediction_frame_from_darts` tests."""

    def test_prediction_frame_from_darts_single_target(self) -> None:
        """Single-target → one :class:`PredictionFrame` of correct shape."""
        preds = [
            _make_prediction_series(entity_id=1, target_columns=["y"]),
            _make_prediction_series(entity_id=2, target_columns=["y"]),
            _make_prediction_series(entity_id=3, target_columns=["y"]),
        ]
        frame = prediction_frame_from_darts(
            predictions=preds,
            entity_id_name="country_id",
            target_columns=["y"],
            level=SpatialLevel.CM,
            clip_negatives=False,
        )
        assert isinstance(frame, PredictionFrame)
        # 3 entities × 3 timesteps = 9 rows.
        assert frame.n_rows == 9
        # Single sample.
        assert frame.sample_count == 1
        # The index level is CM.
        assert frame.index.level == SpatialLevel.CM
        # Entity ids preserved.
        unique_entities = set(np.unique(frame.index.unit).tolist())
        assert unique_entities == {1, 2, 3}
        # Time ids preserved.
        unique_times = set(np.unique(frame.index.time).tolist())
        assert unique_times == {1, 2, 3}


# ----------------------------------------------------------------------
# prediction_frames_from_darts (multi-target)
# ----------------------------------------------------------------------


class TestPredictionFramesFromDartsMulti:
    """Multi-target :func:`prediction_frames_from_darts` tests."""

    def test_prediction_frames_from_darts_multi_target(self) -> None:
        """3 targets → dict of 3 frames, one per target."""
        target_cols = ["y1", "y2", "y3"]
        # Build two prediction series, each carrying all 3 targets.
        preds = [
            _make_prediction_series(
                entity_id=1, target_columns=target_cols
            ),
            _make_prediction_series(
                entity_id=2, target_columns=target_cols
            ),
        ]
        frames = prediction_frames_from_darts(
            predictions=preds,
            entity_id_name="country_id",
            target_columns=target_cols,
            level=SpatialLevel.CM,
            clip_negatives=False,
        )
        assert isinstance(frames, dict)
        assert set(frames.keys()) == set(target_cols)
        # Each frame has 2 entities × 3 timesteps = 6 rows.
        for tgt, frame in frames.items():
            assert isinstance(frame, PredictionFrame)
            assert frame.n_rows == 6
            assert frame.sample_count == 1

    def test_prediction_frames_from_darts_clip_negatives(self) -> None:
        """``clip_negatives=True`` floors all values to 0."""
        target_cols = ["y"]
        # Synthesize values with negatives.
        time_ids = (1, 2, 3)
        values = np.array([[-1.0], [-2.0], [3.0]], dtype=np.float32)
        preds = [
            _make_prediction_series(
                entity_id=1,
                time_ids=time_ids,
                target_columns=target_cols,
                values=values,
            )
        ]
        frames = prediction_frames_from_darts(
            predictions=preds,
            entity_id_name="country_id",
            target_columns=target_cols,
            level=SpatialLevel.CM,
            clip_negatives=True,
        )
        arr = frames["y"].values
        assert np.all(arr >= 0.0)
        # The third value (3.0) is preserved; the first two (-1, -2) → 0.
        assert arr[2, 0] == 3.0
        assert arr[0, 0] == 0.0
        assert arr[1, 0] == 0.0


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


class TestPredictionFrameFromDartsErrors:
    """Error-path tests for the prediction-frame converters."""

    def test_prediction_frame_from_darts_empty_raises(self) -> None:
        """An empty prediction list raises ``ValueError``."""
        with pytest.raises(ValueError, match="empty"):
            prediction_frame_from_darts(
                predictions=[],
                entity_id_name="country_id",
                target_columns=["y"],
                level=SpatialLevel.CM,
            )

    def test_prediction_frame_from_darts_missing_entity_id_raises(
        self,
    ) -> None:
        """A series without the entity-id static covariate raises."""
        preds = [
            _make_prediction_series(
                entity_id=1,
                target_columns=["y"],
                omit_entity_id=True,
            )
        ]
        with pytest.raises(ValueError, match="missing the 'country_id'"):
            prediction_frame_from_darts(
                predictions=preds,
                entity_id_name="country_id",
                target_columns=["y"],
                level=SpatialLevel.CM,
            )

    def test_prediction_frame_from_darts_missing_target_component_raises(
        self,
    ) -> None:
        """A series missing the requested target component raises."""
        # Build a series with component "y" but request target "z".
        preds = [
            _make_prediction_series(
                entity_id=1, target_columns=["y"]
            )
        ]
        with pytest.raises(ValueError, match="missing target component 'z'"):
            prediction_frame_from_darts(
                predictions=preds,
                entity_id_name="country_id",
                target_columns=["z"],
                level=SpatialLevel.CM,
            )

    def test_prediction_frames_from_darts_multi_target_raises_on_missing(
        self,
    ) -> None:
        """Multi-target path: missing target component raises ``ValueError``."""
        # Build a series with only "y1" but request ["y1", "y2"].
        preds = [
            _make_prediction_series(
                entity_id=1, target_columns=["y1"]
            )
        ]
        with pytest.raises(ValueError, match="missing target component 'y2'"):
            prediction_frames_from_darts(
                predictions=preds,
                entity_id_name="country_id",
                target_columns=["y1", "y2"],
                level=SpatialLevel.CM,
            )

    def test_prediction_frames_from_darts_empty_raises(self) -> None:
        """Multi-target path: an empty prediction list raises."""
        with pytest.raises(ValueError, match="empty"):
            prediction_frames_from_darts(
                predictions=[],
                entity_id_name="country_id",
                target_columns=["y"],
                level=SpatialLevel.CM,
            )


# ----------------------------------------------------------------------
# Probabilistic sample preservation
# ----------------------------------------------------------------------


class TestPredictionFrameProbabilistic:
    """Probabilistic (3-D) prediction-frame tests."""

    def test_prediction_frames_from_darts_probabilistic(self) -> None:
        """3-D ``(T, F, S)`` predictions preserve the sample axis in the frame."""
        n_samples = 4
        preds = [
            _make_prediction_series(
                entity_id=1,
                target_columns=["y"],
                n_samples=n_samples,
            ),
            _make_prediction_series(
                entity_id=2,
                target_columns=["y"],
                n_samples=n_samples,
            ),
        ]
        frames = prediction_frames_from_darts(
            predictions=preds,
            entity_id_name="country_id",
            target_columns=["y"],
            level=SpatialLevel.CM,
            clip_negatives=False,
        )
        frame = frames["y"]
        # 2 entities × 3 timesteps = 6 rows × 4 samples.
        assert frame.values.shape == (6, n_samples)
        assert frame.sample_count == n_samples
