"""Tests for per-sequence Zarr cleanup in the PredictionFrame builder.

Covers:
1. Successful per-sequence conversion deletes Zarr after verified memmap.
2. Failed conversion/readback keeps Zarr and raises.
3. Multi-sequence eval deletes each prior sequence's Zarr before next.
4. Forecast single-sequence path deletes Zarr before return.
5. No regression in existing PredictionFrame formatting helpers.
"""
from __future__ import annotations

import sys
import types
import numpy as np
import pytest
import shutil
import tempfile
from pathlib import Path

# Mock darts so we can import views_r2darts2 without the full dep chain.
_mock = types.ModuleType("darts")
_mock.__path__ = []
_mock_ts = types.ModuleType("darts.timeseries")
_mock_ts.TimeSeries = type("TS", (), {})
_mock.TimeSeries = _mock_ts.TimeSeries
_mock_dp = types.ModuleType("darts.dataprocessing")
_mock_dp.__path__ = []
_mock_dp.Pipeline = type("P", (), {})
_mock_dp.Scaler = type("S", (), {})
_mock_tr = types.ModuleType("darts.dataprocessing.transformers")
_mock_tr.Scaler = type("S", (), {})
_mock_dp.transformers = _mock_tr
_mock_pl = types.ModuleType("darts.dataprocessing.pipeline")
_mock_pl.Pipeline = type("P", (), {})
_mock_dp.pipeline = _mock_pl
sys.modules.setdefault("darts", _mock)
sys.modules.setdefault("darts.timeseries", _mock_ts)
sys.modules.setdefault("darts.dataprocessing", _mock_dp)
sys.modules.setdefault("darts.dataprocessing.transformers", _mock_tr)
sys.modules.setdefault("darts.dataprocessing.pipeline", _mock_pl)

from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.transformers.frame_builder import (
    build_prediction_frames_from_dataset,
    PredictionFrameVerificationError,
)
from views_frames import PredictionFrame

TIMES = np.array([528, 529, 530], dtype=np.int64)
ENTITIES = np.array([100, 101, 102], dtype=np.int64)
S = 4


def _build_filled_dataset():
    """Build a fully-populated pgm prediction dataset via the builder."""
    builder = ViewsDataset.builder(
        loa="pgm",
        times=TIMES,
        entities=ENTITIES,
        variables={"pred_ged_sb": "num3"},
        sample_size=S,
        targets=["pred_ged_sb"],
    )
    for t in TIMES:
        vals = np.stack(
            [np.full(S, float(t * 1000 + g), dtype=np.float32) for g in ENTITIES]
        )
        builder.write_batch(
            times=np.full(len(ENTITIES), t),
            entities=ENTITIES,
            columns={"pred_ged_sb": vals},
        )
    return builder, builder.build()


def _build_multi_target_dataset():
    """Build a dataset with multiple pred_ variables."""
    builder = ViewsDataset.builder(
        loa="pgm",
        times=TIMES,
        entities=ENTITIES,
        variables={"pred_a": "num3", "pred_b": "num3"},
        sample_size=S,
        targets=["pred_a", "pred_b"],
    )
    for t in TIMES:
        base = np.stack(
            [np.full(S, float(t * 1000 + g), dtype=np.float32) for g in ENTITIES]
        )
        builder.write_batch(
            times=np.full(len(ENTITIES), t),
            entities=ENTITIES,
            columns={"pred_a": base, "pred_b": base + 1.0},
        )
    return builder, builder.build()


# ============================ Test 1: successful cleanup =====================

def test_successful_conversion_deletes_zarr(tmp_path):
    """After verified PredictionFrame, the Zarr store is deleted."""
    builder, ds = _build_filled_dataset()
    zarr_path = ds._store.path  # capture before close
    assert zarr_path.exists()

    frames_dir = tmp_path / "frames"
    frames = build_prediction_frames_from_dataset(
        ds, ["ged_sb"], frames_dir, zarr_cleanup=True,
    )

    # PredictionFrame is valid.
    assert set(frames.keys()) == {"ged_sb"}
    frame = frames["ged_sb"]
    assert isinstance(frame, PredictionFrame)
    assert frame.values.shape == (len(TIMES) * len(ENTITIES), S)
    assert isinstance(frame.values, np.memmap)

    # Zarr store is deleted.
    assert not zarr_path.exists(), f"Zarr store still exists at {zarr_path}"

    # Memmap files still exist.
    assert (frames_dir / "ged_sb.values.npy").exists()


# ============================ Test 2: failed conversion keeps Zarr ===========

def test_failed_readback_keeps_zarr_and_raises(tmp_path):
    """If verification fails, Zarr is kept and exception is raised.

    We simulate failure by writing a Zarr store with NaN-filled data
    (which the readback check should catch).
    """
    builder = ViewsDataset.builder(
        loa="pgm", times=TIMES, entities=ENTITIES,
        variables={"pred_ged_sb": "num3"}, sample_size=S,
        targets=["pred_ged_sb"],
    )
    # Write NaN values — the readback check should catch this.
    for t in TIMES:
        vals = np.full((len(ENTITIES), S), np.nan, dtype=np.float32)
        builder.write_batch(
            times=np.full(len(ENTITIES), t),
            entities=ENTITIES,
            columns={"pred_ged_sb": vals},
        )
    ds = builder.build()
    zarr_path = ds._store.path

    frames_dir = tmp_path / "frames"
    with pytest.raises(PredictionFrameVerificationError, match="NaN"):
        build_prediction_frames_from_dataset(
            ds, ["ged_sb"], frames_dir, zarr_cleanup=True,
        )

    # Zarr store is NOT deleted on failure.
    assert zarr_path.exists(), "Zarr store should be kept on verification failure"
    ds.close()


# ============================ Test 3: multi-sequence eval ====================

def test_multi_sequence_eval_deletes_each_zarr(tmp_path):
    """Simulate eval: each sequence builds its own Zarr, converts to
    PredictionFrame, and deletes the Zarr before the next sequence starts.
    """
    for seq in range(3):
        builder, ds = _build_filled_dataset()
        zarr_path = ds._store.path
        assert zarr_path.exists()

        frames_dir = tmp_path / f"frames_seq_{seq}"
        frames = build_prediction_frames_from_dataset(
            ds, ["ged_sb"], frames_dir, zarr_cleanup=True,
        )

        # Zarr deleted after each sequence.
        assert not zarr_path.exists(), (
            f"Sequence {seq}: Zarr store should be deleted"
        )
        # Memmap files persist.
        assert (frames_dir / "ged_sb.values.npy").exists()
        # Frame is valid.
        assert frames["ged_sb"].values.shape == (len(TIMES) * len(ENTITIES), S)


# ============================ Test 4: forecast single-sequence ==============

def test_forecast_single_sequence_deletes_zarr(tmp_path):
    """Forecast path: single sequence, Zarr deleted before return."""
    builder, ds = _build_filled_dataset()
    zarr_path = ds._store.path
    assert zarr_path.exists()

    frames_dir = tmp_path / "frames"
    frames = build_prediction_frames_from_dataset(
        ds, ["ged_sb"], frames_dir, zarr_cleanup=True,
    )

    # Zarr deleted before return.
    assert not zarr_path.exists()
    # Frame is valid.
    assert frames["ged_sb"].values.shape == (len(TIMES) * len(ENTITIES), S)


# ============================ Test 5: no regression =========================

def test_values_are_correct_after_zarr_cleanup(tmp_path):
    """PredictionFrame values are correct after Zarr is deleted."""
    builder, ds = _build_filled_dataset()
    frames_dir = tmp_path / "frames"
    frames = build_prediction_frames_from_dataset(
        ds, ["ged_sb"], frames_dir, zarr_cleanup=True,
    )

    frame = frames["ged_sb"]
    E = len(ENTITIES)
    for r in range(len(TIMES) * E):
        t = TIMES[r // E]
        g = ENTITIES[r % E]
        assert np.all(frame.values[r] == float(t * 1000 + g)), (
            f"row {r} (t={t}, g={g}): expected {t*1000+g}, got {frame.values[r]}"
        )


def test_index_is_time_major_after_zarr_cleanup(tmp_path):
    """Index is time-major after Zarr cleanup."""
    builder, ds = _build_filled_dataset()
    frames_dir = tmp_path / "frames"
    frames = build_prediction_frames_from_dataset(
        ds, ["ged_sb"], frames_dir, zarr_cleanup=True,
    )

    frame = frames["ged_sb"]
    E = len(ENTITIES)
    assert np.array_equal(frame.index.time[:E], np.full(E, TIMES[0]))
    assert np.array_equal(frame.index.unit[:E], ENTITIES)
    assert np.array_equal(frame.index.time[-E:], np.full(E, TIMES[-1]))


def test_multiple_targets_after_zarr_cleanup(tmp_path):
    """Multiple targets work correctly after Zarr cleanup."""
    builder, ds = _build_multi_target_dataset()
    zarr_path = ds._store.path
    frames_dir = tmp_path / "frames"
    frames = build_prediction_frames_from_dataset(
        ds, ["a", "b"], frames_dir, zarr_cleanup=True,
    )

    assert set(frames.keys()) == {"a", "b"}
    assert np.array_equal(frames["b"].values, frames["a"].values + 1.0)
    assert not zarr_path.exists()


def test_missing_variable_raises(tmp_path):
    """Missing variable raises KeyError, not PredictionFrameVerificationError."""
    builder, ds = _build_filled_dataset()
    frames_dir = tmp_path / "frames"
    with pytest.raises(KeyError, match="pred_missing"):
        build_prediction_frames_from_dataset(
            ds, ["missing"], frames_dir, zarr_cleanup=True,
        )
    ds.close()


# ============================ Test 6: zarr_cleanup=False ====================

def test_zarr_cleanup_false_keeps_zarr(tmp_path):
    """When zarr_cleanup=False, Zarr store is kept."""
    builder, ds = _build_filled_dataset()
    zarr_path = ds._store.path
    assert zarr_path.exists()

    frames_dir = tmp_path / "frames"
    frames = build_prediction_frames_from_dataset(
        ds, ["ged_sb"], frames_dir, zarr_cleanup=False,
    )

    # Zarr store still exists.
    assert zarr_path.exists()
    # Frame is still valid.
    assert frames["ged_sb"].values.shape == (len(TIMES) * len(ENTITIES), S)
    ds.close()
