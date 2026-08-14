"""Tests for build_prediction_frames_from_dataset (memmap-backed PredictionFrames)."""
from __future__ import annotations

import numpy as np
import pytest

from views_frames import PredictionFrame
from views_r2darts2.dataset import ViewsDataset
from views_r2darts2.transformers.frame_builder import (
    build_prediction_frames_from_dataset,
)

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


def test_build_frames_returns_memmap_backed_frame(tmp_path):
    builder, ds = _build_filled_dataset()
    try:
        frames = build_prediction_frames_from_dataset(
            ds, ["ged_sb"], tmp_path / "frames"
        )
    finally:
        ds.close()

    assert set(frames.keys()) == {"ged_sb"}
    frame = frames["ged_sb"]
    assert isinstance(frame, PredictionFrame)
    assert isinstance(frame.values, np.memmap), "values must be a memmap"
    assert frame.values.shape == (len(TIMES) * len(ENTITIES), S)


def test_build_frames_values_are_correct(tmp_path):
    builder, ds = _build_filled_dataset()
    try:
        frames = build_prediction_frames_from_dataset(
            ds, ["ged_sb"], tmp_path / "frames"
        )
    finally:
        ds.close()

    frame = frames["ged_sb"]
    E = len(ENTITIES)
    for r in range(len(TIMES) * E):
        t = TIMES[r // E]
        g = ENTITIES[r % E]
        expected = float(t * 1000 + g)
        assert np.all(frame.values[r] == expected), (
            f"row {r} (t={t}, g={g}): expected {expected}, got {frame.values[r]}"
        )


def test_build_frames_index_is_time_major(tmp_path):
    builder, ds = _build_filled_dataset()
    try:
        frames = build_prediction_frames_from_dataset(
            ds, ["ged_sb"], tmp_path / "frames"
        )
    finally:
        ds.close()

    frame = frames["ged_sb"]
    E = len(ENTITIES)
    assert np.array_equal(frame.index.time[:E], np.full(E, TIMES[0]))
    assert np.array_equal(frame.index.unit[:E], ENTITIES)
    assert np.array_equal(frame.index.time[-E:], np.full(E, TIMES[-1]))


def test_build_frames_small_entity_block_matches(tmp_path):
    """A tiny entity_block (forcing many passes) must produce identical values."""
    builder, ds = _build_filled_dataset()
    try:
        frames = build_prediction_frames_from_dataset(
            ds, ["ged_sb"], tmp_path / "frames", entity_block=1
        )
    finally:
        ds.close()

    frame = frames["ged_sb"]
    E = len(ENTITIES)
    for r in range(len(TIMES) * E):
        t = TIMES[r // E]
        g = ENTITIES[r % E]
        assert np.all(frame.values[r] == float(t * 1000 + g))


def test_build_frames_missing_variable_raises(tmp_path):
    builder, ds = _build_filled_dataset()
    try:
        with pytest.raises(KeyError, match="pred_missing"):
            build_prediction_frames_from_dataset(
                ds, ["missing"], tmp_path / "frames"
            )
    finally:
        ds.close()


def test_build_frames_multiple_targets(tmp_path):
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
    ds = builder.build()
    try:
        frames = build_prediction_frames_from_dataset(
            ds, ["a", "b"], tmp_path / "frames"
        )
    finally:
        ds.close()

    assert set(frames.keys()) == {"a", "b"}
    assert np.array_equal(frames["b"].values, frames["a"].values + 1.0)
