"""Build memmap-backed PredictionFrames from a zarr-backed ViewsDataset.

When streaming predictions (e.g. MC-dropout inference over the full PGM grid),
the prediction tensor is far larger than RAM. ``DatasetBuilder`` already streams
the writes into a Zarr store; this module provides the complementary read path:
it streams the predictions back out of the Zarr store into a row-major
``(N, S)`` memmap file in entity-aligned blocks, and wraps the result in a
:class:`views_frames.PredictionFrame` whose ``values`` is a read-only
``np.memmap``. Peak memory is one entity block — never the full grid — and the
returned frame occupies ~0 RAM until its values are actually touched.

The read pattern iterates over **entity blocks aligned to the Zarr entity-chunk
size** (256), reading all time steps per block. Because the prediction variable's
time extent is small (< the 256 time-chunk), each Zarr chunk is read exactly once,
so total I/O equals the tensor size with no redundant reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from views_frames import (
    FrameMetadata,
    PredictionFrame,
    SpatioTemporalIndex,
)

__all__ = ["build_prediction_frames_from_dataset"]

#: Entities read per block. A multiple of the Zarr entity-chunk size (256) keeps
#: reads chunk-aligned. Larger = fewer passes, more peak memory.
_DEFAULT_ENTITY_BLOCK = 1024


def build_prediction_frames_from_dataset(
    ds,
    target_names: Iterable[str],
    out_dir: str | Path,
    *,
    entity_block: int = _DEFAULT_ENTITY_BLOCK,
    model_name: str = "darts",
) -> dict[str, PredictionFrame]:
    """Stream a zarr-backed dataset into memmap-backed PredictionFrames.

    For each target this writes a row-major ``(N, S)`` float32 ``values.npy``
    into ``out_dir`` in entity-aligned blocks, then returns a
    :class:`PredictionFrame` whose ``values`` is a read-only memmap over that
    file. Nothing larger than one entity block is ever held in RAM.

    Args:
        ds: A zarr-backed :class:`ViewsDataset` (e.g. ``DatasetBuilder.build()``).
        target_names: Target names to export; each maps to a ``pred_<name>``
            variable in ``ds``.
        out_dir: Directory for the memmap files. Created if missing. **Must stay
            on disk for as long as the returned frames are used.**
        entity_block: Entities read per block. Keep a multiple of the Zarr
            entity-chunk size (256) for aligned reads.
        model_name: Recorded in the frame metadata.

    Returns:
        ``{target_name: PredictionFrame}`` with memmap-backed values.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xr_ds = ds.to_xarray()
    time_id = ds._time_id
    entity_id = ds._entity_id
    times = np.asarray(xr_ds[time_id].values)
    entities = np.asarray(xr_ds[entity_id].values)
    T = times.shape[0]
    E = entities.shape[0]
    S = int(ds.sample_size)
    N = T * E

    # Shared time-major index: row r == (times[r // E], entities[r % E]).
    t_grid, e_grid = np.meshgrid(times, entities, indexing="ij")
    time_flat = np.ascontiguousarray(t_grid.ravel())
    unit_flat = np.ascontiguousarray(e_grid.ravel())
    index = SpatioTemporalIndex(time_flat, unit_flat, level=ds._build_spatial_level())
    meta = FrameMetadata(model=model_name)

    frames: dict[str, PredictionFrame] = {}
    for target_name in target_names:
        pred_var = f"pred_{target_name}"
        if pred_var not in xr_ds:
            raise KeyError(
                f"variable {pred_var!r} not found in dataset; "
                f"available: {sorted(xr_ds.data_vars)}"
            )
        var = xr_ds[pred_var].transpose(time_id, entity_id, "sample")

        values_path = out_dir / f"{target_name}.values.npy"
        mmap = np.lib.format.open_memmap(
            str(values_path), mode="w+", dtype=np.float32, shape=(N, S)
        )
        # Stream entity-aligned blocks (all time steps per block). Each block
        # covers whole Zarr entity-chunks, so every chunk is read exactly once.
        for e0 in range(0, E, entity_block):
            e1 = min(e0 + entity_block, E)
            block = var.isel({entity_id: slice(e0, e1)}).values  # (T, e1-e0, S)
            block = np.ascontiguousarray(block, dtype=np.float32)
            for t in range(T):
                row_start = t * E + e0
                row_end = t * E + e1
                mmap[row_start:row_end, :] = block[t, :, :]
        mmap.flush()
        del mmap

        # Reopen read-only; PredictionFrame keeps the memmap (no copy).
        values = np.load(str(values_path), mmap_mode="r")
        frames[target_name] = PredictionFrame(values, index=index, metadata=meta)

    return frames
