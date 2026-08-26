"""Build memmap-backed PredictionFrames from a zarr-backed ViewsDataset.

When streaming predictions (e.g. MC-dropout inference over the full PGM grid),
the prediction tensor is far larger than RAM. ``DatasetBuilder`` already streams
the writes into a Zarr store; this module provides the complementary read path:
it streams the predictions back out of the Zarr store into a row-major
``(N, S)`` memmap file in entity-aligned blocks, and wraps the result in a
:class:`views_frames.PredictionFrame` whose ``values`` is a read-only
``np.memmap``. Peak memory is one entity block — never the full grid — and the
returned frame occupies ~0 RAM until its values are actually touched.

After the memmap files are written, flushed, and verified (shape, target names,
readback), the Zarr store is deleted to avoid keeping duplicate prediction data
on disk. If verification fails, the Zarr store is kept and an exception is
raised.

The read pattern iterates over **entity blocks aligned to the Zarr entity-chunk
size** (256), reading all time steps per block. Because the prediction variable's
time extent is small (< the 256 time-chunk), each Zarr chunk is read exactly once,
so total I/O equals the tensor size with no redundant reads.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from views_frames import (
    FrameMetadata,
    PredictionFrame,
    SpatioTemporalIndex,
)

__all__ = ["build_prediction_frames_from_dataset"]

logger = logging.getLogger(__name__)

#: Entities read per block. A multiple of the Zarr entity-chunk size (256) keeps
#: reads chunk-aligned. Larger = fewer passes, more peak memory.
_DEFAULT_ENTITY_BLOCK = 1024


class PredictionFrameVerificationError(RuntimeError):
    """Raised when a PredictionFrame memmap fails verification after write."""


def build_prediction_frames_from_dataset(
    ds,
    target_names: Iterable[str],
    out_dir: str | Path,
    *,
    entity_block: int = _DEFAULT_ENTITY_BLOCK,
    model_name: str = "darts",
    zarr_cleanup: bool = True,
) -> dict[str, PredictionFrame]:
    """Stream a zarr-backed dataset into memmap-backed PredictionFrames.

    For each target this writes a row-major ``(N, S)`` float32 ``values.npy``
    into ``out_dir`` in entity-aligned blocks, then returns a
    :class:`PredictionFrame` whose ``values`` is a read-only memmap over that
    file. Nothing larger than one entity block is ever held in RAM.

    After all memmaps are written and verified, the Zarr store backing ``ds``
    is deleted (if ``zarr_cleanup=True``) to avoid keeping duplicate prediction
    data on disk. If verification fails, the Zarr store is kept and an exception
    is raised.

    Args:
        ds: A zarr-backed :class:`ViewsDataset` (e.g. ``DatasetBuilder.build()``).
        target_names: Target names to export; each maps to a ``pred_<name>``
            variable in ``ds``.
        out_dir: Directory for the memmap files. Created if missing. **Must stay
            on disk for as long as the returned frames are used.**
        entity_block: Entities read per block. Keep a multiple of the Zarr
            entity-chunk size (256) for aligned reads.
        model_name: Recorded in the frame metadata.
        zarr_cleanup: When ``True`` (default), delete the Zarr store backing
            ``ds`` after all PredictionFrames are verified. When ``False``,
            the caller is responsible for cleanup.

    Returns:
        ``{target_name: PredictionFrame}`` with memmap-backed values.

    Raises:
        PredictionFrameVerificationError: If a memmap fails verification
            (shape mismatch, NaN in readback, or reopen failure). The Zarr
            store is NOT deleted on failure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture the Zarr store path before we lose the reference (ds.close()
    # may invalidate the internal _store).
    zarr_store_path = None
    zarr_store_obj = getattr(ds, "_store", None)
    if zarr_store_obj is not None:
        zarr_store_path = getattr(zarr_store_obj, "path", None)

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

    target_list = list(target_names)
    frames: dict[str, PredictionFrame] = {}
    written_paths: list[Path] = []

    for target_name in target_list:
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

        written_paths.append(values_path)

    # --- Verification phase -----------------------------------------------
    # For each written memmap file:
    # 1. Reopen read-only.
    # 2. Check shape matches (N, S).
    # 3. Read a small deterministic slice (first row, last row, middle row)
    #    and check for NaN.
    # 4. Check that the index shape matches.
    # If any check fails, raise — the Zarr store is kept on disk.
    for target_name in target_list:
        values_path = out_dir / f"{target_name}.values.npy"
        if not values_path.exists():
            raise PredictionFrameVerificationError(
                f"PredictionFrame file not found after write: {values_path}"
            )
        try:
            values = np.load(str(values_path), mmap_mode="r")
        except Exception as exc:
            raise PredictionFrameVerificationError(
                f"Failed to reopen memmap for target '{target_name}': {exc}"
            ) from exc

        if values.shape != (N, S):
            raise PredictionFrameVerificationError(
                f"Shape mismatch for target '{target_name}': "
                f"expected ({N}, {S}), got {values.shape}"
            )

        # Readback check: first, middle, last row — no NaN.
        check_rows = [0, N // 2, N - 1] if N > 0 else []
        for r in check_rows:
            row_vals = values[r, :]
            if np.isnan(row_vals).any():
                raise PredictionFrameVerificationError(
                    f"NaN in readback for target '{target_name}' at row {r}"
                )

        frames[target_name] = PredictionFrame(values, index=index, metadata=meta)

    # Verify index shape.
    if index.n_rows != N:
        raise PredictionFrameVerificationError(
            f"Index row count mismatch: expected {N}, got {index.n_rows}"
        )

    logger.info(
        "PredictionFrame verification passed: %d targets, %d rows, %d samples. "
        "Memmap files in %s.",
        len(target_list), N, S, out_dir,
    )

    # --- Zarr cleanup phase -----------------------------------------------
    # Only delete the Zarr store AFTER all PredictionFrames are verified.
    if zarr_cleanup and zarr_store_path is not None:
        ds.close()  # Release the zarr-backed dataset references.
        if Path(zarr_store_path).exists():
            shutil.rmtree(str(zarr_store_path), ignore_errors=True)
            logger.info(
                "Zarr store deleted after verified PredictionFrame conversion: %s",
                zarr_store_path,
            )
    elif not zarr_cleanup:
        # Caller is responsible for cleanup.
        pass

    return frames
