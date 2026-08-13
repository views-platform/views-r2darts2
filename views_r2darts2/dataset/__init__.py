"""Disk-backed xarray + Zarr + Dask dataset classes.

A lazy facade over the legacy pandas ``_ViewsDataset``: the whole dataset lives
as chunked Zarr arrays on disk and every accessor returns Dask-backed
``xarray`` objects, so peak memory is bounded by the largest chunk rather than
the row count.

Reading existing data (DataFrame, Parquet, PredictionFrame, FeatureFrame) goes
through the converters in :mod:`views_r2darts2.dataset.converters`. Producing
data incrementally (e.g. prediction batches) goes through
:class:`views_r2darts2.dataset.builder.DatasetBuilder` via
:meth:`ViewsDataset.builder` — the builder pre-allocates a NaN-filled Zarr
skeleton (metadata only) and scatter-writes each batch to disk, so peak memory
is one batch, never the grid. A built dataset supports every existing export
(``to_predictionframe``, ``save_parquet``, ``save_zarr``, ``save_predstore``,
``save_appwrite``).

This module is self-contained — it does NOT import from
``views_pipeline_core``. The ``ViewsDataset`` class and its subclasses live
in the local ``base.py`` / ``subclasses.py`` files.

Imports are lazy (module ``__getattr__``) so importing this package does not
pull in xarray/zarr/dask until a class is actually used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ViewsDataset",
    "PGDataset",
    "PGMDataset",
    "PGYDataset",
    "CDataset",
    "CMDataset",
    "CYDataset",
    "ZarrStore",
    "DatasetBuilder",
]


def __getattr__(name: str) -> Any:
    if name == "ViewsDataset":
        from views_r2darts2.dataset.base import ViewsDataset
        return ViewsDataset
    if name in ("PGDataset", "PGMDataset", "PGYDataset", "CDataset", "CMDataset", "CYDataset"):
        from views_r2darts2.dataset import subclasses
        return getattr(subclasses, name)
    if name == "ZarrStore":
        from views_r2darts2.dataset.zarr_store import ZarrStore
        return ZarrStore
    if name == "DatasetBuilder":
        from views_r2darts2.dataset.builder import DatasetBuilder
        return DatasetBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
