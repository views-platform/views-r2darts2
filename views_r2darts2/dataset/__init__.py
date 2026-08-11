"""Disk-backed xarray + Zarr + Dask dataset classes.

A lazy facade over the legacy pandas ``_ViewsDataset``: the whole dataset lives
as chunked Zarr arrays on disk and every accessor returns Dask-backed
``xarray`` objects, so peak memory is bounded by the largest chunk rather than
the row count.

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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
