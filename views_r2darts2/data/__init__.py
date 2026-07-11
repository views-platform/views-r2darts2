"""Data subpackage: parquet loading + FeatureFrame-backed dataset."""

from __future__ import annotations

from views_r2darts2.data.parquet_loader import (
    ParquetLoadError,
    load_views_parquet,
)
from views_r2darts2.data.views_dataset import ViewsDatasetDarts

__all__ = ["ViewsDatasetDarts", "load_views_parquet", "ParquetLoadError"]
