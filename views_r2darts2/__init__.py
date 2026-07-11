"""views-r2darts2 — Darts-based forecasting for the VIEWS platform.

Public API:
    * :class:`ViewsDatasetDarts` — FeatureFrame-backed data boundary.
    * :class:`DartsForecaster` — model + scaler coupling engine.
    * :class:`DartsForecastingModelManager` — experiment orchestrator.
    * :class:`FeatureScalerManager` — per-feature scaler manager.
    * :class:`ScalerSelector` — sklearn / Darts scaler factory.
    * :class:`ModelCatalog`, :class:`LossCatalog`, :class:`OptimizerCatalog`,
      :class:`SchedulerCatalog` — model/loss/optimizer/scheduler registries.
    * :func:`load_views_parquet` — pandas-free parquet → FeatureFrame loader.
    * :func:`apply_all_patches`, :func:`apply_tide_mc_dropout_patch` — Darts
      monkey-patches.

Pandas-free: the only pandas touchpoint is in
:mod:`views_r2darts2.transformers.darts_bridge`, where Darts' API requires a
``pd.Index`` and ``pd.DataFrame`` for time-index and static-covariate
construction. No other module imports pandas.
"""

from __future__ import annotations

from views_r2darts2.catalogs.loss_catalog import LossCatalog
from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.catalogs.optimizer_catalog import OptimizerCatalog
from views_r2darts2.catalogs.scheduler_catalog import SchedulerCatalog
from views_r2darts2.data.parquet_loader import load_views_parquet
from views_r2darts2.data.views_dataset import ViewsDatasetDarts
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.engines.darts_forecasting_model_manager import (
    DartsForecastingModelManager,
)
from views_r2darts2.infrastructure.patches import (
    apply_all_patches,
    apply_tide_mc_dropout_patch,
)
from views_r2darts2.transformers.feature_scaler_manager import FeatureScalerManager
from views_r2darts2.transformers.scaler_selector import ScalerSelector

__all__ = [
    "ViewsDatasetDarts",
    "DartsForecaster",
    "DartsForecastingModelManager",
    "FeatureScalerManager",
    "ScalerSelector",
    "ModelCatalog",
    "LossCatalog",
    "OptimizerCatalog",
    "SchedulerCatalog",
    "load_views_parquet",
    "apply_all_patches",
    "apply_tide_mc_dropout_patch",
]
