"""views-r2darts2 — Darts-based forecasting for the VIEWS platform.

Public API:
    * :class:`ViewsDataset` — zarr-backed central data interface (input,
      output, slicing, scaling, Darts TimeSeries construction).
    * :class:`DartsForecaster` — slim model + partition orchestrator.
    * :class:`DartsForecastingModelManager` — experiment lifecycle manager.
    * :class:`FeatureScalerManager` — per-feature scaler manager.
    * :class:`ScalerSelector` — sklearn / Darts scaler factory.
    * :class:`ModelCatalog`, :class:`LossCatalog`, :class:`OptimizerCatalog`,
      :class:`SchedulerCatalog` — model/loss/optimizer/scheduler registries.
    * :class:`MarkovModel` — sklearn-backed Markov prediction model (the one
      non-torch forecasting model in the catalog).
    * :func:`apply_all_patches`, :func:`apply_tide_mc_dropout_patch` — Darts
      monkey-patches.

Architecture: The :class:`ViewsDataset` is the single source of truth for all
data operations. The forecaster and manager delegate to it — they hold no
data manipulation logic. The dataset is zarr-backed (disk-resident, lazy
Dask-backed xarray) with optional memmap caching.

Pandas-free: the only pandas touchpoint is in
:mod:`views_r2darts2.transformers.darts_bridge`, where Darts' API requires a
``pd.Index`` and ``pd.DataFrame`` for time-index and static-covariate
construction. No other module imports pandas.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ViewsDataset",
    "DartsForecaster",
    "DartsForecastingModelManager",
    "FeatureScalerManager",
    "ScalerSelector",
    "ModelCatalog",
    "LossCatalog",
    "OptimizerCatalog",
    "SchedulerCatalog",
    "MarkovModel",
    "apply_all_patches",
    "apply_tide_mc_dropout_patch",
]


def __getattr__(name: str) -> Any:
    if name == "ViewsDataset":
        from views_r2darts2.dataset.base import ViewsDataset
        return ViewsDataset
    if name == "DartsForecaster":
        from views_r2darts2.engines.darts_forecaster import DartsForecaster
        return DartsForecaster
    if name == "DartsForecastingModelManager":
        from views_r2darts2.engines.darts_forecasting_model_manager import (
            DartsForecastingModelManager,
        )
        return DartsForecastingModelManager
    if name == "FeatureScalerManager":
        from views_r2darts2.transformers.feature_scaler_manager import (
            FeatureScalerManager,
        )
        return FeatureScalerManager
    if name == "ScalerSelector":
        from views_r2darts2.transformers.scaler_selector import ScalerSelector
        return ScalerSelector
    if name in ("ModelCatalog", "LossCatalog", "OptimizerCatalog", "SchedulerCatalog"):
        mod = {
            "ModelCatalog": "model_catalog",
            "LossCatalog": "loss_catalog",
            "OptimizerCatalog": "optimizer_catalog",
            "SchedulerCatalog": "scheduler_catalog",
        }[name]
        import importlib
        return importlib.import_module(f"views_r2darts2.catalogs.{mod}").__dict__[name]
    if name == "MarkovModel":
        from views_r2darts2.models.markov_model import MarkovModel
        return MarkovModel
    if name in ("apply_all_patches", "apply_tide_mc_dropout_patch"):
        from views_r2darts2.infrastructure import patches
        return getattr(patches, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
