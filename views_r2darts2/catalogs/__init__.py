"""Catalogs subpackage: loss, model, optimizer, scheduler registries."""

from __future__ import annotations

from views_r2darts2.catalogs.loss_catalog import LossCatalog
from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.catalogs.optimizer_catalog import OptimizerCatalog
from views_r2darts2.catalogs.scheduler_catalog import SchedulerCatalog

__all__ = ["LossCatalog", "ModelCatalog", "OptimizerCatalog", "SchedulerCatalog"]
