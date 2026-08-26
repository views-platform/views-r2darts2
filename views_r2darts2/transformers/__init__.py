"""Transformers subpackage: scalers, inverse utilities, Darts boundary."""

from __future__ import annotations

from views_r2darts2.transformers.darts_bridge import (
    build_entity_timeseries,
    prediction_frame_from_darts,
    prediction_frames_from_darts,
)
from views_r2darts2.transformers.feature_scaler_manager import FeatureScalerManager
from views_r2darts2.transformers.frame_builder import (
    build_prediction_frames_from_dataset,
)
from views_r2darts2.transformers.scaler_selector import ScalerSelector
from views_r2darts2.transformers.static_covariates import (
    StaticCovariateConfig,
    StaticCovariateStats,
    compute_static_covariates,
)

__all__ = [
    "FeatureScalerManager",
    "ScalerSelector",
    "StaticCovariateConfig",
    "StaticCovariateStats",
    "compute_static_covariates",
    "build_entity_timeseries",
    "prediction_frame_from_darts",
    "prediction_frames_from_darts",
    "build_prediction_frames_from_dataset",
]
