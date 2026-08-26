"""Infrastructure subpackage: callbacks, encoders, exceptions, patches, gate."""

from __future__ import annotations

from views_r2darts2.infrastructure.exceptions import (
    ArchitectureMismatchError,
    DataLeakageError,
    DataStarvationError,
    MissingHyperparameterError,
    NumericalSanityError,
    PredictionHorizonError,
    ReproducibilityError,
    TemporalDiscontinuityError,
    TemporalHoleError,
)
from views_r2darts2.infrastructure.reproducibility_gate import (
    ReproducibilityGate,
    TemporalContinuityError,  # back-compat alias for TemporalDiscontinuityError
)

__all__ = [
    "ReproducibilityGate",
    "ReproducibilityError",
    "MissingHyperparameterError",
    "ArchitectureMismatchError",
    "TemporalDiscontinuityError",
    "TemporalContinuityError",
    "DataLeakageError",
    "DataStarvationError",
    "NumericalSanityError",
    "TemporalHoleError",
    "PredictionHorizonError",
]
