"""Exception hierarchy for the reproducibility gate.

Pure-Python module: no external dependencies. The class names are part of the
public API — tests and downstream code import them by name.
"""

from __future__ import annotations


class ReproducibilityError(Exception):
    """Base for all reproducibility gate failures."""


class MissingHyperparameterError(ReproducibilityError):
    """A mandatory hyperparameter is missing from the config manifest."""


class ArchitectureMismatchError(ReproducibilityError, ValueError):
    """The model architecture and the forecast horizon are misaligned."""


class TemporalDiscontinuityError(ReproducibilityError):
    """Train and test partitions are not temporally contiguous."""


class DataLeakageError(ReproducibilityError):
    """Test data was detected within a training tensor."""


class DataStarvationError(ReproducibilityError):
    """Training data fails to use the full available history."""


class NumericalSanityError(ReproducibilityError):
    """NaN, Inf, or extreme outliers detected in the data stream."""


class TemporalHoleError(ReproducibilityError):
    """A temporal sequence has missing intermediate steps."""


class PredictionHorizonError(ReproducibilityError):
    """The forecast horizon exceeds the ground-truth boundary of the test set."""
