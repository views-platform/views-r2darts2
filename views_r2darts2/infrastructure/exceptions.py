class ReproducibilityError(Exception):
    """Base class for all reproducibility gate failures."""
    pass

class MissingHyperparameterError(ReproducibilityError):
    """Raised when a mandatory hyperparameter is missing from the config."""
    pass

class ArchitectureMismatchError(ReproducibilityError, ValueError):
    """Raised when model architecture and forecast horizon are misaligned."""
    pass

class TemporalDiscontinuityError(ReproducibilityError):
    """Raised when training and testing sets are not contiguous (t+1 failure)."""
    pass

class DataLeakageError(ReproducibilityError):
    """Raised when test data is detected within a training tensor."""
    pass

class DataStarvationError(ReproducibilityError):
    """Raised when training data fails to utilize the full available history."""
    pass

class NumericalSanityError(ReproducibilityError):
    """Raised when NaNs, Infs, or extreme outliers are detected in the data stream."""
    pass

class TemporalHoleError(ReproducibilityError):
    """Raised when missing months are detected in a historical sequence."""
    pass

class PredictionHorizonError(ReproducibilityError):
    """Raised when a forecast exceeds the ground-truth boundary of a test set."""
    pass
