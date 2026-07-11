"""Device detection utility (no dependencies).

Moved out of :class:`DartsForecaster` so that :class:`ModelCatalog` can call
``get_device()`` without importing the forecaster (which would create a
circular import: ``catalogs → engines → catalogs``).

Google Python Style.
"""

from __future__ import annotations

import torch


def get_device() -> str:
    """Return the device type for model training (``mps`` / ``cuda`` / ``cpu``).

    Sets the default dtype to ``float32`` when MPS is available (matches the
    legacy behavior in :class:`DartsForecaster.get_device`).
    """
    if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
