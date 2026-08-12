"""Models subpackage.

Holds forecasting-model implementations that are too large to live inside the
catalog. Currently exports :class:`MarkovModel` — a sklearn-backed Markov
prediction model that bypasses the torch-specific reproducibility gates.
"""

from __future__ import annotations

from views_r2darts2.models.markov_model import (
    MarkovFatalityModel,
    MarkovModel,
    MarkovState,
    MarkovStateModel,
)

__all__ = [
    "MarkovFatalityModel",
    "MarkovModel",
    "MarkovState",
    "MarkovStateModel",
]
