"""
Callable index encoders for use with Darts' ``add_encoders`` configuration.

These functions receive a raw ``pd.Index`` of integer time-step identifiers
and return a ``np.ndarray`` of ``float32`` covariate values.  They are
defined in an importable package module so that ``torch.save`` / pickle can
serialise them by (module, qualname) without error.

Usage in ``add_encoders``::

    from views_r2darts2.encoders import month_sin, month_cos

    "add_encoders": {
        "custom": {
            "past":   [month_sin, month_cos],
            "future": [month_sin, month_cos],
        },
    }
"""

import numpy as np


def month_sin(idx) -> np.ndarray:
    """Sine component of the month-of-year cycle.

    Assumes ``idx`` contains integer ``month_id`` values where 1 = Jan 1980,
    so ``(idx - 1) % 12`` maps January → 0, February → 1, …, December → 11.

    Parameters
    ----------
    idx:
        Raw ``pd.Index`` (or array-like) of integer month identifiers passed
        by Darts' ``CallableIndexEncoder``.

    Returns
    -------
    np.ndarray of float32
    """
    arr = np.asarray(idx, dtype=np.float32)
    return np.sin(2 * np.pi * ((arr - 1) % 12) / 12)


def month_cos(idx) -> np.ndarray:
    """Cosine component of the month-of-year cycle.

    Complements :func:`month_sin` — a single sinusoidal feature is ambiguous
    (two months share the same sine value), so both components are needed for
    an unambiguous cyclic encoding.

    Parameters
    ----------
    idx:
        Raw ``pd.Index`` (or array-like) of integer month identifiers passed
        by Darts' ``CallableIndexEncoder``.

    Returns
    -------
    np.ndarray of float32
    """
    arr = np.asarray(idx, dtype=np.float32)
    return np.cos(2 * np.pi * ((arr - 1) % 12) / 12)
