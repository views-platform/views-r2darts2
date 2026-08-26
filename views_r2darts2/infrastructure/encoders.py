"""Cyclic time encoders for VIEWS temporal resolutions.

Each encoder takes an integer index-like (any object coercible by
``np.asarray``) and returns a ``float32`` ``sin`` or ``cos`` array of the same
shape, encoding the position within its calendar period using the VIEWS
``(idx - 1) % period`` convention.

The four supported resolutions (per ADR-001) are:

    * ``cm`` / ``pgm`` — monthly, period = 12
    * ``cw`` / ``pgw`` — weekly, period = 52
    * ``cd`` / ``pgd`` — daily, period = 7 (day-of-week) and 365 (day-of-year)
    * ``cy`` / ``pgy`` — yearly, no cyclic encoding (constant)

The :data:`CYCLIC_ENCODERS_BY_RESOLUTION` dict maps the *last character* of a
VIEWS level string (``"cm"`` → ``"m"``, ``"pgd"`` → ``"d"``, etc.) to the list
of encoder functions for that resolution. Yearly maps to ``None`` (no
encoders).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _cyclic(idx: object, period: int, func: Callable[[NDArray], NDArray]) -> NDArray[np.float32]:
    """Encode ``idx`` as a cyclic ``sin``/``cos`` value with the given period."""
    arr = np.asarray(idx, dtype=np.float32)
    return func(2.0 * np.pi * ((arr - 1.0) % period) / period).astype(np.float32)


def month_sin(idx: object) -> NDArray[np.float32]:
    """Sin of the month-of-year cycle (period = 12)."""
    return _cyclic(idx, 12, np.sin)


def month_cos(idx: object) -> NDArray[np.float32]:
    """Cos of the month-of-year cycle (period = 12)."""
    return _cyclic(idx, 12, np.cos)


def week_of_year_sin(idx: object) -> NDArray[np.float32]:
    """Sin of the week-of-year cycle (period = 52)."""
    return _cyclic(idx, 52, np.sin)


def week_of_year_cos(idx: object) -> NDArray[np.float32]:
    """Cos of the week-of-year cycle (period = 52)."""
    return _cyclic(idx, 52, np.cos)


def day_of_week_sin(idx: object) -> NDArray[np.float32]:
    """Sin of the day-of-week cycle (period = 7)."""
    return _cyclic(idx, 7, np.sin)


def day_of_week_cos(idx: object) -> NDArray[np.float32]:
    """Cos of the day-of-week cycle (period = 7)."""
    return _cyclic(idx, 7, np.cos)


def day_of_year_sin(idx: object) -> NDArray[np.float32]:
    """Sin of the day-of-year cycle (period = 365; leap years ignored)."""
    return _cyclic(idx, 365, np.sin)


def day_of_year_cos(idx: object) -> NDArray[np.float32]:
    """Cos of the day-of-year cycle (period = 365; leap years ignored)."""
    return _cyclic(idx, 365, np.cos)


# Resolution → encoder list. ``"y"`` (yearly) maps to ``None`` — no encoders.
CYCLIC_ENCODERS_BY_RESOLUTION: dict[str, list[Callable] | None] = {
    "m": [month_sin, month_cos],
    "w": [week_of_year_sin, week_of_year_cos],
    "d": [day_of_week_sin, day_of_week_cos, day_of_year_sin, day_of_year_cos],
    "y": None,
}
