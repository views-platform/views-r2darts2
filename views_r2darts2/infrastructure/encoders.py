"""
Callable index encoders for use with Darts' ``add_encoders`` configuration.

These functions receive a raw ``pd.Index`` of integer time-step identifiers
and return a ``np.ndarray`` of ``float32`` covariate values.  They are
defined in an importable package module so that ``torch.save`` / pickle can
serialise them by (module, qualname) without error.

VIEWS temporal resolutions and their integer index conventions
--------------------------------------------------------------
- Monthly (cm, pgm): ``month_id`` — 1 = January 1980.
  Encoders: month-of-year (period 12).

- Weekly (cw, pgw): ``week_id`` — 1 = first ISO week of 1980.
  Encoders: week-of-year (period 52).

- Daily (cd, pgd): ``day_id`` — 1 = 1 January 1980.
  Encoders: day-of-week (period 7) + day-of-year (period 365).
  Day-of-week is the primary short-cycle signal; day-of-year adds
  annual seasonality. Leap years are ignored (period fixed at 365).

- Yearly (cy, pgy): single observation per year — no intra-year
  cycle to encode; ``use_cyclic_encoders`` returns None.

All encoders follow the same convention: ``(idx - 1) % period`` so that
the first integer in each series maps to phase 0.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Monthly encoders  (cm / pgm)
# ---------------------------------------------------------------------------

def month_sin(idx) -> np.ndarray:
    """Sine of the month-of-year cycle (period 12).

    ``(idx - 1) % 12`` maps month_id 1 (Jan) → 0, …, 12 (Dec) → 11.
    """
    arr = np.asarray(idx, dtype=np.float32)
    return np.sin(2 * np.pi * ((arr - 1) % 12) / 12)


def month_cos(idx) -> np.ndarray:
    """Cosine of the month-of-year cycle (period 12)."""
    arr = np.asarray(idx, dtype=np.float32)
    return np.cos(2 * np.pi * ((arr - 1) % 12) / 12)


# ---------------------------------------------------------------------------
# Weekly encoders  (cw / pgw)
# ---------------------------------------------------------------------------

def week_of_year_sin(idx) -> np.ndarray:
    """Sine of the week-of-year cycle (period 52).

    ``(idx - 1) % 52`` maps week_id 1 (first week of 1980) → 0, etc.
    """
    arr = np.asarray(idx, dtype=np.float32)
    return np.sin(2 * np.pi * ((arr - 1) % 52) / 52)


def week_of_year_cos(idx) -> np.ndarray:
    """Cosine of the week-of-year cycle (period 52)."""
    arr = np.asarray(idx, dtype=np.float32)
    return np.cos(2 * np.pi * ((arr - 1) % 52) / 52)


# ---------------------------------------------------------------------------
# Daily encoders  (cd / pgd)
# ---------------------------------------------------------------------------

def day_of_week_sin(idx) -> np.ndarray:
    """Sine of the day-of-week cycle (period 7).

    ``(idx - 1) % 7`` maps day_id 1 (Mon 1 Jan 1980) → 0, etc.
    """
    arr = np.asarray(idx, dtype=np.float32)
    return np.sin(2 * np.pi * ((arr - 1) % 7) / 7)


def day_of_week_cos(idx) -> np.ndarray:
    """Cosine of the day-of-week cycle (period 7)."""
    arr = np.asarray(idx, dtype=np.float32)
    return np.cos(2 * np.pi * ((arr - 1) % 7) / 7)


def day_of_year_sin(idx) -> np.ndarray:
    """Sine of the day-of-year cycle (period 365, leap years ignored).

    ``(idx - 1) % 365`` maps day_id 1 (1 Jan 1980) → 0, etc.
    """
    arr = np.asarray(idx, dtype=np.float32)
    return np.sin(2 * np.pi * ((arr - 1) % 365) / 365)


def day_of_year_cos(idx) -> np.ndarray:
    """Cosine of the day-of-year cycle (period 365, leap years ignored)."""
    arr = np.asarray(idx, dtype=np.float32)
    return np.cos(2 * np.pi * ((arr - 1) % 365) / 365)


# ---------------------------------------------------------------------------
# Resolution → encoder list lookup
# ---------------------------------------------------------------------------

#: Maps the temporal-resolution suffix of a VIEWS level string to the
#: ordered list of cyclic encoder functions for that resolution.
#: Yearly resolutions (``y``) produce no encoders (None).
CYCLIC_ENCODERS_BY_RESOLUTION: dict = {
    "m": [month_sin, month_cos],
    "w": [week_of_year_sin, week_of_year_cos],
    "d": [day_of_week_sin, day_of_week_cos, day_of_year_sin, day_of_year_cos],
    "y": None,
}
