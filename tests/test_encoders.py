"""Tests for :mod:`views_r2darts2.infrastructure.encoders`.

Exercises the four cyclic encoders (``month_*``, ``week_of_year_*``,
``day_of_week_*``, ``day_of_year_*``), the :data:`CYCLIC_ENCODERS_BY_RESOLUTION`
mapping, and the input-coercion contract (numpy array, list, scalar).

Pandas-free.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from views_r2darts2.infrastructure.encoders import (
    CYCLIC_ENCODERS_BY_RESOLUTION,
    day_of_week_cos,
    day_of_week_sin,
    day_of_year_cos,
    day_of_year_sin,
    month_cos,
    month_sin,
    week_of_year_cos,
    week_of_year_sin,
)


# ----------------------------------------------------------------------
# Shape / range / dtype
# ----------------------------------------------------------------------


class TestEncodersShapeRangeDtype:
    """Shape, value-range, and dtype contracts for the cyclic encoders."""

    def test_month_sin_cos_shape_and_range(self) -> None:
        """``month_sin``/``month_cos`` of a length-24 array return shape (24,)
        and values in [-1, 1]."""
        idx = np.arange(1, 25)
        sin_vals = month_sin(idx)
        cos_vals = month_cos(idx)
        assert sin_vals.shape == (24,)
        assert cos_vals.shape == (24,)
        assert np.all(np.abs(sin_vals) <= 1.0 + 1e-6)
        assert np.all(np.abs(cos_vals) <= 1.0 + 1e-6)

    def test_encoders_return_float32(self) -> None:
        """Every encoder must return ``float32`` (ADR-010 airlock invariant)."""
        idx = np.arange(1, 13, dtype=np.int64)
        for fn in (
            month_sin, month_cos,
            week_of_year_sin, week_of_year_cos,
            day_of_week_sin, day_of_week_cos,
            day_of_year_sin, day_of_year_cos,
        ):
            out = fn(idx)
            assert out.dtype == np.float32, f"{fn.__name__} returned {out.dtype}"

    def test_encoders_handle_numpy_array_input(self) -> None:
        """A 1-D ``np.ndarray`` input produces a same-shape output."""
        idx = np.array([1, 2, 3], dtype=np.int64)
        out = month_sin(idx)
        assert out.shape == (3,)

    def test_encoders_handle_list_input(self) -> None:
        """A plain Python list input is coerced to a numpy array."""
        out = month_sin([1, 2, 3])
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)


# ----------------------------------------------------------------------
# Period contracts
# ----------------------------------------------------------------------


class TestEncodersPeriods:
    """Periodicity contracts for each encoder family."""

    def test_month_sin_period_12(self) -> None:
        """``month_sin(1)`` ≈ ``month_sin(13)`` (period = 12)."""
        np.testing.assert_allclose(
            month_sin(1), month_sin(13), rtol=1e-6,
        )

    def test_month_cos_at_january(self) -> None:
        """``month_cos(1)`` ≈ ``cos(0) = 1.0`` (VIEWS ``idx-1`` convention)."""
        np.testing.assert_allclose(
            float(month_cos(1)), 1.0, rtol=1e-5, atol=1e-6,
        )

    def test_month_sin_cos_orbit_period_12(self) -> None:
        """For any month ``m``, ``month_sin(m) == month_sin(m + 12)`` and same for cos."""
        for m in (1, 6, 12, 47):
            np.testing.assert_allclose(
                month_sin(m), month_sin(m + 12), rtol=1e-6,
            )
            np.testing.assert_allclose(
                month_cos(m), month_cos(m + 12), rtol=1e-6,
            )

    def test_week_of_year_sin_period_52(self) -> None:
        """``week_of_year_sin`` has period 52."""
        for w in (1, 26, 52):
            np.testing.assert_allclose(
                week_of_year_sin(w), week_of_year_sin(w + 52), rtol=1e-6,
            )

    def test_day_of_week_sin_period_7(self) -> None:
        """``day_of_week_sin`` has period 7."""
        for d in (1, 3, 7):
            np.testing.assert_allclose(
                day_of_week_sin(d), day_of_week_sin(d + 7), rtol=1e-6,
            )

    def test_day_of_year_sin_period_365(self) -> None:
        """``day_of_year_sin`` has period 365 (leap years ignored)."""
        for d in (1, 100, 365):
            np.testing.assert_allclose(
                day_of_year_sin(d), day_of_year_sin(d + 365), rtol=1e-6,
            )


# ----------------------------------------------------------------------
# Resolution mapping
# ----------------------------------------------------------------------


class TestCyclicEncodersByResolution:
    """``CYCLIC_ENCODERS_BY_RESOLUTION`` resolution → encoder-list mapping."""

    def test_cyclic_encoders_by_resolution_dict(self) -> None:
        """``'m'``→2 fns, ``'w'``→2 fns, ``'d'``→4 fns, ``'y'``→None."""
        monthly = CYCLIC_ENCODERS_BY_RESOLUTION["m"]
        weekly = CYCLIC_ENCODERS_BY_RESOLUTION["w"]
        daily = CYCLIC_ENCODERS_BY_RESOLUTION["d"]
        yearly = CYCLIC_ENCODERS_BY_RESOLUTION["y"]

        assert monthly is not None and len(monthly) == 2
        assert weekly is not None and len(weekly) == 2
        assert daily is not None and len(daily) == 4
        assert yearly is None

        # Monthly encoders are the month_sin / month_cos pair.
        monthly_names = {fn.__name__ for fn in monthly}
        assert monthly_names == {"month_sin", "month_cos"}

        # Daily encoders cover day-of-week + day-of-year (4 functions).
        daily_names = {fn.__name__ for fn in daily}
        assert daily_names == {
            "day_of_week_sin", "day_of_week_cos",
            "day_of_year_sin", "day_of_year_cos",
        }

    def test_resolution_keys_are_complete(self) -> None:
        """The mapping must cover every VIEWS temporal resolution."""
        assert set(CYCLIC_ENCODERS_BY_RESOLUTION.keys()) == {"m", "w", "d", "y"}
