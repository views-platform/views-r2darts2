"""Tests for :class:`views_r2darts2.transformers.scaler_selector.ScalerSelector`.

Verifies the sklearn-level factory (``get_scaler``) and the Darts-level
factory (``instantiate_darts_scaler``) over the full 12-name vocabulary and
all four chain-spec forms. Includes round-trip numerical tests for the
elementwise transforms (AsinhTransform, LogTransform, FourthRootTransform).

 ``pandas`` is used only at the Darts boundary (for
``pd.Index``/``pd.DataFrame`` construction in :class:`TimeSeries`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from views_r2darts2.transformers.scaler_selector import ScalerSelector

# The full 12-name vocabulary exposed by ScalerSelector.get_scaler.
ALL_SCALER_NAMES: list[str] = [
    "StandardScaler",
    "RobustScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "PassThrough",
    "YeoJohnsonTransform",
    "LogTransform",
    "SqrtTransform",
    "AsinhTransform",
    "FourthRootTransform",
    "QuantileUniform",
    "QuantileNormal",
]

# Expected class for each name (used by the parametrized factory test).
EXPECTED_SCALER_CLASSES: dict[str, type] = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "PassThrough": FunctionTransformer,
    "YeoJohnsonTransform": PowerTransformer,
    "LogTransform": FunctionTransformer,
    "SqrtTransform": FunctionTransformer,
    "AsinhTransform": FunctionTransformer,
    "FourthRootTransform": FunctionTransformer,
    "QuantileUniform": QuantileTransformer,
    "QuantileNormal": QuantileTransformer,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_test_timeseries(
    values: np.ndarray, columns: list[str] | None = None
) -> TimeSeries:
    """Build a deterministic :class:`TimeSeries` for round-trip tests.

    Args:
        values: 2-D float32 array of shape ``(T, F)``.
        columns: Optional component names; defaults to ``["x"]`` or ``["x0",
            "x1", ...]`` based on the values' second axis.

    Returns:
        A Darts :class:`TimeSeries` with an integer time index (freq=1).
    """
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if columns is None:
        columns = ["x"] if values.shape[1] == 1 else [
            f"x{i}" for i in range(values.shape[1])
        ]
    n_time = values.shape[0]
    time_arr = np.arange(1, n_time + 1, dtype=np.int64)
    return TimeSeries.from_times_and_values(
        times=pd.Index(time_arr),
        values=values.astype(np.float32),
        columns=columns,
        freq=1,
    )


# ----------------------------------------------------------------------
# sklearn-level factory
# ----------------------------------------------------------------------


class TestScalerSelectorGetScaler:
    """Tests for :meth:`ScalerSelector.get_scaler`."""

    @pytest.mark.parametrize("name", ALL_SCALER_NAMES)
    def test_get_scaler_returns_correct_class(self, name: str) -> None:
        """Each of the 12 names must return an instance of the expected class."""
        scaler = ScalerSelector.get_scaler(name)
        assert isinstance(scaler, BaseEstimator)
        assert isinstance(scaler, EXPECTED_SCALER_CLASSES[name])

    def test_get_scaler_unknown_raises(self) -> None:
        """An unrecognized name must raise ``ValueError``."""
        with pytest.raises(ValueError, match="not recognized"):
            ScalerSelector.get_scaler("DoesNotExist")

    def test_get_scaler_with_kwargs(self) -> None:
        """``**kwargs`` must be forwarded to the sklearn constructor."""
        scaler = ScalerSelector.get_scaler("MinMaxScaler", feature_range=(-1, 1))
        assert isinstance(scaler, MinMaxScaler)
        assert scaler.feature_range == (-1, 1)

    def test_get_scaler_quantile_kwargs(self) -> None:
        """``QuantileUniform`` accepts overridden ``n_quantiles``."""
        scaler = ScalerSelector.get_scaler("QuantileUniform", n_quantiles=50)
        assert isinstance(scaler, QuantileTransformer)
        assert scaler.n_quantiles == 50
        assert scaler.output_distribution == "uniform"


# ----------------------------------------------------------------------
# Darts-level factory
# ----------------------------------------------------------------------


class TestScalerSelectorInstantiateDartsScaler:
    """Tests for :meth:`ScalerSelector.instantiate_darts_scaler`."""

    def test_instantiate_darts_scaler_none_returns_none(self) -> None:
        """``None`` config returns ``None`` (passthrough sentinel)."""
        assert ScalerSelector.instantiate_darts_scaler(None) is None

    def test_instantiate_darts_scaler_string_single(self) -> None:
        """A single-name string returns a bare :class:`Scaler`."""
        scaler = ScalerSelector.instantiate_darts_scaler("MaxAbsScaler")
        assert isinstance(scaler, Scaler)
        assert not isinstance(scaler, Pipeline)
        # The wrapped transformer must be the right sklearn class.
        assert isinstance(scaler.transformer, MaxAbsScaler)

    def test_instantiate_darts_scaler_string_chain(self) -> None:
        """``"A->B"`` string returns a :class:`Pipeline` of two scalers."""
        scaler = ScalerSelector.instantiate_darts_scaler(
            "AsinhTransform->MaxAbsScaler"
        )
        assert isinstance(scaler, Pipeline)
        assert not isinstance(scaler, Scaler)
        # Pipeline has 2 inner scalers.
        assert len(scaler._transformers) == 2

    def test_instantiate_darts_scaler_list_single(self) -> None:
        """A one-element list returns a bare :class:`Scaler` (not Pipeline)."""
        scaler = ScalerSelector.instantiate_darts_scaler(["MaxAbsScaler"])
        assert isinstance(scaler, Scaler)
        assert not isinstance(scaler, Pipeline)
        assert isinstance(scaler.transformer, MaxAbsScaler)

    def test_instantiate_darts_scaler_list_chain(self) -> None:
        """A two-element list returns a :class:`Pipeline`."""
        scaler = ScalerSelector.instantiate_darts_scaler(
            ["AsinhTransform", "MaxAbsScaler"]
        )
        assert isinstance(scaler, Pipeline)

    def test_instantiate_darts_scaler_dict_with_chain_string(self) -> None:
        """``{"chain": "A->B"}`` returns a :class:`Pipeline`."""
        scaler = ScalerSelector.instantiate_darts_scaler(
            {"chain": "AsinhTransform->MaxAbsScaler"}
        )
        assert isinstance(scaler, Pipeline)

    def test_instantiate_darts_scaler_dict_with_chain_list(self) -> None:
        """``{"chain": ["A", "B"]}`` returns a :class:`Pipeline`."""
        scaler = ScalerSelector.instantiate_darts_scaler(
            {"chain": ["AsinhTransform", "MaxAbsScaler"]}
        )
        assert isinstance(scaler, Pipeline)

    def test_instantiate_darts_scaler_dict_with_name(self) -> None:
        """``{"name": "A", "kwargs": {...}}`` returns a :class:`Scaler`."""
        scaler = ScalerSelector.instantiate_darts_scaler(
            {"name": "MinMaxScaler", "kwargs": {"feature_range": (-1, 1)}}
        )
        assert isinstance(scaler, Scaler)
        assert isinstance(scaler.transformer, MinMaxScaler)
        assert scaler.transformer.feature_range == (-1, 1)

    def test_instantiate_darts_scaler_dict_with_name_chain(self) -> None:
        """``{"name": "A->B"}`` is also accepted (chain inferred from arrow)."""
        scaler = ScalerSelector.instantiate_darts_scaler(
            {"name": "AsinhTransform->MaxAbsScaler"}
        )
        assert isinstance(scaler, Pipeline)

    def test_instantiate_darts_scaler_dict_without_name_or_chain_raises(
        self,
    ) -> None:
        """A dict without ``name`` or ``chain`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="must have a 'name' key"):
            ScalerSelector.instantiate_darts_scaler({"kwargs": {}})

    def test_instantiate_darts_scaler_empty_list_raises(self) -> None:
        """An empty chain list raises ``ValueError``."""
        with pytest.raises(ValueError, match="non-empty"):
            ScalerSelector.instantiate_darts_scaler([])

    def test_instantiate_darts_scaler_invalid_type_raises(self) -> None:
        """An unsupported type raises ``TypeError``."""
        with pytest.raises(TypeError, match="None, str, list, or dict"):
            ScalerSelector.instantiate_darts_scaler(42)  # type: ignore[arg-type]

    def test_instantiate_darts_scaler_chain_with_non_string_raises(
        self,
    ) -> None:
        """A chain containing a non-string element raises ``TypeError``."""
        with pytest.raises(TypeError, match="non-empty string"):
            ScalerSelector.instantiate_darts_scaler(["MaxAbsScaler", 42])


# ----------------------------------------------------------------------
# Round-trip numerical tests
# ----------------------------------------------------------------------


class TestScalerSelectorRoundTrip:
    """Numerical round-trip tests for the elementwise transforms."""

    @staticmethod
    def _fit_transform_inverse(
        scaler: Scaler | Pipeline, ts: TimeSeries
    ) -> np.ndarray:
        """Fit + transform + inverse_transform a single series.

        Returns the inverse-transformed values as a 2-D float32 array.
        """
        scaler.fit([ts])
        transformed = scaler.transform([ts])[0]
        inv = scaler.inverse_transform([transformed])[0]
        return inv.all_values(copy=False)[:, :, 0]

    def test_round_trip_asinh_maxabs(self) -> None:
        """``AsinhTransform->MaxAbsScaler`` round-trip recovers values (rtol=1e-5)."""
        values = np.linspace(0.1, 100.0, num=50, dtype=np.float32)[:, np.newaxis]
        ts = _build_test_timeseries(values, columns=["x"])
        scaler = ScalerSelector.instantiate_darts_scaler(
            "AsinhTransform->MaxAbsScaler"
        )
        inv = self._fit_transform_inverse(scaler, ts)
        np.testing.assert_allclose(inv, values, rtol=1e-5, atol=1e-5)

    def test_round_trip_log_transform(self) -> None:
        """``LogTransform`` (log1p) round-trip recovers non-negative values."""
        values = np.linspace(0.0, 50.0, num=50, dtype=np.float32)[:, np.newaxis]
        ts = _build_test_timeseries(values, columns=["x"])
        scaler = ScalerSelector.instantiate_darts_scaler("LogTransform")
        inv = self._fit_transform_inverse(scaler, ts)
        np.testing.assert_allclose(inv, values, rtol=1e-5, atol=1e-5)

    def test_round_trip_fourthroot_transform(self) -> None:
        """``FourthRootTransform`` round-trip recovers non-negative values."""
        values = np.linspace(0.0, 200.0, num=50, dtype=np.float32)[:, np.newaxis]
        ts = _build_test_timeseries(values, columns=["x"])
        scaler = ScalerSelector.instantiate_darts_scaler("FourthRootTransform")
        inv = self._fit_transform_inverse(scaler, ts)
        np.testing.assert_allclose(inv, values, rtol=1e-5, atol=1e-5)

    def test_round_trip_sqrt_transform(self) -> None:
        """``SqrtTransform`` round-trip recovers non-negative values."""
        values = np.linspace(0.0, 100.0, num=50, dtype=np.float32)[:, np.newaxis]
        ts = _build_test_timeseries(values, columns=["x"])
        scaler = ScalerSelector.instantiate_darts_scaler("SqrtTransform")
        inv = self._fit_transform_inverse(scaler, ts)
        np.testing.assert_allclose(inv, values, rtol=1e-5, atol=1e-5)

    def test_round_trip_maxabs_single(self) -> None:
        """Bare ``MaxAbsScaler`` round-trip on multi-column data."""
        values = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            dtype=np.float32,
        )
        ts = _build_test_timeseries(values, columns=["a", "b"])
        scaler = ScalerSelector.instantiate_darts_scaler("MaxAbsScaler")
        inv = self._fit_transform_inverse(scaler, ts)
        np.testing.assert_allclose(inv, values, rtol=1e-5, atol=1e-5)
