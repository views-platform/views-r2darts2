"""Factory for sklearn scalers and Darts ``Scaler`` / ``Pipeline`` objects.

This module is pandas-free. It exposes:

    * ``ScalerSelector.get_scaler(name, **kwargs)`` — returns a raw sklearn
      transformer.
    * ``ScalerSelector.instantiate_darts_scaler(cfg)`` — returns a Darts
      ``Scaler`` (single) or ``Pipeline`` (chained), or ``None``.

The legacy ``get_chained_scaler``, ``get_scaler_or_chain``, and ``is_chain_spec``
helpers have been removed — they were dead code (never called by the production
path; ``instantiate_darts_scaler`` is the sole entry point).

Google Python Style.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
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

from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing import Pipeline


# ----------------------------------------------------------------------
# Elementwise transform functions (used by FunctionTransformer)
# ----------------------------------------------------------------------


def _log_transform(x: NDArray) -> NDArray:
    """Elementwise ``log(1 + x)``."""
    return np.log1p(x)


def _inverse_log_transform(x: NDArray) -> NDArray:
    """Inverse of :func:`_log_transform`: ``exp(x) - 1``."""
    return np.expm1(x)


def _sqrt_transform(x: NDArray) -> NDArray:
    """Elementwise ``sqrt(max(x, 0))``."""
    return np.sqrt(np.maximum(x, 0))


def _inverse_sqrt_transform(x: NDArray) -> NDArray:
    """Inverse of :func:`_sqrt_transform`: ``x**2`` (valid for ``x >= 0``)."""
    return np.square(x)


def _asinh_transform(x: NDArray) -> NDArray:
    """Elementwise ``arcsinh(x)``."""
    return np.arcsinh(x)


def _inverse_asinh_transform(x: NDArray) -> NDArray:
    """Inverse of :func:`_asinh_transform`: ``sinh(x)``."""
    return np.sinh(x)


def _fourthroot_transform(x: NDArray) -> NDArray:
    """Elementwise ``(1 + max(x, 0))^0.25 - 1``.

    Same compression range as asinh (~[0,10] for [0,10000]) but with a
    polynomial (quartic) inverse instead of exponential (sinh). 25× less
    explosive than asinh on model overshoot.
    """
    return np.power(1.0 + np.maximum(x, 0.0), 0.25) - 1.0


def _inverse_fourthroot_transform(x: NDArray) -> NDArray:
    """Inverse of :func:`_fourthroot_transform`: ``(1 + max(x, 0))^4 - 1``."""
    return np.power(1.0 + np.maximum(x, 0.0), 4.0) - 1.0


# ----------------------------------------------------------------------
# ScalerSelector
# ----------------------------------------------------------------------


class ScalerSelector:
    """Factory for selecting and instantiating data scalers."""

    # ------------------------------------------------------------------
    # sklearn-level factory
    # ------------------------------------------------------------------

    @staticmethod
    def get_scaler(scaler_name: str, **kwargs: Any) -> BaseEstimator:
        """Return an sklearn scaler instance by name.

        Args:
            scaler_name: Name of the scaler. See ``_SCALERS`` below for the
                supported vocabulary.
            **kwargs: Forwarded to the scaler constructor.

        Raises:
            ValueError: ``scaler_name`` is not in the vocabulary.
        """
        scalers = {
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "MinMaxScaler": MinMaxScaler,
            "MaxAbsScaler": MaxAbsScaler,
            "PassThrough": partial(
                FunctionTransformer, func=None, inverse_func=None, validate=False
            ),
            "YeoJohnsonTransform": partial(PowerTransformer, method="yeo-johnson"),
            "LogTransform": partial(
                FunctionTransformer,
                func=_log_transform,
                inverse_func=_inverse_log_transform,
                validate=True,
            ),
            "SqrtTransform": partial(
                FunctionTransformer,
                func=_sqrt_transform,
                inverse_func=_inverse_sqrt_transform,
                validate=True,
            ),
            "AsinhTransform": partial(
                FunctionTransformer,
                func=_asinh_transform,
                inverse_func=_inverse_asinh_transform,
                validate=True,
            ),
            "FourthRootTransform": partial(
                FunctionTransformer,
                func=_fourthroot_transform,
                inverse_func=_inverse_fourthroot_transform,
                validate=True,
            ),
            "QuantileUniform": partial(
                QuantileTransformer,
                output_distribution="uniform",
                n_quantiles=1000,
                random_state=42,
            ),
            "QuantileNormal": partial(
                QuantileTransformer,
                output_distribution="normal",
                n_quantiles=1000,
                random_state=42,
            ),
        }

        if scaler_name not in scalers:
            raise ValueError(
                f"Scaler '{scaler_name}' is not recognized. "
                f"Available scalers: {list(scalers.keys())}"
            )
        return scalers[scaler_name](**kwargs)

    # ------------------------------------------------------------------
    # Darts-level factory
    # ------------------------------------------------------------------

    @staticmethod
    def instantiate_darts_scaler(scaler_cfg: Any) -> Scaler | Pipeline | None:
        """Instantiate a Darts ``Scaler`` or ``Pipeline`` from a flexible config.

        All four chain-spec forms below produce structurally identical objects:
            - ``"A->B"``               — string with arrow
            - ``["A", "B"]``           — list
            - ``{"chain": "A->B"}``    — dict with string chain
            - ``{"chain": ["A", "B"]}``— dict with list chain

        Single-element forms (``"A"``, ``["A"]``, ``{"chain": ["A"]}``) all
        return a bare ``Scaler``, not a one-element ``Pipeline``. Empty lists
        and empty chain strings raise ``ValueError``.

        Args:
            scaler_cfg: ``None``, ``str``, ``list[str]``, or ``dict`` with a
                ``"chain"`` or ``"name"`` key.

        Returns:
            A Darts :class:`Scaler`, :class:`Pipeline`, or ``None``.

        Raises:
            TypeError: ``scaler_cfg`` is not a supported type.
            ValueError: A chain list is empty or contains a non-string element.
        """
        if scaler_cfg is None:
            return None

        if isinstance(scaler_cfg, str):
            if "->" in scaler_cfg:
                return ScalerSelector._build_chain_or_single(
                    [s.strip() for s in scaler_cfg.split("->")]
                )
            return Scaler(ScalerSelector.get_scaler(scaler_cfg), global_fit=True)

        if isinstance(scaler_cfg, list):
            return ScalerSelector._build_chain_or_single(scaler_cfg)

        if isinstance(scaler_cfg, dict):
            if "chain" in scaler_cfg:
                chain_list = scaler_cfg["chain"]
                if isinstance(chain_list, str):
                    return ScalerSelector._build_chain_or_single(
                        [s.strip() for s in chain_list.split("->")]
                    )
                if isinstance(chain_list, list):
                    return ScalerSelector._build_chain_or_single(chain_list)
                raise TypeError(
                    f"'chain' must be a string or list, "
                    f"got {type(chain_list).__name__}"
                )
            name = scaler_cfg.get("name")
            kwargs = scaler_cfg.get("kwargs", {})
            if name is None:
                raise ValueError(
                    "Scaler config dict must have a 'name' key or a 'chain' key."
                )
            if "->" in name:
                return ScalerSelector._build_chain_or_single(
                    [s.strip() for s in name.split("->")]
                )
            return Scaler(
                ScalerSelector.get_scaler(name, **kwargs), global_fit=True
            )

        raise TypeError(
            f"Scaler config must be None, str, list, or dict. "
            f"Got {type(scaler_cfg).__name__}."
        )

    @staticmethod
    def _build_chain_or_single(scaler_names: list[str]) -> Scaler | Pipeline:
        """Sole chain-construction helper.

        Single-element chains return a bare ``Scaler`` to match the legacy
        single-scaler path; multi-element chains return a Darts ``Pipeline``.

        Args:
            scaler_names: Non-empty list of scaler name strings.

        Raises:
            ValueError: The list is empty.
            TypeError: An element is not a non-empty string.
        """
        if not isinstance(scaler_names, list) or len(scaler_names) == 0:
            raise ValueError(
                "Scaler chain must be a non-empty list of scaler name strings."
            )
        if not all(isinstance(name, str) and name for name in scaler_names):
            raise TypeError(
                "Scaler chain must contain only non-empty string scaler names, "
                f"got {scaler_names!r}."
            )
        if len(scaler_names) == 1:
            return Scaler(
                ScalerSelector.get_scaler(scaler_names[0]), global_fit=True
            )
        darts_scalers = [
            Scaler(ScalerSelector.get_scaler(name), global_fit=True)
            for name in scaler_names
        ]
        return Pipeline(darts_scalers)
