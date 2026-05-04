import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from typing import Any
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    PowerTransformer,
    FunctionTransformer,
    QuantileTransformer,
)
from functools import partial


def _log_transform(x):
    """Log transform function: log(1 + x)"""
    return np.log1p(x)


def _inverse_log_transform(x):
    """Inverse log transform function: exp(x) - 1"""
    return np.expm1(x)


def _sqrt_transform(x):
    """Square root transform: sqrt(max(x, 0))"""
    return np.sqrt(np.maximum(x, 0))


def _inverse_sqrt_transform(x):
    """Inverse square root transform: x^2"""
    return np.square(x)


def _asinh_transform(x):
    """Inverse hyperbolic sine transform."""
    return np.arcsinh(x)


def _inverse_asinh_transform(x):
    """Inverse of asinh transform: sinh(x)"""
    return np.sinh(x)


def _fourthroot_transform(x):
    """Fourth-root power transform: (1+x)^0.25 - 1.

    Same compression range as asinh (~[0,10] for [0,10000]) but with a
    polynomial (quartic) inverse instead of exponential (sinh).

    Comparison at model overshoot of 15 in transformed space:
        sinh(15)              = 1,634,508  (exponential)
        (1+15)^4 - 1         = 65,535     (polynomial)

    25x less explosive. For in-range values the behavior is similar:
        asinh(100)  = 5.3     fourthroot(100) = 2.16
        asinh(1000) = 7.6     fourthroot(1000) = 4.62
        asinh(5000) = 9.2     fourthroot(5000) = 7.41
    """
    return np.power(1.0 + np.maximum(x, 0.0), 0.25) - 1.0


def _inverse_fourthroot_transform(x):
    """Inverse fourth-root: (1+x)^4 - 1. Polynomial, not exponential."""
    return np.power(1.0 + np.maximum(x, 0.0), 4.0) - 1.0


class ScalerSelector:
    """
    Factory for selecting and instantiating data scalers.
    """
    @staticmethod
    def get_scaler(scaler_name: str, **kwargs) -> BaseEstimator:
        """
        Returns a scaler instance based on the provided scaler name.
        """
        scalers = {
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "MinMaxScaler": MinMaxScaler,
            "MaxAbsScaler": MaxAbsScaler,
            "PassThrough": partial(FunctionTransformer, func=None, inverse_func=None, validate=False),
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
                f"Scaler '{scaler_name}' is not recognized. Available scalers: {list(scalers.keys())}"
            )

        return scalers[scaler_name](**kwargs)

    @staticmethod
    def get_chained_scaler(scaler_chain: str) -> Any:
        """
        Create a Darts Pipeline from a chain specification string.
        """
        from darts.dataprocessing import Pipeline

        scaler_names = [s.strip() for s in scaler_chain.split("->")]
        if len(scaler_names) < 2:
            raise ValueError(
                f"Chain specification '{scaler_chain}' must contain at least 2 scalers."
            )

        darts_scalers = [
            Scaler(ScalerSelector.get_scaler(name), global_fit=True)
            for name in scaler_names
        ]
        return Pipeline(darts_scalers)

    @staticmethod
    def is_chain_spec(scaler_name: str) -> bool:
        """Check if a scaler name is a chain specification."""
        return "->" in scaler_name

    @staticmethod
    def get_scaler_or_chain(scaler_spec: str, **kwargs) -> Any:
        """
        Get either a single sklearn scaler or a Darts Pipeline.
        """
        if ScalerSelector.is_chain_spec(scaler_spec):
            return ScalerSelector.get_chained_scaler(scaler_spec)
        return ScalerSelector.get_scaler(scaler_spec, **kwargs)

    @staticmethod
    def instantiate_darts_scaler(scaler_cfg):
        """
        Instantiate a Darts Scaler or Pipeline from a flexible config format.

        All four chain-spec forms below produce structurally identical objects:
          - "A->B"                — string with arrow
          - ["A", "B"]            — list
          - {"chain": "A->B"}     — dict with string chain
          - {"chain": ["A", "B"]} — dict with list chain
        Single-element forms (`"A"`, `["A"]`, `{"chain": ["A"]}`) all return a
        bare `Scaler`, not a one-element `Pipeline`. Empty lists / empty chain
        strings raise `ValueError` instead of silently producing empty Pipelines.

        Accepts:
          - None → returns None
          - String: 'StandardScaler' or 'AsinhTransform->StandardScaler' (chained)
          - List: ['AsinhTransform', 'StandardScaler'] (chained)
          - Dict: {'name': <str>, 'kwargs': <dict>}
          - Dict with chain: {'chain': ['AsinhTransform', 'StandardScaler']}
            or {'chain': 'AsinhTransform->StandardScaler'}

        Returns:
          Darts Scaler (single) or Pipeline (chained) or None.
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
                    f"'chain' must be a string or list, got {type(chain_list).__name__}"
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
            return Scaler(ScalerSelector.get_scaler(name, **kwargs), global_fit=True)

        raise TypeError(
            f"Scaler config must be None, str, list, or dict. Got {type(scaler_cfg).__name__}."
        )

    @staticmethod
    def _build_chain_or_single(scaler_names: list):
        """
        Sole chain-construction helper used by `instantiate_darts_scaler`.

        Collapses the previously-divergent list / dict-chain-list / dict-chain-str
        code paths (flagged by Copilot on PR #10 and tracked as C-03) into a
        single definition. All chain specs — regardless of input form — go
        through here, so adding a new scaler step or changing chain semantics
        only requires one edit.

        Single-element chains return a bare `Scaler` to match the legacy
        single-scaler path; multi-element chains return a `darts.Pipeline`.
        Empty lists and non-string elements raise instead of silently producing
        malformed pipelines.
        """
        from darts.dataprocessing import Pipeline

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
            return Scaler(ScalerSelector.get_scaler(scaler_names[0]), global_fit=True)
        darts_scalers = [
            Scaler(ScalerSelector.get_scaler(name), global_fit=True)
            for name in scaler_names
        ]
        return Pipeline(darts_scalers)
