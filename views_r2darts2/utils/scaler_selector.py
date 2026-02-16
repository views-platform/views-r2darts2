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
