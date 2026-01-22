import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# from sklearn.exceptions import NotFittedError
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
    """
    Square root transform - gentler than log for zero-inflated data.
    
    Benefits for conflict data:
    - Defined at zero (unlike log)
    - Less aggressive compression than log
    - Variance stabilization for count-like data
    """
    return np.sqrt(np.maximum(x, 0))


def _inverse_sqrt_transform(x):
    """Inverse square root transform: x^2"""
    return np.square(x)


def _asinh_transform(x):
    """
    Inverse hyperbolic sine transform - best for zero-inflated continuous data.
    
    Benefits for conflict data:
    - Handles zeros gracefully (asinh(0) = 0)
    - Handles negative values if any exist
    - Similar to log for large values: asinh(x) ≈ log(2x) for large x
    - Smooth and differentiable everywhere
    - Less aggressive than log near zero
    """
    return np.arcsinh(x)


def _inverse_asinh_transform(x):
    """Inverse of asinh transform: sinh(x)"""
    return np.sinh(x)


class ScalerSelector:
    @staticmethod
    def get_scaler(scaler_name: str, **kwargs) -> BaseEstimator:
        """
        Returns a scaler instance based on the provided scaler name.

        Parameters
        ----------
        scaler_name : str
            Name of the scaler to instantiate.
            
            Available scalers:
            - StandardScaler: Zero mean, unit variance. Best for normally distributed data.
            - RobustScaler: Uses median/IQR, resistant to outliers. Good for conflict data.
            - MinMaxScaler: Scales to [0,1]. Sensitive to outliers.
            - MaxAbsScaler: Scales by max absolute value. Preserves sparsity.
            - YeoJohnsonTransform: Power transform to make data more Gaussian-like.
            - LogTransform: log(1+x). Good for positive skewed data, but undefined at 0.
            - SqrtTransform: sqrt(x). Gentler than log, defined at 0.
            - AsinhTransform: arcsinh(x). Best for zero-inflated data, handles zeros/negatives.
            - QuantileUniform: Maps to uniform distribution. Forces uniform output.
            - QuantileNormal: Maps to normal distribution. Forces Gaussian output.

        Returns
        -------
        BaseEstimator
            An instance of the specified scaler.

        Raises
        ------
        ValueError
            If the scaler name is not recognized.
        """
        scalers = {
            # Standard statistical scalers
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "MinMaxScaler": MinMaxScaler,
            "MaxAbsScaler": MaxAbsScaler,
            
            # Power transforms
            "YeoJohnsonTransform": partial(PowerTransformer, method="yeo-johnson"),
            
            # Logarithmic transform (for strictly positive data)
            "LogTransform": partial(
                FunctionTransformer,
                func=_log_transform,
                inverse_func=_inverse_log_transform,
                validate=True,
            ),
            
            # Square root transform (gentler than log, handles zeros)
            "SqrtTransform": partial(
                FunctionTransformer,
                func=_sqrt_transform,
                inverse_func=_inverse_sqrt_transform,
                validate=True,
            ),
            
            # Inverse hyperbolic sine (best for zero-inflated data)
            "AsinhTransform": partial(
                FunctionTransformer,
                func=_asinh_transform,
                inverse_func=_inverse_asinh_transform,
                validate=True,
            ),
            
            # Quantile transforms (force specific output distributions)
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
