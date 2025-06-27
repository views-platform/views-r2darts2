import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

class ScalerSelector:
    @staticmethod
    def get_scaler(scaler_name: str, **kwargs) -> BaseEstimator:
        """
        Returns a scaler instance based on the provided scaler name.
        
        Parameters
        ----------
        scaler_name : str
            Name of the scaler to instantiate.
        
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
            'StandardScaler': StandardScaler,
            'RobustScaler': RobustScaler,
            'MinMaxScaler': MinMaxScaler,
            'MaxAbsScaler': MaxAbsScaler,
        }
        
        if scaler_name not in scalers:
            raise ValueError(f"Scaler '{scaler_name}' is not recognized. Available scalers: {list(scalers.keys())}")
        
        return scalers[scaler_name](**kwargs)

class ConflictScaler(BaseEstimator, TransformerMixin):
    """
    Zero-inflated conflict data scaler compatible with Darts' Scaler wrapper.
    
    Handles both single and multi-feature input while preserving zeros.
    
    Parameters
    ----------
    zero_threshold : float (default=0.01)
        Absolute threshold for considering values as zeros
    """
    
    def __init__(self, zero_threshold=0.01):
        self.zero_threshold = zero_threshold
        self.scaler_ = StandardScaler()
        self.zero_mask_ = None
        self.n_features_in_ = None
        self.is_fitted = False

    def fit(self, X, y=None):
        # Ensure 2D input and convert to numpy array
        X = check_array(X, ensure_2d=True)
        
        # Store feature count for validation
        self.n_features_in_ = X.shape[1]
        
        # Create global zero mask
        self.zero_mask_ = np.abs(X) <= self.zero_threshold
        
        # Always fit scaler to ensure it's available for inverse_transform
        if np.any(~self.zero_mask_):
            # Normal case: fit on non-zero data
            non_zero_data = X[~self.zero_mask_].reshape(-1, self.n_features_in_)
        else:
            # Edge case: all zeros, fit on dummy data to maintain scaler structure
            non_zero_data = np.zeros((1, self.n_features_in_))
        
        self.scaler_.fit(non_zero_data)
        self.is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
            
        X_tr = X.copy()
        current_zero_mask = np.abs(X_tr) <= self.zero_threshold
        
        # Apply transformation only to non-zero values
        if np.any(~current_zero_mask):
            non_zero_data = X_tr[~current_zero_mask].reshape(-1, self.n_features_in_)
            transformed = self.scaler_.transform(non_zero_data)
            X_tr[~current_zero_mask] = transformed.ravel()
            
        return X_tr

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True)
        
        X_inv = X.copy()
        current_zero_mask = np.abs(X_inv) <= self.zero_threshold
        
        # Apply inverse transformation only to non-zero values
        if np.any(~current_zero_mask):
            non_zero_data = X_inv[~current_zero_mask].reshape(-1, self.n_features_in_)
            inverted = self.scaler_.inverse_transform(non_zero_data)
            X_inv[~current_zero_mask] = inverted.ravel()
        
        # Restore original zeros from training data
        X_inv[self.zero_mask_] = 0
        return X_inv

    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def __getstate__(self):
        return {
            'zero_threshold': self.zero_threshold,
            'scaler_': self.scaler_,
            'zero_mask_': self.zero_mask_,
            'n_features_in_': self.n_features_in_,
            'is_fitted': self.is_fitted
        }

    def __setstate__(self, state):
        self.__dict__.update(state)