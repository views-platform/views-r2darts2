import numpy as np


def calculate_mean_pred_true_ratio(y_true, y_pred):
    """Calculates the mean of the ratio of prediction to true value, ignoring zeros."""
    # Ensure no division by zero
    safe_true = np.where(y_true == 0, 1, y_true)
    safe_pred = np.where(y_true == 0, 1, y_pred)

    ratios = safe_pred / safe_true

    # Only consider ratios where the true value was not zero
    non_zero_mask = y_true > 0
    if np.any(non_zero_mask):
        return np.mean(ratios[non_zero_mask])
    return 1.0  # Return neutral value if no non-zero true values exist
