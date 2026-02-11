import torch
import numpy as np
from darts import TimeSeries
from darts.models import NLinearModel
from views_r2darts2.utils.loss import LossSelector
import logging

# --- Suppress excessive logging ---
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# --- Synthetic Data Generation ---
def generate_synthetic_data(n_samples=1000, zero_inflation_prob=0.9, shape=1.5, scale=2.0):
    is_zero = np.random.rand(n_samples) < zero_inflation_prob
    non_zero_values = np.random.gamma(shape, scale, n_samples)
    raw_data = np.where(is_zero, 0, non_zero_values)
    return torch.from_numpy(raw_data.astype(np.float32)).unsqueeze(-1)

# --- Training Function ---
def train(loss_function_name, loss_params):
    """
    Initializes and runs a single, fast training run for verification.
    """
    series_raw = generate_synthetic_data()
    series_transformed = torch.log1p(series_raw)
    ts = TimeSeries.from_values(series_transformed.numpy())
    train_ts, val_ts = ts.split_before(int(0.7 * len(ts)))

    # The ModelCatalog will be invoked here, printing the summary
    loss_fn_instance = LossSelector.get_loss_function(loss_function_name, **loss_params)
    
    model = NLinearModel(
        input_chunk_length=24,
        output_chunk_length=12,
        n_epochs=1, # Minimal epochs for verification
        optimizer_kwargs={"lr": 1e-4}, # A dummy LR
        loss_fn=loss_fn_instance,
        pl_trainer_kwargs={"enable_progress_bar": False, "logger": False}, # Explicitly disable logger
        random_state=42
    )

    print(f"Starting verification run for {loss_function_name}...")
    # We are calling fit simply to trigger the ModelCatalog and its logging
    # The actual training result is not important for this verification
    model.fit(series=train_ts, val_series=val_ts, verbose=False)
    print("Verification run finished.")

# --- Main Verification Execution ---
def main():
    """
    Main function to run the verification tests locally.
    """
    print(f"\n{'='*20} Starting Hyperparameter Pipeline Verification {'='*20}\n")

    # --- Test 1: AsymmetricQuantileLoss ---
    print("--- Verifying: AsymmetricQuantileLoss ---")
    try:
        train("AsymmetricQuantileLoss", {'tau': 0.99})
    except Exception as e:
        print(f"ERROR during AsymmetricQuantileLoss verification: {e}")
    print("-" * 67)

    # --- Test 2: TweedieLoss ---
    print("\n--- Verifying: TweedieLoss ---")
    try:
        train("TweedieLoss", {'p': 1.1})
    except Exception as e:
        print(f"ERROR during TweedieLoss verification: {e}")
    print("-" * 67)

    # --- Test 3: ShrinkageLoss ---
    print("\n--- Verifying: ShrinkageLoss ---")
    try:
        train("ShrinkageLoss", {'a': 50.0})
    except Exception as e:
        print(f"ERROR during ShrinkageLoss verification: {e}")
    
    print(f"\n{'='*25} Verification Complete {'='*25}\n")
    print("Please check the 'Model Configuration Summary' printed for each run above.")


if __name__ == '__main__':
    main()
