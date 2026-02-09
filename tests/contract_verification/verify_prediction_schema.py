
import pandas as pd
import numpy as np
import torch
from darts import TimeSeries
from darts.models.forecasting.tft_model import TFTModel

from views_r2darts2.data.handlers import _ViewsDatasetDarts
from views_r2darts2.model.forecaster import DartsForecaster

def create_mock_dataset() -> _ViewsDatasetDarts:
    """Creates a minimal but valid _ViewsDatasetDarts object."""
    time_stamps = pd.to_datetime(pd.date_range(start="2020-01-01", periods=50, freq="MS"))
    df = pd.DataFrame({
        'month_id': range(1, 51),
        'country_id': 1,
        'lr_ged_sb_best': np.random.rand(50) * 10,
        'feature_1': np.random.rand(50),
    })
    df = df.set_index(['month_id', 'country_id'])
    
    # HACK: Create a dummy class to mimic ViewsDataLoader
    class MockDataLoader:
        partition_dict = {"train": (1, 40), "test": (40, 50)}

    dataset = _ViewsDatasetDarts(source=df, targets=['lr_ged_sb_best'], broadcast_features=True)
    dataset._data_loader = MockDataLoader()
    
    # Manually assign internal properties that are usually set by the manager
    dataset._time_id = 'month_id'
    dataset._entity_id = 'country_id'

    return dataset

def assert_contract_is_valid(df: pd.DataFrame, target: str, num_samples: int):
    """Runs assertions against a prediction DataFrame to validate the contract."""
    print(f"--- Verifying contract for num_samples={num_samples} ---")
    
    # 1. Index Verification
    assert isinstance(df.index, pd.MultiIndex), f"FAIL: Index is {type(df.index)}, not MultiIndex."
    print("PASS: Index is a MultiIndex.")
    assert len(df.index.levels) == 2, f"FAIL: Index has {len(df.index.levels)} levels, not 2."
    print("PASS: Index has 2 levels.")
    assert 'month_id' in df.index.names, "FAIL: 'month_id' not in index names."
    assert 'country_id' in df.index.names, "FAIL: 'country_id' not in index names."
    print("PASS: Index names are correct.")

    # 2. Column Verification
    expected_col = f"pred_{target}"
    assert len(df.columns) == 1, f"FAIL: DataFrame has {len(df.columns)} columns, not 1."
    print("PASS: DataFrame has 1 column.")
    assert df.columns[0] == expected_col, f"FAIL: Column name is '{df.columns[0]}', not '{expected_col}'."
    print(f"PASS: Column name is '{expected_col}'.")

    # 3. Cell Value Verification
    cell_value = df.iloc[0, 0]
    assert isinstance(cell_value, list), f"FAIL: Cell value is {type(cell_value)}, not list."
    print("PASS: Cell value is a list.")
    assert len(cell_value) == num_samples, f"FAIL: List contains {len(cell_value)} samples, not {num_samples}."
    print(f"PASS: List contains {num_samples} sample(s).")
    assert all(isinstance(v, float) for v in cell_value), "FAIL: Not all values in list are floats."
    print("PASS: All values in list are floats.")

    # 4. Data Content Verification
    assert not df.isnull().values.any(), "FAIL: DataFrame contains NaN values."
    print("PASS: No NaN values found.")
    assert all(v >= 0 for v in cell_value), "FAIL: Found negative prediction values."
    print("PASS: All prediction values are non-negative.")
    
    print("--- Contract Verified ---")


def main():
    """Main execution function to verify the prediction schema."""
    print("Phase B: Empirical Validation of the Data Contract")
    
    # --- Setup ---
    dataset = create_mock_dataset()
    target = "lr_ged_sb_best"
    
    # Use a simple, fast model that supports probabilistic forecasts
    model = TFTModel(
        input_chunk_length=12,
        output_chunk_length=12,
        n_epochs=1, # Minimal training
        random_state=42,
        dropout=0.1, # Enable dropout for MC dropout
        add_relative_index=True # Required for TFTModel with no future covariates
    )

    # Instantiate the forecaster, which is the object that formats the data
    forecaster = DartsForecaster(
        dataset=dataset,
        model=model,
        partition_dict=dataset._data_loader.partition_dict,
        log_targets=True, # Ensure inverse transforms are tested
        target_scaler="MinMaxScaler" # Ensure scaler inverse transform is tested
    )
    
    # Minimal training
    print("\nTraining a minimal model...")
    forecaster.train()
    print("Training complete.")

    # --- Verification for Point Prediction ---
    print("\nGenerating point prediction (num_samples=1)...")
    point_prediction_df = forecaster.predict(sequence_number=0, output_length=10, num_samples=1)
    assert_contract_is_valid(point_prediction_df, target, num_samples=1)
    
    # --- Verification for Probabilistic Prediction ---
    print("\nGenerating probabilistic prediction (num_samples=5)...")
    prob_prediction_df = forecaster.predict(sequence_number=0, output_length=10, num_samples=5, mc_dropout=True)
    assert_contract_is_valid(prob_prediction_df, target, num_samples=5)


if __name__ == "__main__":
    main()
