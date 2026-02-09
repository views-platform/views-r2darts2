
import pandas as pd
import numpy as np
from views_evaluation.evaluation.evaluation_manager import EvaluationManager # Assuming this import path is correct

# --- Helper function to create valid mock data ---
def generate_mock_data(target_name: str, location_id_name: str, num_prediction_sequences: int = 1, num_samples: int = 1):
    """Creates mock actuals and point predictions in the required schema."""

    pred_col_name = f"pred_{target_name}"

    # 1. Actuals DataFrame
    actuals_index = pd.MultiIndex.from_product(
        [range(500, 500 + num_prediction_sequences + 3), [10, 20]],
        names=['month_id', location_id_name]
    )
    actuals = pd.DataFrame(
        {target_name: np.random.randint(0, 50, size=len(actuals_index))},
        index=actuals_index
    )

    # 2. Predictions List (2 rolling sequences of 3 steps each)
    predictions_list = []
    # First sequence
    preds_1_index = pd.MultiIndex.from_product(
        [range(500, 500 + 3), [10, 20]],
        names=['month_id', location_id_name]
    )
    preds_1 = pd.DataFrame(
        # Note the required list format for point predictions
        {pred_col_name: [[val] * num_samples for val in np.random.rand(len(preds_1_index)) * 50]},
        index=preds_1_index
    )
    predictions_list.append(preds_1)

    # Second sequence if needed
    if num_prediction_sequences > 1:
        preds_2_index = pd.MultiIndex.from_product(
            [range(501, 501 + 3), [10, 20]],
            names=['month_id', location_id_name]
        )
        preds_2 = pd.DataFrame(
            {pred_col_name: [[val] * num_samples for val in np.random.rand(len(preds_2_index)) * 50]},
            index=preds_2_index
        )
        predictions_list.append(preds_2)


    return actuals, predictions_list

# --- Test Harness ---
def run_evaluation_with_manager(actuals: pd.DataFrame, predictions: list[pd.DataFrame], target: str, description: str):
    """
    Attempts to run evaluation with the EvaluationManager and reports success/failure.
    """
    print(f"\n--- Testing Scenario: {description} ---")
    try:
        manager = EvaluationManager(metrics_list=['RMSLE', 'CRPS'])
        eval_config = {'steps': [1, 2, 3]} # Use minimal steps
        results = manager.evaluate(
            actual=actuals,
            predictions=predictions,
            target=target,
            config=eval_config
        )
        print("SUCCESS: EvaluationManager ran without error.")
        return results
    except Exception as e:
        print(f"FAILURE: EvaluationManager raised an error: {e}")
        return None

def main():
    target_name = "lr_ged_sb_best"
    location_id_name = "country_id"
    
    # Generate a valid baseline set of data
    valid_actuals, valid_predictions = generate_mock_data(target_name, location_id_name, num_prediction_sequences=2)
    
    # --- Action C.2: Test Index Name Assumption (Malformed Index Level Name) ---
    malformed_index_actuals = valid_actuals.copy()
    malformed_index_actuals.index.names = ['month_id', 'invalid_location_id'] # Violate index name
    run_evaluation_with_manager(malformed_index_actuals, valid_predictions, target_name, "Malformed Actuals Index Name")

    # --- Action C.3: Test Column Name Assumption (Malformed Prediction Column Name) ---
    malformed_col_preds = [df.copy() for df in valid_predictions]
    malformed_col_preds[0].rename(columns={f"pred_{target_name}": "wrong_col_name"}, inplace=True) # Violate column name
    run_evaluation_with_manager(valid_actuals, malformed_col_preds, target_name, "Malformed Prediction Column Name")

    # --- Action C.4: Test Prediction List Order Assumption (Shuffled Predictions) ---
    shuffled_predictions = valid_predictions[:] # Create a copy
    np.random.shuffle(shuffled_predictions) # Shuffle the list
    original_results = run_evaluation_with_manager(valid_actuals, valid_predictions, target_name, "Valid Predictions (Baseline for Order Check)")
    shuffled_results = run_evaluation_with_manager(valid_actuals, shuffled_predictions, target_name, "Shuffled Predictions List")
    
    if original_results and shuffled_results:
        # Compare a key metric if both succeeded
        original_rmsle = original_results['step'][1].loc['step01', 'RMSLE']
        shuffled_rmsle = shuffled_results['step'][1].loc['step01', 'RMSLE']
        print(f"  Original RMSLE for step01: {original_rmsle}")
        print(f"  Shuffled RMSLE for step01: {shuffled_rmsle}")
        if not np.isclose(original_rmsle, shuffled_rmsle): # Use np.isclose for float comparison
            print("  OBSERVATION: RMSLE changed due to shuffling. Order matters.")
        else:
            print("  OBSERVATION: RMSLE did NOT change due to shuffling. Order might not matter for this metric or config.")
    
    # --- Add a test for Non-Numeric Pred Values (e.g. string instead of list[float]) ---
    non_numeric_preds = [df.copy() for df in valid_predictions]
    non_numeric_preds[0][f"pred_{target_name}"].iloc[0] = "not a number"
    run_evaluation_with_manager(valid_actuals, non_numeric_preds, target_name, "Non-Numeric Prediction Values")

if __name__ == "__main__":
    main()
