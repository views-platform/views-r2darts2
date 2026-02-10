import pandas as pd
import numpy as np

# Simulate the error condition
target = "lr_ged_sb_dep"
pred_column_name = f"pred_{target}"

# Create a dataframe like the one produced by DartsForecaster
df = pd.DataFrame({"pred_lr_ged_sb_dep": [1, 2, 3]})

print(f"Target: {repr(target)}")
print(f"Pred Col Name: {repr(pred_column_name)}")
print(f"DF Columns: {repr(list(df.columns))}")

# Check inclusion
print(f"Is in columns? {pred_column_name in df.columns}")

# Now simulate the LIST normalization bug
target_list = ["lr_ged_sb_dep"]
pred_col_from_list = f"pred_{target_list}"
print(f"Pred Col from List: {repr(pred_col_from_list)}")
print(f"Is in columns? {pred_col_from_list in df.columns}")

# Now check for the hidden character theory
target_with_space = "lr_ged_sb_dep "
pred_col_space = f"pred_{target_with_space}"
print(f"Pred Col with space: {repr(pred_col_space)}")
print(f"Is in columns? {pred_col_space in df.columns}")