import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import server

def main():
    print("Loading models...")
    server._load_all_models()
    
    print("Verifying predictions...")
    train_df = pd.read_csv("dataset aerothon/train.csv")
    test_df = pd.read_csv("dataset aerothon/test.csv")
    
    # Combine train and test to provide full history
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = combined_df.rename(columns=server.COL_RENAME)
    combined_df["engine_id"] = combined_df["engine_id"].astype(str)
    
    preds_combined = server.compute_features_and_predict(combined_df, mode="batch")
    
    # Filter back to only the test rows
    # We can match on test.csv's EngineID and Cycle
    test_keys = test_df[["EngineID", "Cycle"]].rename(columns={"EngineID": "engine_id", "Cycle": "cycle"})
    test_keys["engine_id"] = test_keys["engine_id"].astype(str)
    
    preds_test = test_keys.merge(preds_combined, on=["engine_id", "cycle"], how="left")
    
    scored_df = pd.read_csv("results/test_predictions_scored.csv")
    scored_df["engine_id"] = scored_df["engine_id"].astype(str)
    
    merged = preds_test.merge(scored_df, on=["engine_id", "cycle"])
    
    cols_to_check = [
        ("HI_compressor", "pred_HI_compressor"),
        ("HI_turbine", "pred_HI_turbine"),
        ("HI_combustor", "pred_HI_combustor"),
        ("HI_overall", "pred_HI_overall"),
        ("thrust_N", "pred_thrust"),
        ("TSFC", "pred_tsfc_derived")
    ]
    
    errors = 0
    for idx, row in merged.iterrows():
        for my_col, truth_col in cols_to_check:
            my_val = row[my_col]
            truth_val = row[truth_col]
            if not np.isclose(my_val, truth_val, atol=1e-4, rtol=1e-4):
                print(f"Mismatch at index {idx} ({row['engine_id']}, {row['cycle']}) for {my_col}: got {my_val}, expected {truth_val}")
                errors += 1
                
    if errors == 0:
        print("✓ Predictions match test_predictions_scored.csv perfectly when full history is provided!")
    else:
        print(f"✗ Found {errors} mismatches.")
        
    print("\nVerifying fleet_fallback logic for unseen engine...")
    unseen_row = test_df.iloc[[0]].copy().rename(columns=server.COL_RENAME)
    unseen_row["engine_id"] = "UNSEEN_9999"
    
    unseen_preds = server.compute_features_and_predict(unseen_row, mode="batch")
    source = unseen_preds["baseline_source"].iloc[0]
    
    if source == "fleet_fallback":
        print(f"✓ Unseen engine correctly fell back to: {source}")
    else:
        print(f"✗ Unseen engine got unexpected baseline source: {source}")

if __name__ == "__main__":
    main()
