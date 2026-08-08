import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error

from train_digital_twin import compute_physics_features, load_and_preprocess_data

print("Loading test data and ground truth...")
test_df = load_and_preprocess_data('test.csv')



print("Loading saved models and baselines...")
base_models = joblib.load('new model/base_models.pkl')
meta_model = joblib.load('new model/meta_model.pkl')
ml_features = joblib.load('new model/feature_names.pkl')
baselines = joblib.load('new model/baselines.pkl')

print("Computing physics features...")
test_df = compute_physics_features(test_df)

physics_features = [
    'eta_compressor', 'eta_turbine', 'eta_combustor',
    'residual_T2', 'residual_T4', 'work_ratio',
    'thermal_efficiency', 'PR_compressor'
]

# 1. Normalize against SAVED baselines (simulating production)
for feature in physics_features:
    # If the engine wasn't in training, default to 1.0 (or fleet avg)
    baseline_values = test_df['engine_id'].map(lambda x: baselines.get(x, {}).get(feature, 1.0))
    test_df[f'{feature}_norm'] = test_df[feature] / (baseline_values + 1e-8)

# 2. Add Temporal Features
print("Computing temporal features...")
norm_features = [f'{feat}_norm' for feat in physics_features]
rolling_features = []
for feat in norm_features:
    test_df[f'{feat}_roll5'] = test_df.groupby('engine_id')[feat].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    test_df[f'{feat}_roll5_std'] = test_df.groupby('engine_id')[feat].transform(
        lambda x: x.rolling(5, min_periods=1).std().fillna(0)
    )
    test_df[f'{feat}_delta'] = test_df.groupby('engine_id')[feat].transform(
        lambda x: x.diff().fillna(0)
    )
    rolling_features += [f'{feat}_roll5', f'{feat}_roll5_std', f'{feat}_delta']

test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.fillna(0, inplace=True)

# 3. Predict using the Meta-Model
print("Running inference...")
X_test = test_df[ml_features].copy()
OOF_test = pd.DataFrame({
    'xgb': base_models['xgb'].predict(X_test),
    'lgb': base_models['lgb'].predict(X_test),
    'cat': base_models['cat'].predict(X_test),
    'rf': base_models['rf'].predict(X_test)
})
test_df['Predicted_Health'] = meta_model.predict(OOF_test)

# Try to load ground truth if it exists
try:
    gt_df = pd.read_csv('ground_truth.csv')
    gt_df.rename(columns={"EngineID": "engine_id", "Cycle": "cycle"}, inplace=True)
    merged = pd.merge(test_df, gt_df, on=['engine_id', 'cycle'])
    r2 = r2_score(merged['OverallHealth'], merged['Predicted_Health'])
    rmse = np.sqrt(mean_squared_error(merged['OverallHealth'], merged['Predicted_Health']))

    print("============================================================")
    print("UNSEEN TEST SET EVALUATION")
    print("============================================================")
    print(f"Number of test rows: {len(merged)}")
    print(f"R² Score:   {r2:.4f}")
    class AdaptiveHealthKalmanFilter:
        def __init__(self, initial_health=1.0, process_noise=0.005, initial_uncertainty=0.01):
            self.x = np.array([[initial_health]])
            self.P = np.array([[initial_uncertainty]])
            self.Q_base = process_noise
            
        def predict(self, dt=1):
            # Scale process noise by dt to account for missing/sparse cycles
            self.P = self.P + self.Q_base * dt
            return self.x[0, 0]
            
        def update(self, measurement, measurement_variance):
            R = np.array([[measurement_variance]])
            S = self.P + R
            K = self.P / S
            y = np.array([[measurement]]) - self.x
            self.x = self.x + K * y
            self.P = (1 - K) * self.P


    print("\nApplying Kalman Filter to Unseen Test Engines...")
    merged = merged.sort_values(['engine_id', 'cycle'])
    merged['Kalman_Health'] = np.nan
    
    # Tuned uncertainty based on validation sweep
    FIXED_UNCERTAINTY = 0.2 

    for engine in merged['engine_id'].unique():
        engine_data = merged[merged['engine_id'] == engine].sort_values('cycle')
        kf = AdaptiveHealthKalmanFilter(initial_health=1.0)
        
        kalman_preds = []
        prev_cycle = None
        for _, row in engine_data.iterrows():
            if prev_cycle is None:
                dt = 1
            else:
                dt = max(1, row['cycle'] - prev_cycle)
                
            kf.predict(dt=dt)
            kf.update(row['Predicted_Health'], FIXED_UNCERTAINTY)
            kalman_preds.append(kf.x[0, 0])
            prev_cycle = row['cycle']
            
        merged.loc[engine_data.index, 'Kalman_Health'] = kalman_preds

    kalman_r2 = r2_score(merged['OverallHealth'], merged['Kalman_Health'])
    kalman_rmse = np.sqrt(mean_squared_error(merged['OverallHealth'], merged['Kalman_Health']))

    print("============================================================")
    print("KALMAN FILTER EVALUATION (UNSEEN DATA)")
    print("============================================================")
    print(f"Raw ML R²     : {r2:.4f}")
    print(f"ML + Kalman R²: {kalman_r2:.4f}")
    print(f"Raw ML RMSE   : {rmse:.4f}")
    print(f"ML + Kalman RMSE: {kalman_rmse:.4f}")
    print("============================================================\n")

    print("Sample Predictions (Showing Kalman Smoothing):")
    print(merged[['engine_id', 'cycle', 'OverallHealth', 'Predicted_Health', 'Kalman_Health']].head(15).to_string(index=False))

except FileNotFoundError:
    print("\n============================================================")
    print("No ground_truth.csv found! Generated predictions without evaluation.")
    print("============================================================")
    
    # Let's show the first engine's degradation to prove cumulative damage is working!
    first_engine = test_df['engine_id'].iloc[0]
    sample = test_df[test_df['engine_id'] == first_engine][['engine_id', 'cycle', 'Predicted_Health']]
    
    print(f"\nSample Predictions for Engine {first_engine} (showing degradation over time):")
    # Show first 5 and last 5 cycles
    print(pd.concat([sample.head(5), sample.tail(5)]).to_string(index=False))
    
    test_df[['engine_id', 'cycle', 'Predicted_Health']].to_csv('test_predictions.csv', index=False)
    print("\nFull predictions saved to test_predictions.csv")
