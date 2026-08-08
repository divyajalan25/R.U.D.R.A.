import pandas as pd
import numpy as np
import joblib
import os
import sys

# Import physics pipeline from training script to guarantee 100% parity
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_digital_twin import compute_physics_features, load_and_preprocess_data

# 1D Kalman Filter class with cycle-gap scaling (dt)
class AdaptiveHealthKalmanFilter:
    def __init__(self, initial_health=1.0, process_noise=0.005, initial_uncertainty=0.01):
        self.x = np.array([[initial_health]])
        self.P = np.array([[initial_uncertainty]])
        self.Q_base = process_noise
        
    def predict(self, dt=1):
        self.P = self.P + self.Q_base * dt
        return self.x[0, 0]
        
    def update(self, measurement, measurement_variance):
        R = np.array([[measurement_variance]])
        S = self.P + R
        K = self.P / S
        y = np.array([[measurement]]) - self.x
        self.x = self.x + K * y
        self.P = (1 - K) * self.P

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <input_test_csv> <output_predictions_csv>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading and preprocessing test data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Rename basic columns to match the internal names
    rename_map = {
        'EngineID': 'engine_id',
        'Cycle': 'cycle',
        'FuelFlow_kg_s': 'Fuel_Flow_kg_s'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Store original row index to restore output order later, and sort chronologically
    df['orig_index'] = df.index
    df = df.sort_values(['engine_id', 'cycle'])
    
    # Load Models & Features
    print("Loading models and baselines...")
    models_dir = 'models'
    ml_features = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    baselines = joblib.load(os.path.join(models_dir, 'baselines.pkl'))
    
    ensemble_models = {}
    targets_list = ['HI_compressor', 'HI_turbine', 'HI_combustor', 'HI_overall', 'thrust_N', 'RUL']
    for target in targets_list:
        base_path = os.path.join(models_dir, f'base_models_{target}.pkl')
        meta_path = os.path.join(models_dir, f'meta_model_{target}.pkl')
        if os.path.exists(base_path) and os.path.exists(meta_path):
            ensemble_models[target] = {
                'base_models': joblib.load(base_path),
                'meta_model': joblib.load(meta_path)
            }
            print(f"Loaded new target-specific {target} models.")
            
    if 'HI_overall' not in ensemble_models:
        old_base_path = os.path.join(models_dir, 'base_models.pkl')
        old_meta_path = os.path.join(models_dir, 'meta_model.pkl')
        if os.path.exists(old_base_path) and os.path.exists(old_meta_path):
            ensemble_models['HI_overall'] = {
                'base_models': joblib.load(old_base_path),
                'meta_model': joblib.load(old_meta_path)
            }
            print("Loaded default base_models.pkl fallback.")
    
    # Run physics feature engineering
    df = compute_physics_features(df)
    
    physics_features = [
        'eta_compressor', 'eta_turbine', 'eta_combustor',
        'residual_T2', 'residual_T4', 'work_ratio',
        'thermal_efficiency', 'PR_compressor'
    ]
    
    # Normalize features using dynamic or training baselines
    dynamic_baselines = {}
    for eid in df['engine_id'].unique():
        engine_mask = df['engine_id'] == eid
        early_rows = df.loc[engine_mask & (df['cycle'] <= 10)]
        if len(early_rows) > 0:
            dynamic_baselines[eid] = early_rows[physics_features].mean().to_dict()
        else:
            str_eid = str(eid)
            int_eid = int(eid) if isinstance(eid, (int, float)) or (isinstance(eid, str) and eid.isdigit()) else None
            
            if baselines is not None and str_eid in baselines:
                dynamic_baselines[eid] = baselines[str_eid]
            elif baselines is not None and int_eid in baselines:
                dynamic_baselines[eid] = baselines[int_eid]
            else:
                dynamic_baselines[eid] = {feat: 1.0 for feat in physics_features}
                
    for feature in physics_features:
        baseline_values = df['engine_id'].map(lambda x: dynamic_baselines.get(x, {}).get(feature, 1.0))
        df[f'{feature}_norm'] = df[feature] / (baseline_values + 1e-8)
        
    # Compute continuous-time EMA features
    lambda_val = 0.2
    norm_features = [f'{feat}_norm' for feat in physics_features]
    ema_cols = [f'{feat}_ema' for feat in norm_features]
    delta_cols = [f'{feat}_delta' for feat in norm_features]
    
    # Initialize EMA states
    ema_states = {}
    last_cycle = {}
    
    # Create empty arrays for fast assignment
    ema_data = {col: np.zeros(len(df)) for col in ema_cols}
    
    # Iterate through rows to maintain state across time gaps
    for i, (idx, row) in enumerate(df.iterrows()):
        eid = row['engine_id']
        current_cycle = row['cycle']
        
        if eid not in ema_states:
            ema_states[eid] = {feat: row[feat] for feat in norm_features}
            last_cycle[eid] = current_cycle
            dt = 1
        else:
            dt = current_cycle - last_cycle[eid]
            if dt <= 0:
                dt = 1
            last_cycle[eid] = current_cycle
            
        # Continuous-time EMA formula: alpha_eff = 1 - (1 - lambda)^dt
        alpha_eff = 1.0 - (1.0 - lambda_val)**dt
        
        for feat, ema_col in zip(norm_features, ema_cols):
            old_ema = ema_states[eid][feat]
            new_ema = old_ema * (1 - alpha_eff) + row[feat] * alpha_eff
            ema_states[eid][feat] = new_ema
            ema_data[ema_col][i] = new_ema
            
    for ema_col in ema_cols:
        df[ema_col] = ema_data[ema_col]
        
    for feat, ema_col, delta_col in zip(norm_features, ema_cols, delta_cols):
        df[delta_col] = df[feat] - df[ema_col]
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Run Ensemble ML Inference
    print("Running Ensemble ML predictions...")
    X_inference = df[ml_features].copy()
    
    for target in targets_list:
        if target in ensemble_models:
            models = ensemble_models[target]
            base_models = models['base_models']
            meta_model = models['meta_model']
            
            OOF_preds = pd.DataFrame({
                'xgb': base_models['xgb'].predict(X_inference),
                'lgb': base_models['lgb'].predict(X_inference),
                'cat': base_models['cat'].predict(X_inference),
                'rf': base_models['rf'].predict(X_inference)
            })
            df[f'{target}_Raw'] = meta_model.predict(OOF_preds)
        else:
            df[f'{target}_Raw'] = np.nan
            
    # Apply Tuned Kalman Filter
    print("Applying tuned Kalman Filter to smooth predictions...")
    df = df.sort_values(['engine_id', 'cycle'])
    
    FIXED_UNCERTAINTY = 0.2
    
    for target in targets_list:
        if target in ensemble_models and target.startswith('HI_'):
            df[f'{target}_Kalman'] = np.nan
            
    for engine in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine].sort_values('cycle')
        
        # Initialize KFs per target. Health indices should strictly start at 1.0!
        kfs = {}
        for target in targets_list:
            if target in ensemble_models and target.startswith('HI_'):
                kfs[target] = AdaptiveHealthKalmanFilter(initial_health=1.0)
                
        kalman_preds = {t: [] for t in kfs.keys()}
        prev_cycle = None
        for _, row in engine_data.iterrows():
            if prev_cycle is None:
                dt = 1
            else:
                dt = max(1, row['cycle'] - prev_cycle)
                
            for t, kf in kfs.items():
                kf.predict(dt=dt)
                kf.update(row[f'{t}_Raw'], FIXED_UNCERTAINTY)
                kalman_preds[t].append(kf.x[0, 0])
                
            prev_cycle = row['cycle']
            
        for t in kfs.keys():
            df.loc[engine_data.index, f'{t}_Kalman'] = kalman_preds[t]
            
    # Restore original sorting order to match the test CSV exactly
    df = df.sort_values('orig_index')
        
    # Format Output CSV
    output_cols = {'EngineID': df['engine_id'], 'Cycle': df['cycle']}
    for target in targets_list:
        if f'{target}_Kalman' in df.columns:
            output_cols[target] = df[f'{target}_Kalman']
            output_cols[f'{target}_Raw'] = df[f'{target}_Raw']
        else:
            output_cols[target] = df[f'{target}_Raw']
            
    # Overwrite the default output columns for backward compatibility
    if 'HI_overall_Kalman' in df.columns:
        output_cols['OverallHealth'] = df['HI_overall_Kalman']
        output_cols['Predicted_Health'] = df['HI_overall_Kalman']
        output_cols['Raw_ML_Health'] = df['HI_overall_Raw']
        
    output_df = pd.DataFrame(output_cols)
    
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved successfully to {output_path}!")

if __name__ == '__main__':
    main()
