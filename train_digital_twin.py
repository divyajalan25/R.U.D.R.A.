import os
import sys
import subprocess

def install_and_import(package, install_name=None):
    if install_name is None:
        install_name = package
    try:
        __import__(package)
    except ImportError:
        print(f"{install_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

# Ensure required packages are installed
install_and_import('catboost')
install_and_import('lightgbm')
install_and_import('matplotlib')

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Constants
GAMMA_C = 1.4
GAMMA_H = 1.33
CP_C = 1005
CP_H = 1148
LHV_FUEL = 43000000  # J/kg

def load_and_preprocess_data(filepath='data/train.csv'):
    print("============================================================")
    print("PHYSICS-INFORMED DIGITAL TWIN TRAINING PIPELINE")
    print("AEROTHON 2026 - HAL x IITI")
    print("============================================================")
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure the data file exists before running the pipeline.")
    
    df = pd.read_csv(filepath)
    
    # Merge with ground_truth.csv to load all targets
    print("Merging with ground_truth.csv to load all targets...")
    gt_path = 'data/ground_truth.csv'
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
        # Drop columns if they exist to avoid duplicate conflicts
        df = df.drop(columns=['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Thrust_N', 'RUL'], errors='ignore')
        df = pd.merge(df, gt_df[['EngineID', 'Cycle', 'CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Thrust_N']], on=['EngineID', 'Cycle'], how='inner')
    else:
        raise FileNotFoundError(f"ground_truth.csv not found at {gt_path} but is required for training labels!")
    
    # Calculate RUL target dynamically (max_cycle - current_cycle)
    max_cycles = df.groupby('EngineID')['Cycle'].transform('max')
    df['RUL'] = max_cycles - df['Cycle']
    
    # Rename columns to match the expected names in the script
    rename_map = {
        'EngineID': 'engine_id',
        'Cycle': 'cycle',
        'FuelFlow_kg_s': 'Fuel_Flow_kg_s'
    }
    df.rename(columns=rename_map, inplace=True)
    
    print(f"Shape: {df.shape}")
    return df

def compute_physics_features(df):
    print("Computing physics features...")
    
    # Stagnation (total) conditions from flight state
    df["T_t_amb"] = df["Tamb_K"] * (1.0 + (GAMMA_C - 1.0) / 2.0 * df["Mach"] ** 2)
    df["P_t_amb"] = df["Pamb_Pa"] * ((1.0 + (GAMMA_C - 1.0) / 2.0 * df["Mach"] ** 2) ** (GAMMA_C / (GAMMA_C - 1.0)))

    # PR_compressor (P2 is compressor exit, P_t_amb is inlet)
    df['PR_compressor'] = df['P2_Pa'] / df['P_t_amb']
    df['PR_combustor'] = df['P3_Pa'] / df['P2_Pa']
    df['PR_turbine'] = df['P3_Pa'] / df['P4_Pa']
    
    # Ideal temperatures
    T2_ideal = df['T_t_amb'] * (df['PR_compressor']) ** ((GAMMA_C - 1) / GAMMA_C)
    T4_ideal = df['T3_K'] / (df['PR_turbine']) ** ((GAMMA_H - 1) / GAMMA_H)
    
    # Efficiencies
    df['eta_compressor'] = np.clip(np.abs((T2_ideal - df['T_t_amb']) / (df['T2_K'] - df['T_t_amb'] + 1e-8)), 0.7, 0.95)
    df['eta_turbine'] = np.clip(np.abs((df['T3_K'] - df['T4_K']) / (df['T3_K'] - T4_ideal + 1e-8)), 0.7, 0.95)
    
    T3_ideal = df['T3_K'] * 1.05 # placeholder for ideal combustor exit
    df['eta_combustor'] = np.clip(np.abs((df['T3_K'] - df['T2_K']) / (T3_ideal - df['T2_K'] + 1e-8)), 0.85, 1.0)
    
    # Residuals
    df['residual_T2'] = np.abs(df['T2_K'] - T2_ideal)
    df['residual_T4'] = np.abs(df['T4_K'] - T4_ideal)
    
    # Work and Efficiency
    Wc = CP_C * (df['T2_K'] - df['T_t_amb'])
    Wt = CP_H * (df['T3_K'] - df['T4_K'])
    df['work_ratio'] = np.abs(Wt / (Wc + 1e-8))
    
    Qin = df['Fuel_Flow_kg_s'] * LHV_FUEL
    df['thermal_efficiency'] = np.abs((Wt - Wc) / (Qin + 1e-8))
    
    # Validation checks inside physics computation
    print("Validating physics constraints...")
    assert df['eta_compressor'].between(0.5, 1.0).all(), "Compressor efficiency out of bounds!"
    assert df['eta_turbine'].between(0.5, 1.0).all(), "Turbine efficiency out of bounds!"
    
    return df

def compute_baselines(df, physics_features):
    print("Computing per-engine baselines...")
    baselines = {}
    
    baseline_df = df[df['cycle'] <= 10].groupby('engine_id')[physics_features].mean()
    
    for engine_id in df['engine_id'].unique():
        if engine_id in baseline_df.index:
            baselines[engine_id] = baseline_df.loc[engine_id].to_dict()
            
    # Normalize features by baseline
    for feature in physics_features:
        baseline_values = df['engine_id'].map(lambda x: baselines.get(x, {}).get(feature, 1.0))
        df[f'{feature}_norm'] = df[feature] / (baseline_values + 1e-8)
        
    return df, baselines

def train_models():
    df = load_and_preprocess_data()
    df = compute_physics_features(df)
    
    physics_features = [
        'eta_compressor', 'eta_turbine', 'eta_combustor',
        'residual_T2', 'residual_T4', 'work_ratio',
        'thermal_efficiency', 'PR_compressor'
    ]
    
    # Ensure no inf or nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    df, baselines = compute_baselines(df, physics_features)

    print("Adding continuous-time EMA features for stateful context...")
    norm_features = [f'{feat}_norm' for feat in physics_features]
    
    lambda_val = 0.2
    ema_features = []
    for feat in norm_features:
        ema_col = f'{feat}_ema'
        delta_col = f'{feat}_delta'
        ema_features.extend([ema_col, delta_col])
        df[ema_col] = df.groupby('engine_id')[feat].transform(lambda x: x.ewm(alpha=lambda_val, adjust=False).mean())
        df[delta_col] = df[feat] - df[ema_col]
        
    ml_features = norm_features + ema_features
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    X = df[ml_features].copy()
    groups = df['engine_id']
    
    targets = {
        'HI_compressor': 'CompressorHealth',
        'HI_combustor': 'CombustorHealth',
        'HI_turbine': 'TurbineHealth',
        'HI_overall': 'OverallHealth',
        'thrust_N': 'Thrust_N',
        'RUL': 'RUL'
    }
    
    os.makedirs('models', exist_ok=True)
    
    print(f"\nTraining ensemble models for all {len(targets)} targets using {len(ml_features)} features...")
    
    for key, col in targets.items():
        print(f"\n--- Training Target: {col} ({key}) ---")
        y = df[col].copy()
        
        # 5-Fold GroupKFold Cross-Validation for Out-Of-Fold predictions
        gkf = GroupKFold(n_splits=5)
        oof_preds_xgb = np.zeros(len(df))
        oof_preds_lgb = np.zeros(len(df))
        oof_preds_cat = np.zeros(len(df))
        oof_preds_rf = np.zeros(len(df))
        
        metrics = {'r2': []}
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # XGBoost
            model_xgb = xgb.XGBRegressor(tree_method='hist', random_state=42)
            model_xgb.fit(X_train, y_train)
            
            # LightGBM
            model_lgb = lgb.LGBMRegressor(random_state=42, verbose=-1)
            model_lgb.fit(X_train, y_train)
            
            # CatBoost
            model_cat = CatBoostRegressor(random_state=42, verbose=0)
            model_cat.fit(X_train, y_train)
            
            # Random Forest (Fast settings to train quickly)
            model_rf = RandomForestRegressor(n_estimators=25, max_depth=12, random_state=42, n_jobs=-1)
            model_rf.fit(X_train, y_train)
            
            p_xgb = model_xgb.predict(X_val)
            p_lgb = model_lgb.predict(X_val)
            p_cat = model_cat.predict(X_val)
            p_rf = model_rf.predict(X_val)
            
            oof_preds_xgb[val_idx] = p_xgb
            oof_preds_lgb[val_idx] = p_lgb
            oof_preds_cat[val_idx] = p_cat
            oof_preds_rf[val_idx] = p_rf
            
            p_avg = (p_xgb + p_lgb + p_cat + p_rf) / 4
            metrics['r2'].append(r2_score(y_val, p_avg))
            
        print(f"GroupKFold Cross-Validation R²: {np.mean(metrics['r2']):.4f}")
        
        # Meta-Model Ridge Regression
        OOF = pd.DataFrame({
            'xgb': oof_preds_xgb,
            'lgb': oof_preds_lgb,
            'cat': oof_preds_cat,
            'rf': oof_preds_rf
        })
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(OOF, y)
        
        # Train final models on ALL data
        final_xgb = xgb.XGBRegressor(tree_method='hist', random_state=42)
        final_xgb.fit(X, y)
        
        final_lgb = lgb.LGBMRegressor(random_state=42, verbose=-1)
        final_lgb.fit(X, y)
        
        final_cat = CatBoostRegressor(random_state=42, verbose=0)
        final_cat.fit(X, y)
        
        final_rf = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)
        final_rf.fit(X, y)
        
        # Save models for this target
        joblib.dump({
            'xgb': final_xgb,
            'lgb': final_lgb,
            'cat': final_cat,
            'rf': final_rf
        }, f'models/base_models_{key}.pkl')
        
        joblib.dump(meta_model, f'models/meta_model_{key}.pkl')
        print(f"✓ Saved models for target: {col} ({key})")
        
    # Save baselines and feature names once
    joblib.dump(baselines, 'models/baselines.pkl')
    joblib.dump(ml_features, 'models/feature_names.pkl')
    
    print("\n============================================================")
    print("✅ ENSEMBLE TRAINING COMPLETE FOR ALL TARGETS!")
    print("============================================================")

if __name__ == "__main__":
    train_models()
