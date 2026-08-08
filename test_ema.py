import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from train_digital_twin import load_and_preprocess_data, compute_physics_features, compute_baselines

df = load_and_preprocess_data('data/train.csv')
df = compute_physics_features(df)
physics_features = ['eta_compressor', 'eta_turbine', 'eta_combustor', 'work_ratio']
df, _ = compute_baselines(df, physics_features)

# Compute EMA and Delta
lambda_val = 0.2
norm_features = [f'{feat}_norm' for feat in physics_features]

ema_features = []
for feat in norm_features:
    ema_col = f'{feat}_ema'
    delta_col = f'{feat}_delta'
    ema_features.extend([ema_col, delta_col])
    
    # Compute continuous EMA using ewm
    df[ema_col] = df.groupby('engine_id')[feat].transform(lambda x: x.ewm(alpha=lambda_val, adjust=False).mean())
    df[delta_col] = df[feat] - df[ema_col]

ml_features = norm_features + ema_features
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

X = df[ml_features]
y = df['OverallHealth']
groups = df['engine_id']

gkf = GroupKFold(n_splits=5)
preds = np.zeros(len(df))

for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    preds[val_idx] = model.predict(X_va)

r2 = r2_score(y, preds)
print(f"EMA Model R2: {r2:.4f}")
