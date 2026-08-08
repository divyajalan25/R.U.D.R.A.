import pandas as pd
df = pd.read_csv('data/train.csv')
gt = pd.read_csv('data/ground_truth.csv')
merged = pd.merge(df, gt, on=['EngineID', 'Cycle'])
from train_digital_twin import compute_physics_features, compute_baselines
merged = compute_physics_features(merged)
physics_features = ['eta_compressor', 'eta_turbine', 'eta_combustor', 'work_ratio']
merged, _ = compute_baselines(merged, physics_features)
for f in physics_features:
    corr = merged[f'{f}_norm'].corr(merged['CompressorHealth'])
    print(f'{f}_norm with CompHealth: {corr:.2f}')
    corr_overall = merged[f'{f}_norm'].corr(merged['OverallHealth'])
    print(f'{f}_norm with OverallHealth: {corr_overall:.2f}')
