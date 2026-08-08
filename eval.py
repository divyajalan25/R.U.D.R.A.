import pandas as pd
from sklearn.metrics import r2_score

preds = pd.read_csv('data/predictions.csv')
gt = pd.read_csv('data/ground_truth.csv')
merged = pd.merge(preds, gt, on=['EngineID', 'Cycle'])

print("--- EVALUATION ---")
for t, gt_t in [
    ('HI_compressor', 'CompressorHealth'),
    ('HI_turbine', 'TurbineHealth'),
    ('HI_combustor', 'CombustorHealth'),
    ('HI_overall', 'OverallHealth_y'),
    ('thrust_N', 'Thrust_N'),
    ('RUL', 'RUL')
]:
    r2 = r2_score(merged[gt_t], merged[t])
    print(f"{t}: R2 = {r2:.4f}")
