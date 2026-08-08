import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

class KF:
    def __init__(self, q, r):
        self.x = 1.0
        self.P = 0.01
        self.q = q
        self.r = r
    def predict(self, dt=1):
        self.P += self.q * dt
    def update(self, z):
        S = self.P + self.r
        K = self.P / S
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x

# Load predictions.csv and ground truth
preds = pd.read_csv('data/predictions.csv')
gt = pd.read_csv('data/ground_truth.csv')
df = pd.merge(preds, gt, on=['EngineID', 'Cycle'])

best_r2 = -1
best_params = None

for q in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]:
    for r in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
        df['kf'] = 0.0
        for eid in df['EngineID'].unique():
            kf = KF(q, r)
            mask = df['EngineID'] == eid
            cycs = df.loc[mask, 'Cycle'].values
            raws = df.loc[mask, 'HI_overall_Raw'].values
            
            kfs = []
            prev_cyc = None
            for cyc, raw in zip(cycs, raws):
                dt = 1 if prev_cyc is None else max(1, cyc - prev_cyc)
                kf.predict(dt)
                kfs.append(kf.update(raw))
                prev_cyc = cyc
            df.loc[mask, 'kf'] = kfs
            
        r2 = r2_score(df['OverallHealth_y'], df['kf'])
        if r2 > best_r2:
            best_r2 = r2
            best_params = (q, r)
            print(f"New Best R2: {best_r2:.4f} with Q={best_params[0]} and R={best_params[1]}")
            
print(f"Final Best R2: {best_r2:.4f} with Q={best_params[0]} and R={best_params[1]}")
