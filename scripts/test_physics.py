import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

GAMMA_C = 1.4
GAMMA_H = 1.33
CP_C = 1005
CP_H = 1148
LHV_FUEL = 43000000

df = pd.read_csv('turbojet_complete_dataset.csv')
rename_map = {'EngineID': 'engine_id', 'Cycle': 'cycle', 'FuelFlow_kg_s': 'Fuel_Flow_kg_s', 'OverallHealth': 'health_index'}
df.rename(columns=rename_map, inplace=True)

df["T_t_amb"] = df["Tamb_K"] * (1.0 + (GAMMA_C - 1.0) / 2.0 * df["Mach"] ** 2)
df["P_t_amb"] = df["Pamb_Pa"] * ((1.0 + (GAMMA_C - 1.0) / 2.0 * df["Mach"] ** 2) ** (GAMMA_C / (GAMMA_C - 1.0)))

df['PR_compressor'] = df['P2_Pa'] / df['P_t_amb']
df['PR_combustor'] = df['P3_Pa'] / df['P2_Pa']
df['PR_turbine'] = df['P3_Pa'] / df['P4_Pa']

T2_ideal = df['T_t_amb'] * (df['PR_compressor']) ** ((GAMMA_C - 1) / GAMMA_C)
T4_ideal = df['T3_K'] * (df['PR_turbine']) ** ((GAMMA_H - 1) / GAMMA_H)

df['eta_compressor'] = (T2_ideal - df['T_t_amb']) / (df['T2_K'] - df['T_t_amb'])
df['eta_turbine'] = (df['T3_K'] - df['T4_K']) / (df['T3_K'] - T4_ideal)

df['residual_T2'] = np.abs(df['T2_K'] - T2_ideal)
df['residual_T4'] = np.abs(df['T4_K'] - T4_ideal)

Wc = CP_C * (df['T2_K'] - df['T_t_amb'])
Wt = CP_H * (df['T3_K'] - df['T4_K'])
df['work_ratio'] = np.abs(Wt / Wc)

Qin = df['Fuel_Flow_kg_s'] * LHV_FUEL
df['thermal_efficiency'] = np.abs((Wt - Wc) / Qin)

physics_features = ['eta_compressor', 'eta_turbine', 'residual_T2', 'residual_T4', 'work_ratio', 'thermal_efficiency', 'PR_compressor', 'PR_turbine']

baseline_df = df[df['cycle'] <= 10].groupby('engine_id')[physics_features].mean()
for engine_id in df['engine_id'].unique():
    if engine_id in baseline_df.index:
        for f in physics_features:
            df.loc[df['engine_id'] == engine_id, f'{f}_norm'] = df.loc[df['engine_id'] == engine_id, f] / (baseline_df.loc[engine_id, f] + 1e-8)

ml_features = [f'{feat}_norm' for feat in physics_features]
df.dropna(inplace=True)
X = df[ml_features]
y = df['health_index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
m = RandomForestRegressor(n_estimators=50, n_jobs=-1)
m.fit(X_train, y_train)
print("R2:", r2_score(y_test, m.predict(X_test)))
