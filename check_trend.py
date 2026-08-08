import pandas as pd
df = pd.read_csv('data/test_evaluation_comparison.csv')
features = ['eta_compressor_norm', 'eta_turbine_norm', 'eta_combustor_norm', 'work_ratio_norm']

for f in features:
    if f in df.columns:
        early = df.loc[df['Cycle'] <= 10, f].mean()
        late = df.loc[df['Cycle'] > 200, f].mean()
        print(f'{f}: early={early:.4f}, late={late:.4f}')
    else:
        print(f"{f} not found")

