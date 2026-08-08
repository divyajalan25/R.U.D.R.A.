import pandas as pd
import numpy as np
import server

df = pd.read_csv("dataset aerothon/train.csv")
df = df.rename(columns=server.COL_RENAME)
GAMMA = 1.4
CP = 1005.0

df["T_t_amb"] = df["T_amb"] * (1.0 + (GAMMA - 1.0) / 2.0 * df["mach"] ** 2)
df["P_t_amb"] = df["P_amb"] * (
    (1.0 + (GAMMA - 1.0) / 2.0 * df["mach"] ** 2) ** (GAMMA / (GAMMA - 1.0))
)
df["PR_compressor"] = df["P2"] / df["P_t_amb"]
df["PR_combustor"]  = df["P3"] / df["P2"]
df["PR_turbine"]    = df["P3"] / df["P4"]
df["PR_overall"]    = df["P4"] / df["P_t_amb"]

exp = (GAMMA - 1.0) / GAMMA  # 0.2857142857142857
T2_ideal = df["T_t_amb"] * df["PR_compressor"] ** exp
denom_c = (df["T2"] - df["T_t_amb"])
calc_eta = (T2_ideal - df["T_t_amb"]) / denom_c

print("Calculated eta_c vs Ground truth eta_c (first 5):")
print("Calculated:", calc_eta.head().values)
# Wait, train.csv doesn't have eta_compressor, does it?
# Let's see if train.csv has it.
if "eta_compressor" in df.columns:
    print("Ground truth:", df["eta_compressor"].head().values)
    diff = np.abs(calc_eta - df["eta_compressor"])
    print("Max diff:", diff.max())
else:
    print("train.csv doesn't have eta_compressor")
