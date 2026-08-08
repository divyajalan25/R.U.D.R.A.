import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

X = np.random.rand(30000, 8)
y = np.random.rand(30000)
start = time.time()
for i in range(10):
    X_boot, y_boot = resample(X, y)
    m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    m.fit(X_boot, y_boot)
print("Time:", time.time() - start)
