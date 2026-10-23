# Reduced preview (4 a values, 5000 points, n_c=50). Full figure: dynachaos run sec11_diagnostics
# 0-1 test statistic K_01 across the logistic map's route to chaos.
import numpy as np

from dynachaos.diagnostics.zero_one_test import zero_one_statistic
from dynachaos.maps.primitives import logistic

for a in (1.2, 1.6, 1.9, 2.0):
    x = 0.1
    for _ in range(2000):
        x = logistic(x, a)
    series = np.empty(5000)
    for i in range(5000):
        x = logistic(x, a)
        series[i] = x
    K = zero_one_statistic(series, n_c=50)
    print(f"a={a:.2f}  K_01={K:.3f}")
