# Figure: 0-1 test statistic K(a) across the logistic map's route to chaos.
import numpy as np

from dynachaos.diagnostics.zero_one_test import zero_one_statistic
from dynachaos.maps.primitives import logistic

for a in (1.2, 1.6, 1.9, 2.0):
    x = 0.1
    for _ in range(2000):
        x = logistic(x, a)
    series = np.empty(2000)
    for i in range(2000):
        x = logistic(x, a)
        series[i] = x
    K = zero_one_statistic(series, n_c=20)
    print(f"a={a:.2f}  K_01={K:.3f}")

# Full figure: dynachaos run sec11_diagnostics
