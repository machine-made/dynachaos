# Figure: permutation entropy H_PE(a) for the logistic map.
import numpy as np

from dynachaos.diagnostics.permutation import permutation_entropy
from dynachaos.maps.primitives import logistic

for a in (1.2, 1.6, 1.9, 2.0):
    x = 0.1
    for _ in range(2000):
        x = logistic(x, a)
    series = np.empty(2000)
    for i in range(2000):
        x = logistic(x, a)
        series[i] = x
    H = permutation_entropy(series, d=5)
    print(f"a={a:.2f}  H_PE={H:.3f}")

# Full figure: dynachaos run sec11_diagnostics
