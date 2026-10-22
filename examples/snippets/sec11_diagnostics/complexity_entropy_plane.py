# Figure: complexity-entropy plane (H, C) locating logistic-map regimes.
import numpy as np

from dynachaos.diagnostics.permutation import complexity_entropy
from dynachaos.maps.primitives import logistic

for a in (1.2, 1.6, 1.9, 2.0):
    x = 0.1
    for _ in range(2000):
        x = logistic(x, a)
    series = np.empty(2000)
    for i in range(2000):
        x = logistic(x, a)
        series[i] = x
    H, C = complexity_entropy(series, d=5)
    print(f"a={a:.2f}  H={H:.3f}  C={C:.3f}")

# Full figure: dynachaos run sec11_diagnostics
