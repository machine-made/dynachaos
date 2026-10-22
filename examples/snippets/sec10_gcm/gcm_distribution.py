# Figure: distribution P(h) of the mean field h_n of a globally coupled
# logistic map; the mean-field histogram stays broad rather than narrowing
# to a delta function as N grows, the hallmark of GCM violating the law
# of large numbers.
import numpy as np

from dynachaos.cml.primitives import gcm_step
from dynachaos.maps.primitives import logistic

a = 1.99
eps = 0.1
rng = np.random.default_rng(42)
for N in (50, 5000):
    x = rng.uniform(-1, 1, N)
    for _ in range(300):
        x = gcm_step(x, a, eps)
    h_series = np.empty(400)
    for t in range(400):
        x = gcm_step(x, a, eps)
        h_series[t] = np.mean(logistic(x, a))
    print(f"N={N:5d}  mean(h)={h_series.mean():.4f}  std(h)={h_series.std():.4f}")

# Full figure: dynachaos run sec10_gcm
