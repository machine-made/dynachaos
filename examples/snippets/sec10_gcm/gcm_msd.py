# Figure: mean-square displacement (variance) of the mean field h_n of a
# globally coupled logistic map vs system size N; Kaneko's result is that
# this MSD decays much slower than the naive N^-1 central-limit rate (and
# saturates at large N/sample size), a law-of-large-numbers violation.
# This demo, at much smaller N and sample size than the full figure,
# already shows a 100x increase in N shrinking the variance by under 5x.
import numpy as np

from dynachaos.cml.primitives import gcm_step
from dynachaos.maps.primitives import logistic

a = 1.99
eps = 0.1
rng = np.random.default_rng(42)
for N in (50, 500, 5000):
    x = rng.uniform(-1, 1, N)
    for _ in range(300):
        x = gcm_step(x, a, eps)
    h_series = np.empty(400)
    for t in range(400):
        x = gcm_step(x, a, eps)
        h_series[t] = np.mean(logistic(x, a))
    print(f"N={N:5d}  Var(h)={np.var(h_series):.3e}")

# Full figure: dynachaos run sec10_gcm
