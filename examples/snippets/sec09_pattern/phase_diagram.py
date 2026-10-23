# Reduced preview (3 (a, eps) states, N = 100, 500-step transient, 200 samples).
# Full figure: dynachaos run sec09_pattern (200 x 160 grid, 2000 samples after 5000 transient).
# Spatial activity <|x_i - x_{i-1}|> of the logistic coupled map lattice
# (Kaneko 1989) separates spatially smooth from spatially rough states.
import numpy as np

from dynachaos.cml.primitives import cml_step_logistic as cml_step

N = 100
rng = np.random.default_rng(42)
for a, eps in ((1.50, 0.10), (1.72, 0.10), (1.90, 0.10)):
    x = rng.uniform(-1, 1, N)
    for _ in range(500):
        x = cml_step(x, a, eps)
    activity = 0.0
    for _ in range(200):
        x = cml_step(x, a, eps)
        activity += np.mean(np.abs(x - np.roll(x, 1)))
    print(f"a={a:.2f} eps={eps:.2f}  spatial activity={activity / 200:.4f}")
