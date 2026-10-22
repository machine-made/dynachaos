# Figure: global phase diagram of the logistic coupled map lattice
# (Kaneko 1989); spatial activity <|x_i - x_{i-1}|> distinguishes frozen,
# pattern, and turbulent phases across (a, eps).
import numpy as np

from dynachaos.cml.primitives import cml_step_logistic as cml_step

N = 40
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

# Full figure: dynachaos run sec09_pattern
