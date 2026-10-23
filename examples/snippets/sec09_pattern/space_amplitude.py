# Reduced preview (2 a states, N = 100). Full figure: dynachaos run sec09_pattern
# Space-amplitude snapshots x(i) vs site i of a logistic coupled map lattice (N = 100).
import numpy as np

from dynachaos.cml.primitives import cml_step_logistic as cml_step

N = 100
rng = np.random.default_rng(42)
for a, eps, label in ((1.50, 0.10, "frozen random"), (1.90, 0.10, "fully developed turbulence")):
    x = rng.uniform(-1, 1, N)
    for _ in range(500):
        x = cml_step(x, a, eps)
    snapshot = x.copy()
    print(f"a={a:.2f} ({label})  amplitude mean={snapshot.mean():.4f} std={snapshot.std():.4f}")
