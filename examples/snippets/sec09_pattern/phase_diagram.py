# Reduced preview (3 (a, eps) states, N = 100). Full figure: dynachaos run sec09_pattern
# Single-site temporal variance log10(var) of the logistic CML across (a, eps).
import numpy as np

from dynachaos.cml.primitives import cml_step_logistic as cml_step

N = 100
rng = np.random.default_rng(42)
for a, eps in ((1.50, 0.10), (1.72, 0.10), (1.90, 0.10)):
    x = rng.uniform(-1, 1, N)
    for _ in range(500):
        x = cml_step(x, a, eps)
    site_0 = np.empty(2000)
    for s in range(2000):
        x = cml_step(x, a, eps)
        site_0[s] = x[0]
    log_var = np.log10(np.var(site_0) + 1e-12)
    print(f"a={a:.2f} eps={eps:.2f}  single-site log10(var)={log_var:.4f}")
