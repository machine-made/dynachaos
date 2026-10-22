# Figure: Kaneko Model-A CML spatiotemporal intermittency -- the turbulent
# fraction (sites with |x(i+1)-x(i)| > delta, the Kaneko 1985 burst
# criterion) rises sharply with coupling eps.
import numpy as np

from dynachaos.cml.spatiotemporal import model_A_f, simulate_cml

N = 80
rng = np.random.default_rng(42)
for eps in (0.05, 0.08, 0.12):
    x0 = rng.uniform(0, 1, N)
    spacetime = simulate_cml(model_A_f, model_A_f, eps, N=N, n_transient=300, n_record=60, x0=x0)
    turbulent_mask = np.abs(spacetime - np.roll(spacetime, -1, axis=1)) > 0.05
    print(f"eps={eps:.2f}  turbulent_fraction={turbulent_mask.mean():.3f}")

# Full figure: dynachaos run sec12_intermittency
