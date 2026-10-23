# Reduced preview (windowed estimator, N = 400, T = 150, 4 velocities).
# Full figure: dynachaos run sec08_sti
# Windowed co-moving Lyapunov exponent lambda(v) for a logistic coupled map lattice (eps = 0.3):
# growth of the perturbation norm over a window of 2 w + 1 sites that moves with velocity v.
# Static figure: N = 3000, T = 1000, w = 15, 51 velocities on [0, 1]; zero crossings give v_p.
import numpy as np

from dynachaos.diagnostics.comoving_lyapunov import (
    comoving_lyapunov_windowed_logistic,
    comoving_zero_crossings,
)

v_values = np.linspace(0.0, 1.0, 4)
for a in (1.70, 1.85, 1.95):
    lam_v = comoving_lyapunov_windowed_logistic(
        a=a, eps=0.3, N=400, T=150, v_values=v_values, n_transient=500, w=5, i0=50
    )
    crossings = comoving_zero_crossings(v_values, lam_v)
    print(f"a={a:.2f}  lambda(v)={np.round(lam_v, 3)}  zero crossings={np.round(crossings, 2)}")
