# Figure: co-moving Lyapunov exponent lambda(v) for a logistic coupled map
# lattice; its zero crossings mark the propagation velocities of chaotic
# information in the spatiotemporal state.
import numpy as np

from dynachaos.diagnostics.comoving_lyapunov import comoving_lyapunov_spectrum_logistic

v_values = np.array([-0.5, 0.0, 0.5])
for a in (1.70, 1.85, 1.95):
    lam_v = comoving_lyapunov_spectrum_logistic(
        a=a, eps=0.3, N=60, v_values=v_values, n_iter=3000, n_transient=500
    )
    print(f"a={a:.2f}  lambda(v)={np.round(lam_v, 4)}")

# Full figure: dynachaos run sec08_sti
