# Figure: Lyapunov spectrum of the 4D coupled delayed logistic map vs D_B,
# at fixed coupling eps, crossing from 3-torus into developed chaos.
import numpy as np

from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

A, eps = 0.4, 5e-3
x0 = np.array([0.5, 0.5, 0.3, 0.3])
for DB in (2.37, 2.478, 2.55):
    DA = DB + 0.1

    def f(state, DA=DA, DB=DB):
        return coupled_delayed(state, A, DA, DB, eps)

    def jac(state, DA=DA, DB=DB):
        return coupled_delayed_jac(state, A, DA, DB, eps)

    spectrum = lyapunov_spectrum(f, jac, x0, n_iter=2000, n_transient=1000)
    print(f"D_B={DB:.3f}  lambda_1={spectrum[0]:+.4f}  lambda_2={spectrum[1]:+.4f}")

# Full figure: dynachaos run sec06_three_torus
