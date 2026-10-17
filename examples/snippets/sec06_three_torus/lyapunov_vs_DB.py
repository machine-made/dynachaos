# Reduced preview (3 D_B values). Full figure: dynachaos run sec06_three_torus
# Lyapunov spectrum of the 4D coupled delayed logistic map vs D_B (alpha=0.4, eps=5e-3).
import numpy as np

from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

alpha, eps = 0.4, 5e-3
x0 = np.array([0.5, 0.5, 0.3, 0.3])
for DB in (2.37, 2.478, 2.55):
    DA = DB + 0.1

    def f(state, DA=DA, DB=DB):
        return coupled_delayed(state, alpha, DA, DB, eps)

    def jac(state, DA=DA, DB=DB):
        return coupled_delayed_jac(state, alpha, DA, DB, eps)

    spectrum = lyapunov_spectrum(f, jac, x0, n_iter=2000, n_transient=1000)
    print(f"D_B={DB:.3f}  lambda_1={spectrum[0]:+.4f}  lambda_2={spectrum[1]:+.4f}")
