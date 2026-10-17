# Reduced preview (3 D values). Full figure: dynachaos run sec04_doubling
# Lyapunov spectrum of Map (IV), the 4D delayed logistic map, vs D (alpha=0.3).
import numpy as np

from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
from dynachaos.maps.torus_doubling import map_IV, map_IV_jac

alpha = 0.3
x0 = np.array([0.5, 0.45, 0.52, 0.48])
for D in (1.50, 1.5206, 1.5212):

    def f(state, D=D):
        return map_IV(state, alpha, D)

    def jac(state, D=D):
        return map_IV_jac(state, alpha, D)

    spectrum = lyapunov_spectrum(f, jac, x0, n_iter=3000, n_transient=2000)
    print(f"D={D}  lambda_1={spectrum[0]:+.4f}  lambda_2={spectrum[1]:+.4f}")
