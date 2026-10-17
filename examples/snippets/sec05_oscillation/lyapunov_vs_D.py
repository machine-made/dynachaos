# Reduced preview (3 D values). Full figure: dynachaos run sec05_oscillation
# Lyapunov spectrum of the delayed logistic map vs D at alpha=0.3.
import numpy as np

from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
from dynachaos.maps.delayed_logistic import delayed_logistic, delayed_logistic_jac

alpha = 0.3
for D in (1.6, 1.95, 2.2):
    fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
    x0 = np.array([fp + 0.01, fp - 0.01])

    def f(state, D=D):
        return delayed_logistic(state, alpha, D)

    def jac(state, D=D):
        return delayed_logistic_jac(state, alpha, D)

    spectrum = lyapunov_spectrum(f, jac, x0, n_iter=4000, n_transient=2000)
    print(f"D={D:.2f}  lambda_1={spectrum[0]:+.4f}  lambda_2={spectrum[1]:+.4f}")
