# Reduced preview (3 D values). Full figure: dynachaos run sec05_oscillation
# Dynamics across the locking-to-chaos transition, D in [1.86, 1.95] at alpha=0.3;
# lambda_1 turns positive as the frequency-locked torus gives way to chaos.
import numpy as np

from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
from dynachaos.maps.delayed_logistic import delayed_logistic, delayed_logistic_jac

alpha = 0.3
for D in (1.860, 1.930, 1.950):
    fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
    x0 = np.array([fp + 0.01, fp - 0.01])

    def f(state, D=D):
        return delayed_logistic(state, alpha, D)

    def jac(state, D=D):
        return delayed_logistic_jac(state, alpha, D)

    spectrum = lyapunov_spectrum(f, jac, x0, n_iter=4000, n_transient=2000)
    print(f"D={D:.3f}  lambda_1={spectrum[0]:+.4f}")
