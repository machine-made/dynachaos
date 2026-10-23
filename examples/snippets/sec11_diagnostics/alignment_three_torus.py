# Reduced preview (2 orbits, 20 000 iterates). Full figure: dynachaos run sec11_diagnostics
# GALI_k decays at the rate sum_{i=2..k} (lambda_i - lambda_1) set by the Lyapunov spectrum.
import numpy as np

from dynachaos.diagnostics.alignment_three_torus import predicted_rates
from dynachaos.diagnostics.lyapunov import lyapunov_spectrum_blocks
from dynachaos.diagnostics.sali_gali import alignment_indices
from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

alpha, eps = 0.4, 5e-3
x0 = np.array([0.5, 0.5, 0.3, 0.3])
n_iter, n_transient = 20_000, 5000

for DB, label in ((2.37, "partial locking"), (2.55, "chaos")):
    DA = DB + 0.1

    def f(s, DA=DA, DB=DB):
        return coupled_delayed(s, alpha, DA, DB, eps)

    def jac(s, DA=DA, DB=DB):
        return coupled_delayed_jac(s, alpha, DA, DB, eps)

    spectrum, se = lyapunov_spectrum_blocks(f, jac, x0, n_iter, n_transient)
    gali = alignment_indices(f, jac, x0, k=4, n_iter=n_iter, n_transient=n_transient)
    rate, _ = predicted_rates(spectrum, se)
    fit = (gali[-1, 1:] - gali[n_iter // 2, 1:]) * np.log(10.0) / (n_iter - n_iter // 2)
    print(f"DB={DB} ({label})  lambda={np.round(spectrum, 4)}")
    print(f"  GALI_2..4 rate  fitted={np.round(fit, 4)}  predicted={np.round(rate, 4)}")
