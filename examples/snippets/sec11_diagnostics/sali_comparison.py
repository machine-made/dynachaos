# Figure: SALI decay distinguishes torus (regular) from chaotic regimes.
import numpy as np

from dynachaos.diagnostics.sali_gali import sali
from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

A, eps = 0.4, 5e-3
x0 = np.array([0.5, 0.5, 0.3, 0.3])

for DB, label in ((2.35, "3-torus"), (2.55, "developed chaos")):
    DA = DB + 0.1

    def f(s, DA=DA, DB=DB):
        return coupled_delayed(s, A, DA, DB, eps)

    def jac(s, DA=DA, DB=DB):
        return coupled_delayed_jac(s, A, DA, DB, eps)

    s = sali(f, jac, x0, n_iter=1000, n_transient=500)
    print(f"DB={DB} ({label})  SALI[-1]={s[-1]:.3e}")

# Full figure: dynachaos run sec11_diagnostics
