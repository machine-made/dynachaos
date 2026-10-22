# Figure: Type-III intermittency (subharmonic/flip bifurcation) -- the
# sign-alternating flip normal form x -> -(1+eps)x - a*x^3 grows slowly
# and laminar-phase length increases as the reinjection amplitude shrinks.
# Illustrative of the method; the full figure uses
# dynachaos.diagnostics.type_iii_intermittency_figure's escape-episode
# statistics, built on this same flip recurrence
# (dynachaos.maps.intermittency.pm_type_iii_oracle).
import numpy as np

from dynachaos.maps.intermittency import pm_type_iii_oracle

eps, a, escape_threshold = 2e-3, 1.0, 0.35
for x0 in (2e-6, 2e-4, 2e-3):
    orbit = pm_type_iii_oracle(8000, x0=x0, eps=eps, a=a)
    escape_idx = int(np.argmax(np.abs(orbit) >= escape_threshold))
    print(f"x0={x0:.0e}  laminar_length={escape_idx} steps before escape")

# Full figure: dynachaos run sec12_intermittency
