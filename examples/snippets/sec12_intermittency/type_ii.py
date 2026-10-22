# Figure: Type-II intermittency (subcritical Hopf) -- an unstable spiral
# with slow growth rate eps produces long near-periodic laminar episodes
# before escaping past a threshold radius; escape length grows as the
# reinjection radius shrinks. Illustrative of the method; the full figure
# uses dynachaos.diagnostics.type_ii_intermittency_figure's bounded-orbit
# escape-time computation, built on the same normal-form recurrence as
# dynachaos.maps.intermittency.pm_type_ii_oracle.
import warnings

import numpy as np

from dynachaos.maps.intermittency import pm_type_ii_oracle

eps, a, theta, escape_threshold = 2e-3, 1.0, 0.17, 0.35
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for r0 in (1e-3, 1e-2, 2e-2):
        orbit = pm_type_ii_oracle(2500, x0=r0, y0=0.0, eps=eps, a=a, theta=theta)
        radius = np.linalg.norm(orbit, axis=1)
        escape_idx = int(np.argmax(radius >= escape_threshold))
        print(f"r0={r0:.0e}  laminar_length={escape_idx} steps before escape")

# Full figure: dynachaos run sec12_intermittency
