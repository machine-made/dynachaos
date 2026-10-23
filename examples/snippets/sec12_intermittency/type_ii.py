# Type-II intermittency (subcritical Hopf): normal-form escape-time computation.
# Full figure: dynachaos run sec12_intermittency
import warnings

import numpy as np

from dynachaos.maps.intermittency import pm_type_ii_oracle

eps, a, theta, escape_threshold = 2e-3, 1.0, 0.17, 0.35
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for r0 in (1e-3, 1e-2, 2e-2):
        orbit = pm_type_ii_oracle(2500, x0=r0, y0=0.0, eps=eps, a=a, theta=theta)
        radius = np.linalg.norm(orbit, axis=1)
        escaped = radius >= escape_threshold
        if escaped.any():
            status = f"{int(np.argmax(escaped))} steps before escape"
        else:
            status = f">={len(orbit)} (no escape)"
        print(f"r0={r0:.0e}  laminar_length={status}")
