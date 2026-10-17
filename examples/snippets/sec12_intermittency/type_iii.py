# Type-III intermittency (flip bifurcation): escape-episode statistics.
# Full figure: dynachaos run sec12_intermittency
import numpy as np

from dynachaos.maps.intermittency import pm_type_iii_oracle

eps, a, escape_threshold = 2e-3, 1.0, 0.35
for x0 in (2e-6, 2e-4, 2e-3):
    orbit = pm_type_iii_oracle(8000, x0=x0, eps=eps, a=a)
    escaped = np.abs(orbit) >= escape_threshold
    if escaped.any():
        status = f"{int(np.argmax(escaped))} steps before escape"
    else:
        status = f">={len(orbit)} (no escape)"
    print(f"x0={x0:.0e}  laminar_length={status}")
