# Figure: devil's staircase -- rotation number rho(A) of the circle map.
import numpy as np

from dynachaos.maps.circle_map import rotation_number

D = 0.25
A_values = np.linspace(0.0, 1.0, 20)
rho = [rotation_number(A, D=D, n_transient=500, n_iter=2000) for A in A_values]

for A, r in zip(A_values[::4], rho[::4], strict=False):
    print(f"A={A:.2f}  rho={r:.4f}")

# Full figure: dynachaos run sec02_circle_map
