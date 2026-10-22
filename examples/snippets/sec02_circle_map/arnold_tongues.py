# Figure: Arnold tongues -- rotation number rho(Omega, K) over a small grid.
import numpy as np

from dynachaos.maps.circle_map import rotation_number

Omega_values = np.linspace(0.0, 1.0, 10)
K_values = np.linspace(0.0, 0.3, 5)

for K in K_values:
    row = [rotation_number(K, D=Omega, n_transient=200, n_iter=1000) for Omega in Omega_values]
    locked = sum(abs(r - round(r)) < 1e-2 for r in row)
    print(f"K={K:.2f}  locked points: {locked}/{len(Omega_values)}")

# Full figure: dynachaos run sec02_circle_map
