# Reduced preview (grid 10 x 5). Full figure: dynachaos run sec02_circle_map
import numpy as np

from dynachaos.maps.circle_map import rotation_number


def is_locked(r: float, max_denom: int = 5, tol: float = 1e-2) -> bool:
    return any(abs(r - p / q) < tol for q in range(1, max_denom + 1) for p in range(q + 1))


Omega_values = np.linspace(0.0, 1.0, 10)
K_values = np.linspace(0.0, 0.3, 5)

for K in K_values:
    row = [rotation_number(K, D=Omega, n_transient=200, n_iter=1000) for Omega in Omega_values]
    locked = sum(is_locked(r) for r in row)
    print(f"K={K:.2f}  locked points (q<=5): {locked}/{len(Omega_values)}")
