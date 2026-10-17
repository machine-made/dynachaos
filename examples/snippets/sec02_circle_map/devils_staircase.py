# Reduced preview (20 values). Full figure: dynachaos run sec02_circle_map
import numpy as np

from dynachaos.maps.circle_map import rotation_number

Omega = 0.25
K_values = np.linspace(0.0, 1.0, 20)
rho = [rotation_number(K, D=Omega, n_transient=500, n_iter=2000) for K in K_values]

for K, r in zip(K_values[::4], rho[::4], strict=False):
    print(f"K={K:.2f}  rho={r:.4f}")
