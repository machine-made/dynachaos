# Figure: double devil's staircase -- rotation number rho_theta(D) of the
# modulated circle map, with rho_phi locked to the golden-mean frequency C.
import numpy as np

from dynachaos.maps.modulated_circle import C_GOLDEN, rotation_numbers

A, eps = 0.10, 0.05
for D in np.linspace(0.0, 1.0, 6):
    rho_theta, rho_phi = rotation_numbers(A, C_GOLDEN, D, eps, n_transient=500, n_iter=2000)
    print(f"D={D:.2f}  rho_theta={rho_theta:.4f}  rho_phi={rho_phi:.4f}")

# Full figure: dynachaos run sec06_three_torus
