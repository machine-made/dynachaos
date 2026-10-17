# Reduced preview (8 values in 1/4 plateau). Full figure: dynachaos run sec06_three_torus
# Zoom into the 1/4 mode-locked plateau of the double devil's staircase.
import numpy as np

from dynachaos.maps.modulated_circle import C_GOLDEN, rotation_numbers

A, eps = 0.10, 0.05
for D in np.linspace(0.2631, 0.2654, 8):
    rho_theta, _ = rotation_numbers(A, C_GOLDEN, D, eps, n_transient=2000, n_iter=8000)
    print(f"D={D:.4f}  rho_theta={rho_theta:.5f}")
