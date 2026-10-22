# Figure: zoom into a mode-locked plateau of the double devil's staircase,
# where rho_theta is pinned at the rational value 1/4 across a narrow
# window of the bare frequency D (window located via the module's
# longest_plateau_window helper on the full sweep).
import numpy as np

from dynachaos.maps.modulated_circle import C_GOLDEN, rotation_numbers

A, eps = 0.10, 0.05
for D in np.linspace(0.2620, 0.2660, 8):
    rho_theta, _ = rotation_numbers(A, C_GOLDEN, D, eps, n_transient=2000, n_iter=8000)
    print(f"D={D:.4f}  rho_theta={rho_theta:.5f}")

# Full figure: dynachaos run sec06_three_torus
