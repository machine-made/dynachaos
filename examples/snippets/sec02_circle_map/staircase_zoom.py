# Figure: staircase zoom -- fine A-sweep near a mode-locked plateau, with
# the Lyapunov exponent confirming the orbit is periodic (lambda < 0) there.
import numpy as np

from dynachaos.maps.circle_map import lyapunov_exponent, rotation_number

D = 0.25
A_values = np.linspace(0.15, 0.25, 10)
for A in A_values:
    rho = rotation_number(A, D=D, n_transient=500, n_iter=2000)
    lam = lyapunov_exponent(A, D=D, n_transient=500, n_iter=2000)
    print(f"A={A:.3f}  rho={rho:.4f}  lambda={lam:+.4f}")

# Full figure: dynachaos run sec02_circle_map
