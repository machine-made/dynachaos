# Reduced preview (K in [0.125, 0.158], 10 values). Full figure: dynachaos run sec02_circle_map
# Below chaos onset (K ~ 0.182) lambda <= 0 here: mode-locked, regular motion.
import numpy as np

from dynachaos.maps.circle_map import lyapunov_exponent, rotation_number

Omega = 0.25
K_values = np.linspace(0.125, 0.158, 10)
for K in K_values:
    rho = rotation_number(K, D=Omega, n_transient=500, n_iter=2000)
    lam = lyapunov_exponent(K, D=Omega, n_transient=500, n_iter=2000)
    print(f"K={K:.3f}  rho={rho:.4f}  lambda={lam:+.4f}")
