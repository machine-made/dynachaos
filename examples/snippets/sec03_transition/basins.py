# Reduced preview (9 initial states). Full figure: dynachaos run sec03_transition
# Basins of attraction for two mirror-image asymmetric period-32 orbits at a = 1.35344, eps = 0.1.
# Orbits are classified by distance to the two reference cycles; the diagonal x = y is symmetric.
import numpy as np

from dynachaos.maps.coupled_logistic import _find_reference_orbit, coupled_logistic

a, eps = 1.35344, 0.1
ref_A = _find_reference_orbit(a, eps, 0.1, 0.6, n_transient=50_000, period=32)
ref_B = ref_A[:, ::-1]

for x0 in np.linspace(-0.05, 0.05, 9):
    state = np.array([x0, -x0])
    for _ in range(5000):
        state = coupled_logistic(state, a, eps)
    orbit = [state]
    for _ in range(31):
        state = coupled_logistic(state, a, eps)
        orbit.append(state)
    orbit = np.array(orbit)
    dA = min(np.min(np.sum((orbit - r) ** 2, axis=1)) for r in ref_A)
    dB = min(np.min(np.sum((orbit - r) ** 2, axis=1)) for r in ref_B)
    if np.isclose(dA, dB, atol=1e-12):
        basin = "diagonal"
    else:
        basin = "A" if dA < dB else "B"
    print(f"x0={x0:+.3f}  basin={basin}  dist_A={dA:.2e}  dist_B={dB:.2e}")
