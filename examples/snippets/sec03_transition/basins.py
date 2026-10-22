# Figure: basin of attraction for two mirror-image asymmetric period-32
# orbits, classified by the sign of (x-y) reached from each initial state.
# Illustrative of the method; the full figure uses
# dynachaos.maps.coupled_logistic.compute_basins, which classifies against
# two reference orbits found from a 500,000-iterate transient.
import numpy as np

from dynachaos.maps.coupled_logistic import coupled_logistic

A, D = 1.35344, 0.1
for x0 in np.linspace(-0.05, 0.05, 9):
    state = np.array([x0, -x0])
    for _ in range(20_000):
        state = coupled_logistic(state, A, D)
    basin = "A" if state[0] > state[1] else "B"
    print(f"x0={x0:+.3f}  basin={basin}  (x-y)={state[0] - state[1]:+.4f}")

# Full figure: dynachaos run sec03_transition
