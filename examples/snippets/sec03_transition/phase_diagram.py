# Figure: symmetry breaking <|x-y|> and finite-time Lyapunov exponent lambda_1
# over a small (a, eps) grid for the coupled logistic map.
import numpy as np

from dynachaos.maps.coupled_logistic import coupled_logistic

D = 0.1
for A in (1.0, 1.2, 1.35, 1.5):
    state = np.array([0.1, 0.2])
    for _ in range(5000):
        state = coupled_logistic(state, A, D)
    asym = 0.0
    for _ in range(2000):
        state = coupled_logistic(state, A, D)
        asym += abs(state[0] - state[1])
    asym /= 2000
    print(f"a={A:.2f}  <|x-y|>={asym:.4f}")

# Full figure: dynachaos run sec03_transition
