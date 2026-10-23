# Reduced preview (grid 4 x 2). Full figure: dynachaos run sec03_transition
# Symmetry breaking <|x-y|> for the coupled logistic map over an (a, eps) grid.
import numpy as np

from dynachaos.maps.coupled_logistic import coupled_logistic

for eps in (0.05, 0.1):
    for a in (1.0, 1.2, 1.35, 1.5):
        state = np.array([0.1, 0.2])
        for _ in range(5000):
            state = coupled_logistic(state, a, eps)
        asym = 0.0
        for _ in range(2000):
            state = coupled_logistic(state, a, eps)
            asym += abs(state[0] - state[1])
        asym /= 2000
        print(f"eps={eps:.2f}  a={a:.2f}  <|x-y|>={asym:.4f}")
