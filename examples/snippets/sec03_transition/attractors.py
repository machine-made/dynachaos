# Reduced preview (3 states, 2000 points). Full figure: dynachaos run sec03_transition
# Representative attractor portraits at eps=0.1 along the broken-symmetry route (2T -> 4T -> 4C).
# Panels (a)-(b) use initial condition (0.1, 0.6); panels (c)-(f) use (0.1, 0.2).
import numpy as np

from dynachaos.maps._iter import trajectory_after_transient
from dynachaos.maps.coupled_logistic import coupled_logistic

eps = 0.1
cases = (
    (1.10, "2T", np.array([0.1, 0.6])),
    (1.25, "4T", np.array([0.1, 0.2])),
    (1.373, "4C", np.array([0.1, 0.2])),
)

for a, label, init_state in cases:
    traj = trajectory_after_transient(
        init_state,
        lambda state, a=a: coupled_logistic(state, a, eps),
        n_transient=5000,
        n_record=2000,
    )
    print(f"a={a:.4f} ({label})  x range=[{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}]")
