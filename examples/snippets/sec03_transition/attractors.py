# Figure: representative attractor portraits at eps=0.1 along the
# broken-symmetry route (2T -> 4T -> 8T -> 4C).
import numpy as np

from dynachaos.maps._iter import trajectory_after_transient
from dynachaos.maps.coupled_logistic import coupled_logistic

D = 0.1
for A, label in ((1.10, "2T"), (1.25, "4T"), (1.373, "4C")):
    traj = trajectory_after_transient(
        np.array([0.1, 0.6]),
        lambda state, A=A: coupled_logistic(state, A, D),
        n_transient=5000,
        n_record=2000,
    )
    print(f"a={A:.4f} ({label})  x range=[{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}]")

# Full figure: dynachaos run sec03_transition
