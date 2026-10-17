# Reduced preview (2 D_B values, 1000 points). Full figure: dynachaos run sec06_three_torus
# (x_n, z_n) projections of the 4D coupled delayed logistic map (alpha=0.4, eps=5e-3).
import numpy as np

from dynachaos.maps._iter import trajectory_after_transient
from dynachaos.maps.coupled_delayed import coupled_delayed

alpha, eps = 0.4, 5e-3
for DB, label in ((2.37, "near-T^3"), (2.55, "developed chaos")):
    DA = DB + 0.1
    traj = trajectory_after_transient(
        np.array([0.5, 0.5, 0.3, 0.3]),
        lambda state, DA=DA, DB=DB: coupled_delayed(state, alpha, DA, DB, eps),
        n_transient=2000,
        n_record=1000,
        project_fn=lambda state: state[[0, 2]],
    )
    print(f"D_B={DB} ({label})  x range=[{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}]")
