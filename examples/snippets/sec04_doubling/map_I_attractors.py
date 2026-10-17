# Figure: representative (X, Y) projections of Map (I) across torus doubling (alpha=0.4).
# Full figure: dynachaos run sec04_doubling
import numpy as np

from dynachaos.maps.torus_doubling import iterate_map, map_I

alpha = 0.4
x0 = np.array([0.5, 0.5, 0.5])
for D, label in ((2.11, "torus"), (2.16, "2x torus"), (2.19, "chaos")):
    traj = iterate_map(map_I, x0, alpha, D, n_transient=3000, n_plot=2000)
    print(f"D={D:.2f} ({label})  X range=[{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}]")
