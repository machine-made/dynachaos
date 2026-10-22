# Figure: attractor portraits of the delayed logistic map at A=0.3, spanning
# torus, periodic window, and chaotic D.
from dynachaos.maps.delayed_logistic import compute_attractor

A = 0.3
for D, label in ((1.55, "torus"), (1.94, "periodic window"), (2.16, "chaos")):
    traj = compute_attractor(A, D, n_transient=3000, n_plot=2000)
    print(f"D={D:.2f} ({label})  y range=[{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}]")

# Full figure: dynachaos run sec05_oscillation
