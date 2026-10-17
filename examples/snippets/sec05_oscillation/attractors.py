# Reduced preview (3 D values). Full figure: dynachaos run sec05_oscillation
# Attractor portraits of the delayed logistic map at alpha=0.3.
from dynachaos.maps.delayed_logistic import compute_attractor

alpha = 0.3
for D, label in ((1.55, "torus"), (1.94, "chaos"), (2.16, "developed chaos")):
    traj = compute_attractor(alpha, D, n_transient=3000, n_plot=2000)
    print(f"D={D:.2f} ({label})  y range=[{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}]")
