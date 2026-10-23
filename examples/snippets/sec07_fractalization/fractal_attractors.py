# Reduced preview (3 D values, 2000 points). Full figure: dynachaos run sec07_fractalization
# Attractor portraits of the delayed logistic map (alpha=0.3) showing progressive fractalization.
from dynachaos.maps.fractalization import iterate

alpha = 0.3
for D, label in ((1.75, "smooth torus"), (1.92, "fractal torus"), (1.945, "chaos")):
    traj = iterate(alpha, D, n_transient=2000, n_record=2000)
    x_lo, x_hi = traj[:, 0].min(), traj[:, 0].max()
    print(f"D={D} ({label})  points={traj.shape[0]}  x range=[{x_lo:.3f}, {x_hi:.3f}]")
