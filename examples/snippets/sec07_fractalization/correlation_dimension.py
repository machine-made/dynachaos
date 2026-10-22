# Figure: correlation dimension D_2(D) rising from 1 (smooth torus) toward
# ~1.3-1.5 as the delayed logistic map approaches chaos onset.
from dynachaos.diagnostics.correlation import correlation_dimension
from dynachaos.maps.fractalization import iterate

A = 0.3
for D in (1.75, 1.90, 1.945):
    traj = iterate(A, D, n_transient=2000, n_record=3000)
    D2, _, _, _, _ = correlation_dimension(traj, n_r=30, max_pairs=200_000)
    print(f"D={D}  D2={D2:.3f}")

# Full figure: dynachaos run sec07_fractalization
