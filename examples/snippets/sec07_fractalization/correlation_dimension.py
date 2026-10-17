# Reduced preview (3 D values, 3000 points). Full figure: dynachaos run sec07_fractalization
# Correlation dimension D_2(D) of the delayed logistic map (alpha=0.3).
from dynachaos.diagnostics.correlation import correlation_dimension
from dynachaos.maps.fractalization import iterate

alpha = 0.3
for D in (1.75, 1.90, 1.945):
    traj = iterate(alpha, D, n_transient=2000, n_record=3000)
    res = correlation_dimension(traj, n_r=30, max_pairs=200_000)
    D2 = res[0]
    print(f"D={D}  D2={D2:.3f}")
