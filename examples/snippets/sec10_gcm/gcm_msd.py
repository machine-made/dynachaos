# Reduced preview (N in 50..5000, a = 1.99, single seed). Full figure: dynachaos run sec10_gcm
# Mean-square deviation (variance) of the mean field h_n of a globally coupled logistic map vs N.
# Full figure averages 8 seeds across 5 a values with saturation fit MSD(N) = A/N + B (B > 0).
from dynachaos.cml.globally_coupled import compute_msd_sweep

result = compute_msd_sweep(
    a_values=(1.99,),
    N_values=(50, 500, 5000),
    n_seeds=1,
    n_transient=300,
    n_sample=400,
    output_path=None,
)
for N, msd in zip(result["N_values"], result["msd_mean"][0], strict=True):
    print(f"N={N:5d}  Var(h)={msd:.3e}")
