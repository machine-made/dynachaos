# Figure: collective Lyapunov exponent lambda_c of the mean field of a
# globally coupled logistic map (Kaneko 1990); lambda_c > 0 marks
# collective chaos of the macroscopic mean field itself.
from dynachaos.cml.gcm_clusters import compute_collective

result = compute_collective(
    eps=0.1,
    n_sites=60,
    a_values=[1.5, 1.85, 1.99],
    n_transient=200,
    n_measure=2000,
    renorm_interval=10,
    output_path=None,
    progress_interval=0,
)
for a, lam_c in zip(result["a_values"], result["lyap_c"]):
    print(f"a={a:.2f}  lambda_c={lam_c:.4f}")

# Full figure: dynachaos run sec10_gcm
