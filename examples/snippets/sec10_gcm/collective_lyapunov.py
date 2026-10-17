# Collective Lyapunov exponent lambda_c of the GCM mean field (N=500, eps=0.1).
# Full figure: dynachaos run sec10_gcm
from dynachaos.cml.gcm_clusters import compute_collective

result = compute_collective(
    eps=0.1,
    n_sites=500,
    a_values=[1.5, 1.85, 1.99],
    n_transient=200,
    n_measure=2000,
    renorm_interval=10,
    output_path=None,
    progress_interval=0,
)
for a, lam_c in zip(result["a_values"], result["lyap_c"]):
    print(f"a={a:.2f}  lambda_c={lam_c:.4f}")
