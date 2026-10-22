# Figure: cluster count of a globally coupled logistic map (Kaneko 1990);
# sites that converge to (near-)identical states form a coherent cluster.
from dynachaos.cml.gcm_clusters import compute_clusters

for eps in (0.1, 0.3):
    result = compute_clusters(
        a=1.55,
        eps=eps,
        n_sites=40,
        n_transient=2000,
        n_record=10,
        output_path=None,
    )
    n_clusters = len(set(result["cluster_labels"][-1]))
    print(f"eps={eps:.2f}  clusters at final step={n_clusters}")

# Full figure: dynachaos run sec10_gcm
