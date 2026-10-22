#!/usr/bin/env python3
"""
gcm_clusters: GCM cluster states.

Reproduces cluster formation in globally coupled maps (Kaneko, 1990).

GCM model:
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + (eps/N) sum_j f(x_n(j))

with f(x) = 1 - a x^2.

Spacetime diagram colored by cluster membership, showing coherent, ordered,
and partially ordered phases.  The collective exponent of the mean field is in
``gcm_collective_fsle``; the largest microscopic exponent is in ``gcm_lyapunov_max``.

OUTPUTS: figures/sec10_gcm/gcm_clusters.npz,
         figures/sec10_gcm/gcm_clusters.png
USAGE:   python src/dynachaos/cml/gcm_clusters.py
"""

import numpy as np

from dynachaos.cml.primitives import (
    cluster_labels_by_tolerance,
    gcm_step,
)
from dynachaos.io.paths import safe_load, section_dir

FIG_DIR = section_dir("sec10_gcm")

CLUSTER_NPZ = FIG_DIR / "gcm_clusters.npz"
CLUSTER_PNG = FIG_DIR / "gcm_clusters.png"


# ---------------------------------------------------------------------------
# Cluster computation
# ---------------------------------------------------------------------------


def detect_clusters(x, tol=1e-6):
    """Backward-compatible alias for shared cluster labelling."""
    return cluster_labels_by_tolerance(x, tol=tol)


def compute_clusters(
    *,
    a=1.55,
    eps=0.1,
    n_sites=100,
    n_transient=20_000,
    n_record=500,
    seed=42,
    output_path=CLUSTER_NPZ,
):
    """Compute GCM cluster spacetime diagram."""
    if output_path is not None:
        output_path = FIG_DIR / output_path if isinstance(output_path, str) else output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n_sites)

    # Transient
    print("  Running transient...")
    for _ in range(n_transient):
        x = gcm_step(x, a, eps)

    # Record
    print("  Recording cluster states...")
    cluster_labels = np.empty((n_record, n_sites), dtype=int)
    x_record = np.empty((n_record, n_sites))

    for t in range(n_record):
        x = gcm_step(x, a, eps)
        cluster_labels[t] = cluster_labels_by_tolerance(x)
        x_record[t] = x

    payload = {
        "cluster_labels": cluster_labels,
        "x_record": x_record,
        "a": np.array([a]),
        "eps": np.array([eps]),
        "N": np.array([n_sites]),
        "n_transient": np.array([n_transient]),
        "n_record": np.array([n_record]),
    }
    if output_path is not None:
        np.savez_compressed(
            output_path,
            **payload,
        )
        print(f"Saved {output_path}")
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_clusters(data):
    """Plot sorted site states to expose the emergent cluster partition."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        CMAP_SEQUENTIAL,
        add_field_colorbar,
        annotate_on_field,
        apply_axes_polish,
        figure_spec,
        setup,
    )

    setup()

    cluster_labels = data["cluster_labels"]
    x_record = data["x_record"]
    a = data["a"][0]
    eps = data["eps"][0]

    order = np.argsort(x_record, axis=1)
    sorted_x = np.take_along_axis(x_record, order, axis=1)
    n_clusters = np.array([len(np.unique(row)) for row in cluster_labels])

    spec = figure_spec("double")
    # Taller panel than the default "double" height: 500 recorded time steps
    # resampled into a short, wide thumbnail aliases badly (moire in the
    # striped transition bands). Giving the vertical axis more room, plus an
    # antialiased (area-averaging) resampling filter instead of "nearest",
    # keeps the decimation from beating against the period-2 structure.
    fig, ax = plt.subplots(figsize=(spec.figsize[0], spec.figsize[1] * 1.8))

    im = ax.imshow(
        sorted_x,
        aspect="auto",
        origin="lower",
        cmap=CMAP_SEQUENTIAL,
        interpolation="antialiased",
    )

    ax.set_xlabel("Ordered site rank")
    ax.set_ylabel("Time step $n$")
    ax.set_title(
        rf"Sorted site states reveal clustering, $a = {a}$, $\varepsilon = {eps}$",
        loc="left",
    )
    apply_axes_polish(ax, kind="double", title_loc="left", grid=False)

    add_field_colorbar(
        fig,
        im,
        ax,
        label=r"Sorted state value $x_{(k)}$",
        pad=0.02,
        shrink=0.85,
    )
    annotate_on_field(
        ax,
        0.02,
        0.02,
        rf"clusters per row: {n_clusters.min()} to {n_clusters.max()}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=spec.legend_size,
    )

    fig.savefig(CLUSTER_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {CLUSTER_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Cluster spacetime diagram
    try:
        cluster_data = safe_load(CLUSTER_NPZ)
        print(f"Loaded {CLUSTER_NPZ}")
    except FileNotFoundError:
        print("Computing GCM cluster states...")
        cluster_data = compute_clusters()
    plot_clusters(cluster_data)


if __name__ == "__main__":
    main()
