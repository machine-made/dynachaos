#!/usr/bin/env python3
"""
gcm_clusters: GCM cluster states and collective Lyapunov exponent.

Reproduces cluster formation in globally coupled maps (Kaneko, 1990)
and measures the collective Lyapunov exponent of the mean field.

GCM model:
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + (eps/N) sum_j f(x_n(j))

with f(x) = 1 - a x^2.

Figure 1 (cluster spacetime): Spacetime diagram colored by cluster membership,
    showing coherent, ordered, and partially ordered phases.

Figure 2 (collective Lyapunov): lambda_c vs a, identifying collective chaos
    (lambda_c > 0) regions in parameter space.

OUTPUTS: figures/sec10_gcm/gcm_clusters.npz,
         figures/sec10_gcm/gcm_clusters.png,
         figures/sec10_gcm/collective_lyapunov.npz,
         figures/sec10_gcm/collective_lyapunov.png
USAGE:   python src/dynachaos/cml/gcm_clusters.py
"""

import numpy as np

from dynachaos.cml.primitives import (
    cluster_labels_by_tolerance,
    gcm_step,
    sustained_positive_mask,
)
from dynachaos.io.paths import safe_load, section_dir

FIG_DIR = section_dir("sec10_gcm")

CLUSTER_NPZ = FIG_DIR / "gcm_clusters.npz"
CLUSTER_PNG = FIG_DIR / "gcm_clusters.png"
COLL_NPZ = FIG_DIR / "collective_lyapunov.npz"
COLL_PNG = FIG_DIR / "collective_lyapunov.png"


# ---------------------------------------------------------------------------
# Cluster computation
# ---------------------------------------------------------------------------


def detect_clusters(x, tol=1e-6):
    """Backward-compatible alias for shared cluster labelling."""
    return cluster_labels_by_tolerance(x, tol=tol)


def broad_positive_mask(values, threshold=0.02, min_run=4):
    """Backward-compatible alias for sustained positive-run masking."""
    return sustained_positive_mask(values, threshold=threshold, min_run=min_run)


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
# Collective Lyapunov computation
# ---------------------------------------------------------------------------


def compute_collective(
    *,
    eps=0.1,
    n_sites=500,
    a_values=None,
    n_transient=5_000,
    n_measure=50_000,
    renorm_interval=10,
    h_delta=1e-8,
    seed=42,
    output_path=COLL_NPZ,
    progress_interval=10,
):
    """Compute collective Lyapunov exponent vs a.

    Uses the Benettin algorithm adapted to the mean field:
    run two copies of the GCM from nearly identical ICs, track
    mean-field divergence, and periodically renormalize.
    """
    if output_path is not None:
        output_path = FIG_DIR / output_path if isinstance(output_path, str) else output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if a_values is None:
        a_values = np.linspace(1.4, 2.0, 100)
    else:
        a_values = np.asarray(a_values, dtype=float)

    rng = np.random.default_rng(seed)
    lyap_c = np.empty(len(a_values))

    for ia, a in enumerate(a_values):
        # Fresh ICs for each a
        x1 = rng.uniform(-1, 1, n_sites)
        x2 = x1.copy()
        # Uniform perturbation: mean(x2 - x1) = h_delta
        x2 += h_delta

        # Transient (both copies)
        for _ in range(n_transient):
            x1 = gcm_step(x1, a, eps)
            x2 = gcm_step(x2, a, eps)

        # Measure growth rate of mean-field difference.
        # Use the same metric (mean-field magnitude) for both
        # measurement and renormalization to avoid size-dependent bias.
        log_sum = 0.0
        n_renorm = 0

        for t in range(n_measure):
            x1 = gcm_step(x1, a, eps)
            x2 = gcm_step(x2, a, eps)

            # Guard: if x2 diverged between renormalization steps, reset
            if not np.all(np.isfinite(x2)):
                x2 = x1.copy()
                x2 += h_delta

            if (t + 1) % renorm_interval == 0:
                diff = x2 - x1
                h_diff = np.abs(np.mean(diff))

                if h_diff > 0:
                    log_sum += np.log(h_diff / h_delta)
                    n_renorm += 1
                    # Renormalize: rescale so |mean(diff)| = h_delta
                    with np.errstate(over="ignore", invalid="ignore"):
                        x2 = x1 + diff * (h_delta / h_diff)
                    if not np.all(np.isfinite(x2)):
                        x2 = x1.copy()
                        x2 += h_delta
                else:
                    x2 = x1.copy()
                    x2 += h_delta

        if n_renorm > 0:
            lyap_c[ia] = log_sum / (n_renorm * renorm_interval)
        else:
            lyap_c[ia] = 0.0

        if output_path is not None and progress_interval and (ia + 1) % progress_interval == 0:
            print(f"  Collective Lyapunov: {ia + 1}/{len(a_values)}")
            np.savez_compressed(
                output_path,
                a_values=a_values[: ia + 1],
                lyap_c=lyap_c[: ia + 1],
                eps=np.array([eps]),
                N=np.array([n_sites]),
            )

    payload = {
        "a_values": a_values,
        "lyap_c": lyap_c,
        "eps": np.array([eps]),
        "N": np.array([n_sites]),
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


def plot_collective(data):
    """Plot collective Lyapunov exponent vs a."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        reference_line,
        series_style,
        setup,
    )

    setup()

    a_values = data["a_values"]
    lyap_c = data["lyap_c"]
    eps = data["eps"][0]
    N = int(data["N"][0])

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)

    positive = sustained_positive_mask(lyap_c)

    sty = series_style(0)
    ax.plot(
        a_values,
        lyap_c,
        color=sty["color"],
        linewidth=1.1,
        label=r"$\lambda_c$",
    )
    if np.any(positive):
        ax.fill_between(
            a_values,
            0.0,
            lyap_c,
            where=positive,
            alpha=0.22,
            color=COLORS["red"],
            edgecolor=COLORS["red"],
            linewidth=0.8,
            hatch="//",
            label=r"sustained $\lambda_c > 0$",
        )

    reference_line(ax, 0, axis="y", lw=0.7)

    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$\lambda_c$")
    ax.set_title(
        rf"Collective Lyapunov exponent, $\varepsilon = {eps}$, $N = {N}$",
        loc="left",
    )
    apply_axes_polish(ax, kind="double", title_loc="left", grid=False)
    finalize_legend(ax, kind="double", loc="upper left")

    fig.savefig(COLL_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {COLL_PNG}")


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

    # Collective Lyapunov exponent
    try:
        coll_data = safe_load(COLL_NPZ)
        print(f"Loaded {COLL_NPZ}")
    except FileNotFoundError:
        print("Computing collective Lyapunov exponent...")
        coll_data = compute_collective()
    plot_collective(coll_data)


if __name__ == "__main__":
    main()
