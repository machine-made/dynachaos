"""Kaneko CML spatiotemporal-intermittency spine figure."""

from pathlib import Path

import numpy as np

from dynachaos.cml.spatiotemporal import model_A_f, simulate_cml
from dynachaos.io.paths import safe_load, section_dir

SECTION_ID = "sec12_intermittency"
FIG_DIR = section_dir(SECTION_ID)
OUTPUT_NPZ = FIG_DIR / "sti_spine.npz"
OUTPUT_PNG = FIG_DIR / "sti_spine.png"
_THIS_FILE = Path(__file__).resolve()

REQUIRED_KEYS = (
    "schema_version",
    "source_file",
    "seed",
    "model_a_parameter",
    "display_eps",
    "sweep_eps",
    "spacetime",
    "turbulent_mask",
    "turbulent_fraction",
    "laminar_cluster_sizes",
    "cluster_size_values",
    "cluster_size_probabilities",
    "cluster_decay_rate",
)

CAPTION_NOTE = (
    "Kaneko Model-A CML spatiotemporal intermittency: sharp turbulent-fraction onset "
    "and an approximately exponential laminar cluster-size tail."
)


def compute(
    output_path=OUTPUT_NPZ,
    *,
    seed=20260602,
    n_sites=512,
    n_transient=1_200,
    n_record=480,
):
    """Compute deterministic CML/STI diagnostics and optionally cache them."""
    display_eps = 0.08
    sweep_eps = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12])
    spacetime = _simulate_model_a(
        display_eps,
        seed=seed,
        n_sites=n_sites,
        n_transient=n_transient,
        n_record=n_record,
    )
    turbulent_mask = _burst_mask(spacetime)
    laminar_clusters = _periodic_run_lengths(~turbulent_mask)

    turbulent_fraction = []
    for idx, eps in enumerate(sweep_eps):
        sample = _simulate_model_a(
            float(eps),
            seed=seed + idx + 1,
            n_sites=n_sites,
            n_transient=max(800, n_transient // 2),
            n_record=max(240, n_record // 2),
        )
        turbulent_fraction.append(float(_burst_mask(sample).mean()))
    turbulent_fraction = np.asarray(turbulent_fraction, dtype=np.float64)

    cluster_values, cluster_counts = np.unique(laminar_clusters, return_counts=True)
    cluster_probabilities = cluster_counts / np.sum(cluster_counts)
    fit_mask = (cluster_values >= 2) & (cluster_probabilities > 0.0)
    coeffs = np.polyfit(
        cluster_values[fit_mask],
        np.log(cluster_probabilities[fit_mask]),
        deg=1,
    )
    cluster_decay_rate = float(-coeffs[0])

    payload = {
        "schema_version": np.array([2], dtype=np.int64),
        "source_file": np.array([_source_file_label()]),
        "seed": np.array([seed], dtype=np.int64),
        "model_a_parameter": np.array([-0.01], dtype=np.float64),
        "display_eps": np.array([display_eps], dtype=np.float64),
        "sweep_eps": sweep_eps.astype(np.float64),
        "spacetime": spacetime.astype(np.float64),
        "turbulent_mask": turbulent_mask.astype(np.bool_),
        "turbulent_fraction": turbulent_fraction.astype(np.float64),
        "laminar_cluster_sizes": laminar_clusters.astype(np.int64),
        "cluster_size_values": cluster_values.astype(np.int64),
        "cluster_size_probabilities": cluster_probabilities.astype(np.float64),
        "cluster_decay_rate": np.array(cluster_decay_rate, dtype=np.float64),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


def plot(data, output_path=OUTPUT_PNG):
    """Render the CML/STI spine figure."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        CMAP_SPACETIME,
        COLORS,
        annotate_on_field,
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
    )

    setup()

    spacetime = np.asarray(data["spacetime"], dtype=np.float64)
    turbulent_mask = np.asarray(data["turbulent_mask"], dtype=np.bool_)
    eps = np.asarray(data["sweep_eps"], dtype=np.float64)
    rho = np.asarray(data["turbulent_fraction"], dtype=np.float64)
    cluster_values = np.asarray(data["cluster_size_values"], dtype=np.float64)
    cluster_prob = np.asarray(data["cluster_size_probabilities"], dtype=np.float64)
    rate = float(np.asarray(data["cluster_decay_rate"], dtype=np.float64))
    display_eps = float(np.asarray(data["display_eps"], dtype=np.float64)[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = figure_spec("grid")
    fig = plt.figure(figsize=spec.figsize, constrained_layout=True)
    axes = fig.subplot_mosaic(
        [
            ["space", "space", "space", "cbar"],
            ["rho", ".", "clusters", "."],
            ["caption", "caption", "caption", "caption"],
        ],
        gridspec_kw={"height_ratios": [1.15, 1.0, 0.13], "width_ratios": [1.0, 0.2, 1.0, 0.045]},
    )
    ax_space = axes["space"]
    ax_rho = axes["rho"]
    ax_clusters = axes["clusters"]
    ax_caption = axes["caption"]
    ax_cbar = axes["cbar"]

    vmin, vmax = np.percentile(spacetime, [1.0, 99.0])
    # A 512x480 field resampled down to web-thumbnail width with "nearest"
    # interpolation aliases into visual noise; an antialiased (area-
    # averaging) resampling filter keeps the turbulent/laminar texture
    # interpretable at display size.
    image = ax_space.imshow(
        spacetime,
        aspect="auto",
        cmap=CMAP_SPACETIME,
        origin="lower",
        interpolation="antialiased",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(image, cax=ax_cbar, label="$x_i(n)$")
    cbar.ax.tick_params(labelsize=spec.tick_size)
    ax_space.contour(
        turbulent_mask.astype(float),
        levels=[0.5],
        colors=[COLORS["black"]],
        linewidths=0.25,
        alpha=0.35,
    )
    annotate_on_field(
        ax_space,
        0.86 * spacetime.shape[1],
        0.10 * spacetime.shape[0],
        rf"$\varepsilon={display_eps:.2f}$",
    )
    ax_space.set_title(
        rf"Kaneko CML spatiotemporal intermittency, $\varepsilon={display_eps:.2f}$",
        loc="left",
    )
    ax_space.set_xlabel("site $i$")
    ax_space.set_ylabel("time $n$")

    ax_rho.plot(eps, rho, marker="o", ms=4.0, lw=1.0, color=COLORS["black"])
    ax_rho.set_title(r"Turbulent fraction", loc="left")
    ax_rho.set_xlabel(r"coupling $\varepsilon$")
    ax_rho.set_ylabel(r"$\rho$")

    ax_clusters.semilogy(
        cluster_values,
        cluster_prob,
        marker="o",
        ms=3.5,
        lw=0,
        color=COLORS["black"],
    )
    finite = (cluster_values >= 2) & (cluster_prob > 0.0)
    ref_x = np.array([cluster_values[finite][0], cluster_values[finite][-1]], dtype=np.float64)
    ref_y = cluster_prob[finite][0] * np.exp(-rate * (ref_x - ref_x[0]))
    ax_clusters.semilogy(ref_x, ref_y, lw=1.0, ls="--", color=COLORS["grey"])
    ax_clusters.set_title(r"Cluster-size tail", loc="left")
    ax_clusters.set_xlabel("domain size")
    ax_clusters.set_ylabel("probability")

    for label, ax in zip(("a", "b", "c"), (ax_space, ax_rho, ax_clusters), strict=True):
        panel_label(ax, f"({label})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        apply_axes_polish(ax, kind="grid")

    ax_caption.axis("off")
    ax_caption.text(
        0.5,
        0.5,
        CAPTION_NOTE,
        ha="center",
        va="center",
        fontsize=spec.tick_size,
        color=COLORS["black"],
    )

    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main():
    """Load or compute the CML/STI spine cache, then render the figure."""
    try:
        data = safe_load(OUTPUT_NPZ)
        missing = tuple(key for key in REQUIRED_KEYS if key not in data.files)
        stale_schema = not missing and int(np.asarray(data["schema_version"])[0]) != 2
        if missing or stale_schema:
            data.close()
            reason = "missing keys (" + ", ".join(missing) + ")" if missing else "stale schema"
            print(f"Cache {OUTPUT_NPZ} {reason}; recomputing...")
            data = compute()
    except FileNotFoundError:
        data = compute()

    try:
        plot(data)
    finally:
        close = getattr(data, "close", None)
        if close is not None:
            close()


def _burst_mask(field, delta=0.05):
    """Kaneko 1985 burst criterion: S(i)=1 if |x(i+1)-x(i)| > delta."""
    return np.abs(field - np.roll(field, -1, axis=1)) > delta


def _simulate_model_a(eps, *, seed, n_sites, n_transient, n_record):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, n_sites)
    return simulate_cml(
        model_A_f,
        model_A_f,
        eps,
        N=n_sites,
        n_transient=n_transient,
        n_record=n_record,
        x0=x0,
    )


def _periodic_run_lengths(mask):
    rows = np.asarray(mask, dtype=np.bool_)
    if rows.ndim != 2:
        raise ValueError("mask must be a 2D spacetime array")

    lengths = []
    for row in rows:
        lengths.extend(_periodic_row_runs(row))
    return np.asarray(lengths, dtype=np.int64)


def _periodic_row_runs(row):
    values = np.asarray(row, dtype=np.bool_)
    if values.size == 0 or not np.any(values):
        return []
    if np.all(values):
        return [int(values.size)]

    start = int(np.flatnonzero(~values)[0])
    ordered = np.concatenate((values[start + 1 :], values[: start + 1]))
    lengths = []
    current = 0
    for flag in ordered:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _source_file_label():
    # as_posix(), not str(): this label is written into the .npz payload as
    # provenance, so it is data rather than a filesystem operation. str() would
    # render it with the OS separator and a cache generated on Windows would
    # record "src\\dynachaos\\..." while every other platform records
    # "src/dynachaos/...", making the metadata non-portable.
    try:
        return _THIS_FILE.relative_to(_THIS_FILE.parents[3]).as_posix()
    except ValueError:
        return _THIS_FILE.name


if __name__ == "__main__":
    main()
