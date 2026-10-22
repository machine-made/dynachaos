#!/usr/bin/env python3
"""
pattern_dynamics: CML pattern dynamics and global phase diagram.

Reproduces Kaneko (1989) "Pattern dynamics in spatiotemporal chaos", Physica D
34, 1-41.

CML model (Eq. 1):
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + eps/2 [f(x_n(i+1)) + f(x_n(i-1))]

with f(x) = 1 - a x^2 (logistic map) and periodic boundary conditions.

Phase diagram: a in [1.5, 2.0], eps in [0, 0.4]

OUTPUTS: figures/sec09_pattern/*.npz, *.png
"""

import textwrap

import numpy as np

from dynachaos.cml.primitives import (
    cml_step_logistic as cml_step,
)
from dynachaos.cml.primitives import (
    cml_step_logistic_batch as _cml_step_batch,
)
from dynachaos.io.paths import safe_load, section_dir

FIG_DIR = section_dir("sec09_pattern")

PHASE_NPZ = FIG_DIR / "phase_diagram.npz"
PHASE_PNG = FIG_DIR / "phase_diagram.png"
SPACE_NPZ = FIG_DIR / "space_amplitude.npz"
SPACE_PNG = FIG_DIR / "space_amplitude.png"

PHASE_SCHEMA_VERSION = 2
SPACE_SCHEMA_VERSION = 3
SPACE_CASES = (
    (1.50, 0.10, "Frozen\nrandom", "a"),
    (1.64, 0.10, "Pattern\nselection", "b"),
    (1.72, 0.10, "Defect\nturbulence", "c"),
    (1.80, 0.10, "Pattern competition\nintermittency", "d"),
    (1.90, 0.10, "Fully developed\nturbulence", "e"),
)


def compute_phase_diagram():
    """Compute phase diagram with temporal and spatial activity measures.

    Vectorized: for each eps, all a values run in parallel as a
    (n_a, N) array.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_a = 200
    n_eps = 160
    a_values = np.linspace(1.5, 2.0, n_a)
    eps_values = np.linspace(0.0, 0.4, n_eps)
    N = 100
    n_transient = 5000
    n_sample = 2000

    # a values as column vector for broadcasting with (n_a, N) arrays
    a_col = a_values[:, np.newaxis]  # shape (n_a, 1)

    lam_grid = np.empty((n_eps, n_a))
    spatial_activity_grid = np.empty((n_eps, n_a))
    rng = np.random.default_rng(42)

    for j, eps in enumerate(eps_values):
        # Each row is one CML instance at a different a
        x = rng.uniform(-1, 1, (n_a, N))

        for _ in range(n_transient):
            x = _cml_step_batch(x, a_col, eps)

        # Accumulate variance of the central site
        sum_v = np.zeros(n_a)
        sum_v2 = np.zeros(n_a)
        sum_spatial_activity = np.zeros(n_a)
        for _ in range(n_sample):
            x = _cml_step_batch(x, a_col, eps)
            mid = x[:, N // 2]
            sum_v += mid
            sum_v2 += mid * mid
            sum_spatial_activity += np.mean(np.abs(x - np.roll(x, 1, axis=1)), axis=1)

        mean_v = sum_v / n_sample
        lam_grid[j] = sum_v2 / n_sample - mean_v * mean_v
        spatial_activity_grid[j] = sum_spatial_activity / n_sample

        if (j + 1) % 40 == 0:
            print(f"  Phase diagram: {j + 1}/{n_eps}")
            np.savez_compressed(
                PHASE_NPZ,
                a=a_values,
                eps=eps_values,
                lam=lam_grid,
                spatial_activity=spatial_activity_grid,
                schema_version=np.array([PHASE_SCHEMA_VERSION]),
            )

    np.savez_compressed(
        PHASE_NPZ,
        a=a_values,
        eps=eps_values,
        lam=lam_grid,
        spatial_activity=spatial_activity_grid,
        schema_version=np.array([PHASE_SCHEMA_VERSION]),
    )
    print(f"Saved {PHASE_NPZ}")


# ---------------------------------------------------------------------------
# Space-amplitude plots
# ---------------------------------------------------------------------------


def compute_space_amplitude():
    """Compute space-amplitude snapshots for representative phases."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    N = 100

    results = {}
    rng = np.random.default_rng(42)

    for a, eps, label, tag in SPACE_CASES:
        print(f"  a={a}, eps={eps} ({label})")
        x = rng.uniform(-1, 1, N)
        for _ in range(10_000):
            x = cml_step(x, a, eps)

        snapshots = np.empty((12, N))
        for t in range(12):
            for _ in range(40):
                x = cml_step(x, a, eps)
            snapshots[t] = x.copy()

        results[f"a_{a}_eps_{eps}_snap"] = snapshots
        results[f"a_{a}_eps_{eps}_label"] = np.array([label])
        results[f"a_{a}_eps_{eps}_tag"] = np.array([tag])

    results["params"] = np.array([(a, eps) for a, eps, _, _ in SPACE_CASES])
    results["schema_version"] = np.array([SPACE_SCHEMA_VERSION])
    np.savez_compressed(SPACE_NPZ, **results)
    print(f"Saved {SPACE_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_phase_diagram(data):
    """Plot the global phase diagram."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        CMAP_SEQUENTIAL,
        COLORS,
        add_field_colorbar,
        annotate_on_field,
        apply_axes_polish,
        figure_spec,
        reference_line,
        setup,
    )

    setup()

    a = data["a"]
    eps = data["eps"]
    activity = data["spatial_activity"] if "spatial_activity" in data else data["lam"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    im = ax.pcolormesh(a, eps, activity, cmap=CMAP_SEQUENTIAL, rasterized=True)
    ax.set_xlabel(r"Nonlinearity $a$")
    ax.set_ylabel(r"Coupling $\varepsilon$")
    ax.set_title("Spatial activity map", loc="left")
    add_field_colorbar(fig, im, ax, label=r"$\langle |x_i - x_{i-1}| \rangle$")
    # Pad the left edge so the a=1.50 state marker does not sit exactly on
    # the spine, and drop the eps=0 row from view: at zero coupling the
    # lattice is uncoupled and the plotted quantity has no dynamical
    # meaning there, which otherwise reads as a corrupted-tile artifact.
    a_pad = 0.01 * (float(a.max()) - float(a.min()))
    ax.set_xlim(float(a.min()) - a_pad, float(a.max()))
    eps_floor = 0.02
    ax.set_ylim(eps_floor, float(eps.max()))
    apply_axes_polish(ax, kind="double", title_loc="left", grid=False)

    eps_slice = 0.10
    reference_line(ax, eps_slice, axis="y")
    for a_case, eps_case, _label, tag in SPACE_CASES:
        ax.scatter(
            [a_case],
            [eps_case],
            s=24,
            facecolors=COLORS["offwhite"],
            edgecolors=COLORS["black"],
            linewidths=0.6,
            zorder=3,
        )
        annotate_on_field(
            ax,
            a_case + 0.008,
            eps_case + 0.008,
            f"({tag})",
            ha="left",
            va="bottom",
        )

    fig.savefig(PHASE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PHASE_PNG}")


def plot_space_amplitude(data):
    """Plot space-amplitude snapshots."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        CMAP_SEQUENTIAL,
        apply_axes_polish,
        figure_spec,
        panel_label,
        reference_line,
        setup,
    )

    setup()

    params = data["params"]
    n_panels = len(params)

    spec = figure_spec("grid")
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(spec.figsize[0], spec.figsize[1] * 0.54),
        sharey=True,
    )

    y_min = np.inf
    y_max = -np.inf
    for idx, (a, eps) in enumerate(params):
        key = f"a_{a}_eps_{eps}_snap"
        snapshots = data[key]
        y_min = min(y_min, float(snapshots.min()))
        y_max = max(y_max, float(snapshots.max()))

    for idx, (a, eps) in enumerate(params):
        ax = axes[idx]
        key = f"a_{a}_eps_{eps}_snap"
        label_key = f"a_{a}_eps_{eps}_label"
        tag_key = f"a_{a}_eps_{eps}_tag"
        snapshots = data[key]
        N = snapshots.shape[1]
        sites = np.arange(N)
        # Colour each snapshot by its time index (sequential map) rather than
        # overplotting all 12 in near-identical dark grey: at lw~1.2 that
        # overplotted into solid ink. A thinner, more transparent, colour-
        # ramped line lets the reader see whether curves coincide (frozen)
        # or spread apart (turbulent).
        cmap = plt.get_cmap(CMAP_SEQUENTIAL)
        n_snap = snapshots.shape[0]
        colors = cmap(np.linspace(0.0, 1.0, n_snap))
        for shade_idx, snap in enumerate(snapshots):
            ax.plot(sites, snap, color=colors[shade_idx], lw=0.6, alpha=0.55)
        label = str(data[label_key][0]) if label_key in data else ""
        label = (
            "\n".join(textwrap.wrap(label.replace("\n", " "), width=12, break_long_words=False))
            if label
            else ""
        )
        tag = str(data[tag_key][0]) if tag_key in data else "?"
        ax.set_title(f"$a={a}$\n{label}", loc="left")
        panel_label(ax, f"({tag})")
        ax.set_xlabel("$i$")
        if idx == 0:
            ax.set_ylabel("$x(i)$")
        ax.set_ylim(y_min - 0.05, y_max + 0.05)
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False)
        reference_line(ax, 0.0, axis="y")

    fig.savefig(SPACE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SPACE_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        phase_data = safe_load(PHASE_NPZ)
        print(f"Loaded {PHASE_NPZ}")
        schema_version = int(np.atleast_1d(phase_data.get("schema_version", np.array([0])))[0])
        if schema_version < PHASE_SCHEMA_VERSION or "spatial_activity" not in phase_data:
            raise KeyError("stale phase-diagram cache")
    except FileNotFoundError:
        print("Computing phase diagram...")
        compute_phase_diagram()
        phase_data = safe_load(PHASE_NPZ)
    except KeyError:
        print("Recomputing phase diagram with spatial activity...")
        compute_phase_diagram()
        phase_data = safe_load(PHASE_NPZ)
    plot_phase_diagram(phase_data)

    try:
        space_data = safe_load(SPACE_NPZ)
        print(f"Loaded {SPACE_NPZ}")
        schema_version = int(np.atleast_1d(space_data.get("schema_version", np.array([0])))[0])
        if schema_version < SPACE_SCHEMA_VERSION:
            raise KeyError("stale space-amplitude cache")
    except FileNotFoundError:
        print("Computing space-amplitude plots...")
        compute_space_amplitude()
        space_data = safe_load(SPACE_NPZ)
    except KeyError:
        print("Recomputing space-amplitude plots with updated panel set...")
        compute_space_amplitude()
        space_data = safe_load(SPACE_NPZ)
    plot_space_amplitude(space_data)


if __name__ == "__main__":
    main()
