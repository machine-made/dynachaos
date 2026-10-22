#!/usr/bin/env python3
"""
fractalization: Fractalization of torus near the onset of chaos.

Reproduces Kaneko (1984) "Fractalization of Torus", PTP 71(5), 1112-1115.

As a control parameter approaches the chaotic threshold, a smooth torus
develops wrinkles at ever finer scales and becomes fractal.  The fractal
(correlation) dimension increases from 1 (smooth torus) toward ~1.3-1.5
(fractal torus) at the chaos onset.

Uses the delayed logistic map (Kaneko 1984):
    x_{n+1} = A x_n + (1 - A)(1 - D y_n^2)
    y_{n+1} = x_n

with A = 0.3.  Torus appears at D_c = 1/(1-A) ≈ 1.429.
Fractalization visible for D ∈ [1.90, 1.95].

Figures:
  - Attractor portraits showing progressive fractalization
  - Correlation dimension D_2 vs D (Grassberger-Procaccia)

OUTPUTS: figures/sec07_fractalization/*.npz, *.png
USAGE:   python src/dynachaos/maps/fractalization.py
"""

import numpy as np

from dynachaos.diagnostics.correlation import correlation_dimension
from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps._iter import run_animation_sweep, trajectory_after_transient
from dynachaos.maps.delayed_logistic import delayed_logistic

FIG_DIR = section_dir("sec07_fractalization")

FRAC_NPZ = FIG_DIR / "fractal_attractors.npz"
FRAC_PNG = FIG_DIR / "fractal_attractors.png"
DIM_NPZ = FIG_DIR / "correlation_dimension.npz"
DIM_PNG = FIG_DIR / "correlation_dimension.png"
ANIM_NPZ = FIG_DIR / "fractalization_animation.npz"
ANIM_GIF = FIG_DIR / "fractalization_animation.gif"


# ---------------------------------------------------------------------------
# Iteration helper
# ---------------------------------------------------------------------------


def iterate(A, D, n_transient=20_000, n_record=100_000, x0=None):
    """Iterate and return trajectory points."""
    if x0 is None:
        fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
        x0 = np.array([fp + 0.01, fp - 0.01])

    return trajectory_after_transient(
        x0,
        lambda state: delayed_logistic(state, A, D),
        n_transient,
        n_record,
    )


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_attractors():
    """Compute attractors at D values showing fractalization."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A = 0.3
    # D values: smooth torus -> wrinkled -> fractal -> chaos
    D_values = [1.75, 1.86, 1.90, 1.92, 1.94, 1.945]
    labels = ["smooth torus", "torus", "wrinkled torus", "fractal torus", "onset of chaos", "chaos"]

    results = {}
    for D, label in zip(D_values, labels):
        print(f"  D={D} ({label})")
        traj = iterate(A, D, n_transient=50_000, n_record=500_000)
        results[f"D_{D}_traj"] = traj

    results["D_values"] = np.array(D_values)
    results["A"] = np.array([A])
    np.savez_compressed(FRAC_NPZ, **results)
    print(f"Saved {FRAC_NPZ}")


def compute_dimensions():
    """Compute correlation dimension D_2 vs D."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A = 0.3
    n_params = 200
    D_values = np.linspace(1.70, 2.00, n_params)
    D2_values = np.empty(n_params)
    D2_err_values = np.empty(n_params)

    for i, D in enumerate(D_values):
        traj = iterate(A, D, n_transient=50_000, n_record=100_000)
        D2, _, _, D2_err, _, _ = correlation_dimension(
            traj, n_r=50, max_pairs=1_000_000, return_stderr=True
        )
        D2_values[i] = D2
        D2_err_values[i] = D2_err

        if (i + 1) % 20 == 0:
            print(f"  Dimension: {i + 1}/{n_params}")
            np.savez_compressed(
                DIM_NPZ,
                D=D_values[: i + 1],
                D2=D2_values[: i + 1],
                D2_err=D2_err_values[: i + 1],
                A=np.array([A]),
            )

    np.savez_compressed(DIM_NPZ, D=D_values, D2=D2_values, D2_err=D2_err_values, A=np.array([A]))
    print(f"Saved {DIM_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_attractors(data):
    """Plot attractor portraits showing fractalization."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, panel_label, setup

    setup()

    D_values = data["D_values"]
    labels = [
        "smooth torus",
        "weakly wrinkled",
        "wrinkled torus",
        "fractalizing torus",
        "onset of chaos",
        "chaos",
    ]

    all_x = np.concatenate([data[f"D_{D}_traj"][:, 0] for D in D_values])
    all_y = np.concatenate([data[f"D_{D}_traj"][:, 1] for D in D_values])
    pad = 0.05
    x_span = all_x.max() - all_x.min()
    y_span = all_y.max() - all_y.min()
    xlim = (all_x.min() - pad * x_span, all_x.max() + pad * x_span)
    ylim = (all_y.min() - pad * y_span, all_y.max() + pad * y_span)

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 3, figsize=spec.figsize)
    fig.subplots_adjust(hspace=0.48, wspace=0.24)
    axes_flat = axes.flatten()

    for idx, D in enumerate(D_values):
        ax = axes_flat[idx]
        traj = data[f"D_{D}_traj"]
        # s=0.012 is far below one device pixel even at 600 dpi: dense panels
        # hid this, but the sparser panels (few hundred distinct points)
        # rendered as hollow white-centred rings instead of filled ink.
        # A larger marker with lower alpha keeps the dense panels from
        # saturating while making sparse points read as dots, not rings.
        ax.scatter(traj[:, 0], traj[:, 1], s=0.35, c=COLORS["black"], alpha=0.05, rasterized=True)
        panel_label(ax, chr(ord("a") + idx))
        ax.set_title(f"$D = {D}$\n{labels[idx]}", loc="left")
        if idx // 3 == 1:
            ax.set_xlabel("$x$")
        if idx % 3 == 0:
            ax.set_ylabel("$y$")
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False, equal=True)
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(
        r"Fractalization of torus, $\alpha = 0.3$",
        fontsize=spec.title_size,
        y=1.02,
    )
    fig.savefig(FRAC_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FRAC_PNG}")


def plot_dimension(data):
    """Plot correlation dimension D_2 vs D."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        reference_line,
        setup,
    )

    setup()

    D = data["D"]
    D2 = data["D2"]
    D2_err = data["D2_err"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    ax.plot(D, D2, color=COLORS["black"], linestyle="-", lw=0.8)
    ax.fill_between(
        D,
        D2 - D2_err,
        D2 + D2_err,
        alpha=0.15,
        color=COLORS["black"],
        linewidth=0,
        zorder=0,
    )
    reference_line(ax, 1.0, axis="y", lw=0.6, alpha=0.7, label="$D_2 = 1$ (smooth torus)")
    ax.axvspan(1.92, 1.95, color=COLORS["grey"], alpha=0.10, zorder=0)
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"Correlation dimension $D_2$")
    ax.set_title(r"Delayed logistic map, $\alpha = 0.3$", loc="left")
    # The old fixed (0.9, 1.45) ylim silently clipped the D2-D2_err dip near
    # D~1.94 (down to ~0.886), cutting off the exact "local dips near the
    # transition" the caption asks the reader to see. Derive the limits from
    # the data (with the +/-1 sigma band included) so nothing is cropped.
    y_lo = float(np.min(D2 - D2_err))
    y_hi = float(np.max(D2 + D2_err))
    y_pad = 0.05 * (y_hi - y_lo)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    apply_axes_polish(ax, kind="double", title_loc="left", grid=False)
    finalize_legend(ax, kind="single", loc="upper left")

    fig.savefig(DIM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {DIM_PNG}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def compute_animation_data():
    """Sweep D from 1.75 to 1.96 for fractalization animation."""
    A = 0.3
    D_sweep = np.linspace(1.75, 1.96, 200)

    def iterate_fn(D):
        return iterate(A, D, n_transient=30_000, n_record=5_000)

    run_animation_sweep(iterate_fn, D_sweep, ANIM_NPZ, n_plot=5_000)


def make_animation_gif(data):
    """Create GIF of fractalization progression."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"],
        data["all_x"],
        data["all_y"],
        ANIM_GIF,
        title_template=r"Fractalization of torus, $\alpha = 0.3$, $D = {param_value}$",
        param_name="D",
        param_fmt=".4f",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        frac_data = safe_load(FRAC_NPZ)
        print(f"Loaded {FRAC_NPZ}")
    except FileNotFoundError:
        print("Computing fractal attractors...")
        compute_attractors()
        frac_data = safe_load(FRAC_NPZ)
    plot_attractors(frac_data)

    try:
        dim_data = safe_load(DIM_NPZ)
        print(f"Loaded {DIM_NPZ}")
    except FileNotFoundError:
        print("Computing correlation dimensions...")
        compute_dimensions()
        dim_data = safe_load(DIM_NPZ)
    plot_dimension(dim_data)

    # Animation
    try:
        anim_data = safe_load(ANIM_NPZ)
        print(f"Loaded {ANIM_NPZ}")
    except FileNotFoundError:
        print("Computing fractalization animation data...")
        compute_animation_data()
        anim_data = safe_load(ANIM_NPZ)
    make_animation_gif(anim_data)


if __name__ == "__main__":
    main()
