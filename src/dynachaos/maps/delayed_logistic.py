#!/usr/bin/env python3
"""
delayed_logistic: Oscillation and doubling of torus in the delayed logistic map.

Reproduces Kaneko (1984) "Oscillation and Doubling of Torus", PTP 72(2), 202-215.

Map (Eq. 2.1):
    x_{n+1} = A x_n + (1 - A)(1 - D y_n^2)
    y_{n+1} = x_n

Fixed point: x = y = (sqrt(1+4D) - 1) / (2D)
Neimark-Sacker bifurcation at D_c = (3-2A)/(4(1-A)^2), torus appears for D > D_c.

Figures:
  - Attractor portraits at A=0.3 for twelve D values from 1.55 to 2.16
    (extending Kaneko's Fig. 1 with finer progression)
  - Lyapunov exponents along the D path
  - Animated GIF of attractor evolution across D

OUTPUTS: figures/sec05_oscillation/attractors.npz,
         figures/sec05_oscillation/attractors.png,
         figures/sec05_oscillation/lyapunov_vs_D.npz,
         figures/sec05_oscillation/lyapunov_vs_D.png,
         figures/sec05_oscillation/attractors_animation.gif
USAGE:   python src/dynachaos/maps/delayed_logistic.py
"""

import numpy as np

from dynachaos.io.paths import load_or_compute_payload, safe_load, section_dir
from dynachaos.io.paths import write_payload as _io_write_payload
from dynachaos.maps._iter import run_animation_sweep, trajectory_after_transient

FIG_DIR = section_dir("sec05_oscillation")
ATTR_NPZ = FIG_DIR / "attractors.npz"
ATTR_PNG = FIG_DIR / "attractors.png"
LYAP_NPZ = FIG_DIR / "lyapunov_vs_D.npz"
LYAP_PNG = FIG_DIR / "lyapunov_vs_D.png"
ANIM_NPZ = FIG_DIR / "attractors_animation.npz"
ANIM_GIF = FIG_DIR / "attractors_animation.gif"
LOCK_NPZ = FIG_DIR / "locking_sequence.npz"
LOCK_PNG = FIG_DIR / "locking_sequence.png"

ATTRACTOR_LABELS_SHORT = [
    "torus",
    "torus",
    "torus",
    "torus",
    "torus",
    "torus",
    "periodic window",
    "chaos",
    "chaos",
    "periodic window",
    "chaos",
    "chaos",
]

LOCKING_LABELS_SHORT = [
    "torus",
    "torus",
    "torus",
    "torus",
    "torus",
    "periodic window",
    "early chaos",
    "chaos",
]


def _validate_unique_d_keys(d_values, fmt):
    keys = [format(float(value), fmt) for value in d_values]
    if len(set(keys)) != len(keys):
        raise ValueError(f"D_values must be unique after {fmt} formatting")


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------


def delayed_logistic(state, A, D):
    """One iteration: state = (x, y) -> (x', y')."""
    x, y = state
    x_new = A * x + (1.0 - A) * (1.0 - D * y * y)
    y_new = x
    return np.array([x_new, y_new])


def delayed_logistic_jac(state, A, D):
    """Jacobian of the delayed logistic map."""
    x, y = state
    return np.array([[A, -2.0 * (1.0 - A) * D * y], [1.0, 0.0]])


# ---------------------------------------------------------------------------
# Attractor computation
# ---------------------------------------------------------------------------


def compute_attractor(A, D, n_transient=10_000, n_plot=50_000, x0=None):
    """Iterate the map and return the attractor points (None if diverged)."""
    if x0 is None:
        # Start near the fixed point
        fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
        x0 = np.array([fp + 0.01, fp - 0.01])

    return trajectory_after_transient(
        x0,
        lambda state: delayed_logistic(state, A, D),
        n_transient,
        n_plot,
        diverged_fn=lambda state: np.any(np.abs(state) > 1e10),
    )


def compute_attractors(
    *,
    A=0.3,
    D_values=None,
    n_transient=20_000,
    n_plot=100_000,
    output_path=ATTR_NPZ,
):
    """Compute attractors for twelve D values spanning torus to chaos."""
    if D_values is None:
        D_values = [1.55, 1.65, 1.75, 1.82, 1.86, 1.90, 1.92, 1.94, 1.95, 2.00, 2.09, 2.16]
    else:
        D_values = list(np.atleast_1d(np.asarray(D_values, dtype=np.float64)))
    _validate_unique_d_keys(D_values, ".2f")
    results = {}
    for idx, D in enumerate(D_values):
        label = ATTRACTOR_LABELS_SHORT[idx] if idx < len(ATTRACTOR_LABELS_SHORT) else ""
        print(f"  D={D} {label}")
        traj = compute_attractor(A, D, n_transient=n_transient, n_plot=n_plot)
        results[f"D_{D:.2f}_x"] = traj[:, 0]
        results[f"D_{D:.2f}_y"] = traj[:, 1]

    results["D_values"] = np.array(D_values)
    results["A"] = np.array([A])
    return _io_write_payload(output_path, results, base_dir=FIG_DIR)


# ---------------------------------------------------------------------------
# Lyapunov spectrum computation
# ---------------------------------------------------------------------------


def compute_lyapunov_spectrum(
    *,
    A=0.3,
    D_values=None,
    n_iter=50_000,
    n_transient=10_000,
    output_path=LYAP_NPZ,
    progress_interval=500,
):
    """Sweep D and compute Lyapunov spectrum."""
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

    if D_values is None:
        D_values = np.linspace(1.3, 2.5, 2000)
    else:
        D_values = np.atleast_1d(np.asarray(D_values, dtype=np.float64))
    n_params = len(D_values)

    spectra = np.empty((n_params, 2))
    spectra_err = np.empty((n_params, 2))
    for i, D in enumerate(D_values):
        fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
        x0 = np.array([fp + 0.01, fp - 0.01])

        def f(state, _D=D):
            return delayed_logistic(state, A, _D)

        def jac(state, _D=D):
            return delayed_logistic_jac(state, A, _D)

        spectra[i], spectra_err[i] = lyapunov_spectrum(
            f, jac, x0, n_iter=n_iter, n_transient=n_transient, return_convergence=True
        )
        if output_path is not None and progress_interval and (i + 1) % progress_interval == 0:
            print(f"  Lyapunov: {i + 1}/{n_params}")
            _io_write_payload(
                output_path,
                {
                    "D": D_values[: i + 1],
                    "spectra": spectra[: i + 1],
                    "spectra_err": spectra_err[: i + 1],
                },
                base_dir=FIG_DIR,
            )

    return _io_write_payload(
        output_path,
        {"D": D_values, "spectra": spectra, "spectra_err": spectra_err},
        base_dir=FIG_DIR,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_attractors(data):
    """Plot the twelve attractor portraits in a 3x4 grid."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, panel_label, setup

    setup()

    D_values = data["D_values"]
    labels_short = ATTRACTOR_LABELS_SHORT
    panel_labels = list("abcdefghijkl")

    n_panels = len(D_values)

    # Compute shared axis limits from all panels' data
    all_x, all_y = [], []
    for D in D_values:
        all_x.append(data[f"D_{D:.2f}_x"])
        all_y.append(data[f"D_{D:.2f}_y"])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    pad = 0.05
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    xlim = (all_x.min() - pad * x_range, all_x.max() + pad * x_range)
    ylim = (all_y.min() - pad * y_range, all_y.max() + pad * y_range)

    spec = figure_spec("grid")
    fig, axes = plt.subplots(3, 4, figsize=(spec.figsize[0], spec.figsize[1] + 0.8))
    fig.subplots_adjust(hspace=0.48, wspace=0.24)
    axes_flat = axes.flatten()

    for idx, D in enumerate(D_values):
        ax = axes_flat[idx]
        x = data[f"D_{D:.2f}_x"]
        y = data[f"D_{D:.2f}_y"]
        ax.scatter(x, y, s=0.012, c=COLORS["black"], alpha=0.22, rasterized=True)
        panel_label(ax, panel_labels[idx])
        ax.set_title(f"$D={D}$\n{labels_short[idx]}", loc="left")
        if idx // 4 == 2:
            ax.set_xlabel("$x$")
        if idx % 4 == 0:
            ax.set_ylabel("$y$")
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False, equal=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Hide unused subplots
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        r"Delayed logistic map, $\alpha = 0.3$",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=spec.title_size,
    )
    fig.savefig(ATTR_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ATTR_PNG}")


def plot_lyapunov(data):
    """Plot Lyapunov spectrum vs D."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        lyap_color,
        reference_line,
        setup,
    )

    setup()

    D = data["D"]
    spectra = data["spectra"]
    spectra_err = data.get("spectra_err", np.zeros_like(spectra))

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    ax.plot(D, spectra[:, 0], color=lyap_color(0), linestyle="-", lw=0.8, label=r"$\lambda_1$")
    # Finite-time convergence band (Benettin running estimate spread). The
    # underlying spread is genuinely narrow, so alpha/edges are pushed up to
    # keep it visible rather than sub-pixel; see plan-figures.md sec05
    # lyapunov_vs_D notes.
    ax.fill_between(
        D,
        spectra[:, 0] - spectra_err[:, 0],
        spectra[:, 0] + spectra_err[:, 0],
        alpha=0.45,
        color=lyap_color(0),
        edgecolor=lyap_color(0),
        linewidth=0.3,
        zorder=1.5,
    )
    ax.plot(D, spectra[:, 1], color=lyap_color(1), linestyle="-", lw=0.8, label=r"$\lambda_2$")
    reference_line(ax, 0, axis="y", lw=0.7)
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"Lyapunov exponent")
    ax.set_title(r"Delayed logistic map, $\alpha = 0.3$", loc="left")
    apply_axes_polish(ax, kind="double", title_loc="left")
    finalize_legend(ax, kind="double", loc="upper right")

    # Mark key transitions
    alpha = 0.3
    D_hopf = (3.0 - 2.0 * alpha) / (4.0 * (1.0 - alpha) ** 2)  # approx 1.2245
    reference_line(ax, D_hopf, axis="x", lw=0.5, ls=":", alpha=0.7)
    ax.text(
        D_hopf + 0.02,
        ax.get_ylim()[1] * 0.8,
        r"$D_c = \frac{3-2\alpha}{4(1-\alpha)^2}$",
        fontsize=spec.tick_size,
        color=COLORS["grey"],
    )

    # The convergence band (spectra_err) is a few x1e-3 wide against a
    # y-range of ~0.6, so it is genuinely sub-pixel at the full-panel scale
    # above -- an inset over the locking/transition region (D in
    # [1.85, 2.15], where the running Benettin estimate is least converged)
    # is the only way the shaded band the caption describes is actually
    # visible, and it reuses the empty lower-right quadrant of the panel.
    inset_mask = (D >= 1.85) & (D <= 2.15)
    if np.any(inset_mask) and np.any(spectra_err[inset_mask, 0] > 0):
        ax_inset = ax.inset_axes((0.60, 0.06, 0.38, 0.42))
        D_in = D[inset_mask]
        lam1_in = spectra[inset_mask, 0]
        err_in = spectra_err[inset_mask, 0]
        ax_inset.plot(D_in, lam1_in, color=lyap_color(0), lw=0.8)
        ax_inset.fill_between(
            D_in,
            lam1_in - err_in,
            lam1_in + err_in,
            alpha=0.45,
            color=lyap_color(0),
            edgecolor=lyap_color(0),
            linewidth=0.3,
        )
        reference_line(ax_inset, 0, axis="y", lw=0.5)
        ax_inset.set_xlim(1.85, 2.15)
        ax_inset.tick_params(labelsize=spec.tick_size - 1)
        ax_inset.set_title("convergence band (inset)", fontsize=spec.tick_size, loc="left", pad=2)

    fig.savefig(LYAP_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {LYAP_PNG}")


# ---------------------------------------------------------------------------
# Locking sequence: zoom into the locking-to-chaos transition
# ---------------------------------------------------------------------------


def compute_locking_sequence(
    *,
    A=0.3,
    D_values=None,
    n_transient=20_000,
    n_plot=100_000,
    output_path=LOCK_NPZ,
):
    """Compute attractors for 8 D values in [1.86, 1.95]."""
    if D_values is None:
        D_values = [1.860, 1.880, 1.895, 1.905, 1.915, 1.930, 1.940, 1.950]
    else:
        D_values = list(np.atleast_1d(np.asarray(D_values, dtype=np.float64)))
    _validate_unique_d_keys(D_values, ".3f")
    results = {}
    for idx, D in enumerate(D_values):
        label = LOCKING_LABELS_SHORT[idx] if idx < len(LOCKING_LABELS_SHORT) else ""
        print(f"  D={D} {label}")
        traj = compute_attractor(A, D, n_transient=n_transient, n_plot=n_plot)
        results[f"D_{D:.3f}_x"] = traj[:, 0]
        results[f"D_{D:.3f}_y"] = traj[:, 1]

    results["D_values"] = np.array(D_values)
    results["A"] = np.array([A])
    return _io_write_payload(output_path, results, base_dir=FIG_DIR)


def plot_locking_sequence(data):
    """Plot 2x4 grid of locking-to-chaos transition."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, panel_label, setup

    setup()

    D_values = data["D_values"]
    labels = LOCKING_LABELS_SHORT
    panel_labels = list("abcdefgh")

    # Compute shared axis limits
    all_x, all_y = [], []
    for D in D_values:
        all_x.append(data[f"D_{D:.3f}_x"])
        all_y.append(data[f"D_{D:.3f}_y"])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    pad = 0.05
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    xlim = (all_x.min() - pad * x_range, all_x.max() + pad * x_range)
    ylim = (all_y.min() - pad * y_range, all_y.max() + pad * y_range)

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 4, figsize=(spec.figsize[0], spec.figsize[1] - 0.5))
    fig.subplots_adjust(hspace=0.48, wspace=0.24)
    axes_flat = axes.flatten()

    for idx, D in enumerate(D_values):
        ax = axes_flat[idx]
        x = data[f"D_{D:.3f}_x"]
        y = data[f"D_{D:.3f}_y"]
        ax.scatter(x, y, s=0.012, c=COLORS["black"], alpha=0.22, rasterized=True)
        panel_label(ax, panel_labels[idx])
        ax.set_title(f"$D={D}$\n{labels[idx]}", loc="left")
        if idx // 4 == 1:
            ax.set_xlabel("$x$")
        if idx % 4 == 0:
            ax.set_ylabel("$y$")
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False, equal=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # The default locator drops any tick that falls outside the padded
        # data limits, which silently removed every negative candidate here
        # (ylim reaches ~-0.35) and left "0.0, 0.5" as the only labels. A
        # fixed step guarantees at least one negative tick survives.
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))

    fig.suptitle(
        r"Locking$\to$chaos transition, $\alpha = 0.3$",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=spec.title_size,
    )
    fig.savefig(LOCK_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {LOCK_PNG}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def compute_animation_data():
    """Sweep D from 1.4 to 3.5 and store attractor trajectories for animation."""
    A = 0.3
    D_sweep = np.linspace(1.4, 3.5, 200)

    def iterate_fn(D):
        return compute_attractor(A, D, n_transient=20_000, n_plot=5_000)

    return run_animation_sweep(iterate_fn, D_sweep, ANIM_NPZ, n_plot=5_000)


def make_animation_gif(data):
    """Create GIF animation of attractors evolving across D."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"],
        data["all_x"],
        data["all_y"],
        ANIM_GIF,
        title_template=r"Delayed logistic map, $\alpha = 0.3$, $D = {param_value}$",
        param_name="D",
        param_fmt=".3f",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    attr_data = load_or_compute_payload(
        ATTR_NPZ,
        "attractors",
        compute_attractors,
        required_keys=("D_values", "A"),
    )
    plot_attractors(attr_data)

    lyap_data = load_or_compute_payload(
        LYAP_NPZ,
        "Lyapunov spectrum",
        compute_lyapunov_spectrum,
        required_keys=("D", "spectra"),
    )
    plot_lyapunov(lyap_data)

    lock_data = load_or_compute_payload(
        LOCK_NPZ,
        "locking sequence",
        compute_locking_sequence,
        required_keys=("D_values", "A"),
    )
    plot_locking_sequence(lock_data)

    # Animation
    try:
        anim_data = safe_load(ANIM_NPZ)
        print(f"Loaded {ANIM_NPZ}")
    except FileNotFoundError:
        print("Computing animation data...")
        anim_data = compute_animation_data()
    make_animation_gif(anim_data)


if __name__ == "__main__":
    main()
