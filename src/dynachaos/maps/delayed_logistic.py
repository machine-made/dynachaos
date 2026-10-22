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
  - Lyapunov exponents along the D path (a) and the power spectra of x_n for a smooth
    torus (D = 1.70) and a wrinkled torus (D = 1.86) (b)
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

# Drive amplitudes of the two power spectra beside the Lyapunov sweep.
PSD_DISPLAY_D = (1.70, 1.86)
LYAP_REQUIRED_KEYS = (
    "D",
    "spectra",
    "spectra_err",
    "A",
    "freqs_170",
    "psd_170",
    "freqs_186",
    "psd_186",
)

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


def _psd_tag(D):
    """Key suffix of a power-spectrum entry: D = 1.70 -> '170'."""
    return f"{round(100.0 * float(D)):d}"


def compute_psd(A, D, n_transient=20_000, n_record=32_768):
    """Normalised power spectrum of x_n of the delayed logistic map at drive ``D``.

    The record is mean-subtracted and Hann-windowed; the spectrum is scaled to a
    maximum of 1.  Returns ``(freqs, psd)`` with frequencies in cycles per iteration.
    """
    fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
    x0 = np.array([fp + 0.01, fp - 0.01], dtype=np.float64)
    traj = trajectory_after_transient(
        x0,
        lambda s: delayed_logistic(s, A, D),
        n_transient,
        n_record,
    )
    x = traj[:, 0]
    xw = (x - np.mean(x)) * np.hanning(len(x))
    psd = np.abs(np.fft.rfft(xw)) ** 2
    return np.fft.rfftfreq(len(x)), psd / np.max(psd)


def compute_lyapunov_spectrum(
    *,
    A=0.3,
    D_values=None,
    n_iter=50_000,
    n_transient=10_000,
    psd_D=(),
    psd_n_transient=20_000,
    psd_n_record=32_768,
    output_path=LYAP_NPZ,
    progress_interval=500,
):
    """Sweep D and compute the Lyapunov spectrum.

    ``D_values`` defaults to 2000 points on [1.20, 2.20].  Every D in ``psd_D``
    also gets a power spectrum, stored as ``freqs_<tag>`` and ``psd_<tag>``
    (tag = 100 D, e.g. ``freqs_170`` for D = 1.70).
    """
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

    if D_values is None:
        D_values = np.linspace(1.20, 2.20, 2000)
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

    payload = {"D": D_values, "spectra": spectra, "spectra_err": spectra_err}
    for D in psd_D:
        print(f"  Power spectrum at D = {D}")
        freqs, psd = compute_psd(A, D, n_transient=psd_n_transient, n_record=psd_n_record)
        payload[f"freqs_{_psd_tag(D)}"] = freqs
        payload[f"psd_{_psd_tag(D)}"] = psd
    payload["A"] = np.array([A])
    return _io_write_payload(output_path, payload, base_dir=FIG_DIR)


def compute_lyapunov_figure_data():
    """Compute the data of the Lyapunov figure: the D sweep and the two power spectra."""
    return compute_lyapunov_spectrum(psd_D=PSD_DISPLAY_D)


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
    """Plot (a) the Lyapunov spectrum vs D and (b) the power spectra of x_n."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        lyap_color,
        panel_label,
        reference_line,
        setup,
    )

    setup()

    D = data["D"]
    spectra = data["spectra"]
    spectra_err = data["spectra_err"]
    alpha = float(data["A"][0])

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(spec.figsize[0], 3.2))
    fig.subplots_adjust(wspace=0.24, bottom=0.18)

    # Panel (a): Lyapunov spectrum with the finite-time convergence band
    ax1.plot(D, spectra[:, 0], color=lyap_color(0), lw=0.9, label=r"$\lambda_1$")
    ax1.fill_between(
        D,
        spectra[:, 0] - spectra_err[:, 0],
        spectra[:, 0] + spectra_err[:, 0],
        alpha=0.30,
        color=lyap_color(0),
        linewidth=0,
    )
    ax1.plot(D, spectra[:, 1], color=lyap_color(1), lw=0.9, label=r"$\lambda_2$")
    reference_line(ax1, 0, axis="y", lw=0.5, alpha=0.6)

    D_hopf = (3.0 - 2.0 * alpha) / (4.0 * (1.0 - alpha) ** 2)  # ~1.2245
    reference_line(ax1, D_hopf, axis="x", lw=0.5, ls=":", alpha=0.7)
    ax1.text(
        D_hopf + 0.02,
        0.25,
        r"$D_c \approx 1.225$",
        fontsize=spec.tick_size - 1,
        color=COLORS["grey"],
    )

    ax1.set_xlabel(r"Drive amplitude $D$")
    ax1.set_ylabel(r"Lyapunov exponent")
    ax1.set_title(r"Lyapunov spectrum ($\alpha = 0.3$)", loc="left")
    ax1.set_ylim(-0.45, 0.45)
    apply_axes_polish(ax1, kind="double", title_loc="left")
    finalize_legend(ax1, kind="single", loc="lower left")
    panel_label(ax1, "(a)")

    # Panel (b): power spectra of a smooth and a wrinkled torus, in dB
    for d_val, color in zip(PSD_DISPLAY_D, (COLORS["blue"], COLORS["red"]), strict=True):
        tag = _psd_tag(d_val)
        psd_db = 10.0 * np.log10(np.maximum(data[f"psd_{tag}"], 1e-6))
        ax2.plot(
            data[f"freqs_{tag}"],
            psd_db,
            color=color,
            lw=0.8,
            alpha=0.85,
            label=rf"$D = {d_val:.2f}$",
        )

    # Primary winding frequency f0 approx 0.225
    f0 = 0.2254
    ax2.axvline(f0, color=COLORS["black"], ls="--", lw=0.6, alpha=0.6)
    ax2.text(f0, -4, r"$f_0$", ha="center", fontsize=spec.tick_size, color=COLORS["black"])

    ax2.set_xlim(0.0, 0.5)
    ax2.set_ylim(-55, 5)
    ax2.set_xlabel(r"Frequency $f$ (dimensionless)")
    ax2.set_ylabel(r"Power spectrum $S(f)$ (dB)")
    ax2.set_title(r"Harmonics of the rotation frequency in $x_n$", loc="left")
    apply_axes_polish(ax2, kind="double", title_loc="left")
    finalize_legend(ax2, kind="single", loc="lower right")
    panel_label(ax2, "(b)")

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
        compute_lyapunov_figure_data,
        required_keys=LYAP_REQUIRED_KEYS,
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
