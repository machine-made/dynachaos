#!/usr/bin/env python3
"""
circle_map: Devil's staircase and Lyapunov exponents for the Kaneko circle map.

Reproduces Kaneko (1982) "On the period-adding phenomena at the frequency
locking in a one-dimensional mapping".

Map:  θ_{n+1} = θ_n + D + A sin(2πθ_n)  (mod 1)

With D = 0.25 fixed and A as the bifurcation parameter.

OUTPUTS:
  figures/sec02_circle_map/devils_staircase.npz
  figures/sec02_circle_map/devils_staircase.png

USAGE:
  python src/dynachaos/maps/circle_map.py
  rm figures/sec02_circle_map/devils_staircase.npz
  python src/dynachaos/maps/circle_map.py
"""

import numpy as np

from dynachaos.io.paths import load_or_compute_npz, section_dir
from dynachaos.maps._iter import iterate_unwrapped, run_transient

FIG_DIR = section_dir("sec02_circle_map")
OUTPUT_NPZ = FIG_DIR / "devils_staircase.npz"
OUTPUT_PNG = FIG_DIR / "devils_staircase.png"
ZOOM_NPZ = FIG_DIR / "staircase_zoom.npz"
ZOOM_PNG = FIG_DIR / "staircase_zoom.png"


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------


def circle_map(theta, A, D=0.25):
    """One iteration of the Kaneko circle map."""
    return (theta + D + A * np.sin(2 * np.pi * theta)) % 1.0


def circle_map_derivative(theta, A, D=0.25):
    """Derivative dF/dθ of the circle map."""
    return 1.0 + 2 * np.pi * A * np.cos(2 * np.pi * theta)


# ---------------------------------------------------------------------------
# Rotation number computation
# ---------------------------------------------------------------------------


def rotation_number(A, D=0.25, n_transient=5000, n_iter=50_000, theta0=0.1):
    """Compute the rotation number for given parameters.

    The rotation number is the average advance per iteration (without mod 1).
    """
    # Transient and accumulation are both done in unwrapped coordinates.
    theta_unwrapped = iterate_unwrapped(
        theta0,
        lambda theta: D + A * np.sin(2 * np.pi * theta),
        n_transient,
    )
    theta_start = theta_unwrapped
    theta_unwrapped = iterate_unwrapped(
        theta_unwrapped,
        lambda theta: D + A * np.sin(2 * np.pi * theta),
        n_iter,
    )

    return (theta_unwrapped - theta_start) / n_iter


def lyapunov_exponent(A, D=0.25, n_transient=5000, n_iter=50_000, theta0=0.1):
    """Compute Lyapunov exponent of the circle map at given A, D."""
    theta = float(run_transient(theta0, lambda th: circle_map(th, A, D), n_transient))

    log_sum = 0.0
    for _ in range(n_iter):
        deriv = abs(circle_map_derivative(theta, A, D))
        if deriv > 0:
            log_sum += np.log(deriv)
        else:
            log_sum += -100.0
        theta = circle_map(theta, A, D)

    return log_sum / n_iter


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute():
    """Sweep A and compute rotation numbers + Lyapunov exponents.

    Vectorized: all parameter values are iterated simultaneously as a
    single NumPy array, giving ~100x speedup over scalar loops.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_params = 200_000
    n_transient = 5000
    n_iter = 50_000
    D = 0.25

    A_values = np.linspace(0.0, 0.25, n_params)
    TWO_PI = 2.0 * np.pi

    # --- Vectorised iteration: all 200k thetas in parallel ---
    theta = np.full(n_params, 0.1)

    # Transient (unwrapped)
    theta = iterate_unwrapped(
        theta,
        lambda th: D + A_values * np.sin(TWO_PI * th),
        n_transient,
    )
    print("  Transient done")

    # Record start for rotation number
    theta_start = theta.copy()

    # Accumulate Lyapunov and iterate for rotation number
    log_sum = np.zeros(n_params)
    for step in range(n_iter):
        deriv = np.abs(1.0 + TWO_PI * A_values * np.cos(TWO_PI * theta))
        log_sum += np.where(deriv > 0, np.log(deriv), -100.0)
        theta += D + A_values * np.sin(TWO_PI * theta)
        if (step + 1) % 5000 == 0:
            print(f"  {step + 1}/{n_iter} iterations")

    rho = (theta - theta_start) / n_iter
    lam = log_sum / n_iter

    np.savez_compressed(OUTPUT_NPZ, A=A_values, rho=rho, lam=lam)
    print(f"Saved {OUTPUT_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(data):
    """Create the devil's staircase with Lyapunov overlay."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        annotate_on_field,
        apply_axes_polish,
        figure_spec,
        panel_label,
        reference_line,
        setup,
    )

    setup()

    A = data["A"]
    rho = data["rho"]
    lam = data["lam"]

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(spec.figsize[0], spec.figsize[1] * 1.64),
        sharex=True,
        height_ratios=[3, 2],
    )
    fig.subplots_adjust(hspace=0.08)

    # --- Top panel: rotation-number staircase ---
    ax1.plot(A, rho, color=COLORS["black"], lw=0.35, rasterized=True)
    ax1.set_ylabel(r"Rotation number $\rho$")
    ax1.set_ylim(0.0, 0.255)
    panel_label(ax1, "(a)")
    reference_line(ax1, 1.0 / 5.0, axis="y", lw=0.5, alpha=0.7)
    ax1.text(0.004, 0.2035, r"$\rho = 1/5$", fontsize=spec.tick_size, color=COLORS["grey"])

    # --- Bottom panel: Lyapunov exponent vs A ---
    ax2.plot(A, lam, color=COLORS["black"], lw=0.2, linestyle="-", rasterized=True)
    ax2.fill_between(A, 0.0, lam, where=lam > 0.0, color=COLORS["red"], alpha=0.18)
    reference_line(ax2, 0.0, axis="y", lw=0.7)
    ax2.set_xlabel(r"Nonlinearity parameter $K$")
    ax2.set_ylabel(r"Lyapunov exponent $\lambda$")
    ax2.set_xlim(0, 0.25)
    panel_label(ax2, "(b)")

    # Mark the onset of 1/5 locking and compute chaos onset live from cached data.
    K_inf = 0.15671685

    # Compute chaos onset: first K where lam stays sustained above 5-sigma noise floor
    noise_floor = lam[A < 0.10].std() * 5
    noise_floor = max(noise_floor, 1e-6)  # Ensure reasonable floor
    mask_K_gt_015 = A > 0.15
    A_scan = A[mask_K_gt_015]
    lam_scan = lam[mask_K_gt_015]
    window_size = 50
    K_chaos_onset = None
    for i in range(len(lam_scan) - window_size):
        if np.all(lam_scan[i : i + window_size] > noise_floor):
            K_chaos_onset = A_scan[i]
            break
    if K_chaos_onset is None:
        K_chaos_onset = 0.182  # fallback

    for ax in (ax1, ax2):
        ax.axvline(K_inf, color=COLORS["blue"], lw=0.7, ls=":", alpha=0.8)
        ax.axvline(K_chaos_onset, color=COLORS["red"], lw=0.7, ls=":", alpha=0.8)
    annotate_on_field(
        ax1,
        K_inf - 0.003,
        0.244,
        r"$K_\infty$: $1/5$ lock",
        fontsize=spec.tick_size,
        color=COLORS["blue"],
        ha="right",
        va="center",
    )
    # Clip (not scale) the Lyapunov axis to the physically interesting band:
    # three isolated superstable spikes reach lam ~ -2.3-2.35 and, if used to
    # set the range, crush the |lambda| < 0.05 band the caption relies on onto
    # the zero line. Let those spikes run off the bottom instead.
    ax2.set_ylim(-0.35, 0.15)
    # Format chaos onset to 3 sig figs for display
    K_chaos_display = round(K_chaos_onset, 3)
    # The band right of the onset marker is solid data at this y-range, so the
    # label sits in the empty quadrant left of the marker, reading toward the
    # feature it names -- no bbox needed there.
    annotate_on_field(
        ax2,
        K_chaos_onset - 0.003,
        0.07,
        rf"$K \approx {K_chaos_display}$: sustained $\lambda>0$",
        fontsize=spec.tick_size,
        color=COLORS["red"],
        ha="right",
        va="bottom",
        bbox=None,
    )

    apply_axes_polish(ax1, kind="double", grid=False)
    apply_axes_polish(ax2, kind="double", grid=False)

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Staircase zoom: self-similar nesting in the Farey region
# ---------------------------------------------------------------------------


def compute_zoom():
    """Recompute the staircase at high resolution over K ∈ [0.10, 0.17]."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_params = 100_000
    n_transient = 5000
    n_iter = 50_000
    D = 0.25

    A_values = np.linspace(0.10, 0.17, n_params)
    TWO_PI = 2.0 * np.pi

    theta = np.full(n_params, 0.1)
    theta = iterate_unwrapped(
        theta,
        lambda th: D + A_values * np.sin(TWO_PI * th),
        n_transient,
    )
    print("  Zoom transient done")

    theta_start = theta.copy()
    for step in range(n_iter):
        theta += D + A_values * np.sin(TWO_PI * theta)
        if (step + 1) % 5000 == 0:
            print(f"  Zoom: {step + 1}/{n_iter}")

    rho = (theta - theta_start) / n_iter
    np.savez_compressed(ZOOM_NPZ, A=A_values, rho=rho)
    print(f"Saved {ZOOM_NPZ}")


def plot_zoom(zoom_data, full_data):
    """Two-panel figure: full staircase with zoom box + zoomed Farey region."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        panel_label,
        reference_line,
        setup,
    )

    setup()

    spec = figure_spec("double")
    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(spec.figsize[0], spec.figsize[1]),
        width_ratios=[1, 1.2],
    )
    fig.subplots_adjust(wspace=0.30)

    # --- Left panel: full staircase with zoom box ---
    A_full = full_data["A"]
    rho_full = full_data["rho"]
    ax_full.plot(A_full, rho_full, color=COLORS["black"], lw=0.3, rasterized=True)
    ax_full.set_xlabel(r"Nonlinearity $K$")
    ax_full.set_ylabel(r"Rotation number $\rho$")
    ax_full.set_title("Global staircase", loc="left")
    ax_full.set_xlim(0, 0.25)
    ax_full.set_ylim(0.0, 0.255)
    panel_label(ax_full, "(a)")

    # Zoom box
    zoom_K = (0.125, 0.158)
    zoom_rho = (0.198, 0.226)
    rect = Rectangle(
        (zoom_K[0], zoom_rho[0]),
        zoom_K[1] - zoom_K[0],
        zoom_rho[1] - zoom_rho[0],
        linewidth=0.8,
        edgecolor=COLORS["red"],
        fill=False,
        ls="--",
    )
    ax_full.add_patch(rect)

    # --- Right panel: zoomed Farey region ---
    A_zoom = zoom_data["A"]
    rho_zoom = zoom_data["rho"]
    ax_zoom.plot(A_zoom, rho_zoom, color=COLORS["black"], lw=0.3, rasterized=True)
    ax_zoom.set_xlabel(r"Nonlinearity $K$")
    ax_zoom.set_ylabel(r"Rotation number $\rho$")
    ax_zoom.set_title(r"Period-adding sequence approaching $1/5$", loc="left")
    ax_zoom.set_xlim(*zoom_K)
    ax_zoom.set_ylim(*zoom_rho)
    # The staircase descends from high rho at low K to 1/5 at high K, so the
    # curve (and the 2/9 plateau) runs through the default upper-left corner;
    # the lower-left corner is empty of data, so the label is moved there.
    panel_label(ax_zoom, "(b)", loc="lower left")

    # Mark the dominant period-adding sequence n/(5n-1) and the 1/5 accumulation plateau.
    rationals = {
        r"$2/9$": 2.0 / 9.0,
        r"$3/14$": 3.0 / 14.0,
        r"$4/19$": 4.0 / 19.0,
        r"$1/5$": 1.0 / 5.0,
    }
    for label, val in rationals.items():
        reference_line(ax_zoom, val, axis="y", lw=0.3, alpha=0.5)
        ax_zoom.text(
            zoom_K[1] - 0.0025,
            val + 0.0006,
            label,
            fontsize=spec.tick_size - 1,
            color=COLORS["grey"],
            ha="right",
        )

    apply_axes_polish(ax_full, kind="double", title_loc="left", grid=False)
    apply_axes_polish(ax_zoom, kind="double", title_loc="left", grid=False)

    fig.savefig(ZOOM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ZOOM_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Devil's staircase
    data = load_or_compute_npz(
        OUTPUT_NPZ,
        "devil's staircase",
        compute,
        required_keys=("A", "rho", "lam"),
    )
    plot(data)

    # Staircase zoom
    zoom_data = load_or_compute_npz(
        ZOOM_NPZ,
        "staircase zoom",
        compute_zoom,
        required_keys=("A", "rho"),
    )
    plot_zoom(zoom_data, data)


if __name__ == "__main__":
    main()
