#!/usr/bin/env python3
"""
modulated_circle: Double devil's staircase in the modulated circle map.

Reproduces Kaneko (1984) "Fates of Three-Torus. I", PTP 71(2), 282-294, Sec 3.

Modulated circle map (Eq. 3.1):
    theta_{n+1} = theta_n + A sin(2 pi theta_n) + D + eps sin(2 pi phi_n)
    phi_{n+1}   = phi_n + C

where phi undergoes pure rigid rotation at irrational frequency C = (sqrt(5)-1)/2.

OUTPUTS: figures/sec06_three_torus/double_staircase.npz, .png
"""

import numpy as np

from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps._iter import iterate_unwrapped

FIG_DIR = section_dir("sec06_three_torus")
OUTPUT_NPZ = FIG_DIR / "double_staircase.npz"
OUTPUT_PNG = FIG_DIR / "double_staircase.png"
ZOOM_PNG = FIG_DIR / "double_staircase_zoom.png"

# Golden-mean inverse (irrational modulation frequency)
C_GOLDEN = (np.sqrt(5) - 1) / 2


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------


def modulated_circle(state, A, C, D, eps):
    """One iteration of the modulated circle map (Kaneko Eq. 3.1)."""
    theta, phi = state
    theta_new = (theta + A * np.sin(2 * np.pi * theta) + D + eps * np.sin(2 * np.pi * phi)) % 1.0
    phi_new = (phi + C) % 1.0
    return np.array([theta_new, phi_new])


# ---------------------------------------------------------------------------
# Rotation numbers
# ---------------------------------------------------------------------------


def rotation_numbers(A, C, D, eps, n_transient=5000, n_iter=30_000, state0=None):
    """Compute both rotation numbers (rho_theta, rho_phi)."""
    if state0 is None:
        state0 = np.array([0.1, 0.1])

    unwrapped = np.array(state0, dtype=np.float64, copy=True)

    def increment(state):
        theta_uw, phi_uw = state
        dtheta = A * np.sin(2 * np.pi * theta_uw) + D + eps * np.sin(2 * np.pi * phi_uw)
        return np.array([dtheta, C], dtype=np.float64)

    unwrapped = iterate_unwrapped(unwrapped, increment, n_transient)
    start = unwrapped.copy()
    unwrapped = iterate_unwrapped(unwrapped, increment, n_iter)

    rho_theta = (unwrapped[0] - start[0]) / n_iter
    rho_phi = (unwrapped[1] - start[1]) / n_iter
    return rho_theta, rho_phi


def longest_plateau_window(D, rho, target, tol):
    """Return the widest contiguous D-window with rho close to target."""
    mask = np.abs(rho - target) <= tol
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None

    start = idx[0]
    best = (start, start)
    prev = idx[0]
    for current in idx[1:]:
        if current != prev + 1:
            if D[prev] - D[start] > D[best[1]] - D[best[0]]:
                best = (start, prev)
            start = current
        prev = current
    if D[prev] - D[start] > D[best[1]] - D[best[0]]:
        best = (start, prev)
    return float(D[best[0]]), float(D[best[1]])


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute():
    """Sweep D (bare frequency) and compute the double devil's staircase."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Parameters following Kaneko Eq. 3.1
    A = 0.10  # subcritical (A < 1/2pi ~ 0.159)
    eps = 0.05  # moderate forcing amplitude
    C = C_GOLDEN  # golden-mean inverse

    n_params = 10_000
    D_values = np.linspace(0.0, 1.0, n_params)

    rho_theta = np.empty(n_params)
    rho_phi = np.empty(n_params)

    for i, D in enumerate(D_values):
        rt, rp = rotation_numbers(A, C, D, eps, n_transient=3000, n_iter=20_000)
        rho_theta[i] = rt
        rho_phi[i] = rp

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_params}")
            np.savez_compressed(
                OUTPUT_NPZ,
                D=D_values[: i + 1],
                rho_theta=rho_theta[: i + 1],
                rho_phi=rho_phi[: i + 1],
                A=np.array([A]),
                C=np.array([C]),
                eps=np.array([eps]),
            )

    np.savez_compressed(
        OUTPUT_NPZ,
        D=D_values,
        rho_theta=rho_theta,
        rho_phi=rho_phi,
        A=np.array([A]),
        C=np.array([C]),
        eps=np.array([eps]),
    )
    print(f"Saved {OUTPUT_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(data):
    """Plot the double devil's staircase."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup

    setup()

    D = data["D"]
    rho_theta = data["rho_theta"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=(spec.figsize[0], spec.figsize[1] * 0.95))

    zoom_targets = [
        (0.25, 5e-4, COLORS["blue"], r"zoom (a): plateau near $1/4$"),
        (C_GOLDEN, 5e-4, COLORS["red"], r"zoom (b): plateau near $C$"),
    ]
    for target, tol, color, label in zoom_targets:
        window = longest_plateau_window(D, rho_theta, target, tol)
        if window is None:
            continue
        pad = 0.003 if target < 0.4 else 0.002
        ax.axvspan(window[0] - pad, window[1] + pad, color=color, alpha=0.10, lw=0.0)
        # Anchor the label to the shaded band with a short leader line rather
        # than printing it at (window[0], target): that point sits directly
        # on the rho_theta=D climb, so the text was struck through by the
        # curve and, being left-aligned, visually drifted away from the
        # sliver it names. The label now sits in the clear band above the
        # curve and points straight down at the band's centre.
        center = 0.5 * (window[0] + window[1])
        # zoom (a) sits under the rho_phi/reference-line annotation block
        # (top-left, x below ~0.5 in axes fraction), so its label is placed
        # lower to stay clear of that text instead of colliding with it.
        label_y = 0.83 if target < 0.4 else 0.97
        ax.annotate(
            label,
            xy=(center, label_y - 0.19),
            xycoords=("data", "axes fraction"),
            xytext=(center, label_y),
            textcoords=("data", "axes fraction"),
            ha="center",
            va="top",
            color=color,
            fontsize=spec.legend_size - 0.3,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 0.6,
                "alpha": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    # Rigid-rotation reference rho_theta = D (grey dashed diagonal).
    ax.plot(D, D, color=COLORS["grey"], linestyle="--", lw=0.75, alpha=0.7)
    ax.plot(D, rho_theta, color=COLORS["black"], linestyle="-", lw=0.9, rasterized=True)
    ax.set_xlabel(r"bare frequency $D$")
    ax.set_ylabel(r"observed rotation $\rho_\theta$")
    ax.set_ylim(-0.02, 1.02)
    ax.text(
        0.03,
        0.96,
        r"$\rho_\varphi \equiv C$ to numerical precision",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=spec.legend_size,
    )
    ax.text(
        0.03,
        0.89,
        r"faint dashed line: rigid-rotation reference $\rho_\theta=D$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=spec.legend_size - 0.2,
        color=COLORS["grey"],
    )

    apply_axes_polish(ax, kind="double", title_loc="left", grid=False)

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Zoomed double staircase: partial locking windows
# ---------------------------------------------------------------------------


def plot_zoom(data):
    """Two-panel zoom into different D windows showing rho_theta staircase detail."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        panel_label,
        reference_line,
        setup,
    )

    setup()

    D = data["D"]
    rho_theta = data["rho_theta"]

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(spec.figsize[0], spec.figsize[1]),
    )
    fig.subplots_adjust(wspace=0.30)

    window1 = longest_plateau_window(D, rho_theta, 0.25, 5e-4)
    window2 = longest_plateau_window(D, rho_theta, C_GOLDEN, 5e-4)
    if window1 is None:
        window1 = (0.255, 0.275)
    if window2 is None:
        window2 = (0.602, 0.622)

    # --- Left panel: narrow plateau around 1/4 ---
    pad1 = 0.008
    mask1 = (D >= window1[0] - pad1) & (D <= window1[1] + pad1)
    rt1 = rho_theta[mask1]
    # The rho=1/4 guide used to be drawn after the data at the default
    # linewidth (1.0) with the curve at a hairline 0.3 -- the guide sat
    # exactly on the locking plateau it was meant to mark and buried it.
    # Draw the guide behind the data (lower zorder) and thinner, and give
    # the data curve enough weight to read as the primary feature.
    reference_line(ax1, 0.25, axis="y", lw=0.5, zorder=1)
    ax1.plot(D[mask1], rt1, color=COLORS["black"], lw=0.9, rasterized=True, zorder=2)
    ax1.axvspan(window1[0], window1[1], color=COLORS["blue"], alpha=0.10, lw=0.0)
    ax1.set_xlabel(r"$D$")
    ax1.set_ylabel(r"$\rho_\theta$")
    ax1.set_title(r"plateau near $1/4$", loc="left")
    panel_label(ax1, "a")
    if rt1.size > 0:
        ax1.set_ylim(rt1.min() - 0.01, rt1.max() + 0.01)

    # --- Right panel: plateau around the irrational modulation frequency C ---
    pad2 = 0.006
    mask2 = (D >= window2[0] - pad2) & (D <= window2[1] + pad2)
    rt2 = rho_theta[mask2]
    reference_line(ax2, C_GOLDEN, axis="y", lw=0.5, zorder=1)
    ax2.plot(D[mask2], rt2, color=COLORS["black"], lw=0.9, rasterized=True, zorder=2)
    ax2.axvspan(window2[0], window2[1], color=COLORS["red"], alpha=0.10, lw=0.0)
    ax2.set_xlabel(r"$D$")
    ax2.set_title(r"plateau near $C$", loc="left")
    panel_label(ax2, "b")
    if rt2.size > 0:
        ax2.set_ylim(rt2.min() - 0.004, rt2.max() + 0.004)

    apply_axes_polish(ax1, kind="double", title_loc="left", grid=False)
    apply_axes_polish(ax2, kind="double", title_loc="left", grid=False)

    fig.savefig(ZOOM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ZOOM_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(OUTPUT_NPZ)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing double devil's staircase...")
        compute()
        data = safe_load(OUTPUT_NPZ)
    plot(data)
    plot_zoom(data)


if __name__ == "__main__":
    main()
