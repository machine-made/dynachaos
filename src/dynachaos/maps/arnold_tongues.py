#!/usr/bin/env python3
"""
arnold_tongues: Arnold tongue structure of the circle map in the (Ω, K) plane.

Displays the rotation number ρ(Ω, K) as a 2D color map, revealing the
wedge-shaped tongue structure.  Above the critical line K_c = 1/(2π),
tongues overlap and chaos appears.

Map:  θ_{n+1} = θ_n + Ω + K sin(2πθ_n)  (mod 1)

OUTPUTS:
  figures/sec02_circle_map/arnold_tongues.npz
  figures/sec02_circle_map/arnold_tongues.png

USAGE:
  python src/dynachaos/maps/arnold_tongues.py
  rm figures/sec02_circle_map/arnold_tongues.npz
  python src/dynachaos/maps/arnold_tongues.py
"""

import numpy as np

from dynachaos.io.paths import safe_load, section_dir

FIG_DIR = section_dir("sec02_circle_map")
OUTPUT_NPZ = FIG_DIR / "arnold_tongues.npz"
OUTPUT_PNG = FIG_DIR / "arnold_tongues.png"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute():
    """Sweep (Ω, K) and compute rotation numbers — fully vectorized."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_omega = 2000
    n_K = 1000
    n_transient = 5000
    n_iter = 50_000

    Omega_values = np.linspace(0.0, 1.0, n_omega)
    K_values = np.linspace(0.0, 0.3, n_K)

    # 2D grid: shape (n_K, n_omega) — K varies along rows, Ω along columns
    Omega_grid, K_grid = np.meshgrid(Omega_values, K_values)
    Omega_flat = Omega_grid.ravel()
    K_flat = K_grid.ravel()
    n_total = len(Omega_flat)

    TWO_PI = 2.0 * np.pi
    theta = np.full(n_total, 0.1)

    # Transient (unwrapped)
    for step in range(n_transient):
        theta += Omega_flat + K_flat * np.sin(TWO_PI * theta)
        if (step + 1) % 1000 == 0:
            print(f"  Transient: {step + 1}/{n_transient}")

    theta_start = theta.copy()

    # Main iteration for rotation number
    for step in range(n_iter):
        theta += Omega_flat + K_flat * np.sin(TWO_PI * theta)
        if (step + 1) % 10_000 == 0:
            print(f"  Iteration: {step + 1}/{n_iter}")

    rho = (theta - theta_start) / n_iter
    rho_2d = rho.reshape(n_K, n_omega)

    np.savez_compressed(
        OUTPUT_NPZ,
        Omega=Omega_values,
        K=K_values,
        rho=rho_2d,
    )
    print(f"Saved {OUTPUT_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(data):
    """Plot the Arnold tongue structure as a 2D color map."""
    import matplotlib.pyplot as plt

    from dynachaos.utils import style as _style
    from dynachaos.utils.style import (
        COLORS,
        add_field_colorbar,
        annotate_on_field,
        apply_axes_polish,
        figure_spec,
        setup,
    )

    setup()

    Omega = data["Omega"]
    K = data["K"]
    rho = data["rho"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)

    pcm = ax.pcolormesh(
        Omega,
        K,
        rho,
        cmap=_style.CMAP_SEQUENTIAL,
        shading="auto",
        rasterized=True,
    )
    pcm.set_clim(0.0, 1.0)

    # Critical line K_c = 1/(2π)
    K_c = 1.0 / (2.0 * np.pi)
    ax.axhline(K_c, color=COLORS["red"], lw=0.8, ls="--", alpha=0.85)
    # No bbox: an opaque box here punches a hole in the field. Contour levels
    # coincide with locked plateaus (rho is exactly constant over a finite
    # area), so contouring them floods the tongue interiors with degenerate
    # hash; the tongue boundaries are already visible as colour
    # discontinuities in the pcolormesh, so no overlay is drawn here.
    annotate_on_field(
        ax,
        0.93,
        K_c + 0.005,
        r"$K_c = 1/(2\pi)$",
        fontsize=spec.tick_size,
        color=COLORS["red"],
        ha="right",
        va="center",
        bbox=None,
    )

    ax.set_xlabel(r"Bare frequency $\Omega$")
    ax.set_ylabel(r"Nonlinearity $K$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, K[-1])

    top = ax.secondary_xaxis("top")
    top.set_xticks([0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0])
    top.set_xticklabels(
        [r"$0/1$", r"$1/3$", r"$1/2$", r"$2/3$", r"$1/1$"],
        fontsize=spec.tick_size,
    )
    top.tick_params(axis="x", pad=2, length=0)
    # Nudge the corner fraction labels (0/1, 1/1) inboard so they don't sit
    # flush against the plot edges.
    top_labels = top.get_xticklabels()
    top_labels[0].set_ha("left")
    top_labels[-1].set_ha("right")

    add_field_colorbar(fig, pcm, ax, label=r"Rotation number $\rho$", pad=0.02, aspect=30)

    apply_axes_polish(ax, kind="double", grid=False)

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(OUTPUT_NPZ)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing Arnold tongues...")
        compute()
        data = safe_load(OUTPUT_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
