#!/usr/bin/env python3
"""
spatiotemporal: Spatiotemporal intermittency in coupled map lattices.

Reproduces Kaneko (1985) "Spatiotemporal Intermittency in Coupled Map Lattices",
PTP 74(5), 1033-1044.

CML model:
    x_{n+1}(i) = f(x_n(i)) + eps/2 [g(x_n(i+1)) + g(x_n(i-1)) - 2 g(x_n(i))]

Three models studied:
  (A) Piecewise map: f(x) = x + x^2 + a for x < c; -3(x-c)+1+a for x > c
      with c = (sqrt(5)-1)/2, g(x) = f(x)
  (B) Coupled circle: f(x) = x + A sin(2 pi x) + C (mod 1), g(x) = sin(2 pi x)
      A=0.2, C=0.55
  (C) Coupled logistic: f(x) = 1 - A x^2, g(x) = f(x)
      A = 1.752

Figures:
  - Spacetime diagrams for each model

OUTPUTS: figures/sec08_sti/spacetime_diagrams.npz, .png
"""

import numpy as np

from dynachaos.cml.primitives import cml_step
from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps.primitives import logistic

FIG_DIR = section_dir("sec08_sti")

STI_NPZ = FIG_DIR / "spacetime_diagrams.npz"
STI_PNG = FIG_DIR / "spacetime_diagrams.png"


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def model_A_f(x, a=-0.01):
    """Piecewise map for model (A) (Kaneko 1985).

    For a <= 0 the laminar branch (x < c) has a stable fixed point
    x* = -sqrt(-a) with slope f'(x*) = 1 - 2*sqrt(-a); a small negative
    a makes it weakly stable, and Pomeau-Manneville intermittency occurs
    at nearby parameters, producing the laminar/burst spatiotemporal
    intermittency Kaneko studies. (a > 0 removes the fixed point.)
    """
    c = (np.sqrt(5) - 1.0) / 2.0
    return np.where(x < c, x + x * x + a, -3.0 * (x - c) + 1.0 + a)


def model_B_f(x, A=0.2, C=0.55):
    """Circle map for model (B)."""
    return (x + A * np.sin(2 * np.pi * x) + C) % 1.0


def model_B_g(x):
    """Coupling function for model (B)."""
    return np.sin(2 * np.pi * x)


def model_C_f(x, A=1.752):
    """Logistic map for model (C)."""
    return logistic(x, A)


def simulate_cml(f, g, eps, N=200, n_transient=2000, n_record=500, x0=None):
    """Run CML and record spacetime diagram."""
    if x0 is None:
        rng = np.random.default_rng(42)
        x0 = rng.uniform(0, 1, N)

    x = x0.copy()
    for _ in range(n_transient):
        x = cml_step(x, f, g, eps)

    spacetime = np.empty((n_record, N))
    for t in range(n_record):
        x = cml_step(x, f, g, eps)
        spacetime[t] = x

    return spacetime


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute():
    """Generate spacetime diagrams for all three models."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    N = 200
    n_record = 500

    results = {}

    print("  Model (A)...")
    for eps in [0.06, 0.07, 0.08]:
        st = simulate_cml(model_A_f, model_A_f, eps, N=N, n_record=n_record)
        results[f"A_eps_{eps}"] = st

    print("  Model (B)...")
    for eps in [0.02, 0.024, 0.03]:
        st = simulate_cml(model_B_f, model_B_g, eps, N=N, n_record=n_record)
        results[f"B_eps_{eps}"] = st

    print("  Model (C)...")
    for eps in [0.16, 0.20, 0.30]:
        st = simulate_cml(model_C_f, model_C_f, eps, N=N, n_record=n_record)
        results[f"C_eps_{eps}"] = st

    np.savez_compressed(STI_NPZ, **results)
    print(f"Saved {STI_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(data):
    """Plot spacetime diagrams for all three models."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        CMAP_SPACETIME,
        add_field_colorbar,
        apply_axes_polish,
        figure_spec,
        setup,
    )

    setup()

    configs = [
        ("A", [0.06, 0.07, 0.08], "Model (A): piecewise"),
        ("B", [0.02, 0.024, 0.03], "Model (B): circle"),
        ("C", [0.16, 0.20, 0.30], "Model (C): logistic"),
    ]

    # The caption commits to the full field (N = 200 sites, 500 steps), so
    # the panels must show all of it: cropping to a prettier window would
    # contradict the text. The moire banding came from resampling 500 rows
    # with nearest-neighbour interpolation, not from the aspect -- antialiased
    # interpolation removes it; the non-isotropic aspect stays, as in
    # Kaneko's own plates, and taller panels keep the stretch modest.
    spec = figure_spec("grid")
    fig, axes = plt.subplots(3, 3, figsize=(spec.figsize[0], spec.figsize[1] * 1.7))

    for row, (model, eps_vals, title) in enumerate(configs):
        row_arrays = [
            data[f"{model}_eps_{eps}"] for eps in eps_vals if f"{model}_eps_{eps}" in data
        ]
        row_stack = np.concatenate([arr.ravel() for arr in row_arrays])
        vmin, vmax = np.percentile(row_stack, [1.0, 99.0])
        if vmax - vmin < 1e-12:
            vmin -= 0.5
            vmax += 0.5
        row_mappable = None
        for col, eps in enumerate(eps_vals):
            ax = axes[row, col]
            key = f"{model}_eps_{eps}"
            if key in data:
                st = data[key]
                row_mappable = ax.imshow(
                    st,
                    aspect="auto",
                    cmap=CMAP_SPACETIME,
                    origin="lower",
                    interpolation="antialiased",
                    vmin=vmin,
                    vmax=vmax,
                )
            ax.set_title(rf"$\varepsilon = {eps}$", loc="left")
            if col == 0:
                ax.set_ylabel(f"{title}\nTime $n$")
            else:
                ax.set_ylabel("")
            if row == len(configs) - 1:
                ax.set_xlabel("Site $i$")
            else:
                ax.set_xlabel("")
            apply_axes_polish(ax, kind="grid", title_loc="left", grid=False)
        if row_mappable is not None:
            add_field_colorbar(fig, row_mappable, axes[row, -1], label=r"$x_i^n$")

    fig.savefig(STI_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {STI_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(STI_NPZ)
        print(f"Loaded {STI_NPZ}")
    except FileNotFoundError:
        print("Computing spacetime diagrams...")
        compute()
        data = safe_load(STI_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
