#!/usr/bin/env python3
"""
coupled_delayed: 4D coupled delayed logistic map for three-torus dynamics.

Reproduces Kaneko (1984) "Fates of Three-Torus. I", PTP 71(2), 282-294.

Map (Eq. 2.2):
    x_{n+1} = A x_n + D_A x_{n-1}(1 - x_{n-1}) + eps h_1(x_n, x_{n-1}, z_n, z_{n-1})
    z_{n+1} = A z_n + D_B z_{n-1}(1 - z_{n-1}) + eps h_2(x_n, x_{n-1}, z_n, z_{n-1})

With y_n = x_{n-1}, w_n = z_{n-1}, the state is (x, y, z, w).
Perturbations: h_1 = z_n - z_{n-1}, h_2 = x_{n-1} - x_n.
D_A = D_B + 0.1, A = 0.4 fixed.

Figures:
  - Lyapunov exponents vs D_B for different eps
  - (x_n, z_n) projections showing 3-torus, lockings, chaos

OUTPUTS: figures/sec06_three_torus/*.npz, *.png
"""

import numpy as np

from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps._iter import run_animation_sweep, trajectory_after_transient

FIG_DIR = section_dir("sec06_three_torus")

LYAP_NPZ = FIG_DIR / "lyapunov_vs_DB.npz"
LYAP_PNG = FIG_DIR / "lyapunov_vs_DB.png"
PROJ_NPZ = FIG_DIR / "xz_projections.npz"
PROJ_PNG = FIG_DIR / "xz_projections.png"
ANIM_NPZ = FIG_DIR / "three_torus_animation.npz"
ANIM_GIF = FIG_DIR / "three_torus_animation.gif"

PROJ_SCHEMA_VERSION = 2
PROJECTION_CASES = (
    (2.37, "near-$T^3$ regime", "line"),
    (2.43, "resonance web", "line"),
    (2.45, "locked invariant circle", "line"),
    (2.478, "weak chaos", "density"),
    (2.497, "periodic window", "points"),
    (2.55, "developed chaos", "density"),
)


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------


def coupled_delayed(state, A, DA, DB, eps):
    """One iteration of the 4D coupled delayed logistic map.

    State = (x, y, z, w) where y = x_{n-1}, w = z_{n-1}.
    """
    x, y, z, w = state
    h1 = z - w  # z_n - z_{n-1}
    h2 = y - x  # x_{n-1} - x_n
    x_new = A * x + DA * y * (1.0 - y) + eps * h1
    z_new = A * z + DB * w * (1.0 - w) + eps * h2
    y_new = x
    w_new = z
    return np.array([x_new, y_new, z_new, w_new])


def coupled_delayed_jac(state, A, DA, DB, eps):
    """Jacobian of the coupled delayed logistic map."""
    x, y, z, w = state
    return np.array(
        [
            [A, DA * (1.0 - 2.0 * y), eps, -eps],
            [1.0, 0.0, 0.0, 0.0],
            [-eps, eps, A, DB * (1.0 - 2.0 * w)],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )


# ---------------------------------------------------------------------------
# Lyapunov computation
# ---------------------------------------------------------------------------


def compute_lyapunov():
    """Compute Lyapunov exponents vs D_B for several epsilon values."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

    A = 0.4
    eps_values = [1e-3, 5e-3, 1e-2]
    n_params = 500
    DB_values = np.linspace(2.1, 2.65, n_params)

    all_spectra = {}
    for eps in eps_values:
        print(f"  eps={eps}")
        spectra = np.empty((n_params, 4))
        for i, DB in enumerate(DB_values):
            DA = DB + 0.1
            x0 = np.array([0.5, 0.5, 0.3, 0.3])

            def f(s, _DA=DA, _DB=DB, _e=eps):
                return coupled_delayed(s, A, _DA, _DB, _e)

            def jac(s, _DA=DA, _DB=DB, _e=eps):
                return coupled_delayed_jac(s, A, _DA, _DB, _e)

            spectra[i] = lyapunov_spectrum(f, jac, x0, n_iter=30_000, n_transient=10_000)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{n_params}")
        all_spectra[f"eps_{eps}_spectra"] = spectra

    np.savez_compressed(LYAP_NPZ, DB=DB_values, eps_values=np.array(eps_values), **all_spectra)
    print(f"Saved {LYAP_NPZ}")


# ---------------------------------------------------------------------------
# (x, z) projections
# ---------------------------------------------------------------------------


def compute_projections():
    """Compute (x_n, z_n) projections at fixed eps, varying D_B."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A = 0.4
    eps = 5e-3
    DB_values = [case[0] for case in PROJECTION_CASES]
    labels = [case[1] for case in PROJECTION_CASES]
    render_modes = [case[2] for case in PROJECTION_CASES]

    results = {}
    for DB, label in zip(DB_values, labels, strict=False):
        DA = DB + 0.1
        print(f"  DB={DB} ({label})")
        traj = trajectory_after_transient(
            np.array([0.5, 0.5, 0.3, 0.3], dtype=np.float64),
            lambda state: coupled_delayed(state, A, DA, DB, eps),
            30_000,
            50_000,
            project_fn=lambda state: state[[0, 2]],
        )
        results[f"DB_{DB}_xz"] = traj

    results["DB_values"] = np.array(DB_values)
    results["labels"] = np.array(labels)
    results["render_modes"] = np.array(render_modes)
    results["schema_version"] = np.array([PROJ_SCHEMA_VERSION])
    np.savez_compressed(PROJ_NPZ, **results)
    print(f"Saved {PROJ_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_lyapunov(data):
    """Plot Lyapunov exponents vs D_B."""
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

    DB = data["DB"]
    eps_values = data["eps_values"]

    panel_spec = figure_spec("grid")
    fig, axes = plt.subplots(
        1,
        len(eps_values),
        figsize=(panel_spec.figsize[0], panel_spec.figsize[1] * 0.58),
        sharey=True,
    )
    gallery_db = np.array([case[0] for case in PROJECTION_CASES])
    for idx, eps in enumerate(eps_values):
        ax = axes[idx]
        spectra = data[f"eps_{eps}_spectra"]
        # Draw the zero-exponent guide first, below and thinner than the
        # data: at the default weight/zorder it rendered on top of
        # lambda_1/lambda_2 exactly where those curves flatten near zero,
        # hiding the two-vs-one-zero-exponent distinction the panel exists
        # to show.
        reference_line(ax, 0, axis="y", lw=0.5, zorder=1)
        for k in range(3):
            ax.plot(
                DB,
                spectra[:, k],
                color=lyap_color(k),
                lw=0.9,
                label=rf"$\lambda_{k + 1}$",
                zorder=2,
            )
        if np.isclose(eps, 5e-3):
            for db in gallery_db:
                ax.axvline(db, color=COLORS["grey"], lw=0.45, alpha=0.3)
        ax.set_xlabel(r"$D_B$")
        ax.set_title(rf"$\varepsilon = {eps}$", loc="left")
        panel_label(ax, chr(97 + idx))
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False)
        if idx == 0:
            ax.set_ylabel("Lyapunov exponent")
            finalize_legend(ax, kind="grid", loc="lower left")
    axes[0].set_ylim(-0.315, 0.115)
    fig.text(
        0.01,
        0.995,
        r"Coupled delayed logistic map: $\alpha = 0.4$, $D_A = D_B + 0.1$",
        ha="left",
        va="top",
        fontsize=panel_spec.legend_size,
    )
    fig.savefig(LYAP_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {LYAP_PNG}")


def plot_projections(data):
    """Plot (x_n, z_n) projections."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        CMAP_SEQUENTIAL,
        COLORS,
        add_field_colorbar,
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
    )

    setup()

    DB_values = data["DB_values"]
    labels = np.array([case[1] for case in PROJECTION_CASES])
    render_modes = data["render_modes"]

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 3, figsize=spec.figsize)
    # Panels (d) and (f) each grow their own colorbar via fig.colorbar(ax=...),
    # which shrinks only that axes and steals from the wspace gap rather than
    # from the whole row -- at the default spacing the colorbar (and its
    # rotated label) intruded into the neighbouring panel, printing "visitation
    # density" over panel (e) and overprinting panel (e)'s y-tick labels.
    fig.subplots_adjust(wspace=0.6, hspace=0.55)
    axes_flat = axes.flatten()
    x_limits = []
    z_limits = []
    for DB in DB_values:
        xz = data[f"DB_{DB}_xz"]
        x_limits.extend([xz[:, 0].min(), xz[:, 0].max()])
        z_limits.extend([xz[:, 1].min(), xz[:, 1].max()])
    x_pad = 0.03 * (max(x_limits) - min(x_limits))
    z_pad = 0.03 * (max(z_limits) - min(z_limits))
    xlim = (min(x_limits) - x_pad, max(x_limits) + x_pad)
    zlim = (min(z_limits) - z_pad, max(z_limits) + z_pad)

    for idx, DB in enumerate(DB_values):
        ax = axes_flat[idx]
        xz = data[f"DB_{DB}_xz"]
        mode = str(render_modes[idx])
        if mode == "line":
            alpha = 0.28 if idx == 0 else 0.09
            ax.scatter(
                xz[:, 0],
                xz[:, 1],
                s=0.08,
                c=COLORS["black"],
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )
        elif mode == "points":
            ax.scatter(xz[:, 0], xz[:, 1], s=2.1, c=COLORS["black"], alpha=0.9, linewidths=0)
        else:
            density = ax.hexbin(
                xz[:, 0],
                xz[:, 1],
                gridsize=60,
                bins="log",
                mincnt=1,
                cmap=CMAP_SEQUENTIAL,
                linewidths=0.0,
                rasterized=True,
            )
            add_field_colorbar(fig, density, ax, label="visitation density")
        ax.set_title(f"$D_B = {DB:.3f}$\n{labels[idx]}", loc="left")
        panel_label(ax, chr(97 + idx))
        if idx >= 3:
            ax.set_xlabel("$x$")
        else:
            ax.set_xlabel("")
        if idx % 3 == 0:
            ax.set_ylabel("$z$")
        else:
            ax.set_ylabel("")
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False, equal=True)

    fig.text(
        0.01,
        0.995,
        r"Representative $(x_n, z_n)$ projections at $\varepsilon = 5 \times 10^{-3}$",
        ha="left",
        va="top",
        fontsize=spec.legend_size,
    )
    fig.savefig(PROJ_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PROJ_PNG}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def compute_animation_data():
    """Sweep DB from 2.1 to 2.65 for the three-torus animation."""
    A = 0.4
    eps = 5e-3
    DB_sweep = np.linspace(2.1, 2.65, 300)
    x0 = np.array([0.5, 0.5, 0.3, 0.3], dtype=np.float64)

    def iterate_fn(DB):
        DA = DB + 0.1
        return trajectory_after_transient(
            x0,
            lambda state: coupled_delayed(state, A, DA, DB, eps),
            30_000,
            10_000,
            project_fn=lambda state: state[[0, 2]],
        )

    run_animation_sweep(
        iterate_fn,
        DB_sweep,
        ANIM_NPZ,
        n_plot=10_000,
        progress_interval=50,
    )


def make_animation_gif(data):
    """Create the three-torus dynamics GIF."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"],
        data["all_x"],
        data["all_y"],
        ANIM_GIF,
        title_template=r"Three-torus dynamics, $\varepsilon = 0.005$, $D_B = {param_value}$",
        param_name="D_B",
        param_fmt=".4f",
        xlabel="$x$",
        ylabel="$z$",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        lyap_data = safe_load(LYAP_NPZ)
        print(f"Loaded {LYAP_NPZ}")
    except FileNotFoundError:
        print("Computing Lyapunov spectra...")
        compute_lyapunov()
        lyap_data = safe_load(LYAP_NPZ)
    plot_lyapunov(lyap_data)

    try:
        proj_data = safe_load(PROJ_NPZ)
        print(f"Loaded {PROJ_NPZ}")
        schema_version = int(np.atleast_1d(proj_data.get("schema_version", np.array([0])))[0])
        if schema_version < PROJ_SCHEMA_VERSION:
            raise KeyError("stale projection cache")
        if "render_modes" not in proj_data.files:
            raise KeyError("projection metadata missing")
    except FileNotFoundError:
        print("Computing projections...")
        compute_projections()
        proj_data = safe_load(PROJ_NPZ)
    except KeyError:
        print("Recomputing projections with updated gallery metadata...")
        compute_projections()
        proj_data = safe_load(PROJ_NPZ)
    plot_projections(proj_data)

    # Animation
    try:
        anim_data = safe_load(ANIM_NPZ)
        print(f"Loaded {ANIM_NPZ}")
    except FileNotFoundError:
        print("Computing three-torus animation data...")
        compute_animation_data()
        anim_data = safe_load(ANIM_NPZ)
    make_animation_gif(anim_data)


if __name__ == "__main__":
    main()
