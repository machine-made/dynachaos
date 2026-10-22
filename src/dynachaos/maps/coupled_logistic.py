#!/usr/bin/env python3
"""
coupled_logistic: Symmetry breaking in the coupled logistic map.

Revisits Kaneko (1983) "Transition from Torus to Chaos Accompanied by
Frequency Lockings with Symmetry Breaking", PTP 69(5), 1427-1442.

Map (Eq. 1.1):
    x_{n+1} = 1 - A x_n^2 + D(y_n - x_n)
    y_{n+1} = 1 - A y_n^2 + D(x_n - y_n)

Figures:
  - Parameter survey in (A, D) using symmetry-breaking and Lyapunov observables
  - Representative attractor portraits at D=0.1 along the broken-symmetry route
  - Basin of attraction showing stripe accumulation near the invariant diagonal

OUTPUTS: figures/sec03_transition/phase_diagram.npz, phase_diagram.png
         figures/sec03_transition/attractors.npz, attractors.png
         figures/sec03_transition/basins.npz, basins.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from dynachaos.io.paths import (
    load_or_compute_npz,
    safe_load,
    section_dir,
)
from dynachaos.io.paths import (
    write_payload as _io_write_payload,
)
from dynachaos.maps._iter import (
    run_animation_sweep,
    run_transient,
    sample_trajectory,
    trajectory_after_transient,
)
from dynachaos.maps.primitives import logistic, logistic_derivative

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import coupled_logistic_basin_grid as _coupled_logistic_basin_grid_rs

    _RUST_AVAILABLE = True
except ImportError:
    _coupled_logistic_basin_grid_rs = None
    _RUST_AVAILABLE = False

FIG_DIR = section_dir("sec03_transition")

PHASE_NPZ = FIG_DIR / "phase_diagram.npz"
PHASE_PNG = FIG_DIR / "phase_diagram.png"
ATTR_NPZ = FIG_DIR / "attractors.npz"
ATTR_PNG = FIG_DIR / "attractors.png"
BASIN_NPZ = FIG_DIR / "basins.npz"
BASIN_PNG = FIG_DIR / "basins.png"
ANIM_NPZ = FIG_DIR / "attractors_animation.npz"
ANIM_GIF = FIG_DIR / "attractors_animation.gif"
PHASE_SCHEMA_VERSION = 3
PHASE_REQUIRED_KEYS = ("A", "D", "asym", "lyap", "schema_version")


@dataclass(frozen=True)
class PhaseDiagramPayload:
    """Typed NPZ payload for the schema-versioned phase diagram cache."""

    A: np.ndarray
    D: np.ndarray
    asym: np.ndarray
    lyap: np.ndarray
    schema_version: int = PHASE_SCHEMA_VERSION

    def to_npz(self) -> dict[str, np.ndarray]:
        return {
            "A": self.A,
            "D": self.D,
            "asym": self.asym,
            "lyap": self.lyap,
            "schema_version": np.array([self.schema_version], dtype=np.int16),
        }

    @classmethod
    def from_npz(cls, data) -> PhaseDiagramPayload:
        missing = tuple(key for key in PHASE_REQUIRED_KEYS if key not in data.files)
        if missing:
            raise KeyError("phase diagram cache missing keys: " + ", ".join(missing))

        version = int(np.atleast_1d(data["schema_version"])[0])
        if version < PHASE_SCHEMA_VERSION:
            raise ValueError("stale phase diagram cache")

        A = np.asarray(data["A"], dtype=np.float64)
        D = np.asarray(data["D"], dtype=np.float64)
        asym = np.asarray(data["asym"], dtype=np.float64)
        lyap = np.asarray(data["lyap"], dtype=np.float64)
        expected_shape = (len(D), len(A))
        if A.ndim != 1 or D.ndim != 1:
            raise ValueError("phase diagram cache axes must be one-dimensional")
        if asym.shape != expected_shape or lyap.shape != expected_shape:
            raise ValueError("phase diagram cache grid shape mismatch")

        return cls(
            A=A,
            D=D,
            asym=asym,
            lyap=lyap,
            schema_version=version,
        )


@dataclass(frozen=True)
class AttractorCase:
    """One representative attractor panel for the D=0.1 gallery."""

    A: float
    label: str
    initial_state: tuple[float, float]
    xlim: tuple[float, float]
    ylim: tuple[float, float]


ATTRACTOR_CASES = (
    AttractorCase(1.10, "2T", (0.1, 0.6), (-1.15, 1.15), (-1.15, 1.15)),
    AttractorCase(1.25, "4T", (0.1, 0.6), (-1.15, 1.15), (-1.15, 1.15)),
    AttractorCase(1.35, "8T", (0.1, 0.2), (-1.05, 1.05), (-1.05, 1.05)),
    AttractorCase(1.3525, "8T (zoom)", (0.1, 0.2), (-0.18, 0.18), (-0.18, 0.18)),
    AttractorCase(1.355, "8C (zoom)", (0.1, 0.2), (-0.18, 0.18), (-0.18, 0.18)),
    AttractorCase(1.373, "4C", (0.1, 0.2), (-1.05, 1.05), (-1.05, 1.05)),
)

# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------


def coupled_logistic(state, A, D):
    """One iteration of the coupled logistic map."""
    x, y = state
    x_new = logistic(x, A) + D * (y - x)
    y_new = logistic(y, A) + D * (x - y)
    return np.array([x_new, y_new])


def coupled_logistic_jac(state, A, D):
    """Jacobian of the coupled logistic map."""
    x, y = state
    return np.array([[logistic_derivative(x, A) - D, D], [D, logistic_derivative(y, A) - D]])


# ---------------------------------------------------------------------------
# Phase diagram computation
# ---------------------------------------------------------------------------


def compute_phase_diagram(
    *,
    A_values=None,
    D_values=None,
    n_transient=5000,
    n_sample=2000,
    output_path=PHASE_NPZ,
    progress_interval=50,
):
    """Compute phase diagram in (A, D) space.

    Vectorized: for each D, all A values are iterated simultaneously as
    NumPy arrays.

    Diagnostics:
      - asym = <|x - y|> : symmetry-breaking order parameter
      - lyap = finite-time largest Lyapunov exponent estimate
    """
    if A_values is None:
        A_values = np.linspace(0.5, 1.65, 500)
    else:
        A_values = np.atleast_1d(np.asarray(A_values, dtype=np.float64))
    if D_values is None:
        D_values = np.linspace(0.0, 0.3, 200)
    else:
        D_values = np.atleast_1d(np.asarray(D_values, dtype=np.float64))
    n_A, n_D = len(A_values), len(D_values)

    asym_grid = np.full((n_D, n_A), np.nan)
    lyap_grid = np.full((n_D, n_A), np.nan)

    for j, D in enumerate(D_values):
        # All A values in parallel
        x = np.full(n_A, 0.1)
        y = np.full(n_A, 0.2)

        # Transient
        for _ in range(n_transient):
            x_new = logistic(x, A_values) + D * (y - x)
            y_new = logistic(y, A_values) + D * (x - y)
            x, y = x_new, y_new
            # Clamp divergent orbits
            mask = (np.abs(x) > 10.0) | (np.abs(y) > 10.0)
            x = np.where(mask, np.nan, x)
            y = np.where(mask, np.nan, y)

        # Sample diagnostics and tangent-space growth
        sum_absdiff = np.zeros(n_A)
        count_absdiff = np.zeros(n_A)
        lyap_sum = np.zeros(n_A)
        lyap_count = np.zeros(n_A)
        vx = np.full(n_A, 1.0 / np.sqrt(2.0))
        vy = np.full(n_A, 1.0 / np.sqrt(2.0))

        for _ in range(n_sample):
            # Tangent evolution for largest Lyapunov exponent estimate.
            j11 = logistic_derivative(x, A_values) - D
            j22 = logistic_derivative(y, A_values) - D
            tx = j11 * vx + D * vy
            ty = D * vx + j22 * vy
            tnorm = np.sqrt(tx * tx + ty * ty)
            valid_tan = np.isfinite(x) & np.isfinite(y) & np.isfinite(tnorm) & (tnorm > 0.0)
            lyap_sum += np.where(valid_tan, np.log(tnorm), 0.0)
            lyap_count += valid_tan.astype(float)
            vx = np.where(valid_tan, tx / tnorm, vx)
            vy = np.where(valid_tan, ty / tnorm, vy)

            x_new = logistic(x, A_values) + D * (y - x)
            y_new = logistic(y, A_values) + D * (x - y)
            x, y = x_new, y_new
            mask = (np.abs(x) > 10.0) | (np.abs(y) > 10.0)
            x = np.where(mask, np.nan, x)
            y = np.where(mask, np.nan, y)
            valid_state = np.isfinite(x) & np.isfinite(y)
            sum_absdiff += np.where(valid_state, np.abs(x - y), 0.0)
            count_absdiff += valid_state.astype(float)

        asym_row = np.full(n_A, np.nan)
        lyap_row = np.full(n_A, np.nan)
        np.divide(sum_absdiff, count_absdiff, out=asym_row, where=count_absdiff > 0.0)
        np.divide(lyap_sum, lyap_count, out=lyap_row, where=lyap_count > 0.0)
        asym_grid[j] = asym_row
        lyap_grid[j] = lyap_row

        if output_path is not None and progress_interval and (j + 1) % progress_interval == 0:
            print(f"  Phase diagram: row {j + 1}/{n_D}")
            _io_write_payload(
                output_path,
                PhaseDiagramPayload(A_values, D_values, asym_grid, lyap_grid).to_npz(),
                base_dir=FIG_DIR,
            )

    return _io_write_payload(
        output_path,
        PhaseDiagramPayload(A_values, D_values, asym_grid, lyap_grid).to_npz(),
        base_dir=FIG_DIR,
    )


# ---------------------------------------------------------------------------
# Attractor portraits at D=0.1
# ---------------------------------------------------------------------------


def compute_attractors(
    *,
    cases=ATTRACTOR_CASES,
    D=0.1,
    n_transient=50_000,
    n_plot=100_000,
    output_path=ATTR_NPZ,
):
    """Compute representative attractor portraits at D=0.1.

    The first two panels use an off-diagonal initial condition to expose the
    symmetry-broken tori that coexist with synchronized cycles on x=y.
    """
    results = {}
    for idx, case in enumerate(cases):
        print(f"  A={case.A} ({case.label})")
        traj = trajectory_after_transient(
            np.array(case.initial_state, dtype=float),
            lambda state: coupled_logistic(state, case.A, D),
            n_transient,
            n_plot,
        )
        results[f"x_{idx}"] = traj[:, 0]
        results[f"y_{idx}"] = traj[:, 1]

    results["A_values"] = np.array([case.A for case in cases], dtype=np.float64)
    results["labels"] = np.array([case.label for case in cases])
    results["initial_states"] = np.array(
        [case.initial_state for case in cases],
        dtype=np.float64,
    )
    results["x_limits"] = np.array([case.xlim for case in cases], dtype=np.float64)
    results["y_limits"] = np.array([case.ylim for case in cases], dtype=np.float64)
    results["D"] = np.array([D])
    results["schema_version"] = np.array([4], dtype=np.int16)
    return _io_write_payload(output_path, results, base_dir=FIG_DIR)


# ---------------------------------------------------------------------------
# Basin of attraction (self-similar stripe structure)
# ---------------------------------------------------------------------------


def _find_reference_orbit(A, D, x0, y0, n_transient=500_000, period=32):
    """Find a reference periodic orbit by iterating from (x0, y0)."""
    state = run_transient(
        np.array([x0, y0], dtype=np.float64),
        lambda s: coupled_logistic(s, A, D),
        n_transient,
    )
    return sample_trajectory(
        state,
        lambda s: coupled_logistic(s, A, D),
        period,
    )


def _basin_grid_python(A, D, x_range, y_range, n_transient, ref_A):
    """Compute coupled-logistic basin labels using the NumPy row loop."""
    ref_B = ref_A[:, ::-1].copy()
    n_grid = len(x_range)
    basin = np.zeros((len(y_range), n_grid), dtype=np.int8)

    for j, y0 in enumerate(y_range):
        # Vectorize across all x values for this row
        x = x_range.copy()
        y = np.full(n_grid, y0)

        # Transient
        for _ in range(n_transient):
            x_new = logistic(x, A) + D * (y - x)
            y_new = logistic(y, A) + D * (x - y)
            x, y = x_new, y_new
            diverged = (np.abs(x) > 100) | (np.abs(y) > 100)
            x = np.where(diverged, np.nan, x)
            y = np.where(diverged, np.nan, y)

        # Classify by minimum distance to either reference orbit.
        # For each grid point, compute distance to all reference points on
        # each reference orbit and take the minimum.
        dist_A = np.full(n_grid, np.inf)
        dist_B = np.full(n_grid, np.inf)
        for k in range(len(ref_A)):
            d_a = (x - ref_A[k, 0]) ** 2 + (y - ref_A[k, 1]) ** 2
            d_b = (x - ref_B[k, 0]) ** 2 + (y - ref_B[k, 1]) ** 2
            dist_A = np.minimum(dist_A, d_a)
            dist_B = np.minimum(dist_B, d_b)

        basin[j] = np.where(
            np.isnan(x), -1, np.where(dist_A < dist_B, 1, np.where(dist_B < dist_A, 2, 0))
        ).astype(np.int8)

        if (j + 1) % 200 == 0:
            print(f"  Basins: row {j + 1}/{len(y_range)}")

    return basin


def _basin_grid(A, D, x_range, y_range, n_transient, ref_A):
    if _RUST_AVAILABLE and _coupled_logistic_basin_grid_rs is not None:
        return np.asarray(
            _coupled_logistic_basin_grid_rs(
                np.ascontiguousarray(x_range, dtype=np.float64),
                np.ascontiguousarray(y_range, dtype=np.float64),
                A,
                D,
                int(n_transient),
                np.ascontiguousarray(ref_A, dtype=np.float64),
            )
        )

    return _basin_grid_python(A, D, x_range, y_range, n_transient, ref_A)


def compute_basins(
    *,
    A=1.35344,
    D=0.1,
    n_grid=800,
    n_transient=50_000,
    reference_transient=500_000,
    period=32,
    output_path=BASIN_NPZ,
):
    """Compute basin of attraction showing stripe structure.

    At A=1.35344, D=0.1, two coexisting asymmetric period-32 cycles
    exist (Kaneko 1983, Fig. 8).  They are mirror images about y=x.
    The basin boundary between them forms self-similar stripes near
    the y=x line.

    Classification: after a long transient, the final state is compared
    to both reference orbits; the closer one determines the basin.

    Vectorized: each row of the grid processes all x values in parallel.
    """
    # Pre-compute the two reference orbits (mirror images about y=x)
    print("  Computing reference orbits...")
    ref_A = _find_reference_orbit(A, D, 0.1, 0.6, n_transient=reference_transient, period=period)

    x_range = np.linspace(-1.0, 1.0, n_grid)
    y_range = np.linspace(-1.0, 1.0, n_grid)
    basin = _basin_grid(A, D, x_range, y_range, n_transient, ref_A)

    return _io_write_payload(
        output_path,
        {"x": x_range, "y": y_range, "basin": basin, "A": np.array([A]), "D": np.array([D])},
        base_dir=FIG_DIR,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_phase_diagram(data):
    """Plot a coarse parameter survey in (A, D) space."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    from dynachaos.utils.style import (
        CMAP_DIVERGING,
        CMAP_SEQUENTIAL,
        COLORS,
        add_field_colorbar,
        annotate_on_field,
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
    )

    setup()

    A = data["A"]
    D = data["D"]
    asym = data["asym"]
    lyap = data["lyap"]

    spec = figure_spec("double")
    fig, axes = plt.subplots(1, 2, figsize=(spec.figsize[0], spec.figsize[1] + 0.25), sharey=True)
    fig.subplots_adjust(wspace=0.18, bottom=0.18)

    asym_cmap = plt.get_cmap(CMAP_SEQUENTIAL).copy()
    asym_cmap.set_bad(COLORS["offwhite"])
    asym_valid = asym[np.isfinite(asym)]
    asym_vmax = float(np.nanpercentile(asym_valid, 99.0)) if asym_valid.size else 0.2
    asym_vmax = max(asym_vmax, 1e-3)
    im0 = axes[0].pcolormesh(
        A,
        D,
        asym,
        cmap=asym_cmap,
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=asym_vmax),
        rasterized=True,
        shading="auto",
    )
    axes[0].axhline(0.1, color=COLORS["red"], linestyle=(0, (4, 2)), lw=0.8)
    annotate_on_field(
        axes[0],
        A[-1],
        0.1 + 0.025 * (D[-1] - D[0]),
        r"$\varepsilon = 0.1$ gallery slice",
        ha="right",
        va="bottom",
        color=COLORS["red"],
        fontsize=spec.legend_size,
    )
    axes[0].set_xlabel(r"$a$")
    axes[0].set_ylabel(r"$\varepsilon$")
    axes[0].set_title(r"Symmetry breaking: $\langle |x-y| \rangle$", loc="left")
    panel_label(axes[0], "(a)")
    apply_axes_polish(axes[0], kind="double", title_loc="left", grid=False)
    # Panel (b) sits immediately right of this colorbar, so a right-side
    # label lands inside its axes; put the label in the pad gap on the left.
    cbar0 = add_field_colorbar(fig, im0, axes[0], label=r"$\langle |x-y| \rangle$", pad=0.12)
    cbar0.ax.yaxis.set_label_position("left")

    lyap_cmap = plt.get_cmap(CMAP_DIVERGING).copy()
    lyap_cmap.set_bad(COLORS["offwhite"])
    lyap_valid = lyap[np.isfinite(lyap)]
    lyap_lim = float(np.nanpercentile(np.abs(lyap_valid), 99.0)) if lyap_valid.size else 0.1
    lyap_lim = max(lyap_lim, 0.02)
    im1 = axes[1].pcolormesh(
        A,
        D,
        lyap,
        cmap=lyap_cmap,
        vmin=-lyap_lim,
        vmax=lyap_lim,
        rasterized=True,
        shading="auto",
    )
    axes[1].axhline(0.1, color=COLORS["red"], linestyle=(0, (4, 2)), lw=0.8)
    annotate_on_field(
        axes[1],
        A[-1],
        0.1 + 0.025 * (D[-1] - D[0]),
        r"$\varepsilon = 0.1$ gallery slice",
        ha="right",
        va="bottom",
        color=COLORS["red"],
        fontsize=spec.legend_size,
    )
    axes[1].set_xlabel(r"$a$")
    axes[1].set_title(r"Chaos onset: $\lambda_1$ (finite-time)", loc="left")
    panel_label(axes[1], "(b)")
    apply_axes_polish(axes[1], kind="double", title_loc="left", grid=False)
    add_field_colorbar(fig, im1, axes[1], label=r"$\lambda_1$", pad=0.02)

    fig.savefig(PHASE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PHASE_PNG}")


def plot_attractors(data):
    """Plot representative attractor portraits."""
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

    A_values = data["A_values"]
    labels = data["labels"]
    initial_states = data["initial_states"]
    x_limits = data["x_limits"]
    y_limits = data["y_limits"]
    panel_labels = list("abcdef")

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 3, figsize=(spec.figsize[0], spec.figsize[1] + 0.15))
    fig.subplots_adjust(wspace=0.22, hspace=0.3, top=0.92)
    fig.text(
        0.01,
        0.975,
        r"Coupled logistic map at $\varepsilon = 0.1$",
        ha="left",
        va="top",
        fontsize=spec.title_size,
    )
    axes_flat = axes.flatten()

    for idx, A in enumerate(A_values):
        ax = axes_flat[idx]
        x = data[f"x_{idx}"]
        y = data[f"y_{idx}"]
        point_size = 0.06 if "zoom" not in labels[idx] else 0.045
        ax.scatter(x, y, s=point_size, c=COLORS["black"], alpha=0.16, rasterized=True)
        reference_line(ax, 0.0, axis="y", lw=0.55, zorder=0)
        reference_line(ax, 0.0, axis="x", lw=0.55, zorder=0)
        ax.set_title(
            f"$a = {float(A):.4g}$, {labels[idx]}",
            loc="left",
            usetex=False,
        )
        panel_label(ax, panel_labels[idx], loc="upper left")
        if idx >= 3:
            ax.set_xlabel("$x$")
        else:
            ax.set_xlabel("")
        if idx % 3 == 0:
            ax.set_ylabel("$y$")
        else:
            ax.set_ylabel("")
        ax.set_xlim(*x_limits[idx])
        ax.set_ylim(*y_limits[idx])
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False, equal=True)
        if idx < 2:
            x0, y0 = initial_states[idx]
            ax.text(
                0.03,
                0.04,
                rf"$({x0:.1f}, {y0:.1f})$ seed",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=spec.legend_size,
                color=COLORS["grey"],
            )

    fig.savefig(ATTR_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ATTR_PNG}")


def plot_basins(data):
    """Plot basin of attraction with a zoom panel near y=x."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
    )

    setup()

    x = data["x"]
    y = data["y"]
    basin = data["basin"]
    A_val = data["A"][0]

    cmap = ListedColormap(
        [
            COLORS["offwhite"],
            COLORS["grey"],
            COLORS["blue"],
            COLORS["red"],
        ]
    )
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(spec.figsize[0], spec.figsize[1] + 0.15))
    fig.subplots_adjust(wspace=0.18, bottom=0.2)
    extent = (x[0], x[-1], y[0], y[-1])

    ax1.imshow(
        basin,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    ax1.plot(x, x, color=COLORS["black"], linestyle="--", lw=0.45, alpha=0.45)
    ax1.set_xlabel("$x_0$")
    ax1.set_ylabel("$y_0$")
    ax1.set_title(f"Full basin at $a = {A_val:.5f}$", loc="left")
    # Opaque backing: the default semi-transparent tag picks up a two-tone
    # tint from the red/blue basin field directly underneath it.
    panel_label(
        ax1,
        "(a)",
        bbox={"facecolor": COLORS["offwhite"], "alpha": 1.0, "pad": 1.2, "edgecolor": "none"},
    )
    apply_axes_polish(ax1, kind="double", title_loc="left", grid=False, equal=True)

    zx0, zx1, zy0, zy1 = -0.16, 0.04, -0.11, 0.09
    rect = plt.Rectangle(
        (zx0, zy0),
        zx1 - zx0,
        zy1 - zy0,
        lw=0.8,
        ec=COLORS["black"],
        fc="none",
    )
    ax1.add_patch(rect)

    ax2.imshow(
        basin,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    ax2.plot(
        [-0.2, 0.1],
        [-0.2, 0.1],
        color=COLORS["black"],
        linestyle="--",
        lw=0.45,
        alpha=0.45,
    )
    ax2.set_xlim(zx0, zx1)
    ax2.set_ylim(zy0, zy1)
    ax2.set_xlabel("$x_0$")
    ax2.set_ylabel("$y_0$")
    ax2.set_title("Zoom near the invariant diagonal", loc="left")
    panel_label(
        ax2,
        "(b)",
        bbox={"facecolor": COLORS["offwhite"], "alpha": 1.0, "pad": 1.2, "edgecolor": "none"},
    )
    apply_axes_polish(ax2, kind="double", title_loc="left", grid=False, equal=True)

    legend_elements = [
        Patch(facecolor=COLORS["grey"], label="Synchronized 4-cycle"),
        Patch(facecolor=COLORS["blue"], label="Asymmetric orbit A"),
        Patch(facecolor=COLORS["red"], label="Asymmetric orbit B"),
    ]
    if np.any(basin == -1):
        legend_elements.insert(
            0,
            Patch(facecolor=COLORS["offwhite"], edgecolor=COLORS["grey"], label="Diverged"),
        )
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        fontsize=spec.legend_size,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Higher dpi than the section default: the 800x800 basin grid needs
    # >=2 device pixels per cell so the web-derivative downsample area-averages
    # the field instead of aliasing it into salt-and-pepper speckle.
    fig.savefig(BASIN_PNG, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {BASIN_PNG}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def compute_animation_data():
    """Sweep A from 0.9 to 1.45 at D=0.1 for animation."""
    D = 0.1
    A_sweep = np.linspace(0.9, 1.45, 200)

    def iterate_fn(A):
        return trajectory_after_transient(
            np.array([0.1, 0.2], dtype=np.float64),
            lambda state: coupled_logistic(state, A, D),
            20_000,
            5_000,
        )

    return run_animation_sweep(iterate_fn, A_sweep, ANIM_NPZ, n_plot=5_000)


def make_animation_gif(data):
    """Create GIF of attractor evolution across A."""
    from dynachaos.utils.animation import make_attractor_gif

    make_attractor_gif(
        data["param_values"],
        data["all_x"],
        data["all_y"],
        ANIM_GIF,
        title_template=r"Coupled logistic map, $\varepsilon = 0.1$, $a = {param_value}$",
        param_name="a",
        param_fmt=".3f",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Phase diagram
    try:
        phase_data = safe_load(PHASE_NPZ)
        print(f"Loaded {PHASE_NPZ}")
        try:
            phase_payload = PhaseDiagramPayload.from_npz(phase_data)
        except (KeyError, ValueError):
            phase_data.close()
            print("Phase cache missing updated diagnostics; recomputing...")
            phase_data = compute_phase_diagram()
        else:
            phase_data.close()
            phase_data = phase_payload.to_npz()
    except FileNotFoundError:
        print("Computing phase diagram...")
        phase_data = compute_phase_diagram()
    plot_phase_diagram(phase_data)

    # Attractors
    _attr_npz = None
    try:
        _attr_npz = safe_load(ATTR_NPZ)
        print(f"Loaded {ATTR_NPZ}")
        attr_needs_recompute = (
            "schema_version" not in _attr_npz.files
            or int(_attr_npz["schema_version"][0]) < 4
            or "A_values" not in _attr_npz.files
            or "labels" not in _attr_npz.files
            or "initial_states" not in _attr_npz.files
            or "x_limits" not in _attr_npz.files
            or "y_limits" not in _attr_npz.files
        )
        if not attr_needs_recompute:
            for idx in range(len(ATTRACTOR_CASES)):
                if f"x_{idx}" not in _attr_npz.files or f"y_{idx}" not in _attr_npz.files:
                    attr_needs_recompute = True
                    break
        if attr_needs_recompute:
            _attr_npz.close()
            _attr_npz = None
            print("Attractor cache schema mismatch; recomputing...")
            attr_data = compute_attractors()
        else:
            attr_data = _attr_npz
    except FileNotFoundError:
        print("Computing attractors...")
        attr_data = compute_attractors()
    try:
        plot_attractors(attr_data)
    finally:
        if _attr_npz is not None:
            _attr_npz.close()

    basin_data = load_or_compute_npz(
        BASIN_NPZ,
        "basins",
        compute_basins,
        required_keys=("x", "y", "basin", "A", "D"),
    )
    plot_basins(basin_data)

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
