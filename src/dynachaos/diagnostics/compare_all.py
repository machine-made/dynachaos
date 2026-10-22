#!/usr/bin/env python3
"""
compare_all: Apply modern diagnostics to Kaneko's maps (Section 11).

Applies all four modern diagnostic tools to representative Kaneko maps:
  1. 0-1 test for chaos
  2. SALI (Smaller Alignment Index)
  3. Permutation entropy and complexity-entropy plane
  4. Recurrence Quantification Analysis

Maps tested:
  - Logistic map (1D): periodic vs chaotic regimes
  - Delayed logistic (2D): torus -> fractalization -> chaos

OUTPUTS: figures/sec11_diagnostics/*.npz, *.png
USAGE:   python src/dynachaos/diagnostics/compare_all.py
"""

import numpy as np

from dynachaos.diagnostics.compare_all_helpers import (
    delayed_logistic_series,
    delayed_logistic_trajectory,
    load_or_compute_npz,
    logistic_series,
    sweep_pair_metric,
    sweep_scalar_metric,
)
from dynachaos.io.paths import section_dir
from dynachaos.maps.coupled_delayed import (
    coupled_delayed as _coupled_delayed_map,
)
from dynachaos.maps.coupled_delayed import (
    coupled_delayed_jac as _coupled_delayed_jac,
)

FIG_DIR = section_dir("sec11_diagnostics")

TEST01_NPZ = FIG_DIR / "test01_sweep.npz"
TEST01_PNG = FIG_DIR / "test01_sweep.png"
SALI_NPZ = FIG_DIR / "sali_comparison.npz"
SALI_PNG = FIG_DIR / "sali_comparison.png"
COUPLED_LYAP_NPZ = section_dir("sec06_three_torus") / "lyapunov_vs_DB.npz"
PE_NPZ = FIG_DIR / "permutation_entropy.npz"
PE_PNG = FIG_DIR / "permutation_entropy.png"
CH_NPZ = FIG_DIR / "complexity_entropy_plane.npz"
CH_PNG = FIG_DIR / "complexity_entropy_plane.png"
RQA_NPZ = FIG_DIR / "rqa_measures.npz"
RQA_PNG = FIG_DIR / "rqa_measures.png"


# ---------------------------------------------------------------------------
# 0-1 test sweep
# ---------------------------------------------------------------------------


def compute_01_test():
    """0-1 test for chaos across the logistic map parameter range."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.zero_one_test import zero_one_statistic

    n_a = 500
    a_values = np.linspace(1.0, 2.0, n_a)
    K_values = np.empty(n_a)

    for i, a in enumerate(a_values):
        series = logistic_series(a, n_transient=5000, n_record=5000)
        K_values[i] = zero_one_statistic(series, n_c=50)
        if (i + 1) % 100 == 0:
            print(f"  0-1 test: {i + 1}/{n_a}")
            np.savez_compressed(TEST01_NPZ, a=a_values[: i + 1], K=K_values[: i + 1])

    np.savez_compressed(TEST01_NPZ, a=a_values, K=K_values)
    print(f"Saved {TEST01_NPZ}")


# ---------------------------------------------------------------------------
# SALI comparison
# ---------------------------------------------------------------------------

SALI_CASES = [
    (2.35, "3-torus"),
    (2.37, "locking"),
    (2.47, "onset of chaos"),
    (2.55, "developed chaos"),
]


def _sali_lambda1_values(DB_values, *, eps=5e-3):
    with np.load(COUPLED_LYAP_NPZ, allow_pickle=False) as data:
        DB_grid = data["DB"]
        spectra = data[f"eps_{eps:g}_spectra"]
        return np.array([spectra[np.argmin(np.abs(DB_grid - DB)), 0] for DB in DB_values])


def compute_sali():
    """SALI time series for coupled delayed logistic (4D) at various regimes.

    Uses the 4D coupled delayed logistic map so that SALI has enough
    dimensions to distinguish torus from chaos — in 2D, both deviation
    vectors must align regardless of dynamics (only the decay rate differs).
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.sali_gali import sali

    A = 0.4
    eps = 5e-3

    results = {}
    for DB, label in SALI_CASES:
        DA = DB + 0.1
        print(f"  SALI: DB={DB} ({label})")
        x0 = np.array([0.5, 0.5, 0.3, 0.3])

        def f(s, _DA=DA, _DB=DB):
            return _coupled_delayed_map(s, A, _DA, _DB, eps)

        def jac(s, _DA=DA, _DB=DB):
            return _coupled_delayed_jac(s, A, _DA, _DB, eps)

        s = sali(f, jac, x0, n_iter=10_000, n_transient=5000)
        results[f"DB_{DB}_sali"] = s

    DB_values = np.array([case[0] for case in SALI_CASES])
    results["DB_values"] = DB_values
    results["lambda1_values"] = _sali_lambda1_values(DB_values, eps=eps)
    np.savez_compressed(SALI_NPZ, **results)
    print(f"Saved {SALI_NPZ}")


# ---------------------------------------------------------------------------
# Permutation entropy sweep
# ---------------------------------------------------------------------------


def compute_permutation_entropy():
    """Permutation entropy across logistic map and delayed logistic."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.permutation import permutation_entropy

    # Logistic map sweep
    n_a = 500
    a_values = np.linspace(1.0, 2.0, n_a)
    H_logistic = sweep_scalar_metric(
        a_values,
        lambda a: logistic_series(a, n_transient=5000, n_record=5000),
        lambda series: permutation_entropy(series, d=5),
        progress_every=100,
        progress_label="PE logistic",
    )

    # Delayed logistic sweep
    n_D = 300
    D_values = np.linspace(1.5, 2.2, n_D)
    H_delayed = sweep_scalar_metric(
        D_values,
        lambda D: delayed_logistic_series(D, n_transient=5000, n_record=5000),
        lambda series: permutation_entropy(series, d=5),
        progress_every=100,
        progress_label="PE delayed",
    )

    np.savez_compressed(PE_NPZ, a=a_values, H_logistic=H_logistic, D=D_values, H_delayed=H_delayed)
    print(f"Saved {PE_NPZ}")


# ---------------------------------------------------------------------------
# Complexity-entropy plane
# ---------------------------------------------------------------------------


def compute_complexity_entropy():
    """Map different dynamical regimes onto the C-H plane."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.permutation import complexity_entropy

    # Sample many parameter values for the logistic map
    a_values = np.linspace(1.0, 2.0, 200)
    H_vals, C_vals = sweep_pair_metric(
        a_values,
        lambda a: logistic_series(a, n_transient=5000, n_record=5000),
        lambda series: complexity_entropy(series, d=5),
        progress_every=50,
        progress_label="C-H logistic",
    )

    # Delayed logistic
    D_values = np.linspace(1.5, 2.2, 200)
    H_del, C_del = sweep_pair_metric(
        D_values,
        lambda D: delayed_logistic_series(D, n_transient=5000, n_record=5000),
        lambda series: complexity_entropy(series, d=5),
        progress_every=50,
        progress_label="C-H delayed",
    )

    np.savez_compressed(
        CH_NPZ,
        a=a_values,
        H_logistic=H_vals,
        C_logistic=C_vals,
        D=D_values,
        H_delayed=H_del,
        C_delayed=C_del,
    )
    print(f"Saved {CH_NPZ}")


# ---------------------------------------------------------------------------
# RQA measures vs parameter
# ---------------------------------------------------------------------------


def compute_rqa():
    """RQA measures along the delayed logistic transition."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from dynachaos.diagnostics.recurrence import recurrence_matrix, rqa

    A = 0.3
    n_D = 80
    D_values = np.linspace(1.5, 2.2, n_D)

    RR = np.empty(n_D)
    DET = np.empty(n_D)
    LAM = np.empty(n_D)
    ENTR = np.empty(n_D)

    for i, D in enumerate(D_values):
        traj = delayed_logistic_trajectory(D, A=A, n_transient=10_000, n_record=2000)

        R, _ = recurrence_matrix(traj, percentile=5)
        stats = rqa(R, l_min=2, v_min=2)
        RR[i] = stats["RR"]
        DET[i] = stats["DET"]
        LAM[i] = stats["LAM"]
        ENTR[i] = stats["ENTR"]

        if (i + 1) % 20 == 0:
            print(f"  RQA: {i + 1}/{n_D}")
            np.savez_compressed(
                RQA_NPZ,
                D=D_values[: i + 1],
                RR=RR[: i + 1],
                DET=DET[: i + 1],
                LAM=LAM[: i + 1],
                ENTR=ENTR[: i + 1],
            )

    np.savez_compressed(RQA_NPZ, D=D_values, RR=RR, DET=DET, LAM=LAM, ENTR=ENTR)
    print(f"Saved {RQA_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_01_test(data):
    """Plot 0-1 test K vs a for the logistic map."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, reference_line, setup

    setup()

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    ax.plot(
        data["a"],
        data["K"],
        "o",
        ms=2.6,
        alpha=0.55,
        markeredgewidth=0.0,
        rasterized=True,
        color=COLORS["black"],
    )
    reference_line(ax, 0.5, lw=0.7, alpha=0.55)
    ax.set_xlabel(r"Nonlinearity $a$")
    ax.set_ylabel(r"$K_{01}$ (0-1 test)")
    ax.set_title(r"0-1 test for chaos: logistic map $f(x) = 1 - ax^2$", loc="left")
    apply_axes_polish(ax, kind="double", title_loc="left")
    ax.set_ylim(-0.1, 1.1)
    ax.text(
        1.5,
        0.6,
        "$K_{01} \\approx 1$: chaos",
        fontsize=spec.tick_size * 1.5,
        color=COLORS["grey"],
    )
    ax.text(
        1.1,
        0.15,
        "$K_{01} \\approx 0$: regular",
        fontsize=spec.tick_size * 1.5,
        color=COLORS["grey"],
    )

    fig.savefig(TEST01_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {TEST01_PNG}")


def _mathtext_sci(value, sig=2):
    """Format a float as consistent mathtext scientific notation, e.g. 9.46e-06."""
    mantissa, exponent = f"{value:.{sig}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def plot_sali(data):
    """Plot SALI time series for different dynamical regimes."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        color_for,
        figure_spec,
        finalize_legend,
        marker_for,
        reference_line,
        setup,
    )

    setup()

    DB_values = data["DB_values"]
    labels = [case[1] for case in SALI_CASES]
    lambda1_values = data["lambda1_values"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    for idx, DB in enumerate(DB_values):
        s = data[f"DB_{DB}_sali"]
        # Avoid log(0)
        floor = 1e-16
        valid = s > floor
        s_safe = np.where(valid, s, floor)
        n = np.arange(len(s))
        lam_str = _mathtext_sci(lambda1_values[idx])
        label = rf"$D_B={DB}$ ({labels[idx]}, $\lambda_1={lam_str}$)"
        ax.semilogy(
            n,
            s_safe,
            lw=1.1,
            color=color_for(idx),
            label=label,
        )
        # Only mark points that are genuinely above the machine-underflow
        # floor: once a curve saturates at 1e-16, drawing periodic markers
        # along the clipped floor reads as a separate dataset lying on the
        # frame rather than as a saturated series.
        step = max(len(s) // 14, 1)
        marker_idx = n[::step]
        marker_idx = marker_idx[valid[marker_idx]]
        ax.plot(
            marker_idx,
            s_safe[marker_idx],
            linestyle="none",
            marker=marker_for(idx),
            markersize=3.0,
            color=color_for(idx),
            markerfacecolor=COLORS["offwhite"],
            markeredgewidth=0.8,
        )

    ax.set_xlabel("Iteration $n$")
    ax.set_ylabel("SALI")
    ax.set_title(
        r"SALI: coupled delayed logistic map, $\alpha = 0.4$, "
        r"$\varepsilon = 5 \times 10^{-3}$",
        loc="left",
    )
    apply_axes_polish(ax, kind="double", title_loc="left")
    ax.set_ylim(1e-16, 10)
    reference_line(ax, 1e-16, lw=0.7, alpha=0.55)
    finalize_legend(ax, kind="double", loc="lower left")

    fig.savefig(SALI_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {SALI_PNG}")


def plot_permutation_entropy(data):
    """Plot permutation entropy sweeps."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
        system_color,
    )

    setup()

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=spec.figsize)
    h_all = np.concatenate([data["H_logistic"], data["H_delayed"]])
    h_range = h_all.max() - h_all.min()
    y_pad_top = 0.04 * h_range
    # The a<1.25 period-doubling plateau sits at the data floor; a 4% pad
    # still lands the curve on the spine, so give the bottom more room.
    y_pad_bottom = 0.10 * h_range

    ax1.plot(
        data["a"],
        data["H_logistic"],
        color=system_color("logistic"),
        lw=0.9,
        rasterized=True,
    )
    ax1.set_xlabel(r"$a$")
    ax1.set_ylabel(r"$H_\mathrm{PE}$")
    ax1.set_title(r"Logistic map $1-ax^2$", loc="left")
    ax1.set_ylim(h_all.min() - y_pad_bottom, h_all.max() + y_pad_top)
    apply_axes_polish(ax1, kind="double", title_loc="left", grid=False)
    panel_label(ax1, "(a)")

    ax2.plot(
        data["D"],
        data["H_delayed"],
        color=system_color("delayed_logistic"),
        lw=0.9,
        rasterized=True,
    )
    ax2.set_xlabel(r"$D$")
    ax2.set_ylabel(r"$H_\mathrm{PE}$")
    ax2.set_title(r"Delayed logistic, $\alpha = 0.3$", loc="left")
    ax2.set_ylim(h_all.min() - y_pad_bottom, h_all.max() + y_pad_top)
    apply_axes_polish(ax2, kind="double", title_loc="left", grid=False)
    panel_label(ax2, "(b)")

    fig.savefig(PE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PE_PNG}")


def plot_complexity_entropy(data):
    """Plot the complexity-entropy plane."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        setup,
        system_color,
    )

    setup()

    spec = figure_spec("single")
    fig, ax = plt.subplots(figsize=(spec.figsize[0] * 1.25, spec.figsize[1] + 0.35))

    ax.scatter(
        data["H_logistic"],
        data["C_logistic"],
        s=8,
        alpha=0.55,
        c=system_color("logistic"),
        label="Logistic map",
        rasterized=True,
    )
    ax.scatter(
        data["H_delayed"],
        data["C_delayed"],
        s=8,
        alpha=0.55,
        c=system_color("delayed_logistic"),
        label="Delayed logistic",
        rasterized=True,
    )

    h_all = np.concatenate([data["H_logistic"], data["H_delayed"]])
    c_all = np.concatenate([data["C_logistic"], data["C_delayed"]])
    h_pad = 0.05 * (h_all.max() - h_all.min())
    c_pad = 0.08 * (c_all.max() - c_all.min())

    ax.set_xlabel(r"Normalised permutation entropy $H$")
    ax.set_ylabel(r"Statistical complexity $C$")
    ax.set_title("Complexity-entropy plane ($d=5$)", loc="left")
    ax.set_xlim(h_all.min() - h_pad, h_all.max() + h_pad)
    ax.set_ylim(max(0.0, c_all.min() - c_pad), c_all.max() + c_pad)
    apply_axes_polish(ax, kind="single", title_loc="left", grid=False)
    finalize_legend(ax, kind="single", markerscale=2.6, loc="upper left")

    fig.savefig(CH_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {CH_PNG}")


def plot_rqa(data):
    """Plot RQA measures vs D for the delayed logistic."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, panel_label, setup

    setup()

    D = data["D"]

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 2, figsize=spec.figsize, sharex=True)
    fig.suptitle(
        "RQA: delayed logistic map, $\\alpha = 0.3$",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=spec.title_size,
    )

    axes[0, 0].plot(D, data["RR"], color=COLORS["black"], linestyle="-", lw=1.0)
    axes[0, 0].set_ylabel("RR")
    axes[0, 0].set_title("Recurrence rate", loc="left")
    apply_axes_polish(axes[0, 0], kind="grid", title_loc="left")
    panel_label(axes[0, 0], "(a)")

    axes[0, 1].plot(D, data["DET"], color=COLORS["black"], linestyle="-", lw=1.0)
    axes[0, 1].set_ylabel("DET")
    axes[0, 1].set_title("Determinism", loc="left")
    apply_axes_polish(axes[0, 1], kind="grid", title_loc="left")
    panel_label(axes[0, 1], "(b)")

    axes[1, 0].plot(D, data["LAM"], color=COLORS["black"], linestyle="-", lw=1.0)
    axes[1, 0].set_ylabel("LAM")
    axes[1, 0].set_xlabel(r"$D$")
    axes[1, 0].set_title("Laminarity", loc="left")
    apply_axes_polish(axes[1, 0], kind="grid", title_loc="left")
    panel_label(axes[1, 0], "(c)")

    axes[1, 1].plot(D, data["ENTR"], color=COLORS["black"], linestyle="-", lw=1.0)
    axes[1, 1].set_ylabel("ENTR")
    axes[1, 1].set_xlabel(r"$D$")
    axes[1, 1].set_title("Entropy of diag. lines", loc="left")
    apply_axes_polish(axes[1, 1], kind="grid", title_loc="left")
    # The ENTR curve dips sharply right where the default upper-left tag
    # sits (D ~ 1.51-1.55); the data line and its close cousin dip cut
    # straight through the label there. Bottom-left is clear for this panel.
    panel_label(axes[1, 1], "(d)", loc="lower left")

    fig.subplots_adjust(wspace=0.24, hspace=0.34)
    fig.savefig(RQA_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RQA_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sections = [
        ("0-1 test", TEST01_NPZ, compute_01_test, plot_01_test, ("a", "K")),
        (
            "SALI",
            SALI_NPZ,
            compute_sali,
            plot_sali,
            (
                "DB_values",
                "lambda1_values",
                "DB_2.35_sali",
                "DB_2.37_sali",
                "DB_2.47_sali",
                "DB_2.55_sali",
            ),
        ),
        (
            "Permutation entropy",
            PE_NPZ,
            compute_permutation_entropy,
            plot_permutation_entropy,
            ("a", "H_logistic", "D", "H_delayed"),
        ),
        (
            "C-H plane",
            CH_NPZ,
            compute_complexity_entropy,
            plot_complexity_entropy,
            ("a", "H_logistic", "C_logistic", "D", "H_delayed", "C_delayed"),
        ),
        ("RQA", RQA_NPZ, compute_rqa, plot_rqa, ("D", "RR", "DET", "LAM", "ENTR")),
    ]

    for name, npz_path, compute_fn, plot_fn, required_keys in sections:
        data = load_or_compute_npz(npz_path, name, compute_fn, required_keys=required_keys)
        plot_fn(data)


if __name__ == "__main__":
    main()
