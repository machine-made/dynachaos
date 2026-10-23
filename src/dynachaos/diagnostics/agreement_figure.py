#!/usr/bin/env python3
"""
agreement_figure: agreement of K_01, H_PE, DET and LAM with lambda_1 (Section 11).

Three test systems, each on the parameter grid of its Section 11 sweep:

  - logistic map x -> 1 - a x^2, a in [1, 2], 500 values
  - delayed logistic map, alpha = 0.3, D in [1.5, 2.2], 300 values
  - coupled delayed logistic map, alpha = 0.4, eps = 5e-3, D_B in [2.1, 2.65], 500 values

At every grid point the Lyapunov spectrum with block standard errors labels the
point chaotic, regular or marginal (see :mod:`dynachaos.diagnostics.agreement`).
The diagnostics use the series and settings of the Section 11 sweeps: 5000
samples after a transient for K_01 (n_c = 50) and H_PE (d = 5), and 2000 states
at the 5th percentile threshold for DET and LAM (l_min = v_min = 2).

OUTPUTS: figures/sec11_diagnostics/agreement_scores.npz, *.png
USAGE:   python src/dynachaos/diagnostics/agreement_figure.py
"""

import numpy as np

from dynachaos.diagnostics.agreement import (
    CHAOTIC,
    MARGINAL,
    REGULAR,
    label_sweep,
    roc_curve,
    scored_agreement,
)
from dynachaos.diagnostics.compare_all_helpers import (
    delayed_logistic_series,
    delayed_logistic_trajectory,
    logistic_series,
)
from dynachaos.diagnostics.lyapunov import lyapunov_spectrum_blocks
from dynachaos.diagnostics.permutation import permutation_entropy
from dynachaos.diagnostics.recurrence import recurrence_matrix, rqa
from dynachaos.diagnostics.zero_one_test import zero_one_statistic
from dynachaos.io.paths import load_or_compute_npz, section_dir
from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac
from dynachaos.maps.delayed_logistic import delayed_logistic, delayed_logistic_jac
from dynachaos.pipelines.profile import choose

FIG_DIR = section_dir("sec11_diagnostics")
AGREE_NPZ = FIG_DIR / "agreement_scores.npz"
AGREE_PNG = FIG_DIR / "agreement_roc.png"

N_BLOCKS = 20
K01_THRESHOLD = 0.5
SYSTEMS = ("logistic", "delayed", "coupled")
# (key, label, chaotic_high, threshold).  DET and LAM are expected to drop when the
# motion turns chaotic, so a low value is the chaotic direction.
DIAGNOSTICS = (
    ("K01", r"$K_{01}$", True, K01_THRESHOLD),
    ("HPE", r"$H_\mathrm{PE}$", True, None),
    ("DET", "DET", False, None),
    ("LAM", "LAM", False, None),
)


def _grids():
    return {
        "logistic": np.linspace(1.0, 2.0, choose(500, 12)),
        "delayed": np.linspace(1.5, 2.2, choose(300, 12)),
        "coupled": np.linspace(2.1, 2.65, choose(500, 12)),
    }


# ---------------------------------------------------------------------------
# One parameter value: Lyapunov spectrum and diagnostics
# ---------------------------------------------------------------------------


def _coupled_state_series(DB, n_transient, n_record):
    """4D states of the coupled delayed logistic map after a transient."""
    DA = DB + 0.1
    state = np.array([0.5, 0.5, 0.3, 0.3])
    for _ in range(n_transient):
        state = coupled_delayed(state, 0.4, DA, DB, 5e-3)
    traj = np.empty((n_record, 4))
    for i in range(n_record):
        state = coupled_delayed(state, 0.4, DA, DB, 5e-3)
        traj[i] = state
    return traj


def _lyapunov(system, p, n_transient, n_iter):
    if system == "logistic":
        x0 = np.array([0.1])

        def f(s):
            return np.array([1.0 - p * s[0] * s[0]])

        def jac(s):
            return np.array([[-2.0 * p * s[0]]])

    elif system == "delayed":
        fp = (np.sqrt(1.0 + 4.0 * p) - 1.0) / (2.0 * p)
        x0 = np.array([fp + 0.01, fp - 0.01])

        def f(s):
            return delayed_logistic(s, 0.3, p)

        def jac(s):
            return delayed_logistic_jac(s, 0.3, p)

    else:
        x0 = np.array([0.5, 0.5, 0.3, 0.3])

        def f(s):
            return coupled_delayed(s, 0.4, p + 0.1, p, 5e-3)

        def jac(s):
            return coupled_delayed_jac(s, 0.4, p + 0.1, p, 5e-3)

    return lyapunov_spectrum_blocks(f, jac, x0, n_iter, n_transient, N_BLOCKS)


def _series_and_trajectory(system, p):
    """Scalar series (K_01, H_PE) and state trajectory (RQA) of the Section 11 sweeps."""
    if system == "logistic":
        series = logistic_series(p, n_transient=5000, n_record=5000)
        traj = logistic_series(p, n_transient=10_000, n_record=2000)
    elif system == "delayed":
        series = delayed_logistic_series(p, n_transient=5000, n_record=5000)
        traj = delayed_logistic_trajectory(p, A=0.3, n_transient=10_000, n_record=2000)
    else:
        states = _coupled_state_series(p, 20_000, 5000)
        series, traj = states[:, 0], _coupled_state_series(p, 20_000, 2000)
    for arr in (series, traj):
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError(f"non-finite orbit: {system}, parameter {p}")
    return series, traj


def _diagnostics(series, traj):
    R, _ = recurrence_matrix(traj, percentile=5)
    stats = rqa(R, l_min=2, v_min=2)
    return (
        zero_one_statistic(series, n_c=50),
        permutation_entropy(series, d=5),
        stats["DET"],
        stats["LAM"],
    )


def compute_agreement():
    """Lyapunov labels and diagnostic values on the three sweeps."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_transient = {"logistic": 5000, "delayed": 10_000, "coupled": 50_000}
    n_iter = {
        "logistic": choose(50_000, 2000),
        "delayed": choose(50_000, 2000),
        "coupled": choose(100_000, 4000),
    }
    out = {}
    for system, grid in _grids().items():
        n_dim = {"logistic": 1, "delayed": 2, "coupled": 4}[system]
        spectra = np.empty((len(grid), n_dim))
        spectra_se = np.empty_like(spectra)
        values = np.empty((len(grid), len(DIAGNOSTICS)))
        for i, p in enumerate(grid):
            spectra[i], spectra_se[i] = _lyapunov(system, p, n_transient[system], n_iter[system])
            values[i] = _diagnostics(*_series_and_trajectory(system, p))
            if (i + 1) % 100 == 0:
                print(f"  agreement {system}: {i + 1}/{len(grid)}")
        out[f"{system}_param"] = grid
        out[f"{system}_spectrum"] = spectra
        out[f"{system}_spectrum_se"] = spectra_se
        out[f"{system}_label"] = label_sweep(spectra, spectra_se)
        for j, (key, *_rest) in enumerate(DIAGNOSTICS):
            out[f"{system}_{key}"] = values[:, j]
    np.savez_compressed(AGREE_NPZ, **out)
    print(f"Saved {AGREE_NPZ}")


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def score_system(data, system):
    """Scores of every diagnostic on one system, from the cached arrays.

    Returns ``{diagnostic key: dict from scored_agreement}``.  ``agreement`` is
    present for K_01 only.
    """
    labels = data[f"{system}_label"]
    return {
        key: scored_agreement(
            data[f"{system}_{key}"], labels, threshold=threshold, chaotic_high=chaotic_high
        )
        for key, _label, chaotic_high, threshold in DIAGNOSTICS
    }


def misranked_table(data, system, key):
    """Rows (parameter, lambda_1, SE, score, label) for the misranked points of a diagnostic."""
    scores = score_system(data, system)[key]
    rows = []
    for name, idx in (
        ("regular above median chaotic", scores["misranked_regular"]),
        ("chaotic below median regular", scores["misranked_chaotic"]),
    ):
        for i in idx:
            rows.append(
                (
                    name,
                    float(data[f"{system}_param"][i]),
                    float(data[f"{system}_spectrum"][i, 0]),
                    float(data[f"{system}_spectrum_se"][i, 0]),
                    float(data[f"{system}_{key}"][i]),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_agreement(data):
    """ROC curves of the four diagnostics, one panel per system."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        color_for,
        figure_spec,
        finalize_legend,
        marker_for,
        setup,
    )

    setup()
    spec = figure_spec("double")
    fig, axes = plt.subplots(1, 3, figsize=(spec.figsize[0], spec.figsize[1] * 0.95))
    for p, (ax, system) in enumerate(zip(axes, SYSTEMS, strict=True)):
        labels = data[f"{system}_label"]
        scored = labels != MARGINAL
        scores = score_system(data, system)
        ax.plot([0, 1], [0, 1], color=COLORS["grey"], ls="--", lw=0.6, zorder=0)
        for j, (key, name, chaotic_high, _thr) in enumerate(DIAGNOSTICS):
            values = data[f"{system}_{key}"][scored]
            fpr, tpr = roc_curve(values if chaotic_high else -values, labels[scored] == CHAOTIC)
            ax.plot(
                fpr,
                tpr,
                color=color_for(j),
                marker=marker_for(j),
                markersize=3.0,
                markerfacecolor=COLORS["offwhite"],
                markevery=max(len(fpr) // 8, 1),
                lw=1.0 + 0.5 * (len(DIAGNOSTICS) - 1 - j),
                label=f"{name} ({scores[key]['auc']:.3f})",
                zorder=j + 1,
            )
        n_c, n_r = (int(np.sum(labels == v)) for v in (CHAOTIC, REGULAR))
        ax.set_title(f"({chr(97 + p)}) {system}", loc="left")
        ax.set_xlabel("False-positive rate")
        if p == 0:
            ax.set_ylabel("True-positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        apply_axes_polish(ax, kind="double", title_loc="left", grid=False)
        finalize_legend(
            ax,
            kind="double",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=1,
            title=f"AUC; {n_c} chaotic, {n_r} regular",
            title_fontsize=figure_spec("double").legend_size,
        )
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(AGREE_PNG, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {AGREE_PNG}")


KEYS = tuple(
    f"{system}_{suffix}"
    for system in SYSTEMS
    for suffix in ("param", "spectrum", "spectrum_se", "label", *(d[0] for d in DIAGNOSTICS))
)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_compute_npz(
        AGREE_NPZ, "diagnostic agreement", compute_agreement, required_keys=KEYS
    )
    plot_agreement(data)


if __name__ == "__main__":
    main()
