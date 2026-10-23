#!/usr/bin/env python3
"""
alignment_three_torus: SALI and GALI on torus, locked and chaotic orbits (Section 11).

Tests the alignment indices on orbits of the coupled delayed logistic map whose
Lyapunov spectrum is known from the same run.  For deviation vectors that align
with the leading direction, ln GALI_k grows at the rate

    s_k = sum_{i=2..k} (lambda_i - lambda_1).

An attracting 2-torus (lambda_1 = lambda_2 = 0) gives s_2 = 0: SALI and GALI_2
stay at a positive value.  GALI_3 decays at lambda_3 and GALI_4 at
lambda_3 + lambda_4.  A partially locked orbit (lambda_1 = 0 > lambda_2) gives
s_2 = lambda_2, and a chaotic orbit gives s_2 = -(lambda_1 - lambda_2).

The indices come from :func:`dynachaos.diagnostics.sali_gali.alignment_indices`,
which keeps the log of the volume and so resolves values far below 1e-16.

OUTPUTS: figures/sec11_diagnostics/alignment_three_torus.npz, *.png
USAGE:   python src/dynachaos/diagnostics/alignment_three_torus.py
"""

import numpy as np

from dynachaos.diagnostics.lyapunov import lyapunov_spectrum_blocks
from dynachaos.diagnostics.sali_gali import alignment_indices, log10_sali_from_gali2
from dynachaos.io.paths import load_or_compute_npz, section_dir
from dynachaos.maps.coupled_delayed import (
    coupled_delayed as _coupled_delayed_map,
)
from dynachaos.maps.coupled_delayed import (
    coupled_delayed_jac as _coupled_delayed_jac,
)
from dynachaos.pipelines.profile import choose

FIG_DIR = section_dir("sec11_diagnostics")
ALIGN_NPZ = FIG_DIR / "alignment_three_torus.npz"
ALIGN_PNG = FIG_DIR / "alignment_three_torus.png"

ALPHA = 0.4
EPS = 5e-3
X0 = np.array([0.5, 0.5, 0.3, 0.3])
N_TRANSIENT = choose(100_000, 2_000)
N_ITER = choose(1_000_000, 20_000)
N_BLOCKS = 20
SEEDS = (0, 1, 2)
SAMPLES_PER_DECADE = 40
K_MAX = 4

# (D_B, regime).  Chosen from a Lyapunov scan of D_B at 1e5 transient and 2e5 iterates.
# ``check_regime`` stops the run if the 1e6-iterate spectrum of an orbit contradicts its label.
ORBITS = (
    (2.3811, "torus"),
    (2.3846, "torus"),
    (2.37, "partial locking"),
    (2.55, "chaos"),
)


def check_regime(spectrum, se, regime, n_sigma=2.0):
    """Raise ValueError if a spectrum does not match the stated regime."""
    lam, band = np.asarray(spectrum), n_sigma * np.asarray(se)
    zero = np.abs(lam) <= band
    if regime == "torus":
        ok = zero[0] and zero[1] and lam[2] < -band[2]
    elif regime == "partial locking":
        ok = zero[0] and lam[1] < -band[1]
    elif regime == "chaos":
        ok = lam[0] > band[0]
    else:
        raise ValueError(f"unknown regime '{regime}'")
    if not ok:
        raise ValueError(f"spectrum {lam} (SE {np.asarray(se)}) does not match regime '{regime}'")


def _sample_indices(n_iter):
    """Zero-based iteration indices, log-spaced with SAMPLES_PER_DECADE points per decade."""
    n_dec = np.log10(n_iter)
    ns = np.unique(np.round(np.logspace(0.0, n_dec, int(n_dec * SAMPLES_PER_DECADE) + 1)))
    return ns.astype(np.int64) - 1


def _window_slope(log10_gali, n1, n2):
    """Least-squares slope of ln GALI per iteration over iterations n1 < n <= n2."""
    n = np.arange(n1, n2, dtype=np.float64)[::10]
    y = log10_gali[n1:n2:10] * np.log(10.0)
    return np.polyfit(n, y, 1)[0]


def predicted_rates(spectrum, se):
    """Predicted ln GALI_k rate and its standard error for k = 2..K_MAX.

    The standard error assumes independent errors on the exponents.
    """
    lam, err = np.asarray(spectrum), np.asarray(se)
    rate = np.array([np.sum(lam[1:k] - lam[0]) for k in range(2, K_MAX + 1)])
    sigma = np.array(
        [np.sqrt((k - 1) ** 2 * err[0] ** 2 + np.sum(err[1:k] ** 2)) for k in range(2, K_MAX + 1)]
    )
    return rate, sigma


def compute_alignment(orbits=None, n_transient=None, n_iter=None, seeds=SEEDS):
    """SALI and GALI_2..4 along each orbit, with the Lyapunov spectrum of the same run."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    orbits = ORBITS if orbits is None else orbits
    n_transient = N_TRANSIENT if n_transient is None else n_transient
    n_iter = N_ITER if n_iter is None else n_iter
    idx = _sample_indices(n_iter)
    n_dec = int(np.floor(np.log10(n_iter)))

    out = {
        "DB_values": np.array([o[0] for o in orbits]),
        "regimes": np.array([o[1] for o in orbits]),
        "seeds": np.array(seeds),
        "n_samples": idx + 1,
        "n_transient": np.array([n_transient]),
        "n_iter": np.array([n_iter]),
    }
    spectra, spectra_se = [], []
    log_g, dec_min, dec_max, fit, fit_prev, pred, pred_se = [], [], [], [], [], [], []

    for DB, regime in orbits:
        DA = DB + 0.1

        def f(s, _DA=DA, _DB=DB):
            return _coupled_delayed_map(s, ALPHA, _DA, _DB, EPS)

        def jac(s, _DA=DA, _DB=DB):
            return _coupled_delayed_jac(s, ALPHA, _DA, _DB, EPS)

        print(f"  alignment: D_B={DB} ({regime})")
        spec, se = lyapunov_spectrum_blocks(
            f, jac, X0, n_iter=n_iter, n_transient=n_transient, n_blocks=N_BLOCKS
        )
        check_regime(spec, se, regime)
        spectra.append(spec)
        spectra_se.append(se)
        rate, sigma = predicted_rates(spec, se)
        pred.append(rate)
        pred_se.append(sigma)

        per_seed = []
        for seed in seeds:
            full = alignment_indices(
                f,
                jac,
                X0,
                k=K_MAX,
                n_iter=n_iter,
                n_transient=n_transient,
                rng=np.random.default_rng(seed),
            )[:, 1:]  # columns: log10 GALI_2, GALI_3, GALI_4
            per_seed.append(
                (
                    full[idx],
                    np.array([full[10**j : 10 ** (j + 1)].min(axis=0) for j in range(n_dec)]),
                    np.array([full[10**j : 10 ** (j + 1)].max(axis=0) for j in range(n_dec)]),
                    [_window_slope(full[:, c], n_iter // 10, n_iter) for c in range(3)],
                    [_window_slope(full[:, c], n_iter // 100, n_iter // 10) for c in range(3)],
                )
            )
        log_g.append([p[0] for p in per_seed])
        dec_min.append([p[1] for p in per_seed])
        dec_max.append([p[2] for p in per_seed])
        fit.append([p[3] for p in per_seed])
        fit_prev.append([p[4] for p in per_seed])

    out["spectrum"] = np.array(spectra)
    out["spectrum_se"] = np.array(spectra_se)
    out["log10_gali"] = np.array(log_g)  # (orbit, seed, sample, k = 2..4)
    out["log10_sali"] = log10_sali_from_gali2(out["log10_gali"][..., 0])
    out["decade_min"] = np.array(dec_min)  # (orbit, seed, decade, k)
    out["decade_max"] = np.array(dec_max)
    out["rate_fit"] = np.array(fit)  # (orbit, seed, k): iterates n_iter/10 to n_iter
    out["rate_fit_early"] = np.array(fit_prev)  # (orbit, seed, k): n_iter/100 to n_iter/10
    out["rate_pred"] = np.array(pred)  # (orbit, k)
    out["rate_pred_se"] = np.array(pred_se)
    np.savez_compressed(ALIGN_NPZ, **out)
    print(f"Saved {ALIGN_NPZ}")


def _symlog_tick_label(value, _pos):
    """Plain number for |value| < 1000, signed power of ten above."""
    if abs(value) < 1000:
        return f"{value:g}"
    sign = "-" if value < 0 else ""
    return rf"${sign}10^{{{int(round(np.log10(abs(value))))}}}$"


def plot_alignment(data):
    """Plot log10 of SALI and GALI_2..4 against the iteration number."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        color_for,
        figure_spec,
        marker_for,
        setup,
    )

    setup()
    n = data["n_samples"]
    regimes = [str(r) for r in data["regimes"]]
    DB = data["DB_values"]
    series = (
        ("SALI", data["log10_sali"][:, 0]),
        ("GALI$_2$", data["log10_gali"][:, 0, :, 0]),
        ("GALI$_3$", data["log10_gali"][:, 0, :, 1]),
        ("GALI$_4$", data["log10_gali"][:, 0, :, 2]),
    )
    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 2, figsize=spec.figsize, sharex=True)
    for p, (ax, (name, values)) in enumerate(zip(axes.flat, series, strict=True)):
        for i in range(len(DB)):
            ax.plot(
                n,
                values[i],
                color=color_for(i),
                marker=marker_for(i),
                markersize=3.0,
                markerfacecolor=COLORS["offwhite"],
                markevery=8,
                lw=1.0 + 0.6 * (len(DB) - 1 - i),
                label=rf"$D_B={DB[i]:g}$, {regimes[i]}",
                zorder=i + 1,
            )
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)  # linear for |log10 index| < 1
        ax.set_ylim(top=1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(_symlog_tick_label))
        ax.set_title(f"({chr(97 + p)}) {name}", loc="left")
        if p >= 2:
            ax.set_xlabel("Iteration $n$")
        ax.set_ylabel(rf"$\log_{{10}}$ {name}")
        apply_axes_polish(ax, kind="grid", title_loc="left", grid=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=figure_spec("grid").legend_size,
    )
    fig.subplots_adjust(wspace=0.28, hspace=0.3)
    fig.savefig(ALIGN_PNG, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ALIGN_PNG}")


KEYS = (
    "DB_values",
    "regimes",
    "n_samples",
    "spectrum",
    "spectrum_se",
    "log10_gali",
    "log10_sali",
    "decade_min",
    "decade_max",
    "rate_fit",
    "rate_fit_early",
    "rate_pred",
    "rate_pred_se",
)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_compute_npz(
        ALIGN_NPZ, "alignment indices", compute_alignment, required_keys=KEYS
    )
    plot_alignment(data)


if __name__ == "__main__":
    main()
