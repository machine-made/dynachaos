#!/usr/bin/env python3
"""
globally_coupled: Globally coupled map violating the law of large numbers.

Reproduces Kaneko (1990) PRL 65(12), 1391-1394.

GCM model (Eq. 1):
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + eps/N sum_j f(x_n(j))

with f(x) = 1 - a x^2 (logistic map).

Key result: MSD of mean field h_n stops decreasing with N.  The MSD figure sweeps five
values of a and eight system sizes with eight random seeds per point, and fits
MSD(N) = A/N + B for each a: a saturation B > 0 means the law of large numbers fails.

OUTPUTS: figures/sec10_gcm/gcm_results.npz, gcm_distribution.png,
         figures/sec10_gcm/gcm_msd.npz, gcm_msd.png
"""

import numpy as np

from dynachaos.cml.primitives import gcm_step
from dynachaos.io.paths import load_or_compute_npz, safe_load, section_dir
from dynachaos.maps.primitives import logistic

FIG_DIR = section_dir("sec10_gcm")

GCM_NPZ = FIG_DIR / "gcm_results.npz"
MSD_NPZ = FIG_DIR / "gcm_msd.npz"
GCM_PNG = FIG_DIR / "gcm_msd.png"
DIST_PNG = FIG_DIR / "gcm_distribution.png"
MSD_REQUIRED_KEYS = (
    "a_values",
    "N_values",
    "eps",
    "n_seeds",
    "msd_all",
    "msd_mean",
    "msd_std",
    "msd_stderr",
    "fits_A",
    "fits_B",
    "fits_B_stderr",
    "fits_r2",
)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute():
    """Sample the mean field h_n for several N (distribution P(h) and its variance)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    a = 1.99
    eps = 0.1
    N_values = [100, 400, 1000, 5000, 20_000]
    n_transient = 10_000
    n_sample = 100_000

    results = {}
    rng = np.random.default_rng(42)

    for N in N_values:
        print(f"  N={N}")
        x = rng.uniform(-1, 1, N)

        for _ in range(n_transient):
            x = gcm_step(x, a, eps)

        h_series = np.empty(n_sample)
        for t in range(n_sample):
            x = gcm_step(x, a, eps)
            fx = logistic(x, a)
            h_series[t] = np.mean(fx)

        results[f"N_{N}_h"] = h_series
        results[f"N_{N}_mean"] = np.array([np.mean(h_series)])
        results[f"N_{N}_msd"] = np.array([np.var(h_series)])

    results["N_values"] = np.array(N_values)
    results["a"] = np.array([a])
    results["eps"] = np.array([eps])

    np.savez_compressed(GCM_NPZ, **results)
    print(f"Saved {GCM_NPZ}")


def compute_msd_sweep(
    a_values=(1.80, 1.85, 1.92, 1.95, 1.99),
    N_values=(100, 200, 500, 1000, 2000, 5000, 10_000, 20_000),
    eps=0.1,
    n_seeds=8,
    n_transient=2000,
    n_sample=10_000,
    seed_base=42,
    output_path=MSD_NPZ,
):
    """MSD of the mean field for every (a, N) over ``n_seeds`` seeds, with the fit MSD(N) = A/N + B.

    The mean field is h_n = mean(f(x_n)).  Each (a, N, seed) run starts from
    ``np.random.default_rng(seed_base + 10000 * i_a + 100 * i_N + seed)``, discards
    ``n_transient`` steps and takes the variance of ``n_sample`` values of h_n.  The
    fit is an ordinary least-squares line of the seed-averaged MSD against 1/N, so
    the intercept B is the saturation value and ``fits_B_stderr`` its standard error.
    """
    from scipy.stats import linregress

    n_a = len(a_values)
    n_N = len(N_values)
    msd_all = np.empty((n_a, n_N, n_seeds), dtype=np.float64)

    for ia, a in enumerate(a_values):
        print(f"  GCM MSD for a = {a} across {n_N} sizes, {n_seeds} seeds...")
        for jn, N in enumerate(N_values):
            for s in range(n_seeds):
                rng = np.random.default_rng(seed_base + 10000 * ia + 100 * jn + s)
                x = rng.uniform(-1, 1, N)
                for _ in range(n_transient):
                    x = gcm_step(x, a, eps)
                h_vals = np.empty(n_sample, dtype=np.float64)
                for t in range(n_sample):
                    x = gcm_step(x, a, eps)
                    h_vals[t] = np.mean(logistic(x, a))
                msd_all[ia, jn, s] = np.var(h_vals)

    msd_mean = np.mean(msd_all, axis=2)
    msd_std = np.std(msd_all, axis=2, ddof=1)
    msd_stderr = msd_std / np.sqrt(n_seeds)

    inv_N = 1.0 / np.array(N_values, dtype=np.float64)
    fits_A = np.empty(n_a, dtype=np.float64)
    fits_B = np.empty(n_a, dtype=np.float64)
    fits_B_stderr = np.empty(n_a, dtype=np.float64)
    fits_r2 = np.empty(n_a, dtype=np.float64)
    print("  Saturation fits MSD(N) = A/N + B:")
    for ia, a in enumerate(a_values):
        res = linregress(inv_N, msd_mean[ia])
        fits_A[ia] = res.slope
        fits_B[ia] = res.intercept
        fits_B_stderr[ia] = res.intercept_stderr
        fits_r2[ia] = res.rvalue**2
        print(
            f"    a = {a:.2f}: B = {res.intercept:.4e} +/- {res.intercept_stderr:.2e}, "
            f"A = {res.slope:.4f}, R^2 = {res.rvalue**2:.4f}"
        )

    payload = {
        "a_values": np.array(a_values, dtype=np.float64),
        "N_values": np.array(N_values, dtype=np.int64),
        "eps": np.array([eps], dtype=np.float64),
        "n_seeds": np.array([n_seeds], dtype=np.int64),
        "msd_all": msd_all,
        "msd_mean": msd_mean,
        "msd_std": msd_std,
        "msd_stderr": msd_stderr,
        "fits_A": fits_A,
        "fits_B": fits_B,
        "fits_B_stderr": fits_B_stderr,
        "fits_r2": fits_r2,
    }
    if output_path is not None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_msd(data):
    """Plot MSD vs N: seed-averaged points with seed spread, and the fit A/N + B."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        CMAP_SEQUENTIAL,
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        series_style,
        setup,
    )

    setup()

    N_values = data["N_values"]
    a_values = data["a_values"]
    msd_mean = data["msd_mean"]
    msd_std = data["msd_std"]
    fits_A = data["fits_A"]
    fits_B = data["fits_B"]
    fits_B_stderr = data["fits_B_stderr"]
    eps = float(data["eps"][0])
    n_seeds = int(data["n_seeds"][0])

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)

    # a is an ordered parameter; a sequential colormap lets the reader read
    # the ordering of the five curves directly instead of decoding an
    # unordered qualitative palette against the legend.
    cmap = plt.get_cmap(CMAP_SEQUENTIAL)
    n_a = len(a_values)

    for ia, a_val in enumerate(a_values):
        sty = series_style(ia)
        color = cmap(ia / max(n_a - 1, 1))
        B_val = fits_B[ia]
        B_err = fits_B_stderr[ia]
        label = (
            rf"$a = {a_val:.2f}$ ($B = {B_val * 1e3:.2f} \pm {B_err * 1e3:.2f} \times 10^{{-3}}$)"
        )
        ax.errorbar(
            N_values,
            msd_mean[ia],
            yerr=msd_std[ia],
            label=label,
            color=color,
            marker=sty["marker"],
            markersize=3.5,
            markerfacecolor=COLORS["offwhite"],
            markeredgewidth=0.8,
            linewidth=1.2,
            capsize=2.0,
            capthick=0.8,
        )
        # Fitted saturation line A/N + B
        N_dense = np.geomspace(N_values[0], N_values[-1], 100)
        ax.plot(
            N_dense, fits_A[ia] / N_dense + B_val, color=color, linestyle=":", lw=0.7, alpha=0.7
        )

    # Reference 1/N decay anchored at the smallest N, a = 1.99
    ref_idx = int(np.argmin(np.abs(a_values - 1.99)))
    N_ref = np.array([N_values[0], N_values[-1]], dtype=np.float64)
    ref_anchor = msd_mean[ref_idx, 0] * N_values[0]
    ax.loglog(
        N_ref,
        ref_anchor / N_ref,
        color=COLORS["grey"],
        linestyle="--",
        lw=0.8,
        label=r"$\propto 1/N$ decay (no saturation)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"System size $N$")
    ax.set_ylabel(r"MSD $\langle (\delta h)^2 \rangle$")
    ax.set_title(
        rf"GCM mean-field fluctuations ($\varepsilon = {eps:.1f}$, {n_seeds} seeds)", loc="left"
    )
    # The 1/N guide falling away below ~5e-4 buys nothing but empty panel, so the
    # floor is set there; the ceiling follows the data.
    ax.set_ylim(5e-4, ax.get_ylim()[1])
    apply_axes_polish(ax, kind="double", title_loc="left", grid=False)
    finalize_legend(
        ax,
        kind="double",
        loc="lower left",
        ncol=2,
        columnspacing=0.9,
        fontsize=spec.legend_size - 0.5,
    )

    fig.savefig(GCM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {GCM_PNG}")


def plot_distribution(data):
    """Plot distribution P(h) for different N."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLOR_CYCLE,
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        setup,
    )

    setup()

    N_values = data["N_values"]

    spec = figure_spec("single")
    fig, ax = plt.subplots(figsize=spec.figsize)
    for idx, N in enumerate(N_values):
        h = data[f"N_{N}_h"]
        ax.hist(
            h,
            bins=100,
            density=True,
            alpha=0.9,
            color=COLOR_CYCLE[idx],
            label=f"$N = {N}$",
            histtype="step",
            lw=1.0,
        )

    ax.set_xlabel(r"Mean field $h$")
    ax.set_ylabel(r"$P(h)$")
    ax.set_title(rf"$a = {data['a'][0]}$, $\varepsilon = {data['eps'][0]}$", loc="left")
    # The histograms are all confined to a narrow band; the generous default
    # x-range is mostly empty and leaves less room for the inset.
    all_h = np.concatenate([data[f"N_{N}_h"] for N in N_values])
    ax.set_xlim(max(float(all_h.min()) - 0.02, 0.1), min(float(all_h.max()) + 0.02, 0.55))
    apply_axes_polish(ax, kind="single", title_loc="left", grid=False)
    finalize_legend(ax, kind="single", loc="upper left")

    inset = ax.inset_axes([0.16, 0.12, 0.30, 0.24])
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    inset.set_zorder(5)
    msd_values = np.array([float(data[f"N_{N}_msd"][0]) for N in N_values])
    for idx, (N, msd) in enumerate(zip(N_values, msd_values, strict=False)):
        inset.loglog(N, msd, color=COLOR_CYCLE[idx], marker="o", ms=3.2, ls="none")
    inset.loglog(N_values, msd_values, color=COLORS["grey"], lw=0.8, zorder=1)
    # 1/N guide anchored at the first point, so the reader can judge how
    # much the LLN would have demanded the variance to fall by N=20000.
    n_ref = np.array([float(N_values[0]), float(N_values[-1])])
    inset.loglog(
        n_ref,
        msd_values[0] * N_values[0] / n_ref,
        color=COLORS["grey"],
        lw=0.8,
        ls="--",
        zorder=1,
    )
    inset.set_title(r"Var$(h)$", fontsize=spec.legend_size, pad=2.0)
    inset.set_xlabel("$N$", fontsize=spec.legend_size - 1.0, labelpad=1.0)
    inset.tick_params(axis="both", which="both", labelsize=spec.legend_size - 1.0, pad=1.5)
    inset.grid(False)

    fig.savefig(DIST_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {DIST_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(GCM_NPZ)
        print(f"Loaded {GCM_NPZ}")
    except FileNotFoundError:
        print("Computing GCM results...")
        compute()
        data = safe_load(GCM_NPZ)
    plot_distribution(data)

    msd_data = load_or_compute_npz(
        MSD_NPZ,
        "GCM MSD sweep",
        compute_msd_sweep,
        required_keys=MSD_REQUIRED_KEYS,
    )
    plot_msd(msd_data)


if __name__ == "__main__":
    main()
