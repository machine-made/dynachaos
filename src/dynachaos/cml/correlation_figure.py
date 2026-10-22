#!/usr/bin/env python3
"""
correlation_figure: Spatial correlation decay and KS entropy density of the logistic CML.

Two-panel figure for Section 8 (thermodynamics of spatiotemporal chaos):
  (a) Spatial autocorrelation |C(r)|/C(0) vs separation r (semilogy)
  (b) Kolmogorov-Sinai entropy density h_KS(L)/L vs subsystem size L, where h_KS(L)
      is the sum of the positive Lyapunov exponents of the Jacobian of an L-site
      subsystem (the L x L sub-block of the full lattice Jacobian)

When the correlation length xi is much smaller than the lattice, the density
h_KS(L)/L is constant in L (extensive, a thermodynamic quantity).  A pattern-selection
state with long-range order has no such plateau.

CML model: logistic f(x) = 1 - a*x^2, g = f, periodic BC, N=200, eps=0.3.
Four regimes:  a=1.5 (frozen random), a=1.7 (pattern selection),
               a=1.85 (defect turbulence), a=1.95 (fully-developed turbulence).

OUTPUTS: figures/sec08_sti/correlation_decay.npz, correlation_decay.png
USAGE:   python src/dynachaos/cml/correlation_figure.py
"""

import numpy as np

from dynachaos.cml.primitives import (
    cml_jacobian_subblock_logistic as _cml_jacobian_subblock,
)
from dynachaos.cml.primitives import (
    cml_step_logistic as cml_step,
)
from dynachaos.io.paths import safe_load, section_dir

FIG_DIR = section_dir("sec08_sti")
CORR_NPZ = FIG_DIR / "correlation_decay.npz"
CORR_PNG = FIG_DIR / "correlation_decay.png"


# ---------------------------------------------------------------------------
# Panel 1: Spatial correlations
# ---------------------------------------------------------------------------


def compute_correlations(
    N=200,
    eps=0.3,
    a_values=(1.50, 1.70, 1.85, 1.95),
    n_transient=5000,
    n_sample=1000,
    sample_interval=10,
    seed=42,
):
    """Compute the spatial autocorrelation C(r) for several a values via FFT.

    C(r) = <x(i) x(i+r)> - <x>^2, averaged over sites and ``n_sample`` snapshots
    taken every ``sample_interval`` steps, and normalised so that C(0) = 1.  One
    generator seeded with ``seed`` draws the initial lattice of every a value in turn.
    """
    rng = np.random.default_rng(seed)
    r_max = N // 2
    all_corr = np.empty((len(a_values), r_max + 1), dtype=np.float64)

    for ia, a in enumerate(a_values):
        print(f"  Spatial correlation: a = {a}...")
        x = rng.uniform(-0.5, 0.5, N)
        for _ in range(n_transient):
            x = cml_step(x, a, eps)

        # Decorrelated snapshots
        snapshots = np.empty((n_sample, N), dtype=np.float64)
        for s in range(n_sample):
            for _ in range(sample_interval):
                x = cml_step(x, a, eps)
            snapshots[s] = x

        # Circular autocorrelation of each fluctuation snapshot via FFT
        fluct = snapshots - snapshots.mean()
        corr_sum = np.zeros(r_max + 1, dtype=np.float64)
        for s in range(n_sample):
            fft_f = np.fft.rfft(fluct[s])
            power = np.real(fft_f * np.conj(fft_f))
            full_corr = np.fft.irfft(power, n=N) / N
            corr_sum += full_corr[: r_max + 1]

        corr_sum /= n_sample
        if abs(corr_sum[0]) > 1e-15:
            all_corr[ia] = corr_sum / corr_sum[0]
        else:
            all_corr[ia] = corr_sum

    return np.array(a_values, dtype=np.float64), np.arange(r_max + 1), all_corr


def fit_correlation_length(r, corr_normalized):
    """Correlation length xi: where the initial decay of |C(r)| first reaches 1/e.

    Scans |C(r)|/C(0) from r = 0 and interpolates linearly between the two sites
    that bracket e^-1.  Only the initial decay is used, so an oscillating
    correlation (pattern selection) gives the length of its first fall, not that of
    its envelope.  Returns NaN when |C| never falls to 1/e.
    """
    abs_corr = np.abs(corr_normalized)
    target = np.exp(-1.0)
    for idx in range(1, len(abs_corr)):
        if abs_corr[idx] <= target:
            x0, x1 = float(r[idx - 1]), float(r[idx])
            y0, y1 = abs_corr[idx - 1], abs_corr[idx]
            if y0 != y1:
                return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return x1
    return float("nan")


# ---------------------------------------------------------------------------
# Panel 2: KS entropy density
# ---------------------------------------------------------------------------


def compute_ks_entropy_density(
    N=200,
    eps=0.3,
    a_values=(1.50, 1.70, 1.85, 1.95),
    L_values=(10, 20, 30, 40, 60, 80, 100),
    n_transient=2000,
    n_iter=2000,
    seed=42,
):
    """Compute the KS entropy density h_KS(L)/L for several subsystem sizes L.

    The full N-site lattice is evolved; the L x L sub-block of its Jacobian
    (sites 0..L-1) is propagated with QR re-orthonormalisation, giving L exponents.
    h_KS(L) is the sum of the positive ones.  The lattice of the a value with
    index ``ia`` starts from ``np.random.default_rng(seed + ia)``.
    """
    h_density = np.empty((len(a_values), len(L_values)), dtype=np.float64)

    for ia, a in enumerate(a_values):
        print(f"  KS entropy density: a = {a}...")
        rng = np.random.default_rng(seed + ia)
        x = rng.uniform(-0.5, 0.5, N)
        for _ in range(n_transient):
            x = cml_step(x, a, eps)

        for iL, L in enumerate(L_values):
            x_run = x.copy()
            Q = np.eye(L, dtype=np.float64)
            exps = np.zeros(L, dtype=np.float64)
            for _ in range(n_iter):
                x_run = cml_step(x_run, a, eps)
                J = _cml_jacobian_subblock(x_run, a, eps, L)
                Q, R = np.linalg.qr(J @ Q)
                exps += np.log(np.abs(np.diag(R)))
            exps /= n_iter
            h_density[ia, iL] = np.sum(exps[exps > 0]) / L

    return np.array(a_values, dtype=np.float64), np.array(L_values, dtype=np.int64), h_density


# ---------------------------------------------------------------------------
# Combined computation
# ---------------------------------------------------------------------------


def compute():
    """Run both computations and save to a single .npz."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing spatial correlations...")
    a_corr, r_vals, all_corr = compute_correlations()
    xi_values = np.array(
        [fit_correlation_length(r_vals, all_corr[ia]) for ia in range(len(a_corr))]
    )

    print("Computing KS entropy density...")
    a_lyap, L_vals, h_density = compute_ks_entropy_density()

    np.savez_compressed(
        CORR_NPZ,
        a_corr=a_corr,
        r_vals=r_vals,
        all_corr=all_corr,
        xi_values=xi_values,
        a_lyap=a_lyap,
        L_vals=L_vals,
        h_density=h_density,
    )
    print(f"Saved {CORR_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(data):
    """Two-panel figure: correlation decay + KS entropy density."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        apply_axes_polish,
        figure_spec,
        series_style,
        setup,
    )

    setup()

    a_corr = data["a_corr"]
    r_vals = data["r_vals"]
    all_corr = data["all_corr"]
    xi_values = data["xi_values"]
    a_lyap = data["a_lyap"]
    L_vals = data["L_vals"]
    h_density = data["h_density"]

    labels = [
        r"$a=1.50$",
        r"$a=1.70$",
        r"$a=1.85$",
        r"$a=1.95$",
    ]

    spec = figure_spec("double")
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(spec.figsize[0], spec.figsize[1] + 0.25),
        gridspec_kw={"width_ratios": (1.25, 0.95)},
    )

    # --- Panel (a): Spatial correlation decay ---
    for ia in range(len(a_corr)):
        sty = series_style(ia)
        sty.pop("marker", None)
        sty.pop("markersize", None)
        sty.pop("markerfacecolor", None)
        sty.pop("markeredgewidth", None)
        abs_corr = np.abs(all_corr[ia])
        mask = abs_corr > 0
        r_plot = r_vals[mask]
        c_plot = abs_corr[mask]
        xi = xi_values[ia]
        label = labels[ia]
        if np.isfinite(xi):
            label += rf", $\xi \approx {xi:.1f}$"
        ax1.semilogy(r_plot, c_plot, label=label, **sty)

        if np.isfinite(xi) and xi > 0:
            r_fit_max = min(3.0 * xi, float(r_plot.max()) if r_plot.size else xi)
            r_fit = np.linspace(0.0, r_fit_max, 100)
            ax1.semilogy(
                r_fit,
                np.exp(-r_fit / xi),
                color=sty["color"],
                linestyle=":",
                linewidth=1.1,
                alpha=0.8,
                zorder=sty.get("zorder", 2) + 0.5,
            )

    ax1.set_xlabel(r"Separation $r$")
    ax1.set_ylabel(r"$|C(r)|/C(0)$")
    ax1.set_title(r"(a) Spatial correlation decay", loc="left")
    apply_axes_polish(ax1, kind="double", title_loc="left", grid=False)
    handles, labels_out = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=spec.legend_size,
        bbox_to_anchor=(0.5, 1.01),
    )

    # --- Panel (b): KS entropy density extensivity ---
    for ia in range(len(a_lyap)):
        sty = series_style(ia)
        ax2.plot(L_vals, h_density[ia], markevery=1, **sty)

    ax2.set_xlabel(r"Subsystem size $L$")
    ax2.set_ylabel(r"$h_{\mathrm{KS}}(L)/L$")
    ax2.set_title(r"(b) KS entropy density $h_{\mathrm{KS}}(L)/L$", loc="left")
    ax2.set_ylim(0.0, 1.25 * float(np.max(h_density)))
    apply_axes_polish(ax2, kind="double", title_loc="left", grid=False)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(CORR_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {CORR_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = safe_load(CORR_NPZ)
        print(f"Loaded {CORR_NPZ}")
    except FileNotFoundError:
        print("Computing correlation and Lyapunov density data...")
        compute()
        data = safe_load(CORR_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
