#!/usr/bin/env python3
"""
comoving_figure: Windowed co-moving Lyapunov exponent for the coupled logistic CML.

Computes lambda(v) -- the co-moving Lyapunov exponent -- for a coupled
map lattice with logistic local dynamics f(x) = 1 - a*x^2 at three
values of the nonlinearity parameter a spanning pattern selection,
defect turbulence, and fully developed turbulence.  The definition is the
windowed one of Kaneko (1986) and Deissler & Kaneko (1987): the growth rate
of the perturbation norm summed over a window of 2 w + 1 sites that moves
with velocity v, renormalised at every step.

The lattice is large (N = 3000) and the averaging time long (T = 1000) so that
the moving window never wraps around the periodic domain (largest site index
500 + 1000 + 15 = 1515 < 3000).  v is sampled densely on [0, 1] (51 points) to
resolve the zero crossings v_p of lambda(v), which mark the propagation
velocity of information in the spatiotemporal chaotic state.

CML model:
    x_{n+1}(i) = f(x_n(i)) + (eps/2)[f(x_n(i+1)) + f(x_n(i-1)) - 2 f(x_n(i))]

Parameters: N=3000, T=1000, w=15, eps=0.3

OUTPUTS: figures/sec08_sti/comoving_lyapunov.npz,
         figures/sec08_sti/comoving_lyapunov.png
USAGE:   python src/dynachaos/cml/comoving_figure.py
"""

import numpy as np

from dynachaos.io.paths import safe_load, section_dir

FIG_DIR = section_dir("sec08_sti")
OUTPUT_NPZ = FIG_DIR / "comoving_lyapunov.npz"
OUTPUT_PNG = FIG_DIR / "comoving_lyapunov.png"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute(
    a_values=(1.70, 1.85, 1.95),
    eps=0.3,
    N=3000,
    T=1000,
    n_transient=2000,
    w=15,
    i0=500,
    n_v=51,
    seed=42,
):
    """Compute the windowed co-moving Lyapunov exponent for each a and save the cache.

    Each a value gets its own random initial lattice, drawn with seed ``seed + index``.
    Besides ``lambda_a<a>`` the cache holds the zero crossings ``zc_a<a>`` of lambda(v).
    """
    from dynachaos.diagnostics.comoving_lyapunov import (
        comoving_lyapunov_windowed_logistic,
        comoving_zero_crossings,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    v_values = np.linspace(0.0, 1.0, n_v)
    payload = {
        "v_values": v_values,
        "a_values": np.array(a_values, dtype=np.float64),
        "eps": np.array([eps], dtype=np.float64),
        "N": np.array([N], dtype=np.int64),
        "T": np.array([T], dtype=np.int64),
        "w": np.array([w], dtype=np.int64),
        "i0": np.array([i0], dtype=np.int64),
    }
    for idx, a in enumerate(a_values):
        print(f"Computing co-moving spectrum for a = {a}...")
        lam = comoving_lyapunov_windowed_logistic(
            a=a,
            eps=eps,
            N=N,
            T=T,
            v_values=v_values,
            n_transient=n_transient,
            w=w,
            i0=i0,
            seed=seed + idx,
        )
        zero_crossings = comoving_zero_crossings(v_values, lam)
        print(f"  a = {a}: zero crossings at {zero_crossings}")
        payload[f"lambda_a{a:.2f}"] = lam
        payload[f"zc_a{a:.2f}"] = np.array(zero_crossings, dtype=np.float64)

    np.savez_compressed(OUTPUT_NPZ, **payload)
    print(f"Saved {OUTPUT_NPZ}")
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(data):
    """Plot lambda(v) vs v for each nonlinearity parameter, with the zero crossings marked."""
    import matplotlib.pyplot as plt

    from dynachaos.diagnostics.comoving_lyapunov import comoving_zero_crossings
    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        reference_line,
        setup,
    )

    setup()

    v = data["v_values"]
    a_values = data["a_values"]
    eps = float(data["eps"][0])
    N = int(data["N"][0])

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)

    colors = {1.70: COLORS["blue"], 1.85: COLORS["black"], 1.95: COLORS["red"]}
    labels = {
        1.70: r"$a = 1.70$ (pattern selection)",
        1.85: r"$a = 1.85$ (defect turbulence)",
        1.95: r"$a = 1.95$ (fully developed turbulence)",
    }

    reference_line(ax, 0.0, axis="y", lw=0.6, alpha=0.7)

    for a in a_values:
        a = float(a)
        lam = np.asarray(data[f"lambda_a{a:.2f}"], dtype=float)
        col = colors.get(a, COLORS["black"])
        ax.plot(v, lam, color=col, lw=1.1, marker="o", ms=2.5, label=labels.get(a, rf"$a = {a}$"))

        # Zero crossings with v > 0.2 are the primary propagation speed v_p.
        for zc in comoving_zero_crossings(v, lam):
            if zc > 0.2:
                reference_line(ax, zc, axis="x", lw=0.7, ls="--", color=col)
                ax.text(
                    zc + 0.015,
                    0.20 if a == 1.95 else 0.12,
                    rf"$v_p \approx {zc:.3f}$",
                    color=col,
                    fontsize=spec.tick_size - 1,
                    ha="left",
                )

    ax.set_xlabel(r"Velocity $v$ (lattice sites per iteration)")
    ax.set_ylabel(r"Co-moving Lyapunov exponent $\lambda(v)$")
    ax.set_title(
        rf"Windowed co-moving Lyapunov exponent ($\varepsilon = {eps:.1f}$, $N = {N}$)",
        loc="left",
    )
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(-0.35, 0.45)
    apply_axes_polish(ax, kind="double", title_loc="left")
    finalize_legend(ax, kind="single", loc="lower left")

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
        print("Computing co-moving Lyapunov exponent...")
        compute()
        data = safe_load(OUTPUT_NPZ)
    plot(data)


if __name__ == "__main__":
    main()
