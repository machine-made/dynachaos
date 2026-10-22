#!/usr/bin/env python3
"""
comoving_figure: Co-moving Lyapunov exponent for coupled logistic CML.

Computes lambda(v) — the co-moving Lyapunov exponent — for a coupled
map lattice with logistic local dynamics f(x) = 1 - a*x^2 at three
values of the nonlinearity parameter a spanning pattern selection,
defect turbulence, and fully developed turbulence.

Zero crossings of lambda(v) identify the propagation velocities of
information in the spatiotemporal chaotic state.

CML model:
    x_{n+1}(i) = f(x_n(i)) + (eps/2)[f(x_n(i+1)) + f(x_n(i-1)) - 2 f(x_n(i))]

Parameters: N=500, eps=0.3

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


def compute():
    """Compute co-moving Lyapunov exponent for three values of a."""
    from dynachaos.diagnostics.comoving_lyapunov import comoving_lyapunov_spectrum_logistic

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    N = 500
    eps = 0.3
    v_values = np.linspace(-1.5, 1.5, 301)
    n_iter = 100_000
    n_transient = 20_000

    a_values = [1.70, 1.85, 1.95]
    a_labels = ["pattern selection", "defect turbulence", "fully developed turbulence"]

    all_lambda = {}
    for ia, (a, label) in enumerate(zip(a_values, a_labels)):
        print(f"a={a} ({label})...")
        lam_v = comoving_lyapunov_spectrum_logistic(
            a=a,
            eps=eps,
            N=N,
            v_values=v_values,
            n_iter=n_iter,
            n_transient=n_transient,
        )
        all_lambda[f"lambda_a{a:.2f}"] = lam_v

        # Save iteratively after each a value
        np.savez_compressed(
            OUTPUT_NPZ,
            v_values=v_values,
            a_values=np.array(a_values[: ia + 1]),
            eps=np.array([eps]),
            N=np.array([N]),
            **{k: v for k, v in all_lambda.items()},
        )
        print(f"  Saved checkpoint ({ia + 1}/{len(a_values)})")

    print(f"Saved {OUTPUT_NPZ}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(data):
    """Plot lambda(v) vs v for each nonlinearity parameter."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        reference_line,
        series_style,
        setup,
    )

    setup()

    v_values = data["v_values"]
    a_values = data["a_values"]

    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)

    a_labels = {
        1.70: "pattern selection",
        1.85: "defect turbulence",
        1.95: "fully developed turbulence",
    }

    y_min = np.inf
    y_max = -np.inf
    crossings = []
    for i, a in enumerate(a_values):
        key = f"lambda_a{a:.2f}"
        lam_v = np.array(data[key], dtype=float)
        valid = lam_v > -9.5
        lam_plot = lam_v.copy()
        lam_plot[~valid] = np.nan
        sty = series_style(i)
        sty.pop("marker", None)
        sty.pop("markersize", None)
        sty.pop("markerfacecolor", None)
        sty.pop("markeredgewidth", None)
        label_str = a_labels.get(float(a), "")
        ax.plot(v_values, lam_plot, label=rf"$a = {a:.2f}$ ({label_str})", **sty)
        if np.any(valid):
            y_min = min(y_min, np.nanmin(lam_plot[valid]))
            y_max = max(y_max, np.nanmax(lam_plot[valid]))

        for j in range(len(lam_v) - 1):
            if not (valid[j] and valid[j + 1]):
                continue
            if lam_v[j] * lam_v[j + 1] < 0:
                v_cross = (v_values[j] * abs(lam_v[j + 1]) + v_values[j + 1] * abs(lam_v[j])) / (
                    abs(lam_v[j]) + abs(lam_v[j + 1])
                )
                ax.axvline(v_cross, color=sty["color"], lw=1.4, ls="--", alpha=0.9, zorder=4)
                if v_cross > 0:
                    crossings.append((a, sty["color"], v_cross))

    for k, (a, color, v_cross) in enumerate(sorted(crossings, key=lambda c: -c[2])):
        ax.text(
            0.98,
            0.97 - 0.06 * k,
            rf"$v_p(a={a:.2f}) \approx {v_cross:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize="small",
            color=color,
        )

    reference_line(ax, 0, axis="y", label=None)
    ax.set_xlabel(r"Velocity $v$ (sites/iteration)")
    ax.set_ylabel(r"$\lambda(v)$")
    if np.isfinite(y_min) and np.isfinite(y_max):
        ax.set_ylim(y_min - 0.15, y_max + 0.08)
    ax.set_title(r"Co-moving Lyapunov exponent, logistic CML", loc="left")

    apply_axes_polish(ax, kind="double", title_loc="left", grid=False)
    finalize_legend(
        ax,
        kind="double",
        loc="lower center",
        frameon=True,
        framealpha=1.0,
        facecolor=COLORS["offwhite"],
    )

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
