#!/usr/bin/env python3
"""
gcm_lyapunov_max: largest Lyapunov exponent of the globally coupled map vs a.

The GCM
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + (eps/N) sum_j f(x_n(j)),   f(x) = 1 - a x^2,
has identical elements (Kaneko 1990).  One tangent vector v evolves under the exact
Jacobian,
    v -> (1 - eps) f'(x) v + eps <f'(x) v>,
and is renormalised every step (Benettin et al. 1980).  The mean log growth is the
largest Lyapunov exponent lambda_max.

This exponent belongs to the microscopic (site) dynamics.  Shibata and Kaneko (1998)
show that the finite-size Lyapunov exponent of the mean field tends to it as the
perturbation goes to zero; their collective exponent lambda_C needs a finite
perturbation scale (see ``gcm_collective_fsle``).

The figure runs 8 seeds for N = 500, 1000 and 2000 over a in [1.40, 2.00] at eps = 0.1.

OUTPUTS: figures/sec10_gcm/gcm_lyapunov_max.npz,
         figures/sec10_gcm/gcm_lyapunov_max.png
USAGE:   python src/dynachaos/cml/gcm_lyapunov_max.py
"""

import numpy as np

from dynachaos.cml.primitives import gcm_step
from dynachaos.io.paths import load_or_compute_npz, section_dir

FIG_DIR = section_dir("sec10_gcm")
OUTPUT_NPZ = FIG_DIR / "gcm_lyapunov_max.npz"
OUTPUT_PNG = FIG_DIR / "gcm_lyapunov_max.png"

N_VALUES = (500, 1000, 2000)
REQUIRED_KEYS = (
    "a_values",
    "N_values",
    "eps",
    "n_seeds",
    "n_transient",
    "n_measure",
    "seed_base",
    *(f"runs_{N}" for N in N_VALUES),
    *(f"mean_{N}" for N in N_VALUES),
    *(f"se_{N}" for N in N_VALUES),
)


def lyapunov_max(a, eps, n_sites, n_transient, n_measure, seed):
    """Return lambda_max of the GCM for one random initial condition.

    The initial state is uniform on [-1, 1] and the tangent vector standard normal,
    both drawn from ``np.random.default_rng(seed)``.  Raises ``RuntimeError`` if the
    orbit leaves [-1, 1], so a divergent run cannot enter an average.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n_sites)
    for _ in range(n_transient):
        x = gcm_step(x, a, eps)
    v = rng.standard_normal(n_sites)
    v /= np.linalg.norm(v)
    log_sum = 0.0
    for _ in range(n_measure):
        fp = -2.0 * a * x
        v = (1.0 - eps) * fp * v + eps * np.mean(fp * v)
        x = gcm_step(x, a, eps)
        r = np.linalg.norm(v)
        log_sum += np.log(r)
        v /= r
    if not np.all(np.abs(x) <= 1.0):
        raise RuntimeError(f"orbit left [-1, 1]: a={a}, N={n_sites}, seed={seed}")
    return log_sum / n_measure


def compute(
    a_values=None,
    eps=0.1,
    N_values=N_VALUES,
    n_seeds=8,
    n_transient=1000,
    n_measure=10_000,
    seed_base=42,
    output_path=OUTPUT_NPZ,
):
    """Compute lambda_max for every (N, seed, a) and cache the ensemble.

    The run for size N, seed index s and a-index ia uses the generator seed
    ``seed_base + 1000 * N + s + ia``.  ``runs_<N>`` has shape ``(n_seeds, len(a))``;
    ``mean_<N>`` and ``se_<N>`` are the mean over seeds and its standard error
    (population standard deviation over sqrt(n_seeds)).
    """
    if a_values is None:
        a_values = np.linspace(1.40, 2.00, 100)
    a_values = np.asarray(a_values, dtype=np.float64)
    payload = {
        "a_values": a_values,
        "N_values": np.array(N_values, dtype=np.int64),
        "eps": np.array([eps]),
        "n_seeds": np.array([n_seeds], dtype=np.int64),
        "n_transient": np.array([n_transient], dtype=np.int64),
        "n_measure": np.array([n_measure], dtype=np.int64),
        "seed_base": np.array([seed_base], dtype=np.int64),
    }
    for N in N_values:
        print(f"  Running N = {N} ({n_seeds} seeds, {len(a_values)} a values)...", flush=True)
        runs = np.array(
            [
                [
                    lyapunov_max(a, eps, N, n_transient, n_measure, seed_base + 1000 * N + s + ia)
                    for ia, a in enumerate(a_values)
                ]
                for s in range(n_seeds)
            ]
        )
        payload[f"runs_{N}"] = runs
        payload[f"mean_{N}"] = runs.mean(axis=0)
        payload[f"se_{N}"] = runs.std(axis=0) / np.sqrt(n_seeds)
        print(f"  N = {N}: lambda_max in [{runs.min():+.4f}, {runs.max():+.4f}]", flush=True)

    if output_path is not None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


def plot(data):
    """Plot lambda_max vs a for each N, with +-1 standard-error bands."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, reference_line, setup

    setup()

    a = data["a_values"]
    spec = figure_spec("double")
    fig, ax = plt.subplots(figsize=spec.figsize)
    reference_line(ax, 0.0, axis="y", lw=0.8, label=r"$\lambda_\mathrm{max} = 0$")
    # The thickest line goes at the back.
    for N, lw, ms, col in (
        (2000, 3.0, 4.5, COLORS["red"]),
        (1000, 2.0, 3.5, COLORS["black"]),
        (500, 1.0, 2.5, COLORS["blue"]),
    ):
        mean, se = data[f"mean_{N}"], data[f"se_{N}"]
        ax.fill_between(a, mean - se, mean + se, color=col, alpha=0.15, lw=0)
        ax.plot(a, mean, color=col, lw=lw, marker="o", ms=ms, markevery=4, label=rf"$N = {N}$")
    ax.set_xlabel(r"Nonlinearity $a$")
    ax.set_ylabel(r"$\lambda_\mathrm{max}$ (per iteration)")
    ax.set_xlim(a[0], a[-1])
    ax.legend(loc="upper left", frameon=True)
    apply_axes_polish(ax, kind="double", grid=False)

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_compute_npz(
        OUTPUT_NPZ,
        "GCM largest Lyapunov exponent",
        compute,
        required_keys=REQUIRED_KEYS,
    )
    plot(data)


if __name__ == "__main__":
    main()
