#!/usr/bin/env python3
"""
gcm_collective_fsle: collective Lyapunov exponent of a heterogeneous GCM.

Reproduces Shibata & Kaneko, Phys. Rev. Lett. 81, 4116 (1998), Figs. 3-5.

Model, their Eq. (1):
    x_{n+1}(i) = (1 - eps) f_i(x_n(i)) + (eps/N) sum_j f_j(x_n(j)),
    f_i(x) = 1 - a(i) x^2,  a(i) = a0 + Delta_a (2i - N)/(2N),  i = 1..N.
Mean field, Eq. (2): h_n = (1/N) sum_i f_i(x_n(i)).

Finite-size Lyapunov exponent (FSLE), Eq. (3):
    lambda_delta0(Delta) = log(Delta/delta0) / tau(Delta),
with tau(Delta) the first time |h'_n - h_n| >= Delta, averaged over samples (<tau> is
used).  The perturbed copy is x' = x + 1e-3 delta0 sigma, sigma uniform in [-1, 1] (the
paper's Fig. 2 scheme, scaled down); tau counts from the first step with
|h'_n - h_n| >= delta0, so the perturbation is aligned with the growing direction before
the clock starts (Aurell et al. 1997).

Normalised time, Eq. (6) (chaotic case): t(Delta) = tau(Delta) + log(delta0)/lambda_m,
    t = log(Delta)/lambda_m                                   for Delta < Delta_m,
    t = log(Delta/Delta_m)/lambda_C + log(Delta_m)/lambda_m   for Delta > Delta_m.
lambda_m (microscopic largest exponent) comes from the tangent map (Benettin et al. 1980).

Cases: chaotic a0 = 1.9, Delta_a = 0.05, eps = 0.098 (their Figs. 3, 4a) and the
identical-element control a0 = 1.6962, Delta_a = 0, eps = 0.008 (their Fig. 4c).  The
full run covers N = 1e4, 1e5, 1e6.  Larger sizes are added one at a time with ``--add N``
(N = 1e7 needs several hours and about 0.6 GB; ``--estimate N`` prints the cost first); each
goes to ``gcm_collective_fsle_N<N>.npz`` beside the cache and the plot merges it in.

OUTPUTS: figures/sec10_gcm/gcm_collective_fsle.npz,
         figures/sec10_gcm/gcm_collective_fsle.png
USAGE:   python src/dynachaos/cml/gcm_collective_fsle.py              # N <= 1e6
         python src/dynachaos/cml/gcm_collective_fsle.py --estimate 10000000
         python src/dynachaos/cml/gcm_collective_fsle.py --add 10000000
"""

import argparse
import time
from pathlib import Path

import numpy as np

from dynachaos.io.paths import safe_load, section_dir

FIG_DIR = section_dir("sec10_gcm")
OUTPUT_NPZ = FIG_DIR / "gcm_collective_fsle.npz"
OUTPUT_PNG = FIG_DIR / "gcm_collective_fsle.png"

CHAOS = {"a0": 1.9, "delta_a": 0.05, "eps": 0.098}  # their Figs. 3, 4(a)
CONTROL = {"a0": 1.6962, "delta_a": 0.0, "eps": 0.008}  # their Fig. 4(c), identical GCM
DELTAS = np.logspace(-6.5, -0.5, 31)

# Run lengths: n_max caps tau(Delta); n_transient precedes the reference orbit; n_gap
# separates successive perturbations along it.
N_MAX = 3000
N_TRANSIENT = 2000
N_GAP = 100
DEFAULT_NS = (10_000, 100_000, 1_000_000)
DEFAULT_SAMPLES = 100

REQUIRED_KEYS = (
    "deltas",
    "Ns",
    "n_max",
    "n_samples",
    "chaos_a0",
    "chaos_delta_a",
    "chaos_eps",
    "control_a0",
    "control_delta_a",
    "control_eps",
    "chaos_tau_N10000",
    "chaos_lam_m_N10000",
    "chaos_tau_N100000",
    "chaos_lam_m_N100000",
    "chaos_tau_N1000000",
    "chaos_lam_m_N1000000",
    "chaos_tau_N1000000_d1e-06",
    "chaos_tau_N1000000_d1e-05",
    "control_tau_N100000",
    "control_lam_m_N100000",
    "control_tau_N1000000",
    "control_lam_m_N1000000",
)


# ---------------------------------------------------------------------------
# Model and estimators
# ---------------------------------------------------------------------------


def a_profile(a0, delta_a, N):
    """Nonlinearity of site i = 1..N: a0 + delta_a (2i - N) / (2N)."""
    i = np.arange(1, N + 1, dtype=np.float64)
    return a0 + delta_a * (2.0 * i - N) / (2.0 * N)


def step(x, a, eps):
    """One GCM step.  Return the new state and the mean field h_n of the old state."""
    fx = 1.0 - a * x * x
    h = fx.mean()
    return (1.0 - eps) * fx + eps * h, h


def check(x, where):
    """Raise ``RuntimeError`` when the orbit is not finite or leaves [-1, 1]."""
    if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1.0:
        raise RuntimeError(f"orbit left [-1, 1] ({where})")


def lambda_m(a, eps, x, n_transient=2000, n_measure=5000, seed=1):
    """Largest microscopic Lyapunov exponent by the tangent map (Benettin et al. 1980).

    ``a`` is the per-site nonlinearity (scalar or array) and ``x`` the starting state.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_transient):
        x, _ = step(x, a, eps)
    v = rng.standard_normal(x.size)
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(n_measure):
        fp = -2.0 * a * x
        v = (1.0 - eps) * fp * v + eps * np.mean(fp * v)
        x, _ = step(x, a, eps)
        r = np.linalg.norm(v)
        s += np.log(r)
        v /= r
    check(x, "lambda_m")
    return s / n_measure


def first_passage(x, a, eps, delta0, rng, n_max, kick=1e-3, deltas=None):
    """Return tau(Delta) for every Delta in ``deltas`` and the reference state after the run.

    tau is ``n_max + 1`` where Delta was not reached within ``n_max`` steps of the clock
    start.  The copy starts at x + kick * delta0 * sigma, sigma uniform in [-1, 1].  The
    clock starts when |h'_n - h_n| first reaches delta0.
    """
    deltas = DELTAS if deltas is None else deltas
    sigma = rng.uniform(-1.0, 1.0, x.size)
    xp = x + kick * delta0 * sigma
    tau = np.full(deltas.size, n_max + 1, dtype=np.int64)
    n0 = None  # step at which |h'_n - h_n| first reaches delta0
    k = 0  # index of the next Delta to reach
    for n in range(10 * n_max):
        x_new, h = step(x, a, eps)
        xp_new, hp = step(xp, a, eps)
        d = abs(hp - h)
        if n0 is None and d >= delta0:
            n0 = n
        if n0 is not None:
            if n - n0 > n_max:
                break
            while k < deltas.size and d >= deltas[k]:
                tau[k] = n - n0
                k += 1
        x, xp = x_new, xp_new
        if k == deltas.size:
            break
    if n0 is None:
        raise RuntimeError("mean-field separation never reached delta0")
    check(x, "reference")
    check(xp, "perturbed copy")
    return tau, x


def fsle_run(params, N, delta0, n_samples, n_max, n_transient, n_gap, seed):
    """Average tau(Delta) over ``n_samples`` perturbations along one long reference orbit.

    Returns the ``(n_samples, len(DELTAS))`` array of tau and the microscopic exponent
    lambda_m measured on the same orbit afterwards.
    """
    rng = np.random.default_rng(seed)
    a = a_profile(params["a0"], params["delta_a"], N)
    x = rng.uniform(-1.0, 1.0, N)
    for _ in range(n_transient):
        x, _ = step(x, a, params["eps"])
    taus = np.empty((n_samples, DELTAS.size), dtype=np.int64)
    t0 = time.perf_counter()
    for s in range(n_samples):
        taus[s], x = first_passage(x, a, params["eps"], delta0, rng, n_max)
        for _ in range(n_gap):
            x, _ = step(x, a, params["eps"])
    lam = lambda_m(a, params["eps"], x)
    print(
        f"  N={N:.0e} delta0={delta0:.0e}: lambda_m={lam:.4f}, {time.perf_counter() - t0:.0f} s",
        flush=True,
    )
    return taus, lam


def t_model(logD, lam_C, log_Dm, lam_m):
    """Eq. (6), chaotic case, with the microscopic branch below Delta_m."""
    return np.where(logD < log_Dm, logD / lam_m, (logD - log_Dm) / lam_C + log_Dm / lam_m)


def fit_t(taus, delta0, lam_m, n_max, d_max):
    """Fit Eq. (6) to <tau> over Delta <= d_max, where every sample reached Delta.

    Returns ``(popt, perr, ok)``: popt = (lambda_C, log Delta_m), perr their standard errors
    and ``ok`` the mask of the Delta values used.
    """
    from scipy.optimize import curve_fit

    ok = np.all(taus <= n_max, axis=0) & (DELTAS <= d_max) & (DELTAS > delta0)
    t = taus[:, ok].mean(axis=0) + np.log(delta0) / lam_m
    logD = np.log(DELTAS[ok])
    popt, pcov = curve_fit(
        lambda L, lc, ldm: t_model(L, lc, ldm, lam_m), logD, t, p0=(0.05, np.log(1e-3))
    )
    return popt, np.sqrt(np.diag(pcov)), ok


def fsle_curve(taus, delta0, n_max):
    """FSLE lambda_delta0(Delta) = log(Delta/delta0)/<tau> where every sample reached Delta.

    Returns ``(Delta, lambda)`` restricted to Delta > delta0.
    """
    ok = np.all(taus <= n_max, axis=0) & (DELTAS > delta0)
    return DELTAS[ok], np.log(DELTAS[ok] / delta0) / taus[:, ok].mean(axis=0)


def estimate_step_time(N, params=CHAOS, n_steps=50):
    """Seconds per two-copy GCM step at size N (timed over ``n_steps``)."""
    a = a_profile(params["a0"], params["delta_a"], N)
    x = np.random.default_rng(0).uniform(-1.0, 1.0, N)
    xp = x + 1e-7
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x, _ = step(x, a, params["eps"])
        xp, _ = step(xp, a, params["eps"])
    return (time.perf_counter() - t0) / n_steps


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute(Ns=DEFAULT_NS, samples=DEFAULT_SAMPLES, output_path=OUTPUT_NPZ):
    """Run the chaotic case at every N (delta0 = 1e-7), at the largest N for delta0 = 1e-6 and
    1e-5, and the identical-element control at the two largest N; cache the tau arrays.

    The run at size N uses ``np.random.default_rng(N)`` (N + 7 for the other delta0, N + 3
    for the control).
    """
    Ns = [int(N) for N in Ns]
    out = {
        "deltas": DELTAS,
        "Ns": np.array(Ns),
        "n_max": np.array([N_MAX]),
        "n_samples": np.array([samples]),
    }
    for k, v in CHAOS.items():
        out[f"chaos_{k}"] = np.array([v])
    for k, v in CONTROL.items():
        out[f"control_{k}"] = np.array([v])

    print("Chaotic case, delta0 = 1e-7, all N", flush=True)
    for N in Ns:
        taus, lam = fsle_run(CHAOS, N, 1e-7, samples, N_MAX, N_TRANSIENT, N_GAP, seed=N)
        out[f"chaos_tau_N{N}"], out[f"chaos_lam_m_N{N}"] = taus, np.array([lam])
    Nf = Ns[-1]
    print(f"Chaotic case, N = {Nf:.0e}, other delta0", flush=True)
    for d0 in (1e-6, 1e-5):
        taus, lam = fsle_run(CHAOS, Nf, d0, samples, N_MAX, N_TRANSIENT, N_GAP, seed=Nf + 7)
        out[f"chaos_tau_N{Nf}_d{d0:.0e}"] = taus
    print("Control (identical GCM)", flush=True)
    for N in Ns[-2:]:
        taus, lam = fsle_run(CONTROL, N, 1e-7, samples, N_MAX, N_TRANSIENT, N_GAP, seed=N + 3)
        out[f"control_tau_N{N}"], out[f"control_lam_m_N{N}"] = taus, np.array([lam])

    if output_path is not None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **out)
        print(f"Saved {output_path}", flush=True)
    return out


def _extra_size_path(path, N):
    path = Path(path)
    return path.with_name(f"{path.stem}_N{int(N)}.npz")


def add_system_size(N, path=OUTPUT_NPZ):
    """Compute the chaotic delta0 = 1e-7 set at size N and store it beside the cache.

    The result goes to ``gcm_collective_fsle_N<N>.npz`` so that the base cache stays the
    reproducible product of ``dynachaos run``; ``load_data`` merges it back in.  The run
    uses the same number of samples as the base cache and ``np.random.default_rng(N)``.
    """
    N = int(N)
    with safe_load(path) as cached:
        n_samples = int(cached["n_samples"][0])
    taus, lam = fsle_run(CHAOS, N, 1e-7, n_samples, N_MAX, N_TRANSIENT, N_GAP, seed=N)
    extra = _extra_size_path(path, N)
    np.savez_compressed(
        extra,
        **{f"chaos_tau_N{N}": taus, f"chaos_lam_m_N{N}": np.array([lam])},
        n_samples=np.array([n_samples]),
    )
    print(f"Saved {extra}", flush=True)
    return extra


def load_data(path=OUTPUT_NPZ):
    """Load the base cache and merge every ``gcm_collective_fsle_N<N>.npz`` beside it.

    Raises ``FileNotFoundError`` when the base cache is missing and ``KeyError`` when it
    lacks a required key.
    """
    path = Path(path)
    with safe_load(path) as cached:
        data = dict(cached)
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"cache lacks {missing}")
    sizes = {int(v) for v in data["Ns"]}
    for extra in sorted(path.parent.glob(f"{path.stem}_N*.npz")):
        with safe_load(extra) as cached:
            for key in cached.files:
                if key.startswith("chaos_"):
                    data[key] = cached[key]
                    if key.startswith("chaos_tau_N"):
                        sizes.add(int(key.removeprefix("chaos_tau_N")))
    data["Ns"] = np.array(sorted(sizes))
    return data


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------


def analyse(data, d_max=0.05):
    """Fit Eq. (6) at every N and Delta_m ~ N^p.

    Returns ``{"fits": {N: (popt, perr, lam_m)}, "slope": p, "slope_err": err}``; the error
    is NaN with two or fewer sizes.
    """
    Ns = [int(N) for N in data["Ns"]]
    n_max = int(data["n_max"][0])
    fits = {}
    for N in Ns:
        taus, lam = data[f"chaos_tau_N{N}"], float(data[f"chaos_lam_m_N{N}"][0])
        popt, perr, _ = fit_t(taus, 1e-7, lam, n_max, d_max)
        fits[N] = (popt, perr, lam)
    log_N = np.log(np.array(Ns, dtype=float))
    log_dm = np.array([fits[N][0][1] for N in Ns])
    if len(Ns) > 2:
        slope, cov = np.polyfit(log_N, log_dm, 1, cov=True)
        err = float(np.sqrt(cov[0, 0]))
    else:
        slope, err = np.polyfit(log_N, log_dm, 1), float("nan")
    return {"fits": fits, "slope": float(slope[0]), "slope_err": err}


def plot(data, d_max=0.05):
    """Three panels: (a) FSLE vs Delta, (b) normalised time with Eq. (6) fits, (c) Delta_m(N)."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import COLORS, apply_axes_polish, figure_spec, setup

    setup()

    Ns = [int(N) for N in data["Ns"]]
    n_max = int(data["n_max"][0])
    spec = figure_spec("double")
    fig, axs = plt.subplots(1, 3, figsize=(spec.figsize[0] * 1.9, spec.figsize[1] * 1.2))
    Nf = max(N for N in Ns if f"chaos_tau_N{N}_d1e-06" in data)  # largest N with all delta0
    lam_f = float(data[f"chaos_lam_m_N{Nf}"][0])

    # (a) FSLE vs Delta for several delta0 at the largest N
    for (key, d0), lw in zip(
        (
            (f"chaos_tau_N{Nf}", 1e-7),
            (f"chaos_tau_N{Nf}_d1e-06", 1e-6),
            (f"chaos_tau_N{Nf}_d1e-05", 1e-5),
        ),
        (3.0, 2.0, 1.0),
        strict=True,
    ):
        delta, lam = fsle_curve(data[key], d0, n_max)
        axs[0].plot(
            delta, lam, lw=lw, marker="o", ms=3, label=rf"$\delta_0 = 10^{{{int(np.log10(d0))}}}$"
        )
    axs[0].axhline(
        lam_f, color=COLORS["grey"], ls="--", lw=0.8, label=rf"$\lambda_m = {lam_f:.3f}$"
    )
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"Scale $\Delta$ of $|h'_n - h_n|$")
    axs[0].set_ylabel(r"$\lambda_{\delta_0}(\Delta)$ (per iteration)")
    axs[0].set_title(rf"(a) FSLE, $N = 10^{{{int(np.log10(Nf))}}}$", loc="left")
    axs[0].legend(frameon=True, fontsize=spec.legend_size - 1)

    # (b) t(Delta) per N with the Eq. (6) fit; thickest line at the back
    analysis = analyse(data, d_max)
    fits = analysis["fits"]
    colors = (COLORS["red"], COLORS["black"], COLORS["blue"], COLORS["grey"])
    for N, lw, col in zip(Ns[::-1], (3.0, 2.0, 1.0, 0.8), colors, strict=False):
        taus, lam = data[f"chaos_tau_N{N}"], float(data[f"chaos_lam_m_N{N}"][0])
        popt, _, ok = fit_t(taus, 1e-7, lam, n_max, d_max)
        t = taus[:, ok].mean(axis=0) + np.log(1e-7) / lam
        axs[1].plot(
            DELTAS[ok],
            t,
            color=col,
            lw=lw,
            marker="o",
            ms=3,
            ls="none" if lw < 1 else "-",
            label=rf"$N = 10^{{{int(np.log10(N))}}}$, $\lambda_C = {popt[0]:.3f}$",
        )
        axs[1].plot(
            DELTAS[ok],
            t_model(np.log(DELTAS[ok]), popt[0], popt[1], lam),
            color=col,
            lw=0.8,
            ls=":",
        )
    for N in [N for N in Ns if f"control_tau_N{N}" in data]:
        taus, lam = data[f"control_tau_N{N}"], float(data[f"control_lam_m_N{N}"][0])
        ok = np.all(taus <= n_max, axis=0) & (DELTAS > 1e-7)
        t = taus[:, ok].mean(axis=0) + np.log(1e-7) / lam
        axs[1].plot(
            DELTAS[ok],
            t,
            color=COLORS["grey"],
            lw=1.0,
            marker="s",
            ms=3,
            label=rf"control $\Delta a = 0$, $N = 10^{{{int(np.log10(N))}}}$",
        )
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"Scale $\Delta$")
    axs[1].set_ylabel(r"$t(\Delta)$ (iterations)")
    axs[1].set_title("(b) Normalised time, Eq. (6) fits", loc="left")
    axs[1].legend(frameon=True, fontsize=spec.legend_size - 2)

    # (c) Delta_m vs N
    Nn = np.array(Ns, dtype=float)
    dm = np.array([np.exp(fits[N][0][1]) for N in Ns])
    dm_err = np.array([fits[N][1][1] for N in Ns]) * dm
    axs[2].errorbar(Nn, dm, yerr=dm_err, marker="o", lw=1.5, capsize=3, label=r"fitted $\Delta_m$")
    axs[2].plot(
        Nn,
        dm[-1] * np.sqrt(Nn[-1] / Nn),
        color=COLORS["grey"],
        ls="--",
        lw=0.8,
        label=r"$\propto N^{-1/2}$",
    )
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel(r"System size $N$")
    axs[2].set_ylabel(r"$\Delta_m$")
    axs[2].set_title(r"(c) Microscopic scale $\Delta_m(N)$", loc="left")
    axs[2].legend(frameon=True, fontsize=spec.legend_size - 1)

    for ax in axs:
        apply_axes_polish(ax, kind="double", title_loc="left", grid=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")

    for N in Ns:
        (lc, ldm), (elc, eldm), lam = fits[N]
        print(
            f"N={N:.0e}: lambda_m={lam:.4f} lambda_C={lc:.4f}+-{elc:.4f} "
            f"Delta_m={np.exp(ldm):.2e} (log err {eldm:.3f})"
        )
    print(f"Delta_m ~ N^{analysis['slope']:.3f} +- {analysis['slope_err']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--estimate", type=int, default=0, help="time one step pair at this N, print the cost, exit"
    )
    ap.add_argument("--Ns", type=float, nargs="+", default=list(DEFAULT_NS))
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument(
        "--add",
        type=int,
        default=0,
        help="compute the chaotic delta0 = 1e-7 set at this N, store it beside the cache, re-plot",
    )
    args = ap.parse_args(argv)

    if args.estimate:
        per_step = estimate_step_time(args.estimate)
        steps_per_sample = 1300 + N_GAP  # measured: about 1000 steps per sample, up to n_max
        total = per_step * (N_TRANSIENT + args.samples * steps_per_sample) + per_step / 2 * 7000
        memory_gb = 8 * args.estimate * 8 / 1e9
        print(
            f"N={args.estimate:.0e}: {per_step * 1e3:.1f} ms per two-copy step; one delta0 set of "
            f"{args.samples} samples about {total / 3600:.2f} h; arrays about {memory_gb:.2f} GB"
        )
        return

    if args.add:
        add_system_size(args.add)
        plot(load_data())
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = load_data()
        print(f"Loaded {OUTPUT_NPZ} (sizes {', '.join(f'{N:.0e}' for N in data['Ns'])})")
    except (FileNotFoundError, KeyError) as exc:
        print(f"Computing FSLE of the heterogeneous GCM ({exc})...")
        data = compute([int(N) for N in args.Ns], args.samples)
    plot(data)


if __name__ == "__main__":
    main()
