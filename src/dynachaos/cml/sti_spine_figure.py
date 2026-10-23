"""Kaneko CML spatiotemporal-intermittency spine figure.

Kaneko Model-A CML:
  - the turbulent-fraction curve phi_turb(eps) locates the critical coupling eps_c
    (first sweep coupling where phi_turb exceeds 0.001, interpolated linearly and
    rounded to three decimals);
  - a power law, an exponential and a truncated power law are fitted by maximum
    likelihood to the laminar cluster sizes (lengths >= 2) at eps_c, 0.060 and 0.080,
    and compared by BIC and by the Kolmogorov-Smirnov distance.

Panels: (a) spacetime diagram at eps = 0.08, (b) phi_turb vs eps with eps_c marked,
(c) laminar-length distributions at the three fitted couplings.
"""

from pathlib import Path

import numpy as np

from dynachaos.cml.spatiotemporal import model_A_f, simulate_cml
from dynachaos.io.paths import safe_load, section_dir

SECTION_ID = "sec12_intermittency"
FIG_DIR = section_dir(SECTION_ID)
OUTPUT_NPZ = FIG_DIR / "sti_spine.npz"
OUTPUT_PNG = FIG_DIR / "sti_spine.png"
_THIS_FILE = Path(__file__).resolve()

SCHEMA_VERSION = 5  # 5: no tail fit when all tail lengths are equal; 4: KS at support values
TURBULENT_ONSET_FRACTION = 0.001
FIT_FIELDS = (
    "x_min",
    "n_tail",
    "alpha_pl",
    "ll_pl",
    "bic_pl",
    "ks_pl",
    "lam_exp",
    "ll_exp",
    "bic_exp",
    "ks_exp",
    "alpha_tr",
    "lam_tr",
    "ll_tr",
    "bic_tr",
    "ks_tr",
)

REQUIRED_KEYS = (
    "schema_version",
    "source_file",
    "seed",
    "model_a_parameter",
    "display_eps",
    "sweep_eps",
    "turbulent_fraction",
    "eps_c",
    "fit_eps_values",
    "spacetime",
    "turbulent_mask",
    "laminar_values_0",
    "laminar_probs_0",
    "laminar_values_1",
    "laminar_probs_1",
    "laminar_values_2",
    "laminar_probs_2",
    *(f"fit_{field}" for field in FIT_FIELDS),
    "fit_preferred",
)


def locate_critical_coupling(eps_values, turbulent_fraction, threshold=TURBULENT_ONSET_FRACTION):
    """Critical coupling: where the turbulent fraction first exceeds ``threshold``.

    Linear interpolation between the last sweep point at or below the threshold and the
    first one above it, rounded to three decimals.  Returns NaN when the first sweep point
    is already turbulent or no point is.
    """
    eps_values = np.asarray(eps_values, dtype=np.float64)
    turbulent_fraction = np.asarray(turbulent_fraction, dtype=np.float64)
    active = np.flatnonzero(turbulent_fraction > threshold)
    if active.size == 0 or active[0] == 0:
        return float("nan")
    i1 = int(active[0])
    i0 = i1 - 1
    frac0, frac1 = turbulent_fraction[i0], turbulent_fraction[i1]
    eps_c = eps_values[i0] + (threshold - frac0) * (eps_values[i1] - eps_values[i0]) / (
        frac1 - frac0
    )
    return round(float(eps_c), 3)


def fit_tail_models(lengths, x_min=2):
    """Fit a power law, an exponential and a truncated power law to the tail ``lengths >= x_min``.

    Discrete maximum likelihood on the support x_min..max(tail).  ``bic_*`` is
    k ln n - 2 ln L (k = 1, 1, 2 parameters) and ``ks_*`` the largest distance between the
    empirical and the model CDF.  ``preferred`` names the model of lowest BIC.  Returns an
    empty dict when the tail has fewer than 10 values or fewer than 2 distinct values (a fully
    laminar lattice has one run length, so the distribution is not fitted).
    """
    from scipy.optimize import minimize

    lengths = np.asarray(lengths)
    tail = lengths[lengths >= x_min]
    n = len(tail)
    if n < 10 or np.unique(tail).size < 2:
        return {}

    x_max = int(np.max(tail))
    x_supp = np.arange(x_min, x_max + 1, dtype=np.float64)
    # The lengths are integers with many ties. Compare the CDFs at each support value, not at
    # each sample: the rank of a tied sample is not its CDF value.
    emp_cdf = np.searchsorted(np.sort(tail), x_supp, side="right") / n
    sum_log = np.sum(np.log(tail))
    sum_tail = np.sum(tail)

    def ks_distance(pmf):
        return float(np.max(np.abs(emp_cdf - np.cumsum(pmf))))

    # 1. Power law
    def nll_pl(a):
        Z = np.sum(x_supp ** (-a[0]))
        if Z <= 0 or not np.isfinite(Z):
            return 1e12
        return -(-a[0] * sum_log - n * np.log(Z))

    res_pl = minimize(nll_pl, [2.0], bounds=[(0.1, 10.0)])
    alpha_pl = float(res_pl.x[0])
    ll_pl = float(-res_pl.fun)
    bic_pl = float(1 * np.log(n) - 2 * ll_pl)
    ks_pl = ks_distance((x_supp ** (-alpha_pl)) / np.sum(x_supp ** (-alpha_pl)))

    # 2. Exponential
    def nll_exp(lam):
        Z = np.sum(np.exp(-lam[0] * x_supp))
        if Z <= 0 or not np.isfinite(Z):
            return 1e12
        return -(-lam[0] * sum_tail - n * np.log(Z))

    res_exp = minimize(nll_exp, [0.1], bounds=[(1e-5, 10.0)])
    lam_exp = float(res_exp.x[0])
    ll_exp = float(-res_exp.fun)
    bic_exp = float(1 * np.log(n) - 2 * ll_exp)
    pmf_exp = np.exp(-lam_exp * x_supp) / np.sum(np.exp(-lam_exp * x_supp))
    ks_exp = ks_distance(pmf_exp)

    # 3. Truncated power law: P(x) ~ x^(-alpha) exp(-lambda x)
    def nll_tr(params):
        a, lam = params
        Z = np.sum((x_supp ** (-a)) * np.exp(-lam * x_supp))
        if Z <= 0 or not np.isfinite(Z):
            return 1e12
        return -(-a * sum_log - lam * sum_tail - n * np.log(Z))

    res_tr = minimize(nll_tr, [alpha_pl, lam_exp], bounds=[(0.0, 10.0), (0.0, 10.0)])
    alpha_tr = float(res_tr.x[0])
    lam_tr = float(res_tr.x[1])
    ll_tr = float(-res_tr.fun)
    bic_tr = float(2 * np.log(n) - 2 * ll_tr)
    terms_tr = (x_supp ** (-alpha_tr)) * np.exp(-lam_tr * x_supp)
    ks_tr = ks_distance(terms_tr / np.sum(terms_tr))

    bics = [bic_pl, bic_exp, bic_tr]
    preferred = ["power_law", "exponential", "truncated_power_law"][int(np.argmin(bics))]
    return {
        "x_min": x_min,
        "n_tail": n,
        "alpha_pl": alpha_pl,
        "ll_pl": ll_pl,
        "bic_pl": bic_pl,
        "ks_pl": ks_pl,
        "lam_exp": lam_exp,
        "ll_exp": ll_exp,
        "bic_exp": bic_exp,
        "ks_exp": ks_exp,
        "alpha_tr": alpha_tr,
        "lam_tr": lam_tr,
        "ll_tr": ll_tr,
        "bic_tr": bic_tr,
        "ks_tr": ks_tr,
        "preferred": preferred,
    }


def compute(
    output_path=OUTPUT_NPZ,
    *,
    seed=20260602,
    n_sites=512,
    n_transient=1_200,
    n_record=480,
    sweep_eps=None,
    sweep_n_transient=1_000,
    sweep_n_record=300,
    fit_n_transient=2_000,
    fit_n_record=1_000,
):
    """Compute the STI spine: spacetime at eps = 0.08, phi_turb(eps), eps_c and tail fits.

    ``n_transient``/``n_record`` apply to the display run; the sweep and the tail runs have
    their own lengths.  Seeds: the display run uses ``seed``, sweep point ``i`` uses
    ``seed + i + 1`` and the laminar-tail run at coupling eps uses
    ``seed + int(eps * 10000)``; each draws the
    initial lattice with ``np.random.default_rng``.  The three fitted couplings are eps_c
    (the smallest swept coupling when no onset is bracketed), 0.060 and 0.080.
    """
    display_eps = 0.08
    if sweep_eps is None:
        sweep_eps = np.linspace(0.03, 0.12, 19)
    sweep_eps = np.asarray(sweep_eps, dtype=np.float64)

    spacetime = _simulate_model_a(
        display_eps,
        seed=seed,
        n_sites=n_sites,
        n_transient=n_transient,
        n_record=n_record,
    )
    turbulent_mask = _burst_mask(spacetime)

    turbulent_fraction = []
    for idx, eps in enumerate(sweep_eps):
        sample = _simulate_model_a(
            float(eps),
            seed=seed + idx + 1,
            n_sites=n_sites,
            n_transient=sweep_n_transient,
            n_record=sweep_n_record,
        )
        turbulent_fraction.append(float(_burst_mask(sample).mean()))
    turbulent_fraction = np.asarray(turbulent_fraction, dtype=np.float64)

    eps_c = locate_critical_coupling(sweep_eps, turbulent_fraction)
    fit_eps_values = np.array(
        [eps_c if np.isfinite(eps_c) else sweep_eps[0], 0.060, 0.080], dtype=np.float64
    )

    payload = {
        "schema_version": np.array([SCHEMA_VERSION], dtype=np.int64),
        "source_file": np.array([_source_file_label()]),
        "seed": np.array([seed], dtype=np.int64),
        "model_a_parameter": np.array([-0.01], dtype=np.float64),
        "display_eps": np.array([display_eps], dtype=np.float64),
        "sweep_eps": sweep_eps,
        "turbulent_fraction": turbulent_fraction,
        "eps_c": np.array([eps_c], dtype=np.float64),
        "fit_eps_values": fit_eps_values,
        "spacetime": spacetime.astype(np.float64),
        "turbulent_mask": turbulent_mask.astype(np.bool_),
    }

    fits = []
    for k, eps_fit in enumerate(fit_eps_values):
        field = _simulate_model_a(
            float(eps_fit),
            seed=seed + int(eps_fit * 10000),
            n_sites=n_sites,
            n_transient=fit_n_transient,
            n_record=fit_n_record,
        )
        lengths = _periodic_run_lengths(~_burst_mask(field))
        values, counts = np.unique(lengths, return_counts=True)
        payload[f"laminar_values_{k}"] = values.astype(np.int64)
        payload[f"laminar_probs_{k}"] = (counts / np.sum(counts)).astype(np.float64)
        fits.append(fit_tail_models(lengths, x_min=2))
    for name in FIT_FIELDS:
        payload[f"fit_{name}"] = np.array([fit.get(name, np.nan) for fit in fits], dtype=np.float64)
    payload["fit_preferred"] = np.array([fit.get("preferred", "none") for fit in fits])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


def plot(data, output_path=OUTPUT_PNG):
    """Render the STI spine figure: spacetime (a), turbulent fraction (b), laminar lengths (c)."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        CMAP_SPACETIME,
        COLORS,
        annotate_on_field,
        apply_axes_polish,
        figure_spec,
        panel_label,
        reference_line,
        setup,
    )

    setup()

    spacetime = np.asarray(data["spacetime"], dtype=np.float64)
    turbulent_mask = np.asarray(data["turbulent_mask"], dtype=np.bool_)
    sweep_eps = np.asarray(data["sweep_eps"], dtype=np.float64)
    turbulent_fraction = np.asarray(data["turbulent_fraction"], dtype=np.float64)
    eps_c = float(np.asarray(data["eps_c"])[0])
    display_eps = float(np.asarray(data["display_eps"])[0])
    fit_eps_values = np.asarray(data["fit_eps_values"], dtype=np.float64)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = figure_spec("double")
    fig = plt.figure(figsize=(spec.figsize[0], spec.figsize[1] * 1.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.24)
    ax_space = fig.add_subplot(gs[0, :])
    ax_rho = fig.add_subplot(gs[1, 0])
    ax_clusters = fig.add_subplot(gs[1, 1])

    # (a) Spacetime diagram with the burst contour
    im = ax_space.imshow(
        spacetime,
        aspect="auto",
        origin="lower",
        cmap=CMAP_SPACETIME,
        # A 512x480 field resampled to figure width with "nearest" aliases into noise;
        # an antialiased (area-averaging) filter keeps the turbulent/laminar texture.
        interpolation="antialiased",
        vmin=-0.2,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax_space, pad=0.02, fraction=0.03, label=r"$x_i(n)$")
    cbar.ax.tick_params(labelsize=spec.tick_size)
    ax_space.contour(
        turbulent_mask.astype(float),
        levels=[0.5],
        colors=[COLORS["black"]],
        linewidths=0.25,
        alpha=0.35,
    )
    annotate_on_field(
        ax_space,
        0.84 * spacetime.shape[1],
        0.88 * spacetime.shape[0],
        rf"$\varepsilon={display_eps:.2f}$",
    )
    ax_space.set_title(
        rf"Model-A CML spatiotemporal intermittency ($\varepsilon={display_eps:.2f}$)",
        loc="left",
    )
    ax_space.set_xlabel("Lattice site $i$")
    ax_space.set_ylabel("Time $n$")
    panel_label(ax_space, "(a)")
    apply_axes_polish(ax_space, kind="double")

    # (b) Turbulent fraction with the critical coupling
    ax_rho.plot(sweep_eps, turbulent_fraction, marker="o", ms=4.0, lw=1.0, color=COLORS["black"])
    if np.isfinite(eps_c):
        reference_line(ax_rho, eps_c, axis="x", lw=0.6, ls="--", color=COLORS["red"])
        ax_rho.text(
            eps_c + 0.003,
            0.2,
            rf"$\varepsilon_c \approx {eps_c:.3f}$",
            color=COLORS["red"],
            fontsize=spec.tick_size,
        )
    ax_rho.set_title(r"Turbulent fraction $\phi_{\mathrm{turb}}$", loc="left")
    ax_rho.set_xlabel(r"Coupling $\varepsilon$")
    ax_rho.set_ylabel(r"$\phi_{\mathrm{turb}}$")
    ax_rho.set_ylim(-0.05, 1.0)
    panel_label(ax_rho, "(b)")
    apply_axes_polish(ax_rho, kind="double")

    # (c) Laminar-length distributions at the three fitted couplings
    colors = [COLORS["blue"], COLORS["black"], COLORS["red"]]
    for k, (eps_val, col) in enumerate(zip(fit_eps_values, colors, strict=True)):
        values = data[f"laminar_values_{k}"]
        probs = data[f"laminar_probs_{k}"]
        if len(values) < 2:
            # The whole lattice is laminar: no laminar-length distribution to show.
            continue
        ax_clusters.semilogy(
            values,
            probs,
            marker="o",
            ms=3.0,
            lw=0.5,
            color=col,
            label=rf"$\varepsilon={eps_val:.3f}$",
        )

    # Exponential fit of the last coupling (eps = 0.08), anchored at its first tail point
    values_last = data["laminar_values_2"]
    tail = values_last >= 2
    rate = float(np.asarray(data["fit_lam_exp"])[2])
    if np.any(tail) and np.isfinite(rate):
        rx = np.array([2.0, 15.0])
        ry = data["laminar_probs_2"][tail][0] * np.exp(-rate * (rx - 2.0))
        ax_clusters.semilogy(rx, ry, lw=1.0, ls="--", color=COLORS["grey"])

    ax_clusters.set_xlim(1, 50)
    ax_clusters.set_ylim(1e-4, 1.0)
    ax_clusters.set_title(r"Laminar length distributions", loc="left")
    ax_clusters.set_xlabel(r"Laminar run length $\ell$")
    ax_clusters.set_ylabel(r"Probability $P(\ell)$")
    ax_clusters.legend(loc="upper right", fontsize=spec.tick_size - 1, framealpha=0.9)
    panel_label(ax_clusters, "(c)")
    apply_axes_polish(ax_clusters, kind="double")

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main():
    """Load or compute the CML/STI spine cache, then render the figure."""
    try:
        data = safe_load(OUTPUT_NPZ)
        missing = tuple(key for key in REQUIRED_KEYS if key not in data.files)
        stale_schema = not missing and int(np.asarray(data["schema_version"])[0]) != SCHEMA_VERSION
        if missing or stale_schema:
            data.close()
            reason = "missing keys (" + ", ".join(missing) + ")" if missing else "stale schema"
            print(f"Cache {OUTPUT_NPZ} {reason}; recomputing...")
            data = compute()
    except FileNotFoundError:
        data = compute()

    try:
        plot(data)
    finally:
        close = getattr(data, "close", None)
        if close is not None:
            close()


def _burst_mask(field, delta=0.05):
    """Kaneko 1985 burst criterion: S(i)=1 if |x(i+1)-x(i)| > delta."""
    return np.abs(field - np.roll(field, -1, axis=1)) > delta


def _simulate_model_a(eps, *, seed, n_sites, n_transient, n_record):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, n_sites)
    return simulate_cml(
        model_A_f,
        model_A_f,
        eps,
        N=n_sites,
        n_transient=n_transient,
        n_record=n_record,
        x0=x0,
    )


def _periodic_run_lengths(mask):
    rows = np.asarray(mask, dtype=np.bool_)
    if rows.ndim != 2:
        raise ValueError("mask must be a 2D spacetime array")

    lengths = []
    for row in rows:
        lengths.extend(_periodic_row_runs(row))
    return np.asarray(lengths, dtype=np.int64)


def _periodic_row_runs(row):
    values = np.asarray(row, dtype=np.bool_)
    if values.size == 0 or not np.any(values):
        return []
    if np.all(values):
        return [int(values.size)]

    start = int(np.flatnonzero(~values)[0])
    ordered = np.concatenate((values[start + 1 :], values[: start + 1]))
    lengths = []
    current = 0
    for flag in ordered:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _source_file_label():
    # as_posix(), not str(): this label is written into the .npz payload as
    # provenance, so it is data rather than a filesystem operation. str() would
    # render it with the OS separator and a cache generated on Windows would
    # record "src\\dynachaos\\..." while every other platform records
    # "src/dynachaos/...", making the metadata non-portable.
    try:
        return _THIS_FILE.relative_to(_THIS_FILE.parents[3]).as_posix()
    except ValueError:
        return _THIS_FILE.name


if __name__ == "__main__":
    main()
