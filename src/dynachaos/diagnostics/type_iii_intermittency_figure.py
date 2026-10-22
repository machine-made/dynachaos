"""Type-III intermittency proof figure pipeline."""

from pathlib import Path

import numpy as np

from dynachaos.diagnostics.intermittency import (
    fit_power_law_mle,
    powerlaw_alpha_ci,
    powerlaw_gof,
    reinjection_Mx,
)
from dynachaos.io.paths import safe_load, section_dir

SECTION_ID = "sec12_intermittency"
FIG_DIR = section_dir(SECTION_ID)
OUTPUT_NPZ = FIG_DIR / "type_iii_intermittency.npz"
OUTPUT_PNG = FIG_DIR / "type_iii_intermittency.png"
_THIS_FILE = Path(__file__).resolve()

REQUIRED_KEYS = (
    "schema_version",
    "source_file",
    "seed",
    "eps",
    "a",
    "escape_threshold",
    "return_grid",
    "f2_return_points",
    "f2_linear_slope",
    "f2_cubic_coefficient",
    "reinjection_points",
    "series",
    "laminar_mask",
    "laminar_lengths",
    "laminar_tail_alpha",
    "laminar_tail_alpha_ci",
    "laminar_tail_gof_p",
    "rpd_thresholds",
    "rpd_conditional_means",
    "rpd_slope",
    "rpd_intercept",
    "rpd_alpha",
    "rpd_rvalue",
)


def compute(
    output_path=OUTPUT_NPZ,
    *,
    seed=20260602,
    n_reinjections=1_200,
    series_sample_size=25_000,
    powerlaw_gof_bootstrap=100,
    alpha_ci_bootstrap=200,
):
    """Compute deterministic FIG C Type-III flip diagnostics and optionally cache them."""
    eps = 2e-3
    a = 1.0
    escape_threshold = 0.35
    rng = np.random.default_rng(seed)

    return_grid = np.linspace(-0.08, 0.08, 2_000, dtype=np.float64)
    f2_values = _type_iii_f2(return_grid, eps=eps, a=a)
    f2_return_points = np.column_stack((return_grid, f2_values))
    cubic_fit = np.polyfit(return_grid, f2_values, deg=3)

    reinjection_points = rng.uniform(2e-6, 2e-3, n_reinjections)
    full_series, full_laminar_mask, laminar_lengths = _escape_episodes(
        reinjection_points,
        eps=eps,
        a=a,
        escape_threshold=escape_threshold,
    )
    rpd = reinjection_Mx(full_series, full_laminar_mask)
    tail_fit = fit_power_law_mle(laminar_lengths, discrete=False)
    tail_gof = powerlaw_gof(
        laminar_lengths,
        fit=tail_fit,
        n_bootstrap=powerlaw_gof_bootstrap,
        rng=seed,
    )
    tail_ci = powerlaw_alpha_ci(
        laminar_lengths,
        fit=tail_fit,
        n_bootstrap=alpha_ci_bootstrap,
        rng=seed + 1,
    )
    series = full_series[:series_sample_size]
    laminar_mask = full_laminar_mask[:series_sample_size]

    payload = {
        "schema_version": np.array([2], dtype=np.int64),
        "source_file": np.array([_source_file_label()]),
        "seed": np.array([seed], dtype=np.int64),
        "eps": np.array([eps], dtype=np.float64),
        "a": np.array([a], dtype=np.float64),
        "escape_threshold": np.array([escape_threshold], dtype=np.float64),
        "return_grid": return_grid.astype(np.float64),
        "f2_return_points": f2_return_points.astype(np.float64),
        "f2_linear_slope": np.array([cubic_fit[2]], dtype=np.float64),
        "f2_cubic_coefficient": np.array([cubic_fit[0]], dtype=np.float64),
        "reinjection_points": reinjection_points.astype(np.float64),
        "series": series.astype(np.float64),
        "laminar_mask": laminar_mask.astype(np.bool_),
        "laminar_lengths": laminar_lengths.astype(np.int64),
        "laminar_tail_alpha": np.array([tail_fit.alpha], dtype=np.float64),
        "laminar_tail_alpha_ci": tail_ci.astype(np.float64),
        "laminar_tail_gof_p": np.array([tail_gof.p_value], dtype=np.float64),
        "rpd_thresholds": rpd.thresholds.astype(np.float64),
        "rpd_conditional_means": rpd.conditional_means.astype(np.float64),
        "rpd_slope": np.array([rpd.slope], dtype=np.float64),
        "rpd_intercept": np.array([rpd.intercept], dtype=np.float64),
        "rpd_alpha": np.array([rpd.alpha], dtype=np.float64),
        "rpd_rvalue": np.array([rpd.rvalue], dtype=np.float64),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


def plot(data, output_path=OUTPUT_PNG):
    """Render the Type-III proof figure."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
    )

    setup()

    f2_points = np.asarray(data["f2_return_points"], dtype=np.float64)
    series = np.asarray(data["series"], dtype=np.float64)
    laminar_mask = np.asarray(data["laminar_mask"], dtype=np.bool_)
    lengths = np.asarray(data["laminar_lengths"], dtype=np.int64)
    tail_alpha = float(np.asarray(data["laminar_tail_alpha"], dtype=np.float64)[0])
    tail_ci = np.asarray(data["laminar_tail_alpha_ci"], dtype=np.float64)
    tail_gof_p = float(np.asarray(data["laminar_tail_gof_p"], dtype=np.float64)[0])
    thresholds = np.asarray(data["rpd_thresholds"], dtype=np.float64)
    conditional_means = np.asarray(data["rpd_conditional_means"], dtype=np.float64)
    rpd_slope = float(np.asarray(data["rpd_slope"], dtype=np.float64)[0])
    rpd_intercept = float(np.asarray(data["rpd_intercept"], dtype=np.float64)[0])
    rpd_alpha = float(np.asarray(data["rpd_alpha"], dtype=np.float64)[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 2, figsize=spec.figsize, constrained_layout=True)
    ax_return, ax_series, ax_lengths, ax_rpd = axes.ravel()

    ax_return.plot(f2_points[:, 0], f2_points[:, 1], color=COLORS["black"], lw=1.2)
    bounds = np.array([np.min(f2_points), np.max(f2_points)], dtype=np.float64)
    ax_return.plot(bounds, bounds, color=COLORS["black"], lw=0.8, ls="--")
    ax_return.set_title("Type-III two-step flip return", loc="left")
    ax_return.set_xlabel("$x_n$")
    ax_return.set_ylabel("$x_{n+2}$")

    window = slice(0, min(5_000, series.size))
    time = np.arange(series[window].size)
    ax_series.semilogy(time, series[window], color=COLORS["black"], lw=0.7)
    ax_series.scatter(
        time[laminar_mask[window]],
        series[window][laminar_mask[window]],
        s=1.8,
        color=COLORS["grey"],
        alpha=0.45,
        linewidths=0,
    )
    ax_series.set_title("Escape episodes after reinjection", loc="left")
    ax_series.set_xlabel("$n$")
    ax_series.set_ylabel("$|x_n|$")

    bins = np.arange(np.min(lengths), np.max(lengths) + 2)
    ax_lengths.hist(lengths, bins=bins, density=True, color=COLORS["black"], alpha=0.75)
    # Bottom-row titles used to be long enough to overwrite the neighbouring
    # panel's title (e.g. "GoF p=0.03" ran into "M(x) slope=0.482"); keep the
    # title short and put the fit statistics in an in-axes text box instead.
    ax_lengths.set_title("Laminar tail (power-law check)", loc="left")
    ax_lengths.text(
        0.97,
        0.95,
        f"$\\alpha$={tail_alpha:.2f}\n"
        f"95% CI [{tail_ci[0]:.2f}, {tail_ci[1]:.2f}]\n"
        f"GoF p={tail_gof_p:.2f}",
        transform=ax_lengths.transAxes,
        ha="right",
        va="top",
        fontsize=spec.legend_size,
    )
    ax_lengths.set_xlabel("laminar length $\\ell$")
    ax_lengths.set_ylabel("density")

    ax_rpd.plot(thresholds, conditional_means, color=COLORS["black"], lw=1.2)
    fit = rpd_slope * thresholds + rpd_intercept
    ax_rpd.plot(thresholds, fit, color=COLORS["grey"], lw=0.9, ls="--")
    ax_rpd.set_title("Recurrence-plot-diagram slope", loc="left")
    ax_rpd.text(
        0.97,
        0.08,
        f"slope={rpd_slope:.3f}\n$\\alpha$={rpd_alpha:.3f}",
        transform=ax_rpd.transAxes,
        ha="right",
        va="bottom",
        fontsize=spec.legend_size,
    )
    ax_rpd.set_xlabel("threshold $x$")
    ax_rpd.set_ylabel("$M(x)$")

    for label, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        panel_label(ax, f"({label})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        apply_axes_polish(ax, kind="grid")

    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main():
    """Load or compute the Type-III cache, then render the figure."""
    try:
        data = safe_load(OUTPUT_NPZ)
        missing = tuple(key for key in REQUIRED_KEYS if key not in data.files)
        stale_schema = not missing and int(np.asarray(data["schema_version"])[0]) != 2
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


def _type_iii_f2(x, *, eps, a):
    return _type_iii_step(_type_iii_step(x, eps=eps, a=a), eps=eps, a=a)


def _type_iii_step(x, *, eps, a):
    return -(1.0 + eps) * x - a * x * x * x


def _escape_episodes(reinjection_points, *, eps, a, escape_threshold, max_steps=5_000):
    series = [escape_threshold * 1.2]
    laminar_mask = [False]
    laminar_lengths = []
    for x0 in reinjection_points:
        orbit = [float(x0)]
        while abs(orbit[-1]) < escape_threshold and len(orbit) < max_steps:
            orbit.append(float(_type_iii_step(orbit[-1], eps=eps, a=a)))
        if abs(orbit[-1]) < escape_threshold:
            raise RuntimeError("Type-III laminar episode reached max_steps before escape")
        laminar = np.abs(np.asarray(orbit[:-1], dtype=np.float64))
        if laminar.size == 0:
            continue
        series.extend(laminar)
        laminar_mask.extend([True] * laminar.size)
        series.append(escape_threshold * 1.2)
        laminar_mask.append(False)
        laminar_lengths.append(laminar.size)
    return (
        np.asarray(series, dtype=np.float64),
        np.asarray(laminar_mask, dtype=np.bool_),
        np.asarray(laminar_lengths, dtype=np.int64),
    )


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
