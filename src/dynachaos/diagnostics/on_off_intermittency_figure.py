"""On-off intermittency proof-triad figure pipeline."""

from pathlib import Path

import numpy as np

from dynachaos.diagnostics.intermittency import (
    burst_amplitude_distribution,
    fit_power_law_mle,
    laminar_burst_symmetry,
    mean_laminar_scaling,
    powerlaw_alpha_ci,
    powerlaw_gof,
)
from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps.intermittency import (
    ON_OFF_SKEW_LOGISTIC_ONSET,
    on_off_skew_logistic_oracle,
)

SECTION_ID = "sec12_intermittency"
FIG_DIR = section_dir(SECTION_ID)
OUTPUT_NPZ = FIG_DIR / "on_off_intermittency.npz"
OUTPUT_PNG = FIG_DIR / "on_off_intermittency.png"
_THIS_FILE = Path(__file__).resolve()

REQUIRED_KEYS = (
    "schema_version",
    "source_file",
    "seed",
    "benchmark_eps",
    "benchmark_lambda_perp",
    "benchmark_series",
    "benchmark_laminar_mask",
    "benchmark_laminar_lengths",
    "benchmark_burst_lengths",
    "benchmark_burst_amplitudes",
    "benchmark_threshold_percentile",
    "off_time_alpha",
    "off_time_alpha_ci",
    "off_time_gof_p",
    "burst_amplitude_alpha",
    "burst_amplitude_alpha_ci",
    "burst_amplitude_gof_p",
    "scaling_eps_values",
    "lambda_abs_values",
    "mean_off_lengths",
    "mean_off_beta",
    "skew_driver_series",
    "skew_transverse_series",
)


def compute(
    output_path=OUTPUT_NPZ,
    *,
    seed=20260602,
    n_benchmark=40_000,
    powerlaw_gof_bootstrap=100,
    alpha_ci_bootstrap=200,
):
    """Compute deterministic FIG B on-off diagnostics and optionally cache them."""
    threshold_percentile = 90.0
    transient = 1_000
    benchmark_eps = ON_OFF_SKEW_LOGISTIC_ONSET - 0.01
    skew = on_off_skew_logistic_oracle(
        n_benchmark,
        x0=0.217,
        y0=1e-2,
        eps=benchmark_eps,
    )
    benchmark = np.abs(skew[transient:, 1])
    threshold = np.percentile(benchmark, threshold_percentile)
    laminar_mask = benchmark <= threshold
    symmetry = laminar_burst_symmetry(benchmark, laminar_mask)
    burst_distribution = burst_amplitude_distribution(
        benchmark,
        laminar_mask,
        min_tail=np.count_nonzero(~laminar_mask),
    )
    off_time_fit = fit_power_law_mle(symmetry.laminar_lengths)
    off_time_gof = powerlaw_gof(
        symmetry.laminar_lengths,
        fit=off_time_fit,
        n_bootstrap=powerlaw_gof_bootstrap,
        rng=seed,
    )
    off_time_ci = powerlaw_alpha_ci(
        symmetry.laminar_lengths,
        fit=off_time_fit,
        n_bootstrap=alpha_ci_bootstrap,
        rng=seed + 1,
    )
    burst_gof = powerlaw_gof(
        burst_distribution.amplitudes,
        fit=burst_distribution.power_law,
        n_bootstrap=powerlaw_gof_bootstrap,
        rng=seed + 2,
    )
    burst_ci = powerlaw_alpha_ci(
        burst_distribution.amplitudes,
        fit=burst_distribution.power_law,
        n_bootstrap=alpha_ci_bootstrap,
        rng=seed + 3,
    )

    scaling_eps_values = np.array([0.45, 0.46, 0.47, 0.48, 0.49, 0.495])
    lambda_abs_values = np.abs(np.log(2.0 * scaling_eps_values))
    mean_lengths = _mean_off_lengths(
        scaling_eps_values,
        n=100_000,
        threshold=1e-12,
    )
    scaling = mean_laminar_scaling(lambda_abs_values, mean_lengths, min_points=3)

    payload = {
        "schema_version": np.array([2], dtype=np.int64),
        "source_file": np.array([_source_file_label()]),
        "seed": np.array([seed], dtype=np.int64),
        "benchmark_eps": np.array([benchmark_eps], dtype=np.float64),
        "benchmark_lambda_perp": np.array([np.log(2.0 * benchmark_eps)], dtype=np.float64),
        "benchmark_series": benchmark.astype(np.float64),
        "benchmark_laminar_mask": laminar_mask.astype(np.bool_),
        "benchmark_laminar_lengths": symmetry.laminar_lengths.astype(np.int64),
        "benchmark_burst_lengths": symmetry.burst_lengths.astype(np.int64),
        "benchmark_burst_amplitudes": burst_distribution.amplitudes.astype(np.float64),
        "benchmark_threshold_percentile": np.array([threshold_percentile], dtype=np.float64),
        "off_time_alpha": np.array([off_time_fit.alpha], dtype=np.float64),
        "off_time_alpha_ci": off_time_ci.astype(np.float64),
        "off_time_gof_p": np.array([off_time_gof.p_value], dtype=np.float64),
        "burst_amplitude_alpha": np.array(
            [burst_distribution.power_law.alpha],
            dtype=np.float64,
        ),
        "burst_amplitude_alpha_ci": burst_ci.astype(np.float64),
        "burst_amplitude_gof_p": np.array([burst_gof.p_value], dtype=np.float64),
        "scaling_eps_values": scaling_eps_values.astype(np.float64),
        "lambda_abs_values": lambda_abs_values.astype(np.float64),
        "mean_off_lengths": mean_lengths.astype(np.float64),
        "mean_off_beta": np.array([scaling.beta], dtype=np.float64),
        "skew_driver_series": skew[transient:, 0].astype(np.float64),
        "skew_transverse_series": benchmark.astype(np.float64),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


def plot(data, output_path=OUTPUT_PNG):
    """Render the on-off proof-triad figure."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
    )

    setup()

    benchmark = np.asarray(data["benchmark_series"], dtype=np.float64)
    laminar_mask = np.asarray(data["benchmark_laminar_mask"], dtype=np.bool_)
    off_lengths = np.asarray(data["benchmark_laminar_lengths"], dtype=np.float64)
    burst_amplitudes = np.asarray(data["benchmark_burst_amplitudes"], dtype=np.float64)
    off_alpha = float(np.asarray(data["off_time_alpha"], dtype=np.float64)[0])
    off_ci = np.asarray(data["off_time_alpha_ci"], dtype=np.float64)
    off_gof_p = float(np.asarray(data["off_time_gof_p"], dtype=np.float64)[0])
    burst_alpha = float(np.asarray(data["burst_amplitude_alpha"], dtype=np.float64)[0])
    burst_ci = np.asarray(data["burst_amplitude_alpha_ci"], dtype=np.float64)
    burst_gof_p = float(np.asarray(data["burst_amplitude_gof_p"], dtype=np.float64)[0])
    lambda_abs = np.asarray(data["lambda_abs_values"], dtype=np.float64)
    mean_lengths = np.asarray(data["mean_off_lengths"], dtype=np.float64)
    skew_transverse = np.asarray(data["skew_transverse_series"], dtype=np.float64)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 2, figsize=spec.figsize, constrained_layout=True)
    ax_series, ax_off, ax_burst, ax_scaling = axes.ravel()

    window = slice(0, 5_000)
    time = np.arange(benchmark[window].size)
    ax_series.semilogy(time, benchmark[window], color=COLORS["black"], lw=0.75)
    ax_series.scatter(
        time[laminar_mask[window]],
        benchmark[window][laminar_mask[window]],
        s=2.0,
        color=COLORS["grey"],
        alpha=0.45,
        linewidths=0,
    )
    inset = ax_series.inset_axes([0.58, 0.57, 0.36, 0.34])
    inset.semilogy(skew_transverse[:2_000], color=COLORS["black"], lw=0.55)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    # Tick-free is fine, but the inset needs to say what window it shows:
    # it is the same |y_n| trace as the main panel, zoomed to n in [0, 2000].
    inset.set_title("zoom, $n\\in[0,2000]$", fontsize=spec.legend_size, pad=2.0)
    inset.set_xlabel("$n$", fontsize=spec.legend_size, labelpad=1.0)
    inset.set_ylabel("$|y_n|$", fontsize=spec.legend_size, labelpad=1.0)
    ax_series.set_title("On-off laminar epochs and bursts", loc="left")
    ax_series.set_xlabel("$n$")
    ax_series.set_ylabel("$|y_n|$")

    values, counts = np.unique(off_lengths.astype(np.int64), return_counts=True)
    probabilities = counts / np.sum(counts)
    ax_off.loglog(values, probabilities, marker="o", ms=3.0, lw=0, color=COLORS["black"])
    reference_x = np.array([values[0], np.max(values)], dtype=np.float64)
    reference_y = probabilities[0] * (reference_x / reference_x[0]) ** -1.5
    ax_off.loglog(reference_x, reference_y, color=COLORS["grey"], lw=1.0, ls="--")
    ax_off.set_title(
        f"Off-time $\\alpha$={off_alpha:.2f}, "
        f"95% CI [{off_ci[0]:.2f}, {off_ci[1]:.2f}], GoF p={off_gof_p:.2f}",
        loc="left",
    )
    ax_off.set_xlabel("off time $\\tau$")
    ax_off.set_ylabel("$P(\\tau)$")

    bins = np.geomspace(np.min(burst_amplitudes), np.max(burst_amplitudes), 80)
    density, edges = np.histogram(burst_amplitudes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax_burst.loglog(centers, density, marker="o", ms=2.5, lw=0, color=COLORS["black"])
    finite = density > 0.0
    reference_x = np.array([centers[finite][0], centers[finite][-1]], dtype=np.float64)
    reference_y = density[finite][0] * (reference_x / reference_x[0]) ** -1.0
    ax_burst.loglog(reference_x, reference_y, color=COLORS["grey"], lw=1.0, ls="--")
    ax_burst.set_title(
        f"Burst $\\alpha$={burst_alpha:.2f}, "
        f"95% CI [{burst_ci[0]:.2f}, {burst_ci[1]:.2f}], GoF p={burst_gof_p:.2f}",
        loc="left",
    )
    ax_burst.set_xlabel("burst amplitude $|y|$")
    ax_burst.set_ylabel("density")

    ax_scaling.loglog(
        lambda_abs,
        mean_lengths,
        marker="o",
        ms=4.0,
        color=COLORS["black"],
        lw=1.0,
    )
    scale_x = np.array([np.min(lambda_abs), np.max(lambda_abs)], dtype=np.float64)
    scale_y = mean_lengths[0] * (scale_x / lambda_abs[0]) ** -1.0
    ax_scaling.loglog(scale_x, scale_y, color=COLORS["grey"], lw=1.0, ls="--")
    ax_scaling.set_title("$\\langle \\tau \\rangle \\sim |\\lambda_\\perp|^{-1}$", loc="left")
    ax_scaling.set_xlabel("$|\\lambda_\\perp|$")
    ax_scaling.set_ylabel("$\\langle \\tau \\rangle$")

    # Panels (a), (c), (d) are monotone-decreasing curves that start right at
    # the default upper-left tag position, stamping the label on top of the
    # y-axis spine and the leading data points. Each curve leaves the
    # opposite (lower-left) corner clear.
    label_locs = {"a": "lower left", "c": "lower left", "d": "lower left"}
    for label, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        panel_label(ax, f"({label})", loc=label_locs.get(label, "upper left"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        apply_axes_polish(ax, kind="grid")

    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main():
    """Load or compute the on-off cache, then render the figure."""
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

    plot(data)


def _mean_off_lengths(eps_values, *, n, threshold):
    means = []
    for eps in eps_values:
        lengths = []
        for x0 in np.linspace(0.123, 0.923, 40):
            series = np.abs(
                on_off_skew_logistic_oracle(
                    n,
                    x0=float(x0),
                    y0=1e-2,
                    eps=float(eps),
                )[:, 1]
            )
            hits = np.flatnonzero(series < threshold)
            lengths.append(int(hits[0] + 1) if hits.size else series.size)
        means.append(np.mean(lengths))
    return np.asarray(means, dtype=np.float64)


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
