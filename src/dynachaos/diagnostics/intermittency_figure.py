"""Type-I intermittency proof-triad figure pipeline.

This module supersedes the former mixed intermittency diagnostics summary.  It
now builds the FIG A Type-I demonstration: logistic period-3 tangent-channel
geometry, the laminar-length tail, the normal-form mean-laminar scaling law,
and the Lorenz rho=166.2 near-diagonal return-map channel.
"""

import warnings
from pathlib import Path

import numpy as np

from dynachaos.diagnostics.intermittency import (
    compare_powerlaw_exponential,
    detect_laminar_phases,
    mean_laminar_scaling,
    powerlaw_alpha_ci,
    powerlaw_gof,
    return_map_reconstruction,
)
from dynachaos.io.paths import safe_load, section_dir
from dynachaos.maps.intermittency import (
    LOGISTIC_TYPE_I_ONSET,
    LORENZ_INTERMITTENCY_RHO,
    logistic_type_i_oracle,
    lorenz_1662_oracle,
)

SECTION_ID = "sec12_intermittency"
FIG_DIR = section_dir(SECTION_ID)
OUTPUT_NPZ = FIG_DIR / "type_i_intermittency.npz"
OUTPUT_PNG = FIG_DIR / "type_i_intermittency.png"
_THIS_FILE = Path(__file__).resolve()

REQUIRED_KEYS = (
    "schema_version",
    "source_file",
    "seed",
    "logistic_mechanism_r",
    "logistic_tail_r",
    "logistic_period",
    "logistic_series",
    "logistic_laminar_mask",
    "logistic_laminar_lengths",
    "logistic_laminar_percentile",
    "logistic_f3_return_points",
    "logistic_f3_channel_points",
    "logistic_f3_channel_slope",
    "type_i_tail_alpha",
    "type_i_tail_alpha_ci",
    "type_i_tail_gof_p",
    "type_i_vuong_z",
    "normal_form_eps",
    "normal_form_mean_lengths",
    "normal_form_beta",
    "lorenz_rho",
    "lorenz_time",
    "lorenz_observable",
    "lorenz_return_points",
    "lorenz_channel_points",
    "lorenz_channel_slope",
)


def compute(
    output_path=OUTPUT_NPZ,
    *,
    seed=20260601,
    n_logistic=200_000,
    powerlaw_gof_bootstrap=100,
    alpha_ci_bootstrap=200,
):
    """Compute deterministic FIG A Type-I diagnostics and optionally cache them."""
    logistic_tail = logistic_type_i_oracle(n_logistic, x0=0.2)
    logistic_mask, logistic_lengths = detect_laminar_phases(
        logistic_tail,
        method="period",
        period=3,
        percentile=70.0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        comparison = compare_powerlaw_exponential(logistic_lengths)
        tail_gof = powerlaw_gof(
            logistic_lengths,
            fit=comparison.power_law,
            n_bootstrap=powerlaw_gof_bootstrap,
            rng=seed,
        )
        tail_ci = powerlaw_alpha_ci(
            logistic_lengths,
            fit=comparison.power_law,
            n_bootstrap=alpha_ci_bootstrap,
            rng=seed + 1,
        )

    f3_return_points, f3_derivative = _logistic_f3_grid()
    f3_channel_points = _logistic_f3_tangent_channels(
        f3_return_points,
        f3_derivative,
    )
    f3_channel_slope = np.polyfit(
        f3_channel_points[:, 0],
        f3_channel_points[:, 1],
        deg=1,
    )[0]

    eps_values = np.geomspace(1e-5, 1e-3, 8)
    mean_lengths = _normal_form_escape_lengths(eps_values)
    scaling = mean_laminar_scaling(eps_values, mean_lengths, min_points=3)

    lorenz = lorenz_1662_oracle(t_span=(0.0, 80.0), dt=0.01, t_transient=20.0)
    lorenz_observable = lorenz[:, 1].astype(np.float64)
    lorenz_time = np.arange(lorenz_observable.size, dtype=np.float64) * 0.01 + 20.0
    lorenz_return = return_map_reconstruction(
        lorenz_observable,
        fs=100.0,
        channel_percentile=30.0,
        min_channel_points=5,
    )
    lorenz_return_points = lorenz_return.extrema.points.astype(np.float64)

    payload = {
        "schema_version": np.array([6], dtype=np.int64),
        "source_file": np.array([_source_file_label()]),
        "seed": np.array([seed], dtype=np.int64),
        "logistic_mechanism_r": np.array([LOGISTIC_TYPE_I_ONSET], dtype=np.float64),
        "logistic_tail_r": np.array([LOGISTIC_TYPE_I_ONSET - 1e-4], dtype=np.float64),
        "logistic_period": np.array([3], dtype=np.int64),
        "logistic_series": logistic_tail.astype(np.float64),
        "logistic_laminar_mask": logistic_mask.astype(np.bool_),
        "logistic_laminar_lengths": logistic_lengths.astype(np.int64),
        "logistic_laminar_percentile": np.array([70.0], dtype=np.float64),
        "logistic_f3_return_points": f3_return_points,
        "logistic_f3_channel_points": f3_channel_points,
        "logistic_f3_channel_slope": np.array([f3_channel_slope], dtype=np.float64),
        "type_i_tail_alpha": np.array(
            [comparison.power_law.alpha],
            dtype=np.float64,
        ),
        "type_i_tail_alpha_ci": tail_ci.astype(np.float64),
        "type_i_tail_gof_p": np.array([tail_gof.p_value], dtype=np.float64),
        "type_i_vuong_z": np.array([comparison.z], dtype=np.float64),
        "normal_form_eps": eps_values.astype(np.float64),
        "normal_form_mean_lengths": mean_lengths.astype(np.float64),
        "normal_form_beta": np.array([scaling.beta], dtype=np.float64),
        "lorenz_rho": np.array([LORENZ_INTERMITTENCY_RHO], dtype=np.float64),
        "lorenz_time": lorenz_time,
        "lorenz_observable": lorenz_observable,
        "lorenz_return_points": lorenz_return_points,
        "lorenz_channel_points": lorenz_return.tangent_channel.points.astype(np.float64),
        "lorenz_channel_slope": np.array(
            [lorenz_return.tangent_channel.slope],
            dtype=np.float64,
        ),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


def plot(data, output_path=OUTPUT_PNG):
    """Render the Type-I proof-triad figure."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
    )

    setup()

    f3_points = np.asarray(data["logistic_f3_return_points"], dtype=np.float64)
    f3_channel = np.asarray(data["logistic_f3_channel_points"], dtype=np.float64)
    lengths = np.asarray(data["logistic_laminar_lengths"], dtype=np.float64)
    tail_alpha = float(np.asarray(data["type_i_tail_alpha"], dtype=np.float64)[0])
    tail_ci = np.asarray(data["type_i_tail_alpha_ci"], dtype=np.float64)
    tail_gof_p = float(np.asarray(data["type_i_tail_gof_p"], dtype=np.float64)[0])
    eps = np.asarray(data["normal_form_eps"], dtype=np.float64)
    mean_lengths = np.asarray(data["normal_form_mean_lengths"], dtype=np.float64)
    lorenz_time = np.asarray(data["lorenz_time"], dtype=np.float64)
    lorenz_observable = np.asarray(data["lorenz_observable"], dtype=np.float64)
    lorenz_return = np.asarray(data["lorenz_return_points"], dtype=np.float64)
    lorenz_channel = np.asarray(data["lorenz_channel_points"], dtype=np.float64)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 2, figsize=spec.figsize, constrained_layout=True)
    ax_return, ax_tail, ax_scaling, ax_lorenz = axes.ravel()

    sample = _even_sample(f3_points, 20_000)
    ax_return.scatter(
        sample[:, 0],
        sample[:, 1],
        s=2.0,
        color=COLORS["black"],
        alpha=0.18,
        linewidths=0,
    )
    ax_return.scatter(
        f3_channel[:, 0],
        f3_channel[:, 1],
        s=5.0,
        color=COLORS["red"],
        alpha=0.8,
        linewidths=0,
    )
    bounds = np.array([np.min(f3_points), np.max(f3_points)], dtype=np.float64)
    ax_return.plot(bounds, bounds, color=COLORS["black"], lw=0.8, ls="--")
    ax_return.set_title("Logistic $f^3$ tangent channels", loc="left")
    ax_return.set_xlabel("$x_n$")
    ax_return.set_ylabel("$x_{n+3}$")

    values, counts = np.unique(lengths.astype(np.int64), return_counts=True)
    probabilities = counts / np.sum(counts)
    ax_tail.loglog(
        values,
        probabilities,
        marker="o",
        ms=3.0,
        lw=0,
        color=COLORS["black"],
        alpha=0.8,
    )
    tail = values >= 3
    reference_x = np.array([values[tail][0], np.max(values)], dtype=np.float64)
    reference_y = probabilities[tail][0] * (reference_x / reference_x[0]) ** -1.5
    ax_tail.loglog(reference_x, reference_y, color=COLORS["grey"], lw=1.0, ls="--")
    ax_tail.set_title(
        f"Laminar tail $\\alpha$={tail_alpha:.2f}, "
        f"95% CI [{tail_ci[0]:.2f}, {tail_ci[1]:.2f}], GoF p={tail_gof_p:.2f}",
        loc="left",
    )
    ax_tail.set_xlabel("laminar length $\\ell$")
    ax_tail.set_ylabel("$P(\\ell)$")

    ax_scaling.loglog(
        eps,
        mean_lengths,
        marker="o",
        ms=4.0,
        color=COLORS["black"],
        lw=1.0,
    )
    scale_x = np.array([np.min(eps), np.max(eps)], dtype=np.float64)
    scale_y = mean_lengths[0] * (scale_x / eps[0]) ** -0.5
    ax_scaling.loglog(scale_x, scale_y, color=COLORS["grey"], lw=1.0, ls="--")
    ax_scaling.set_title(
        "Normal-form $\\langle \\ell \\rangle \\sim \\epsilon^{-1/2}$",
        loc="left",
    )
    ax_scaling.set_xlabel("$\\epsilon$")
    ax_scaling.set_ylabel("$\\langle \\ell \\rangle$")

    lorenz_sample = _even_sample(lorenz_return, 4_000)
    ax_lorenz.scatter(
        lorenz_sample[:, 0],
        lorenz_sample[:, 1],
        s=5.0,
        color=COLORS["black"],
        alpha=0.45,
        linewidths=0,
    )
    ax_lorenz.scatter(
        lorenz_channel[:, 0],
        lorenz_channel[:, 1],
        s=12.0,
        color=COLORS["red"],
        alpha=0.9,
        linewidths=0,
    )
    lorenz_bounds = np.array([np.min(lorenz_return), np.max(lorenz_return)], dtype=np.float64)
    ax_lorenz.plot(lorenz_bounds, lorenz_bounds, color=COLORS["black"], lw=0.8, ls="--")
    # Moved out of the upper-left corner: it used to sit directly over the
    # panel letter (clipping the closing parenthesis) and over part of the
    # negative-y_k branch of the scatter. The lower-right of this panel is
    # comparatively empty.
    inset = ax_lorenz.inset_axes([0.56, 0.08, 0.40, 0.32])
    inset.plot(
        lorenz_time[:1_200],
        lorenz_observable[:1_200],
        color=COLORS["black"],
        lw=0.6,
    )
    inset.set_xlabel("time", fontsize=spec.legend_size - 1.0, labelpad=1.0)
    inset.set_ylabel("$y$", fontsize=spec.legend_size - 1.0, labelpad=1.0)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    ax_lorenz.set_title("Lorenz $\\rho=166.2$ $y$-return channel", loc="left")
    ax_lorenz.set_xlabel("$y_k$")
    ax_lorenz.set_ylabel("$y_{k+1}$")

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
    """Load or compute the Type-I cache, then render the figure."""
    try:
        data = safe_load(OUTPUT_NPZ)
        missing = tuple(key for key in REQUIRED_KEYS if key not in data.files)
        stale_schema = not missing and int(np.asarray(data["schema_version"])[0]) != 6
        if missing or stale_schema:
            data.close()
            if missing:
                reason = "missing keys (" + ", ".join(missing) + ")"
            else:
                reason = "stale schema"
            print(f"Cache {OUTPUT_NPZ} {reason}; recomputing...")
            data = compute()
    except FileNotFoundError:
        data = compute()

    plot(data)


def _normal_form_escape_lengths(eps_values):
    lengths = []
    for eps in eps_values:
        x = 0.0
        length = 0
        while x < 1.0:
            x = x + float(eps) + x * x
            length += 1
        lengths.append(length)
    return np.asarray(lengths, dtype=np.float64)


def _logistic_f3_grid(n_points=4_000):
    x = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    y = x.copy()
    derivative = np.ones_like(x)
    for _ in range(3):
        derivative *= LOGISTIC_TYPE_I_ONSET * (1.0 - 2.0 * y)
        y = LOGISTIC_TYPE_I_ONSET * y * (1.0 - y)
    return np.column_stack((x, y)), derivative


def _logistic_f3_tangent_channels(points, derivative):
    points = np.asarray(points, dtype=np.float64)
    derivative = np.asarray(derivative, dtype=np.float64)
    diagonal_distance = np.abs(points[:, 1] - points[:, 0])
    mask = (points[:, 0] > 0.05) & (diagonal_distance <= 0.01) & (np.abs(derivative - 1.0) < 1.0)
    channel = points[mask]
    if channel.shape[0] < 20:
        msg = "logistic f^3 grid did not resolve the tangent channels"
        raise RuntimeError(msg)
    return channel


def _even_sample(points, max_points):
    points = np.asarray(points)
    if points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices]


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
