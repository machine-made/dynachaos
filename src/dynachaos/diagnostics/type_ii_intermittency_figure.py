"""Type-II intermittency normal-form proof figure pipeline."""

from pathlib import Path

import numpy as np

from dynachaos.diagnostics.intermittency import fit_power_law_mle, powerlaw_alpha_ci, powerlaw_gof
from dynachaos.io.paths import safe_load, section_dir

SECTION_ID = "sec12_intermittency"
FIG_DIR = section_dir(SECTION_ID)
OUTPUT_NPZ = FIG_DIR / "type_ii_intermittency.npz"
OUTPUT_PNG = FIG_DIR / "type_ii_intermittency.png"
_THIS_FILE = Path(__file__).resolve()

REQUIRED_KEYS = (
    "schema_version",
    "source_file",
    "seed",
    "eps",
    "a",
    "theta",
    "escape_threshold",
    "spiral_orbit",
    "spiral_radius",
    "spiral_escape_index",
    "reinjection_radii",
    "laminar_lengths",
    "laminar_histogram_edges",
    "laminar_histogram_density",
    "laminar_tail_alpha",
    "laminar_tail_alpha_ci",
    "laminar_tail_gof_p",
    "exponential_rate",
    "exponential_intercept",
    "exponential_rvalue",
)

CAPTION_NOTE = (
    "Normal-form Type-II demonstration; clean physical exemplars are scarce "
    "(p-n diode, forced jet)."
)


def compute(
    output_path=OUTPUT_NPZ,
    *,
    seed=20260602,
    n_reinjections=1_200,
    powerlaw_gof_bootstrap=100,
    alpha_ci_bootstrap=200,
):
    """Compute deterministic FIG D Type-II diagnostics and optionally cache them."""
    eps = 2e-3
    a = 1.0
    theta = 0.17
    escape_threshold = 0.35
    rng = np.random.default_rng(seed)

    spiral_orbit, spiral_escape_index = _bounded_orbit(
        x0=2e-3,
        y0=0.0,
        eps=eps,
        a=a,
        theta=theta,
        escape_threshold=escape_threshold,
        max_steps=2_000,
    )
    spiral_radius = np.linalg.norm(spiral_orbit, axis=1)

    reinjection_radii = rng.uniform(1e-3, 2e-2, n_reinjections)
    laminar_lengths = _escape_lengths(
        reinjection_radii,
        eps=eps,
        a=a,
        theta=theta,
        escape_threshold=escape_threshold,
    )
    density, edges = np.histogram(laminar_lengths, bins=48, density=True)
    fit_mask = density > 0.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    slope, intercept = np.polyfit(centers[fit_mask], np.log(density[fit_mask]), deg=1)
    rvalue = np.corrcoef(centers[fit_mask], np.log(density[fit_mask]))[0, 1]
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

    payload = {
        "schema_version": np.array([2], dtype=np.int64),
        "source_file": np.array([_source_file_label()]),
        "seed": np.array([seed], dtype=np.int64),
        "eps": np.array([eps], dtype=np.float64),
        "a": np.array([a], dtype=np.float64),
        "theta": np.array([theta], dtype=np.float64),
        "escape_threshold": np.array([escape_threshold], dtype=np.float64),
        "spiral_orbit": spiral_orbit.astype(np.float64),
        "spiral_radius": spiral_radius.astype(np.float64),
        "spiral_escape_index": np.array([spiral_escape_index], dtype=np.int64),
        "reinjection_radii": reinjection_radii.astype(np.float64),
        "laminar_lengths": laminar_lengths.astype(np.int64),
        "laminar_histogram_edges": edges.astype(np.float64),
        "laminar_histogram_density": density.astype(np.float64),
        "laminar_tail_alpha": np.array([tail_fit.alpha], dtype=np.float64),
        "laminar_tail_alpha_ci": tail_ci.astype(np.float64),
        "laminar_tail_gof_p": np.array([tail_gof.p_value], dtype=np.float64),
        "exponential_rate": np.array([-slope], dtype=np.float64),
        "exponential_intercept": np.array([intercept], dtype=np.float64),
        "exponential_rvalue": np.array([rvalue], dtype=np.float64),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"Saved {output_path}")
    return payload


def plot(data, output_path=OUTPUT_PNG):
    """Render the Type-II normal-form proof figure."""
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        panel_label,
        setup,
    )

    setup()

    orbit = np.asarray(data["spiral_orbit"], dtype=np.float64)
    radius = np.asarray(data["spiral_radius"], dtype=np.float64)
    lengths = np.asarray(data["laminar_lengths"], dtype=np.int64)
    edges = np.asarray(data["laminar_histogram_edges"], dtype=np.float64)
    density = np.asarray(data["laminar_histogram_density"], dtype=np.float64)
    exp_rate = float(np.asarray(data["exponential_rate"], dtype=np.float64)[0])
    exp_intercept = float(np.asarray(data["exponential_intercept"], dtype=np.float64)[0])
    tail_alpha = float(np.asarray(data["laminar_tail_alpha"], dtype=np.float64)[0])
    tail_ci = np.asarray(data["laminar_tail_alpha_ci"], dtype=np.float64)
    tail_gof_p = float(np.asarray(data["laminar_tail_gof_p"], dtype=np.float64)[0])
    escape_threshold = float(np.asarray(data["escape_threshold"], dtype=np.float64)[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = figure_spec("grid")
    fig = plt.figure(figsize=spec.figsize, constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["spiral", "radius"], ["hist", "survival"], ["caption", "caption"]],
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.12]},
    )
    ax_spiral = axes["spiral"]
    ax_radius = axes["radius"]
    ax_hist = axes["hist"]
    ax_survival = axes["survival"]
    ax_caption = axes["caption"]

    sample = _even_sample(orbit, 1_600)
    ax_spiral.plot(sample[:, 0], sample[:, 1], color=COLORS["black"], lw=0.75)
    ax_spiral.scatter(
        sample[::32, 0],
        sample[::32, 1],
        s=6.0,
        color=COLORS["grey"],
        alpha=0.65,
        linewidths=0,
    )
    ax_spiral.set_aspect("equal", adjustable="box")
    ax_spiral.set_title("Type-II Hopf normal-form spiral", loc="left")
    ax_spiral.set_xlabel("$x_n$")
    ax_spiral.set_ylabel("$y_n$")

    time = np.arange(radius.size)
    ax_radius.semilogy(time, radius, color=COLORS["black"], lw=0.9)
    ax_radius.axhline(escape_threshold, color=COLORS["black"], lw=0.8, ls="--")
    ax_radius.set_title("Radial escape from the fixed point", loc="left")
    ax_radius.set_xlabel("$n$")
    ax_radius.set_ylabel("$r_n$")

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    ax_hist.bar(centers, density, width=width, color=COLORS["black"], alpha=0.7, align="center")
    fit_x = np.linspace(np.min(centers), np.max(centers), 200)
    fit_y = np.exp(exp_intercept - exp_rate * fit_x)
    ax_hist.plot(fit_x, fit_y, color=COLORS["grey"], lw=1.0, ls="--")
    ax_hist.set_title("Laminar lengths: exponential envelope", loc="left")
    ax_hist.set_xlabel("laminar length $\\ell$")
    ax_hist.set_ylabel("density")

    sorted_lengths = np.sort(lengths)
    survival = 1.0 - np.arange(sorted_lengths.size, dtype=np.float64) / sorted_lengths.size
    ax_survival.semilogy(sorted_lengths, survival, color=COLORS["black"], lw=1.1)
    ax_survival.set_title(
        f"Power-law check $\\alpha$={tail_alpha:.2f}, "
        f"95% CI [{tail_ci[0]:.2f}, {tail_ci[1]:.2f}], GoF p={tail_gof_p:.2f}",
        loc="left",
    )
    ax_survival.set_xlabel("laminar length $\\ell$")
    ax_survival.set_ylabel("$P(L \\geq \\ell)$")

    # Panel (c)'s tall leading bar plus a wide y-tick label ("0.0025") leave
    # no margin for the default upper-left tag, which then renders with its
    # opening parenthesis clipped by the left spine. The histogram's upper
    # right is empty (density falls off with laminar length), so put it there.
    label_locs = {"c": "upper right"}
    for label, ax in zip(
        ("a", "b", "c", "d"),
        (ax_spiral, ax_radius, ax_hist, ax_survival),
        strict=True,
    ):
        panel_label(ax, f"({label})", loc=label_locs.get(label, "upper left"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        apply_axes_polish(ax, kind="grid")

    ax_caption.axis("off")
    ax_caption.text(
        0.5,
        0.5,
        CAPTION_NOTE,
        ha="center",
        va="center",
        fontsize=spec.tick_size,
        color=COLORS["black"],
    )
    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main():
    """Load or compute the Type-II cache, then render the figure."""
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


def _bounded_orbit(*, x0, y0, eps, a, theta, escape_threshold, max_steps):
    x = float(x0)
    y = float(y0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    orbit = []
    for step in range(1, max_steps + 1):
        r2 = x * x + y * y
        growth = 1.0 + eps + a * r2
        xr = cos_theta * x - sin_theta * y
        yr = sin_theta * x + cos_theta * y
        x = growth * xr
        y = growth * yr
        orbit.append((x, y))
        if np.hypot(x, y) >= escape_threshold:
            return np.asarray(orbit, dtype=np.float64), step
    raise RuntimeError("Type-II spiral orbit reached max_steps before escape")


def _escape_lengths(reinjection_radii, *, eps, a, theta, escape_threshold, max_steps=5_000):
    lengths = []
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    for r0 in reinjection_radii:
        x = float(r0)
        y = 0.0
        for length in range(1, max_steps + 1):
            r2 = x * x + y * y
            growth = 1.0 + eps + a * r2
            xr = cos_theta * x - sin_theta * y
            yr = sin_theta * x + cos_theta * y
            x = growth * xr
            y = growth * yr
            if np.hypot(x, y) >= escape_threshold:
                lengths.append(length)
                break
        else:
            raise RuntimeError("Type-II laminar episode reached max_steps before escape")
    return np.asarray(lengths, dtype=np.int64)


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
