#!/usr/bin/env python3
"""
benchmark_gp_vs_multifractal: Compare GP and multifractal D2 estimates.

For synthetic measures with known reference dimensions, this script compares:
1) Accuracy: |D2_est - D2_ref|
2) Performance: wall-clock runtime

GP estimate:
    dynachaos.diagnostics.correlation_dimension on sampled point clouds.

Multifractal estimate:
    dynachaos.diagnostics.multifractal_spectrum on a binned measure field,
    extracting D_q at q=2.

OUTPUTS: benchmark_gp_vs_multifractal.npz, benchmark_gp_vs_multifractal.png
USAGE:   python examples/benchmark_gp_vs_multifractal.py
         rm examples/benchmark_gp_vs_multifractal.npz
         python examples/benchmark_gp_vs_multifractal.py
"""

import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")

from _pipeline import load_jsonc  # noqa: E402

CONFIG = load_jsonc(SCRIPT.with_suffix(".jsonc"))


def _build_reference_field(kind, grid_size, carpet_level):
    if kind == "square":
        return np.ones((grid_size, grid_size), dtype=np.float64)
    if kind == "line":
        field = np.zeros((grid_size, grid_size), dtype=np.float64)
        field[grid_size // 2, :] = 1.0
        return field
    if kind == "carpet":
        pattern = np.array(
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float64,
        )
        field = np.array([[1.0]], dtype=np.float64)
        for _ in range(carpet_level):
            field = np.kron(field, pattern)
        if field.shape != (grid_size, grid_size):
            raise ValueError(
                f"carpet field shape {field.shape} != grid_size ({grid_size}, {grid_size})"
            )
        return field
    raise ValueError(f"unknown scenario kind: {kind}")


def _sample_points_from_field(field, n_samples, rng):
    ny, nx = field.shape
    flat = field.ravel()
    total = float(flat.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("field must have positive finite total mass")
    probs = flat / total
    idx = rng.choice(flat.size, size=n_samples, replace=True, p=probs)
    iy = idx // nx
    ix = idx % nx

    # Jitter each sample uniformly within the selected pixel.
    x = (ix + rng.random(n_samples)) / float(nx)
    y = (iy + rng.random(n_samples)) / float(ny)
    return np.column_stack([x, y]).astype(np.float64, copy=False)


def _points_to_field(points, grid_size):
    field = np.zeros((grid_size, grid_size), dtype=np.float64)
    ix = np.floor(points[:, 0] * grid_size).astype(np.int64)
    iy = np.floor(points[:, 1] * grid_size).astype(np.int64)
    ix = np.clip(ix, 0, grid_size - 1)
    iy = np.clip(iy, 0, grid_size - 1)
    np.add.at(field, (iy, ix), 1.0)
    return field


def _estimate_gp_d2(points, n_r, theiler_window):
    from dynachaos.diagnostics import correlation_dimension

    d2, _, _, _, _ = correlation_dimension(
        points,
        n_r=n_r,
        theiler_window=theiler_window,
    )
    return float(d2)


def _estimate_mf_d2(field, box_sizes, q_values):
    from dynachaos.diagnostics import multifractal_spectrum

    spec = multifractal_spectrum(field, box_sizes=box_sizes, q_values=q_values)
    q = np.asarray(spec["q"], dtype=np.float64)
    dq = np.asarray(spec["Dq"], dtype=np.float64)
    idx = np.where(np.isclose(q, 2.0))[0]
    if idx.size == 0:
        raise ValueError("q_values must include q=2.0")
    return float(dq[idx[0]])


def compute():
    seed = int(CONFIG["seed"])
    n_samples_sweep = np.asarray(CONFIG["n_samples_sweep"], dtype=np.int64)
    repeats_accuracy = int(CONFIG["repeats_accuracy"])
    repeats_timing = int(CONFIG["repeats_timing"])
    gp_n_r = int(CONFIG["gp_n_r"])
    gp_theiler_window = int(CONFIG["gp_theiler_window"])
    grid_size = int(CONFIG["grid_size"])
    box_sizes = np.asarray(CONFIG["box_sizes"], dtype=np.int64)
    q_values = np.asarray(CONFIG["q_values"], dtype=np.float64)
    carpet_level = int(CONFIG["carpet_level"])
    scenarios = list(CONFIG["scenarios"])

    n_s = len(scenarios)
    n_n = len(n_samples_sweep)

    d2_gp = np.full((n_s, n_n), np.nan, dtype=np.float64)
    d2_mf = np.full((n_s, n_n), np.nan, dtype=np.float64)
    err_gp = np.full((n_s, n_n), np.nan, dtype=np.float64)
    err_mf = np.full((n_s, n_n), np.nan, dtype=np.float64)
    t_gp = np.full((n_s, n_n), np.nan, dtype=np.float64)
    t_mf_core = np.full((n_s, n_n), np.nan, dtype=np.float64)
    t_mf_total = np.full((n_s, n_n), np.nan, dtype=np.float64)
    t_binning = np.full((n_s, n_n), np.nan, dtype=np.float64)

    scenario_names = []
    ref_d2 = np.full(n_s, np.nan, dtype=np.float64)

    for si, scenario in enumerate(scenarios):
        name = str(scenario["name"])
        kind = str(scenario["kind"])
        ref = float(scenario["ref_D2"])
        scenario_names.append(name)
        ref_d2[si] = ref

        field_ref = _build_reference_field(
            kind=kind,
            grid_size=grid_size,
            carpet_level=carpet_level,
        )

        print(f"[{name}] ref_D2={ref:.6f}")
        for ni, n_samples in enumerate(n_samples_sweep):
            gp_vals = []
            mf_vals = []
            gp_times = []
            mf_core_times = []
            mf_total_times = []
            binning_times = []

            for rep in range(repeats_accuracy):
                rng = np.random.default_rng(seed + 1000 * si + 100 * ni + rep)
                points = _sample_points_from_field(field_ref, int(n_samples), rng)
                gp = _estimate_gp_d2(points, n_r=gp_n_r, theiler_window=gp_theiler_window)
                field_emp = _points_to_field(points, grid_size=grid_size)
                mf = _estimate_mf_d2(field_emp, box_sizes=box_sizes, q_values=q_values)
                gp_vals.append(gp)
                mf_vals.append(mf)

            for rep in range(repeats_timing):
                rng = np.random.default_rng(seed + 20000 + 1000 * si + 100 * ni + rep)
                points = _sample_points_from_field(field_ref, int(n_samples), rng)

                t0 = time.perf_counter()
                _estimate_gp_d2(points, n_r=gp_n_r, theiler_window=gp_theiler_window)
                gp_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                field_emp = _points_to_field(points, grid_size=grid_size)
                t_bin = time.perf_counter() - t0

                t0 = time.perf_counter()
                _estimate_mf_d2(field_emp, box_sizes=box_sizes, q_values=q_values)
                t_core = time.perf_counter() - t0

                binning_times.append(t_bin)
                mf_core_times.append(t_core)
                mf_total_times.append(t_bin + t_core)

            d2_gp[si, ni] = float(np.nanmedian(gp_vals))
            d2_mf[si, ni] = float(np.nanmedian(mf_vals))
            err_gp[si, ni] = abs(d2_gp[si, ni] - ref)
            err_mf[si, ni] = abs(d2_mf[si, ni] - ref)
            t_gp[si, ni] = float(np.nanmedian(gp_times))
            t_mf_core[si, ni] = float(np.nanmedian(mf_core_times))
            t_mf_total[si, ni] = float(np.nanmedian(mf_total_times))
            t_binning[si, ni] = float(np.nanmedian(binning_times))

            ratio_total = (
                t_gp[si, ni] / t_mf_total[si, ni] if t_mf_total[si, ni] > 0.0 else np.nan
            )
            ratio_core = (
                t_gp[si, ni] / t_mf_core[si, ni] if t_mf_core[si, ni] > 0.0 else np.nan
            )
            print(
                f"  N={int(n_samples):4d} | "
                f"GP D2={d2_gp[si, ni]:.3f}, err={err_gp[si, ni]:.3f}, t={t_gp[si, ni]:.4f}s | "
                f"MF D2={d2_mf[si, ni]:.3f}, err={err_mf[si, ni]:.3f}, "
                f"t_core={t_mf_core[si, ni]:.4f}s, t_total={t_mf_total[si, ni]:.4f}s | "
                f"speedup_core={ratio_core:.2f}x, speedup_total={ratio_total:.2f}x"
            )

    np.savez_compressed(
        OUTPUT_NPZ,
        scenario_names=np.array(scenario_names),
        ref_d2=ref_d2,
        n_samples_sweep=n_samples_sweep,
        d2_gp=d2_gp,
        d2_mf=d2_mf,
        err_gp=err_gp,
        err_mf=err_mf,
        t_gp=t_gp,
        t_mf=t_mf_total,
        t_mf_core=t_mf_core,
        t_mf_total=t_mf_total,
        t_binning=t_binning,
    )
    print(f"Saved {OUTPUT_NPZ}")


def plot(data):
    import matplotlib.pyplot as plt

    from dynachaos.utils.style import (
        COLORS,
        apply_axes_polish,
        figure_spec,
        finalize_legend,
        setup,
    )

    setup()
    spec = figure_spec("grid")

    scenario_names = [str(s) for s in data["scenario_names"]]
    ref_d2 = np.asarray(data["ref_d2"], dtype=np.float64)
    n_samples = np.asarray(data["n_samples_sweep"], dtype=np.int64)
    d2_gp = np.asarray(data["d2_gp"], dtype=np.float64)
    d2_mf = np.asarray(data["d2_mf"], dtype=np.float64)
    err_gp = np.asarray(data["err_gp"], dtype=np.float64)
    err_mf = np.asarray(data["err_mf"], dtype=np.float64)
    t_gp = np.asarray(data["t_gp"], dtype=np.float64)
    if "t_mf_total" in data:
        t_mf_total = np.asarray(data["t_mf_total"], dtype=np.float64)
    else:
        t_mf_total = np.asarray(data["t_mf"], dtype=np.float64)
    if "t_mf_core" in data:
        t_mf_core = np.asarray(data["t_mf_core"], dtype=np.float64)
    else:
        t_mf_core = t_mf_total

    fig, axes = plt.subplots(2, 2, figsize=(spec.figsize[0], spec.figsize[1] + 0.6))
    fig.subplots_adjust(hspace=0.45, wspace=0.34)
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    color_cycle = [COLORS["black"], COLORS["blue"], COLORS["red"], COLORS["green"]]

    for si, name in enumerate(scenario_names):
        c = color_cycle[si % len(color_cycle)]
        ax_a.plot(
            n_samples,
            err_gp[si],
            color=c,
            ls="-",
            marker="o",
            ms=3,
            label=f"{name} (GP)",
        )
        ax_a.plot(
            n_samples,
            err_mf[si],
            color=c,
            ls="--",
            marker="s",
            ms=3,
            label=f"{name} (MF)",
        )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Sample size $N$")
    ax_a.set_ylabel(r"Absolute error $|D_2 - D_{2,ref}|$")
    ax_a.set_title("(a) Accuracy", loc="left")
    apply_axes_polish(ax_a, kind="grid", title_loc="left")
    finalize_legend(ax_a, kind="grid", ncol=2)

    median_t_gp = np.nanmedian(t_gp, axis=0)
    median_t_mf_total = np.nanmedian(t_mf_total, axis=0)
    median_t_mf_core = np.nanmedian(t_mf_core, axis=0)
    ax_b.loglog(n_samples, median_t_gp, color=COLORS["black"], marker="o", ms=4, label="GP")
    ax_b.loglog(
        n_samples,
        median_t_mf_total,
        color=COLORS["red"],
        marker="s",
        ms=4,
        label="MF (total)",
    )
    ax_b.loglog(
        n_samples,
        median_t_mf_core,
        color=COLORS["red"],
        marker="^",
        ms=3,
        ls="--",
        label="MF (core only)",
    )
    ax_b.set_xlabel("Sample size $N$")
    ax_b.set_ylabel("Median wall-clock (s)")
    ax_b.set_title("(b) Runtime (median across scenarios)", loc="left")
    apply_axes_polish(ax_b, kind="grid", title_loc="left")
    finalize_legend(ax_b, kind="grid")

    median_err_gp = np.nanmedian(err_gp, axis=0)
    median_err_mf = np.nanmedian(err_mf, axis=0)
    ax_c.semilogx(
        n_samples,
        median_err_gp,
        color=COLORS["black"],
        marker="o",
        ms=4,
        label="GP median abs. error",
    )
    ax_c.semilogx(
        n_samples,
        median_err_mf,
        color=COLORS["red"],
        marker="s",
        ms=4,
        label="MF median abs. error",
    )
    speedup_total = np.divide(
        median_t_gp,
        median_t_mf_total,
        out=np.full_like(median_t_gp, np.nan),
        where=median_t_mf_total > 0.0,
    )
    ax_c2 = ax_c.twinx()
    ax_c2.semilogx(
        n_samples,
        speedup_total,
        color=COLORS["blue"],
        marker="^",
        ms=4,
        ls="--",
        label="Speedup total (GP/MF)",
    )
    ax_c.set_xlabel("Sample size $N$")
    ax_c.set_ylabel("Median abs. error")
    ax_c2.set_ylabel("Speedup (GP/MF)")
    ax_c.set_title("(c) Aggregate comparison", loc="left")
    apply_axes_polish(ax_c, kind="grid", title_loc="left")
    apply_axes_polish(ax_c2, kind="grid", title_loc="left")
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc="best", fontsize=spec.tick_size - 1)

    ax_d.axis("off")
    n_last = int(n_samples[-1])
    lines = [f"Comparison at N={n_last}", ""]
    lines.append(f"{'Scenario':<20} {'Ref D2':>7} {'GP':>7} {'MF':>7}")
    lines.append("-" * 48)
    for si, name in enumerate(scenario_names):
        lines.append(
            f"{name:<20} {ref_d2[si]:>7.3f} "
            f"{d2_gp[si, -1]:>7.3f} {d2_mf[si, -1]:>7.3f}"
        )
    lines.append("")
    lines.append(
        f"Median |err|: GP={median_err_gp[-1]:.4f}, MF={median_err_mf[-1]:.4f}"
    )
    speedup_total_last = speedup_total[-1]
    if np.isfinite(speedup_total_last):
        lines.append(f"Median speedup total (GP/MF): {speedup_total_last:.2f}x")
    else:
        lines.append("Median speedup total (GP/MF): n/a")
    ax_d.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax_d.transAxes,
        fontsize=spec.tick_size - 0.3,
        va="top",
        fontfamily="monospace",
    )
    ax_d.set_title("(d) Summary", loc="left")
    apply_axes_polish(ax_d, kind="grid", title_loc="left")

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


def main():
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing GP vs multifractal benchmark...")
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
