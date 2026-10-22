#!/usr/bin/env python3
"""
benchmark_performance: Measure Rust vs Python performance for key diagnostics.

Sweeps N = [1000, 2000, 5000, 10_000, 20_000, 50_000] and measures wall-clock
time for: correlation_integral, average_mutual_information, cao_method,
false_nearest_neighbors.

OUTPUTS: benchmark_performance.npz, benchmark_performance.png
USAGE:   python examples/benchmark_performance.py
         rm examples/benchmark_performance.npz && python examples/benchmark_performance.py
"""

import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")

N_VALUES = [1000, 2000, 5000, 10_000, 20_000, 50_000]
N_REPEATS = 3
ALGORITHMS = ["correlation_integral", "ami", "cao", "fnn"]


def _make_series(n, a=1.99, burn=500):
    """Generate logistic map series for benchmarking."""
    x = 0.123456789
    series = np.empty(n)
    for i in range(n + burn):
        x = 1.0 - a * x * x
        if i >= burn:
            series[i - burn] = x
    return series


def _time_algorithm(name, series, use_rust):
    """Run one algorithm and return wall-clock time."""
    from dynachaos.diagnostics import correlation as corr_mod
    from dynachaos.diagnostics import embedding as emb_mod

    if name == "correlation_integral":
        old_flag = corr_mod._RUST_AVAILABLE
        corr_mod._RUST_AVAILABLE = use_rust
        try:
            from dynachaos.diagnostics.recurrence import embed_time_delay
            embedded = embed_time_delay(series, d=3, tau=1)
            r_values = np.logspace(-2, 0, 20)
            t0 = time.perf_counter()
            corr_mod.correlation_integral(embedded, r_values, theiler_window=1)
            return time.perf_counter() - t0
        finally:
            corr_mod._RUST_AVAILABLE = old_flag

    elif name == "ami":
        old_flag = emb_mod._RUST_AVAILABLE
        emb_mod._RUST_AVAILABLE = use_rust
        try:
            t0 = time.perf_counter()
            emb_mod.average_mutual_information(series, tau_max=30)
            return time.perf_counter() - t0
        finally:
            emb_mod._RUST_AVAILABLE = old_flag

    elif name == "cao":
        old_flag = emb_mod._RUST_AVAILABLE
        emb_mod._RUST_AVAILABLE = use_rust
        try:
            dims = np.arange(1, 8)
            t0 = time.perf_counter()
            emb_mod.cao_method(series, tau=1, d_max=int(dims[-1]))
            return time.perf_counter() - t0
        finally:
            emb_mod._RUST_AVAILABLE = old_flag

    elif name == "fnn":
        old_flag = emb_mod._RUST_AVAILABLE
        emb_mod._RUST_AVAILABLE = use_rust
        try:
            dims = np.arange(1, 8)
            t0 = time.perf_counter()
            emb_mod.false_nearest_neighbors(series, tau=1, d_max=int(dims[-1]))
            return time.perf_counter() - t0
        finally:
            emb_mod._RUST_AVAILABLE = old_flag

    raise ValueError(f"Unknown algorithm: {name}")


def _check_rust():
    """Check if Rust extension is available."""
    try:
        from dynachaos._rust import correlation_counts  # noqa: F401
        return True
    except ImportError:
        return False


def compute():
    has_rust = _check_rust()
    n_alg = len(ALGORITHMS)
    n_N = len(N_VALUES)

    times_python = np.full((n_alg, n_N), np.nan)
    times_rust = np.full((n_alg, n_N), np.nan)

    for j, N in enumerate(N_VALUES):
        print(f"  N = {N:,}")
        series = _make_series(N)

        for i, alg in enumerate(ALGORITHMS):
            # Python timings (always available)
            trial_times = []
            for _ in range(N_REPEATS):
                t = _time_algorithm(alg, series, use_rust=False)
                trial_times.append(t)
            times_python[i, j] = np.median(trial_times)

            # Rust timings
            if has_rust:
                trial_times = []
                for _ in range(N_REPEATS):
                    t = _time_algorithm(alg, series, use_rust=True)
                    trial_times.append(t)
                times_rust[i, j] = np.median(trial_times)

            label = f"    {alg}: py={times_python[i, j]:.4f}s"
            if has_rust:
                speedup = times_python[i, j] / times_rust[i, j]
                label += f", rs={times_rust[i, j]:.4f}s, speedup={speedup:.1f}x"
            print(label)

    np.savez_compressed(
        OUTPUT_NPZ,
        N_values=np.array(N_VALUES),
        algorithms=np.array(ALGORITHMS),
        times_python=times_python,
        times_rust=times_rust,
        has_rust=has_rust,
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

    N_vals = data["N_values"]
    algorithms = list(data["algorithms"])
    t_py = data["times_python"]
    t_rs = data["times_rust"]
    has_rust = bool(data["has_rust"])

    spec = figure_spec("grid")
    fig, axes = plt.subplots(2, 2, figsize=(spec.figsize[0], spec.figsize[1] + 1.0))
    fig.subplots_adjust(hspace=0.50, wspace=0.38)

    alg_labels = {
        "correlation_integral": "Correlation integral",
        "ami": "AMI",
        "cao": "Cao",
        "fnn": "FNN",
    }
    colors_list = [COLORS["black"], COLORS["red"], COLORS["blue"], COLORS["green"]]

    # -- (a) Wall-clock vs N --
    ax = axes[0, 0]
    for i, alg in enumerate(algorithms):
        c = colors_list[i]
        ax.loglog(N_vals, t_py[i], color=c, ls="-", marker="o", ms=3,
                  label=f"{alg_labels[alg]} (Py)")
        if has_rust and not np.all(np.isnan(t_rs[i])):
            ax.loglog(N_vals, t_rs[i], color=c, ls="--", marker="s", ms=3,
                      label=f"{alg_labels[alg]} (Rs)")
    ax.set_xlabel("$N$")
    ax.set_ylabel("Wall-clock (s)")
    ax.set_title("(a) Timing", loc="left")
    apply_axes_polish(ax, kind="grid", title_loc="left")
    finalize_legend(ax, kind="grid", loc="upper left", ncol=2)

    # -- (b) Speedup ratio --
    ax = axes[0, 1]
    if has_rust:
        for i, alg in enumerate(algorithms):
            speedup = t_py[i] / t_rs[i]
            ax.semilogx(N_vals, speedup, color=colors_list[i],
                        marker="o", ms=4, lw=1.2, label=alg_labels[alg])
        ax.axhline(1.0, color=COLORS["grey"], ls=":", lw=0.6)
        ax.set_ylabel("Speedup (Python / Rust)")
    else:
        ax.text(0.5, 0.5, "Rust not available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=spec.tick_size)
    ax.set_xlabel("$N$")
    ax.set_title("(b) Speedup", loc="left")
    apply_axes_polish(ax, kind="grid", title_loc="left")
    if has_rust:
        finalize_legend(ax, kind="grid")

    # -- (c) Bar chart at largest N --
    ax = axes[1, 0]
    x_pos = np.arange(len(algorithms))
    width = 0.35
    py_times = t_py[:, -1]
    ax.bar(x_pos - width / 2, py_times, width, color=COLORS["blue"],
           label="Python", alpha=0.8)
    if has_rust:
        rs_times = t_rs[:, -1]
        ax.bar(x_pos + width / 2, rs_times, width, color=COLORS["red"],
               label="Rust", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([alg_labels[a] for a in algorithms], rotation=30, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"(c) At N={N_vals[-1]:,}", loc="left")
    apply_axes_polish(ax, kind="grid", title_loc="left")
    finalize_legend(ax, kind="grid")

    # -- (d) Summary table --
    ax = axes[1, 1]
    ax.axis("off")
    lines = ["Performance summary", f"N values: {list(N_vals)}", ""]
    lines.append(f"{'Algorithm':<20} {'Py (s)':>8} {'Rs (s)':>8} {'Speedup':>8}")
    lines.append("-" * 48)
    for i, alg in enumerate(algorithms):
        py_t = t_py[i, -1]
        if has_rust and not np.isnan(t_rs[i, -1]):
            rs_t = t_rs[i, -1]
            sp = py_t / rs_t
            lines.append(f"{alg_labels[alg]:<20} {py_t:>8.4f} {rs_t:>8.4f} {sp:>7.1f}x")
        else:
            lines.append(f"{alg_labels[alg]:<20} {py_t:>8.4f}     N/A      N/A")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=spec.tick_size - 0.5, verticalalignment="top",
            fontfamily="monospace")
    ax.set_title("(d) Summary", loc="left")
    apply_axes_polish(ax, kind="grid", title_loc="left")

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


def main():
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Running performance benchmark...")
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
