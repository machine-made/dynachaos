"""Shared analysis and plotting pipeline for benchmark examples.

Provides five functions:
- ``load_jsonc``: Load a JSONC file (JSON with // comments)
- ``run_embedding_analysis``: AMI -> Cao -> FNN -> embedding -> D2
- ``plot_benchmark``: multi-panel figure with attractor, diagnostics,
  D2 comparison (GP vs multifractal), and summary
- ``plot_zero_one_test``: 0-1 test figure (p-q trajectory, K histogram, summary)
- ``plot_multifractal``: multifractal figure (Dq, f(alpha), tau(q), summary)

Every benchmark script calls these with system-specific data and reference values.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynachaos.diagnostics import (
    average_mutual_information,
    cao_method,
    correlation_dimension,
    false_nearest_neighbors,
    multifractal_spectrum,
    select_dimension_cao,
)
from dynachaos.diagnostics.recurrence import embed_time_delay
from dynachaos.diagnostics.zero_one_test import zero_one_series
from dynachaos.utils.style import (
    COLORS,
    apply_axes_polish,
    figure_spec,
    finalize_legend,
    setup,
)


def _normalize_unit_interval(values):
    """Normalize a 1D array into [0, 1] with a robust constant-axis fallback."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if not finite.any():
        return out
    v = arr[finite]
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    if v_max > v_min:
        out[finite] = (v - v_min) / (v_max - v_min)
    else:
        out[finite] = 0.5
    return out


def _projection_points(x, y):
    """Build finite normalized 2D projection points in [0, 1]^2."""
    x_u = _normalize_unit_interval(x)
    y_u = _normalize_unit_interval(y)
    finite = np.isfinite(x_u) & np.isfinite(y_u)
    if np.sum(finite) < 32:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([x_u[finite], y_u[finite]]).astype(np.float64, copy=False)


def _gp_d2_from_projection_points(points, n_r=40, theiler_window=1):
    """Estimate GP D2 directly from normalized 2D projection points."""
    if points.shape[0] < 64:
        return np.nan
    tw = max(int(theiler_window), 0)
    d2, _, _, _, _ = correlation_dimension(points, n_r=n_r, theiler_window=tw)
    return float(d2)


def _multifractal_d2_from_projection_points(points, grid_size=128):
    """Estimate multifractal D2 from normalized 2D projection points."""
    if points.shape[0] < 64:
        return np.nan, "n/a"

    xv = points[:, 0]
    yv = points[:, 1]
    field = np.zeros((grid_size, grid_size), dtype=np.float64)
    ix = np.floor(xv * grid_size).astype(np.int64)
    iy = np.floor(yv * grid_size).astype(np.int64)
    ix = np.clip(ix, 0, grid_size - 1)
    iy = np.clip(iy, 0, grid_size - 1)
    np.add.at(field, (iy, ix), 1.0)

    max_box = grid_size // 2
    box_sizes = np.array([2, 4, 8, 16, 32, 64], dtype=np.int64)
    box_sizes = box_sizes[box_sizes <= max_box]
    if box_sizes.size < 2:
        return np.nan, "n/a"

    q_values = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    spec = multifractal_spectrum(field, box_sizes=box_sizes, q_values=q_values)
    q = np.asarray(spec["q"], dtype=np.float64)
    dq = np.asarray(spec["Dq"], dtype=np.float64)
    idx = np.where(np.isclose(q, 2.0))[0]
    if idx.size == 0:
        return np.nan, str(spec.get("backend", "n/a"))
    return float(dq[idx[0]]), str(spec.get("backend", "n/a"))


def load_jsonc(path):
    """Load a JSONC file (JSON with // comments)."""
    text = Path(path).read_text()
    text = re.sub(r'//.*', '', text)
    return json.loads(text)


def run_embedding_analysis(x, tau_max=100, d_max=15, n_r=40, verbose=False):
    """Run AMI -> Cao -> FNN -> embed -> D2 pipeline.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Scalar time series.
    tau_max : int
        Maximum delay for AMI scan.
    d_max : int
        Maximum embedding dimension for Cao/FNN.
    n_r : int
        Number of r values for correlation dimension.
    verbose : bool
        If True, print timing and memory info during D2 computation.

    Returns
    -------
    dict with keys:
        tau_opt, d_opt, taus, mi_values, E1, E2,
        f1, f2, f3, r_values, C_values, D2, embedded
    """
    # Ensure contiguous array (e.g. traj[:, 0] is a non-contiguous view)
    x = np.ascontiguousarray(x, dtype=np.float64)

    # AMI: find optimal delay
    taus, mi_values = average_mutual_information(x, tau_max=tau_max)

    # Find first local minimum of AMI
    tau_opt = 1
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
            tau_opt = int(taus[i])
            break

    # Cao's method: find optimal dimension
    # cao_method takes d_max as int, returns E1/E2 of shape (d_max-1,)
    E1, E2 = cao_method(x, tau_opt, d_max=d_max)
    dims = np.arange(1, d_max)  # E1[i] = E(d=i+1)/E(d=i), so dims 1..d_max-1

    # FNN fractions: d_max as int, returns arrays of shape (d_max,)
    f1, f2, f3 = false_nearest_neighbors(x, tau_opt, d_max=d_max)
    fnn_dims = np.arange(1, d_max + 1)

    # Robust Cao selector: choose onset of stable near-1 E1 plateau.
    d_opt = int(select_dimension_cao(E1, min_dim=2, max_dim=d_max))

    # Embed and compute correlation dimension
    embedded = embed_time_delay(x, d_opt, tau_opt)
    D2, r_values, C_values, local_slopes, scaling_mask = correlation_dimension(
        embedded, n_r=n_r, theiler_window=tau_opt, verbose=verbose,
    )

    return {
        "tau_opt": tau_opt,
        "d_opt": d_opt,
        "taus": taus,
        "mi_values": mi_values,
        "dims": dims,
        "E1": E1,
        "E2": E2,
        "fnn_dims": fnn_dims,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "r_values": r_values,
        "C_values": C_values,
        "D2": D2,
        "local_slopes": local_slopes,
        "scaling_mask": scaling_mask,
        "embedded": embedded,
    }


def plot_benchmark(results, attractor_xy, output_png, system_name,
                   ref_D2=None, ref_lambda1=None, computed_lambda1=None,
                   computed_spectrum=None, ref_spectrum=None,
                   attractor_xlabel="$x$", attractor_ylabel="$y$",
                   attractor_scatter_kw=None,
                   attractor_plot_fn=None):
    """Produce a benchmark figure with GP-vs-multifractal D2 comparison panel.

    Panels:
        (a) Attractor phase portrait
        (b) AMI I(tau) with tau_opt marked
        (c) Cao E1(d)/E2(d) with d_opt marked
        (d) FNN fractions f1, f2, f3 vs d
        (e) log C(r) vs log r with D2 slope
        (f) D2 comparison on the same 2D projection
        (g) Summary text

    Parameters
    ----------
    results : dict
        Output of ``run_embedding_analysis``.
    attractor_xy : tuple of (ndarray, ndarray)
        (x, y) data for attractor scatter plot.
    output_png : Path
        Output filename.
    system_name : str
        System name for titles.
    ref_D2 : float or None
        Literature D2 value.
    ref_lambda1 : float or None
        Literature lambda_1 value.
    computed_lambda1 : float or None
        Computed lambda_1 value.
    computed_spectrum : ndarray or None
        Full computed Lyapunov spectrum.
    ref_spectrum : ndarray or None
        Literature Lyapunov spectrum.
    attractor_xlabel, attractor_ylabel : str
        Axis labels for attractor panel.
    attractor_scatter_kw : dict or None
        Extra kwargs for attractor scatter.
    """
    setup()
    spec = figure_spec("grid")

    fig, axes = plt.subplot_mosaic(
        [["a", "b", "c", "d"], ["e", "f", "g", "g"]],
        figsize=(spec.figsize[0] + 3.0, spec.figsize[1] + 0.8),
    )
    fig.subplots_adjust(hspace=0.44, wspace=0.34)

    ax_a = axes["a"]
    ax_b = axes["b"]
    ax_c = axes["c"]
    ax_d = axes["d"]
    ax_e = axes["e"]
    ax_f = axes["f"]
    ax_g = axes["g"]

    # ── (a) Attractor ──
    if attractor_plot_fn is not None:
        attractor_plot_fn(ax_a)
    else:
        scatter_kw = {"s": 0.05, "c": COLORS["black"], "alpha": 0.3, "rasterized": True}
        if attractor_scatter_kw:
            scatter_kw.update(attractor_scatter_kw)
        ax_a.scatter(*attractor_xy, **scatter_kw)
        ax_a.set_xlabel(attractor_xlabel)
        ax_a.set_ylabel(attractor_ylabel)
    ax_a.set_title("(a) Attractor", loc="left")
    ax_a.grid(False)
    apply_axes_polish(ax_a, kind="grid", title_loc="left")

    # ── (b) AMI ──
    ax_b.plot(results["taus"], results["mi_values"], color=COLORS["black"], lw=1.2)
    ax_b.axvline(results["tau_opt"], color=COLORS["red"], ls="--", lw=0.8,
                 label=rf"$\tau = {results['tau_opt']}$")
    ax_b.set_xlabel(r"$\tau$")
    ax_b.set_ylabel(r"$I(\tau)$")
    ax_b.set_title("(b) Average Mutual Information", loc="left")
    apply_axes_polish(ax_b, kind="grid", title_loc="left")
    finalize_legend(ax_b, kind="grid")

    # ── (c) Cao ──
    ax_c.plot(results["dims"], results["E1"], color=COLORS["black"], lw=1.2,
              marker="o", ms=3, label="$E_1(d)$")
    ax_c.plot(results["dims"], results["E2"], color=COLORS["blue"], lw=1.2,
              marker="s", ms=3, label="$E_2(d)$")
    ax_c.axhline(0.95, color=COLORS["grey"], ls=":", lw=0.6)
    ax_c.axvline(results["d_opt"], color=COLORS["red"], ls="--", lw=0.8,
                 label=rf"$d = {results['d_opt']}$")
    ax_c.set_xlabel("$d$")
    ax_c.set_ylabel("$E_1, E_2$")
    ax_c.set_title("(c) Cao's method", loc="left")
    apply_axes_polish(ax_c, kind="grid", title_loc="left")
    finalize_legend(ax_c, kind="grid", loc="center right")

    # ── (d) FNN ──
    ax_d.plot(results["fnn_dims"], results["f1"], color=COLORS["black"], lw=1.2,
              marker="o", ms=3, label="$f_1$")
    ax_d.plot(results["fnn_dims"], results["f2"], color=COLORS["blue"], lw=1.2,
              marker="s", ms=3, label="$f_2$")
    ax_d.plot(results["fnn_dims"], results["f3"], color=COLORS["red"], lw=1.2,
              marker="^", ms=3, label="$f_3$")
    ax_d.set_xlabel("$d$")
    ax_d.set_ylabel("FNN fraction")
    ax_d.set_title("(d) False nearest neighbors", loc="left")
    apply_axes_polish(ax_d, kind="grid", title_loc="left")
    finalize_legend(ax_d, kind="grid")

    # ── (e) Correlation dimension ──
    r_vals = results["r_values"]
    C_vals = results["C_values"]
    scaling = results.get("scaling_mask", np.zeros(len(r_vals), dtype=bool))
    mask = C_vals > 0
    if np.any(mask):
        # Plot all valid points in grey
        ax_e.plot(np.log(r_vals[mask]), np.log(C_vals[mask]),
                  color=COLORS["grey"], lw=0.8, marker="o", ms=2, alpha=0.5)
        # Highlight scaling region points
        scaling_valid = mask & scaling
        if np.any(scaling_valid):
            ax_e.plot(np.log(r_vals[scaling_valid]), np.log(C_vals[scaling_valid]),
                      color=COLORS["black"], lw=1.2, marker="o", ms=3)
            # Fit line over scaling region
            lr = np.log(r_vals[scaling_valid])
            lc = np.log(C_vals[scaling_valid])
            coeffs = np.polyfit(lr, lc, 1)
            ax_e.plot(lr, np.polyval(coeffs, lr), color=COLORS["red"],
                      ls="--", lw=0.8, label=rf"$D_2 = {results['D2']:.2f}$")
    ax_e.set_xlabel(r"$\ln\, r$")
    ax_e.set_ylabel(r"$\ln\, C(r)$")
    ax_e.set_title("(e) Correlation integral", loc="left")
    apply_axes_polish(ax_e, kind="grid", title_loc="left")
    finalize_legend(ax_e, kind="grid")

    # ── (f) D2 comparison on same 2D projection data ──
    d2_gp_embed = float(results["D2"])
    points_proj = _projection_points(attractor_xy[0], attractor_xy[1])
    projection_theiler = max(int(results.get("tau_opt", 1)), 1)
    d2_gp_proj = _gp_d2_from_projection_points(
        points_proj,
        theiler_window=projection_theiler,
    )
    d2_mf_proj, mf_backend = _multifractal_d2_from_projection_points(points_proj)

    compare_vals = np.array([d2_gp_proj, d2_mf_proj], dtype=np.float64)
    compare_labels = ["GP (2D proj)", "Multifractal (2D proj)"]
    x_pos = np.arange(len(compare_labels))

    compare_plot_vals = np.where(np.isfinite(compare_vals), compare_vals, 0.0)
    bar_colors = [COLORS["black"], COLORS["red"]]
    bars = ax_f.bar(x_pos, compare_plot_vals, color=bar_colors, alpha=0.82)
    ax_f.set_xticks(x_pos)
    ax_f.set_xticklabels(compare_labels)
    ax_f.set_ylabel(r"$D_2$")
    ax_f.set_title("(f) D2 comparison (same 2D projection)", loc="left")
    ax_f.axhline(2.0, color=COLORS["grey"], ls=":", lw=0.8, label="2D ceiling")
    finalize_legend(ax_f, kind="grid")

    finite_vals = compare_vals[np.isfinite(compare_vals)]
    if finite_vals.size > 0:
        lo = float(np.min(finite_vals))
        hi = float(np.max(finite_vals))
        lo = min(lo, 0.0)
        hi = max(hi, 2.0)
        span = max(hi - lo, 0.05)
        pad = 0.25 * span
        ax_f.set_ylim(lo - pad, hi + pad)
    else:
        ax_f.set_ylim(-0.2, 2.2)
        ax_f.text(
            0.5,
            0.5,
            "No finite projection D2 estimates",
            transform=ax_f.transAxes,
            ha="center",
            va="center",
            fontsize=spec.tick_size - 0.8,
        )

    for i, (bar, value) in enumerate(zip(bars, compare_vals, strict=True)):
        if np.isfinite(value):
            ax_f.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=spec.tick_size - 0.8,
            )
        else:
            ax_f.text(
                x_pos[i],
                0.5,
                "n/a",
                ha="center",
                va="center",
                transform=ax_f.get_xaxis_transform(),
                fontsize=spec.tick_size - 0.8,
            )
            bar.set_alpha(0.25)
    apply_axes_polish(ax_f, kind="grid", title_loc="left")

    # ── (g) Summary ──
    ax_g.axis("off")
    lines = [f"System: {system_name}", ""]

    # Embedding parameters
    lines.append(f"Optimal delay: tau = {results['tau_opt']}")
    lines.append(f"Embedding dim: d = {results['d_opt']}")
    lines.append("")

    # Correlation dimension
    lines.append(f"D2 GP (embedding): {d2_gp_embed:.3f}")
    if np.isfinite(d2_gp_proj):
        lines.append(f"D2 GP (2D projection): {d2_gp_proj:.3f}")
    else:
        lines.append("D2 GP (2D projection): n/a")
    lines.append(f"Projection Theiler window: {projection_theiler}")
    if np.isfinite(d2_mf_proj):
        lines.append(f"D2 multifractal (2D proj): {d2_mf_proj:.3f} ({mf_backend})")
    else:
        lines.append("D2 multifractal (2D proj): n/a")
    if ref_D2 is not None:
        lines.append(f"D2 (literature): {ref_D2}")
    lines.append("")

    # Lyapunov
    if computed_lambda1 is not None:
        lines.append(f"lambda_1 (computed): {computed_lambda1:.4f}")
    if ref_lambda1 is not None:
        lines.append(f"lambda_1 (literature): {ref_lambda1}")

    if computed_spectrum is not None:
        spec_str = ", ".join(f"{v:.4f}" for v in computed_spectrum)
        lines.append(f"Spectrum: [{spec_str}]")
    if ref_spectrum is not None:
        ref_str = ", ".join(f"{v}" for v in ref_spectrum)
        lines.append(f"Ref spectrum: [{ref_str}]")

    summary_text = "\n".join(lines)
    ax_g.text(0.03, 0.95, summary_text, transform=ax_g.transAxes,
              fontsize=spec.tick_size, verticalalignment="top",
              fontfamily="monospace")
    ax_g.set_title("(g) Summary", loc="left")
    apply_axes_polish(ax_g, kind="grid", title_loc="left")

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")


def plot_zero_one_test(series, output_png, system_name, n_c=100, rng_seed=42):
    """Produce a 0-1 test figure with p-q trajectory, K histogram, and summary.

    For continuous-time systems the caller must subsample the series to a
    decorrelated rate (e.g. every tau_opt-th sample) *before* passing it here.
    Oversampled continuous data will give spurious K ≈ 0 results because
    consecutive values are nearly identical and the MSD cannot grow.

    Parameters
    ----------
    series : array_like
        Scalar time series (1D observable of the dynamical system).
        Should be pre-subsampled for continuous-time data.
    output_png : Path
        Output filename.
    system_name : str
        System name for titles.
    n_c : int
        Number of random c values for the 0-1 test.
    rng_seed : int
        RNG seed for reproducibility.
    """
    setup()
    spec = figure_spec("grid")

    series = np.asarray(series, dtype=np.float64).ravel()
    N = len(series)
    rng = np.random.default_rng(rng_seed)

    # Compute K values for all c
    c_values, K_values = zero_one_series(series, n_c=n_c, rng=rng)

    K_med = float(np.median(K_values))
    verdict = "chaotic" if K_med > 0.5 else "regular"

    # Pick representative c (closest K to median)
    idx_rep = int(np.argmin(np.abs(K_values - K_med)))
    c_rep = c_values[idx_rep]

    # Rebuild p, q for representative c
    js = np.arange(1, N + 1, dtype=np.float64)
    p = np.cumsum(series * np.cos(js * c_rep))
    q = np.cumsum(series * np.sin(js * c_rep))

    fig, axes = plt.subplot_mosaic(
        [["a", "b"], ["c", "c"]],
        figsize=(spec.figsize[0], spec.figsize[1] + 0.4),
    )
    fig.subplots_adjust(hspace=0.48, wspace=0.34)

    # ── (a) p-q trajectory ──
    ax_a = axes["a"]
    ax_a.plot(p, q, color=COLORS["black"], lw=0.3, alpha=0.6, rasterized=True)
    ax_a.set_xlabel("$p_n$")
    ax_a.set_ylabel("$q_n$")
    ax_a.set_title(f"(a) p-q trajectory (c = {c_rep:.3f})", loc="left")
    ax_a.set_aspect("equal", adjustable="datalim")
    ax_a.grid(False)
    apply_axes_polish(ax_a, kind="grid", title_loc="left")

    # ── (b) K histogram ──
    ax_b = axes["b"]
    verdict_color = COLORS["red"] if verdict == "chaotic" else COLORS["blue"]
    # Adaptive binning: zoom to the data range with padding
    K_lo = float(np.min(K_values))
    K_hi = float(np.max(K_values))
    K_span = max(K_hi - K_lo, 0.02)
    bin_lo = max(K_lo - 0.15 * K_span, -0.05)
    bin_hi = min(K_hi + 0.15 * K_span, 1.05)
    ax_b.hist(K_values, bins=25, range=(bin_lo, bin_hi),
              color=verdict_color, alpha=0.7,
              edgecolor=COLORS["black"], linewidth=0.5)
    ax_b.axvline(K_med, color=COLORS["black"], ls="--", lw=1.2,
                 label=f"median K = {K_med:.3f}")
    ax_b.set_xlabel("$K$")
    ax_b.set_ylabel("Count")
    ax_b.set_title("(b) K(c) distribution", loc="left")
    ax_b.set_xlim(bin_lo, bin_hi)
    apply_axes_polish(ax_b, kind="grid", title_loc="left")
    finalize_legend(ax_b, kind="grid")

    # ── (c) Summary ──
    ax_c = axes["c"]
    ax_c.axis("off")
    lines = [
        f"System: {system_name}",
        "",
        "0-1 test for chaos (Gottwald & Melbourne 2009)",
        "",
        f"K (median):  {K_med:.4f}",
        f"Verdict:     {verdict} (K {'\u2248 1' if verdict == 'chaotic' else '\u2248 0'})",
        "",
        f"n_c (random frequencies): {n_c}",
        f"N (series length):       {N}",
        f"Representative c:        {c_rep:.4f}  (K = {K_values[idx_rep]:.4f})",
        f"K range:  [{K_lo:.4f}, {K_hi:.4f}]",
    ]
    ax_c.text(0.03, 0.95, "\n".join(lines), transform=ax_c.transAxes,
              fontsize=spec.tick_size, verticalalignment="top",
              fontfamily="monospace")
    ax_c.set_title("(c) Summary", loc="left")
    apply_axes_polish(ax_c, kind="grid", title_loc="left")

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")


def plot_multifractal(attractor_xy, output_png, system_name,
                      grid_size=256, q_range=(-5, 5), n_q=41):
    """Produce a multifractal figure: Dq(q), f(alpha), tau(q), summary.

    Parameters
    ----------
    attractor_xy : tuple of (ndarray, ndarray)
        (x, y) attractor data (will be normalized and binned).
    output_png : Path
        Output filename.
    system_name : str
        System name for titles.
    grid_size : int
        Bin resolution for the 2D histogram.
    q_range : tuple of (float, float)
        Range of moment orders q.  Kept moderate (default [-5, 5]) to avoid
        noise-dominated extreme-q values.
    n_q : int
        Number of q values.
    """
    setup()
    spec = figure_spec("grid")

    # Build 2D field from attractor projection
    points = _projection_points(attractor_xy[0], attractor_xy[1])
    if points.shape[0] < 64:
        print(f"Skipping multifractal figure: too few projection points ({points.shape[0]})")
        return

    field = np.zeros((grid_size, grid_size), dtype=np.float64)
    ix = np.clip(np.floor(points[:, 0] * grid_size).astype(np.int64), 0, grid_size - 1)
    iy = np.clip(np.floor(points[:, 1] * grid_size).astype(np.int64), 0, grid_size - 1)
    np.add.at(field, (iy, ix), 1.0)

    # Box sizes: dyadic up to grid_size/2
    max_box = grid_size // 2
    box_sizes = np.array([2, 4, 8, 16, 32, 64, 128], dtype=np.int64)
    box_sizes = box_sizes[box_sizes <= max_box]

    q_values = np.linspace(q_range[0], q_range[1], n_q, dtype=np.float64)

    mf = multifractal_spectrum(field, box_sizes=box_sizes, q_values=q_values)

    q = np.asarray(mf["q"], dtype=np.float64)
    dq = np.asarray(mf["Dq"], dtype=np.float64)
    tau = np.asarray(mf["tau"], dtype=np.float64)
    # Use canonical (direct regression) alpha/f — much smoother than the
    # Legendre transform which amplifies noise via finite-difference gradient.
    alpha = np.asarray(mf["alpha_canonical"], dtype=np.float64)
    f_alpha = np.asarray(mf["f_canonical"], dtype=np.float64)
    r2_alpha = np.asarray(mf["r2_alpha"], dtype=np.float64)
    r2_tau = np.asarray(mf["r2_tau"], dtype=np.float64)
    phi = float(mf["phi"])
    backend = str(mf["backend"])

    # Extract D0, D1, D2
    def _get_dq(target_q):
        idx = np.argmin(np.abs(q - target_q))
        return float(dq[idx]) if np.isfinite(dq[idx]) else np.nan

    D0 = _get_dq(0.0)
    D1 = _get_dq(1.0)
    D2 = _get_dq(2.0)

    # Spectrum width from the well-behaved part of the Legendre spectrum:
    # restrict to q in [-3, 3] to avoid noisy tails in the alpha estimate.
    core_mask = np.isfinite(alpha) & (q >= -3.0) & (q <= 3.0)
    core_alpha = alpha[core_mask] if np.any(core_mask) else alpha[np.isfinite(alpha)]
    delta_alpha = (float(np.max(core_alpha) - np.min(core_alpha))
                   if core_alpha.size > 1 else 0.0)

    # Verdict
    if delta_alpha < 0.05:
        mf_verdict = "monofractal"
    else:
        mf_verdict = "multifractal"

    fig, axes = plt.subplot_mosaic(
        [["a", "b"], ["c", "d"]],
        figsize=(spec.figsize[0], spec.figsize[1] + 0.8),
    )
    fig.subplots_adjust(hspace=0.48, wspace=0.34)

    # ── (a) Dq vs q ──
    ax_a = axes["a"]
    # Only show Dq values with good tau regression (R² > 0.95)
    good_dq = np.isfinite(dq) & (r2_tau > 0.95)
    ax_a.plot(q[good_dq], dq[good_dq], color=COLORS["black"], lw=1.2)
    # Mark D0, D1, D2
    for label, val, target_q, color in [
        ("$D_0$", D0, 0.0, COLORS["blue"]),
        ("$D_1$", D1, 1.0, COLORS["green"]),
        ("$D_2$", D2, 2.0, COLORS["red"]),
    ]:
        if np.isfinite(val):
            ax_a.axhline(val, color=color, ls=":", lw=0.7, alpha=0.7)
            ax_a.plot(target_q, val, "o", color=color, ms=5, zorder=5)
            ax_a.annotate(f"{label}={val:.2f}", xy=(target_q, val),
                          xytext=(8, 4), textcoords="offset points",
                          fontsize=spec.legend_size, color=color)
    ax_a.set_xlabel("$q$")
    ax_a.set_ylabel("$D_q$")
    ax_a.set_title("(a) Generalized dimensions", loc="left")
    apply_axes_polish(ax_a, kind="grid", title_loc="left")

    # ── (b) f(alpha) vs alpha ──
    # Sort by alpha and filter by R² quality to suppress noise at extreme q.
    ax_b = axes["b"]
    finite_fa = np.isfinite(alpha) & np.isfinite(f_alpha) & (r2_alpha > 0.95)
    if np.any(finite_fa):
        a_plot = alpha[finite_fa]
        f_plot = f_alpha[finite_fa]
        sort_idx = np.argsort(a_plot)
        a_plot = a_plot[sort_idx]
        f_plot = f_plot[sort_idx]
        ax_b.plot(a_plot, f_plot, color=COLORS["black"], lw=1.2)
        # Mark peak
        peak_idx = int(np.argmax(f_plot))
        a_peak = float(a_plot[peak_idx])
        f_peak = float(f_plot[peak_idx])
        ax_b.plot(a_peak, f_peak, "o", color=COLORS["red"], ms=5, zorder=5)
        ax_b.annotate(f"peak ({a_peak:.2f}, {f_peak:.2f})",
                      xy=(a_peak, f_peak), xytext=(8, -10),
                      textcoords="offset points", fontsize=spec.legend_size,
                      color=COLORS["red"])
    ax_b.set_xlabel(r"$\alpha$")
    ax_b.set_ylabel(r"$f(\alpha)$")
    ax_b.set_title(r"(b) Singularity spectrum $f(\alpha)$", loc="left")
    apply_axes_polish(ax_b, kind="grid", title_loc="left")

    # ── (c) tau(q) vs q ──
    ax_c = axes["c"]
    finite_tau = np.isfinite(tau)
    ax_c.plot(q[finite_tau], tau[finite_tau], color=COLORS["black"], lw=1.2)
    # Mark q=0, 1, 2
    for target_q, color in [(0.0, COLORS["blue"]), (1.0, COLORS["green"]), (2.0, COLORS["red"])]:
        idx = np.argmin(np.abs(q - target_q))
        if np.isfinite(tau[idx]):
            ax_c.plot(q[idx], tau[idx], "o", color=color, ms=5, zorder=5)
    ax_c.set_xlabel("$q$")
    ax_c.set_ylabel(r"$\tau(q)$")
    ax_c.set_title(r"(c) Mass exponent $\tau(q)$", loc="left")
    apply_axes_polish(ax_c, kind="grid", title_loc="left")

    # ── (d) Summary ──
    ax_d = axes["d"]
    ax_d.axis("off")
    lines = [
        f"System: {system_name}",
        "",
        "Multifractal analysis",
        "",
        f"D0 (capacity):     {D0:.3f}" if np.isfinite(D0) else "D0: n/a",
        f"D1 (information):  {D1:.3f}" if np.isfinite(D1) else "D1: n/a",
        f"D2 (correlation):  {D2:.3f}" if np.isfinite(D2) else "D2: n/a",
        "",
        f"Spectrum width:    {delta_alpha:.3f}",
        f"phi (std alpha):   {phi:.4f}",
        f"Verdict:           {mf_verdict}",
        "",
        f"Grid: {grid_size}x{grid_size}, backend: {backend}",
    ]
    ax_d.text(0.03, 0.95, "\n".join(lines), transform=ax_d.transAxes,
              fontsize=spec.tick_size, verticalalignment="top",
              fontfamily="monospace")
    ax_d.set_title("(d) Summary", loc="left")
    apply_axes_polish(ax_d, kind="grid", title_loc="left")

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
