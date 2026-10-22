"""Export decimated, web-ready JSON payloads from cached figure .npz arrays.

This is the data layer for an interactive gallery: it reads the arrays already
cached under ``figures/<section>/<name>.npz`` by the reproduction pipelines and
writes self-describing JSON payloads under ``site/data/<section>/<name>.json``
that a browser can render as charts. No HTML/CSS/JS is produced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# Emit floats with 6 significant figures to keep payloads small.
_SIG_FIGS = 6

# Heatmap z-grids only ever feed a 256-step colour ramp and a hover readout,
# neither of which can show a 6th significant figure -- 3 sig figs keeps the
# on-screen result identical while cutting the largest payloads by ~5x. Axis
# coordinates and line/scatter data keep the full _SIG_FIGS precision above.
_SIG_FIGS_Z = 3


def _round_to(value: float, sig_figs: int) -> float:
    """Round a single float to ``sig_figs`` significant figures."""
    if not np.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    return float(f"{value:.{sig_figs}g}")


def _round_sig(value: float) -> float:
    """Round a single float to 6 significant figures."""
    return _round_to(value, _SIG_FIGS)


def _round_list(values: np.ndarray) -> list[float]:
    """Round a 1-D array to 6 significant figures, as a plain list."""
    return [_round_sig(float(v)) for v in values]


def _round_grid(values: np.ndarray) -> list[list[float]]:
    """Round a 2-D z-grid to the coarser 3 significant figures a colour ramp
    and hover readout actually need, as nested lists."""
    return [[_round_to(float(v), _SIG_FIGS_Z) for v in row] for row in values]


def _require(npz: np.lib.npyio.NpzFile, key: str, source: str) -> np.ndarray:
    """Fetch ``key`` from ``npz``, failing loudly if it is missing."""
    if key not in npz.files:
        raise KeyError(f"{source}: expected key {key!r}, found {list(npz.files)}")
    return npz[key]


def _load(section: str, name: str) -> np.lib.npyio.NpzFile:
    path = FIGURES_DIR / section / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    return np.load(path, allow_pickle=False)


def _no_decimation(n: int) -> dict[str, Any]:
    return {"method": "none", "from": n, "to": n}


def _stride_decimation(n_from: int, n_to: int) -> dict[str, Any]:
    return {"method": "stride", "from": n_from, "to": n_to}


def _uniform_decimation(pairs: set[tuple[int, int]]) -> dict[str, Any]:
    """One decimation record for a multi-panel figure.

    The JSON carries a single figure-level record, which is only truthful if
    every panel was decimated identically -- fail rather than let the record
    silently describe just the last panel.
    """
    if len(pairs) != 1:
        raise SystemExit(
            f"panels decimated unevenly, one record cannot describe them: {sorted(pairs)}"
        )
    n_from, n_to = next(iter(pairs))
    return _stride_decimation(n_from, n_to)


def _minmax_bin_decimation(n_from: int, n_bins: int) -> dict[str, Any]:
    return {
        "method": "minmax-bin",
        "from": n_from,
        "to": n_bins,
        "note": (
            "each bin emits the bin's min- and max-y sample (in original order) "
            "rather than a stride subsample, so plateau edges in a devil's "
            "staircase are not flattened by decimation"
        ),
    }


def _minmax_decimate(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin ``y`` into ``n_bins`` equal-width chunks along the sample index and
    keep, from each bin, only the samples at the bin's minimum and maximum
    y-value (in their original index order, de-duplicated when they coincide).

    A plain stride subsample of a devil's staircase erases the very feature
    the figure exists to show: the plateau step edges collapse to a smooth
    ramp once most in-between samples are dropped. Keeping each bin's extremal
    pair instead preserves the step onset and the step's flat interior without
    keeping every sample.
    """
    n = x.shape[0]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    xs: list[float] = []
    ys: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        if hi <= lo:
            continue
        chunk = y[lo:hi]
        i_min = lo + int(np.argmin(chunk))
        i_max = lo + int(np.argmax(chunk))
        for i in sorted({i_min, i_max}):
            xs.append(float(x[i]))
            ys.append(float(y[i]))
    return np.asarray(xs), np.asarray(ys)


#: Delay mixing weight of the delayed logistic map used throughout sec05/sec07.
ALPHA = 0.3


def _neimark_sacker_dc(alpha: float) -> float:
    """Neimark-Sacker threshold D_c = (3 - 2a) / [4 (1 - a)^2]."""
    return (3.0 - 2.0 * alpha) / (4.0 * (1.0 - alpha) ** 2)


def _lam(i: int) -> str:
    """Lyapunov exponent name with a real subscript, so the chart legend reads
    the same as the caption."""
    return "λ" + "₀₁₂₃₄₅₆₇₈₉"[i]


def export_rqa_measures() -> dict[str, Any]:
    """sec11_diagnostics/rqa_measures: four RQA measures vs embedding delay D."""
    source = "sec11_diagnostics/rqa_measures"
    npz = _load("sec11_diagnostics", "rqa_measures")
    d = _require(npz, "D", source)
    n = d.shape[0]
    traces = []
    for key, name in (
        ("RR", "Recurrence rate"),
        ("DET", "Determinism"),
        ("LAM", "Laminarity"),
        ("ENTR", "Entropy of diagonal lines"),
    ):
        y = _require(npz, key, source)
        traces.append({"name": name, "x": _round_list(d), "y": _round_list(y)})
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D"}, "y": {"label": "RQA measure"}},
        "panels": [{"title": "Recurrence quantification measures", "traces": traces}],
    }


def export_test01_sweep() -> dict[str, Any]:
    """sec11_diagnostics/test01_sweep: 0-1 test statistic vs logistic nonlinearity a."""
    source = "sec11_diagnostics/test01_sweep"
    npz = _load("sec11_diagnostics", "test01_sweep")
    a = _require(npz, "a", source)
    k = _require(npz, "K", source)
    n = a.shape[0]
    return {
        "figure": source,
        "kind": "scatter",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "Nonlinearity a"}, "y": {"label": "K₀₁ (0–1 test)"}},
        "panels": [
            {
                "title": "0-1 test for chaos: logistic map",
                "traces": [{"name": "K₀₁", "x": _round_list(a), "y": _round_list(k)}],
            }
        ],
    }


def export_permutation_entropy() -> dict[str, Any]:
    """sec11_diagnostics/permutation_entropy: PE vs a (logistic) and vs D (delayed logistic).

    Keys are ``a``, ``H_logistic``, ``D``, ``H_delayed`` (matches the brief).
    """
    source = "sec11_diagnostics/permutation_entropy"
    npz = _load("sec11_diagnostics", "permutation_entropy")
    a = _require(npz, "a", source)
    h_logistic = _require(npz, "H_logistic", source)
    d = _require(npz, "D", source)
    h_delayed = _require(npz, "H_delayed", source)
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(a.shape[0] + d.shape[0]),
        "axes": {"x": {"label": "a or D"}, "y": {"label": "Hₚₑ (permutation entropy)"}},
        "panels": [
            {
                "title": "Logistic map",
                "traces": [{"name": "Hₚₑ", "x": _round_list(a), "y": _round_list(h_logistic)}],
            },
            {
                "title": "Delayed logistic, α = 0.3",
                "traces": [{"name": "Hₚₑ", "x": _round_list(d), "y": _round_list(h_delayed)}],
            },
        ],
    }


def export_complexity_entropy_plane() -> dict[str, Any]:
    """sec11_diagnostics/complexity_entropy_plane: (H, C) scatter for two maps.

    Keys are ``a``, ``H_logistic``, ``C_logistic``, ``D``, ``H_delayed``, ``C_delayed``.
    """
    source = "sec11_diagnostics/complexity_entropy_plane"
    npz = _load("sec11_diagnostics", "complexity_entropy_plane")
    h_logistic = _require(npz, "H_logistic", source)
    c_logistic = _require(npz, "C_logistic", source)
    h_delayed = _require(npz, "H_delayed", source)
    c_delayed = _require(npz, "C_delayed", source)
    n = h_logistic.shape[0] + h_delayed.shape[0]
    return {
        "figure": source,
        "kind": "scatter",
        "decimation": _no_decimation(n),
        "axes": {
            "x": {"label": "Normalised permutation entropy H"},
            "y": {"label": "Statistical complexity C"},
        },
        "panels": [
            {
                "title": "Complexity-entropy plane (d=5)",
                "traces": [
                    {
                        "name": "Logistic map",
                        "x": _round_list(h_logistic),
                        "y": _round_list(c_logistic),
                    },
                    {
                        "name": "Delayed logistic map",
                        "x": _round_list(h_delayed),
                        "y": _round_list(c_delayed),
                    },
                ],
            }
        ],
    }


def export_correlation_dimension() -> dict[str, Any]:
    """sec07_fractalization/correlation_dimension: D2 vs D with an error band."""
    source = "sec07_fractalization/correlation_dimension"
    npz = _load("sec07_fractalization", "correlation_dimension")
    d = _require(npz, "D", source)
    d2 = _require(npz, "D2", source)
    d2_err = _require(npz, "D2_err", source)
    n = d.shape[0]
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D"}, "y": {"label": "Correlation dimension D₂"}},
        "panels": [
            {
                "title": "Delayed logistic map, α = 0.3",
                "traces": [
                    {
                        "name": "D₂",
                        "x": _round_list(d),
                        "y": _round_list(d2),
                        "yerr": _round_list(d2_err),
                    }
                ],
            }
        ],
    }


def export_lyapunov_vs_d() -> dict[str, Any]:
    """sec05_oscillation/lyapunov_vs_D: two-component Lyapunov spectrum vs D."""
    source = "sec05_oscillation/lyapunov_vs_D"
    npz = _load("sec05_oscillation", "lyapunov_vs_D")
    d = _require(npz, "D", source)
    spectra = _require(npz, "spectra", source)
    spectra_err = _require(npz, "spectra_err", source)
    n = d.shape[0]
    traces = []
    for col in range(spectra.shape[1]):
        traces.append(
            {
                "name": _lam(col + 1),
                "x": _round_list(d),
                "y": _round_list(spectra[:, col]),
                "yerr": _round_list(spectra_err[:, col]),
            }
        )
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D"}, "y": {"label": "Lyapunov exponent"}},
        # The caption promises a dashed line at the Neimark-Sacker bifurcation.
        # Same closed form and same alpha as the matplotlib figure, so the
        # interactive view cannot drift from the static one.
        "marks": [
            {"axis": "x", "value": _neimark_sacker_dc(ALPHA), "label": "D_c"},
            {"axis": "y", "value": 0.0, "label": "λ = 0"},
        ],
        "panels": [{"title": "Delayed logistic map, α = 0.3", "traces": traces}],
    }


def export_lyapunov_vs_db() -> dict[str, Any]:
    """sec06_three_torus/lyapunov_vs_DB: Lyapunov spectrum vs D_B for three epsilon.

    Keys are ``DB``, ``eps_values`` (the three epsilon values, in the order the
    ``eps_<value>_spectra`` keys are named), and one ``eps_<value>_spectra``
    array of shape (500, 4) per epsilon. Only the first 3 columns are used
    (per the brief).
    """
    source = "sec06_three_torus/lyapunov_vs_DB"
    npz = _load("sec06_three_torus", "lyapunov_vs_DB")
    db = _require(npz, "DB", source)
    eps_values = _require(npz, "eps_values", source)
    n = db.shape[0]
    panels = []
    for eps in eps_values:
        key = f"eps_{eps:g}_spectra"
        spectra = _require(npz, key, source)
        traces = [
            {
                "name": _lam(col + 1),
                "x": _round_list(db),
                "y": _round_list(spectra[:, col]),
            }
            for col in range(3)
        ]
        panels.append({"title": f"ε = {eps:g}", "traces": traces})
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D_B"}, "y": {"label": "Lyapunov exponent"}},
        "panels": panels,
    }


def export_map_i_attractors(max_points: int = 20000) -> dict[str, Any]:
    """sec04_doubling/map_I_attractors: (X, Y) projections for three D values.

    Keys are ``D_2.11_traj``, ``D_2.16_traj``, ``D_2.19_traj`` (each (100000, 3))
    and ``D_values`` giving the order (matches labels torus / 2x torus / chaos
    used by the reproduction script).
    """
    source = "sec04_doubling/map_I_attractors"
    npz = _load("sec04_doubling", "map_I_attractors")
    d_values = _require(npz, "D_values", source)
    labels = ["torus", "2x torus", "chaos"]
    panels = []
    pairs: set[tuple[int, int]] = set()
    for d_value, label in zip(d_values, labels, strict=True):
        key = f"D_{d_value:g}_traj"
        traj = _require(npz, key, source)
        n_from = traj.shape[0]
        idx = np.arange(0, n_from, max(1, n_from // max_points))[:max_points]
        pairs.add((n_from, idx.shape[0]))
        panels.append(
            {
                "title": f"D={d_value:g} ({label})",
                "traces": [
                    {
                        "name": label,
                        "x": _round_list(traj[idx, 0]),
                        "y": _round_list(traj[idx, 1]),
                    }
                ],
            }
        )
    return {
        "figure": source,
        "kind": "scatter",
        "decimation": _uniform_decimation(pairs),
        "axes": {"x": {"label": "X"}, "y": {"label": "Y"}},
        "panels": panels,
    }


def export_arnold_tongues(max_axis: int = 600) -> dict[str, Any]:
    """sec02_circle_map/arnold_tongues: rotation number heatmap over (Omega, K)."""
    source = "sec02_circle_map/arnold_tongues"
    npz = _load("sec02_circle_map", "arnold_tongues")
    omega = _require(npz, "Omega", source)
    k = _require(npz, "K", source)
    rho = _require(npz, "rho", source)
    n_k, n_omega = rho.shape

    k_step = max(1, -(-n_k // max_axis))
    omega_step = max(1, -(-n_omega // max_axis))

    k_idx = np.arange(0, n_k, k_step)
    omega_idx = np.arange(0, n_omega, omega_step)
    rho_ds = rho[np.ix_(k_idx, omega_idx)]

    return {
        "figure": source,
        "kind": "heatmap",
        "decimation": _stride_decimation(n_k * n_omega, rho_ds.shape[0] * rho_ds.shape[1]),
        "axes": {"x": {"label": "Bare frequency Ω"}, "y": {"label": "Nonlinearity K"}},
        # The critical line the caption names: above it the map is noninvertible.
        "marks": [{"axis": "y", "value": 1.0 / (2.0 * np.pi), "label": "K_c = 1/2π"}],
        "panels": [
            {
                "title": "Rotation number over the circle-map parameter plane",
                "zlabel": "Rotation number ρ",
                "cmap": "viridis",
                "x": _round_list(omega[omega_idx]),
                "y": _round_list(k[k_idx]),
                "z": _round_grid(rho_ds),
            }
        ],
    }


def export_phase_diagram() -> dict[str, Any]:
    """sec09_pattern/phase_diagram: activity maps over (a, epsilon), full resolution.

    Keys are ``a``, ``eps``, ``lam``, ``spatial_activity`` (plus a
    ``schema_version`` int that is not chart data and is not emitted).
    """
    source = "sec09_pattern/phase_diagram"
    npz = _load("sec09_pattern", "phase_diagram")
    a = _require(npz, "a", source)
    eps = _require(npz, "eps", source)
    lam = _require(npz, "lam", source)
    spatial_activity = _require(npz, "spatial_activity", source)
    n = lam.shape[0] * lam.shape[1]
    return {
        "figure": source,
        "kind": "heatmap",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "Nonlinearity a"}, "y": {"label": "Coupling ε"}},
        "panels": [
            {
                "title": "Largest Lyapunov exponent",
                "zlabel": "λ₁",
                # Signed quantity: a diverging ramp pinned at zero keeps the
                # locked / marginal / chaotic reading the rest of the site uses.
                "cmap": "signed",
                "x": _round_list(a),
                "y": _round_list(eps),
                "z": _round_grid(lam),
            },
            {
                "title": "Spatial activity map",
                "zlabel": "Spatial activity",
                "cmap": "viridis",
                "x": _round_list(a),
                "y": _round_list(eps),
                "z": _round_grid(spatial_activity),
            },
        ],
    }


def export_map_iv_lyapunov() -> dict[str, Any]:
    """sec04_doubling/map_IV_lyapunov: Lyapunov exponents for Map (IV) vs D.

    ``spectra`` has 4 columns, but ``plot_lyapunov`` (torus_doubling.py) only
    plots the first two -- the caption notes the third and fourth are large
    negative -- so only those two are exported to match the published figure
    exactly rather than adding two traces nobody drew.
    """
    source = "sec04_doubling/map_IV_lyapunov"
    npz = _load("sec04_doubling", "map_IV_lyapunov")
    d = _require(npz, "D", source)
    spectra = _require(npz, "spectra", source)
    n = d.shape[0]
    traces = [
        {"name": _lam(col + 1), "x": _round_list(d), "y": _round_list(spectra[:, col])}
        for col in range(2)
    ]
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "D"}, "y": {"label": "Lyapunov exponent"}},
        "marks": [{"axis": "y", "value": 0.0, "label": "λ = 0"}],
        "panels": [{"title": "Map (IV), α = 0.3", "traces": traces}],
    }


def export_phase_diagram_sec03() -> dict[str, Any]:
    """sec03_transition/phase_diagram: symmetry-breaking and Lyapunov maps over (a, ε).

    Keys are ``A``, ``D`` (the coupling axis, plotted as ε), ``asym``, ``lyap``
    (plus a ``schema_version`` int that is not chart data). Both fields are
    plotted directly by ``plot_phase_diagram`` (coupled_logistic.py); only the
    matplotlib colour-norm (PowerNorm/percentile clipping) is display-only and
    is not reproduced here, since the JSON always carries the raw z-values.
    """
    source = "sec03_transition/phase_diagram"
    npz = _load("sec03_transition", "phase_diagram")
    a = _require(npz, "A", source)
    eps = _require(npz, "D", source)
    asym = _require(npz, "asym", source)
    lyap = _require(npz, "lyap", source)
    n = asym.shape[0] * asym.shape[1]
    return {
        "figure": source,
        "kind": "heatmap",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "Nonlinearity a"}, "y": {"label": "Coupling ε"}},
        # Both static panels mark the eps=0.1 slice used elsewhere in the
        # gallery; a plain constant, not a derived quantity.
        "marks": [{"axis": "y", "value": 0.1, "label": "ε = 0.1 gallery slice"}],
        "panels": [
            {
                "title": "Symmetry breaking: <|x-y|>",
                "zlabel": "<|x-y|>",
                "cmap": "viridis",
                "x": _round_list(a),
                "y": _round_list(eps),
                "z": _round_grid(asym),
            },
            {
                "title": "Chaos onset: λ1 (finite-time)",
                "zlabel": "λ₁",
                "cmap": "signed",
                "x": _round_list(a),
                "y": _round_list(eps),
                "z": _round_grid(lyap),
            },
        ],
    }


def export_collective_lyapunov() -> dict[str, Any]:
    """sec10_gcm/collective_lyapunov: collective Lyapunov exponent lambda_c vs a.

    ``plot_collective`` (gcm_clusters.py) shades a "sustained lambda_c > 0"
    region using ``sustained_positive_mask`` -- a threshold-plus-run-length
    scan over the data, computed at draw time. That derived mask is not
    reproduced here (it would be a second implementation of the same logic,
    risking drift); only the raw line and the lambda=0 reference the mask is
    drawn against are exported.
    """
    source = "sec10_gcm/collective_lyapunov"
    npz = _load("sec10_gcm", "collective_lyapunov")
    a_values = _require(npz, "a_values", source)
    lyap_c = _require(npz, "lyap_c", source)
    eps = float(_require(npz, "eps", source)[0])
    n_count = int(_require(npz, "N", source)[0])
    n = a_values.shape[0]
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "a"}, "y": {"label": "λ_c"}},
        "marks": [{"axis": "y", "value": 0.0, "label": "λ_c = 0"}],
        "panels": [
            {
                "title": f"Collective Lyapunov exponent, ε = {eps:g}, N = {n_count}",
                "traces": [{"name": "λ_c", "x": _round_list(a_values), "y": _round_list(lyap_c)}],
            }
        ],
    }


def export_double_staircase(max_points: int = 4000) -> dict[str, Any]:
    """sec06_three_torus/double_staircase: rho_theta and rho_phi vs D.

    ``plot`` (modulated_circle.py) only draws rho_theta against D (plus a
    rigid-rotation diagonal reference and plateau annotations); rho_phi is
    only mentioned in a static text label ("rho_phi = C to numerical
    precision"). Both arrays are raw data straight from the npz -- no
    derivation -- so both are exported as lines for the interactive view,
    even though the static PNG only draws one of them.
    """
    source = "sec06_three_torus/double_staircase"
    npz = _load("sec06_three_torus", "double_staircase")
    d = _require(npz, "D", source)
    rho_theta = _require(npz, "rho_theta", source)
    rho_phi = _require(npz, "rho_phi", source)
    n_from = d.shape[0]
    idx = np.arange(0, n_from, max(1, n_from // max_points))[:max_points]
    n_to = idx.shape[0]
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _stride_decimation(n_from, n_to),
        "axes": {"x": {"label": "bare frequency D"}, "y": {"label": "rotation number"}},
        "panels": [
            {
                "title": "Double devil's staircase",
                "traces": [
                    {"name": "ρ_θ", "x": _round_list(d[idx]), "y": _round_list(rho_theta[idx])},
                    {"name": "ρ_φ", "x": _round_list(d[idx]), "y": _round_list(rho_phi[idx])},
                ],
            }
        ],
    }


def export_comoving_lyapunov() -> dict[str, Any]:
    """sec08_sti/comoving_lyapunov: co-moving Lyapunov exponent lambda(v).

    ``plot`` (comoving_figure.py) masks entries with ``lam_v > -9.5`` to NaN --
    a sentinel filter for missing/invalid samples, not a derived quantity --
    and reproduces that filter here so the interactive line does not show
    placeholder values. The per-series zero-crossing markers it also draws
    (linear-interpolated crossing points) are a derived quantity and are not
    reproduced; only the lambda=0 reference they are drawn against is kept.
    """
    source = "sec08_sti/comoving_lyapunov"
    npz = _load("sec08_sti", "comoving_lyapunov")
    v_values = _require(npz, "v_values", source)
    a_values = _require(npz, "a_values", source)
    n = v_values.shape[0]
    a_labels = {
        1.70: "pattern selection",
        1.85: "defect turbulence",
        1.95: "fully developed turbulence",
    }
    traces = []
    for a in a_values:
        key = f"lambda_a{a:.2f}"
        lam_v = _require(npz, key, source)
        valid = lam_v > -9.5
        y = np.where(valid, lam_v, np.nan)
        label = a_labels.get(float(a), "")
        name = f"a = {a:.2f} ({label})" if label else f"a = {a:.2f}"
        traces.append({"name": name, "x": _round_list(v_values), "y": _round_list(y)})
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(n),
        "axes": {"x": {"label": "Velocity v (sites/iteration)"}, "y": {"label": "λ(v)"}},
        "marks": [{"axis": "y", "value": 0.0, "label": "λ = 0"}],
        "panels": [{"title": "Co-moving Lyapunov exponent, logistic CML", "traces": traces}],
    }


def export_attractors_sec03(max_points: int = 12000) -> dict[str, Any]:
    """sec03_transition/attractors: six (x, y) attractor portraits.

    Keys are ``x_0..x_5``, ``y_0..y_5``, ``A_values``, ``labels`` (plus
    per-panel ``x_limits``/``y_limits`` the static plot pads the view with,
    and ``initial_states``/``D``/``schema_version`` that are not chart data).
    All plotted arrays are raw per-panel trajectories; the canvas engine
    autoranges from the decimated data, so the static padding is not carried
    across, but no derived quantity is at risk of drifting from the paper.
    """
    source = "sec03_transition/attractors"
    npz = _load("sec03_transition", "attractors")
    a_values = _require(npz, "A_values", source)
    labels = _require(npz, "labels", source)
    panels = []
    pairs: set[tuple[int, int]] = set()
    for idx in range(a_values.shape[0]):
        x = _require(npz, f"x_{idx}", source)
        y = _require(npz, f"y_{idx}", source)
        n_from = x.shape[0]
        sel = np.arange(0, n_from, max(1, n_from // max_points))[:max_points]
        pairs.add((n_from, sel.shape[0]))
        panels.append(
            {
                "title": f"a = {float(a_values[idx]):.4g}, {labels[idx]}",
                "traces": [
                    {"name": str(labels[idx]), "x": _round_list(x[sel]), "y": _round_list(y[sel])}
                ],
            }
        )
    return {
        "figure": source,
        "kind": "scatter",
        "decimation": _uniform_decimation(pairs),
        "axes": {"x": {"label": "x"}, "y": {"label": "y"}},
        "panels": panels,
    }


def export_xz_projections(max_points: int = 12000) -> dict[str, Any]:
    """sec06_three_torus/xz_projections: six (x_n, z_n) projections.

    Two of the six panels are rendered as hexbin visitation-density maps in
    the static figure (``plot_projections``, coupled_delayed.py); that density
    binning is a derived quantity and is not reproduced. Instead every panel
    exports the same raw (x, z) point cloud the density panels were built
    from, decimated for interactivity -- honest raw data, not a second
    implementation of the density estimate.
    """
    source = "sec06_three_torus/xz_projections"
    npz = _load("sec06_three_torus", "xz_projections")
    db_values = _require(npz, "DB_values", source)
    labels = _require(npz, "labels", source)
    panels = []
    n_from = None
    n_to = None
    for db, label in zip(db_values, labels, strict=True):
        key = f"DB_{db}_xz"
        xz = _require(npz, key, source)
        n_from = xz.shape[0]
        sel = np.arange(0, n_from, max(1, n_from // max_points))[:max_points]
        n_to = sel.shape[0]
        panels.append(
            {
                "title": f"D_B = {db:.3f}, {label}",
                "traces": [
                    {"name": str(label), "x": _round_list(xz[sel, 0]), "y": _round_list(xz[sel, 1])}
                ],
            }
        )
    return {
        "figure": source,
        "kind": "scatter",
        "decimation": _stride_decimation(n_from, n_to),
        "axes": {"x": {"label": "x"}, "y": {"label": "z"}},
        "panels": panels,
    }


def export_devils_staircase(n_bins: int = 4000) -> dict[str, Any]:
    """sec02_circle_map/devils_staircase: rotation number and Lyapunov exponent vs K.

    ``plot`` (circle_map.py) draws both rho and lam directly from the npz;
    the K_chaos_onset it also computes (a windowed noise-floor scan) only
    feeds a text annotation and is not reproduced. A plain stride decimation
    would flatten the plateau structure the figure exists to show, so this
    uses the min/max-per-bin helper instead.
    """
    source = "sec02_circle_map/devils_staircase"
    npz = _load("sec02_circle_map", "devils_staircase")
    a = _require(npz, "A", source)
    rho = _require(npz, "rho", source)
    lam = _require(npz, "lam", source)
    n = a.shape[0]
    a_rho, rho_ds = _minmax_decimate(a, rho, n_bins)
    a_lam, lam_ds = _minmax_decimate(a, lam, n_bins)
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _minmax_bin_decimation(n, n_bins),
        "axes": {"x": {"label": "Nonlinearity K"}, "y": {"label": "value"}},
        "marks": [{"axis": "y", "value": 1.0 / 5.0, "label": "ρ = 1/5"}],
        "panels": [
            {
                "title": "Rotation number staircase",
                "traces": [{"name": "ρ", "x": _round_list(a_rho), "y": _round_list(rho_ds)}],
            },
            {
                "title": "Lyapunov exponent",
                "traces": [{"name": "λ", "x": _round_list(a_lam), "y": _round_list(lam_ds)}],
            },
        ],
    }


def export_staircase_zoom(n_bins: int = 4000) -> dict[str, Any]:
    """sec02_circle_map/staircase_zoom: zoomed rotation-number staircase.

    ``rho`` is plotted directly from the npz by ``plot_zoom`` (circle_map.py);
    the zoom-box rectangle and rational-locking reference lines it also draws
    are static annotations computed from fixed constants, not derived from
    the data, and are reproduced here as marks. Min/max-per-bin decimation
    for the same plateau-preservation reason as devils_staircase.
    """
    source = "sec02_circle_map/staircase_zoom"
    npz = _load("sec02_circle_map", "staircase_zoom")
    a = _require(npz, "A", source)
    rho = _require(npz, "rho", source)
    n = a.shape[0]
    a_ds, rho_ds = _minmax_decimate(a, rho, n_bins)
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _minmax_bin_decimation(n, n_bins),
        "axes": {"x": {"label": "Nonlinearity K"}, "y": {"label": "Rotation number ρ"}},
        "marks": [
            {"axis": "y", "value": 2.0 / 9.0, "label": "2/9"},
            {"axis": "y", "value": 3.0 / 14.0, "label": "3/14"},
            {"axis": "y", "value": 4.0 / 19.0, "label": "4/19"},
            {"axis": "y", "value": 1.0 / 5.0, "label": "1/5"},
        ],
        "panels": [
            {
                "title": "Period-adding sequence approaching 1/5",
                "traces": [{"name": "ρ", "x": _round_list(a_ds), "y": _round_list(rho_ds)}],
            }
        ],
    }


def export_space_amplitude() -> dict[str, Any]:
    """sec09_pattern/space_amplitude: space-amplitude snapshot overlays.

    Each ``a_<a>_eps_<eps>_snap`` array (12, 100) is plotted directly, one
    trace per stored snapshot, by ``plot_space_amplitude`` (pattern_dynamics.py)
    -- no derivation. All 5 parameter panels x 12 snapshots are exported; the
    legend-wrap fix makes the resulting 12-trace legends displayable.
    """
    source = "sec09_pattern/space_amplitude"
    npz = _load("sec09_pattern", "space_amplitude")
    params = _require(npz, "params", source)
    panels = []
    for a, eps in params:
        key = f"a_{a}_eps_{eps}_snap"
        label_key = f"a_{a}_eps_{eps}_label"
        snapshots = _require(npz, key, source)
        n_snap, n_sites = snapshots.shape
        sites = np.arange(n_sites)
        # Stored label carries a literal newline for the static plot's two-line
        # title; a single space reads the same words on one line for the JSON.
        label = str(npz[label_key][0]).replace("\n", " ") if label_key in npz.files else ""
        traces = [
            {"name": f"iteration {i}", "x": _round_list(sites), "y": _round_list(snapshots[i])}
            for i in range(n_snap)
        ]
        title = f"a = {a:g}" + (f", {label}" if label else "")
        panels.append({"title": title, "traces": traces})
    return {
        "figure": source,
        "kind": "lines",
        "decimation": _no_decimation(int(params.shape[0] * 12 * 100)),
        "axes": {"x": {"label": "site i"}, "y": {"label": "x(i)"}},
        "marks": [{"axis": "y", "value": 0.0, "label": "x = 0"}],
        "panels": panels,
    }


EXPORTERS: tuple[tuple[str, str, Any], ...] = (
    ("sec11_diagnostics", "rqa_measures", export_rqa_measures),
    ("sec11_diagnostics", "test01_sweep", export_test01_sweep),
    ("sec11_diagnostics", "permutation_entropy", export_permutation_entropy),
    ("sec11_diagnostics", "complexity_entropy_plane", export_complexity_entropy_plane),
    ("sec07_fractalization", "correlation_dimension", export_correlation_dimension),
    ("sec05_oscillation", "lyapunov_vs_D", export_lyapunov_vs_d),
    ("sec06_three_torus", "lyapunov_vs_DB", export_lyapunov_vs_db),
    ("sec04_doubling", "map_I_attractors", export_map_i_attractors),
    ("sec02_circle_map", "arnold_tongues", export_arnold_tongues),
    ("sec09_pattern", "phase_diagram", export_phase_diagram),
    ("sec04_doubling", "map_IV_lyapunov", export_map_iv_lyapunov),
    ("sec03_transition", "phase_diagram", export_phase_diagram_sec03),
    ("sec10_gcm", "collective_lyapunov", export_collective_lyapunov),
    ("sec06_three_torus", "double_staircase", export_double_staircase),
    ("sec08_sti", "comoving_lyapunov", export_comoving_lyapunov),
    ("sec03_transition", "attractors", export_attractors_sec03),
    ("sec06_three_torus", "xz_projections", export_xz_projections),
    ("sec02_circle_map", "devils_staircase", export_devils_staircase),
    ("sec02_circle_map", "staircase_zoom", export_staircase_zoom),
    ("sec09_pattern", "space_amplitude", export_space_amplitude),
)


def main() -> int:
    """Export all figure-data JSON payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "site" / "data",
        help="Output root directory (default: site/data)",
    )
    args = parser.parse_args()

    rows: list[tuple[str, str, int]] = []
    for section, name, exporter in EXPORTERS:
        payload = exporter()
        out_dir = args.outdir / section
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.json"
        text = json.dumps(payload, separators=(",", ":"))
        out_path.write_text(text)
        rows.append((f"{section}/{name}", payload["kind"], len(text.encode("utf-8"))))

    header = f"{'figure':<45} {'kind':<8} {'bytes':>10}"
    print(header)
    print("-" * len(header))
    for figure, kind, nbytes in rows:
        print(f"{figure:<45} {kind:<8} {nbytes:>10,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
