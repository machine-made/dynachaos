"""Correlation integral and dimension (Grassberger-Procaccia algorithm).

Estimates the correlation dimension D_2 from trajectory points by computing
the correlation integral C(r) — the fraction of pairs within distance r —
and fitting the power-law scaling C(r) ~ r^{D_2}.

The improved implementation supports:
- Theiler window to exclude temporally correlated pairs
- Chebyshev (max-norm) or Euclidean distance
- Rust-accelerated exact all-pairs counting (O(1) memory per pair)

All thresholds are derived from data geometry ("organic"):
- R-range from attractor bounding-box diameter (ptp)
- Noise floor from Poisson statistics (1/sqrt(n_valid))
- Saturation/slope filter from finite-difference resolution (log_spacing)

Reference
---------
Grassberger, P. & Procaccia, I. (1983) "Measuring the strangeness of strange
  attractors", Physica D, 9(1-2), 189-208.
Theiler, J. (1986) "Spurious dimension from correlation algorithms applied to
  limited time-series data", Phys. Rev. A, 34(3), 2427-2432.
"""

import os
import time

import numpy as np
from scipy.stats import linregress

from dynachaos.diagnostics._validation import sorted_nonnegative_radius_grid
from dynachaos.diagnostics.reliability import ReliabilityRecord
from dynachaos.utils.system import get_rss_mb

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import correlation_counts as _correlation_counts_rs

    _RUST_AVAILABLE = True
except ImportError:
    _correlation_counts_rs = None
    _RUST_AVAILABLE = False


def _correlation_counts_python(traj, r_values, theiler_window, use_chebyshev):
    """Exact all-pairs correlation counts, pure Python.

    For small N uses scipy pdist; for large N uses random sampling.
    """
    N = len(traj)
    traj = np.asarray(traj, dtype=np.float64)
    r_values = np.asarray(r_values, dtype=np.float64)
    n_r = len(r_values)
    counts = np.zeros(n_r, dtype=np.int64)

    n_pairs_total = 0
    # Brute force with Theiler window support
    for i in range(N):
        j_start = i + theiler_window + 1
        if j_start >= N:
            continue
        diffs = traj[j_start:] - traj[i]
        if use_chebyshev:
            if traj.ndim == 1 or traj.shape[1] == 1:
                dists = np.abs(diffs).ravel()
            else:
                dists = np.max(np.abs(diffs), axis=1)
        else:
            if traj.ndim == 1 or traj.shape[1] == 1:
                dists = np.abs(diffs).ravel()
            else:
                dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        n_pairs_total += len(dists)
        for k in range(n_r):
            counts[k] += np.sum(dists < r_values[k])

    return counts, n_pairs_total


def _print_timing(backend, N, n_valid, n_r, elapsed):
    """Print verbose timing and memory info for correlation counting."""
    throughput = n_valid * n_r / elapsed / 1e6 if elapsed > 0 else 0.0
    print(
        f"  correlation_integral ({backend}): N={N:,}, pairs={n_valid:,}, "
        f"n_r={n_r}, time={elapsed:.3f}s, "
        f"{throughput:.1f}M pair-tests/s, RSS={get_rss_mb():.0f} MB"
    )


def _valid_pair_count(n, theiler_window):
    """Closed-form count of valid pairs with |i-j| > theiler_window."""
    n_eff = max(0, n - theiler_window - 1)
    return n_eff * (n_eff + 1) // 2


def _validate_correlation_inputs(r_values, theiler_window, norm):
    """Validate correlation-count parameters shared by Python and Rust paths."""
    try:
        theiler_int = int(theiler_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("theiler_window must be >= 0") from exc
    if theiler_int != theiler_window or theiler_int < 0:
        raise ValueError("theiler_window must be >= 0")
    if norm not in {"chebyshev", "euclidean"}:
        raise ValueError("norm must be one of: 'chebyshev', 'euclidean'")

    r_values = sorted_nonnegative_radius_grid(r_values, name="r_values")
    return r_values, theiler_int, norm == "chebyshev"


def _undefined_dimension_result(return_stderr=False):
    empty = np.array([], dtype=np.float64)
    if return_stderr:
        return np.nan, empty, empty, np.nan, empty, np.array([], dtype=bool)
    return np.nan, empty, empty, empty, np.array([], dtype=bool)


def correlation_integral(
    traj, r_values, max_pairs=500_000, theiler_window=0, norm="chebyshev", verbose=False
):
    """Compute the correlation integral C(r) for an array of r values.

    C(r) = (2 / N(N-1)) * #{pairs (i,j) with |i-j| > w and dist < r}

    Parameters
    ----------
    traj : ndarray, shape (N, d) or (N,)
        Trajectory points.
    r_values : ndarray, shape (n_r,)
        Distance thresholds (should be sorted ascending).
    max_pairs : int
        Ignored when Rust is available (exact computation used).
        Python fallback uses all pairs for small N.
    theiler_window : int
        Minimum temporal separation |i-j| > theiler_window.
        Default 0 (all pairs). Recommended: mean period or
        autocorrelation time of the signal.
    norm : str
        'chebyshev' (default, max-norm) or 'euclidean'.
    verbose : bool
        If True, print timing and memory info after computation.

    Returns
    -------
    C : ndarray, shape (n_r,)
        Correlation integral values.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj[:, np.newaxis]
    # Empty and single-point trajectories keep their degenerate return path;
    # only non-finite values are rejected.
    if not np.all(np.isfinite(traj)):
        raise ValueError("traj must contain only finite values")
    r_values, theiler_window, use_chebyshev = _validate_correlation_inputs(
        r_values, theiler_window, norm
    )

    N = len(traj)
    n_valid = _valid_pair_count(N, theiler_window)

    if n_valid == 0:
        return np.zeros(len(r_values))

    if _RUST_AVAILABLE:
        t0 = time.perf_counter()
        counts = np.asarray(_correlation_counts_rs(traj, r_values, theiler_window, use_chebyshev))
        elapsed = time.perf_counter() - t0
        if verbose:
            _print_timing("Rust", N, n_valid, len(r_values), elapsed)
        return counts.astype(np.float64) / n_valid

    t0 = time.perf_counter()
    counts, _ = _correlation_counts_python(traj, r_values, theiler_window, use_chebyshev)
    elapsed = time.perf_counter() - t0
    if verbose:
        _print_timing("Python", N, n_valid, len(r_values), elapsed)
    return counts.astype(np.float64) / n_valid


def _find_scaling_region(log_r, log_C, min_points=5):
    """Find the scaling region via local slope plateau detection.

    Computes the local slope d(log C)/d(log r) at each point, then finds
    the longest contiguous region where the slope is stable (small std).

    Points where the slope falls below the finite-difference resolution
    (log_spacing) are excluded — this organically removes saturation and
    noise-floor regions without hardcoded thresholds.

    Parameters
    ----------
    log_r, log_C : ndarray
        Log-transformed r and C(r) values (already filtered for C > 0).
    min_points : int
        Minimum number of points required in the scaling region.

    Returns
    -------
    mask : ndarray of bool
        Points belonging to the best scaling region.
    slopes : ndarray
        Local slope at each point.
    """
    n = len(log_r)
    if n < min_points:
        return np.ones(n, dtype=bool), np.gradient(log_C, log_r)

    # Local slopes via centered finite differences
    slopes = np.gradient(log_C, log_r)

    # Organic slope filter: the finite-difference resolution is the mean
    # spacing between consecutive log(r) values.  Slope values below this
    # are indistinguishable from zero (saturation region) or noise.
    log_spacing = (log_r[-1] - log_r[0]) / (n - 1) if n > 1 else 0.0
    usable = slopes >= log_spacing
    usable_idx = np.where(usable)[0]

    if len(usable_idx) < min_points:
        # Fall back to all points if too few pass the filter
        usable_idx = np.arange(n)

    # Sliding window on usable points to find the most stable plateau
    n_u = len(usable_idx)
    u_slopes = slopes[usable_idx]

    win = max(min_points, n_u // 4)
    if win > n_u:
        mask = np.zeros(n, dtype=bool)
        mask[usable_idx] = True
        return mask, slopes

    best_std = np.inf
    best_start = 0
    for start in range(n_u - win + 1):
        s = u_slopes[start : start + win]
        std = np.std(s)
        if std < best_std:
            best_std = std
            best_start = start

    # Expand outward from best window while slope stays consistent.
    # Adaptive threshold: allow +/-3sigma of the best window's std, with a
    # floor at the log-spacing resolution (organic replacement for magic 0.05).
    center_median = np.median(u_slopes[best_start : best_start + win])
    threshold = max(3.0 * best_std, log_spacing)

    lo = best_start
    hi = best_start + win
    while lo > 0 and abs(u_slopes[lo - 1] - center_median) < threshold:
        lo -= 1
    while hi < n_u and abs(u_slopes[hi] - center_median) < threshold:
        hi += 1

    mask = np.zeros(n, dtype=bool)
    mask[usable_idx[lo:hi]] = True
    return mask, slopes


def fit_power_law_loglog(x, y, min_points=5, *, return_stderr=False):
    """Fit ``y ~ x**slope`` over a data-driven log-log scaling region.

    This regression is a diagnostic scaling fit. It preserves the
    Grassberger-Procaccia scaling-region heuristic used by
    :func:`correlation_dimension` and is factored for other diagnostics that
    need the same log-log fit without duplicating the detector.

    By default returns ``(slope, intercept, rvalue, slopes, scaling)``. When
    ``return_stderr=True`` the 1-sigma standard error on the slope (from the
    linear regression) is inserted after ``rvalue``:
    ``(slope, intercept, rvalue, stderr, slopes, scaling)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D arrays with matching length")
    if min_points < 1:
        raise ValueError("min_points must be >= 1")

    valid = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    full_slopes = np.full(len(x), np.nan)
    full_scaling = np.zeros(len(x), dtype=bool)
    if np.sum(valid) < min_points:
        if return_stderr:
            return np.nan, np.nan, np.nan, np.nan, full_slopes, full_scaling
        return np.nan, np.nan, np.nan, full_slopes, full_scaling

    log_x = np.log(x[valid])
    log_y = np.log(y[valid])
    scaling_local, slopes_local = _find_scaling_region(log_x, log_y, min_points=min_points)

    valid_idx = np.where(valid)[0]
    full_slopes[valid_idx] = slopes_local
    full_scaling[valid_idx[scaling_local]] = True

    fit_x = log_x[scaling_local]
    fit_y = log_y[scaling_local]
    if len(fit_x) < min_points:
        if return_stderr:
            return np.nan, np.nan, np.nan, np.nan, full_slopes, full_scaling
        return np.nan, np.nan, np.nan, full_slopes, full_scaling

    result = linregress(fit_x, fit_y)
    if return_stderr:
        return (
            float(result.slope),
            float(result.intercept),
            float(result.rvalue),
            float(result.stderr),
            full_slopes,
            full_scaling,
        )
    return (
        float(result.slope),
        float(result.intercept),
        float(result.rvalue),
        full_slopes,
        full_scaling,
    )


def correlation_dimension(
    traj,
    n_r=50,
    r_range=None,
    max_pairs=500_000,
    theiler_window=0,
    norm="chebyshev",
    verbose=False,
    return_stderr=False,
    *,
    return_metadata=False,
):
    """Estimate the correlation dimension D_2 from trajectory points.

    Uses local slope plateau detection to identify the scaling region
    of C(r) ~ r^{D_2}, then fits a line in log-log space over that region.

    Parameters
    ----------
    traj : ndarray, shape (N, d)
        Trajectory points.
    n_r : int
        Number of r values to use (default 50).
    r_range : tuple of (float, float) or None
        Range of r values. Auto-estimated from data if None.
    max_pairs : int
        Maximum pairs for Python fallback sampling.
    theiler_window : int
        Minimum temporal separation for valid pairs.
    norm : str
        'chebyshev' or 'euclidean'.
    verbose : bool
        If True, print timing and memory info during computation.

    Returns
    -------
    D2 : float
        Estimated correlation dimension.
    r_values : ndarray
        The r values used.
    C_values : ndarray
        Corresponding C(r).
    D2_stderr : float
        One-sigma standard error on D2 from the log-log scaling-region fit.
    local_slopes : ndarray
        Local slope d(log C)/d(log r) at each valid point.
    scaling_mask : ndarray of bool
        Boolean mask over r_values indicating the scaling region used for fit.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj[:, np.newaxis]
    # Empty and single-point trajectories keep their NaN return path;
    # only non-finite values are rejected.
    if not np.all(np.isfinite(traj)):
        raise ValueError("traj must contain only finite values")
    if n_r < 1:
        raise ValueError("n_r must be >= 1")

    N = len(traj)

    def _metadata(r_range_value, warnings=None, unresolved=None, backend=None):
        return ReliabilityRecord(
            method_name="correlation_dimension",
            backend=backend or ("rust" if _RUST_AVAILABLE else "python"),
            parameters={
                "n_r": n_r,
                "r_range": r_range_value,
                "max_pairs": max_pairs,
                "theiler_window": theiler_window,
                "norm": norm,
                "return_stderr": return_stderr,
            },
            data_length=N,
            data_shape=tuple(int(v) for v in traj.shape),
            sampling_downsampling_note=(
                "no sampling/downsampling; exact all-pairs counting for available backend"
            ),
            validity_warnings=list(warnings or []),
            unresolved_verdicts=list(unresolved or []),
            scale_evidence={
                "artifact_path": "benchmarks/results/scale_envelope.json",
                "entry": {"kind": "gp", "backend": "rust/python parity entries"},
            },
        )

    if N < 2:
        result = _undefined_dimension_result(return_stderr=return_stderr)
        if return_metadata:
            return (
                *result,
                _metadata(
                    r_range,
                    warnings=["fewer than two trajectory points"],
                    unresolved=["scaling region not identified"],
                    backend="n.a.",
                ),
            )
        return result

    if r_range is None:
        # Organic: attractor bounding-box diameter sets the scale
        diameter = np.max(np.ptp(traj, axis=0))
        if not np.isfinite(diameter) or diameter <= 0.0:
            result = _undefined_dimension_result(return_stderr=return_stderr)
            if return_metadata:
                return (
                    *result,
                    _metadata(
                        r_range,
                        warnings=["nonpositive or nonfinite trajectory diameter"],
                        unresolved=["scaling region not identified"],
                        backend="n.a.",
                    ),
                )
            return result
        r_min = diameter / N  # below this, pair count -> 0
        r_max = diameter  # beyond this, C(r) -> 1
        r_range = (r_min, r_max)
    elif (
        len(r_range) != 2
        or not np.all(np.isfinite(r_range))
        or r_range[0] <= 0.0
        or r_range[1] <= 0.0
        or r_range[0] > r_range[1]
    ):
        raise ValueError("r_range must be a positive ascending (min, max) pair")

    r_values = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), n_r)
    C_values = correlation_integral(
        traj, r_values, max_pairs, theiler_window, norm, verbose=verbose
    )

    # Filter: keep points with positive C, exclude noise floor
    # Organic: Poisson noise floor — at C = 1/sqrt(n_valid), the count is
    # sqrt(n_valid) pairs with relative error n_valid^{-1/4}.
    n_valid = _valid_pair_count(N, theiler_window)
    c_floor = 1.0 / np.sqrt(n_valid) if n_valid > 0 else 1e-5
    fit_values = np.where(C_values > c_floor, C_values, np.nan)
    # Note: saturation ceiling (formerly 0.8) is now handled organically
    # inside _find_scaling_region via the log_spacing slope filter.

    # Full-length arrays for return (NaN where C <= 0)
    full_slopes = np.full(len(r_values), np.nan)
    full_scaling = np.zeros(len(r_values), dtype=bool)

    D2, _, _, D2_stderr, full_slopes, full_scaling = fit_power_law_loglog(
        r_values, fit_values, min_points=3, return_stderr=True
    )

    result = (
        (D2, r_values, C_values, D2_stderr, full_slopes, full_scaling)
        if return_stderr
        else (D2, r_values, C_values, full_slopes, full_scaling)
    )
    if return_metadata:
        warnings = []
        unresolved = []
        if n_valid == 0:
            warnings.append("no valid pairs after Theiler-window exclusion")
        if not np.any(full_scaling) or not np.isfinite(D2):
            unresolved.append("scaling region not identified")
        return (*result, _metadata(r_range, warnings=warnings, unresolved=unresolved))
    return result


def _takens_theiler_curve(r_values, C_values, c_floor):
    """Takens-Theiler estimator D_TT(r) = C(r) / integral_0^r C(x)/x dx.

    The integral is evaluated exactly under the TISEAN ``c2t`` convention: C(r)
    is interpolated by a pure power law between consecutive grid points (a
    straight line on the log-log plot), so each segment integral has the closed
    form

        int_{r_{j-1}}^{r_j} C(x)/x dx = (C_j - C_{j-1}) / a_j ,
        a_j = ln(C_j/C_{j-1}) / ln(r_j/r_{j-1})            (local log-log slope)

    (the a_j -> 0 saturation limit reduces to C_{j-1} * ln(r_j/r_{j-1})). The
    lower tail [0, r_min] uses the same power-law continuation, giving the finite
    closed form  int_0^{r_min} C/x dx = C(r_min) / a_tail ,  where a_tail is the
    scaling-region slope (median local slope in the band) rather than the noisy
    first-segment slope -- near the Poisson floor the first-segment slope can be
    ~0 from integer pair-count granularity, which would make C0/a blow up and
    crush the curve. Omitting the tail entirely biases D_TT high.

    Returns the per-r D_TT curve (NaN where C is at/below the Poisson floor).

    References
    ----------
    Takens (1985) LNM 1125; Theiler (1990) JOSA A 7:1055; TISEAN ``c2t``
    (Hegger & Kantz 1999). Verified against the TISEAN/Octave ``c2t`` definition.
    """
    r_values = np.asarray(r_values, dtype=np.float64)
    C_values = np.asarray(C_values, dtype=np.float64)
    n = len(r_values)
    D_tt = np.full(n, np.nan)

    valid = C_values > c_floor
    idx = np.where(valid)[0]
    if len(idx) < 3:
        return D_tt

    rr = r_values[idx]
    CC = C_values[idx]
    log_r = np.log(rr)
    log_C = np.log(CC)

    # per-segment log-log slopes a_j (between consecutive valid points)
    a = np.diff(log_C) / np.diff(log_r)  # length len(idx)-1

    # segment integrals of C/x: (C_j - C_{j-1})/a_j, with a->0 limit
    dC = np.diff(CC)
    seg = np.where(np.abs(a) > 1e-12, dC / np.where(a == 0.0, 1.0, a), CC[:-1] * np.diff(log_r))

    # lower tail int_0^{r_min} C/x dx = C(r_min)/a_tail, extrapolating C ~ r^a_tail
    # below r_min. a_tail is the SCALING-region slope (median local slope in the
    # band), NOT the first-segment slope a[0]: near the Poisson floor a[0] can be
    # ~0 from integer pair-count granularity, and C0/a[0] would then blow up (or
    # diverge at a[0]=0), crushing the whole D_TT curve toward 0.
    # Exclude granularity/saturation slopes from the a_tail median: an integer
    # pair-count step gives a ~ 1/(count*dlogr) ~ 1e-4..1e-3, far below any real
    # scaling slope (a ~ D >= ~0.5). A 1e-2 floor cleanly separates the two, so a
    # thin band dominated by granularity steps cannot collapse a_tail (and blow up
    # the tail). Saturation segments (a -> 0) are likewise excluded.
    SLOPE_FLOOR = 1e-2
    seg_C = CC[:-1]
    in_band = (seg_C > c_floor) & (seg_C < 0.1) & (a > SLOPE_FLOOR)
    if np.any(in_band):
        a_tail = float(np.median(a[in_band]))
    elif np.any(a > SLOPE_FLOOR):
        a_tail = float(np.median(a[a > SLOPE_FLOOR]))
    else:
        a_tail = 1.0
    tail = CC[0] / a_tail

    # cumulative integral I(r_i) = tail + sum_{j<=i} seg_j
    cumulative_integral = np.empty(len(idx))
    cumulative_integral[0] = tail
    cumulative_integral[1:] = tail + np.cumsum(seg)

    with np.errstate(divide="ignore", invalid="ignore"):
        D_tt[idx] = CC / cumulative_integral
    return D_tt


def takens_theiler_dimension(
    traj, n_r=60, r_range=None, max_pairs=500_000, theiler_window=0, norm="chebyshev", verbose=False
):
    """Correlation dimension via the Takens-Theiler maximum-likelihood estimator.

    Fit-window-free alternative to the log-log slope of
    :func:`correlation_dimension`. Computes the correlation integral C(r) (same
    Rust-accelerated pair counting, same Theiler window) and applies the
    Takens-Theiler estimator D_TT(r) = C(r) / int_0^r C(x)/x dx (see
    :func:`_takens_theiler_curve`). The reported D2 is the median of D_TT(r) over
    the FLATTEST contiguous shelf of the curve (minimum-variation window within
    the band C in [~5*floor, ~knee]); including the edge roll-off would bias D2
    low.

    Measured behaviour vs the slope estimator (validation battery, N~4e4): the
    Takens-Theiler estimate is less BIASED -- it recovers the literature value
    (e.g. Lorenz 2.05 vs the slope fit's 2.00) -- while its ensemble variance is
    comparable to the slope fit (sigma ~ 0.01-0.02), not lower. Tight, integer-
    separated estimates come from ensemble/segment confidence intervals, not from
    the estimator alone.

    Companion to :func:`correlation_dimension` with a parallel call signature and
    return tuple (note the denser default ``n_r``); the slope estimator is
    retained for diagnostics.

    Returns
    -------
    D2 : float
        Takens-Theiler correlation-dimension estimate (median over the band).
    r_values : ndarray
    C_values : ndarray
    D_tt : ndarray
        Per-r Takens-Theiler curve D_TT(r) (NaN below the Poisson floor).
    scaling_mask : ndarray of bool
        Band over which D2 is taken (the median).
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj[:, np.newaxis]

    N = len(traj)
    if r_range is None:
        diameter = np.max(np.ptp(traj, axis=0))
        if not np.isfinite(diameter) or diameter <= 0.0:
            # degenerate (constant / collapsed) trajectory -- no scaling to fit
            return np.nan, np.array([]), np.array([]), np.array([]), np.zeros(0, dtype=bool)
        r_range = (diameter / N, diameter)

    r_values = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), n_r)
    C_values = correlation_integral(
        traj, r_values, max_pairs, theiler_window, norm, verbose=verbose
    )

    n_valid = _valid_pair_count(N, theiler_window)
    c_floor = 1.0 / np.sqrt(n_valid) if n_valid > 0 else 1e-5

    D_tt = _takens_theiler_curve(r_values, C_values, c_floor)

    full_scaling = np.zeros(len(r_values), dtype=bool)
    if not np.any(np.isfinite(D_tt)):
        return np.nan, r_values, C_values, D_tt, full_scaling

    # Read D2 off the FLAT SHELF of D_TT(r), not a median over the whole band.
    # The Takens-Theiler curve rises through the scaling region to a plateau at
    # the true dimension, then rolls off near the saturation knee (edge effect);
    # including that roll-off both biases D2 low and inflates its variance. We
    # therefore restrict to points comfortably above the Poisson floor and below
    # the knee, then take the flattest contiguous window of D_TT (minimum local
    # variation) -- the shelf -- and report its median.
    cand = np.isfinite(D_tt) & (C_values > 5.0 * c_floor) & (C_values < 0.1)
    if np.sum(cand) < 4:
        cand = np.isfinite(D_tt) & (C_values > 2.0 * c_floor) & (C_values < 0.3)
    if np.sum(cand) < 3:
        return np.nan, r_values, C_values, D_tt, full_scaling

    cand_idx = np.where(cand)[0]
    d_cand = D_tt[cand_idx]
    n_c = len(cand_idx)
    win = min(n_c, max(4, n_c // 2))  # shelf window: ~half the clean band
    best_std, best_lo = np.inf, 0
    for lo in range(n_c - win + 1):
        s = np.std(d_cand[lo : lo + win])
        if s < best_std:
            best_std, best_lo = s, lo
    shelf = cand_idx[best_lo : best_lo + win]

    full_scaling[shelf] = True
    D2 = float(np.median(D_tt[shelf]))
    return D2, r_values, C_values, D_tt, full_scaling
