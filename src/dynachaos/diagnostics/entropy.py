"""Entropy-based diagnostics for nonlinear time series.

Implements four commonly used complexity measures:
- Sample entropy (SampEn)
- Approximate entropy (ApEn)
- Fuzzy entropy (FuzzyEn)
- Multiscale entropy (MSE)

References
----------
Richman, J.S. & Moorman, J.R. (2000). Physiological time-series analysis
  using approximate entropy and sample entropy. Am. J. Physiol. Heart Circ.
  Physiol., 278(6), H2039-H2049.
Pincus, S.M. (1991). Approximate entropy as a measure of system complexity.
  Proc. Natl. Acad. Sci. USA, 88(6), 2297-2301.
Chen, W. et al. (2007). Characterization of surface EMG signal based on fuzzy
  entropy. Medical Engineering & Physics, 29(2), 164-169.
Costa, M. et al. (2002). Multiscale entropy analysis of complex physiologic
  time series. Phys. Rev. Lett., 89(6), 068102.
"""

import os
import time

import numpy as np

from dynachaos.diagnostics._validation import finite_positive_scalar, finite_series_1d
from dynachaos.diagnostics.recurrence import embed_time_delay
from dynachaos.utils.system import get_rss_mb

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import apen_counts as _apen_counts_rs
    from dynachaos._rust import correlation_counts as _correlation_counts_rs
    from dynachaos._rust import fuzzy_entropy_sum as _fuzzy_entropy_sum_rs

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _apen_counts_rs = None
    _correlation_counts_rs = None
    _fuzzy_entropy_sum_rs = None


def _default_r(x):
    """Default tolerance: 0.2 * std(x)."""
    return 0.2 * np.std(x, ddof=1)


def _as_finite_series(x):
    """Return a 1D finite float64 series."""
    return finite_series_1d(x, name="x")


def _print_timing(backend, N, n_pairs, elapsed):
    """Print timing and memory usage in the diagnostics style."""
    throughput = n_pairs / elapsed / 1e6 if elapsed > 0 else 0.0
    print(
        f"  entropy ({backend}): N={N:,}, pairs={n_pairs:,}, "
        f"time={elapsed:.3f}s, {throughput:.1f}M pairs/s, RSS={get_rss_mb():.0f} MB"
    )


def _embed_pair(x, m):
    """Return (traj_m, traj_m1) with traj_m trimmed to match traj_m1 length."""
    traj_m1 = embed_time_delay(x, m + 1, 1)
    traj_m = embed_time_delay(x, m, 1)[: len(traj_m1)]
    return traj_m, traj_m1


def _correlation_count_python(traj, r, theiler_window=0):
    """Count upper-triangle template matches with Chebyshev distance < r."""
    n_pts = len(traj)
    count = 0
    for i in range(n_pts):
        j_start = i + theiler_window + 1
        if j_start >= n_pts:
            continue
        dists = np.max(np.abs(traj[j_start:] - traj[i]), axis=1)
        count += int(np.count_nonzero(dists < r))
    return count


def _fuzzy_entropy_sum_python(traj, r, n, theiler_window=0):
    """Sum exp(-(d/r)^n) over upper-triangle pairs using Chebyshev distance."""
    n_pts = len(traj)
    total = 0.0
    n_float = float(n)
    for i in range(n_pts):
        j_start = i + theiler_window + 1
        if j_start >= n_pts:
            continue
        dists = np.max(np.abs(traj[j_start:] - traj[i]), axis=1)
        total += float(np.exp(-((dists / r) ** n_float)).sum())
    return total


def _apen_counts_python(traj, r):
    """Count self-included template matches with Chebyshev distance <= r."""
    n_templates = len(traj)
    counts = np.empty(n_templates, dtype=np.int64)
    for i in range(n_templates):
        count = 0
        for j in range(n_templates):
            d_max = np.max(np.abs(traj[i] - traj[j]))
            if d_max <= r:  # ApEn uses <= per Pincus (1991)
                count += 1
        counts[i] = count
    return counts


def sample_entropy(x, m=2, r=None, verbose=False):
    """Compute sample entropy (SampEn) from a scalar time series.

    SampEn measures regularity by comparing template-match probabilities at
    lengths ``m`` and ``m + 1`` without counting self-matches.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input scalar time series.
    m : int, default=2
        Embedding dimension for the template length.
    r : float or None, default=None
        Match tolerance. If None, uses ``0.2 * std(x, ddof=1)``.
    verbose : bool, default=False
        If True, print timing and memory information.

    Returns
    -------
    float
        Sample entropy value ``-log(A / B)``.  Returns ``np.nan`` when no
        m-length template matches exist (B = 0, undefined).  Returns
        ``np.inf`` when m-length matches exist but no (m+1)-length ones
        (A = 0, equivalent to −ln(0)).

    References
    ----------
    Richman, J.S. & Moorman, J.R. (2000), Am. J. Physiol. Heart Circ.
    Physiol. 278(6), H2039-H2049.
    """
    x = _as_finite_series(x)
    if m < 1:
        raise ValueError("m must be >= 1")
    if len(x) < m + 1:
        return np.inf

    if r is None:
        r = _default_r(x)
    r = finite_positive_scalar(r, name="r")

    traj_m, traj_m1 = _embed_pair(x, m)
    n_m = len(traj_m)
    if n_m < 2:
        return np.inf

    r_values = np.array([r], dtype=np.float64)
    n_pairs = n_m * (n_m - 1)  # both trajectories same length; total upper-triangle pairs
    t0 = time.perf_counter()

    if _RUST_AVAILABLE:
        b_count = int(
            np.asarray(_correlation_counts_rs(traj_m, r_values, 0, True), dtype=np.int64)[0]
        )
        a_count = int(
            np.asarray(_correlation_counts_rs(traj_m1, r_values, 0, True), dtype=np.int64)[0]
        )
        backend = "Rust"
    else:
        b_count = _correlation_count_python(traj_m, r, theiler_window=0)
        a_count = _correlation_count_python(traj_m1, r, theiler_window=0)
        backend = "Python"

    elapsed = time.perf_counter() - t0
    if verbose:
        _print_timing(backend, len(x), n_pairs, elapsed)

    if b_count == 0:
        return np.nan  # no m-length matches: undefined
    if a_count == 0:
        return np.inf  # no (m+1)-length matches: -ln(0/B) = inf
    return float(-np.log(a_count / b_count))


def approximate_entropy(x, m=2, r=None, verbose=False):
    """Compute approximate entropy (ApEn) from a scalar time series.

    ApEn follows Pincus (1991) and includes self-matches in template counts.
    This inclusion introduces a known bias for short series.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input scalar time series.
    m : int, default=2
        Embedding dimension for the template length.
    r : float or None, default=None
        Match tolerance. If None, uses ``0.2 * std(x, ddof=1)``.
    verbose : bool, default=False
        If True, print timing and memory information.

    Returns
    -------
    float
        Approximate entropy value ``phi(m) - phi(m + 1)``.

    References
    ----------
    Pincus, S.M. (1991), Proc. Natl. Acad. Sci. USA 88(6), 2297-2301.
    """
    x = _as_finite_series(x)
    if m < 1:
        raise ValueError("m must be >= 1")
    if len(x) < m + 1:
        return np.inf

    if r is None:
        r = _default_r(x)
    r = finite_positive_scalar(r, name="r")

    def _phi(order):
        traj = embed_time_delay(x, order, 1)
        n_templates = len(traj)
        if _RUST_AVAILABLE:
            counts = np.asarray(_apen_counts_rs(traj, r), dtype=np.float64)
        else:
            counts = _apen_counts_python(traj, r).astype(np.float64)
        c = counts / n_templates
        return float(np.mean(np.log(c)))

    n_m = len(x) - m + 1
    n_m1 = len(x) - m
    n_pairs = n_m * n_m + n_m1 * n_m1
    t0 = time.perf_counter()
    apen = _phi(m) - _phi(m + 1)
    elapsed = time.perf_counter() - t0

    if verbose:
        backend = "Rust" if _RUST_AVAILABLE else "Python"
        _print_timing(backend, len(x), n_pairs, elapsed)
    return float(apen)


def fuzzy_entropy(x, m=2, r=None, n=2, verbose=False):
    """Compute fuzzy entropy (FuzzyEn) from a scalar time series.

    FuzzyEn replaces hard match counting with smooth membership
    ``exp(-(d/r)^n)`` and uses mean-centered templates before distance
    evaluation, following the original formulation.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input scalar time series.
    m : int, default=2
        Embedding dimension for the template length.
    r : float or None, default=None
        Fuzzy tolerance scale. If None, uses ``0.2 * std(x, ddof=1)``.
    n : int, default=2
        Fuzzy exponent in ``exp(-(d/r)^n)``.
    verbose : bool, default=False
        If True, print timing and memory information.

    Returns
    -------
    float
        Fuzzy entropy value ``-log(phi_{m+1} / phi_m)``.

    References
    ----------
    Chen, W. et al. (2007), Medical Engineering & Physics 29(2), 164-169.
    """
    x = _as_finite_series(x)
    if m < 1:
        raise ValueError("m must be >= 1")
    if n <= 0:
        raise ValueError("n must be > 0")
    if len(x) < m + 1:
        return np.inf

    if r is None:
        r = _default_r(x)
    r = finite_positive_scalar(r, name="r")

    traj_m, traj_m1 = _embed_pair(x, m)
    n_m = len(traj_m)
    if n_m < 2:
        return np.inf

    traj_m = traj_m - traj_m.mean(axis=1, keepdims=True)
    traj_m1 = traj_m1 - traj_m1.mean(axis=1, keepdims=True)

    n_pairs_total = n_m * (n_m - 1)  # both same length; total upper-triangle pairs
    t0 = time.perf_counter()

    if _RUST_AVAILABLE:
        sum_m = float(_fuzzy_entropy_sum_rs(traj_m, r, int(n), 0))
        sum_m1 = float(_fuzzy_entropy_sum_rs(traj_m1, r, int(n), 0))
        backend = "Rust"
    else:
        sum_m = _fuzzy_entropy_sum_python(traj_m, r, n, theiler_window=0)
        sum_m1 = _fuzzy_entropy_sum_python(traj_m1, r, n, theiler_window=0)
        backend = "Python"

    elapsed = time.perf_counter() - t0
    if verbose:
        _print_timing(backend, len(x), n_pairs_total, elapsed)

    denom = n_m * (n_m - 1)  # same for both (trimmed to equal length)
    phi_m = 2.0 * sum_m / denom
    phi_m1 = 2.0 * sum_m1 / denom

    if phi_m <= 0.0 or phi_m1 <= 0.0:
        return np.inf
    return float(-np.log(phi_m1 / phi_m))


def multiscale_entropy(x, scales=None, m=2, r=None, verbose=False):
    """Compute multiscale entropy (MSE) using fixed-tolerance SampEn.

    Coarse-grains the signal at each scale and computes sample entropy with a
    tolerance fixed from the original (unscaled) series.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input scalar time series.
    scales : iterable of int or None, default=None
        Scale factors. If None, uses ``range(1, 21)``.
    m : int, default=2
        Embedding dimension passed to sample entropy.
    r : float or None, default=None
        SampEn tolerance fixed across scales. If None, uses
        ``0.15 * std(x, ddof=1)`` from the original series (Costa et al. 2002
        convention). Pass an explicit value to use the SampEn default of 0.2.
    verbose : bool, default=False
        If True, print aggregate timing and memory information.

    Returns
    -------
    numpy.ndarray
        MSE values for each requested scale (same order as ``scales``).

    References
    ----------
    Costa, M. et al. (2002), Phys. Rev. Lett. 89(6), 068102.
    """
    x = _as_finite_series(x)
    if m < 1:
        raise ValueError("m must be >= 1")

    if scales is None:
        scales = range(1, 21)
    scales = list(scales)

    if r is None:
        r = 0.15 * np.std(x, ddof=1)  # Costa et al. (2002) convention
    r = finite_positive_scalar(r, name="r")

    t0 = time.perf_counter()
    n_pairs_total = 0
    results = []
    for tau in scales:
        tau_int = int(tau)
        if tau_int < 1:
            raise ValueError("all scales must be >= 1")

        n_tau = len(x) // tau_int
        y = x[: n_tau * tau_int].reshape(n_tau, tau_int).mean(axis=1)
        results.append(sample_entropy(y, m=m, r=r, verbose=False))

        n_templates = max(0, len(y) - m)
        n_pairs_total += n_templates * (n_templates - 1)

    elapsed = time.perf_counter() - t0
    if verbose:
        _print_timing("MSE", len(x), n_pairs_total, elapsed)

    return np.array(results, dtype=np.float64)


__all__ = [
    "sample_entropy",
    "approximate_entropy",
    "fuzzy_entropy",
    "multiscale_entropy",
]
