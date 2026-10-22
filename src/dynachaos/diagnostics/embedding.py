"""Nonlinear time series embedding: delay selection and dimension estimation.

Provides tools for selecting embedding parameters (time delay tau, embedding
dimension d) required before applying delay-coordinate reconstruction to a
scalar time series.

Algorithms
----------
- **AMI** (Average Mutual Information): selects optimal tau as the first local
  minimum of the delayed mutual information I(tau).  Ref: Fraser & Swinney (1986).
- **Cao's method**: determines the minimum sufficient embedding dimension via
  the E1(d)/E2(d) criterion.  Ref: Cao (1997).
- **FNN** (False Nearest Neighbors): estimates dimension by tracking false
  neighbors that unfold when the dimension is increased.  Ref: Kennel et al. (1992).

References
----------
Fraser, A.M. & Swinney, H.L. (1986) "Independent coordinates for strange
  attractors from mutual information", Phys. Rev. A, 33(2), 1134-1140.
Cao, L. (1997) "Practical method for determining the minimum embedding
  dimension of a scalar time series", Physica D, 110(1-2), 43-50.
Kennel, M.B., Brown, R. & Abarbanel, H.D.I. (1992) "Determining minimum
  embedding dimension using a geometrical construction", Phys. Rev. A,
  45(6), 3403-3411.

Usage
-----
    from dynachaos.diagnostics.embedding import (
        average_mutual_information, optimal_delay,
        cao_method, false_nearest_neighbors, optimal_dimension,
    )

    tau = optimal_delay(x, tau_max=100)
    E1, E2 = cao_method(x, tau, d_max=15)
    d_opt = optimal_dimension(x, tau, d_max=15, method='cao')
"""

import os

import numpy as np

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    import dynachaos._rust as _rust_mod

    _ami_histogram_rs = _rust_mod.ami_histogram
    _select_dimension_cao_rs = getattr(_rust_mod, "select_dimension_cao", None)

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _select_dimension_cao_rs = None

# Cao/FNN always use Python (scipy cKDTree) — Rust brute-force is 70-100x slower.


def _embed(x, d, tau):
    """Time-delay embedding of a scalar series (internal helper).

    Returns ndarray of shape (N - (d-1)*tau, d).
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    M = N - (d - 1) * tau
    if M <= 0:
        raise ValueError(f"Series too short (N={N}) for d={d}, tau={tau}: need N > {(d - 1) * tau}")
    X = np.empty((M, d))
    for j in range(d):
        X[:, j] = x[j * tau : j * tau + M]
    return X


# ── AMI ──────────────────────────────────────────────────────────────────


def _ami_python(x, tau_max, n_bins):
    """Histogram-based AMI, pure Python/NumPy."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    x_min, x_max = x.min(), x.max()
    # Slight padding to ensure max value falls inside last bin
    x_max_padded = x_max + (x_max - x_min) * 1e-10

    I_values = np.empty(tau_max)
    for t in range(1, tau_max + 1):
        n = N - t
        if n < 2:
            I_values[t - 1] = 0.0
            continue
        x1 = x[:n]
        x2 = x[t : t + n]
        hist2d, _, _ = np.histogram2d(
            x1, x2, bins=n_bins, range=[[x_min, x_max_padded], [x_min, x_max_padded]]
        )
        # Joint and marginal probabilities
        p_joint = hist2d / n
        p_x = p_joint.sum(axis=1)
        p_y = p_joint.sum(axis=0)

        # MI = sum p(i,j) * log(p(i,j) / (p(i) * p(j)))
        mi = 0.0
        for i in range(n_bins):
            if p_x[i] == 0.0:
                continue
            for j in range(n_bins):
                if p_joint[i, j] > 0.0 and p_y[j] > 0.0:
                    mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_x[i] * p_y[j]))
        I_values[t - 1] = mi

    return I_values


def average_mutual_information(x, tau_max=100, n_bins=64):
    """Delayed mutual information I(tau) for tau = 1..tau_max.

    Parameters
    ----------
    x : array_like, shape (N,)
        Scalar time series.
    tau_max : int
        Maximum delay to evaluate.
    n_bins : int
        Number of histogram bins per axis.

    Returns
    -------
    tau_values : ndarray, shape (tau_max,)
        Delay values 1, 2, ..., tau_max.
    I_values : ndarray, shape (tau_max,)
        Mutual information at each delay.

    Reference
    ---------
    Fraser, A.M. & Swinney, H.L. (1986), Phys. Rev. A, 33(2), 1134-1140.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x.ndim != 1:
        x = x.ravel()
    if len(x) < 2:
        raise ValueError("x must contain at least two values")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    try:
        tau_max_int = int(tau_max)
        n_bins_int = int(n_bins)
    except (TypeError, ValueError) as exc:
        raise ValueError("tau_max and n_bins must be positive integers") from exc
    if tau_max_int != tau_max or n_bins_int != n_bins or tau_max_int < 1 or n_bins_int < 1:
        raise ValueError("tau_max and n_bins must be positive integers")
    tau_max = tau_max_int
    n_bins = n_bins_int
    tau_values = np.arange(1, tau_max + 1)

    if _RUST_AVAILABLE:
        I_values = np.asarray(_ami_histogram_rs(x, tau_max, n_bins))
    else:
        I_values = _ami_python(x, tau_max, n_bins)

    return tau_values, I_values


def optimal_delay(x, tau_max=100, n_bins=64):
    """First local minimum of the average mutual information.

    Returns
    -------
    tau_opt : int
        Optimal delay (first local minimum of AMI).
        Returns 1 if no local minimum is found.
    """
    _, mi = average_mutual_information(x, tau_max, n_bins)
    # Find first local minimum
    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i - 1] and mi[i] <= mi[i + 1]:
            return i + 1  # tau is 1-indexed
    return 1


# ── Cao's method ─────────────────────────────────────────────────────────


def _cao_python(x, tau, d_max, theiler_window):
    """Cao's E(d) and E*(d) via scipy cKDTree, pure Python."""
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=np.float64)
    E = np.empty(d_max)
    E_star = np.empty(d_max)

    for d in range(1, d_max + 1):
        # Embed in d and d+1 dimensions
        # y1 uses x[:-tau] so indices align with y2
        M = len(x) - d * tau
        if M < 2:
            E[d - 1] = np.nan
            E_star[d - 1] = np.nan
            continue

        y1 = _embed(x[: M + (d - 1) * tau], d, tau)  # shape (M, d)
        y2 = _embed(x, d + 1, tau)  # shape (M, d+1)

        # Find nearest neighbor in d-dimensional embedding (Chebyshev)
        tree = cKDTree(y1)
        # Query k=2 because the first is the point itself
        dists, indices = tree.query(y1, k=2, p=np.inf)
        nn_idx = indices[:, 1]
        nn_dist_d = dists[:, 1]

        # Theiler window: re-query excluding temporal neighbors
        if theiler_window > 0:
            for i in range(M):
                if abs(i - nn_idx[i]) <= theiler_window:
                    # Search with larger k, pick first outside window
                    k_search = min(2 + 2 * theiler_window, M)
                    d_all, i_all = tree.query(y1[i], k=k_search, p=np.inf)
                    for ki in range(1, len(i_all)):
                        if abs(i - i_all[ki]) > theiler_window:
                            nn_idx[i] = i_all[ki]
                            nn_dist_d[i] = d_all[ki]
                            break

        # Compute a(i,d) = ||y2[i] - y2[nn]||_inf / ||y1[i] - y1[nn]||_inf
        nn_dist_d1 = np.max(np.abs(y2 - y2[nn_idx]), axis=1)

        # Avoid division by zero
        safe_mask = nn_dist_d > 0
        a = np.ones(M)
        a[safe_mask] = nn_dist_d1[safe_mask] / nn_dist_d[safe_mask]

        E[d - 1] = np.mean(a)

        # E*(d) = mean |x[i + d*tau] - x[nn_i + d*tau]|
        # The (d+1)-th coordinate difference
        E_star[d - 1] = np.mean(np.abs(y2[:, -1] - y2[nn_idx, -1]))

    return E, E_star


def cao_method(x, tau, d_max=15, theiler_window=0):
    """Cao's method for estimating the minimum embedding dimension.

    Computes the E1(d) and E2(d) statistics:
    - E1(d) = E(d+1)/E(d) saturates near 1 when d >= true dimension.
    - E2(d) != 1 indicates deterministic structure (distinguishes from noise).

    Parameters
    ----------
    x : array_like, shape (N,)
        Scalar time series.
    tau : int
        Time delay for embedding.
    d_max : int
        Maximum embedding dimension to test.
    theiler_window : int
        Minimum temporal separation for nearest-neighbor search.

    Returns
    -------
    E1 : ndarray, shape (d_max - 1,)
        E1(d) = E(d+1)/E(d) for d = 1, ..., d_max-1.
    E2 : ndarray, shape (d_max - 1,)
        E2(d) = E*(d+1)/E*(d) for d = 1, ..., d_max-1.

    Reference
    ---------
    Cao, L. (1997), Physica D, 110(1-2), 43-50.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))

    E, E_star = _cao_python(x, tau, d_max, theiler_window)

    # E1[d] = E[d+1] / E[d], E2[d] = E*[d+1] / E*[d]
    E1 = np.empty(d_max - 1)
    E2 = np.empty(d_max - 1)
    for d in range(d_max - 1):
        E1[d] = E[d + 1] / E[d] if E[d] > 0 else np.nan
        E2[d] = E_star[d + 1] / E_star[d] if E_star[d] > 0 else np.nan

    return E1, E2


# ── FNN ──────────────────────────────────────────────────────────────────


def _fnn_python(x, tau, d_max, R_tol, A_tol, theiler_window):
    """False nearest neighbors, pure Python with scipy cKDTree."""
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=np.float64)
    sigma = np.std(x)

    f1 = np.empty(d_max)
    f2 = np.empty(d_max)
    f3 = np.empty(d_max)

    for d in range(1, d_max + 1):
        M = len(x) - d * tau
        if M < 2:
            f1[d - 1] = np.nan
            f2[d - 1] = np.nan
            f3[d - 1] = np.nan
            continue

        y1 = _embed(x[: M + (d - 1) * tau], d, tau)
        y2 = _embed(x, d + 1, tau)

        # Find NN using Euclidean norm (standard for FNN)
        tree = cKDTree(y1)
        dists, indices = tree.query(y1, k=2, p=2)
        nn_idx = indices[:, 1]
        nn_dist = dists[:, 1]

        # Theiler window
        if theiler_window > 0:
            for i in range(M):
                if abs(i - nn_idx[i]) <= theiler_window:
                    k_search = min(2 + 2 * theiler_window, M)
                    d_all, i_all = tree.query(y1[i], k=k_search, p=2)
                    for ki in range(1, len(i_all)):
                        if abs(i - i_all[ki]) > theiler_window:
                            nn_idx[i] = i_all[ki]
                            nn_dist[i] = d_all[ki]
                            break

        # Test I: |x[i+d*tau] - x[nn+d*tau]| / nn_dist > R_tol
        extra_dist = np.abs(y2[:, -1] - y2[nn_idx, -1])
        safe_mask = nn_dist > 0
        ratio = np.zeros(M)
        ratio[safe_mask] = extra_dist[safe_mask] / nn_dist[safe_mask]
        test1 = ratio > R_tol

        # Test II: ||y2[i] - y2[nn]||_2 / sigma > A_tol
        full_dist = np.sqrt(np.sum((y2 - y2[nn_idx]) ** 2, axis=1))
        test2 = (full_dist / sigma) > A_tol

        f1[d - 1] = np.mean(test1)
        f2[d - 1] = np.mean(test2)
        f3[d - 1] = np.mean(test1 | test2)

    return f1, f2, f3


def false_nearest_neighbors(x, tau, d_max=15, R_tol=15.0, A_tol=2.0, theiler_window=0):
    """False nearest neighbors fraction per embedding dimension.

    Three test statistics are returned:
    - f1: fraction failing Test I (distance ratio criterion)
    - f2: fraction failing Test II (absolute distance criterion)
    - f3: fraction failing either test (union)

    Parameters
    ----------
    x : array_like, shape (N,)
        Scalar time series.
    tau : int
        Time delay.
    d_max : int
        Maximum embedding dimension to test.
    R_tol : float
        Threshold for Test I (default 15.0, Kennel et al.).
    A_tol : float
        Threshold for Test II (default 2.0).
    theiler_window : int
        Minimum temporal separation for nearest-neighbor search.

    Returns
    -------
    f1 : ndarray, shape (d_max,)
        False neighbor fraction (Test I) for d = 1, ..., d_max.
    f2 : ndarray, shape (d_max,)
        False neighbor fraction (Test II) for d = 1, ..., d_max.
    f3 : ndarray, shape (d_max,)
        False neighbor fraction (Test I or II) for d = 1, ..., d_max.

    Reference
    ---------
    Kennel, M.B. et al. (1992), Phys. Rev. A, 45(6), 3403-3411.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))

    return _fnn_python(x, tau, d_max, R_tol, A_tol, theiler_window)


# ── Convenience wrappers ─────────────────────────────────────────────────


def _smooth_series(values, window):
    """Moving-average smoothing helper for 1D diagnostic curves."""

    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    left = int(window) // 2
    right = int(window) - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def select_dimension_cao(
    E1,
    *,
    near_one_lower=0.95,
    near_one_upper=1.05,
    saturation_tol=0.02,
    plateau_span=3,
    smoothing_window=1,
    min_dim=2,
    max_dim=None,
):
    """Select embedding dimension from Cao's E1 curve.

    Strategy:
    1) Find the *onset* of a near-1 stable plateau using a forward window.
    2) Fallback to first d with E1 >= near_one_lower.
    3) Fallback to d with E1 closest to 1.

    Parameters
    ----------
    E1 : array_like
        Cao E1(d) values for d = 1..(d_max-1).
    near_one_lower, near_one_upper : float
        Acceptable near-1 band for saturation detection.
    saturation_tol : float
        Maximum local variation tolerated for plateau stability.
    plateau_span : int
        Number of consecutive dimensions required for a stable plateau.
    smoothing_window : int
        Optional moving-average smoothing window on E1.
    min_dim : int
        Minimum embedding dimension to consider.
    max_dim : int or None
        Optional upper clamp for returned dimension.

    Returns
    -------
    d_opt : int
        Selected embedding dimension.
    """

    e1 = np.ascontiguousarray(np.asarray(E1, dtype=np.float64))
    min_dim = max(1, int(min_dim))
    if e1.size == 0:
        return max(1, min_dim)

    if _RUST_AVAILABLE and _select_dimension_cao_rs is not None:
        max_dim_arg = None if max_dim is None else int(max_dim)
        return int(
            _select_dimension_cao_rs(
                e1,
                near_one_lower=float(near_one_lower),
                near_one_upper=float(near_one_upper),
                saturation_tol=float(saturation_tol),
                plateau_span=int(plateau_span),
                smoothing_window=int(smoothing_window),
                min_dim=min_dim,
                max_dim=max_dim_arg,
            )
        )

    smoothed = _smooth_series(e1, int(smoothing_window))
    finite = np.isfinite(smoothed)
    if not np.any(finite):
        return max(1, min_dim)

    lo = float(near_one_lower)
    hi = float(near_one_upper)
    if lo > hi:
        lo, hi = hi, lo

    span = max(2, int(plateau_span))
    d_hi = int(max_dim) if max_dim is not None else int(smoothed.size + 1)

    def _clamp(d):
        return max(min_dim, min(int(d), d_hi))

    # Primary: onset of stable near-1 plateau.
    for idx, value in enumerate(smoothed):
        dim = idx + 1
        if dim < min_dim or not np.isfinite(value):
            continue
        if not (lo <= value <= hi):
            continue
        window_vals = smoothed[idx : idx + span]
        if window_vals.size < span or not np.all(np.isfinite(window_vals)):
            continue
        if np.any((window_vals < lo) | (window_vals > hi)):
            continue
        if np.ptp(window_vals) <= 1.5 * saturation_tol:
            return _clamp(dim)
        diffs = np.abs(np.diff(window_vals))
        if diffs.size > 0 and np.nanmax(diffs) <= saturation_tol:
            return _clamp(dim)

    # Fallback 1: first near-one crossing.
    for idx, value in enumerate(smoothed):
        dim = idx + 1
        if dim >= min_dim and np.isfinite(value) and value >= lo:
            return _clamp(dim)

    # Fallback 2: closest to 1.
    dims = np.arange(1, smoothed.size + 1, dtype=np.int64)
    valid = (dims >= min_dim) & np.isfinite(smoothed)
    if np.any(valid):
        local_dims = dims[valid]
        local_vals = smoothed[valid]
        closest = int(local_dims[np.argmin(np.abs(local_vals - 1.0))])
        return _clamp(closest)

    return _clamp(int(e1.size + 1))


def optimal_dimension(x, tau, d_max=15, method="cao", theiler_window=0):
    """Estimate the minimum sufficient embedding dimension.

    Parameters
    ----------
    x : array_like
        Scalar time series.
    tau : int
        Time delay.
    d_max : int
        Maximum dimension to test.
    method : str
        'cao' (default, robust plateau-onset selector), 'cao_legacy'
        (first E1 > 0.95), or 'fnn'.
    theiler_window : int
        Minimum temporal separation for nearest-neighbor search.

    Returns
    -------
    d_opt : int
        Estimated optimal embedding dimension.
    """
    if d_max < 1:
        raise ValueError(f"d_max must be >= 1 (got {d_max})")
    if method in ("cao", "cao_legacy") and d_max < 2:
        raise ValueError(
            f"d_max must be >= 2 for Cao's method (got {d_max}); "
            "at least two dimensions are needed to compute E1(d)"
        )

    if method == "cao":
        E1, _ = cao_method(x, tau, d_max, theiler_window=theiler_window)
        return select_dimension_cao(E1, min_dim=2, max_dim=d_max)
    elif method == "cao_legacy":
        E1, _ = cao_method(x, tau, d_max, theiler_window=theiler_window)
        threshold = 0.95
        for d in range(len(E1)):
            if E1[d] > threshold:
                return d + 1
        return d_max
    elif method == "fnn":
        _, _, f3 = false_nearest_neighbors(x, tau, d_max, theiler_window=theiler_window)
        # Find first d where FNN fraction drops below threshold
        threshold = 0.01
        for d in range(len(f3)):
            if f3[d] < threshold:
                return d + 1
        return d_max
    else:
        raise ValueError(f"Unknown method '{method}', use 'cao', 'cao_legacy', or 'fnn'")
