"""Low-level reusable CML/GCM kernels and numeric helpers."""

import os

import numpy as np

from dynachaos.maps.primitives import logistic, logistic_derivative

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import cml_jacobian_logistic as _cml_jacobian_logistic_rs

    _RUST_AVAILABLE = True
except ImportError:
    _cml_jacobian_logistic_rs = None
    _RUST_AVAILABLE = False


def _roll_pair(values, axis):
    """Return left/right rolls, preserving legacy flatten semantics by default."""
    if axis is None:
        return np.roll(values, -1), np.roll(values, 1)
    return np.roll(values, -1, axis=axis), np.roll(values, 1, axis=axis)


def cml_step(x, f, g, eps, axis=None):
    """One CML step with periodic boundary conditions.

    `axis=None` preserves the historical NumPy flatten-then-roll behavior.
    Pass an explicit axis for multidimensional lattice arrays.
    """
    fx = f(x)
    gx = g(x)
    gx_left, gx_right = _roll_pair(gx, axis)
    coupling = eps / 2.0 * (gx_left + gx_right - 2.0 * gx)
    return fx + coupling


def cml_step_logistic(x, a, eps, axis=None):
    """One CML step with f=g=logistic (Model C shorthand).

    `axis=None` preserves the historical NumPy flatten-then-roll behavior.
    Pass an explicit axis for multidimensional lattice arrays.
    """
    fx = logistic(x, a)
    fx_left, fx_right = _roll_pair(fx, axis)
    coupling = eps / 2.0 * (fx_left + fx_right - 2.0 * fx)
    return fx + coupling


def cml_step_logistic_batch(x, a_col, eps, axis=1):
    """Batched logistic CML update for arrays shaped (n_a, n_sites)."""
    fx = logistic(x, a_col)
    fx_left = np.roll(fx, -1, axis=axis)
    fx_right = np.roll(fx, 1, axis=axis)
    return (1.0 - eps) * fx + eps / 2.0 * (fx_left + fx_right)


def gcm_step(x, a, eps):
    """One globally-coupled logistic-map step."""
    fx = logistic(x, a)
    mean_field = np.mean(fx)
    return (1.0 - eps) * fx + eps * mean_field


def cluster_labels_by_tolerance(x, tol=1e-6):
    """Assign cluster labels from sorted one-dimensional state values."""
    values = np.asarray(x)
    if values.ndim != 1:
        raise ValueError("cluster_labels_by_tolerance expects a 1D array")
    if values.size == 0:
        return np.empty(0, dtype=int)

    labels = -np.ones(values.size, dtype=int)
    idx_sorted = np.argsort(values)
    x_sorted = values[idx_sorted]

    cluster_id = 0
    labels[idx_sorted[0]] = cluster_id
    for k in range(1, values.size):
        if x_sorted[k] - x_sorted[k - 1] > tol:
            cluster_id += 1
        labels[idx_sorted[k]] = cluster_id
    return labels


def sustained_positive_mask(values, threshold=0.02, min_run=4):
    """Mask entries that belong to sustained runs above a threshold."""
    mask = np.asarray(values) > threshold
    broad = np.zeros_like(mask, dtype=bool)
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_run:
                broad[start:idx] = True
            start = None
    if start is not None and len(mask) - start >= min_run:
        broad[start:] = True
    return broad


def cml_jacobian_subblock_logistic(x, a, eps, L):
    """Jacobian block for logistic CML subsystem (sites 0..L-1)."""
    values = np.asarray(x, dtype=np.float64)
    N = len(values)
    if L < 1 or L > N:
        raise ValueError(f"L must satisfy 1 <= L <= N (got L={L}, N={N})")

    if _RUST_AVAILABLE and _cml_jacobian_logistic_rs is not None:
        flat = _cml_jacobian_logistic_rs(np.ascontiguousarray(values), a, eps, L)
        return np.asarray(flat).reshape((L, L))

    dfx = logistic_derivative(values, a)
    J = np.zeros((L, L))
    for i in range(L):
        J[i, i] = (1.0 - eps) * dfx[i]
        i_left = (i - 1) % N
        if 0 <= i_left < L:
            J[i, i_left] = (eps / 2.0) * dfx[i_left]
        i_right = (i + 1) % N
        if 0 <= i_right < L:
            J[i, i_right] = (eps / 2.0) * dfx[i_right]
    return J


__all__ = [
    "cml_step",
    "cml_step_logistic",
    "cml_step_logistic_batch",
    "gcm_step",
    "cluster_labels_by_tolerance",
    "sustained_positive_mask",
    "cml_jacobian_subblock_logistic",
]
