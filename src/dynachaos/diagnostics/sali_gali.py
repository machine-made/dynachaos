"""SALI and GALI fast chaos indicators for discrete maps.

The Smaller Alignment Index (SALI, Skokos 2001) and Generalized Alignment
Index (GALI, Skokos et al. 2007) detect chaos by tracking the alignment
of deviation vectors under the tangent map.

For chaotic orbits all deviation vectors align with the most unstable
direction, so:

    SALI → 0  exponentially fast  (chaos)
    SALI ≈ const > 0              (regular)

    GALI_k → 0  exponentially     (chaos, rate depends on Lyapunov gaps)
    GALI_k ≈ const > 0            (regular on k-torus)

GALI_k is the volume of the parallelepiped spanned by k deviation vectors,
computed as a product of singular values.  GALI_2 ≡ SALI (up to
normalisation).

References
----------
Skokos, Ch. (2001) "Alignment indices: a new, simple method for
  determining the ordered or chaotic nature of orbits", J. Phys. A,
  34(47), 10029-10043.
Skokos, Ch., Bountis, T. & Antonopoulos, Ch. (2007) "Geometrical
  properties of local dynamics in Hamiltonian systems: the Generalized
  Alignment Index (GALI) method", Physica D, 231(1), 30-54.

Usage
-----
    from dynachaos.diagnostics.sali_gali import sali, gali

    sali_series = sali(f, jac, x0, n_iter=10000)
    gali_series = gali(f, jac, x0, k=4, n_iter=10000)
"""

import numpy as np


def _random_orthonormal(dim, k, rng):
    """Generate k random orthonormal vectors in R^dim."""
    A = rng.standard_normal((dim, k))
    Q, _ = np.linalg.qr(A)
    return Q  # columns are orthonormal


def sali(f, jac, x0, n_iter=10_000, n_transient=1000, rng=None):
    """Compute the Smaller Alignment Index (SALI) time series.

    Parameters
    ----------
    f : callable
        The map x_{n+1} = f(x_n).
    jac : callable
        The Jacobian J(x), returns (N, N) array.
    x0 : array_like
        Initial condition, shape (N,).
    n_iter : int
        Number of iterations to track.
    n_transient : int
        Transient iterations before tracking.
    rng : numpy.random.Generator or None
        For reproducibility of initial deviation vectors.

    Returns
    -------
    sali_values : ndarray, shape (n_iter,)
        SALI at each iteration.  Exponential decay → chaos.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    dim = len(x)

    if rng is None:
        rng = np.random.default_rng(42)

    # Transient
    for _ in range(n_transient):
        x = f(x)

    # Two initial deviation vectors (random, normalised)
    Q = _random_orthonormal(dim, 2, rng)
    v1, v2 = Q[:, 0].copy(), Q[:, 1].copy()

    sali_values = np.empty(n_iter)

    for i in range(n_iter):
        J = jac(x)
        x = f(x)

        # Evolve deviation vectors
        v1 = J @ v1
        v2 = J @ v2

        # Normalise
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0:
            v1 /= n1
        if n2 > 0:
            v2 /= n2

        # SALI = min(||v1_hat + v2_hat||, ||v1_hat - v2_hat||)
        # When vectors align: one of these → 0
        sali_values[i] = min(np.linalg.norm(v1 + v2), np.linalg.norm(v1 - v2))

    return sali_values


def gali(f, jac, x0, k=None, n_iter=10_000, n_transient=1000, rng=None):
    """Compute the Generalized Alignment Index (GALI_k) time series.

    Parameters
    ----------
    f : callable
        The map x_{n+1} = f(x_n).
    jac : callable
        The Jacobian J(x), returns (N, N) array.
    x0 : array_like
        Initial condition, shape (N,).
    k : int or None
        Number of deviation vectors.  Default: N (full dimension).
    n_iter : int
        Number of iterations to track.
    n_transient : int
        Transient iterations before tracking.
    rng : numpy.random.Generator or None
        For reproducibility.

    Returns
    -------
    gali_values : ndarray, shape (n_iter,)
        GALI_k at each iteration.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    dim = len(x)

    if k is None:
        k = dim
    k = min(k, dim)

    if rng is None:
        rng = np.random.default_rng(42)

    # Transient
    for _ in range(n_transient):
        x = f(x)

    # k initial orthonormal deviation vectors
    Q = _random_orthonormal(dim, k, rng)
    vectors = [Q[:, j].copy() for j in range(k)]

    gali_values = np.empty(n_iter)

    for i in range(n_iter):
        J = jac(x)
        x = f(x)

        # Evolve and normalise all deviation vectors
        for j in range(k):
            vectors[j] = J @ vectors[j]
            norm = np.linalg.norm(vectors[j])
            if norm > 0:
                vectors[j] /= norm

        # GALI_k = product of singular values of the matrix
        # whose columns are the normalised deviation vectors
        W = np.column_stack(vectors)
        sv = np.linalg.svd(W, compute_uv=False)
        gali_values[i] = np.prod(sv)

    return gali_values


def sali_at_time(f, jac, x0, n_iter=10_000, n_transient=1000, rng=None):
    """Return only the final SALI value (for parameter sweeps).

    Returns
    -------
    float
        SALI at iteration n_iter.
    """
    vals = sali(f, jac, x0, n_iter, n_transient, rng)
    return vals[-1]
