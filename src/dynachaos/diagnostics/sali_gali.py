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


def alignment_indices(f, jac, x0, k=4, n_iter=10_000, n_transient=1000, rng=None):
    """Return log10 GALI_j (j = 1..k) at every iteration, without a floor.

    :func:`gali` multiplies singular values of the normalised deviation
    vectors.  Double precision cannot resolve a GALI below about 1e-16, so a
    decaying index stops at that floor.  This function keeps the log of the
    volume instead.

    The deviation frame is written as ``W = Q S``.  ``Q`` has orthonormal
    columns and ``S`` is upper triangular with unit-norm columns.  One step
    is ``J Q = Q' R``, then ``S' = (R S) / column norms``.  Because ``S`` is
    triangular, the product of its first ``j`` diagonal entries is the volume
    of the first ``j`` normalised vectors, i.e. GALI_j.  The diagonal is
    accumulated as a log and re-inserted, so it never underflows.

    The nested frames share one set of ``k`` initial vectors.  Column ``j``
    of the frame never depends on columns after ``j``.

    Parameters
    ----------
    f, jac, x0, n_iter, n_transient, rng
        As in :func:`gali`.
    k : int
        Number of deviation vectors, at most the dimension of the map.

    Returns
    -------
    log10_gali : ndarray, shape (n_iter, k)
        Column ``j - 1`` holds log10 GALI_j.  Column 0 is zero.  Column 1
        is log10 GALI_2.

    Raises
    ------
    FloatingPointError
        If the orbit or the frame stops being finite.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    dim = len(x)
    if not 1 <= k <= dim:
        raise ValueError("k must be between 1 and the dimension of the map")
    if rng is None:
        rng = np.random.default_rng(42)

    for _ in range(n_transient):
        x = f(x)

    Q = _random_orthonormal(dim, k, rng)
    S = np.eye(k)
    log_diag = np.zeros(k)  # natural log of |S_ii|
    sign = np.ones(k)
    out = np.empty((n_iter, k))
    idx = np.arange(k)

    for i in range(n_iter):
        J = jac(x)
        x = f(x)
        Q, R = np.linalg.qr(J @ Q)
        T = R @ S
        norms = np.sqrt(np.einsum("ij,ij->j", T, T))
        r_diag = np.diag(R)
        log_diag += np.log(np.abs(r_diag)) - np.log(norms)
        sign *= np.sign(r_diag)
        S = T / norms
        S[idx, idx] = sign * np.exp(log_diag)
        out[i] = np.cumsum(log_diag)

    if not (np.all(np.isfinite(out)) and np.all(np.isfinite(x))):
        raise FloatingPointError("non-finite alignment index")
    return out / np.log(10.0)


def log10_sali_from_gali2(log10_gali2):
    """log10 SALI from log10 GALI_2 of the same two normalised vectors.

    With angle ``theta`` between the vectors, GALI_2 = sin(theta) and
    SALI = 2 sin(min(theta, pi - theta) / 2).  Hence
    SALI = GALI_2 * sqrt(2 / (1 + sqrt(1 - GALI_2**2))).  The factor lies in
    [1, sqrt(2)] and tends to 1 as the vectors align, so SALI follows GALI_2
    far below the 1e-16 floor of :func:`sali`.
    """
    lg = np.asarray(log10_gali2, dtype=np.float64)
    g = 10.0**lg
    return lg + 0.5 * np.log10(2.0 / (1.0 + np.sqrt(np.maximum(1.0 - g * g, 0.0))))
