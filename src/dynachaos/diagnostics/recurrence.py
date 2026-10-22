"""Recurrence plots and Recurrence Quantification Analysis (RQA).

A recurrence plot visualises the times at which a trajectory returns close
to a previously visited state.  The binary recurrence matrix is

    R_{ij} = Θ(ε - ||x_i - x_j||)

where Θ is the Heaviside function and ε is a threshold.

From R one extracts diagonal and vertical line structures that quantify:
  - RR   (recurrence rate):  fraction of recurrent points
  - DET  (determinism):      fraction in diagonal lines (≥ l_min)
  - LAM  (laminarity):       fraction in vertical lines (≥ v_min)
  - L    (mean diagonal):    mean diagonal line length
  - TT   (trapping time):    mean vertical line length
  - ENTR (entropy):          Shannon entropy of diagonal line length dist.

The public :func:`rqa` API enforces a non-empty square symmetric recurrence
matrix before computing RQA metrics. The direct line extractors used underneath,
including the Rust-accelerated helpers, are lower-level square-matrix scanners;
callers should not treat them as full RQA validators.

Reference
---------
Marwan, N. et al. (2007) "Recurrence plots for the analysis of complex
  systems", Physics Reports, 438(5-6), 237-329.

Usage
-----
    from dynachaos.diagnostics.recurrence import recurrence_matrix, rqa, rqa_from_trajectory

    R = recurrence_matrix(trajectory, eps=0.1)
    stats = rqa(R, l_min=2, v_min=2)
    large_stats = rqa_from_trajectory(trajectory, eps=0.1)
"""

import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform

from dynachaos.diagnostics._validation import (
    finite_nonnegative_scalar,
    finite_trajectory,
    nonnegative_int,
    positive_int,
    square_bool_matrix,
)
from dynachaos.diagnostics.reliability import ReliabilityRecord

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import count_line_lengths as _count_line_lengths_rs
    from dynachaos._rust import diagonal_lines as _diagonal_lines_rs
    from dynachaos._rust import vertical_lines as _vertical_lines_rs

    _RUST_AVAILABLE = True
except ImportError:
    _count_line_lengths_rs = None
    _RUST_AVAILABLE = False


_STREAMING_METRICS = frozenset({"euclidean", "sqeuclidean", "cityblock", "manhattan", "chebyshev"})
_STREAMING_METRICS_MSG = (
    "rqa_from_trajectory currently supports metric values: "
    "euclidean, sqeuclidean, cityblock, manhattan, chebyshev"
)


@dataclass(frozen=True)
class LaminarLengthsResult:
    """Laminar vertical-line distribution and RQA laminarity measures."""

    lengths: np.ndarray
    LAM: float
    TT: float
    eps: float


def recurrence_matrix(X, eps=None, metric="euclidean", percentile=5):
    """Compute the binary recurrence matrix.

    Parameters
    ----------
    X : ndarray, shape (N, d) or (N,)
        Trajectory: N points in d-dimensional phase space.
        If 1D, automatically reshaped to (N, 1).
    eps : float or None
        Recurrence threshold.  If None, set to the `percentile`-th
        percentile of all pairwise distances.
    metric : str
        Distance metric (passed to scipy.spatial.distance.pdist).
    percentile : float
        Used to auto-select eps when eps is None.

    Returns
    -------
    R : ndarray, shape (N, N), dtype bool
        The recurrence matrix.
    eps_used : float
        The threshold actually used (useful when auto-selected).
    """
    X = finite_trajectory(X, name="X")
    if eps is not None:
        eps = finite_nonnegative_scalar(eps, name="eps")
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")

    dists = squareform(pdist(X, metric=metric))

    if eps is None:
        positive_dists = dists[dists > 0]
        eps = 0.0 if positive_dists.size == 0 else float(np.percentile(positive_dists, percentile))

    R = dists <= eps
    return R, eps


def _diagonal_lines(R, l_min=2):
    """Extract diagonal line lengths from recurrence matrix (upper triangle)."""
    if l_min < 1:
        raise ValueError("l_min must be >= 1")
    if _RUST_AVAILABLE:
        return _diagonal_lines_rs(R, l_min)

    N = R.shape[0]
    lengths = []

    # Direct indexing avoids per-diagonal np.diag allocation
    for k in range(1, N):
        current = 0
        for i in range(N - k):
            if R[i, i + k]:
                current += 1
            else:
                if current >= l_min:
                    lengths.append(current)
                current = 0
        if current >= l_min:
            lengths.append(current)

    return np.array(lengths, dtype=int)


def _vertical_lines(R, v_min=2):
    """Extract vertical line lengths from recurrence matrix."""
    if v_min < 1:
        raise ValueError("v_min must be >= 1")
    if _RUST_AVAILABLE:
        return _vertical_lines_rs(R, v_min)

    N = R.shape[0]
    lengths = []

    for j in range(N):
        current = 0
        for i in range(N):
            if R[i, j]:
                current += 1
            else:
                if current >= v_min:
                    lengths.append(current)
                current = 0
        if current >= v_min:
            lengths.append(current)

    return np.array(lengths, dtype=int)


def _validate_recurrence_threshold(eps, percentile):
    if eps is not None:
        eps = finite_nonnegative_scalar(eps, name="eps")
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    return eps, percentile


def _paired_distances(A, B, metric):
    if metric == "euclidean":
        return np.linalg.norm(A - B, axis=1)
    if metric == "sqeuclidean":
        diff = A - B
        return np.einsum("ij,ij->i", diff, diff)
    if metric in {"cityblock", "manhattan"}:
        return np.sum(np.abs(A - B), axis=1)
    if metric == "chebyshev":
        return np.max(np.abs(A - B), axis=1)
    raise ValueError(_STREAMING_METRICS_MSG)


def _validate_streaming_metric(metric):
    if metric not in _STREAMING_METRICS:
        raise ValueError(_STREAMING_METRICS_MSG)


def _line_lengths(mask, min_length):
    if min_length < 1:
        raise ValueError("min_length must be >= 1")
    if _RUST_AVAILABLE and _count_line_lengths_rs is not None:
        return [
            int(length)
            for length in _count_line_lengths_rs(np.asarray(mask, dtype=np.bool_), int(min_length))
        ]

    lengths = []
    current = 0
    for value in mask:
        if value:
            current += 1
        else:
            if current >= min_length:
                lengths.append(current)
            current = 0
    if current >= min_length:
        lengths.append(current)
    return lengths


def _rqa_from_line_lengths(total_points, recurrent_all, recurrent_upper, diag_lens, vert_lens):
    RR = recurrent_all / total_points

    if diag_lens:
        diag_lens_array = np.asarray(diag_lens, dtype=int)
        diag_sum = np.sum(diag_lens_array)
        DET = diag_sum / recurrent_upper if recurrent_upper > 0 else 0.0
        L = np.mean(diag_lens_array)
        Lmax = np.max(diag_lens_array)
        unique, counts = np.unique(diag_lens_array, return_counts=True)
        probs = counts / np.sum(counts)
        ENTR = -np.sum(probs * np.log(probs))
    else:
        DET, L, Lmax, ENTR = 0.0, 0.0, 0, 0.0

    if vert_lens:
        vert_lens_array = np.asarray(vert_lens, dtype=int)
        vert_sum = np.sum(vert_lens_array)
        LAM = vert_sum / recurrent_all if recurrent_all > 0 else 0.0
        TT = np.mean(vert_lens_array)
    else:
        LAM, TT = 0.0, 0.0

    return {
        "RR": float(RR),
        "DET": float(DET),
        "LAM": float(LAM),
        "L": float(L),
        "TT": float(TT),
        "ENTR": float(ENTR),
        "Lmax": int(Lmax),
    }


def rqa(R, l_min=2, v_min=2):
    """Compute Recurrence Quantification Analysis measures.

    Parameters
    ----------
    R : ndarray, shape (N, N), dtype bool
        Recurrence matrix.
    l_min : int
        Minimum diagonal line length.
    v_min : int
        Minimum vertical line length.

    Returns
    -------
    dict
        Keys: 'RR', 'DET', 'LAM', 'L', 'TT', 'ENTR', 'Lmax'.
    """
    R = square_bool_matrix(R, name="R", symmetric=True)
    l_min = positive_int(l_min, "l_min")
    v_min = positive_int(v_min, "v_min")

    n_recurrent_all = int(np.sum(R))
    diag_lens = _diagonal_lines(R, l_min)
    vert_lens = _vertical_lines(R, v_min)
    n_recurrent_upper = (n_recurrent_all - int(np.trace(R))) // 2
    return _rqa_from_line_lengths(
        R.shape[0] * R.shape[0],
        n_recurrent_all,
        n_recurrent_upper,
        list(diag_lens),
        list(vert_lens),
    )


def laminar_lengths(X, eps=None, metric="euclidean", percentile=5, v_min=2):
    """Return recurrence-plot laminar lengths plus LAM and TT for a trajectory.

    Laminar phases are represented by vertical recurrence-plot line lengths.
    ``eps`` is estimated from the requested distance percentile when omitted.
    LAM follows the Marwan et al. RQA convention: its numerator sums vertical
    lines with length at least ``v_min``, while its denominator sums all
    recurrent points, equivalently vertical lines with length at least one.
    """
    v_min = positive_int(v_min, "v_min")
    R, eps_used = recurrence_matrix(X, eps=eps, metric=metric, percentile=percentile)

    lengths = np.asarray(_vertical_lines(R, v_min), dtype=int)
    all_lengths = np.asarray(_vertical_lines(R, 1), dtype=int)
    denominator = int(np.sum(all_lengths))

    if lengths.size:
        numerator = int(np.sum(lengths))
        LAM = numerator / denominator if denominator > 0 else 0.0
        TT = float(np.mean(lengths))
    else:
        LAM, TT = 0.0, 0.0

    return LaminarLengthsResult(
        lengths=lengths,
        LAM=float(LAM),
        TT=float(TT),
        eps=float(eps_used),
    )


def _threshold_from_recurrence_percentile(X, metric, percentile):
    pdist_metric = "cityblock" if metric == "manhattan" else metric
    condensed = pdist(X, metric=pdist_metric)
    positive_dists = condensed[condensed > 0]
    if positive_dists.size == 0:
        return 0.0
    # recurrence_matrix percentiles over the squareform matrix, where every
    # positive pair appears twice; duplicate the condensed vector to match.
    return float(np.percentile(np.repeat(positive_dists, 2), percentile))


def _trajectory_rqa_scan(X, eps, metric, percentile, l_min, v_min, theiler=0):
    X = finite_trajectory(X, name="X")
    eps, percentile = _validate_recurrence_threshold(eps, percentile)
    l_min = positive_int(l_min, "l_min")
    v_min = positive_int(v_min, "v_min")
    if theiler is None:
        theiler = 0
    theiler = nonnegative_int(theiler, "theiler")
    _validate_streaming_metric(metric)

    if eps is None:
        eps = _threshold_from_recurrence_percentile(X, metric, percentile)

    N = X.shape[0]
    recurrent_upper = 0
    diag_lens = []
    for k in range(1, N):
        recurrent = _paired_distances(X[:-k], X[k:], metric) <= eps
        if k <= theiler:
            recurrent = np.zeros_like(recurrent, dtype=bool)
        recurrent_upper += int(np.sum(recurrent))
        diag_lens.extend(_line_lengths(recurrent, l_min))

    vert_lens = []
    recurrent_all = 0
    rows = np.arange(N)
    for j in range(N):
        recurrent = _paired_distances(X, X[j : j + 1], metric) <= eps
        if theiler > 0:
            recurrent = np.logical_and(recurrent, np.abs(rows - j) > theiler)
        recurrent_all += int(np.sum(recurrent))
        vert_lens.extend(_line_lengths(recurrent, v_min))

    if theiler == 0:
        recurrent_all = N + 2 * recurrent_upper
    stats = _rqa_from_line_lengths(N * N, recurrent_all, recurrent_upper, diag_lens, vert_lens)
    details = {
        "eps": float(eps),
        "theiler": int(theiler),
        "diagonal_lengths": np.asarray(diag_lens, dtype=int),
        "vertical_lengths": np.asarray(vert_lens, dtype=int),
        "data_length": int(N),
        "data_shape": tuple(int(v) for v in X.shape),
    }
    return stats, details


def rqa_from_trajectory(
    X, eps=None, metric="euclidean", percentile=5, l_min=2, v_min=2, *, return_metadata=False
):
    """Compute RQA measures from a trajectory without materializing ``R``.

    The dense :func:`recurrence_matrix` API remains the right interface when a
    caller needs the recurrence matrix itself.  This function is for large-RQA
    workflows that only need the scalar measures.
    """
    stats, details = _trajectory_rqa_scan(X, eps, metric, percentile, l_min, v_min)
    if return_metadata:
        warnings = []
        unresolved = []
        if details["diagonal_lengths"].size == 0:
            warnings.append("no diagonal lines at or above l_min")
        if details["vertical_lengths"].size == 0:
            warnings.append("no vertical lines at or above v_min")
        metadata = ReliabilityRecord(
            method_name="rqa_from_trajectory",
            backend=(
                "rust" if (_RUST_AVAILABLE and _count_line_lengths_rs is not None) else "python"
            ),
            parameters={
                "eps": details["eps"],
                "metric": metric,
                "percentile": float(percentile),
                "l_min": positive_int(l_min, "l_min"),
                "v_min": positive_int(v_min, "v_min"),
            },
            data_length=details["data_length"],
            data_shape=details["data_shape"],
            sampling_downsampling_note=(
                "no sampling/downsampling; streaming trajectory scan without dense "
                "recurrence matrix"
            ),
            validity_warnings=warnings,
            unresolved_verdicts=unresolved,
            scale_evidence={
                "artifact_path": "benchmarks/results/scale_envelope.json",
                "entry": {"kind": "dense_recurrence_rqa", "backend": "python dense reference"},
            },
        )
        return stats, metadata
    return stats


def embed_time_delay(x, d, tau):
    """Time-delay embedding of a scalar time series.

    Parameters
    ----------
    x : array_like, shape (N,)
        Scalar time series.
    d : int
        Embedding dimension.
    tau : int
        Time delay.

    Returns
    -------
    X : ndarray, shape (N - (d-1)*tau, d)
        Embedded trajectory.
    """
    x = np.asarray(x, dtype=np.float64)
    d = positive_int(d, "d")
    tau = positive_int(tau, "tau")
    N = len(x)
    M = N - (d - 1) * tau
    if M <= 0:
        raise ValueError(f"Series too short (N={N}) for d={d}, tau={tau}")
    X = np.empty((M, d))
    for j in range(d):
        X[:, j] = x[j * tau : j * tau + M]
    return X
