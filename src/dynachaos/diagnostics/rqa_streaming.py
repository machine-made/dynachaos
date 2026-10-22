"""Streaming recurrence quantification without materializing a dense matrix."""

from __future__ import annotations

import numpy as np

from dynachaos.diagnostics.recurrence import _trajectory_rqa_scan


def rqa_streaming_from_trajectory(
    X,
    eps=None,
    metric="euclidean",
    percentile=5,
    l_min=2,
    v_min=2,
    *,
    theiler=0,
    return_counts=False,
):
    """Compute exact RQA metrics by scanning diagonals and columns.

    The default return value is byte-compatible with
    :func:`dynachaos.diagnostics.recurrence.rqa`: a dict containing ``RR``,
    ``DET``, ``LAM``, ``L``, ``TT``, ``ENTR``, and ``Lmax``.  Set
    ``return_counts=True`` to also return the diagonal/vertical run lengths and
    the exact diagonal-length bins used for ``ENTR``.

    ``theiler=0`` preserves current dense-RQA semantics.  Positive values mask
    all pairs with ``abs(i - j) <= theiler`` before counting recurrences.
    """
    stats, scan_details = _trajectory_rqa_scan(X, eps, metric, percentile, l_min, v_min, theiler)
    if not return_counts:
        return stats

    diag_lens = scan_details["diagonal_lengths"]
    if diag_lens.size:
        bins, counts = np.unique(diag_lens, return_counts=True)
    else:
        bins = np.asarray([], dtype=int)
        counts = np.asarray([], dtype=int)
    details = {
        "eps": scan_details["eps"],
        "theiler": scan_details["theiler"],
        "diagonal_lengths": diag_lens,
        "vertical_lengths": scan_details["vertical_lengths"],
        "entr_bins": bins,
        "entr_counts": counts,
    }
    return stats, details
