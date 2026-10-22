import numpy as np
import pytest

from dynachaos.diagnostics.recurrence import recurrence_matrix, rqa
from dynachaos.diagnostics.rqa_streaming import rqa_streaming_from_trajectory


def _mask_theiler(R, theiler):
    out = np.array(R, dtype=bool, copy=True)
    if theiler > 0:
        i, j = np.indices(out.shape)
        out[np.abs(i - j) <= theiler] = False
    return out


def _dense_diag_bins(R, l_min):
    lengths = []
    n = R.shape[0]
    for k in range(1, n):
        current = 0
        for value in np.diag(R, k=k):
            if value:
                current += 1
            else:
                if current >= l_min:
                    lengths.append(current)
                current = 0
        if current >= l_min:
            lengths.append(current)
    if not lengths:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    return np.unique(np.asarray(lengths, dtype=int), return_counts=True)


def _assert_streaming_matches_dense(
    X, *, eps=None, percentile=5, metric="euclidean", l_min=2, v_min=2, theiler=0
):
    R, eps_used = recurrence_matrix(X, eps=eps, metric=metric, percentile=percentile)
    R = _mask_theiler(R, theiler)
    dense = rqa(R, l_min=l_min, v_min=v_min)
    streaming, details = rqa_streaming_from_trajectory(
        X,
        eps=eps,
        percentile=percentile,
        metric=metric,
        l_min=l_min,
        v_min=v_min,
        theiler=theiler,
        return_counts=True,
    )
    assert details["eps"] == pytest.approx(eps_used)
    assert streaming.keys() == dense.keys()
    for key in ["RR", "DET", "LAM", "L", "TT", "ENTR"]:
        assert streaming[key] == pytest.approx(dense[key])
    assert streaming["Lmax"] == dense["Lmax"]
    bins, counts = _dense_diag_bins(R, l_min)
    np.testing.assert_array_equal(details["entr_bins"], bins)
    np.testing.assert_array_equal(details["entr_counts"], counts)
    return dense, streaming, details


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eps": 0.23, "metric": "euclidean", "l_min": 2, "v_min": 2, "theiler": 0},
        {"eps": 0.23, "metric": "euclidean", "l_min": 3, "v_min": 2, "theiler": 0},
        {"eps": 0.31, "metric": "cityblock", "l_min": 2, "v_min": 3, "theiler": 0},
        {"eps": 0.42, "metric": "chebyshev", "l_min": 2, "v_min": 2, "theiler": 1},
        {
            "eps": None,
            "percentile": 40,
            "metric": "euclidean",
            "l_min": 2,
            "v_min": 2,
            "theiler": 0,
        },
    ],
)
def test_rqa_streaming_exact_dense_parity(kwargs):
    X = np.array(
        [
            [0.00, 0.20],
            [0.10, 0.24],
            [0.35, 0.34],
            [0.36, 0.38],
            [0.80, 0.52],
            [0.83, 0.55],
            [0.20, 0.21],
        ],
        dtype=float,
    )
    _assert_streaming_matches_dense(X, **kwargs)


def test_rqa_streaming_degenerate_all_one_recurrence_matrix():
    X = np.zeros((5, 1), dtype=float)
    dense, streaming, details = _assert_streaming_matches_dense(X, eps=0.0, l_min=1, v_min=1)
    assert dense == streaming
    np.testing.assert_array_equal(details["entr_bins"], np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(details["entr_counts"], np.array([1, 1, 1, 1]))


def test_rqa_streaming_degenerate_all_zero_after_theiler_mask():
    X = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=float)
    dense, streaming, details = _assert_streaming_matches_dense(
        X, eps=0.0, l_min=2, v_min=2, theiler=4
    )
    assert dense == {"RR": 0.0, "DET": 0.0, "LAM": 0.0, "L": 0.0, "TT": 0.0, "ENTR": 0.0, "Lmax": 0}
    assert streaming == dense
    assert details["diagonal_lengths"].size == 0
    assert details["vertical_lengths"].size == 0


@pytest.mark.parametrize("theiler", [-1, -0.5, 0.5, True, "1"])
def test_rqa_streaming_rejects_invalid_theiler(theiler):
    with pytest.raises(ValueError, match="theiler must be a non-negative integer"):
        rqa_streaming_from_trajectory(np.arange(4.0), eps=0.1, theiler=theiler)


def test_rqa_streaming_percentile_threshold_matches_dense():
    # interpolated percentile differs between the condensed pdist vector and
    # the squareform multiset; this fixture exposed eps 4.6 vs 5.0 pre-fix
    X = np.array([[15.0], [18.0], [23.0]])
    _assert_streaming_matches_dense(X, eps=None, percentile=40, l_min=1, v_min=1)


# ── Coverage: rqa_streaming edge cases ────────────────────────────────────────


def test_rqa_streaming_early_return_when_return_counts_false():
    """rqa_streaming_from_trajectory returns early when return_counts=False (line 34)."""
    X = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]], dtype=float)
    result = rqa_streaming_from_trajectory(X, eps=0.15, return_counts=False)
    assert isinstance(result, dict)
    assert "RR" in result
    assert "DET" in result
    assert not isinstance(result, tuple)
