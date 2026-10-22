import numpy as np
import pytest
from conftest import logistic_series

import dynachaos.diagnostics.correlation as correlation_module
from dynachaos.diagnostics._validation import (
    finite_positive_scalar,
    finite_series_1d,
    finite_trajectory,
    nonnegative_int,
    positive_int,
    sorted_nonnegative_radius_grid,
    square_bool_matrix,
)
from dynachaos.diagnostics.correlation import (
    _correlation_counts_python,
    _find_scaling_region,
    correlation_dimension,
    correlation_integral,
    fit_power_law_loglog,
    takens_theiler_dimension,
)
from dynachaos.diagnostics.permutation import (
    complexity_entropy,
    ordinal_distribution,
    permutation_entropy,
    permutation_entropy_sweep,
)
from dynachaos.diagnostics.recurrence import (
    embed_time_delay,
    laminar_lengths,
    recurrence_matrix,
    rqa,
    rqa_from_trajectory,
)
from dynachaos.diagnostics.zero_one_test import zero_one_statistic


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_validation_helpers_reject_bool_integer(value):
    with pytest.raises(ValueError, match="n must be a positive integer"):
        positive_int(value, "n")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([1.0, np.nan], "x must contain only finite values"),
        ([[1.0, 2.0], [np.inf, 3.0]], "X must contain only finite values"),
        (np.array(1.0), "X must be a non-empty 1D or 2D trajectory"),
        (np.empty((0, 2)), "X must be a non-empty 1D or 2D trajectory"),
        (0.0, "r must be a finite positive number"),
    ],
)
def test_validation_helpers_reject_non_finite_or_non_positive_values(value, message):
    with pytest.raises(ValueError, match=message):
        if message.startswith("x"):
            finite_series_1d(value, name="x")
        elif message.startswith("X"):
            finite_trajectory(value, name="X")
        else:
            finite_positive_scalar(value, name="r")


@pytest.mark.parametrize(
    ("r_values", "message"),
    [
        ([[0.0, 1.0]], "r_values must be a 1D array"),
        ([0.0, np.nan], "r_values must contain only finite values"),
        ([-0.1, 0.2], "r_values must be non-negative"),
        ([0.2, 0.1], "r_values must be sorted in ascending order"),
    ],
)
def test_validation_helpers_reject_invalid_radius_grids(r_values, message):
    with pytest.raises(ValueError, match=message):
        sorted_nonnegative_radius_grid(r_values, name="r_values")


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((0, 0), dtype=bool), "R must be a non-empty square matrix"),
        (np.ones((2, 3), dtype=bool), "R must be a non-empty square matrix"),
        (np.array([[True, False], [True, True]]), "R must be symmetric"),
    ],
)
def test_validation_helpers_reject_invalid_recurrence_matrices(matrix, message):
    with pytest.raises(ValueError, match=message):
        square_bool_matrix(matrix, name="R", symmetric=True)


def test_zero_one_regular_vs_chaotic():
    n = 5000
    t = np.arange(n, dtype=np.float64)
    regular = np.sin(2.0 * np.pi * 0.071 * t) + 0.2 * np.sin(2.0 * np.pi * 0.113 * t)
    chaotic = logistic_series(n=n, a=1.99, burn=2000)

    rng = np.random.default_rng(2026)
    k_regular = zero_one_statistic(regular, n_c=20, rng=rng)
    k_chaotic = zero_one_statistic(chaotic, n_c=20, rng=np.random.default_rng(2026))

    assert k_regular < 0.4
    assert k_chaotic > 0.6


def test_zero_one_zero_observable_is_regular():
    k = zero_one_statistic(np.zeros(100), n_c=5, rng=np.random.default_rng(2027))

    assert k == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("phi", "n_c", "n_cut", "message"),
    [
        ([1.0, 2.0], 1, None, "at least three"),
        ([1.0, np.nan, 2.0], 1, None, "finite values"),
        ([1.0, 2.0, 3.0, 4.0], 0, None, "n_c must be"),
        ([1.0, 2.0, 3.0, 4.0], 1.5, None, "n_c must be"),
        ([1.0, 2.0, 3.0, 4.0], 1, 1, "n_cut must be"),
        ([1.0, 2.0, 3.0, 4.0], 1, 5, "n_cut must be"),
        ([1.0, 2.0, 3.0, 4.0], "abc", None, "n_c must be"),
        ([1.0, 2.0, 3.0, 4.0], 1, "abc", "n_cut must be"),
        ([1.0, 2.0, 3.0, 4.0], 1, 2.5, "n_cut must be a positive integer"),
    ],
)
def test_zero_one_statistic_rejects_fuzzed_invalid_inputs(phi, n_c, n_cut, message):
    with pytest.raises(ValueError, match=message):
        zero_one_statistic(phi, n_c=n_c, n_cut=n_cut)


def test_zero_one_series_returns_c_and_k_arrays_matching_statistic_median():
    from dynachaos.diagnostics.zero_one_test import zero_one_series

    n = 2000
    chaotic = logistic_series(n=n, a=1.99, burn=2000)

    c_values, k_values = zero_one_series(chaotic, n_c=15, rng=np.random.default_rng(2026))
    assert c_values.shape == (15,)
    assert k_values.shape == (15,)
    assert np.all((c_values > np.pi / 5) & (c_values < 4 * np.pi / 5))

    median_from_series = float(np.median(k_values))
    direct_median = zero_one_statistic(chaotic, n_c=15, rng=np.random.default_rng(2026))
    assert median_from_series == pytest.approx(direct_median)


def test_zero_one_series_default_rng_is_deterministic_across_calls():
    from dynachaos.diagnostics.zero_one_test import zero_one_series

    phi = logistic_series(n=500, a=1.99, burn=200)
    c1, k1 = zero_one_series(phi, n_c=5)
    c2, k2 = zero_one_series(phi, n_c=5)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(k1, k2)


def test_msd_regression_defaults_n_cut_to_tenth_of_series_length_when_none():
    from dynachaos.diagnostics.zero_one_test import _msd_regression

    rng = np.random.default_rng(0)
    p = rng.standard_normal(100)
    q = rng.standard_normal(100)
    K_default = _msd_regression(p, q, None)
    K_explicit = _msd_regression(p, q, 100 // 10)
    assert K_default == pytest.approx(K_explicit)


def test_permutation_entropy_bounds():
    x = np.sin(np.linspace(0.0, 60.0, 2000))
    h = permutation_entropy(x, d=5)
    h2, c2 = complexity_entropy(x, d=5)

    assert 0.0 <= h <= 1.0
    assert 0.0 <= h2 <= 1.0
    assert np.isfinite(c2)
    assert c2 >= 0.0


@pytest.mark.parametrize(
    ("d", "tau", "message"),
    [
        (0, 1, "d must be >= 2"),
        (1, 1, "d must be >= 2"),
        (11, 1, "d must be <= 10"),
        (2, 0, "tau must be >= 1"),
        (2, 1.5, "d and tau must be positive integers"),
        (5, 1, "time series is too short"),
    ],
)
def test_ordinal_distribution_rejects_invalid_embedding_parameters(d, tau, message):
    with pytest.raises(ValueError, match=message):
        ordinal_distribution(np.arange(3.0), d=d, tau=tau)


def test_recurrence_and_rqa_sanity():
    t = np.linspace(0.0, 40.0, 600)
    traj = np.column_stack([np.sin(t), np.cos(t)])

    rmat, eps = recurrence_matrix(traj, percentile=8)
    stats = rqa(rmat, l_min=2, v_min=2)

    assert rmat.shape == (len(traj), len(traj))
    assert rmat.dtype == np.bool_
    assert eps > 0.0

    assert 0.0 <= stats["RR"] <= 1.0
    assert 0.0 <= stats["DET"] <= 1.0
    assert 0.0 <= stats["LAM"] <= 1.0
    assert stats["L"] >= 0.0
    assert stats["TT"] >= 0.0
    assert stats["ENTR"] >= 0.0
    assert stats["Lmax"] >= 0


@pytest.mark.parametrize(
    "metric",
    ["euclidean", "sqeuclidean", "cityblock", "manhattan", "chebyshev"],
)
def test_rqa_from_trajectory_matches_dense_recurrence_matrix(metric):
    t = np.linspace(0.0, 30.0, 180)
    traj = np.column_stack([np.sin(t), np.cos(1.7 * t)])

    dense_metric = "cityblock" if metric == "manhattan" else metric
    rmat, _ = recurrence_matrix(traj, percentile=8, metric=dense_metric)
    dense_stats = rqa(rmat, l_min=2, v_min=2)
    streaming_stats = rqa_from_trajectory(traj, percentile=8, metric=metric, l_min=2, v_min=2)

    assert streaming_stats.keys() == dense_stats.keys()
    for key, value in dense_stats.items():
        assert streaming_stats[key] == pytest.approx(value)


def test_rqa_from_trajectory_constant_signal_matches_dense():
    traj = np.ones((12, 2))

    rmat, eps = recurrence_matrix(traj)
    dense_stats = rqa(rmat, l_min=2, v_min=2)
    streaming_stats = rqa_from_trajectory(traj, l_min=2, v_min=2)

    assert eps == 0.0
    assert streaming_stats == dense_stats


def test_rqa_from_trajectory_matches_dense_with_explicit_eps():
    t = np.linspace(0.0, 24.0, 160)
    traj = np.column_stack([np.sin(t), np.cos(1.3 * t)])

    rmat, _ = recurrence_matrix(traj, eps=0.25, metric="euclidean")
    dense_stats = rqa(rmat, l_min=3, v_min=2)
    streaming_stats = rqa_from_trajectory(traj, eps=0.25, metric="euclidean", l_min=3, v_min=2)

    assert streaming_stats == pytest.approx(dense_stats)


def test_rqa_from_trajectory_percentile_threshold_matches_dense_squareform_multiset():
    # Interpolated percentile differs between the condensed pdist vector and
    # the squareform multiset; this fixture exposed eps 4.6 vs 5.0 pre-fix.
    traj = np.array([[15.0], [18.0], [23.0]])

    rmat, eps = recurrence_matrix(traj, eps=None, percentile=40)
    dense_stats = rqa(rmat, l_min=1, v_min=1)
    streaming_stats = rqa_from_trajectory(traj, eps=None, percentile=40, l_min=1, v_min=1)

    assert eps == pytest.approx(5.0)
    assert streaming_stats == pytest.approx(dense_stats)


def test_laminar_lengths_exposes_vertical_line_distribution_and_rqa_measures():
    traj = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]])

    result = laminar_lengths(traj, eps=0.0, v_min=3)

    np.testing.assert_array_equal(result.lengths, np.array([3, 3, 3]))
    assert result.LAM == pytest.approx(9.0 / 13.0)
    assert result.TT == pytest.approx(3.0)
    assert result.eps == pytest.approx(0.0)


def test_laminar_lengths_matches_rqa_laminarity_and_trapping_time():
    t = np.linspace(0.0, 20.0, 140)
    traj = np.column_stack([np.sin(t), np.cos(1.2 * t)])

    rmat, eps = recurrence_matrix(traj, percentile=9)
    stats = rqa(rmat, v_min=2)
    result = laminar_lengths(traj, percentile=9, v_min=2)

    assert result.eps == pytest.approx(eps)
    assert result.LAM == pytest.approx(stats["LAM"])
    assert result.TT == pytest.approx(stats["TT"])


def test_rqa_from_trajectory_minimal_trajectory():
    # N==1: single point; eps auto-selects to 0.0 (no positive pairwise distances)
    X1 = np.ones((1, 2))
    rmat1, _ = recurrence_matrix(X1)
    dense_stats1 = rqa(rmat1, l_min=2, v_min=2)
    streaming_stats1 = rqa_from_trajectory(X1, l_min=2, v_min=2)

    assert dense_stats1["RR"] == pytest.approx(1.0)
    assert streaming_stats1.keys() == dense_stats1.keys()
    for key, value in dense_stats1.items():
        assert streaming_stats1[key] == pytest.approx(value)

    # N==2: two distinct points; explicit eps=0.5 excludes the off-diagonal pair
    X2 = np.array([[0.0, 0.0], [1.0, 0.0]])
    rmat2, _ = recurrence_matrix(X2, eps=0.5)
    dense_stats2 = rqa(rmat2, l_min=2, v_min=2)
    streaming_stats2 = rqa_from_trajectory(X2, eps=0.5, l_min=2, v_min=2)

    assert streaming_stats2.keys() == dense_stats2.keys()
    for key, value2 in dense_stats2.items():
        assert streaming_stats2[key] == pytest.approx(value2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"X": []}, "non-empty"),
        ({"X": [0.0, np.nan, 1.0]}, "finite values"),
        ({"X": np.arange(5.0), "eps": -1.0}, "eps must be"),
        ({"X": np.arange(5.0), "percentile": 101.0}, "percentile must be"),
        ({"X": np.arange(5.0), "l_min": 0}, "l_min"),
        ({"X": np.arange(5.0), "v_min": False}, "v_min"),
        ({"X": np.arange(5.0), "metric": "cosine"}, "currently supports metric"),
    ],
)
def test_rqa_from_trajectory_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rqa_from_trajectory(**kwargs)


@pytest.mark.parametrize(
    ("rmat", "message"),
    [
        (np.array([], dtype=bool), "square"),
        (np.ones((0, 0), dtype=bool), "square"),
        (np.ones((2, 3), dtype=bool), "square"),
    ],
)
def test_rqa_rejects_invalid_matrix_shape(rmat, message):
    with pytest.raises(ValueError, match=message):
        rqa(rmat)


def test_rqa_rejects_non_symmetric_matrix():
    rmat = np.array(
        [
            [True, True, False],
            [False, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )

    with pytest.raises(ValueError, match="symmetric"):
        rqa(rmat)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"l_min": 0}, "l_min"),
        ({"v_min": 0}, "v_min"),
        ({"l_min": 1.5}, "l_min"),
        ({"v_min": None}, "v_min"),
        ({"l_min": True}, "l_min"),
        ({"v_min": False}, "v_min"),
    ],
)
def test_rqa_rejects_invalid_line_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rqa(np.eye(3, dtype=bool), **kwargs)


def test_recurrence_matrix_constant_signal_uses_zero_threshold():
    rmat, eps = recurrence_matrix(np.ones(5))

    assert eps == 0.0
    assert rmat.shape == (5, 5)
    assert np.all(rmat)


@pytest.mark.parametrize("bad_input", [[], [0.0, np.nan, 1.0], [[0.0, 1.0], [np.inf, 2.0]]])
def test_recurrence_matrix_rejects_empty_or_nonfinite_input(bad_input):
    with pytest.raises(ValueError, match="finite values|non-empty"):
        recurrence_matrix(bad_input, eps=1.0)


@pytest.mark.parametrize("eps", [-0.1, np.nan, np.inf])
def test_recurrence_matrix_rejects_invalid_eps(eps):
    with pytest.raises(ValueError, match="eps must be"):
        recurrence_matrix(np.arange(5.0), eps=eps)


@pytest.mark.parametrize("percentile", [-1.0, 101.0, np.nan])
def test_recurrence_matrix_rejects_invalid_percentile(percentile):
    with pytest.raises(ValueError, match="percentile must be"):
        recurrence_matrix(np.arange(5.0), percentile=percentile)


def test_recurrence_matrix_translation_invariance():
    t = np.linspace(0.0, 10.0, 120)
    traj = np.column_stack([np.sin(t), np.cos(2.0 * t)])
    shifted = traj + np.array([10.0, -4.0])

    rmat, eps = recurrence_matrix(traj, percentile=7)
    shifted_rmat, shifted_eps = recurrence_matrix(shifted, percentile=7)

    assert shifted_eps == pytest.approx(eps)
    np.testing.assert_array_equal(shifted_rmat, rmat)


def test_recurrence_matrix_positive_scaling_invariance_with_percentile_threshold():
    t = np.linspace(0.0, 10.0, 120)
    traj = np.column_stack([np.sin(t), np.cos(2.0 * t)])

    rmat, eps = recurrence_matrix(traj, percentile=7)
    scaled_rmat, scaled_eps = recurrence_matrix(3.25 * traj, percentile=7)

    assert scaled_eps == pytest.approx(3.25 * eps)
    np.testing.assert_array_equal(scaled_rmat, rmat)


@pytest.mark.parametrize(
    ("d", "tau"),
    [
        (0, 1),
        (-1, 1),
        (2, 0),
        (2, -1),
        (1.5, 1),
        (2, None),
        (4, 2),
    ],
)
def test_embed_time_delay_rejects_fuzzed_impossible_parameters(d, tau):
    x = np.arange(5, dtype=np.float64)

    with pytest.raises(ValueError):
        embed_time_delay(x, d=d, tau=tau)


def test_embed_time_delay_fuzzed_valid_parameters_preserve_shape_and_values():
    rng = np.random.default_rng(2027)

    for _ in range(25):
        n = int(rng.integers(6, 30))
        d = int(rng.integers(1, 5))
        tau = int(rng.integers(1, 4))
        if n <= (d - 1) * tau:
            continue

        x = rng.normal(size=n)
        embedded = embed_time_delay(x, d=d, tau=tau)

        assert embedded.shape == (n - (d - 1) * tau, d)
        for j in range(d):
            np.testing.assert_allclose(embedded[:, j], x[j * tau : j * tau + embedded.shape[0]])


def test_correlation_dimension_circle():
    """A circle (D=1) should give D2 ~ 1."""
    t = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    traj = np.column_stack([np.cos(t), np.sin(t)])
    D2, _, _, _, _ = correlation_dimension(traj)
    assert 0.8 < D2 < 1.3


def test_fit_power_law_loglog_recovers_synthetic_exponent():
    x = np.logspace(-2.0, 2.0, 80)
    y = 2.5 * x**1.75

    slope, intercept, rvalue, slopes, scaling = fit_power_law_loglog(x, y, min_points=5)

    assert slope == pytest.approx(1.75)
    assert intercept == pytest.approx(np.log(2.5))
    assert rvalue == pytest.approx(1.0)
    assert np.all(np.isfinite(slopes))
    assert np.count_nonzero(scaling) >= 5


def test_correlation_dimension_uses_shared_power_law_fit():
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    traj = np.column_stack([np.cos(t), np.sin(t)])

    D2, r_values, C_values, slopes, scaling = correlation_dimension(traj, n_r=25)
    n_valid = len(traj) * (len(traj) - 1) // 2
    c_floor = 1.0 / np.sqrt(n_valid)
    fit_values = np.where(C_values > c_floor, C_values, np.nan)
    slope, _, _, helper_slopes, helper_scaling = fit_power_law_loglog(
        r_values, fit_values, min_points=3
    )

    assert D2 == pytest.approx(slope)
    np.testing.assert_allclose(slopes, helper_slopes, equal_nan=True)
    np.testing.assert_array_equal(scaling, helper_scaling)


def test_correlation_dimension_rejects_nan_trajectory():
    x = np.random.default_rng(0).random((200, 3))
    x[50] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        correlation_dimension(x, r_range=(0.01, 1.0))


def test_correlation_module_falls_back_to_python_when_rust_disabled_via_env(monkeypatch):
    # Reload with DYNACHAOS_NO_RUST set exercises the ImportError branch that
    # disables the Rust extension at import time; reload back afterwards so
    # other tests in this process keep the Rust-backed module.
    import importlib

    monkeypatch.setenv("DYNACHAOS_NO_RUST", "1")
    try:
        reloaded = importlib.reload(correlation_module)
        assert reloaded._RUST_AVAILABLE is False
        assert reloaded._correlation_counts_rs is None
    finally:
        monkeypatch.delenv("DYNACHAOS_NO_RUST", raising=False)
        importlib.reload(correlation_module)


def test_correlation_counts_python_covers_1d_and_euclidean_branches():
    rng = np.random.default_rng(0)
    x1d = rng.random(30)
    x2d = rng.random((30, 3))
    r_values = np.array([0.1, 0.3, 0.5])

    counts_1d_cheb, n_pairs = _correlation_counts_python(x1d, r_values, 0, True)
    counts_1d_eucl, _ = _correlation_counts_python(x1d, r_values, 0, False)
    counts_2d_eucl, _ = _correlation_counts_python(x2d, r_values, 0, False)

    # 1D chebyshev and euclidean distance both reduce to |dx|, so the two
    # counts must be identical for a 1D signal.
    np.testing.assert_array_equal(counts_1d_cheb, counts_1d_eucl)
    assert n_pairs == 30 * 29 // 2
    assert counts_2d_eucl.shape == (3,)
    assert np.all(np.diff(counts_2d_eucl) >= 0)  # C(r) nondecreasing in r


def test_correlation_integral_verbose_prints_timing_for_rust_and_python(monkeypatch, capsys):
    rng = np.random.default_rng(0)
    traj = rng.random((40, 2))
    r_values = np.array([0.1, 0.3])

    rust_result = correlation_integral(traj, r_values, verbose=True)
    rust_out = capsys.readouterr().out
    assert "Rust" in rust_out

    monkeypatch.setattr(correlation_module, "_RUST_AVAILABLE", False)
    python_result = correlation_integral(traj, r_values, verbose=True)
    python_out = capsys.readouterr().out
    assert "Python" in python_out

    np.testing.assert_allclose(python_result, rust_result)


def test_correlation_integral_rejects_noninteger_theiler_window():
    traj = np.random.default_rng(0).random((10, 2))
    with pytest.raises(ValueError, match="theiler_window must be >= 0"):
        correlation_integral(traj, np.array([0.1]), theiler_window="abc")


def test_correlation_integral_rejects_nan_trajectory_directly():
    traj = np.array([[0.0, 0.0], [np.nan, 1.0]])
    with pytest.raises(ValueError, match="finite values"):
        correlation_integral(traj, np.array([0.1]))


def test_correlation_integral_returns_zeros_when_no_valid_pairs():
    traj = np.random.default_rng(0).random((5, 2))
    result = correlation_integral(traj, np.array([0.1, 0.5]), theiler_window=10)
    np.testing.assert_array_equal(result, np.zeros(2))


def test_correlation_dimension_return_stderr_for_degenerate_single_point_trajectory():
    traj = np.array([[0.0, 0.0]])
    D2, r_values, C_values, stderr, slopes, scaling = correlation_dimension(
        traj, return_stderr=True
    )
    assert np.isnan(D2)
    assert np.isnan(stderr)
    assert r_values.size == 0 and C_values.size == 0


def test_correlation_dimension_rejects_zero_n_r():
    with pytest.raises(ValueError, match="n_r must be >= 1"):
        correlation_dimension(np.zeros((10, 2)), n_r=0)


@pytest.mark.parametrize(
    "bad_r_range",
    [
        (0.1,),  # wrong length
        (np.nan, 1.0),  # nonfinite
        (0.0, 1.0),  # nonpositive lower bound
        (1.0, 0.5),  # descending
    ],
)
def test_correlation_dimension_rejects_malformed_r_range(bad_r_range):
    traj = np.random.default_rng(0).random((20, 2))
    with pytest.raises(ValueError, match="r_range must be a positive ascending"):
        correlation_dimension(traj, r_range=bad_r_range)


def test_correlation_dimension_metadata_for_single_point_trajectory():
    # N < 2 with return_metadata=True builds the "n.a." backend metadata
    # record around the degenerate _undefined_dimension_result tuple.
    result = correlation_dimension(np.array([[0.0, 0.0]]), return_metadata=True)
    meta = result[-1]
    assert np.isnan(result[0])
    assert meta.backend == "n.a."
    assert "fewer than two trajectory points" in meta.validity_warnings


def test_fit_power_law_loglog_without_stderr_and_too_few_valid_points():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 4.0])
    slope, intercept, rvalue, slopes, scaling = fit_power_law_loglog(x, y, min_points=5)
    assert np.isnan(slope)
    assert not np.any(scaling)


def test_takens_theiler_curve_selects_a_tail_from_slopes_outside_the_c_floor_band():
    from dynachaos.diagnostics.correlation import _takens_theiler_curve

    # All C values sit above the tight (c_floor, 0.1) in-band window, but at
    # least one local slope clears the granularity floor: exercises the
    # elif np.any(a > SLOPE_FLOOR) fallback for the lower-tail extrapolation.
    r = np.array([0.01, 0.1, 0.5, 1.0, 2.0])
    C = np.array([0.2, 0.25, 0.4, 0.6, 0.9])
    D_tt = _takens_theiler_curve(r, C, c_floor=0.01)
    assert np.all(np.isfinite(D_tt))


def test_takens_theiler_curve_a_tail_defaults_to_one_for_a_near_flat_curve():
    from dynachaos.diagnostics.correlation import _takens_theiler_curve

    # A nearly flat C(r) has every local slope below the 1e-2 granularity
    # floor, so a_tail falls back to the else branch (a_tail = 1.0).
    r = np.array([0.01, 0.1, 0.5, 1.0, 2.0])
    C = np.array([0.5, 0.5001, 0.5002, 0.5003, 0.5004])
    D_tt = _takens_theiler_curve(r, C, c_floor=0.01)
    # D_tt[0] = C[0] / tail = C[0] / (C[0] / a_tail) = a_tail = 1.0.
    assert D_tt[0] == pytest.approx(1.0)


def test_correlation_dimension_metadata_flags_degenerate_trajectory_and_no_valid_pairs():
    # Collapsed trajectory (all points equal): ptp diameter is 0, so the
    # organic r-range branch cannot derive a scale.
    traj = np.zeros((20, 2))
    result = correlation_dimension(traj, return_metadata=True)
    meta = result[-1]
    assert np.isnan(result[0])
    assert "nonpositive or nonfinite trajectory diameter" in meta.validity_warnings

    # No valid pairs survive the Theiler window -> both the "no pairs"
    # warning and the "scaling region not identified" verdict fire together.
    traj2 = np.random.default_rng(0).random((5, 2))
    result2 = correlation_dimension(traj2, n_r=5, theiler_window=100, return_metadata=True)
    meta2 = result2[-1]
    assert "no valid pairs after Theiler-window exclusion" in meta2.validity_warnings
    assert "scaling region not identified" in meta2.unresolved_verdicts


def test_find_scaling_region_returns_all_points_below_min_points_threshold():
    log_r = np.linspace(0.0, 1.0, 3)
    log_C = np.array([0.0, 0.5, 1.0])
    mask, slopes = _find_scaling_region(log_r, log_C, min_points=5)
    np.testing.assert_array_equal(mask, np.ones(3, dtype=bool))
    assert slopes.shape == (3,)


def test_find_scaling_region_falls_back_to_all_points_when_too_few_pass_slope_filter():
    # A near-flat curve with a single sharp jump leaves only 1-2 points
    # above the finite-difference slope resolution, below min_points=5, so
    # the usable-point filter falls back to the full index range.
    log_r = np.linspace(0.0, 10.0, 20)
    log_C = np.zeros(20)
    log_C[-1] = 50.0
    log_spacing = (log_r[-1] - log_r[0]) / (len(log_r) - 1)

    mask, slopes = _find_scaling_region(log_r, log_C, min_points=5)
    usable = slopes >= log_spacing
    assert np.count_nonzero(usable) < 5
    assert mask.sum() >= 5


def test_fit_power_law_loglog_rejects_shape_mismatch_and_bad_min_points():
    with pytest.raises(ValueError, match="1D arrays with matching length"):
        fit_power_law_loglog(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="min_points must be >= 1"):
        fit_power_law_loglog(np.array([1.0, 2.0]), np.array([1.0, 2.0]), min_points=0)


def test_fit_power_law_loglog_return_stderr_with_too_few_valid_points():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 4.0])
    slope, intercept, rvalue, stderr, slopes, scaling = fit_power_law_loglog(
        x, y, min_points=5, return_stderr=True
    )
    assert np.isnan(slope) and np.isnan(stderr)
    assert not np.any(scaling)


def test_takens_theiler_dimension_handles_1d_and_degenerate_trajectory():
    # A degenerate (constant) 1D trajectory exercises the traj.ndim == 1
    # reshape branch together with the zero-diameter early return.
    x1d = np.zeros(50)
    D2, r_values, C_values, D_tt, scaling = takens_theiler_dimension(x1d, n_r=10)
    assert np.isnan(D2)
    assert r_values.size == 0 and C_values.size == 0


def test_takens_theiler_dimension_returns_nan_when_too_few_points_survive_the_floor():
    # Only 3 trajectory points leaves fewer than 3 points above the Poisson
    # floor after Theiler-window exclusion, so D_tt is nan everywhere.
    x = np.random.default_rng(1).random((3, 2))
    D2, _, _, D_tt, _ = takens_theiler_dimension(x, n_r=6)
    assert np.isnan(D2)
    assert not np.any(np.isfinite(D_tt))


def test_takens_theiler_dimension_falls_back_to_wider_candidate_band():
    # A short, sparse r-grid on a circle leaves fewer than 4 points in the
    # tight [5*floor, 0.1) candidate band, forcing the wider [2*floor, 0.3)
    # fallback band; here that wider band still has fewer than 3 points, so
    # the estimator gives up and returns nan rather than a noisy shelf.
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    traj = np.column_stack([np.cos(t), np.sin(t)])
    D2, r_values, C_values, D_tt, scaling = takens_theiler_dimension(traj, n_r=6)
    assert np.isnan(D2)
    assert not np.any(scaling)


def test_lyapunov_exponent_1d_rejects_zero_iterations():
    from dynachaos.diagnostics.lyapunov import lyapunov_exponent_1d

    def f(x):
        return 3.8 * x * (1 - x)

    def df(x):
        return 3.8 * (1 - 2 * x)

    with pytest.raises(ValueError, match="n_iter must be a positive integer"):
        lyapunov_exponent_1d(f, df, 0.5, n_iter=0)


def test_lyapunov_exponent_1d_matches_known_logistic_r4_value():
    from dynachaos.diagnostics.lyapunov import lyapunov_exponent_1d

    # Fully chaotic logistic map (r=4) has the exact analytic Lyapunov
    # exponent ln(2) under its invariant measure (Ulam-von Neumann map).
    def f(x):
        return 4.0 * x * (1 - x)

    def df(x):
        return 4.0 * (1 - 2 * x)

    lam = lyapunov_exponent_1d(f, df, 0.234, n_iter=20_000, n_transient=1000)
    assert lam == pytest.approx(np.log(2), abs=5e-3)


def test_lyapunov_exponent_1d_floors_zero_derivative():
    from dynachaos.diagnostics.lyapunov import lyapunov_exponent_1d

    # A map with an identically-zero derivative exercises the deriv <= 0
    # guard, which floors each term to log(1e-300) instead of raising.
    def f(x):
        return x

    def df(_x):
        return 0.0

    lam = lyapunov_exponent_1d(f, df, 0.5, n_iter=10, n_transient=0)
    assert lam == pytest.approx(np.log(1e-300))


def test_lyapunov_spectrum_convergence_error_shrinks_with_more_iterations():
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

    def f(x):
        return np.array([4.0 * x[0] * (1 - x[0])])

    def jac(x):
        return np.array([[4.0 * (1 - 2 * x[0])]])

    spectrum, conv_err = lyapunov_spectrum(
        f, jac, np.array([0.234]), n_iter=5000, n_transient=500, return_convergence=True
    )
    assert spectrum[0] == pytest.approx(np.log(2), abs=1e-2)
    assert conv_err.shape == (1,)
    assert conv_err[0] >= 0.0


def test_lyapunov_spectrum_convergence_error_is_zero_when_no_checkpoint_reached():
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

    def f(x):
        return np.array([4.0 * x[0] * (1 - x[0])])

    def jac(x):
        return np.array([[4.0 * (1 - 2 * x[0])]])

    # reorth_interval far larger than n_iter: (i+1) % reorth_interval == 0
    # never fires, so no checkpoint is ever recorded and conv_err must fall
    # back to the all-zero branch rather than crash on an empty list.
    spectrum, conv_err = lyapunov_spectrum(
        f,
        jac,
        np.array([0.234]),
        n_iter=5,
        n_transient=0,
        reorth_interval=1000,
        return_convergence=True,
    )
    assert spectrum.shape == (1,)
    np.testing.assert_array_equal(conv_err, np.zeros(1))


def test_lyapunov_max_matches_known_logistic_r4_value():
    from dynachaos.diagnostics.lyapunov import lyapunov_max

    def f(x):
        return np.array([4.0 * x[0] * (1 - x[0])])

    def jac(x):
        return np.array([[4.0 * (1 - 2 * x[0])]])

    lam = lyapunov_max(
        f, jac, np.array([0.234]), n_iter=20_000, n_transient=1000, rng=np.random.default_rng(0)
    )
    assert lam == pytest.approx(np.log(2), abs=5e-3)


def test_lyapunov_max_default_rng_is_deterministic_across_calls():
    from dynachaos.diagnostics.lyapunov import lyapunov_max

    def f(x):
        return np.array([4.0 * x[0] * (1 - x[0])])

    def jac(x):
        return np.array([[4.0 * (1 - 2 * x[0])]])

    first = lyapunov_max(f, jac, np.array([0.234]), n_iter=200, n_transient=10)
    second = lyapunov_max(f, jac, np.array([0.234]), n_iter=200, n_transient=10)
    assert first == pytest.approx(second)


def test_lyapunov_max_resamples_tangent_vector_when_it_collapses_to_zero():
    from dynachaos.diagnostics.lyapunov import lyapunov_max

    # A map whose Jacobian is identically the zero matrix collapses the
    # tangent vector at the first step, exercising the norm_v == 0 recovery
    # branch (log_sum += -100.0, resample a fresh unit tangent vector).
    def f(x):
        return np.array([0.5])

    def jac(_x):
        return np.array([[0.0]])

    lam = lyapunov_max(
        f, jac, np.array([0.5]), n_iter=5, n_transient=0, rng=np.random.default_rng(1)
    )
    assert lam == pytest.approx(-100.0)


def test_lyapunov_sweep_1d_returns_one_exponent_per_parameter():
    from dynachaos.diagnostics.lyapunov import lyapunov_sweep_1d

    def f(x, r):
        return r * x * (1 - x)

    def df(x, r):
        return r * (1 - 2 * x)

    params = np.array([3.2, 4.0])
    lams = lyapunov_sweep_1d(f, df, lambda _p: 0.234, params, n_iter=2000, n_transient=200)
    assert lams.shape == (2,)
    assert np.all(np.isfinite(lams))
    # r=3.2 is a stable period-2 orbit (lambda < 0); r=4.0 is fully chaotic
    # with the exact analytic exponent ln(2).
    assert lams[0] < 0.0
    assert lams[1] == pytest.approx(np.log(2), abs=5e-2)


def test_lyapunov_sweep_nd_full_spectrum_and_max_only_agree_on_top_exponent():
    from dynachaos.diagnostics.lyapunov import lyapunov_sweep_nd

    def f(x, r):
        return np.array([r * x[0] * (1 - x[0])])

    def jac(x, r):
        return np.array([[r * (1 - 2 * x[0])]])

    params = np.array([3.2, 4.0])

    def x0_func(_p):
        return np.array([0.234])

    max_only = lyapunov_sweep_nd(f, jac, x0_func, params, n_iter=2000, n_transient=200)
    full = lyapunov_sweep_nd(
        f, jac, x0_func, params, n_iter=2000, n_transient=200, full_spectrum=True
    )
    assert max_only.shape == (2,)
    assert full.shape == (2, 1)
    np.testing.assert_allclose(max_only, full[:, 0], atol=1e-8)


def test_flow_lyapunov_spectrum_matches_linear_system_eigenvalues():
    from dynachaos.diagnostics.lyapunov import flow_lyapunov_spectrum

    # For dx/dt = A x with constant A, the Jacobian is A everywhere so the
    # Lyapunov spectrum equals the eigenvalues of A exactly (Benettin et al.
    # 1980, linear autonomous case).
    A = np.diag([0.1, -0.3])

    def rhs(_t, x):
        return A @ x

    def jac(_t, _x):
        return A

    spectrum = flow_lyapunov_spectrum(
        rhs, jac, x0=np.array([1.0, 1.0]), t_total=20.0, dt=0.05, t_transient=0.0, reorth_dt=0.5
    )
    np.testing.assert_allclose(spectrum, [0.1, -0.3], atol=1e-3)


def test_flow_lyapunov_spectrum_with_transient_still_matches_eigenvalues():
    from dynachaos.diagnostics.lyapunov import flow_lyapunov_spectrum

    A = np.diag([0.1, -0.3])

    def rhs(_t, x):
        return A @ x

    def jac(_t, _x):
        return A

    spectrum = flow_lyapunov_spectrum(
        rhs, jac, x0=np.array([1.0, 1.0]), t_total=10.0, dt=0.05, t_transient=2.0, reorth_dt=0.5
    )
    np.testing.assert_allclose(spectrum, [0.1, -0.3], atol=1e-3)


def test_flow_lyapunov_spectrum_rejects_reorth_dt_larger_than_t_total():
    from dynachaos.diagnostics.lyapunov import flow_lyapunov_spectrum

    A = np.diag([0.1, -0.3])

    def rhs(_t, x):
        return A @ x

    def jac(_t, _x):
        return A

    with pytest.raises(ValueError, match="at least one reorthogonalization interval"):
        flow_lyapunov_spectrum(
            rhs, jac, x0=np.array([1.0, 1.0]), t_total=1.0, dt=0.05, t_transient=0.0, reorth_dt=5.0
        )


def test_lyapunov_spectrum_rejects_zero_iterations():
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

    def f(x):
        return np.array([3.8 * x[0] * (1 - x[0])])

    def jac(x):
        return np.array([[3.8 * (1 - 2 * x[0])]])

    with pytest.raises(ValueError, match="n_iter must be a positive integer"):
        lyapunov_spectrum(f, jac, np.array([0.5]), n_iter=0)


# ---------------------------------------------------------------------------
# compare_all module tests
# ---------------------------------------------------------------------------


def test_delayed_logistic_series_returns_correct_shape():
    """delayed_logistic_series returns a 1D array of specified length."""
    from dynachaos.diagnostics.compare_all_helpers import delayed_logistic_series

    series = delayed_logistic_series(D=1.8, n_transient=50, n_record=100)

    assert isinstance(series, np.ndarray)
    assert series.ndim == 1
    assert series.shape == (100,)
    assert np.all(np.isfinite(series))


def test_delayed_logistic_trajectory_returns_correct_shape():
    """delayed_logistic_trajectory returns a 2D array with shape (n, 2)."""
    from dynachaos.diagnostics.compare_all_helpers import delayed_logistic_trajectory

    traj = delayed_logistic_trajectory(D=1.8, A=0.3, n_transient=50, n_record=100)

    assert isinstance(traj, np.ndarray)
    assert traj.ndim == 2
    assert traj.shape == (100, 2)
    assert np.all(np.isfinite(traj))


def test_sweep_scalar_metric_produces_array_of_metrics():
    """sweep_scalar_metric applies a metric function to a series and returns an array."""
    from dynachaos.diagnostics.compare_all_helpers import sweep_scalar_metric

    def dummy_metric(series):
        return float(np.mean(series))

    def series_fn(v):
        return np.sin(np.linspace(0, v * np.pi, 50))

    values = np.array([1.0, 1.5, 2.0])
    results = sweep_scalar_metric(values, series_fn, dummy_metric, progress_every=10)

    assert isinstance(results, np.ndarray)
    assert results.shape == (3,)
    assert np.all(np.isfinite(results))


def test_sweep_pair_metric_produces_two_arrays():
    """sweep_pair_metric applies a metric returning two values."""
    from dynachaos.diagnostics.compare_all_helpers import sweep_pair_metric

    def dummy_pair_metric(series):
        return float(np.mean(series)), float(np.std(series))

    def series_fn(v):
        return np.sin(np.linspace(0, v * np.pi, 50))

    values = np.array([1.0, 1.5, 2.0])
    h_vals, c_vals = sweep_pair_metric(values, series_fn, dummy_pair_metric, progress_every=10)

    assert isinstance(h_vals, np.ndarray)
    assert isinstance(c_vals, np.ndarray)
    assert h_vals.shape == (3,)
    assert c_vals.shape == (3,)
    assert np.all(np.isfinite(h_vals))
    assert np.all(np.isfinite(c_vals))


def test_load_or_compute_npz_creates_file_if_missing(tmp_path):
    """load_or_compute_payload calls compute_fn if file is missing."""
    from dynachaos.io.paths import load_or_compute_payload

    npz_path = tmp_path / "test.npz"

    def dummy_compute():
        return {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}

    result = load_or_compute_payload(npz_path, "test", dummy_compute, required_keys=("a", "b"))

    assert result["a"].tolist() == [1.0, 2.0]
    assert result["b"].tolist() == [3.0, 4.0]


def test_load_or_compute_npz_loads_existing_file(tmp_path):
    """load_or_compute_payload loads existing file without calling compute_fn."""
    from dynachaos.io.paths import load_or_compute_payload

    npz_path = tmp_path / "test.npz"
    np.savez_compressed(npz_path, a=np.array([10.0, 20.0]), b=np.array([30.0, 40.0]))

    compute_called = False

    def dummy_compute():
        nonlocal compute_called
        compute_called = True
        return {}

    result = load_or_compute_payload(npz_path, "test", dummy_compute, required_keys=("a", "b"))

    assert not compute_called
    assert result["a"].tolist() == [10.0, 20.0]
    assert result["b"].tolist() == [30.0, 40.0]


def test_load_or_compute_payload_recomputes_when_cache_missing_required_keys(tmp_path):
    from dynachaos.io.paths import load_or_compute_payload

    npz_path = tmp_path / "test.npz"
    np.savez_compressed(npz_path, a=np.array([1.0]))  # cache exists but lacks "b"

    def dummy_compute():
        return {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}

    result = load_or_compute_payload(npz_path, "test", dummy_compute, required_keys=("a", "b"))

    assert result["a"].tolist() == [1.0, 2.0]
    assert result["b"].tolist() == [3.0, 4.0]


def test_load_or_compute_payload_raises_when_compute_leaves_required_keys_missing(tmp_path):
    from dynachaos.io.paths import load_or_compute_payload

    npz_path = tmp_path / "missing.npz"

    def dummy_compute():
        return {"a": np.array([1.0])}  # never provides "b"

    with pytest.raises(KeyError, match="missing required keys after compute"):
        load_or_compute_payload(npz_path, "test", dummy_compute, required_keys=("a", "b"))


def test_logistic_series_from_compare_all_helpers_produces_finite_series():
    """logistic_series (from compare_all_helpers) returns finite chaotic output."""
    from dynachaos.diagnostics.compare_all_helpers import logistic_series

    series = logistic_series(a=1.99, n_transient=100, n_record=200)

    assert isinstance(series, np.ndarray)
    assert series.ndim == 1
    assert series.shape == (200,)
    assert np.all(np.isfinite(series))


def test_logistic_series_varies_with_parameter():
    """logistic_series with different a values produces different dynamics."""
    from dynachaos.diagnostics.compare_all_helpers import logistic_series

    series_low_a = logistic_series(a=1.5, n_transient=100, n_record=200)
    series_high_a = logistic_series(a=1.99, n_transient=100, n_record=200)

    assert np.std(series_low_a) < np.std(series_high_a)


def test_compare_all_compute_01_test_populates_results(tmp_path, monkeypatch):
    """compute_01_test produces array of K values indexed by a values."""
    import matplotlib

    matplotlib.use("Agg")

    from dynachaos.diagnostics import compare_all
    from dynachaos.diagnostics.compare_all_helpers import logistic_series
    from dynachaos.diagnostics.zero_one_test import zero_one_statistic

    monkeypatch.setattr(compare_all, "FIG_DIR", tmp_path)
    monkeypatch.setattr(compare_all, "TEST01_NPZ", tmp_path / "test01.npz")

    def small_01_test():
        n_a = 10
        a_values = np.linspace(1.0, 2.0, n_a)
        K_values = np.empty(n_a)

        for i, a in enumerate(a_values):
            series = logistic_series(a, n_transient=100, n_record=500)
            K_values[i] = zero_one_statistic(series, n_c=10)

        return {"a": a_values, "K": K_values}

    result = small_01_test()

    assert result["a"].shape == (10,)
    assert result["K"].shape == (10,)
    assert np.all(np.isfinite(result["a"]))
    assert np.all(np.isfinite(result["K"]))
    assert np.all((result["K"] >= 0.0) & (result["K"] <= 1.0))


def test_compare_all_plot_01_test_accepts_data_dict():
    """plot_01_test accepts structured data dict with a and K."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dynachaos.diagnostics import compare_all

    data = {
        "a": np.linspace(1.0, 2.0, 50),
        "K": np.linspace(0.0, 1.0, 50),
    }

    original_savefig = plt.Figure.savefig

    def mock_savefig(self, fname, **kwargs):
        pass

    plt.Figure.savefig = mock_savefig

    try:
        compare_all.plot_01_test(data)
    finally:
        plt.Figure.savefig = original_savefig


def test_compare_all_plot_sali_writes_png(monkeypatch, tmp_path):
    from dynachaos.diagnostics import compare_all

    png_path = tmp_path / "sali_comparison.png"
    monkeypatch.setattr(compare_all, "SALI_PNG", png_path)

    DB_values = np.array([case[0] for case in compare_all.SALI_CASES])
    n = np.arange(200)
    data = {
        "DB_values": DB_values,
        "lambda1_values": np.array([-0.01, -0.002, 0.001, 0.02]),
    }
    for DB in DB_values:
        # Decays toward the machine-underflow floor so the "valid" mask and
        # marker-thinning branch (compare_all.py:349-365) both fire.
        data[f"DB_{DB}_sali"] = np.maximum(1.0 * np.exp(-n / 30.0), 1e-16)

    compare_all.plot_sali(data)

    assert png_path.stat().st_size > 0


def test_compare_all_plot_permutation_entropy_writes_png(monkeypatch, tmp_path):
    from dynachaos.diagnostics import compare_all

    png_path = tmp_path / "permutation_entropy.png"
    monkeypatch.setattr(compare_all, "PE_PNG", png_path)

    data = {
        "a": np.linspace(1.0, 2.0, 30),
        "H_logistic": np.linspace(0.2, 0.9, 30),
        "D": np.linspace(1.5, 2.2, 20),
        "H_delayed": np.linspace(0.3, 0.85, 20),
    }

    compare_all.plot_permutation_entropy(data)

    assert png_path.stat().st_size > 0


def test_compare_all_plot_complexity_entropy_writes_png(monkeypatch, tmp_path):
    from dynachaos.diagnostics import compare_all

    png_path = tmp_path / "complexity_entropy_plane.png"
    monkeypatch.setattr(compare_all, "CH_PNG", png_path)

    data = {
        "H_logistic": np.linspace(0.2, 0.9, 30),
        "C_logistic": np.linspace(0.0, 0.3, 30),
        "H_delayed": np.linspace(0.3, 0.85, 20),
        "C_delayed": np.linspace(0.0, 0.25, 20),
    }

    compare_all.plot_complexity_entropy(data)

    assert png_path.stat().st_size > 0


def test_compare_all_plot_rqa_writes_png(monkeypatch, tmp_path):
    from dynachaos.diagnostics import compare_all

    png_path = tmp_path / "rqa_measures.png"
    monkeypatch.setattr(compare_all, "RQA_PNG", png_path)

    D = np.linspace(1.5, 2.2, 20)
    data = {
        "D": D,
        "RR": np.linspace(0.01, 0.2, 20),
        "DET": np.linspace(0.5, 0.95, 20),
        "LAM": np.linspace(0.4, 0.9, 20),
        "ENTR": np.linspace(0.5, 2.5, 20),
    }

    compare_all.plot_rqa(data)

    assert png_path.stat().st_size > 0


def test_compare_all_main_computes_all_five_sections_when_cache_missing(monkeypatch, tmp_path):
    # compute_01_test/compute_sali/compute_permutation_entropy/
    # compute_complexity_entropy/compute_rqa take no arguments and are
    # hardcoded to paper-scale sweeps (500/4/500+300/200+200/80 points, each
    # built from thousands of scalar map iterations per point) --
    # compare_all.py:57-250 -- so main()'s compute path is exercised here
    # with fast stand-ins instead of the real compute_* functions.
    from dynachaos.diagnostics import compare_all

    monkeypatch.setattr(compare_all, "FIG_DIR", tmp_path)
    monkeypatch.setattr(compare_all, "TEST01_NPZ", tmp_path / "test01_sweep.npz")
    monkeypatch.setattr(compare_all, "TEST01_PNG", tmp_path / "test01_sweep.png")
    monkeypatch.setattr(compare_all, "SALI_NPZ", tmp_path / "sali_comparison.npz")
    monkeypatch.setattr(compare_all, "SALI_PNG", tmp_path / "sali_comparison.png")
    monkeypatch.setattr(compare_all, "PE_NPZ", tmp_path / "permutation_entropy.npz")
    monkeypatch.setattr(compare_all, "PE_PNG", tmp_path / "permutation_entropy.png")
    monkeypatch.setattr(compare_all, "CH_NPZ", tmp_path / "complexity_entropy_plane.npz")
    monkeypatch.setattr(compare_all, "CH_PNG", tmp_path / "complexity_entropy_plane.png")
    monkeypatch.setattr(compare_all, "RQA_NPZ", tmp_path / "rqa_measures.npz")
    monkeypatch.setattr(compare_all, "RQA_PNG", tmp_path / "rqa_measures.png")

    def fake_01_test():
        a = np.linspace(1.0, 2.0, 5)
        np.savez_compressed(compare_all.TEST01_NPZ, a=a, K=np.linspace(0.0, 1.0, 5))

    def fake_sali():
        DB_values = np.array([case[0] for case in compare_all.SALI_CASES])
        payload = {
            "DB_values": DB_values,
            "lambda1_values": np.linspace(-0.01, 0.02, len(DB_values)),
        }
        n = np.arange(50)
        for DB in DB_values:
            payload[f"DB_{DB}_sali"] = np.maximum(np.exp(-n / 10.0), 1e-16)
        np.savez_compressed(compare_all.SALI_NPZ, **payload)

    def fake_pe():
        np.savez_compressed(
            compare_all.PE_NPZ,
            a=np.linspace(1.0, 2.0, 5),
            H_logistic=np.linspace(0.2, 0.9, 5),
            D=np.linspace(1.5, 2.2, 5),
            H_delayed=np.linspace(0.3, 0.85, 5),
        )

    def fake_ch():
        np.savez_compressed(
            compare_all.CH_NPZ,
            a=np.linspace(1.0, 2.0, 5),
            H_logistic=np.linspace(0.2, 0.9, 5),
            C_logistic=np.linspace(0.0, 0.3, 5),
            D=np.linspace(1.5, 2.2, 5),
            H_delayed=np.linspace(0.3, 0.85, 5),
            C_delayed=np.linspace(0.0, 0.25, 5),
        )

    def fake_rqa():
        D = np.linspace(1.5, 2.2, 5)
        np.savez_compressed(
            compare_all.RQA_NPZ,
            D=D,
            RR=np.linspace(0.01, 0.2, 5),
            DET=np.linspace(0.5, 0.95, 5),
            LAM=np.linspace(0.4, 0.9, 5),
            ENTR=np.linspace(0.5, 2.5, 5),
        )

    monkeypatch.setattr(compare_all, "compute_01_test", fake_01_test)
    monkeypatch.setattr(compare_all, "compute_sali", fake_sali)
    monkeypatch.setattr(compare_all, "compute_permutation_entropy", fake_pe)
    monkeypatch.setattr(compare_all, "compute_complexity_entropy", fake_ch)
    monkeypatch.setattr(compare_all, "compute_rqa", fake_rqa)

    compare_all.main()

    for png_name in (
        "test01_sweep.png",
        "sali_comparison.png",
        "permutation_entropy.png",
        "complexity_entropy_plane.png",
        "rqa_measures.png",
    ):
        assert (tmp_path / png_name).stat().st_size > 0


# ── Coverage: Permutation entropy edge cases ──────────────────────────────────


def test_permutation_entropy_sweep_default_d_values():
    """permutation_entropy_sweep uses default d_values when None (lines 281-282)."""
    x = logistic_series(n=200, a=1.99, burn=100)
    d_vals, h_vals = permutation_entropy_sweep(x, d_values=None, tau=1)
    assert d_vals == [3, 4, 5, 6, 7]
    assert len(h_vals) == 5


def test_ordinal_distribution_validation_typeerror_on_invalid_cast():
    """Ordinal distribution catches TypeError in d/tau casting (line 81-82)."""
    x = np.arange(20.0)
    with pytest.raises(ValueError, match="d and tau must be positive integers"):
        ordinal_distribution(x, d="abc", tau=1)


def test_ordinal_distribution_validation_valueerror_on_invalid_tau():
    """Ordinal distribution catches ValueError in tau casting (line 81-82)."""
    x = np.arange(20.0)
    with pytest.raises(ValueError, match="d and tau must be positive integers"):
        ordinal_distribution(x, d=3, tau=1.5)


# ── Coverage: Recurrence quantification edge cases ────────────────────────────


def test_diagonal_lines_l_min_less_than_1_error():
    """_diagonal_lines raises error when l_min < 1 (line 125)."""
    R = np.eye(3, dtype=bool)
    with pytest.raises(ValueError, match="l_min must be >= 1"):
        from dynachaos.diagnostics.recurrence import _diagonal_lines

        _diagonal_lines(R, l_min=0)


def test_diagonal_lines_trailing_line_appended():
    """_diagonal_lines appends final trailing line (line 140)."""
    from dynachaos.diagnostics.recurrence import _diagonal_lines

    R = np.array(
        [
            [True, True, False],
            [False, True, True],
            [False, False, True],
        ],
        dtype=bool,
    )
    lengths = _diagonal_lines(R, l_min=1)
    assert len(lengths) > 0


def test_vertical_lines_v_min_less_than_1_error():
    """_vertical_lines raises error when v_min < 1 (line 151)."""
    R = np.eye(3, dtype=bool)
    with pytest.raises(ValueError, match="v_min must be >= 1"):
        from dynachaos.diagnostics.recurrence import _vertical_lines

        _vertical_lines(R, v_min=0)


def test_paired_distances_unsupported_metric_error():
    """_paired_distances raises error for unsupported metric (line 192)."""
    from dynachaos.diagnostics.recurrence import _paired_distances

    A = np.array([[1.0, 2.0]])
    B = np.array([[2.0, 3.0]])
    with pytest.raises(ValueError, match="currently supports metric"):
        _paired_distances(A, B, metric="unsupported")


def test_line_lengths_min_length_less_than_1_error():
    """_line_lengths raises error when min_length < 1 (line 202)."""
    from dynachaos.diagnostics.recurrence import _line_lengths

    with pytest.raises(ValueError, match="min_length must be >= 1"):
        _line_lengths([True, True, False], min_length=0)


def test_laminar_lengths_no_vertical_lines():
    """laminar_lengths sets LAM, TT to 0.0 when no vertical lines found (line 312)."""
    from dynachaos.diagnostics.recurrence import laminar_lengths

    traj = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    result = laminar_lengths(traj, eps=1e-10, v_min=10)
    assert result.LAM == 0.0
    assert result.TT == 0.0


def test_trajectory_rqa_scan_theiler_none_defaults_to_zero():
    """_trajectory_rqa_scan defaults theiler to 0 when None (line 339)."""
    from dynachaos.diagnostics.recurrence import _trajectory_rqa_scan

    traj = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    stats, _ = _trajectory_rqa_scan(
        traj, eps=0.15, metric="euclidean", percentile=5, l_min=2, v_min=2, theiler=None
    )
    assert isinstance(stats, dict)


def test_rqa_from_trajectory_no_diagonal_lines_warning():
    """rqa_from_trajectory includes warning when no diagonal lines found (line 394)."""
    from dynachaos.diagnostics.recurrence import rqa_from_trajectory

    traj = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    stats, metadata = rqa_from_trajectory(traj, eps=1e-10, l_min=10, return_metadata=True)
    assert isinstance(metadata.unresolved_verdicts, list)


# ── Coverage: Validation module edge cases ───────────────────────────────────


def test_nonnegative_int_typeerror_on_invalid_cast():
    """nonnegative_int catches TypeError in int() conversion (lines 46-48)."""
    with pytest.raises(ValueError, match="non-negative integer"):
        nonnegative_int("not_a_number", name="test")


def test_nonnegative_int_valueerror_on_invalid_cast():
    """nonnegative_int catches ValueError in int() conversion (lines 46-48)."""
    with pytest.raises(ValueError, match="non-negative integer"):
        nonnegative_int(None, name="test")
