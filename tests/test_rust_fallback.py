"""Test that Rust and Python implementations produce identical results.

These tests force both paths and compare outputs to ensure the Rust
acceleration is a transparent drop-in.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import assert_threshold_well_separated, logistic_series

try:
    import dynachaos._rust  # noqa: F401

    _RUST_IMPORTABLE = True
except ImportError:
    _RUST_IMPORTABLE = False

rust_extension = pytest.mark.skipif(not _RUST_IMPORTABLE, reason="compiled extension not built")
_NO_RUST_ENV = bool(os.environ.get("DYNACHAOS_NO_RUST"))
_GOLDEN_PATH = Path(__file__).with_name("data") / "rust_parity_goldens.npz"


def _golden(name):
    with np.load(_GOLDEN_PATH) as goldens:
        return goldens[name]


def _assert_golden(name, value, **kwargs):
    np.testing.assert_allclose(np.asarray(value), _golden(name), **kwargs)


class TestRecurrenceParity:
    """Verify Rust and Python _diagonal_lines / _vertical_lines agree."""

    def _recurrence_matrix(self):
        """Build a recurrence matrix that is identical on every platform.

        This trajectory is a uniformly sampled circle, so it is highly
        degenerate: 79800 pairs take only 399 distinct distances, and 272 of
        them land within 1e-9 of the threshold. That tie-cluster straddles
        ``eps``, and last-bit differences between x86_64/OpenBLAS and
        arm64/Accelerate split it differently on each machine. Thresholding the
        raw distances therefore gave R.sum() 13316 on Linux against 13318 on
        macOS, moving 6 of 77 diagonal line lengths and failing pinned goldens.

        Quantising onto a 1e-9 grid before comparing makes the whole tie-cluster
        move as a unit, so R is bit-identical everywhere (verified: R sha256
        matches on Linux x86_64 and macOS arm64) and the goldens mean the same
        thing on every machine.
        """
        from scipy.spatial.distance import pdist, squareform

        t = np.linspace(0.0, 40.0, 400)
        traj = np.column_stack([np.sin(t), np.cos(t)])

        d = np.round(pdist(traj, metric="euclidean"), 9)
        eps = round(float(np.percentile(d, 8)), 9)
        R = squareform(d) <= eps
        np.fill_diagonal(R, True)
        return R

    def test_diagonal_lines_parity(self):
        R = self._recurrence_matrix()

        from dynachaos.diagnostics import recurrence as rec_mod

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            py_result = rec_mod._diagonal_lines(R, l_min=2)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(
            np.asarray(sorted(py_result), dtype=np.int64),
            _golden("diag_lines"),
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import diagonal_lines as rust_diag

            rs_result = np.asarray(sorted(rust_diag(R, l_min=2)), dtype=np.int64)
            np.testing.assert_array_equal(rs_result, _golden("diag_lines"))
            # The parity contract itself: both backends, same input, same answer.
            # Holds on every platform regardless of how R was built.
            np.testing.assert_array_equal(rs_result, np.asarray(sorted(py_result), dtype=np.int64))

    def test_vertical_lines_parity(self):
        R = self._recurrence_matrix()
        from dynachaos.diagnostics import recurrence as rec_mod

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            py_result = rec_mod._vertical_lines(R, v_min=2)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(
            np.asarray(sorted(py_result), dtype=np.int64),
            _golden("vert_lines"),
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import vertical_lines as rust_vert

            rs_result = np.asarray(sorted(rust_vert(R, v_min=2)), dtype=np.int64)
            np.testing.assert_array_equal(rs_result, _golden("vert_lines"))
            np.testing.assert_array_equal(rs_result, np.asarray(sorted(py_result), dtype=np.int64))

    @pytest.mark.parametrize(
        ("mask", "min_length"),
        [
            ([], 2),
            ([False, False, False], 2),
            ([True, False, True, True, False, True, True, True], 2),
            ([True, True, False, True, False, True, True], 1),
            ([False, True, True, True], 3),
        ],
    )
    def test_count_line_lengths_parity(self, mask, min_length, request):
        from dynachaos.diagnostics import recurrence as rec_mod

        mask_array = np.asarray(mask, dtype=np.bool_)

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            python_result = rec_mod._line_lengths(mask_array, min_length)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag
        golden_name = f"line_lengths_{request.node.callspec.indices['mask']}"
        np.testing.assert_array_equal(
            np.asarray(python_result, dtype=np.int64),
            _golden(golden_name),
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import count_line_lengths as rust_line_lengths

            rust_result = np.asarray(rust_line_lengths(mask_array, min_length))
            np.testing.assert_array_equal(rust_result, _golden(golden_name))

    @rust_extension
    def test_count_line_lengths_rejects_invalid_min_length(self):
        from dynachaos._rust import count_line_lengths as rust_line_lengths

        with pytest.raises(ValueError, match="min_length"):
            rust_line_lengths(np.array([True, False], dtype=np.bool_), 0)

    def test_streaming_rqa_uses_line_scanner_transparently(self):
        from scipy.spatial.distance import pdist

        from dynachaos.diagnostics import recurrence as rec_mod

        t = np.linspace(0.0, 18.0, 120)
        traj = np.column_stack([np.sin(t), np.cos(1.4 * t)])

        # This Lissajous curve is non-degenerate: all 7140 pairwise distances are
        # distinct and the nearest sits ~6e-5 from the threshold, eleven orders of
        # magnitude above double-precision noise. That separation is what makes
        # the pinned values below portable, so assert it rather than assume it.
        distances = pdist(traj, metric="chebyshev")
        assert_threshold_well_separated(distances, float(np.percentile(distances, 7)))

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            python_stats = rec_mod.rqa_from_trajectory(
                traj, percentile=7, metric="chebyshev", l_min=2, v_min=3
            )
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(
            np.asarray(sorted(python_stats)),
            _golden("streaming_rqa_keys"),
        )
        _assert_golden(
            "streaming_rqa_values",
            [python_stats[k] for k in sorted(python_stats)],
            atol=0.0,
            rtol=0.0,
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                rec_mod._RUST_AVAILABLE = True
                rust_stats = rec_mod.rqa_from_trajectory(
                    traj, percentile=7, metric="chebyshev", l_min=2, v_min=3
                )
            finally:
                rec_mod._RUST_AVAILABLE = old_flag

            np.testing.assert_array_equal(
                np.asarray(sorted(rust_stats)),
                _golden("streaming_rqa_keys"),
            )
            _assert_golden(
                "streaming_rqa_values",
                [rust_stats[k] for k in sorted(rust_stats)],
                atol=0.0,
                rtol=0.0,
            )
            # The transparency claim in this test's name: swapping the line
            # scanner must not change a single bit of the result.
            assert [rust_stats[k] for k in sorted(rust_stats)] == [
                python_stats[k] for k in sorted(python_stats)
            ]

    @rust_extension
    @pytest.mark.parametrize(
        ("function_name", "kwargs", "message"),
        [
            ("diagonal_lines", {"l_min": 0}, "l_min"),
            ("vertical_lines", {"v_min": 0}, "v_min"),
            ("diagonal_lines", {}, "square"),
            ("vertical_lines", {}, "square"),
        ],
    )
    def test_direct_rust_recurrence_rejects_invalid_inputs(self, function_name, kwargs, message):
        import dynachaos._rust as rust_mod

        rmat = np.ones((2, 3), dtype=bool)
        if "l_min" in kwargs or "v_min" in kwargs:
            rmat = np.eye(3, dtype=bool)

        with pytest.raises(ValueError, match=message):
            getattr(rust_mod, function_name)(rmat, **kwargs)

    def test_rqa_parity(self):
        """Full RQA pipeline should give same results via either path."""
        R = self._recurrence_matrix()
        from dynachaos.diagnostics import recurrence as rec_mod

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            rqa_python = rec_mod.rqa(R)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(np.asarray(sorted(rqa_python)), _golden("rqa_keys"))
        _assert_golden(
            "rqa_values",
            [rqa_python[k] for k in sorted(rqa_python)],
            atol=1e-12,
            rtol=0.0,
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            rqa_rust = rec_mod.rqa(R)
            _assert_golden(
                "rqa_values",
                [rqa_rust[k] for k in sorted(rqa_rust)],
                atol=1e-12,
                rtol=0.0,
            )


class TestPermutationParity:
    """Verify Rust and Python ordinal_distribution agree."""

    def test_ordinal_distribution_parity(self):
        series = logistic_series(n=5000)

        from dynachaos.diagnostics import permutation as perm_mod

        old_flag = perm_mod._RUST_AVAILABLE
        try:
            perm_mod._RUST_AVAILABLE = False
            py_probs, py_total = perm_mod.ordinal_distribution(series, d=5, tau=1)
        finally:
            perm_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(
            np.asarray([py_total], dtype=np.int64),
            _golden("ordinal_total"),
        )
        np.testing.assert_array_equal(
            np.asarray(sorted(py_probs), dtype=np.int64),
            _golden("ordinal_indices"),
        )
        _assert_golden(
            "ordinal_probs",
            [py_probs[k] for k in sorted(py_probs)],
            atol=1e-12,
            rtol=0.0,
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import ordinal_distribution as rust_ord
            from dynachaos.diagnostics.permutation import _lehmer_to_permutation

            rs_counts, rs_total = rust_ord(series, d=5, tau=1)
            rs_counts = np.asarray(rs_counts)
            np.testing.assert_array_equal(
                np.asarray([rs_total], dtype=np.int64),
                _golden("ordinal_total"),
            )

            rs_probs = {}
            for idx in np.nonzero(rs_counts)[0]:
                perm = _lehmer_to_permutation(int(idx), 5)
                rs_probs[perm] = int(rs_counts[idx]) / rs_total

            np.testing.assert_array_equal(
                np.asarray(sorted(rs_probs), dtype=np.int64),
                _golden("ordinal_indices"),
            )
            _assert_golden(
                "ordinal_probs",
                [rs_probs[k] for k in sorted(rs_probs)],
                atol=1e-12,
                rtol=0.0,
            )

    @rust_extension
    @pytest.mark.parametrize(
        ("d", "tau", "message"),
        [
            (0, 1, "d must be >= 2"),
            (1, 1, "d must be >= 2"),
            (11, 1, "d must be <= 10"),
            (2, 0, "tau must be >= 1"),
            (5, 1, "time series is too short"),
        ],
    )
    def test_ordinal_distribution_rejects_invalid_direct_rust_inputs(self, d, tau, message):
        from dynachaos._rust import ordinal_distribution as rust_ord

        with pytest.raises(ValueError, match=message):
            rust_ord(np.arange(3.0), d=d, tau=tau)

    def test_permutation_entropy_parity(self):
        """Full permutation entropy should agree regardless of backend."""
        series = logistic_series(n=5000)
        from dynachaos.diagnostics import permutation as perm_mod

        old_flag = perm_mod._RUST_AVAILABLE
        try:
            perm_mod._RUST_AVAILABLE = False
            h_python = perm_mod.permutation_entropy(series, d=5)
        finally:
            perm_mod._RUST_AVAILABLE = old_flag

        _assert_golden("permutation_entropy", [h_python], atol=1e-10, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            h_rust = perm_mod.permutation_entropy(series, d=5)
            _assert_golden("permutation_entropy", [h_rust], atol=1e-10, rtol=0.0)

    def test_complexity_entropy_parity(self):
        """Complexity-entropy plane should agree regardless of backend."""
        series = logistic_series(n=5000)
        from dynachaos.diagnostics import permutation as perm_mod

        old_flag = perm_mod._RUST_AVAILABLE
        try:
            perm_mod._RUST_AVAILABLE = False
            h_python, c_python = perm_mod.complexity_entropy(series, d=5)
        finally:
            perm_mod._RUST_AVAILABLE = old_flag

        _assert_golden("complexity_entropy", [h_python, c_python], atol=1e-10, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            h_rust, c_rust = perm_mod.complexity_entropy(series, d=5)
            _assert_golden("complexity_entropy", [h_rust, c_rust], atol=1e-10, rtol=0.0)


class TestAMIParity:
    """Verify Rust and Python AMI agree."""

    def test_ami_parity(self):
        series = logistic_series(n=3000)
        from dynachaos.diagnostics import embedding as emb_mod

        I_python = emb_mod._ami_python(series, tau_max=30, n_bins=32)

        _assert_golden("ami", I_python, atol=1e-10, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            old_flag = emb_mod._RUST_AVAILABLE
            try:
                emb_mod._RUST_AVAILABLE = True
                _, I_rust = emb_mod.average_mutual_information(series, tau_max=30, n_bins=32)
            finally:
                emb_mod._RUST_AVAILABLE = old_flag
            _assert_golden("ami", I_rust, atol=1e-10, rtol=0.0)

    @rust_extension
    def test_ami_direct_rust_rejects_invalid_inputs(self):
        from dynachaos._rust import ami_histogram

        with pytest.raises(ValueError, match="at least two"):
            ami_histogram(np.array([1.0]), tau_max=3, n_bins=4)
        with pytest.raises(ValueError, match="finite values"):
            ami_histogram(np.array([1.0, np.nan, 2.0]), tau_max=3, n_bins=4)

    @rust_extension
    def test_ami_direct_rust_constant_series_returns_zero(self):
        from dynachaos._rust import ami_histogram

        mi = ami_histogram(np.ones(10), tau_max=5, n_bins=4)

        np.testing.assert_array_equal(mi, np.zeros(5))


# TestCaoParity omitted: Cao statistics remain in Python/SciPy.


@rust_extension
def test_disabled_embedding_statistics_are_not_exported():
    import dynachaos._rust as rust_mod

    assert not hasattr(rust_mod, "cao_statistic")
    assert not hasattr(rust_mod, "fnn_statistic")
    assert hasattr(rust_mod, "select_dimension_cao")


class TestCaoSelectorParity:
    """Verify Rust and Python Cao selector agree."""

    @pytest.mark.parametrize(
        ("e1", "kwargs", "expected"),
        [
            (
                np.array([0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000]),
                dict(
                    near_one_lower=0.97,
                    near_one_upper=1.03,
                    saturation_tol=0.02,
                    plateau_span=3,
                    smoothing_window=1,
                    min_dim=2,
                    max_dim=15,
                ),
                4,
            ),
            (
                np.array([0.08, 0.20, 0.35, 0.70, 0.96, 0.93]),
                dict(
                    near_one_lower=0.95,
                    near_one_upper=1.05,
                    saturation_tol=0.02,
                    plateau_span=4,
                    smoothing_window=1,
                    min_dim=2,
                    max_dim=15,
                ),
                5,
            ),
            (
                np.array([0.10, 0.20, 0.30, 0.40]),
                dict(
                    near_one_lower=0.95,
                    near_one_upper=1.05,
                    saturation_tol=0.02,
                    plateau_span=3,
                    smoothing_window=1,
                    min_dim=2,
                    max_dim=15,
                ),
                4,
            ),
            (
                np.array([np.nan, np.inf, np.nan]),
                dict(
                    near_one_lower=0.95,
                    near_one_upper=1.05,
                    saturation_tol=0.02,
                    plateau_span=3,
                    smoothing_window=1,
                    min_dim=3,
                    max_dim=15,
                ),
                3,
            ),
            # NaN mid-plateau: window must be rejected identically by both paths
            (
                np.array([0.50, 0.98, np.nan, 0.99, 1.00, 1.00, 1.00]),
                dict(
                    near_one_lower=0.95,
                    near_one_upper=1.05,
                    saturation_tol=0.02,
                    plateau_span=3,
                    smoothing_window=1,
                    min_dim=2,
                    max_dim=15,
                ),
                4,
            ),
        ],
    )
    def test_select_dimension_cao_parity(self, e1, kwargs, expected):
        from dynachaos.diagnostics import embedding as emb_mod

        e1 = np.asarray(e1, dtype=np.float64)

        old_selector = emb_mod._select_dimension_cao_rs
        emb_mod._select_dimension_cao_rs = None
        try:
            d_python = int(emb_mod.select_dimension_cao(e1, **kwargs))
        finally:
            emb_mod._select_dimension_cao_rs = old_selector

        assert d_python == expected

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import select_dimension_cao as rust_selector

            assert int(rust_selector(e1, **kwargs)) == expected


# TestFNNParity omitted: FNN statistics remain in Python/SciPy.


class TestCorrelationCountsParity:
    """Verify Rust and Python correlation integral agree."""

    def test_correlation_counts_parity(self):
        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        r_values = np.logspace(-2, 0, 15)

        from dynachaos.diagnostics import correlation as corr_mod

        old_flag = corr_mod._RUST_AVAILABLE
        try:
            corr_mod._RUST_AVAILABLE = False
            C_py = corr_mod.correlation_integral(traj, r_values, theiler_window=5, norm="chebyshev")
        finally:
            corr_mod._RUST_AVAILABLE = old_flag

        _assert_golden("correlation_counts", C_py, atol=1e-10, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                corr_mod._RUST_AVAILABLE = True
                C_rust = corr_mod.correlation_integral(
                    traj, r_values, theiler_window=5, norm="chebyshev"
                )
            finally:
                corr_mod._RUST_AVAILABLE = old_flag
            _assert_golden("correlation_counts", C_rust, atol=1e-10, rtol=0.0)

    @rust_extension
    def test_correlation_counts_huge_theiler_window_has_no_pairs(self):
        from dynachaos._rust import correlation_counts

        traj = np.arange(12.0).reshape(6, 2)
        r_values = np.array([0.1, 10.0], dtype=np.float64)

        counts = np.asarray(correlation_counts(traj, r_values, sys.maxsize, True))

        np.testing.assert_array_equal(counts, np.zeros_like(r_values, dtype=np.int64))


@rust_extension
class TestEntropyRustBoundaries:
    def test_apen_counts_uses_inclusive_self_matches(self):
        from dynachaos._rust import apen_counts

        traj = np.array(
            [
                [0.0, 0.1],
                [0.1, 0.2],
                [0.2, 0.4],
                [0.4, 0.45],
            ],
            dtype=np.float64,
        )

        counts = np.asarray(apen_counts(traj, 0.2))

        np.testing.assert_array_equal(counts, np.array([2, 3, 3, 2], dtype=np.int64))

    def test_apen_counts_rejects_invalid_r(self):
        from dynachaos._rust import apen_counts

        with pytest.raises(ValueError, match="r must be positive"):
            apen_counts(np.arange(6.0).reshape(3, 2), 0.0)

    def test_fuzzy_entropy_sum_huge_theiler_window_has_no_pairs(self):
        from dynachaos._rust import fuzzy_entropy_sum

        traj = np.arange(12.0).reshape(6, 2)

        assert fuzzy_entropy_sum(traj, 1.0, 2, sys.maxsize) == 0.0


class TestCMLJacobianParity:
    @pytest.mark.parametrize("L", [1, 3, 5])
    def test_cml_jacobian_logistic_direct_rust_matches_python_path(self, L):
        from dynachaos.cml import primitives as cml_mod

        x = np.array([0.125, -0.25, 0.375, -0.5, 0.625], dtype=np.float64)
        a = 1.73
        eps = 0.31

        old_flag = cml_mod._RUST_AVAILABLE
        cml_mod._RUST_AVAILABLE = False
        try:
            py_jacobian = cml_mod.cml_jacobian_subblock_logistic(x, a, eps, L)
        finally:
            cml_mod._RUST_AVAILABLE = old_flag

        _assert_golden(f"cml_jacobian_direct_{L}", py_jacobian, atol=0.0, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import cml_jacobian_logistic as rust_jacobian

            rust_jacobian_flat = np.asarray(rust_jacobian(x, a=a, eps=eps, L=L))
            rust_jacobian_matrix = rust_jacobian_flat.reshape((L, L))
            _assert_golden(f"cml_jacobian_direct_{L}", rust_jacobian_matrix, atol=0.0, rtol=0.0)

    def test_cml_jacobian_public_dispatcher_matches_python_path(self):
        from dynachaos.cml import primitives as cml_mod

        x = np.array([0.2, -0.1, 0.4, -0.3], dtype=np.float64)
        a = 1.91
        eps = 0.27
        L = len(x)

        old_flag = cml_mod._RUST_AVAILABLE
        try:
            cml_mod._RUST_AVAILABLE = False
            python_path = cml_mod.cml_jacobian_subblock_logistic(x, a, eps, L)
        finally:
            cml_mod._RUST_AVAILABLE = old_flag

        _assert_golden("cml_jacobian_public", python_path, atol=0.0, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                cml_mod._RUST_AVAILABLE = True
                rust_path = cml_mod.cml_jacobian_subblock_logistic(x, a, eps, L)
            finally:
                cml_mod._RUST_AVAILABLE = old_flag
            _assert_golden("cml_jacobian_public", rust_path, atol=0.0, rtol=0.0)

    @rust_extension
    def test_cml_jacobian_logistic_rejects_invalid_l(self):
        from dynachaos._rust import cml_jacobian_logistic as rust_jacobian

        x = np.array([0.1, -0.2, 0.3], dtype=np.float64)

        with pytest.raises(ValueError, match="L must satisfy"):
            rust_jacobian(x, 1.5, 0.2, 0)

        with pytest.raises(ValueError, match="L must satisfy"):
            rust_jacobian(x, 1.5, 0.2, len(x) + 1)

    def test_cml_lyapunov_density_inner_loop_parity(self):
        python_density = self._small_lyapunov_density(use_rust=False)

        _assert_golden("cml_lyapunov_density", python_density, atol=1e-14, rtol=1e-14)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            rust_density = self._small_lyapunov_density(use_rust=True)
            _assert_golden("cml_lyapunov_density", rust_density, atol=1e-14, rtol=1e-14)

    def _small_lyapunov_density(self, use_rust):
        from dynachaos.cml import primitives as cml_mod

        N = 6
        eps = 0.3
        a_values = np.array([1.55, 1.82])
        L_values = np.array([2, 4, 6])
        n_transient = 5
        n_iter = 10
        rng = np.random.default_rng(123)

        density = np.empty((len(a_values), len(L_values)))
        old_flag = cml_mod._RUST_AVAILABLE
        cml_mod._RUST_AVAILABLE = use_rust
        try:
            for ia, a in enumerate(a_values):
                x = rng.uniform(-0.5, 0.5, N)
                for _ in range(n_transient):
                    x = cml_mod.cml_step_logistic(x, a, eps)

                for iL, L in enumerate(L_values):
                    x_run = x.copy()
                    v = rng.standard_normal(L)
                    v /= np.linalg.norm(v)

                    log_sum = 0.0
                    for _ in range(n_iter):
                        jacobian = cml_mod.cml_jacobian_subblock_logistic(x_run, a, eps, L)
                        v = jacobian @ v
                        norm_v = np.linalg.norm(v)
                        if norm_v > 0:
                            log_sum += np.log(norm_v)
                            v /= norm_v
                        else:
                            log_sum += -100.0
                            v = rng.standard_normal(L)
                            v /= np.linalg.norm(v)
                        x_run = cml_mod.cml_step_logistic(x_run, a, eps)

                    density[ia, iL] = log_sum / n_iter / L
        finally:
            cml_mod._RUST_AVAILABLE = old_flag

        return density


class TestComovingLyapunovParity:
    """Verify specialized Rust and Python co-moving Lyapunov paths agree."""

    def test_comoving_lyapunov_logistic_kernel_parity(self):
        from dynachaos.diagnostics.comoving_lyapunov import _comoving_lyapunov_spectrum_python
        from dynachaos.maps.primitives import logistic, logistic_derivative

        a = 1.7
        eps = 0.3
        x_init = np.linspace(-0.75, 0.85, 12)
        v_values = np.array([-0.5, 0.0, 0.75], dtype=np.float64)
        n_iter = 37
        n_transient = 11

        python_result = _comoving_lyapunov_spectrum_python(
            lambda x: logistic(x, a),
            lambda x: logistic_derivative(x, a),
            lambda x: logistic(x, a),
            lambda x: logistic_derivative(x, a),
            eps,
            x_init,
            v_values,
            n_iter,
            n_transient,
        )
        _assert_golden("comoving_kernel", python_result, rtol=1e-12, atol=1e-12)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import comoving_lyapunov_logistic

            rust_result = np.asarray(
                comoving_lyapunov_logistic(
                    x_init,
                    v_values,
                    a,
                    eps,
                    n_iter,
                    n_transient,
                )
            )
            _assert_golden("comoving_kernel", rust_result, rtol=1e-12, atol=1e-12)

    def test_comoving_lyapunov_logistic_public_dispatch_parity(self):
        import importlib

        comoving_mod = importlib.import_module("dynachaos.diagnostics.comoving_lyapunov")

        kwargs = {
            "a": 1.85,
            "eps": 0.3,
            "N": 10,
            "v_values": np.array([-0.25, 0.25], dtype=np.float64),
            "n_iter": 25,
            "n_transient": 7,
            "x_init": np.linspace(-0.5, 0.6, 10),
        }
        old_flag = comoving_mod._RUST_AVAILABLE
        try:
            comoving_mod._RUST_AVAILABLE = False
            python_result = comoving_mod.comoving_lyapunov_spectrum_logistic(**kwargs)
        finally:
            comoving_mod._RUST_AVAILABLE = old_flag

        _assert_golden("comoving_public", python_result, rtol=1e-12, atol=1e-12)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                comoving_mod._RUST_AVAILABLE = True
                rust_result = comoving_mod.comoving_lyapunov_spectrum_logistic(**kwargs)
            finally:
                comoving_mod._RUST_AVAILABLE = old_flag
            _assert_golden("comoving_public", rust_result, rtol=1e-12, atol=1e-12)


class TestCoupledLogisticBasinsParity:
    """Verify Rust and Python coupled-logistic basin grids agree."""

    def test_coupled_logistic_basin_grid_parity(self):
        from dynachaos.maps.coupled_logistic import _basin_grid_python, _find_reference_orbit

        A = 1.35344
        D = 0.1
        n_grid = 5
        n_transient = 6
        reference_transient = 8
        period = 4

        ref_a = _find_reference_orbit(
            A,
            D,
            0.1,
            0.6,
            n_transient=reference_transient,
            period=period,
        )
        x_range = np.linspace(-1.0, 1.0, n_grid)
        y_range = np.linspace(-1.0, 1.0, n_grid)

        python_basin = _basin_grid_python(A, D, x_range, y_range, n_transient, ref_a)

        np.testing.assert_array_equal(python_basin, _golden("basin_grid"))

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import coupled_logistic_basin_grid

            rust_basin = np.asarray(
                coupled_logistic_basin_grid(
                    x_range,
                    y_range,
                    A,
                    D,
                    n_transient,
                    ref_a,
                )
            )
            np.testing.assert_array_equal(rust_basin, _golden("basin_grid"))

    @rust_extension
    def test_coupled_logistic_basin_grid_divergence_and_tie_cases(self):
        from dynachaos._rust import coupled_logistic_basin_grid

        ref_a = np.array([[0.0, 0.0]], dtype=np.float64)

        diverged = np.asarray(
            coupled_logistic_basin_grid(
                np.array([1.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
                200.0,
                0.1,
                1,
                ref_a,
            )
        )
        np.testing.assert_array_equal(diverged, np.array([[-1]], dtype=np.int8))

        tied = np.asarray(
            coupled_logistic_basin_grid(
                np.array([0.0], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
                1.0,
                0.1,
                0,
                ref_a,
            )
        )
        np.testing.assert_array_equal(tied, np.array([[0]], dtype=np.int8))

    def test_compute_basins_parity(self, tmp_path):
        import importlib

        basin_mod = importlib.import_module("dynachaos.maps.coupled_logistic")

        kwargs = {
            "n_grid": 5,
            "n_transient": 6,
            "reference_transient": 8,
            "period": 4,
        }
        old_flag = basin_mod._RUST_AVAILABLE
        try:
            basin_mod._RUST_AVAILABLE = False
            python_payload = basin_mod.compute_basins(
                **kwargs,
                output_path=tmp_path / "basins_python.npz",
            )
        finally:
            basin_mod._RUST_AVAILABLE = old_flag

        expected_keys = {"x", "y", "basin", "A", "D"}
        assert set(python_payload) == expected_keys
        for key in python_payload:
            expected = _golden(f"compute_basins_{key}")
            if key == "basin":
                np.testing.assert_array_equal(python_payload[key], expected)
            else:
                np.testing.assert_allclose(python_payload[key], expected)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                basin_mod._RUST_AVAILABLE = True
                rust_payload = basin_mod.compute_basins(
                    **kwargs,
                    output_path=tmp_path / "basins_rust.npz",
                )
            finally:
                basin_mod._RUST_AVAILABLE = old_flag

            assert set(rust_payload) == expected_keys
            for key in rust_payload:
                expected = _golden(f"compute_basins_{key}")
                if key == "basin":
                    np.testing.assert_array_equal(rust_payload[key], expected)
                else:
                    np.testing.assert_allclose(rust_payload[key], expected)


class TestIntermittencyOracleParity:
    """Verify Rust intermittency oracle kernels match Python fallbacks."""

    def test_direct_pm_type_i_parity(self):
        from dynachaos.maps.intermittency import _pm_type_i_oracle_python

        kwargs = dict(n=80, x0=0.012, eps=2e-4, a=0.8, modulo=True)
        python_result = _pm_type_i_oracle_python(**kwargs)

        _assert_golden("pm_type_i_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import pm_type_i_oracle

            _assert_golden("pm_type_i_direct", pm_type_i_oracle(**kwargs))

    def test_direct_pm_type_ii_parity(self):
        from dynachaos.maps.intermittency import _pm_type_ii_oracle_python

        kwargs = dict(n=80, x0=1e-3, y0=2e-3, eps=1e-3, a=-1.0, theta=0.37)
        python_result = _pm_type_ii_oracle_python(**kwargs)

        _assert_golden("pm_type_ii_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import pm_type_ii_oracle

            _assert_golden("pm_type_ii_direct", pm_type_ii_oracle(**kwargs))

    def test_direct_pm_type_iii_parity(self):
        from dynachaos.maps.intermittency import _pm_type_iii_oracle_python

        kwargs = dict(n=80, x0=1e-3, eps=1e-3, a=1.0)
        python_result = _pm_type_iii_oracle_python(**kwargs)

        _assert_golden("pm_type_iii_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import pm_type_iii_oracle

            _assert_golden("pm_type_iii_direct", pm_type_iii_oracle(**kwargs))

    def test_direct_on_off_parity(self):
        from dynachaos.maps.intermittency import _on_off_oracle_python

        driver = np.random.default_rng(2026).normal(size=80)
        python_result = _on_off_oracle_python(driver, 1e-6, 0.0, 0.25)

        _assert_golden("on_off_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import on_off_oracle

            _assert_golden("on_off_direct", on_off_oracle(driver, 1e-6, 0.0, 0.25))

    def test_direct_on_off_skew_logistic_parity(self):
        from dynachaos.maps.intermittency import _on_off_skew_logistic_oracle_python

        kwargs = dict(n=80, x0=0.217, y0=1e-2, eps=0.499)
        python_result = _on_off_skew_logistic_oracle_python(**kwargs)

        _assert_golden("on_off_skew_logistic_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import on_off_skew_logistic_oracle

            _assert_golden("on_off_skew_logistic_direct", on_off_skew_logistic_oracle(**kwargs))

    def test_on_off_skew_logistic_public_dispatch_and_bursting(self):
        from dynachaos.maps import ON_OFF_SKEW_LOGISTIC_ONSET, on_off_skew_logistic_oracle

        np.testing.assert_allclose(ON_OFF_SKEW_LOGISTIC_ONSET, 0.5)

        out = on_off_skew_logistic_oracle(n=4_000, x0=0.217, y0=1e-2, eps=0.499)

        np.testing.assert_equal(out.shape, (4_000, 2))
        np.testing.assert_allclose(out[:, 0], np.clip(out[:, 0], 0.0, 1.0))
        y_abs = np.abs(out[:, 1])
        median = np.quantile(y_abs, 0.50)
        q95 = np.quantile(y_abs, 0.95)
        np.testing.assert_array_less(10.0 * median, q95)
        np.testing.assert_array_less(1e-3, np.max(y_abs))

    def test_direct_logistic_type_i_parity(self):
        from dynachaos.maps.intermittency import (
            LOGISTIC_TYPE_I_ONSET,
            _logistic_type_i_oracle_python,
        )

        kwargs = dict(n=80, x0=0.2, r=LOGISTIC_TYPE_I_ONSET - 1e-4)
        python_result = _logistic_type_i_oracle_python(**kwargs)

        _assert_golden("logistic_type_i_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import logistic_type_i_oracle

            _assert_golden("logistic_type_i_direct", logistic_type_i_oracle(**kwargs))

    def test_public_dispatch_parity(self):
        from dynachaos.maps import intermittency as int_mod

        calls = [
            ("dispatch_pm_type_i", int_mod.pm_type_i_oracle, dict(n=50, x0=0.01, eps=1e-4, a=1.0)),
            (
                "dispatch_pm_type_ii",
                int_mod.pm_type_ii_oracle,
                dict(n=50, x0=1e-3, y0=2e-3, eps=1e-3, a=-1.0),
            ),
            (
                "dispatch_pm_type_iii",
                int_mod.pm_type_iii_oracle,
                dict(n=50, x0=1e-3, eps=1e-3, a=1.0),
            ),
            ("dispatch_logistic_type_i", int_mod.logistic_type_i_oracle, dict(n=50, x0=0.2)),
            ("dispatch_on_off", int_mod.on_off_oracle, dict(n=50, seed=2027)),
            (
                "dispatch_on_off_skew_logistic",
                int_mod.on_off_skew_logistic_oracle,
                dict(n=50, x0=0.217, y0=1e-2, eps=0.499),
            ),
        ]

        old_flag = int_mod._RUST_AVAILABLE
        try:
            for golden_name, func, kwargs in calls:
                int_mod._RUST_AVAILABLE = False
                python_result = func(**kwargs)
                _assert_golden(golden_name, python_result)

                if _RUST_IMPORTABLE and not _NO_RUST_ENV:
                    int_mod._RUST_AVAILABLE = True
                    rust_result = func(**kwargs)
                    _assert_golden(golden_name, rust_result)
        finally:
            int_mod._RUST_AVAILABLE = old_flag


class TestDiscreteMap:
    """Test the new DiscreteMap convenience class."""

    def test_logistic_trajectory(self):
        from dynachaos.maps.base import LogisticMap

        lm = LogisticMap(a=1.99)
        traj = lm.trajectory(x0=0.1, n_iter=100, n_transient=50)
        assert traj.shape == (100,)
        assert np.all(np.isfinite(traj))

    def test_logistic_lyapunov(self):
        from dynachaos.maps.base import LogisticMap

        lm = LogisticMap(a=1.99)
        lam = lm.lyapunov(x0=0.1, n_iter=50_000, n_transient=5_000)
        assert lam > 0.5  # chaotic

    def test_no_jacobian_raises(self):
        from dynachaos.maps.base import DiscreteMap

        m = DiscreteMap(f=lambda x: 1 - 1.99 * x * x, name="bare")
        with pytest.raises(ValueError, match="No derivative"):
            m.lyapunov(x0=0.1)

    def test_repr(self):
        from dynachaos.maps.base import LogisticMap

        lm = LogisticMap(a=1.5)
        assert "Logistic" in repr(lm)

    def test_henon_lyapunov_uses_nd_spectrum_branch(self):
        # dim=2 routes through lyapunov_spectrum instead of
        # lyapunov_exponent_1d; the largest exponent of the canonical Henon
        # map (a=1.4, b=0.3) is the textbook ~0.42.
        from dynachaos.maps.base import HenonMap

        hm = HenonMap(a=1.4, b=0.3)
        spectrum = hm.lyapunov(x0=np.array([0.1, 0.1]), n_iter=50_000, n_transient=5_000)
        assert spectrum.shape == (2,)
        assert spectrum[0] == pytest.approx(0.42, abs=0.05)

    def test_bifurcation_returns_param_values_and_attractors(self):
        from dynachaos.maps.base import BifurcationData, LogisticMap

        lm = LogisticMap(a=1.5)
        params = np.array([1.5, 1.99])
        data = lm.bifurcation(
            params, lambda a: LogisticMap(a=a).f, x0=0.1, n_transient=100, n_record=20
        )
        assert isinstance(data, BifurcationData)
        np.testing.assert_array_equal(data.param_values, params)
        assert len(data.attractors) == 2
        for pts in data.attractors:
            assert pts.shape == (20,)
            assert np.all(np.isfinite(pts))

    def test_bifurcation_extracts_first_component_for_nd_maps(self):
        from dynachaos.maps.base import HenonMap

        hm = HenonMap(a=1.4, b=0.3)
        params = np.array([1.4])
        data = hm.bifurcation(
            params,
            lambda a: HenonMap(a=a, b=0.3).f,
            x0=np.array([0.1, 0.1]),
            n_transient=50,
            n_record=10,
        )
        assert len(data.attractors) == 1
        assert data.attractors[0].shape == (10,)

    def test_circle_map_constructor(self):
        from dynachaos.maps.base import CircleMap

        cm = CircleMap(omega=0.2, K=0.5)
        traj = cm.trajectory(x0=0.1, n_iter=20)
        assert traj.shape == (20,)
        assert "Circle" in repr(cm)

    def test_standard_map_constructor(self):
        from dynachaos.maps.base import StandardMap

        sm = StandardMap(K=1.0)
        traj = sm.trajectory(x0=np.array([0.1, 0.1]), n_iter=20)
        assert traj.shape == (20, 2)
        assert "StandardMap" in repr(sm)


class TestViz:
    """Smoke tests for viz subpackage (non-interactive)."""

    def test_bifurcation_import(self):
        from dynachaos.viz import bifurcation_diagram

        assert callable(bifurcation_diagram)

    def test_cobweb_import(self):
        from dynachaos.viz import cobweb_diagram

        assert callable(cobweb_diagram)

    def test_return_map_import(self):
        from dynachaos.viz import return_map_plot

        assert callable(return_map_plot)


class TestVersion:
    def test_version_matches_package_metadata(self):
        """__version__ must not drift from the version declared in pyproject.toml."""
        from importlib.metadata import version

        import dynachaos

        assert dynachaos.__version__ == version("dynachaos")


@rust_extension
class TestCircleMapTileParity:
    """The Rust rotation-number tile must match the NumPy reference sweep.

    The reference is the loop `dynachaos.maps.arnold_tongues.compute` runs:
    theta advances by ``Omega + K sin(2 pi theta)`` without reduction modulo 1,
    and the rotation number is the mean drift over the second stretch.
    """

    N_OMEGA = 7
    N_K = 5
    N_TRANSIENT = 200
    N_ITER = 500
    THETA0 = 0.1
    # Closure search: periods up to Q_MAX, over the first SEARCH_STEPS angles.
    Q_MAX = 32
    SEARCH_STEPS = 96
    CLOSURE_EPS = 1e-12
    K_CRITICAL = 1.0 / (2.0 * np.pi)

    def _numpy_reference(self):
        two_pi = 2.0 * np.pi
        omega_values = np.linspace(0.0, 1.0, self.N_OMEGA)
        k_values = np.linspace(0.0, 0.3, self.N_K)
        omega_grid, k_grid = np.meshgrid(omega_values, k_values)
        omega_flat = omega_grid.ravel()
        k_flat = k_grid.ravel()

        theta = np.full(omega_flat.size, self.THETA0)
        for _ in range(self.N_TRANSIENT):
            theta += omega_flat + k_flat * np.sin(two_pi * theta)
        theta_start = theta.copy()
        for _ in range(self.N_ITER):
            theta += omega_flat + k_flat * np.sin(two_pi * theta)
        return ((theta - theta_start) / self.N_ITER).reshape(self.N_K, self.N_OMEGA)

    def _rust_tile(self):
        from dynachaos._rust import rotation_number_tile

        return rotation_number_tile(
            0.0,
            1.0,
            self.N_OMEGA,
            0.0,
            0.3,
            self.N_K,
            self.N_TRANSIENT,
            self.N_ITER,
            self.THETA0,
        )

    def test_rust_tile_matches_numpy_reference(self):
        rust_tile = self._rust_tile()
        reference = self._numpy_reference()

        assert rust_tile.shape == (self.N_K, self.N_OMEGA)
        # The Rust kernel stops early once the orbit locks or the running
        # estimate is within half a colour step (1/256). NumPy here always runs
        # the full stretch, so this bound covers the cells that stopped on the
        # estimate. It is deliberately loose, and on its own it would hide an
        # error of up to one colour step inside a locked tongue, where the two
        # must agree exactly. test_locked_cells_return_the_exact_rational is
        # what holds that population; do not delete one and keep the other.
        display_tol = 1.0 / 256.0
        np.testing.assert_allclose(rust_tile, reference, rtol=0.0, atol=display_tol)

    def _locked_rationals(self):
        """Per cell, the (p, q) an orbit closes on, or None if it does not close.

        The orbit is advanced exactly as _numpy_reference advances it. That
        matters: `theta += omega + k sin(...)` sums the increment before adding,
        and a different association changes the last bit. Above K_c that one bit
        grows into a different trajectory.
        """
        two_pi = 2.0 * np.pi
        omega_values = np.linspace(0.0, 1.0, self.N_OMEGA)
        k_values = np.linspace(0.0, 0.3, self.N_K)
        omega_grid, k_grid = np.meshgrid(omega_values, k_values)
        omega_flat = omega_grid.ravel()
        k_flat = k_grid.ravel()

        theta = np.full(omega_flat.size, self.THETA0)
        for _ in range(self.N_TRANSIENT):
            theta += omega_flat + k_flat * np.sin(two_pi * theta)
        history = []
        for _ in range(self.SEARCH_STEPS):
            history.append(theta.copy())
            theta += omega_flat + k_flat * np.sin(two_pi * theta)
        angles = np.asarray(history)

        rationals = [None] * omega_flat.size
        for cell in range(omega_flat.size):
            column = angles[:, cell]
            for start in range(self.SEARCH_STEPS - self.Q_MAX):
                for q in range(1, self.Q_MAX + 1):
                    drift = column[start + q] - column[start]
                    if abs(drift - round(drift)) < self.CLOSURE_EPS:
                        rationals[cell] = (int(round(drift)), q)
                        break
                if rationals[cell] is not None:
                    break
        return k_flat, rationals

    def test_locked_cells_return_the_exact_rational(self):
        """A closed orbit has an exactly rational rotation number, and the kernel returns it.

        Below K_c the map is a diffeomorphism, so the closure found here is the
        closure the kernel finds and p / q is exact: the two must agree to
        machine precision, not to the displayed colour. Above K_c the map is not
        invertible and a last-bit difference changes the trajectory, so a closure
        detected here need not be the one the kernel took; those cells are left
        to the display bound in the test above.
        """
        rust_tile = self._rust_tile().ravel()
        k_flat, rationals = self._locked_rationals()

        checked = 0
        for cell, rational in enumerate(rationals):
            if rational is None or k_flat[cell] > self.K_CRITICAL:
                continue
            p, q = rational
            assert abs(rust_tile[cell] - p / q) <= 1e-12, (
                f"cell {cell} (K = {k_flat[cell]:.4f}) closes on {p}/{q}, "
                f"but the kernel returned {rust_tile[cell]!r}"
            )
            checked += 1
        assert checked >= 5, (
            f"only {checked} locked cells below K_c were checked; "
            "this assertion would prove almost nothing"
        )

    def test_zero_coupling_row_returns_omega(self):
        """The K = 0 row is a known analytic case: the drift per step is Omega."""
        rust_tile = self._rust_tile()

        np.testing.assert_allclose(
            rust_tile[0], np.linspace(0.0, 1.0, self.N_OMEGA), rtol=0.0, atol=1e-12
        )

    def test_invalid_grid_shape_is_rejected(self):
        from dynachaos._rust import rotation_number_tile

        with pytest.raises(ValueError, match="n_omega and n_k must be >= 1"):
            rotation_number_tile(0.0, 1.0, 0, 0.0, 0.3, 4, 10, 10, 0.1)


@rust_extension
class TestZeroOneParity:
    """Verify the Rust 0-1 kernel matches the Python estimator per c value.

    The kernel takes the c values as input, so both backends see the same
    frequencies and can be compared element by element. The bound is 1e-9:
    both backends form the autocovariance by FFT, so the two differ only by
    rounding, not by method.
    """

    N_C = 15
    N_CUT = 200
    ATOL = 1e-9

    def _series(self):
        n = 2000
        t = np.arange(n, dtype=np.float64)
        regular = np.sin(2.0 * np.pi * 0.071 * t) + 0.2 * np.sin(2.0 * np.pi * 0.113 * t)
        chaotic = logistic_series(n=n, a=1.99, burn=2000)
        return {"regular": regular, "chaotic": chaotic}

    def _c_values(self):
        return np.random.default_rng(2026).uniform(np.pi / 5, 4 * np.pi / 5, self.N_C)

    def test_per_c_k_matches_python(self):
        from dynachaos._rust import zero_one_k
        from dynachaos.diagnostics import zero_one_test as zo_mod

        c_values = self._c_values()
        for name, phi in self._series().items():
            k_python = zo_mod._k_per_c_python(phi, c_values, self.N_CUT)
            k_rust = np.asarray(zero_one_k(phi, c_values, self.N_CUT))

            worst = float(np.max(np.abs(k_rust - k_python)))
            np.testing.assert_allclose(k_rust, k_python, rtol=0.0, atol=self.ATOL)
            assert worst <= self.ATOL, f"{name}: worst per-c difference {worst:.3e}"

    def test_per_c_k_matches_python_at_long_n(self):
        """Same per-c bound at N = 30001, where the FFT path must not drift."""
        from dynachaos._rust import zero_one_k
        from dynachaos.diagnostics import zero_one_test as zo_mod

        phi = logistic_series(n=30_001, a=1.99, burn=2000)
        c_values = np.random.default_rng(2026).uniform(np.pi / 5, 4 * np.pi / 5, 5)
        n_cut = 3000

        k_python = zo_mod._k_per_c_python(phi, c_values, n_cut)
        k_rust = np.asarray(zero_one_k(phi, c_values, n_cut))

        worst = float(np.max(np.abs(k_rust - k_python)))
        np.testing.assert_allclose(k_rust, k_python, rtol=0.0, atol=self.ATOL)
        assert worst <= self.ATOL, f"N = 30001: worst per-c difference {worst:.3e}"

    def test_median_matches_python(self):
        from dynachaos._rust import zero_one_k
        from dynachaos.diagnostics import zero_one_test as zo_mod

        c_values = self._c_values()
        for name, phi in self._series().items():
            k_python = zo_mod._k_per_c_python(phi, c_values, self.N_CUT)
            k_rust = np.asarray(zero_one_k(phi, c_values, self.N_CUT))

            med_python = float(np.median(k_python))
            med_rust = float(np.median(k_rust))
            assert abs(med_rust - med_python) <= self.ATOL, (
                f"{name}: medians differ by {abs(med_rust - med_python):.3e}"
            )

    def test_public_dispatch_uses_rust_when_available(self, monkeypatch):
        from dynachaos.diagnostics import zero_one_test as zo_mod

        phi = self._series()["chaotic"]
        c_values = self._c_values()
        sentinel = np.full(self.N_C, -7.0)

        calls = []

        def stub(phi_arg, c_arg, n_cut_arg):
            calls.append((phi_arg, c_arg, n_cut_arg))
            return sentinel

        monkeypatch.setattr(zo_mod, "_zero_one_k_rs", stub)
        old_flag = zo_mod._RUST_AVAILABLE
        try:
            zo_mod._RUST_AVAILABLE = True
            dispatched = zo_mod._k_per_c(phi, c_values, self.N_CUT)
            # The public functions must reach the kernel too, not only the helper.
            statistic = zo_mod.zero_one_statistic(phi, n_c=self.N_C, n_cut=self.N_CUT)
            _, series_k = zo_mod.zero_one_series(phi, n_c=self.N_C, n_cut=self.N_CUT)
        finally:
            zo_mod._RUST_AVAILABLE = old_flag

        assert len(calls) == 3, "the Rust kernel was not called on every path"
        np.testing.assert_array_equal(dispatched, sentinel)
        assert statistic == -7.0
        np.testing.assert_array_equal(series_k, sentinel)

    def test_direct_rust_rejects_invalid_inputs(self):
        from dynachaos._rust import zero_one_k

        phi = np.array([0.1, 0.2, 0.3, 0.4])
        c = np.array([0.5, 0.7])
        with pytest.raises(ValueError, match="finite values"):
            zero_one_k(np.array([0.1, np.nan, 0.3]), c, 2)
        with pytest.raises(ValueError, match="finite values"):
            zero_one_k(phi, np.array([np.nan]), 2)
        with pytest.raises(ValueError, match="at least one"):
            zero_one_k(phi, np.array([]), 2)
        with pytest.raises(ValueError, match="n_cut"):
            zero_one_k(phi, c, 1)
        with pytest.raises(ValueError, match="n_cut"):
            zero_one_k(phi, c, 5)

    def test_direct_rust_zero_series_returns_zero(self):
        from dynachaos._rust import zero_one_k

        ks = zero_one_k(np.zeros(100), self._c_values()[:5], 10)

        np.testing.assert_array_equal(ks, np.zeros(5))


@rust_extension
class TestZeroOneScaling:
    """The Rust 0-1 kernel must not scale as N^2.

    The kernel it replaced summed the autocovariance directly: n_c x n_cut x N
    multiply-adds, which at the default n_cut = N/10 grows as N^2. From N to
    32N that sum takes 1024x longer. An N log N kernel takes
    32 * log(32N) / log(N) = 46.5x at these sizes (N = 2048, 32N = 65536;
    log2(65536) / log2(2048) = 16/11). The bound of 200 is 4.3x above 46.5
    and 5.1x below 1024, clear of both by more than 3x.

    Timing uses time.process_time, not wall-clock. The old N-to-4N wall-clock
    bound of 8 (expected ~4.5x) failed on CI at ratio 8.68 when other workers
    stole time from the longer burst. process_time is this process's CPU time,
    all threads included, so preemption does not inflate one size more than
    the other. The wider step is what separates the two asymptotes; a higher
    bound on the old pair would not.
    """

    N_C = 8
    N_SMALL = 2_048
    N_LARGE = 65_536
    ROUNDS = 5

    def test_rust_kernel_scales_below_n_squared(self):
        from dynachaos._rust import zero_one_k

        c_values = np.random.default_rng(7).uniform(np.pi / 5, 4 * np.pi / 5, self.N_C)
        cases = {n: logistic_series(n=n, a=1.99, burn=2000) for n in (self.N_SMALL, self.N_LARGE)}
        best = dict.fromkeys(cases, float("inf"))

        for n, phi in cases.items():
            zero_one_k(phi, c_values, n // 10)  # warm up: first call pays pool spin-up
        # Windows updates process CPU time on the scheduler tick (~15.6 ms).
        # A burst that finishes between ticks reads as 0, and the minimum
        # over rounds then divides by zero. Span several ticks of wall time
        # (perf_counter moves between ticks; process_time may not) and
        # divide by the call count. Interleave the sizes so a busy interval
        # slows both, not one; keep the fastest per-call sample of each.
        tick_span_s = 0.10
        for _ in range(self.ROUNDS):
            for n, phi in cases.items():
                count = 0
                cpu_start = time.process_time()
                wall_start = time.perf_counter()
                while True:
                    zero_one_k(phi, c_values, n // 10)
                    count += 1
                    cpu = time.process_time() - cpu_start
                    if time.perf_counter() - wall_start >= tick_span_s and cpu > 0.0:
                        best[n] = min(best[n], cpu / count)
                        break

        ratio = best[self.N_LARGE] / best[self.N_SMALL]
        assert ratio < 200.0, (
            f"zero_one_k took {best[self.N_SMALL]:.4f}s at N = {self.N_SMALL} and "
            f"{best[self.N_LARGE]:.4f}s at N = {self.N_LARGE}: ratio {ratio:.1f} "
            "approaches the 1024x of an N^2 kernel"
        )


@rust_extension
class TestMapAttractorParity:
    """Verify delayed-logistic and torus trajectories against Python."""

    N_TRANSIENT = 2000
    N_PLOT = 4096

    def test_delayed_logistic_trajectory_parity(self):
        from dynachaos._rust import delayed_logistic_attractor_tile
        from dynachaos.maps._iter import trajectory_after_transient
        from dynachaos.maps.delayed_logistic import delayed_logistic

        state0 = np.array([0.4, 0.35], dtype=np.float64)
        d_values = np.array([1.55, 1.95, 2.16], dtype=np.float64)
        rust = delayed_logistic_attractor_tile(0.3, d_values, self.N_TRANSIENT, self.N_PLOT, state0)

        expected = []
        for d in d_values:
            trajectory = trajectory_after_transient(
                state0,
                lambda state, d=d: delayed_logistic(state, 0.3, float(d)),
                self.N_TRANSIENT,
                self.N_PLOT,
                diverged_fn=lambda state: np.any(np.abs(state) > 1e10),
            )
            assert trajectory is not None
            expected.append(trajectory)
        np.testing.assert_array_equal(rust, np.stack(expected))

    @pytest.mark.parametrize(
        ("map_kind", "A", "D", "state0", "map_function"),
        [
            (1, 0.4, 2.19, [0.5, 0.5, 0.5], "map_I"),
            (4, 0.3, 1.5212, [0.5, 0.45, 0.52, 0.48], "map_IV"),
        ],
    )
    def test_torus_doubling_trajectory_parity(self, map_kind, A, D, state0, map_function):
        from dynachaos._rust import torus_doubling_attractor_tile
        from dynachaos.maps import torus_doubling

        initial = np.array(state0, dtype=np.float64)
        python_trajectory = torus_doubling.iterate_map(
            getattr(torus_doubling, map_function),
            initial,
            A,
            D,
            n_transient=self.N_TRANSIENT,
            n_plot=self.N_PLOT,
        )
        assert python_trajectory is not None
        rust = torus_doubling_attractor_tile(
            map_kind,
            A,
            D,
            self.N_TRANSIENT,
            self.N_PLOT,
            initial,
        )
        np.testing.assert_array_equal(rust, python_trajectory)

    def test_torus_doubling_record_divergence_is_partial(self):
        from dynachaos._rust import torus_doubling_attractor_tile
        from dynachaos.maps import torus_doubling

        initial = np.array([0.5, 0.45, 0.52, 0.48], dtype=np.float64)
        python_trajectory = torus_doubling.iterate_map(
            torus_doubling.map_IV,
            initial,
            0.3,
            2.12,
            n_transient=self.N_TRANSIENT,
            n_plot=self.N_PLOT,
        )
        assert python_trajectory is not None
        assert len(python_trajectory) < self.N_PLOT
        rust = torus_doubling_attractor_tile(
            4,
            0.3,
            2.12,
            self.N_TRANSIENT,
            self.N_PLOT,
            initial,
        )
        np.testing.assert_array_equal(rust, python_trajectory)


@rust_extension
class TestModulatedCircleParity:
    """Verify Rust modulated-circle rotation numbers against Python.

    The kernel accumulates the same unwrapped increments in the same
    operation order as ``rotation_numbers``, so the comparison is exact
    (``np.array_equal``): ``np.sin`` and the platform ``sin`` the Rust build
    calls agree bit for bit on this platform. ``rho_phi`` is the accumulated
    drift of ``phi += C``, not ``C`` itself, so it is compared exactly too.
    """

    A = 0.1
    C = (np.sqrt(5) - 1) / 2
    EPS = 0.05
    N_TRANSIENT = 3000
    N_ITER = 20_000

    def test_rotation_numbers_match_python_exactly(self):
        from dynachaos._rust import modulated_circle_rotation_tile
        from dynachaos.maps.modulated_circle import rotation_numbers

        d_values = np.linspace(0.0, 1.0, 50)
        state0 = np.array([0.1, 0.1], dtype=np.float64)
        rust = modulated_circle_rotation_tile(
            self.A,
            self.C,
            d_values,
            self.EPS,
            self.N_TRANSIENT,
            self.N_ITER,
            state0,
        )
        expected = np.array(
            [
                rotation_numbers(
                    self.A,
                    self.C,
                    float(d),
                    self.EPS,
                    n_transient=self.N_TRANSIENT,
                    n_iter=self.N_ITER,
                    state0=state0,
                )
                for d in d_values
            ]
        )
        np.testing.assert_array_equal(rust, expected)

    def test_invalid_input_is_rejected(self):
        from dynachaos._rust import modulated_circle_rotation_tile

        state0 = np.array([0.1, 0.1], dtype=np.float64)
        with pytest.raises(ValueError):
            modulated_circle_rotation_tile(self.A, self.C, np.array([]), self.EPS, 10, 10, state0)
        with pytest.raises(ValueError):
            modulated_circle_rotation_tile(self.A, self.C, np.array([0.3]), self.EPS, 10, 0, state0)
        with pytest.raises(ValueError):
            modulated_circle_rotation_tile(
                self.A, self.C, np.array([0.3]), self.EPS, 10, 10, np.array([0.1])
            )


@rust_extension
class TestCmlSpacetimeParity:
    """Verify Rust CML space-time fields against ``simulate_cml``.

    The kernel evaluates ``f(x_i) + eps / 2 * (g(x_{i+1}) + g(x_{i-1}) -
    2 g(x_i))`` in the same operation order as ``cml_step``, so the recorded
    field is compared exactly (``np.array_equal``). Model B's ``sin`` agrees
    bit for bit between NumPy and the platform ``sin`` the Rust build calls,
    so the exact comparison holds for all three models at the paper's
    settings.
    """

    N = 200
    N_TRANSIENT = 2000
    N_RECORD = 500

    @pytest.fixture()
    def x0(self):
        return np.random.default_rng(42).uniform(0, 1, self.N)

    @pytest.mark.parametrize("eps", [0.06, 0.07, 0.08])
    def test_model_a_matches_python_exactly(self, x0, eps):
        from dynachaos._rust import cml_spacetime_tile
        from dynachaos.cml.spatiotemporal import model_A_f, simulate_cml

        rust = cml_spacetime_tile(0, eps, self.N_TRANSIENT, self.N_RECORD, x0)
        expected = simulate_cml(
            model_A_f,
            model_A_f,
            eps,
            N=self.N,
            n_transient=self.N_TRANSIENT,
            n_record=self.N_RECORD,
            x0=x0,
        )
        np.testing.assert_array_equal(rust, expected)

    @pytest.mark.parametrize("eps", [0.02, 0.024, 0.03])
    def test_model_b_matches_python_exactly(self, x0, eps):
        from dynachaos._rust import cml_spacetime_tile
        from dynachaos.cml.spatiotemporal import model_B_f, model_B_g, simulate_cml

        rust = cml_spacetime_tile(1, eps, self.N_TRANSIENT, self.N_RECORD, x0)
        expected = simulate_cml(
            model_B_f,
            model_B_g,
            eps,
            N=self.N,
            n_transient=self.N_TRANSIENT,
            n_record=self.N_RECORD,
            x0=x0,
        )
        np.testing.assert_array_equal(rust, expected)

    @pytest.mark.parametrize("eps", [0.16, 0.20, 0.30])
    def test_model_c_matches_python_exactly(self, x0, eps):
        from dynachaos._rust import cml_spacetime_tile
        from dynachaos.cml.spatiotemporal import model_C_f, simulate_cml

        rust = cml_spacetime_tile(2, eps, self.N_TRANSIENT, self.N_RECORD, x0)
        expected = simulate_cml(
            model_C_f,
            model_C_f,
            eps,
            N=self.N,
            n_transient=self.N_TRANSIENT,
            n_record=self.N_RECORD,
            x0=x0,
        )
        np.testing.assert_array_equal(rust, expected)

    def test_model_b_wraps_negative_state_with_floored_modulo(self):
        """A negative site exercises the floored modulo in model B's map.

        Python ``x % 1.0`` is floored, so ``f(-0.9)`` returns the wrapped
        value 0.7675570504584948; Rust's truncating ``%`` would return the
        raw value -0.23244294954150524. The kernel uses ``rem_euclid`` and
        must match the Python field exactly.
        """
        from dynachaos._rust import cml_spacetime_tile
        from dynachaos.cml.spatiotemporal import model_B_f, model_B_g, simulate_cml

        x0 = np.array([-0.9, 0.2, 0.5, -0.3], dtype=np.float64)
        rust = cml_spacetime_tile(1, 0.024, 0, 8, x0)
        expected = simulate_cml(model_B_f, model_B_g, 0.024, N=4, n_transient=0, n_record=8, x0=x0)
        np.testing.assert_array_equal(rust, expected)

    def test_invalid_input_is_rejected(self, x0):
        from dynachaos._rust import cml_spacetime_tile

        with pytest.raises(ValueError):
            cml_spacetime_tile(3, 0.2, 10, 4, x0)
        with pytest.raises(ValueError):
            cml_spacetime_tile(0, 0.07, 10, 4, np.array([]))
        with pytest.raises(ValueError):
            cml_spacetime_tile(0, 0.07, 10, 0, x0)
        with pytest.raises(ValueError):
            cml_spacetime_tile(0, np.nan, 10, 4, x0)


@rust_extension
class TestCircleMapLyapunovParity:
    """Verify the Rust circle-map exponent against ``lyapunov_exponent``.

    The wrapped map and the derivative evaluate the same expression in the
    same operation order as ``lyapunov_exponent``; the only nondeterminism
    is the last bit of ``sin``/``cos`` between the Rust and numpy system
    libraries. The bound is absolute 1e-9, fixed by the dispatch. Measured
    worst absolute difference over the 25-value grid below: 8.1e-20.
    """

    A = 0.1
    D = 0.25
    N_TRANSIENT = 5000
    N_ITER = 50_000
    THETA0 = 0.1

    def test_exponent_matches_python(self):
        from dynachaos._rust import circle_map_lyapunov_sum
        from dynachaos.maps.circle_map import lyapunov_exponent

        # 25 A values across the staircase sweep, D fixed at the paper's 0.25.
        a_values = np.linspace(0.0, 0.25, 25)
        for a in a_values:
            rust = circle_map_lyapunov_sum(
                float(a), self.D, self.N_TRANSIENT, self.N_ITER, self.THETA0
            )
            expected = lyapunov_exponent(
                float(a),
                self.D,
                n_transient=self.N_TRANSIENT,
                n_iter=self.N_ITER,
                theta0=self.THETA0,
            )
            np.testing.assert_allclose(rust, expected, atol=1e-9, rtol=0.0)

    def test_invalid_input_is_rejected(self):
        from dynachaos._rust import circle_map_lyapunov_sum

        with pytest.raises(ValueError):
            circle_map_lyapunov_sum(0.1, 0.25, 10, 0, 0.1)
        with pytest.raises(ValueError):
            circle_map_lyapunov_sum(np.nan, 0.25, 10, 10, 0.1)
        with pytest.raises(ValueError):
            circle_map_lyapunov_sum(0.1, np.inf, 10, 10, 0.1)


@rust_extension
class TestDelayedLogisticLyapunovParity:
    """Verify the Rust delayed-logistic spectrum against ``lyapunov_spectrum``.

    The orbit is bit-identical (the map uses only +, -, *), but the Python
    reference uses ``np.linalg.qr`` (LAPACK Householder) and the Rust kernel
    runs its own Householder QR, so the tangent frames differ in the last
    bits even though the state does not. The comparison is therefore
    absolute 1e-9, fixed by the dispatch. Measured worst absolute difference
    on the grid below: 8.3e-16.
    """

    A = 0.3
    D_VALUES = np.linspace(1.3, 2.5, 20)
    N_TRANSIENT = 10_000
    N_ITER = 50_000

    def test_spectrum_matches_python(self):
        from dynachaos._rust import delayed_logistic_lyapunov_tile
        from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
        from dynachaos.maps.delayed_logistic import delayed_logistic, delayed_logistic_jac

        # The kernel takes one shared state0 per call, while the Python
        # sweep (delayed_logistic.py:170) rebuilds the fixed-point seed
        # fp(D) +- 0.01 for every D. To reproduce the paper orbit bit for
        # bit, the kernel is called once per D with that D's seed; only the
        # QR then differs from np.linalg.qr.
        rust = np.empty((len(self.D_VALUES), 2))
        expected = np.empty((len(self.D_VALUES), 2))
        for k, d in enumerate(self.D_VALUES):
            fp = (np.sqrt(1.0 + 4.0 * d) - 1.0) / (2.0 * d)
            x0 = np.array([fp + 0.01, fp - 0.01])
            rust[k] = delayed_logistic_lyapunov_tile(
                self.A,
                np.array([d], dtype=np.float64),
                self.N_TRANSIENT,
                self.N_ITER,
                x0,
            )[0]
            expected[k] = lyapunov_spectrum(
                lambda s, d=d: delayed_logistic(s, self.A, d),
                lambda s, d=d: delayed_logistic_jac(s, self.A, d),
                x0,
                n_iter=self.N_ITER,
                n_transient=self.N_TRANSIENT,
            )
        np.testing.assert_allclose(rust, expected, atol=1e-9, rtol=0.0)

    def test_invalid_input_is_rejected(self):
        from dynachaos._rust import delayed_logistic_lyapunov_tile

        with pytest.raises(ValueError):
            delayed_logistic_lyapunov_tile(0.3, np.array([]), 10, 10, np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            delayed_logistic_lyapunov_tile(0.3, np.array([1.55]), 10, 0, np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            delayed_logistic_lyapunov_tile(0.3, np.array([1.55]), 10, 10, np.array([0.5]))


@rust_extension
class TestTorusDoublingLyapunovParity:
    """Verify the Rust torus spectra against ``lyapunov_spectrum``.

    Both maps are polynomial, so the orbit is bit-identical; only the QR
    differs from ``np.linalg.qr``. Absolute 1e-9 bound, fixed by the
    dispatch. Measured worst absolute difference on the grids below:
    1.4e-14 (map IV), 2.9e-16 (map I).
    """

    N_TRANSIENT = 10_000
    N_ITER = 50_000

    @pytest.mark.parametrize(
        ("map_kind", "A", "d_values", "state0", "map_function", "jac_function"),
        [
            (
                1,
                0.4,
                np.linspace(1.9, 2.25, 20),
                [0.5, 0.5, 0.5],
                "map_I",
                "map_I_jac",
            ),
            (
                4,
                0.3,
                np.linspace(1.48, 1.53, 20),
                [0.5, 0.45, 0.52, 0.48],
                "map_IV",
                "map_IV_jac",
            ),
        ],
    )
    def test_spectrum_matches_python(
        self, map_kind, A, d_values, state0, map_function, jac_function
    ):
        from dynachaos._rust import torus_doubling_lyapunov_tile
        from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
        from dynachaos.maps import torus_doubling

        f = getattr(torus_doubling, map_function)
        jac = getattr(torus_doubling, jac_function)
        expected = np.array(
            [
                lyapunov_spectrum(
                    lambda s, d=d: f(s, A, d),
                    lambda s, d=d: jac(s, A, d),
                    np.array(state0),
                    n_iter=self.N_ITER,
                    n_transient=self.N_TRANSIENT,
                )
                for d in d_values
            ]
        )
        rust = torus_doubling_lyapunov_tile(
            map_kind,
            A,
            np.asarray(d_values, dtype=np.float64),
            self.N_TRANSIENT,
            self.N_ITER,
            np.array(state0),
        )
        np.testing.assert_allclose(rust, expected, atol=1e-9, rtol=0.0)

    def test_invalid_input_is_rejected(self):
        from dynachaos._rust import torus_doubling_lyapunov_tile

        with pytest.raises(ValueError):
            torus_doubling_lyapunov_tile(2, 0.4, np.array([2.19]), 10, 10, np.array([0.5] * 3))
        with pytest.raises(ValueError):
            torus_doubling_lyapunov_tile(1, 0.4, np.array([2.19]), 10, 10, np.array([0.5] * 4))
        with pytest.raises(ValueError):
            torus_doubling_lyapunov_tile(1, 0.4, np.array([2.19]), 10, 0, np.array([0.5] * 3))


@rust_extension
class TestCoupledDelayedLyapunovParity:
    """Verify the Rust coupled-delayed spectrum against ``lyapunov_spectrum``.

    Same construction as the torus tests: polynomial map, bit-identical
    orbit, QR in Rust instead of LAPACK. Absolute 1e-9 bound, fixed by the
    dispatch. Measured worst absolute difference on the grid below: 4.9e-15.
    """

    A = 0.4
    EPS = 5e-3
    DB_VALUES = np.linspace(2.1, 2.65, 20)
    N_TRANSIENT = 10_000
    N_ITER = 30_000
    STATE0 = np.array([0.5, 0.5, 0.3, 0.3])

    def test_spectrum_matches_python(self):
        from dynachaos._rust import coupled_delayed_lyapunov_tile
        from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
        from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

        expected = np.array(
            [
                lyapunov_spectrum(
                    lambda s, db=db: coupled_delayed(s, self.A, db + 0.1, db, self.EPS),
                    lambda s, db=db: coupled_delayed_jac(s, self.A, db + 0.1, db, self.EPS),
                    self.STATE0,
                    n_iter=self.N_ITER,
                    n_transient=self.N_TRANSIENT,
                )
                for db in self.DB_VALUES
            ]
        )
        rust = coupled_delayed_lyapunov_tile(
            self.A,
            self.DB_VALUES,
            self.EPS,
            self.N_TRANSIENT,
            self.N_ITER,
            self.STATE0,
        )
        np.testing.assert_allclose(rust, expected, atol=1e-9, rtol=0.0)

    def test_invalid_input_is_rejected(self):
        from dynachaos._rust import coupled_delayed_lyapunov_tile

        with pytest.raises(ValueError):
            coupled_delayed_lyapunov_tile(0.4, np.array([]), 0.005, 10, 10, self.STATE0)
        with pytest.raises(ValueError):
            coupled_delayed_lyapunov_tile(0.4, np.array([2.3]), 0.005, 10, 10, np.array([0.5] * 3))


@rust_extension
class TestCoupledLogisticTrajectoryParity:
    """Verify phase, attractor and basin kernels against the Python map.

    Trajectories and the asymmetry grid are exact (``np.array_equal``). The
    phase Lyapunov column is not: NumPy ``log`` and Rust ``ln`` differ in the
    last bit. Measured worst absolute difference on the 20 by 10 paper subset
    below is 8.7e-19, inside the fixed 1e-9 bound.
    """

    N_TRANSIENT = 2_000
    N_RECORD = 5_000

    def test_phase_tile_matches_python(self):
        from dynachaos._rust import coupled_logistic_phase_tile
        from dynachaos.maps.coupled_logistic import compute_phase_diagram

        a_values = np.ascontiguousarray(np.linspace(0.5, 1.65, 500)[::25])
        d_values = np.ascontiguousarray(np.linspace(0.0, 0.3, 200)[::20])
        assert a_values.shape == (20,)
        assert d_values.shape == (10,)
        python = compute_phase_diagram(
            A_values=a_values,
            D_values=d_values,
            n_transient=self.N_TRANSIENT,
            n_sample=self.N_RECORD,
            output_path=None,
            progress_interval=0,
        )
        rust = coupled_logistic_phase_tile(
            a_values,
            d_values,
            self.N_TRANSIENT,
            self.N_RECORD,
            0.1,
            0.2,
        )
        np.testing.assert_array_equal(rust[..., 0], python["asym"])
        lyap_py = python["lyap"]
        lyap_rs = rust[..., 1]
        np.testing.assert_array_equal(np.isnan(lyap_rs), np.isnan(lyap_py))
        finite = np.isfinite(lyap_py)
        worst_lyap = float(np.max(np.abs(lyap_rs[finite] - lyap_py[finite])))
        assert worst_lyap <= 1e-9
        np.testing.assert_allclose(lyap_rs[finite], lyap_py[finite], atol=1e-9, rtol=0.0)

    def test_attractor_cases_match_python(self):
        from dynachaos._rust import coupled_logistic_attractor_tile
        from dynachaos.maps.coupled_logistic import ATTRACTOR_CASES, compute_attractors

        python = compute_attractors(
            n_transient=self.N_TRANSIENT,
            n_plot=self.N_RECORD,
            output_path=None,
        )
        for index, case in enumerate(ATTRACTOR_CASES):
            rust = coupled_logistic_attractor_tile(
                np.array([case.A], dtype=np.float64),
                0.1,
                self.N_TRANSIENT,
                self.N_RECORD,
                np.array(case.initial_state, dtype=np.float64),
            )
            expected = np.column_stack((python[f"x_{index}"], python[f"y_{index}"]))
            np.testing.assert_array_equal(rust[0], expected)

    def test_basin_labels_match_python_on_a_64_grid(self):
        from dynachaos._rust import coupled_logistic_basin_grid
        from dynachaos.maps.coupled_logistic import _basin_grid_python, _find_reference_orbit

        a = 1.35344
        d = 0.1
        n_grid = 64
        n_transient = 200
        reference = _find_reference_orbit(a, d, 0.1, 0.6, n_transient=80, period=32)
        x_range = np.linspace(-1.0, 1.0, n_grid)
        y_range = np.linspace(-1.0, 1.0, n_grid)
        python = _basin_grid_python(a, d, x_range, y_range, n_transient, reference)
        rust = coupled_logistic_basin_grid(x_range, y_range, a, d, n_transient, reference)
        np.testing.assert_array_equal(rust, python)


@rust_extension
class TestCoupledDelayedProjectionParity:
    """Verify ``(x, z)`` projections against ``coupled_delayed``.

    ``compute_projections`` writes the figure cache and hardcodes a 50_000
    record, so this test calls that function's map and projector at every
    ``PROJECTION_CASES`` value. ``n_record`` is 5_000, the shortest count the
    dispatch allows. Python passes no divergence check; the samples compared
    here stay finite, so the recorded pairs are exact.
    """

    A = 0.4
    EPS = 5e-3
    STATE0 = np.array([0.5, 0.5, 0.3, 0.3], dtype=np.float64)
    N_TRANSIENT = 2_000
    N_RECORD = 5_000

    def test_projection_cases_match_python(self):
        from dynachaos._rust import coupled_delayed_projection_tile
        from dynachaos.maps._iter import trajectory_after_transient
        from dynachaos.maps.coupled_delayed import PROJECTION_CASES, coupled_delayed

        db_values = np.array([case[0] for case in PROJECTION_CASES], dtype=np.float64)
        rust = coupled_delayed_projection_tile(
            self.A,
            db_values,
            self.EPS,
            self.N_TRANSIENT,
            self.N_RECORD,
            self.STATE0,
        )
        for index, db in enumerate(db_values):
            da = float(db) + 0.1
            python = trajectory_after_transient(
                self.STATE0,
                lambda state, da=da, db=float(db): coupled_delayed(state, self.A, da, db, self.EPS),
                self.N_TRANSIENT,
                self.N_RECORD,
                project_fn=lambda state: state[[0, 2]],
            )
            assert python is not None
            np.testing.assert_array_equal(rust[index], python)


@rust_extension
class TestFractalizationAttractorParity:
    """Verify fractalization orbits against ``iterate``.

    Fractalization is the delayed logistic map. The Rust side is
    ``delayed_logistic_attractor_tile``, the kernel the wasm export calls.
    ``iterate`` builds ``fp(D) ± 0.01`` when ``x0`` is omitted and passes no
    divergence check; the reused kernel instead fills a block with NaN when
    any ``|component| > 1e10``. The paper D values stay finite, so the
    recorded pairs are exact. ``n_record`` is 5_000, shorter than the
    figure's 500_000 and at the dispatch floor.
    """

    A = 0.3
    D_VALUES = (1.75, 1.86, 1.90, 1.92, 1.94, 1.945)
    N_TRANSIENT = 2_000
    N_RECORD = 5_000

    def test_paper_d_values_match_iterate(self):
        from dynachaos._rust import delayed_logistic_attractor_tile
        from dynachaos.maps.fractalization import iterate

        for d in self.D_VALUES:
            python = iterate(self.A, d, n_transient=self.N_TRANSIENT, n_record=self.N_RECORD)
            fp = (np.sqrt(1.0 + 4.0 * d) - 1.0) / (2.0 * d)
            state0 = np.array([fp + 0.01, fp - 0.01], dtype=np.float64)
            rust = delayed_logistic_attractor_tile(
                self.A,
                np.array([d], dtype=np.float64),
                self.N_TRANSIENT,
                self.N_RECORD,
                state0,
            )
            np.testing.assert_array_equal(rust[0], python)
