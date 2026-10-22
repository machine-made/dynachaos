"""Tests for embedding parameter selection (AMI, Cao, FNN).

Correctness tests use known-dimension systems:
- Sinusoid: AMI first minimum near quarter-period
- Circle trajectory: FNN should vanish at d=2
- Delayed logistic: Cao E1 should saturate near d=2-3
"""

import importlib

import numpy as np
import pytest
from conftest import logistic_series

from dynachaos.diagnostics.embedding import (
    _embed,
    _smooth_series,
    average_mutual_information,
    cao_method,
    false_nearest_neighbors,
    optimal_delay,
    optimal_dimension,
    select_dimension_cao,
)


def test_embed_rejects_series_too_short_for_requested_embedding():
    with pytest.raises(ValueError, match="Series too short"):
        _embed(np.arange(5.0), d=3, tau=3)


def test_smooth_series_applies_moving_average_when_window_greater_than_one():
    smoothed = _smooth_series(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), window=3)
    # Edge-padded moving average of length-3 window centered at each point.
    np.testing.assert_allclose(smoothed, [4 / 3, 2.0, 3.0, 4.0, 14 / 3])


class TestAMI:
    def test_ami_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 20 * np.pi, 2000)) + 0.01 * rng.standard_normal(2000)
        taus, mi = average_mutual_information(x, tau_max=50, n_bins=64)
        assert taus.shape == (50,)
        assert mi.shape == (50,)
        assert np.all(mi >= 0)

    def test_ami_sinusoid_first_minimum(self):
        """Sinusoid with period 20: AMI first minimum near tau=5 (quarter-period)."""
        t = np.arange(2000, dtype=np.float64)
        x = np.sin(2 * np.pi * 0.05 * t)
        tau_opt = optimal_delay(x, tau_max=50)
        # Quarter period = 5, allow some slack
        assert 3 <= tau_opt <= 7, f"Expected tau_opt near 5, got {tau_opt}"

    def test_ami_decreasing_initially(self):
        """AMI should generally decrease from tau=1."""
        x = logistic_series(n=5000)
        _, mi = average_mutual_information(x, tau_max=30)
        # First few values should decrease (MI drops with delay for chaotic series)
        assert mi[0] > mi[2], "AMI should decrease initially for chaotic series"

    def test_ami_rejects_invalid_public_inputs(self):
        with pytest.raises(ValueError, match="at least two"):
            average_mutual_information([1.0], tau_max=3)
        with pytest.raises(ValueError, match="finite values"):
            average_mutual_information([1.0, np.nan, 2.0], tau_max=3)
        with pytest.raises(ValueError, match="positive integers"):
            average_mutual_information(np.arange(10.0), tau_max=0)
        with pytest.raises(ValueError, match="positive integers"):
            average_mutual_information(np.arange(10.0), n_bins=1.5)

    def test_ami_constant_series_returns_zero(self):
        _, mi = average_mutual_information(np.ones(10), tau_max=5, n_bins=4)

        np.testing.assert_array_equal(mi, np.zeros(5))

    def test_ami_ravels_2d_input(self):
        x2d = np.sin(np.linspace(0, 20 * np.pi, 300)).reshape(-1, 1)
        _, mi = average_mutual_information(x2d, tau_max=5, n_bins=16)
        assert mi.shape == (5,)

    def test_ami_rejects_noncastable_tau_max(self):
        with pytest.raises(ValueError, match="positive integers"):
            average_mutual_information(np.arange(10.0), tau_max="abc")

    def test_ami_python_fallback_zeros_out_when_lag_leaves_too_few_pairs(self):
        # For a delay close to the series length, N - t < 2 leaves fewer
        # than two pairs, so the Python fallback floors those lags to 0
        # instead of computing a degenerate histogram.
        import dynachaos.diagnostics.embedding as emb_mod

        old = emb_mod._RUST_AVAILABLE
        emb_mod._RUST_AVAILABLE = False
        try:
            x = np.sin(np.linspace(0, 5, 8))
            _, mi = average_mutual_information(x, tau_max=7, n_bins=4)
        finally:
            emb_mod._RUST_AVAILABLE = old
        assert mi[-1] == 0.0
        assert mi[-2] == 0.0

    def test_ami_python_fallback_matches_rust_within_binning_noise(self):
        import dynachaos.diagnostics.embedding as emb_mod

        x = np.sin(np.linspace(0, 20 * np.pi, 500))
        _, mi_rust = average_mutual_information(x, tau_max=10, n_bins=16)

        old = emb_mod._RUST_AVAILABLE
        emb_mod._RUST_AVAILABLE = False
        try:
            _, mi_python = average_mutual_information(x, tau_max=10, n_bins=16)
        finally:
            emb_mod._RUST_AVAILABLE = old

        np.testing.assert_allclose(mi_python, mi_rust, atol=1e-8)

    def test_optimal_delay_returns_one_when_series_too_short_for_local_minimum(self):
        # tau_max=1 leaves a single MI value, so the search range for a local
        # minimum is empty and the function falls back to tau=1.
        x = np.sin(np.linspace(0, 20 * np.pi, 300))
        assert optimal_delay(x, tau_max=1) == 1


class TestCao:
    def test_cao_returns_correct_shape(self):
        x = logistic_series(n=3000)
        E1, E2 = cao_method(x, tau=1, d_max=10)
        assert E1.shape == (9,)  # d_max - 1
        assert E2.shape == (9,)

    def test_cao_logistic_saturation(self):
        """Logistic map (1D map -> d_opt=1 or 2): E1 should saturate quickly."""
        x = logistic_series(n=5000, a=1.99)
        E1, E2 = cao_method(x, tau=1, d_max=10)
        # E1 should be near 1 for d >= 2
        assert E1[2] > 0.9, f"E1(3) should be near 1, got {E1[2]}"

    def test_cao_deterministic_e2(self):
        """E2 should deviate from 1 for deterministic series."""
        x = logistic_series(n=5000, a=1.99)
        _, E2 = cao_method(x, tau=1, d_max=8)
        # For a deterministic system, E2 should not all be 1
        assert not np.allclose(E2, 1.0, atol=0.1), (
            "E2 should deviate from 1 for deterministic signal"
        )

    def test_cao_marks_e1_nan_when_series_too_short_for_high_dimensions(self):
        # len(x) - d*tau < 2 for the larger d's tested here, exercising the
        # M < 2 guard that fills E1/E2 with NaN instead of raising.
        x = np.sin(np.linspace(0, 10, 15))
        E1, E2 = cao_method(x, tau=3, d_max=6)
        assert np.any(np.isnan(E1))
        assert np.any(np.isnan(E2))

    def test_cao_theiler_window_still_returns_finite_statistics(self):
        # Densely sampled sinusoid puts many points within the Theiler
        # window of their nearest neighbor, exercising the re-query branch
        # that searches for a neighbor outside the window.
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 40 * np.pi, 400)) + 1e-3 * rng.standard_normal(400)
        E1, E2 = cao_method(x, tau=2, d_max=6, theiler_window=5)
        assert np.all(np.isfinite(E1))
        assert np.all(np.isfinite(E2))


class TestFNN:
    def test_fnn_returns_correct_shape(self):
        x = logistic_series(n=3000)
        f1, f2, f3 = false_nearest_neighbors(x, tau=1, d_max=8)
        assert f1.shape == (8,)
        assert f2.shape == (8,)
        assert f3.shape == (8,)

    def test_fnn_circle_low_at_d2(self):
        """Circle in 2D (d=1 intrinsic): FNN should vanish by d=2."""
        t = np.linspace(0, 100, 3000)
        x = np.sin(t)
        f1, f2, f3 = false_nearest_neighbors(x, tau=15, d_max=6)
        # FNN fraction should be low at d=2
        assert f3[1] < 0.15, f"FNN(d=2) should be low, got {f3[1]}"

    def test_fnn_bounded(self):
        """All FNN fractions should be in [0, 1]."""
        x = logistic_series(n=3000)
        f1, f2, f3 = false_nearest_neighbors(x, tau=1, d_max=6)
        for arr in (f1, f2, f3):
            valid = arr[np.isfinite(arr)]
            assert np.all(valid >= 0) and np.all(valid <= 1)

    def test_fnn_marks_nan_when_series_too_short_for_high_dimensions(self):
        x = np.sin(np.linspace(0, 10, 15))
        f1, f2, f3 = false_nearest_neighbors(x, tau=3, d_max=6)
        assert np.any(np.isnan(f1))
        assert np.any(np.isnan(f2))
        assert np.any(np.isnan(f3))

    def test_fnn_theiler_window_still_returns_finite_fractions(self):
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 40 * np.pi, 400)) + 1e-3 * rng.standard_normal(400)
        f1, f2, f3 = false_nearest_neighbors(x, tau=2, d_max=6, theiler_window=5)
        assert np.all(np.isfinite(f1))
        assert np.all(np.isfinite(f2))
        assert np.all(np.isfinite(f3))


class TestOptimalDimension:
    def test_cao_selector_env_forced_python_backend(self, monkeypatch):
        import dynachaos.diagnostics.embedding as emb_mod

        monkeypatch.setenv("DYNACHAOS_NO_RUST", "1")
        emb_mod = importlib.reload(emb_mod)
        try:
            e1 = np.array(
                [0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000],
                dtype=np.float64,
            )
            d = emb_mod.select_dimension_cao(
                e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2
            )
            assert emb_mod._RUST_AVAILABLE is False
            assert d == 4
        finally:
            monkeypatch.delenv("DYNACHAOS_NO_RUST", raising=False)
            importlib.reload(emb_mod)

    def test_cao_selector_python_fallback_path(self):
        from dynachaos.diagnostics import embedding as emb_mod

        e1 = np.array(
            [0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000],
            dtype=np.float64,
        )
        old_selector = emb_mod._select_dimension_cao_rs
        emb_mod._select_dimension_cao_rs = None
        try:
            d = select_dimension_cao(e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2)
        finally:
            emb_mod._select_dimension_cao_rs = old_selector

        assert d == 4

    def test_cao_selector_onset_not_tail(self):
        e1 = np.array(
            [0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000],
            dtype=np.float64,
        )
        d = select_dimension_cao(e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2)
        assert d == 4

    def test_cao_selector_ignores_low_dim_artifacts(self):
        e1 = np.array(
            [0.0010, 0.1340, 0.1587, 0.8184, 0.8658, 0.9646, 0.9961, 0.9948, 0.9989],
            dtype=np.float64,
        )
        d = select_dimension_cao(e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2)
        assert d >= 6

    def test_cao_method(self):
        x = logistic_series(n=5000, a=1.99)
        d = optimal_dimension(x, tau=1, d_max=10, method="cao")
        assert 1 <= d <= 4, f"Expected d_opt near 1-3 for logistic, got {d}"

    def test_cao_legacy_method(self):
        x = logistic_series(n=5000, a=1.99)
        d = optimal_dimension(x, tau=1, d_max=10, method="cao_legacy")
        assert 1 <= d <= 4, f"Expected d_opt near 1-3 for logistic, got {d}"

    def test_fnn_method(self):
        x = logistic_series(n=5000, a=1.99)
        d = optimal_dimension(x, tau=1, d_max=10, method="fnn")
        assert 1 <= d <= 5, f"Expected d_opt near 1-3 for logistic, got {d}"

    def test_cao_rejects_d_max_below_2(self):
        series = logistic_series(n=500)
        with pytest.raises(ValueError, match="d_max must be >= 2"):
            optimal_dimension(series, tau=1, d_max=1, method="cao")

    def test_cao_legacy_rejects_d_max_below_2(self):
        series = logistic_series(n=500)
        with pytest.raises(ValueError, match="d_max must be >= 2"):
            optimal_dimension(series, tau=1, d_max=1, method="cao_legacy")

    def test_cao_d_max_2_succeeds(self):
        """Boundary: d_max=2 is the minimum valid value for Cao."""
        series = logistic_series(n=500)
        d = optimal_dimension(series, tau=1, d_max=2, method="cao")
        assert 1 <= d <= 2

    def test_d_max_zero_rejects_all_methods(self):
        series = logistic_series(n=500)
        with pytest.raises(ValueError, match="d_max must be >= 1"):
            optimal_dimension(series, tau=1, d_max=0, method="fnn")

    def test_invalid_method_raises(self):
        x = logistic_series(n=1000)
        with pytest.raises(ValueError, match="Unknown method"):
            optimal_dimension(x, tau=1, method="invalid")

    def test_cao_legacy_falls_back_to_d_max_when_e1_never_exceeds_threshold(self):
        # Pure noise: E1 stays well below 0.95 across all tested dimensions,
        # so the legacy selector exhausts its search and returns d_max.
        rng = np.random.default_rng(3)
        x = rng.standard_normal(3000)
        d = optimal_dimension(x, tau=1, d_max=5, method="cao_legacy")
        assert d == 5

    def test_fnn_falls_back_to_d_max_when_fraction_never_drops_below_threshold(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(3000)
        d = optimal_dimension(x, tau=1, d_max=5, method="fnn")
        assert d == 5

    def test_select_dimension_cao_python_path_handles_empty_and_all_nan_input(self):
        from dynachaos.diagnostics import embedding as emb_mod

        old_selector = emb_mod._select_dimension_cao_rs
        emb_mod._select_dimension_cao_rs = None
        try:
            assert select_dimension_cao(np.array([]), min_dim=2) == 2
            assert select_dimension_cao(np.array([np.nan, np.nan]), min_dim=2) == 2
        finally:
            emb_mod._select_dimension_cao_rs = old_selector

    def test_select_dimension_cao_swaps_inverted_near_one_band(self):
        from dynachaos.diagnostics import embedding as emb_mod

        e1 = np.array(
            [0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000],
            dtype=np.float64,
        )
        old_selector = emb_mod._select_dimension_cao_rs
        emb_mod._select_dimension_cao_rs = None
        try:
            normal = select_dimension_cao(e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2)
            # near_one_lower/upper passed reversed must still work: the
            # function swaps them internally rather than raising.
            swapped = select_dimension_cao(e1, near_one_lower=1.03, near_one_upper=0.97, min_dim=2)
        finally:
            emb_mod._select_dimension_cao_rs = old_selector
        assert normal == swapped == 4

    def test_select_dimension_cao_python_path_uses_fallback_ladder(self):
        from dynachaos.diagnostics import embedding as emb_mod

        old_selector = emb_mod._select_dimension_cao_rs
        emb_mod._select_dimension_cao_rs = None
        try:
            # An isolated near-1 point too close to the array end for a full
            # plateau window forces the primary onset search to fall through
            # to Fallback 1 (first near-one crossing).
            e1_fallback1 = np.array([0.1, 0.99, 0.5, 0.4, 1.0])
            d1 = select_dimension_cao(
                e1_fallback1, near_one_lower=0.95, near_one_upper=1.05, min_dim=2, plateau_span=3
            )
            assert d1 == 2

            # No value ever enters the near-one band -> Fallback 2 (closest
            # to 1 overall).
            e1_fallback2 = np.array([0.1, 0.3, 0.5, 0.7])
            d2 = select_dimension_cao(
                e1_fallback2, near_one_lower=0.95, near_one_upper=1.05, min_dim=2
            )
            assert d2 == 4

            # min_dim beyond every available dimension: no candidate passes
            # any fallback, so the final catch-all return fires.
            e1_none = np.array([0.5, 0.6])
            d3 = select_dimension_cao(e1_none, near_one_lower=0.95, near_one_upper=1.05, min_dim=10)
            assert d3 == 10

            # A monotonically rising in-band window has a peak-to-peak span
            # above the 1.5*saturation_tol ptp criterion, but each
            # consecutive step stays within saturation_tol, exercising the
            # separate max-consecutive-diff acceptance criterion.
            e1_small_steps = np.array([0.1, 0.96, 0.97, 0.98, 0.99, 1.00])
            d4 = select_dimension_cao(
                e1_small_steps,
                near_one_lower=0.95,
                near_one_upper=1.05,
                min_dim=2,
                plateau_span=5,
                saturation_tol=0.02,
            )
            assert d4 == 2
        finally:
            emb_mod._select_dimension_cao_rs = old_selector


class TestCorrelationIntegralImproved:
    def test_backward_compat(self):
        """Old-style call (no theiler_window, no norm) should still work."""
        from dynachaos.diagnostics.correlation import correlation_integral

        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        r_values = np.logspace(-2, 0, 10)
        C = correlation_integral(traj, r_values)
        assert C.shape == (10,)
        assert np.all(C >= 0)
        assert np.all(C <= 1)

    def test_theiler_window_reduces_counts(self):
        """With Theiler window, counts should be <= without."""
        from dynachaos.diagnostics.correlation import correlation_integral

        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        r_values = np.logspace(-2, 0, 10)
        C0 = correlation_integral(traj, r_values, theiler_window=0)
        C10 = correlation_integral(traj, r_values, theiler_window=10)
        # With theiler window, C should generally change (fewer pairs)
        assert not np.allclose(C0, C10), "Theiler window should affect C(r)"

    def test_correlation_integral_is_monotone_in_radius(self):
        from dynachaos.diagnostics.correlation import correlation_integral

        rng = np.random.default_rng(123)
        traj = rng.normal(size=(200, 2))
        r_values = np.array([0.05, 0.1, 0.2, 0.4, 0.8], dtype=np.float64)

        C = correlation_integral(traj, r_values, theiler_window=3)

        assert np.all(np.diff(C) >= 0.0)

    def test_valid_pair_count_decreases_with_theiler_window(self):
        from dynachaos.diagnostics.correlation import _valid_pair_count

        counts = np.array([_valid_pair_count(20, w) for w in range(6)])

        assert np.all(np.diff(counts) < 0)

    def test_invalid_norm_raises(self):
        from dynachaos.diagnostics.correlation import correlation_integral

        with pytest.raises(ValueError, match="norm must be one of"):
            correlation_integral(np.arange(5), [1.0], norm="manhattan")

    def test_negative_theiler_window_raises(self):
        from dynachaos.diagnostics.correlation import correlation_integral

        with pytest.raises(ValueError, match="theiler_window must be >= 0"):
            correlation_integral(np.arange(5), [1.0], theiler_window=-1)

    def test_constant_trajectory_returns_nan_dimension(self):
        from dynachaos.diagnostics.correlation import correlation_dimension

        D2, r_values, C_values, slopes, scaling = correlation_dimension(np.ones((10, 2)))

        assert np.isnan(D2)
        assert r_values.shape == (0,)
        assert C_values.shape == (0,)
        assert slopes.shape == (0,)
        assert scaling.shape == (0,)

    def test_too_short_trajectory_returns_nan_dimension(self):
        from dynachaos.diagnostics.correlation import correlation_dimension

        for traj in (np.array([]), np.array([1.0])):
            D2, r_values, C_values, slopes, scaling = correlation_dimension(traj)

            assert np.isnan(D2)
            assert r_values.shape == (0,)
            assert C_values.shape == (0,)
            assert slopes.shape == (0,)
            assert scaling.shape == (0,)

    def test_correlation_dimension_circle(self):
        """Circle (D=1) should give D2 ~ 1 with improved G-P."""
        from dynachaos.diagnostics.correlation import correlation_dimension

        t = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        D2, _, _, _, _ = correlation_dimension(traj)
        assert 0.7 < D2 < 1.4, f"Expected D2 ~ 1, got {D2}"
