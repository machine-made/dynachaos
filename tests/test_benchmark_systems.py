"""Fast CI validation tests for benchmark systems.

Uses small N and wide tolerances for speed. Validates that each system
recovers D2 and Lyapunov exponents in the right ballpark.
"""

import numpy as np


class TestLogisticBenchmark:
    """Logistic map a=2.0: lambda_1 = ln(2) = 0.6931, D2 = 1.0."""

    def test_lyapunov_exact(self):
        from dynachaos.diagnostics.lyapunov import lyapunov_exponent_1d
        from dynachaos.maps.base import LogisticMap

        lm = LogisticMap(a=2.0)
        lam = lyapunov_exponent_1d(lm.f, lm.df, x0=0.1, n_iter=50_000, n_transient=5_000)
        assert abs(lam - np.log(2)) < 0.01

    def test_d2(self):
        from dynachaos.diagnostics import correlation_dimension
        from dynachaos.diagnostics.recurrence import embed_time_delay
        from dynachaos.maps.base import LogisticMap

        lm = LogisticMap(a=2.0)
        traj = lm.trajectory(x0=0.1, n_iter=10_000, n_transient=2_000)
        embedded = embed_time_delay(traj, d=2, tau=1)
        D2, _, _, _, _ = correlation_dimension(embedded, n_r=30, theiler_window=1)
        assert 0.8 < D2 < 1.3


class TestHenonBenchmark:
    """Henon map a=1.4, b=0.3: lambda_1 ~ 0.42, D2 ~ 1.21."""

    def test_trajectory_bounded(self):
        from dynachaos.maps.base import HenonMap

        hm = HenonMap(a=1.4, b=0.3)
        traj = hm.trajectory(np.array([0.1, 0.1]), n_iter=5000, n_transient=2000)
        assert np.all(np.abs(traj) < 3)

    def test_lyapunov(self):
        from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
        from dynachaos.maps.henon import henon, henon_jac

        a, b = 1.4, 0.3

        def f(state):
            return henon(state, a, b)

        def jac(state):
            return henon_jac(state, a, b)

        spectrum = lyapunov_spectrum(f, jac, np.array([0.1, 0.1]), n_iter=50_000, n_transient=5_000)
        assert 0.35 < spectrum[0] < 0.50

    def test_d2(self):
        from dynachaos.diagnostics import correlation_dimension
        from dynachaos.diagnostics.recurrence import embed_time_delay
        from dynachaos.maps.base import HenonMap

        hm = HenonMap(a=1.4, b=0.3)
        traj = hm.trajectory(np.array([0.1, 0.1]), n_iter=10_000, n_transient=2_000)
        embedded = embed_time_delay(traj[:, 0], d=3, tau=1)
        D2, _, _, _, _ = correlation_dimension(embedded, n_r=30, theiler_window=1)
        assert 1.0 < D2 < 1.5


class TestLorenzBenchmark:
    """Lorenz system: lambda_1 ~ 0.91, D2 ~ 2.05."""

    def test_trajectory_bounded(self):
        from dynachaos.maps.flows import lorenz_trajectory

        traj = lorenz_trajectory(t_span=(0, 50), dt=0.01, t_transient=10.0)
        assert np.all(np.abs(traj[:, 0]) < 30)
        assert np.all(traj[:, 2] > 0)

    def test_lyapunov_spectrum(self):
        from dynachaos.diagnostics.lyapunov import flow_lyapunov_spectrum
        from dynachaos.maps.flows import lorenz_jac, lorenz_rhs

        spectrum = flow_lyapunov_spectrum(
            lorenz_rhs,
            lorenz_jac,
            x0=np.array([1.0, 1.0, 1.0]),
            t_total=100.0,
            dt=0.01,
            t_transient=20.0,
            reorth_dt=1.0,
        )
        assert 0.7 < spectrum[0] < 1.1
        assert abs(spectrum[1]) < 0.1

    def test_d2(self):
        from dynachaos.diagnostics import correlation_dimension
        from dynachaos.diagnostics.recurrence import embed_time_delay
        from dynachaos.maps.flows import lorenz_trajectory

        traj = lorenz_trajectory(t_span=(0, 100), dt=0.01, t_transient=20.0)
        x = traj[::5, 0]
        embedded = embed_time_delay(x, d=5, tau=5)
        D2, _, _, _, _ = correlation_dimension(embedded, n_r=30, theiler_window=5)
        assert 1.5 < D2 < 2.8


class TestRosslerBenchmark:
    """Rossler system: lambda_1 ~ 0.07, D2 ~ 2.01."""

    def test_trajectory_bounded(self):
        from dynachaos.maps.flows import rossler_trajectory

        traj = rossler_trajectory(t_span=(0, 200), dt=0.05, t_transient=50.0)
        assert np.all(np.abs(traj[:, 0]) < 20)

    def test_lyapunov_spectrum(self):
        from dynachaos.diagnostics.lyapunov import flow_lyapunov_spectrum
        from dynachaos.maps.flows import rossler_jac, rossler_rhs

        spectrum = flow_lyapunov_spectrum(
            rossler_rhs,
            rossler_jac,
            x0=np.array([1.0, 1.0, 0.0]),
            t_total=200.0,
            dt=0.01,
            t_transient=50.0,
            reorth_dt=1.0,
        )
        assert 0.03 < spectrum[0] < 0.12

    def test_d2(self):
        from dynachaos.diagnostics import correlation_dimension
        from dynachaos.diagnostics.recurrence import embed_time_delay
        from dynachaos.maps.flows import rossler_trajectory

        traj = rossler_trajectory(t_span=(0, 500), dt=0.05, t_transient=100.0)
        x = traj[::2, 0]
        embedded = embed_time_delay(x, d=5, tau=5)
        D2, _, _, _, _ = correlation_dimension(embedded, n_r=30, theiler_window=5)
        assert 1.5 < D2 < 2.8


class TestMackeyGlassBenchmark:
    """Mackey-Glass DDE tau=17: D2 ~ 2.1."""

    def test_bounded(self):
        from dynachaos.maps.flows import mackey_glass_series

        x = mackey_glass_series(n_points=2000, tau=17, t_transient=200)
        assert np.all(x > 0)
        assert np.all(x < 2.5)

    def test_d2(self):
        from dynachaos.diagnostics import correlation_dimension
        from dynachaos.diagnostics.recurrence import embed_time_delay
        from dynachaos.maps.flows import mackey_glass_series

        x = mackey_glass_series(n_points=5000, tau=17, t_transient=300)
        embedded = embed_time_delay(x, d=5, tau=4)
        D2, _, _, _, _ = correlation_dimension(embedded, n_r=30, theiler_window=4)
        assert 1.5 < D2 < 3.0
