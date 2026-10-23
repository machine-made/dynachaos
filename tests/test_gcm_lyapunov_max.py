import warnings

import numpy as np
import pytest

from dynachaos.cml import gcm_lyapunov_max
from dynachaos.cml.primitives import gcm_step
from dynachaos.pipelines.registry import get_section


def _dense_lyapunov_max(a, eps, n_sites, n_transient, n_measure, seed):
    """Reference: iterate one tangent vector with the full N x N Jacobian."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n_sites)
    for _ in range(n_transient):
        x = gcm_step(x, a, eps)
    v = rng.standard_normal(n_sites)
    v /= np.linalg.norm(v)
    log_sum = 0.0
    for _ in range(n_measure):
        fp = -2.0 * a * x
        # d x_{n+1}(i) / d x_n(j) = (1 - eps) delta_ij f'(x_j) + (eps / N) f'(x_j)
        jacobian = (1.0 - eps) * np.diag(fp) + (eps / n_sites) * np.outer(np.ones(n_sites), fp)
        v = jacobian @ v
        x = gcm_step(x, a, eps)
        r = np.linalg.norm(v)
        log_sum += np.log(r)
        v /= r
    return log_sum / n_measure


@pytest.mark.parametrize("a", [1.5, 1.8, 1.95])
def test_lyapunov_max_equals_dense_jacobian_tangent_map(a):
    kwargs = {"eps": 0.1, "n_sites": 12, "n_transient": 200, "n_measure": 3000, "seed": 11}

    fast = gcm_lyapunov_max.lyapunov_max(a, **kwargs)
    dense = _dense_lyapunov_max(a, **kwargs)

    assert fast == pytest.approx(dense, abs=1e-10)


def test_lyapunov_max_of_fully_coupled_map_is_the_logistic_exponent():
    # eps = 1 makes all sites equal after one step, so the tangent vector follows
    # the single map f(x) = 1 - 2 x^2, whose exponent is ln 2.
    lam = gcm_lyapunov_max.lyapunov_max(
        2.0, 1.0, n_sites=16, n_transient=1000, n_measure=100_000, seed=3
    )

    assert lam == pytest.approx(np.log(2.0), abs=2e-3)


def test_lyapunov_max_of_uncoupled_maps_is_the_logistic_exponent():
    lam = gcm_lyapunov_max.lyapunov_max(
        2.0, 0.0, n_sites=4, n_transient=1000, n_measure=100_000, seed=3
    )

    assert lam == pytest.approx(np.log(2.0), abs=2e-3)


def test_lyapunov_max_at_a_stable_fixed_point_is_ln_of_the_multiplier():
    # For a = 0.5 every site sits on the fixed point x* with f'(x*) = 1 - sqrt(1 + 4 a).
    # The uniform mode has multiplier f'(x*) and the others (1 - eps) f'(x*), so
    # lambda_max = ln |f'(x*)|. The finite-time mean differs by ln(c) / n_measure, where c is
    # the weight of the uniform mode in the first tangent vector, so n_measure is large.
    a, eps = 0.5, 0.1
    multiplier = abs(1.0 - np.sqrt(1.0 + 4.0 * a))

    lam = gcm_lyapunov_max.lyapunov_max(
        a, eps, n_sites=8, n_transient=500, n_measure=40_000, seed=5
    )

    assert lam == pytest.approx(np.log(multiplier), abs=2e-4)


def test_lyapunov_max_rejects_an_orbit_that_leaves_the_interval():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError, match="orbit left"):
            gcm_lyapunov_max.lyapunov_max(3.0, 0.1, 8, n_transient=50, n_measure=50, seed=1)


def test_compute_seeds_each_run_by_size_seed_and_a_index(tmp_path):
    a_values = np.array([1.6, 1.9])
    output_path = tmp_path / "gcm_lyapunov_max.npz"

    payload = gcm_lyapunov_max.compute(
        a_values=a_values,
        N_values=(6, 10),
        n_seeds=3,
        n_transient=20,
        n_measure=60,
        seed_base=7,
        output_path=output_path,
    )

    assert tuple(payload) == (
        "a_values",
        "N_values",
        "eps",
        "n_seeds",
        "n_transient",
        "n_measure",
        "seed_base",
        "runs_6",
        "mean_6",
        "se_6",
        "runs_10",
        "mean_10",
        "se_10",
    )
    runs = payload["runs_10"]
    assert runs.shape == (3, 2)
    for s in range(3):
        for ia, a in enumerate(a_values):
            expected = gcm_lyapunov_max.lyapunov_max(a, 0.1, 10, 20, 60, 7 + 1000 * 10 + s + ia)
            assert runs[s, ia] == expected
    np.testing.assert_allclose(payload["mean_10"], runs.mean(axis=0))
    np.testing.assert_allclose(payload["se_10"], runs.std(axis=0, ddof=1) / np.sqrt(3))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_array_equal(saved["runs_6"], payload["runs_6"])


def test_plot_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "gcm_lyapunov_max.png"
    monkeypatch.setattr(gcm_lyapunov_max, "OUTPUT_PNG", png_path)
    payload = gcm_lyapunov_max.compute(
        a_values=np.linspace(1.5, 1.9, 5),
        n_seeds=2,
        n_transient=20,
        n_measure=40,
        N_values=gcm_lyapunov_max.N_VALUES,
        output_path=None,
    )

    gcm_lyapunov_max.plot(payload)

    assert png_path.stat().st_size > 0


def test_registry_contract_matches_module_keys():
    spec = get_section("sec10_gcm")

    assert "dynachaos.cml.gcm_lyapunov_max" in spec.modules
    assert "gcm_lyapunov_max.npz" in spec.cache_files
    assert "gcm_lyapunov_max.png" in spec.output_files
    assert set(spec.required_npz_keys("gcm_lyapunov_max.npz")) == set(
        gcm_lyapunov_max.REQUIRED_KEYS
    )


def test_committed_cache_shows_chaos_above_the_onset_and_no_size_dependence():
    with np.load("figures/sec10_gcm/gcm_lyapunov_max.npz", allow_pickle=False) as data:
        a = data["a_values"]
        means = {N: data[f"mean_{N}"] for N in (500, 1000, 2000)}
        ses = {N: data[f"se_{N}"] for N in (500, 1000, 2000)}
        assert data["runs_500"].shape == (int(data["n_seeds"][0]), a.size)

    assert a[0] == pytest.approx(1.40)
    assert a[-1] == pytest.approx(2.00)
    for N in means:
        assert np.all(np.isfinite(means[N]))
        assert np.all(ses[N] >= 0.0)
    # The exponent of the site dynamics at a = 2 is the one of the logistic map, ln 2,
    # reduced by the coupling; it must stay positive and below ln 2.
    for N, mean in means.items():
        assert 0.0 < mean[-1] < np.log(2.0), N
    # Sites are exchangeable, so lambda_max must not depend on N beyond seed scatter.
    spread = np.max(np.abs(means[2000] - means[500]))
    assert spread < 0.05
