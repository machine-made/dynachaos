import numpy as np
import pytest

from dynachaos.cml import gcm_collective_fsle as fsle
from dynachaos.cml import gcm_lyapunov_max
from dynachaos.pipelines.registry import get_section

DELTA0 = 1e-7
CACHE = "figures/sec10_gcm/gcm_collective_fsle.npz"


def _dense_lambda_m(a, eps, x, n_transient, n_measure, seed):
    """Reference: tangent map with the full N x N Jacobian of the heterogeneous GCM."""
    n = x.size
    rng = np.random.default_rng(seed)
    for _ in range(n_transient):
        x, _ = fsle.step(x, a, eps)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    total = 0.0
    for _ in range(n_measure):
        fp = -2.0 * a * x
        # d x_{n+1}(i) / d x_n(j) = (1 - eps) delta_ij f_j'(x_j) + (eps / N) f_j'(x_j)
        jacobian = (1.0 - eps) * np.diag(fp) + (eps / n) * np.outer(np.ones(n), fp)
        v = jacobian @ v
        x, _ = fsle.step(x, a, eps)
        r = np.linalg.norm(v)
        total += np.log(r)
        v /= r
    return total / n_measure


def _synthetic_data(sizes=(10_000, 100_000, 1_000_000), lam_c=0.025, lam_m=0.41):
    """Cache-shaped dict whose <tau> follows Eq. (6) exactly, with Delta_m = 0.3 N^(-1/2)."""
    data = {
        "deltas": fsle.DELTAS,
        "Ns": np.array(sizes),
        "n_max": np.array([fsle.N_MAX]),
        "n_samples": np.array([4]),
    }
    log_d = np.log(fsle.DELTAS)
    for N in sizes:
        log_dm = np.log(0.3 / np.sqrt(N))
        t = fsle.t_model(log_d, lam_c, log_dm, lam_m)
        tau = np.tile(t - np.log(DELTA0) / lam_m, (4, 1))
        data[f"chaos_tau_N{N}"] = tau
        data[f"chaos_lam_m_N{N}"] = np.array([lam_m])
        data[f"control_tau_N{N}"] = tau
        data[f"control_lam_m_N{N}"] = np.array([lam_m])
    n_f = sizes[-1]
    data[f"chaos_tau_N{n_f}_d1e-06"] = data[f"chaos_tau_N{n_f}"]
    data[f"chaos_tau_N{n_f}_d1e-05"] = data[f"chaos_tau_N{n_f}"]
    return data


def test_a_profile_spans_a0_plus_minus_half_delta_a():
    a = fsle.a_profile(1.9, 0.05, 1000)

    assert a.shape == (1000,)
    assert a[-1] == pytest.approx(1.9 + 0.025)
    assert a[0] == pytest.approx(1.9 + 0.05 * (2 - 1000) / 2000)
    assert np.all(np.diff(a) > 0.0)
    np.testing.assert_allclose(fsle.a_profile(1.6962, 0.0, 5), 1.6962)


def test_step_returns_the_old_mean_field_and_conserves_it():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, 64)
    a = fsle.a_profile(1.9, 0.05, 64)

    x_new, h = fsle.step(x, a, 0.098)

    assert h == pytest.approx(np.mean(1.0 - a * x * x))
    # The coupling redistributes the mean field, so the mean of the new state is h itself.
    assert x_new.mean() == pytest.approx(h)


def test_check_rejects_non_finite_and_escaped_orbits():
    fsle.check(np.array([-1.0, 0.0, 1.0]), "ok")
    with pytest.raises(RuntimeError, match="orbit left"):
        fsle.check(np.array([0.0, 1.0001]), "escaped")
    with pytest.raises(RuntimeError, match="orbit left"):
        fsle.check(np.array([0.0, np.nan]), "nan")


@pytest.mark.parametrize("seed", [1, 2])
def test_lambda_m_equals_dense_jacobian_tangent_map(seed):
    n = 10
    a = fsle.a_profile(1.9, 0.05, n)
    x = np.random.default_rng(seed).uniform(-1.0, 1.0, n)

    fast = fsle.lambda_m(a, 0.098, x, n_transient=100, n_measure=2000, seed=seed)
    dense = _dense_lambda_m(a, 0.098, x, 100, 2000, seed)

    assert fast == pytest.approx(dense, abs=1e-10)


def test_lambda_m_of_identical_elements_equals_gcm_lyapunov_max():
    # Same model and same tangent map as gcm_lyapunov_max; the two start from different
    # tangent vectors, so they agree to ln(c) / n_measure.
    n, a0, eps = 24, 1.8, 0.1
    n_transient, n_measure = 500, 40_000
    x0 = np.random.default_rng(9).uniform(-1.0, 1.0, n)

    lam_fsle = fsle.lambda_m(fsle.a_profile(a0, 0.0, n), eps, x0, n_transient, n_measure, seed=4)
    lam_max = gcm_lyapunov_max.lyapunov_max(a0, eps, n, n_transient, n_measure, seed=9)

    assert lam_fsle == pytest.approx(lam_max, abs=2e-3)


def test_first_passage_clock_starts_at_delta0_and_is_monotone():
    n = 400
    a = fsle.a_profile(1.9, 0.05, n)
    rng = np.random.default_rng(5)
    x = rng.uniform(-1.0, 1.0, n)
    for _ in range(500):
        x, _ = fsle.step(x, a, 0.098)

    tau, x_after = fsle.first_passage(x, a, 0.098, DELTA0, rng, fsle.N_MAX)

    assert tau.shape == fsle.DELTAS.shape
    assert tau.dtype == np.int64
    assert np.all(np.diff(tau) >= 0)
    assert tau.min() >= 0
    assert tau.max() <= fsle.N_MAX + 1
    assert x_after.shape == x.shape
    # Delta = delta0 is reached at the clock start by definition.
    tau0, _ = fsle.first_passage(
        x_after, a, 0.098, DELTA0, rng, fsle.N_MAX, deltas=np.array([DELTA0])
    )
    assert tau0[0] == 0


def test_first_passage_marks_unreached_scales_with_n_max_plus_one():
    n = 400
    a = fsle.a_profile(1.9, 0.05, n)
    rng = np.random.default_rng(6)
    x = rng.uniform(-1.0, 1.0, n)
    for _ in range(500):
        x, _ = fsle.step(x, a, 0.098)

    # The mean-field difference cannot exceed 2, so Delta = 5 is never reached.
    tau, _ = fsle.first_passage(x, a, 0.098, DELTA0, rng, 50, deltas=np.array([DELTA0, 5.0]))

    np.testing.assert_array_equal(tau, [0, 51])


def test_first_passage_raises_when_delta0_is_never_reached():
    n = 50
    a = fsle.a_profile(0.5, 0.0, n)  # stable fixed point: the perturbation decays
    rng = np.random.default_rng(7)
    x = np.full(n, 0.5)

    with pytest.raises(RuntimeError, match="never reached delta0"):
        fsle.first_passage(x, a, 0.1, 1e-3, rng, 20)


def test_t_model_is_continuous_with_the_two_slopes():
    lam_m, lam_c, log_dm = 0.41, 0.025, np.log(1e-3)

    below = fsle.t_model(np.array([log_dm - 2.0, log_dm - 1.0]), lam_c, log_dm, lam_m)
    above = fsle.t_model(np.array([log_dm + 1.0, log_dm + 2.0]), lam_c, log_dm, lam_m)

    assert np.diff(below)[0] == pytest.approx(1.0 / lam_m)
    assert np.diff(above)[0] == pytest.approx(1.0 / lam_c)
    at_dm = fsle.t_model(np.array([log_dm - 1e-12, log_dm + 1e-12]), lam_c, log_dm, lam_m)
    assert at_dm[0] == pytest.approx(at_dm[1], abs=1e-9)


def test_fit_t_recovers_the_parameters_of_a_synthetic_cache():
    data = _synthetic_data(sizes=(10_000, 100_000, 1_000_000), lam_c=0.025, lam_m=0.41)
    taus = data["chaos_tau_N100000"]

    popt, perr, ok = fsle.fit_t(taus, DELTA0, 0.41, fsle.N_MAX + 1_000_000, 0.05)

    assert popt[0] == pytest.approx(0.025, rel=1e-6)
    assert popt[1] == pytest.approx(np.log(0.3 / np.sqrt(100_000)), abs=1e-6)
    assert np.all(perr >= 0.0)
    assert np.all(fsle.DELTAS[ok] <= 0.05)


def test_fsle_curve_inverts_the_definition():
    lam = 0.3
    taus = np.tile(np.log(fsle.DELTAS / DELTA0) / lam, (3, 1))

    delta, curve = fsle.fsle_curve(taus, DELTA0, 10_000)

    np.testing.assert_allclose(curve, lam)
    assert np.all(delta > DELTA0)


def test_fsle_curve_drops_scales_that_some_sample_did_not_reach():
    taus = np.tile(np.arange(1, fsle.DELTAS.size + 1), (2, 1))
    taus[1, -1] = 3001

    delta, _ = fsle.fsle_curve(taus, DELTA0, 3000)

    assert delta[-1] == fsle.DELTAS[-2]


def test_analyse_recovers_the_size_scaling_of_delta_m():
    data = _synthetic_data()

    analysis = fsle.analyse(data)

    assert analysis["slope"] == pytest.approx(-0.5, abs=1e-6)
    assert np.isfinite(analysis["slope_err"])
    for N in (10_000, 100_000, 1_000_000):
        popt, _, lam = analysis["fits"][N]
        assert popt[0] == pytest.approx(0.025, rel=1e-6)
        assert lam == pytest.approx(0.41)


def test_analyse_with_two_sizes_has_no_slope_error():
    data = _synthetic_data(sizes=(100_000, 1_000_000))

    assert np.isnan(fsle.analyse(data)["slope_err"])


def test_plot_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "gcm_collective_fsle.png"
    monkeypatch.setattr(fsle, "OUTPUT_PNG", png_path)

    fsle.plot(_synthetic_data())

    assert png_path.stat().st_size > 0


def test_compute_load_and_add_system_size_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(fsle, "FIG_DIR", tmp_path)
    monkeypatch.setattr(fsle, "N_MAX", 600)
    monkeypatch.setattr(fsle, "N_TRANSIENT", 300)
    output_path = tmp_path / "gcm_collective_fsle.npz"

    payload = fsle.compute(Ns=(200, 300, 400), samples=2, output_path=output_path)

    assert set(payload) >= {
        "deltas",
        "Ns",
        "n_max",
        "n_samples",
        "chaos_tau_N200",
        "chaos_lam_m_N400",
        "chaos_tau_N400_d1e-06",
        "chaos_tau_N400_d1e-05",
        "control_tau_N300",
        "control_tau_N400",
        "control_lam_m_N400",
    }
    assert payload["chaos_tau_N200"].shape == (2, fsle.DELTAS.size)
    assert "control_tau_N200" not in payload
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)

    # An extra size goes beside the cache and leaves the cache itself untouched.
    before = {k: v.copy() for k, v in np.load(output_path, allow_pickle=False).items()}
    extra = fsle.add_system_size(500, path=output_path)
    assert extra.name == "gcm_collective_fsle_N500.npz"
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(before)
        for key in before:
            np.testing.assert_array_equal(saved[key], before[key])

    # load_data checks required keys before merging, and the tiny cache has other sizes.
    with pytest.raises(KeyError, match="cache lacks"):
        fsle.load_data(output_path)


def test_load_data_merges_extra_sizes(tmp_path):
    base = _synthetic_data()
    # Keep the REQUIRED_KEYS layout of the default run.
    for key in fsle.REQUIRED_KEYS:
        base.setdefault(key, np.array([0.0]))
    base.update({f"chaos_{k}": np.array([v]) for k, v in fsle.CHAOS.items()})
    path = tmp_path / "gcm_collective_fsle.npz"
    np.savez_compressed(path, **base)
    taus = np.full((4, fsle.DELTAS.size), 7, dtype=np.int64)
    np.savez_compressed(
        tmp_path / "gcm_collective_fsle_N10000000.npz",
        chaos_tau_N10000000=taus,
        chaos_lam_m_N10000000=np.array([0.41]),
        n_samples=np.array([4]),
    )

    merged = fsle.load_data(path)

    np.testing.assert_array_equal(merged["Ns"], [10_000, 100_000, 1_000_000, 10_000_000])
    np.testing.assert_array_equal(merged["chaos_tau_N10000000"], taus)
    # The extra file's n_samples must not replace the base value.
    np.testing.assert_array_equal(merged["n_samples"], base["n_samples"])


def test_load_data_requires_the_base_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        fsle.load_data(tmp_path / "gcm_collective_fsle.npz")


def test_main_estimate_prints_cost_and_exits(capsys):
    fsle.main(["--estimate", "2000"])

    out = capsys.readouterr().out
    assert "N=2e+03" in out
    assert "ms per two-copy step" in out


def test_main_plots_from_cache(monkeypatch, tmp_path):
    base = _synthetic_data()
    for key in fsle.REQUIRED_KEYS:
        base.setdefault(key, np.array([0.0]))
    path = tmp_path / "gcm_collective_fsle.npz"
    np.savez_compressed(path, **base)
    monkeypatch.setattr(fsle, "FIG_DIR", tmp_path)
    monkeypatch.setattr(fsle, "OUTPUT_NPZ", path)
    monkeypatch.setattr(fsle, "OUTPUT_PNG", tmp_path / "gcm_collective_fsle.png")

    fsle.main([])

    assert (tmp_path / "gcm_collective_fsle.png").stat().st_size > 0


def test_registry_contract_matches_module_keys():
    spec = get_section("sec10_gcm")

    assert "dynachaos.cml.gcm_collective_fsle" in spec.modules
    assert "gcm_collective_fsle.npz" in spec.cache_files
    assert "gcm_collective_fsle.png" in spec.output_files
    assert set(spec.required_npz_keys("gcm_collective_fsle.npz")) == set(fsle.REQUIRED_KEYS)


# ---------------------------------------------------------------------------
# Physics of the committed cache (Shibata & Kaneko 1998)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cached():
    # The base cache plus the N = 1e7 set stored beside it.
    return fsle.load_data(CACHE)


def test_cached_lambda_m_equals_a_fresh_tangent_map_measurement(cached):
    n = 10_000
    a = fsle.a_profile(1.9, 0.05, n)
    rng = np.random.default_rng(123)
    x = rng.uniform(-1.0, 1.0, n)

    fresh = fsle.lambda_m(a, 0.098, x, n_transient=2000, n_measure=10_000, seed=3)

    assert float(cached[f"chaos_lam_m_N{n}"][0]) == pytest.approx(fresh, abs=0.01)
    # The microscopic exponent does not depend on N.
    lam = [float(cached[f"chaos_lam_m_N{N}"][0]) for N in (10_000, 100_000, 1_000_000)]
    assert max(lam) - min(lam) < 0.01


@pytest.mark.parametrize("N", [10_000, 100_000, 1_000_000, 10_000_000])
def test_cached_fsle_plateau_below_1e4_equals_lambda_m(cached, N):
    n_max = int(cached["n_max"][0])
    lam_m = float(cached[f"chaos_lam_m_N{N}"][0])

    delta, curve = fsle.fsle_curve(cached[f"chaos_tau_N{N}"], DELTA0, n_max)
    plateau = curve[delta < 1e-4].mean()

    assert plateau == pytest.approx(lam_m, abs=0.02)


@pytest.mark.parametrize("N", [10_000, 100_000])
def test_cached_fsle_plateau_below_1e3_equals_lambda_m(cached, N):
    # From N = 1e6 on the microscopic scale Delta_m is below 1e-3, so that range is
    # not a plateau there; the check above uses Delta < 1e-4 for every size.
    n_max = int(cached["n_max"][0])
    lam_m = float(cached[f"chaos_lam_m_N{N}"][0])

    delta, curve = fsle.fsle_curve(cached[f"chaos_tau_N{N}"], DELTA0, n_max)
    plateau = curve[delta < 1e-3].mean()

    assert plateau == pytest.approx(lam_m, abs=0.02)


def test_cached_collective_exponent_is_small_and_positive_at_1e5(cached):
    analysis = fsle.analyse(cached)

    lam_c = analysis["fits"][100_000][0][0]
    assert 0.015 <= lam_c <= 0.03
    # lambda_C is a factor of about 20 below the microscopic exponent at every size.
    for N, ((lc, _), _, lam_m) in analysis["fits"].items():
        assert 0.0 < lc < 0.1 * lam_m, N


def test_cached_microscopic_scale_decreases_with_size(cached):
    analysis = fsle.analyse(cached)

    sizes = sorted(analysis["fits"])
    log_dm = [analysis["fits"][N][0][1] for N in sizes]
    assert np.all(np.diff(log_dm) < 0.0)
    assert -0.65 < analysis["slope"] < -0.3


def test_cached_identical_element_control_plateau_equals_lambda_m(cached):
    n_max = int(cached["n_max"][0])
    for N in (100_000, 1_000_000):
        delta, curve = fsle.fsle_curve(cached[f"control_tau_N{N}"], DELTA0, n_max)
        lam_m = float(cached[f"control_lam_m_N{N}"][0])

        assert curve[delta < 1e-4].mean() == pytest.approx(lam_m, abs=0.03), N
