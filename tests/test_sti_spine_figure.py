import numpy as np
import pytest

from dynachaos.cml import sti_spine_figure
from dynachaos.pipelines.registry import get_section


def test_sti_spine_compute_writes_golden_cache(tmp_path):
    output_path = tmp_path / "sti_spine.npz"

    payload = sti_spine_figure.compute(output_path)

    assert output_path.exists()
    assert tuple(payload) == sti_spine_figure.REQUIRED_KEYS
    np.testing.assert_array_equal(payload["schema_version"], [3])
    np.testing.assert_array_equal(payload["seed"], [20260602])
    np.testing.assert_equal(payload["source_file"][0], "src/dynachaos/cml/sti_spine_figure.py")
    np.testing.assert_allclose(payload["model_a_parameter"], [-0.01])
    np.testing.assert_allclose(payload["display_eps"], [0.08])
    np.testing.assert_equal(payload["spacetime"].shape, (480, 512))
    np.testing.assert_equal(payload["turbulent_mask"].shape, (480, 512))
    np.testing.assert_allclose(payload["sweep_eps"], np.linspace(0.03, 0.12, 19))
    np.testing.assert_equal(payload["turbulent_fraction"].shape, (19,))
    # Kaneko (1985) model-A STI with a=-0.01: sub-critical coupling is laminar
    # (no bursts), super-critical coupling is burst-dominated, with a sharp onset
    # in between (the eps sweep brackets the transition).
    turbulent_fraction = payload["turbulent_fraction"]
    np.testing.assert_array_less(turbulent_fraction[0], 0.05)
    np.testing.assert_array_less(0.5, turbulent_fraction[-1])
    np.testing.assert_array_less(0.7, turbulent_fraction.max())
    # The critical coupling lies between the last laminar and the first turbulent sweep point.
    eps_c = float(payload["eps_c"][0])
    first_turbulent = int(np.argmax(turbulent_fraction > 0.001))
    assert (
        payload["sweep_eps"][first_turbulent - 1] <= eps_c <= payload["sweep_eps"][first_turbulent]
    )
    np.testing.assert_allclose(payload["fit_eps_values"], [eps_c, 0.060, 0.080])
    # One laminar-length distribution and one fit per fitted coupling.
    for k in range(3):
        probs = payload[f"laminar_probs_{k}"]
        # eps_c = 0.05 is a sweep-grid point whose lattice is fully laminar (one run of length L).
        assert payload[f"laminar_values_{k}"].size >= (1 if k == 0 else 2)
        assert probs.sum() == pytest.approx(1.0)
    assert payload["fit_n_tail"].shape == (3,)
    assert np.all(payload["fit_n_tail"] > 20)
    assert np.all(payload["fit_lam_exp"] > 0.0)
    assert set(payload["fit_preferred"]) <= {"power_law", "exponential", "truncated_power_law"}


def test_sti_spine_locates_critical_coupling_by_interpolation():
    eps = np.array([0.1, 0.2, 0.3, 0.4])

    # Threshold 0.001 is crossed between 0.2 (phi = 0) and 0.3 (phi = 0.5).
    located = sti_spine_figure.locate_critical_coupling(eps, [0.0, 0.0, 0.5, 0.8])
    assert located == pytest.approx(round(0.2 + 0.001 * 0.1 / 0.5, 3))
    # No onset in the sweep, or a first point that is already turbulent: no estimate.
    assert np.isnan(sti_spine_figure.locate_critical_coupling(eps, [0.0, 0.0, 0.0, 0.0]))
    assert np.isnan(sti_spine_figure.locate_critical_coupling(eps, [0.3, 0.5, 0.6, 0.8]))


def test_sti_spine_tail_fit_recovers_exponential_tail():
    rng = np.random.default_rng(3)
    lam = 0.5
    # P(x) proportional to exp(-lam x) for x >= 2: a shifted geometric distribution.
    lengths = rng.geometric(1.0 - np.exp(-lam), size=20_000) + 1

    fit = sti_spine_figure.fit_tail_models(lengths, x_min=2)

    assert fit["n_tail"] == lengths.size
    assert fit["lam_exp"] == pytest.approx(lam, rel=0.05)
    assert fit["preferred"] == "exponential"
    # ks_* uses the rank ECDF at each sample, so tied integer lengths inflate it
    # (as in the cited paper values); it still ranks the true model first.
    assert fit["ks_exp"] < fit["ks_pl"]
    assert fit["bic_exp"] < fit["bic_pl"]


def test_sti_spine_tail_fit_recovers_power_law_tail():
    rng = np.random.default_rng(4)
    alpha = 2.2
    support = np.arange(2, 400)
    pmf = support ** (-alpha) / np.sum(support ** (-alpha))
    lengths = rng.choice(support, size=20_000, p=pmf)

    fit = sti_spine_figure.fit_tail_models(lengths, x_min=2)

    assert fit["alpha_pl"] == pytest.approx(alpha, rel=0.03)
    assert fit["preferred"] in {"power_law", "truncated_power_law"}
    assert fit["bic_pl"] < fit["bic_exp"]


def test_sti_spine_tail_fit_returns_empty_dict_for_short_tail():
    assert sti_spine_figure.fit_tail_models(np.array([1, 2, 2, 3, 5]), x_min=2) == {}


def test_sti_spine_plot_writes_png(tmp_path):
    payload = sti_spine_figure.compute(
        None,
        n_sites=128,
        n_transient=200,
        n_record=120,
        sweep_n_transient=200,
        sweep_n_record=60,
        fit_n_transient=200,
        fit_n_record=150,
    )
    output_path = tmp_path / "sti_spine.png"

    result = sti_spine_figure.plot(payload, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_sti_spine_laminar_cluster_runs_wrap_periodic_rows():
    mask = np.array([[True, True, False, True], [False, True, True, False]])

    lengths = sti_spine_figure._periodic_run_lengths(mask)

    np.testing.assert_array_equal(np.sort(lengths), [2, 3])


def test_sti_spine_laminar_cluster_runs_reject_non_spacetime_mask():
    with pytest.raises(ValueError, match="2D spacetime"):
        sti_spine_figure._periodic_run_lengths(np.array([True, False]))


def test_sti_spine_registry_contract_matches_module_keys():
    spec = get_section("sec12_intermittency")
    contract_keys = spec.required_npz_keys("sti_spine.npz")

    assert "dynachaos.cml.sti_spine_figure" in spec.modules
    assert "sti_spine.npz" in spec.cache_files
    assert "sti_spine.npz" in spec.output_files
    assert "sti_spine.png" in spec.output_files
    assert tuple(contract_keys) == sti_spine_figure.REQUIRED_KEYS
