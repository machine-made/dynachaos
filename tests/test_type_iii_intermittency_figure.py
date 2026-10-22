import numpy as np
import pytest

from dynachaos.diagnostics import type_iii_intermittency_figure
from dynachaos.maps.intermittency import pm_type_iii_oracle
from dynachaos.pipelines.registry import get_section


def test_type_iii_intermittency_compute_writes_golden_cache(tmp_path):
    output_path = tmp_path / "type_iii_intermittency.npz"

    payload = type_iii_intermittency_figure.compute(
        output_path,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )

    assert output_path.exists()
    assert tuple(payload) == type_iii_intermittency_figure.REQUIRED_KEYS
    np.testing.assert_array_equal(payload["schema_version"], [2])
    np.testing.assert_array_equal(payload["seed"], [20260602])
    np.testing.assert_equal(
        payload["source_file"][0],
        "src/dynachaos/diagnostics/type_iii_intermittency_figure.py",
    )
    np.testing.assert_allclose(payload["eps"], [2e-3])
    np.testing.assert_allclose(payload["a"], [1.0])
    np.testing.assert_allclose(payload["escape_threshold"], [0.35])
    np.testing.assert_equal(payload["return_grid"].shape, (2_000,))
    np.testing.assert_equal(payload["f2_return_points"].shape, (2_000, 2))
    np.testing.assert_equal(payload["reinjection_points"].shape, (1_200,))
    np.testing.assert_equal(payload["series"].shape, (25_000,))
    np.testing.assert_equal(payload["laminar_mask"].shape, (25_000,))
    np.testing.assert_equal(payload["laminar_lengths"].shape, (1_200,))
    np.testing.assert_allclose(payload["laminar_tail_alpha"], [-5.09301742])
    np.testing.assert_allclose(payload["laminar_tail_alpha_ci"], [-5.38442901, -4.80155101])
    np.testing.assert_allclose(payload["laminar_tail_gof_p"], [0.0])
    np.testing.assert_equal(payload["rpd_thresholds"].shape, (1_200,))
    np.testing.assert_equal(payload["rpd_conditional_means"].shape, (1_200,))
    np.testing.assert_allclose(payload["f2_linear_slope"], [1.00397433])
    np.testing.assert_allclose(payload["f2_cubic_coefficient"], [2.02958326])
    np.testing.assert_allclose(payload["rpd_slope"], [0.48240774])
    np.testing.assert_allclose(payload["rpd_intercept"], [0.0], atol=1e-6)
    np.testing.assert_allclose(payload["rpd_alpha"], [-0.06797727])
    np.testing.assert_allclose(payload["rpd_rvalue"], [0.9998407])
    np.testing.assert_array_less(1.0, payload["f2_linear_slope"])
    np.testing.assert_array_less(1.0, payload["f2_cubic_coefficient"])
    np.testing.assert_array_less(1_000, np.min(payload["laminar_lengths"]))
    np.testing.assert_array_less(np.max(payload["laminar_lengths"]), 4_999)
    np.testing.assert_array_less(-6.0, payload["laminar_tail_alpha"])
    np.testing.assert_array_less(payload["laminar_tail_alpha"], -4.0)
    np.testing.assert_array_less(payload["laminar_tail_alpha_ci"][0], payload["laminar_tail_alpha"])
    np.testing.assert_array_less(payload["laminar_tail_alpha"], payload["laminar_tail_alpha_ci"][1])
    np.testing.assert_array_less(0.0, payload["laminar_tail_gof_p"] + 1e-12)
    np.testing.assert_array_less(payload["laminar_tail_gof_p"], 1.0 + 1e-12)
    np.testing.assert_array_less(0.3, payload["rpd_slope"])
    np.testing.assert_array_less(payload["rpd_slope"], 0.7)
    np.testing.assert_array_less(0.99, payload["rpd_rvalue"])
    assert np.isfinite(payload["rpd_alpha"][0])


def test_type_iii_two_step_return_matches_oracle_at_grid_points():
    payload = type_iii_intermittency_figure.compute(
        None,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )
    points = payload["f2_return_points"][::251]
    eps = float(payload["eps"][0])
    a = float(payload["a"][0])

    expected = []
    for x0 in points[:, 0]:
        expected.append(pm_type_iii_oracle(2, x0=float(x0), eps=eps, a=a)[-1])

    np.testing.assert_allclose(points[:, 1], expected)


def test_type_iii_intermittency_plot_writes_png(tmp_path):
    payload = type_iii_intermittency_figure.compute(
        None,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )
    output_path = tmp_path / "type_iii_intermittency.png"

    result = type_iii_intermittency_figure.plot(payload, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_type_iii_escape_episodes_reject_censored_laminar_runs():
    with pytest.raises(RuntimeError, match="max_steps before escape"):
        type_iii_intermittency_figure._escape_episodes(
            np.array([1e-7]),
            eps=2e-3,
            a=1.0,
            escape_threshold=0.35,
        )


def test_type_iii_intermittency_registry_contract_matches_module_keys():
    spec = get_section("sec12_intermittency")
    contract_keys = spec.required_npz_keys("type_iii_intermittency.npz")

    assert "dynachaos.diagnostics.type_iii_intermittency_figure" in spec.modules
    assert "type_iii_intermittency.npz" in spec.cache_files
    assert "type_iii_intermittency.npz" in spec.output_files
    assert "type_iii_intermittency.png" in spec.output_files
    assert tuple(contract_keys) == type_iii_intermittency_figure.REQUIRED_KEYS
