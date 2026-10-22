import numpy as np
import pytest

from dynachaos.diagnostics import type_ii_intermittency_figure
from dynachaos.maps.intermittency import pm_type_ii_oracle
from dynachaos.pipelines.registry import get_section


def test_type_ii_intermittency_compute_writes_golden_cache(tmp_path):
    output_path = tmp_path / "type_ii_intermittency.npz"

    payload = type_ii_intermittency_figure.compute(
        output_path,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )

    assert output_path.exists()
    assert tuple(payload) == type_ii_intermittency_figure.REQUIRED_KEYS
    np.testing.assert_array_equal(payload["schema_version"], [2])
    np.testing.assert_array_equal(payload["seed"], [20260602])
    np.testing.assert_equal(
        payload["source_file"][0],
        "src/dynachaos/diagnostics/type_ii_intermittency_figure.py",
    )
    np.testing.assert_allclose(payload["eps"], [2e-3])
    np.testing.assert_allclose(payload["a"], [1.0])
    np.testing.assert_allclose(payload["theta"], [0.17])
    np.testing.assert_allclose(payload["escape_threshold"], [0.35])
    np.testing.assert_equal(payload["spiral_orbit"].shape, (1555, 2))
    np.testing.assert_equal(payload["spiral_radius"].shape, (1555,))
    np.testing.assert_array_equal(payload["spiral_escape_index"], [1555])
    np.testing.assert_equal(payload["reinjection_radii"].shape, (1_200,))
    np.testing.assert_equal(payload["laminar_lengths"].shape, (1_200,))
    np.testing.assert_equal(payload["laminar_histogram_edges"].shape, (49,))
    np.testing.assert_equal(payload["laminar_histogram_density"].shape, (48,))
    np.testing.assert_allclose(payload["laminar_tail_alpha"], [-3.23082964])
    np.testing.assert_allclose(payload["laminar_tail_alpha_ci"], [-3.32929902, -3.10493551])
    np.testing.assert_allclose(payload["laminar_tail_gof_p"], [0.0])
    np.testing.assert_allclose(payload["exponential_rate"], [0.0019453190566928037])
    np.testing.assert_allclose(payload["exponential_intercept"], [-5.345020077030128])
    np.testing.assert_allclose(payload["exponential_rvalue"], [-0.9411751253152798])
    np.testing.assert_array_less(400, np.min(payload["laminar_lengths"]))
    np.testing.assert_array_less(np.max(payload["laminar_lengths"]), 2_000)
    np.testing.assert_array_less(-4.0, payload["laminar_tail_alpha"])
    np.testing.assert_array_less(payload["laminar_tail_alpha"], -2.0)
    np.testing.assert_array_less(payload["laminar_tail_alpha_ci"][0], payload["laminar_tail_alpha"])
    np.testing.assert_array_less(payload["laminar_tail_alpha"], payload["laminar_tail_alpha_ci"][1])
    np.testing.assert_array_less(0.0, payload["laminar_tail_gof_p"] + 1e-12)
    np.testing.assert_array_less(payload["laminar_tail_gof_p"], 1.0 + 1e-12)
    np.testing.assert_array_less(0.0, payload["exponential_rate"])
    np.testing.assert_array_less(payload["exponential_rvalue"], -0.7)
    np.testing.assert_array_less(payload["spiral_radius"][0], payload["escape_threshold"])
    np.testing.assert_array_less(payload["escape_threshold"], payload["spiral_radius"][-1])


def test_type_ii_spiral_orbit_matches_oracle_prefix():
    payload = type_ii_intermittency_figure.compute(
        None,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )
    n = int(payload["spiral_escape_index"][0])
    expected = pm_type_ii_oracle(
        n,
        x0=2e-3,
        y0=0.0,
        eps=float(payload["eps"][0]),
        a=float(payload["a"][0]),
        theta=float(payload["theta"][0]),
    )

    np.testing.assert_allclose(payload["spiral_orbit"], expected)


def test_type_ii_intermittency_plot_writes_png(tmp_path):
    payload = type_ii_intermittency_figure.compute(
        None,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )
    output_path = tmp_path / "type_ii_intermittency.png"

    result = type_ii_intermittency_figure.plot(payload, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_type_ii_caption_note_is_honest_about_normal_form_scope():
    note = type_ii_intermittency_figure.CAPTION_NOTE
    required_fragments = (
        "Normal-form Type-II demonstration",
        "clean physical exemplars are scarce",
        "p-n diode",
        "forced jet",
    )

    missing = [fragment for fragment in required_fragments if fragment not in note]
    np.testing.assert_equal(missing, [])


def test_type_ii_escape_lengths_reject_censored_laminar_runs():
    with pytest.raises(RuntimeError, match="max_steps before escape"):
        type_ii_intermittency_figure._escape_lengths(
            np.array([1e-7]),
            eps=1e-3,
            a=1.0,
            theta=float(np.sqrt(5.0)),
            escape_threshold=0.35,
        )


def test_type_ii_intermittency_registry_contract_matches_module_keys():
    spec = get_section("sec12_intermittency")
    contract_keys = spec.required_npz_keys("type_ii_intermittency.npz")

    assert "dynachaos.diagnostics.type_ii_intermittency_figure" in spec.modules
    assert "type_ii_intermittency.npz" in spec.cache_files
    assert "type_ii_intermittency.npz" in spec.output_files
    assert "type_ii_intermittency.png" in spec.output_files
    assert tuple(contract_keys) == type_ii_intermittency_figure.REQUIRED_KEYS
