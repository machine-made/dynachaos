import numpy as np

from dynachaos.diagnostics import on_off_intermittency_figure
from dynachaos.pipelines.registry import get_section


def test_on_off_intermittency_compute_writes_golden_cache(tmp_path):
    output_path = tmp_path / "on_off_intermittency.npz"

    payload = on_off_intermittency_figure.compute(
        output_path,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )

    assert output_path.exists()
    assert tuple(payload) == on_off_intermittency_figure.REQUIRED_KEYS
    np.testing.assert_array_equal(payload["schema_version"], [2])
    np.testing.assert_array_equal(payload["seed"], [20260602])
    np.testing.assert_equal(
        payload["source_file"][0],
        "src/dynachaos/diagnostics/on_off_intermittency_figure.py",
    )
    np.testing.assert_allclose(payload["benchmark_eps"], [0.49])
    np.testing.assert_allclose(payload["benchmark_lambda_perp"], [np.log(0.98)])
    np.testing.assert_equal(payload["benchmark_series"].shape, (39_000,))
    np.testing.assert_equal(payload["benchmark_laminar_mask"].shape, (39_000,))
    np.testing.assert_equal(payload["benchmark_laminar_lengths"].shape, (21,))
    np.testing.assert_equal(payload["benchmark_burst_lengths"].shape, (21,))
    np.testing.assert_equal(payload["benchmark_burst_amplitudes"].shape, (3_900,))
    np.testing.assert_allclose(payload["benchmark_threshold_percentile"], [90.0])
    np.testing.assert_allclose(payload["off_time_alpha"], [-1.65714225])
    np.testing.assert_allclose(payload["off_time_alpha_ci"], [-2.40538453, -1.46694202])
    np.testing.assert_allclose(payload["off_time_gof_p"], [1.0 / 3.0])
    np.testing.assert_allclose(payload["burst_amplitude_alpha"], [-1.02539174])
    np.testing.assert_allclose(
        payload["burst_amplitude_alpha_ci"],
        [-1.02595041, -1.02469471],
    )
    np.testing.assert_allclose(payload["burst_amplitude_gof_p"], [0.0])
    np.testing.assert_allclose(payload["mean_off_beta"], [0.96042981])
    np.testing.assert_allclose(
        payload["scaling_eps_values"],
        [0.45, 0.46, 0.47, 0.48, 0.49, 0.495],
    )
    np.testing.assert_allclose(
        payload["lambda_abs_values"],
        np.abs(np.log(2.0 * payload["scaling_eps_values"])),
    )
    np.testing.assert_equal(payload["skew_driver_series"].shape, (39_000,))
    np.testing.assert_equal(payload["skew_transverse_series"].shape, (39_000,))
    np.testing.assert_array_less(-1.8, payload["off_time_alpha"])
    np.testing.assert_array_less(payload["off_time_alpha"], -1.3)
    np.testing.assert_array_less(payload["off_time_alpha_ci"][0], payload["off_time_alpha"])
    np.testing.assert_array_less(payload["off_time_alpha"], payload["off_time_alpha_ci"][1])
    np.testing.assert_array_less(0.0, payload["off_time_gof_p"] + 1e-12)
    np.testing.assert_array_less(payload["off_time_gof_p"], 1.0 + 1e-12)
    np.testing.assert_array_less(-1.2, payload["burst_amplitude_alpha"])
    np.testing.assert_array_less(payload["burst_amplitude_alpha"], -0.8)
    np.testing.assert_array_less(
        payload["burst_amplitude_alpha_ci"][0],
        payload["burst_amplitude_alpha"],
    )
    np.testing.assert_array_less(
        payload["burst_amplitude_alpha"],
        payload["burst_amplitude_alpha_ci"][1],
    )
    np.testing.assert_array_less(0.0, payload["burst_amplitude_gof_p"] + 1e-12)
    np.testing.assert_array_less(payload["burst_amplitude_gof_p"], 1.0 + 1e-12)
    np.testing.assert_array_less(0.7, payload["mean_off_beta"])
    np.testing.assert_array_less(payload["mean_off_beta"], 1.3)
    np.testing.assert_array_less(payload["lambda_abs_values"][-1], payload["lambda_abs_values"][0])
    np.testing.assert_array_less(payload["mean_off_lengths"][0], payload["mean_off_lengths"][-1])
    np.testing.assert_array_less(
        10.0 * np.median(payload["skew_transverse_series"]),
        np.max(payload["skew_transverse_series"]),
    )


def test_on_off_intermittency_plot_writes_png(tmp_path):
    payload = on_off_intermittency_figure.compute(
        None,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )
    output_path = tmp_path / "on_off_intermittency.png"

    result = on_off_intermittency_figure.plot(payload, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_on_off_intermittency_registry_contract_matches_module_keys():
    spec = get_section("sec12_intermittency")
    contract_keys = spec.required_npz_keys("on_off_intermittency.npz")

    assert "dynachaos.diagnostics.on_off_intermittency_figure" in spec.modules
    assert "on_off_intermittency.npz" in spec.cache_files
    assert "on_off_intermittency.png" in spec.output_files
    assert set(contract_keys).issubset(on_off_intermittency_figure.REQUIRED_KEYS)
