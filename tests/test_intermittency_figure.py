import numpy as np
from conftest import is_reference_platform

from dynachaos.diagnostics import intermittency_figure
from dynachaos.pipelines.registry import get_section


def test_intermittency_figure_compute_writes_golden_cache(tmp_path):
    output_path = tmp_path / "type_i_intermittency.npz"

    payload = intermittency_figure.compute(
        output_path,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )

    assert output_path.exists()
    assert tuple(payload) == intermittency_figure.REQUIRED_KEYS
    np.testing.assert_array_equal(payload["schema_version"], [6])
    np.testing.assert_array_equal(payload["seed"], [20260601])
    np.testing.assert_equal(
        payload["source_file"][0],
        "src/dynachaos/diagnostics/intermittency_figure.py",
    )
    np.testing.assert_array_less(payload["logistic_tail_r"], payload["logistic_mechanism_r"])
    np.testing.assert_array_equal(payload["logistic_period"], [3])
    np.testing.assert_allclose(payload["logistic_laminar_percentile"], [70.0])
    np.testing.assert_equal(payload["logistic_f3_return_points"].shape, (4000, 2))
    np.testing.assert_equal(
        _count_x_clusters(payload["logistic_f3_channel_points"]),
        3,
    )
    np.testing.assert_equal(payload["normal_form_eps"].size, 8)
    np.testing.assert_equal(payload["normal_form_mean_lengths"].size, 8)
    # Counts of laminar phases and Poincare crossings, and the fitted exponents
    # derived from them, come out of chaotic trajectories (logistic period-3
    # channel, Lorenz return map). They are pinned to values measured on the
    # reference architecture; elsewhere a last-bit difference relocates a
    # crossing and the counts shift (observed on macOS arm64: 276 Lorenz return
    # points against 274). The physics assertions below hold on every platform.
    if is_reference_platform():
        np.testing.assert_equal(payload["logistic_laminar_lengths"].size, 6228)
        np.testing.assert_equal(payload["logistic_f3_channel_points"].shape, (177, 2))
        np.testing.assert_equal(payload["lorenz_return_points"].shape, (274, 2))
        np.testing.assert_equal(payload["lorenz_channel_points"].shape, (82, 2))
        np.testing.assert_allclose(payload["type_i_tail_alpha"], [-1.50748972])
        np.testing.assert_allclose(payload["type_i_tail_alpha_ci"], [-1.52037343, -1.49315214])
        np.testing.assert_allclose(payload["type_i_tail_gof_p"], [2.0 / 3.0])
        np.testing.assert_allclose(payload["type_i_vuong_z"], [1.31575131])
        np.testing.assert_allclose(payload["logistic_f3_channel_slope"], [1.00042829])
        np.testing.assert_allclose(payload["normal_form_beta"], [0.4919801])
        np.testing.assert_allclose(payload["lorenz_channel_slope"], [0.98549932])
    else:
        # Still assert the structure is sound and the sets are non-degenerate.
        assert payload["logistic_laminar_lengths"].size > 1000
        assert payload["logistic_f3_channel_points"].shape[1] == 2
        assert payload["lorenz_return_points"].shape[1] == 2
        assert payload["lorenz_channel_points"].shape[1] == 2
        assert payload["lorenz_return_points"].shape[0] > 100
    np.testing.assert_array_less(-1.7, payload["type_i_tail_alpha"])
    np.testing.assert_array_less(payload["type_i_tail_alpha"], -1.3)
    np.testing.assert_array_less(payload["type_i_tail_alpha_ci"][0], payload["type_i_tail_alpha"])
    np.testing.assert_array_less(payload["type_i_tail_alpha"], payload["type_i_tail_alpha_ci"][1])
    np.testing.assert_array_less(0.0, payload["type_i_tail_gof_p"])
    np.testing.assert_array_less(payload["type_i_tail_gof_p"], 1.0)
    np.testing.assert_array_less(0.0, payload["type_i_vuong_z"])
    np.testing.assert_array_less(0.35, payload["normal_form_beta"])
    np.testing.assert_array_less(payload["normal_form_beta"], 0.65)
    # lorenz_channel_slope was asserted to lie in [0.8, 1.2] on the reasoning
    # that type-I intermittency makes the return map tangent to the diagonal in
    # the laminar channel, so the fitted slope should sit near 1. The physics is
    # right; this estimator does not resolve it at this configuration, and the
    # band held on the reference machine by luck rather than by construction.
    #
    # Measured. At the shipped settings (t_max=80, channel_percentile=30, ~274
    # extracted maxima) perturbing the Lorenz initial condition by 1e-12 -- the
    # scale of cross-platform last-bit divergence -- moves the fitted slope
    # across [0.617, 1.540]. The committed 0.98549932 is one draw from that
    # distribution. CI runners land outside the band; 2.048 was observed.
    #
    # More data does not rescue it: at t_max=1500 the spread across the same
    # perturbations is still 0.48. And the falsifying check fails outright --
    # if the fit were measuring the tangency, narrowing the channel would drive
    # the slope to 1, but at t_max=1500 it runs 1.223, 0.519, 0.652, 0.725,
    # 0.850, 0.822, 0.819 for percentiles 30 down to 0.5. Non-monotonic, and
    # converging near 0.82 rather than 1.
    #
    # The same estimator returns 1.00042829 for logistic_f3_channel_slope, a
    # deterministic 1-D map, so the estimator is not simply broken -- roughly
    # 274 maxima off a chaotic ODE do not determine the channel. Until the fit
    # is reworked, assert what the measurement supports: the channel was found
    # and the fit produced a finite, positive slope. Widening the band instead
    # would keep the appearance of a physics check while asserting nothing.
    assert payload["lorenz_channel_slope"].shape == (1,)
    slope = float(payload["lorenz_channel_slope"][0])
    assert np.isfinite(slope), f"channel slope must be finite, got {slope!r}"
    assert slope > 0.0, (
        f"channel slope must be positive -- the laminar channel runs alongside "
        f"the diagonal, so a non-positive slope means the fit did not find it "
        f"(got {slope!r})"
    )
    assert payload["lorenz_channel_points"].shape[1] == 2
    assert payload["lorenz_channel_points"].shape[0] >= 5, (
        "min_channel_points=5 was requested, so fewer than 5 means the channel "
        "extraction silently returned an under-determined set"
    )


def test_intermittency_figure_plot_writes_png(tmp_path):
    payload = intermittency_figure.compute(
        None,
        powerlaw_gof_bootstrap=3,
        alpha_ci_bootstrap=20,
    )
    output_path = tmp_path / "type_i_intermittency.png"

    result = intermittency_figure.plot(payload, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_intermittency_figure_registry_contract_matches_module_keys():
    spec = get_section("sec12_intermittency")
    contract_keys = spec.required_npz_keys("type_i_intermittency.npz")

    assert "dynachaos.diagnostics.intermittency_figure" in spec.modules
    assert "type_i_intermittency.npz" in spec.cache_files
    assert "type_i_intermittency.npz" in spec.output_files
    assert "type_i_intermittency.png" in spec.output_files
    assert set(contract_keys).issubset(intermittency_figure.REQUIRED_KEYS)


def _count_x_clusters(points):
    x = np.sort(np.asarray(points)[:, 0])
    return 1 + int(np.count_nonzero(np.diff(x) > 0.05))
