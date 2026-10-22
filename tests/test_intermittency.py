import numpy as np
import pytest

from dynachaos.diagnostics.correlation import fit_power_law_loglog
from dynachaos.diagnostics.intermittency import (
    BurstAmplitudeDistribution,
    CandidateScalingLaw,
    DiagnosticLogLogFit,
    ExtremaReturnMap,
    IntermittencySummary,
    LaminarBurstSymmetry,
    LaminarLengthDistribution,
    MeanLaminarScaling,
    ReinjectionMx,
    ReturnMapReconstruction,
    TangentChannel,
    burst_amplitude_distribution,
    compare_powerlaw_exponential,
    detect_laminar_phases,
    extrema_return_map,
    fit_exponential,
    fit_power_law_mle,
    intermittency_summary,
    laminar_burst_symmetry,
    laminar_length_distribution,
    mean_laminar_scaling,
    near_diagonal_tangent_channel,
    pooled_laminar_lengths,
    powerlaw_alpha_ci,
    powerlaw_gof,
    reinjection_Mx,
    return_map_reconstruction,
)
from dynachaos.diagnostics.intermittency import (
    fit_power_law_loglog as fit_intermittency_power_law_loglog,
)
from dynachaos.maps.intermittency import (
    logistic_type_i_oracle,
    lorenz_1662_oracle,
    on_off_oracle,
    pm_type_i_oracle,
)


def test_recurrence_laminar_detection_reuses_vertical_lengths():
    signal = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    mask, lengths = detect_laminar_phases(signal, method="recurrence", eps=0.0, v_min=3)

    np.testing.assert_array_equal(lengths, np.array([3, 3, 3]))
    np.testing.assert_array_equal(mask, np.array([False, False, True, True, True]))


def test_period_laminar_detection_estimates_period_and_threshold():
    signal = np.array([0.0, 1.0, 0.0, 1.02, 0.01, 1.01, 0.6, -0.4])

    mask, lengths = detect_laminar_phases(signal, method="period", percentile=50)

    assert mask.dtype == np.bool_
    np.testing.assert_equal(mask.shape, signal.shape)
    assert lengths.size > 0
    assert np.all(lengths >= 1)


def test_variance_laminar_detection_estimates_window_and_threshold():
    signal = np.r_[np.ones(8), np.linspace(-1.0, 1.0, 8), np.ones(8)]

    mask, lengths = detect_laminar_phases(signal, method="variance", percentile=40)

    assert mask.dtype == np.bool_
    np.testing.assert_equal(mask.shape, signal.shape)
    assert lengths.size > 0
    assert np.all(lengths >= 1)


def test_laminar_length_distribution_returns_fd_bins_and_discrete_counts():
    lengths = np.array([1, 1, 2, 4, 4, 4, 8])

    dist = laminar_length_distribution(lengths)

    assert isinstance(dist, LaminarLengthDistribution)
    assert dist.bin_edges.size >= 2
    np.testing.assert_array_equal(dist.values, np.array([1, 2, 4, 8]))
    np.testing.assert_array_equal(dist.counts, np.array([2, 1, 3, 1]))
    assert np.sum(dist.probabilities) == pytest.approx(1.0)


def test_type_i_logistic_period_lengths_have_negative_loglog_tail():
    signal = logistic_type_i_oracle(8000, x0=0.2)

    mask, lengths = detect_laminar_phases(signal, method="period", percentile=10)
    dist = laminar_length_distribution(lengths)
    slope, _, _, _, scaling = fit_power_law_loglog(dist.values, dist.probabilities, min_points=3)

    np.testing.assert_equal(mask.shape, signal.shape)
    assert lengths.size >= 20
    assert np.count_nonzero(scaling) >= 3
    assert slope < -0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": [1.0], "method": "period"}, "at least two"),
        ({"x": [1.0, np.nan], "method": "period"}, "finite"),
        ({"x": [1.0, 2.0], "method": "unknown"}, "method"),
        ({"x": [1.0, 2.0], "method": "period", "period": 2}, "shorter"),
    ],
)
def test_detect_laminar_phases_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        detect_laminar_phases(**kwargs)


def test_laminar_length_distribution_rejects_nonpositive_lengths():
    with pytest.raises(ValueError, match="positive"):
        laminar_length_distribution([1, 0, 2])


def test_laminar_length_distribution_drop_final_censored_is_opt_in():
    lengths = np.array([2, 4, 4, 8, 16], dtype=np.int64)

    default_dist = laminar_length_distribution(lengths)
    explicit_default = laminar_length_distribution(lengths, drop_final_censored=False)
    corrected = laminar_length_distribution(lengths, drop_final_censored=True)

    np.testing.assert_array_equal(default_dist.values, explicit_default.values)
    np.testing.assert_array_equal(default_dist.counts, explicit_default.counts)
    np.testing.assert_array_equal(corrected.values, [2, 4, 8])
    np.testing.assert_array_equal(corrected.counts, [1, 2, 1])


def test_detect_laminar_phases_can_drop_terminal_censored_run():
    signal = np.array([0.0, 1.0, 0.0, 1.02, 0.01, 1.01, 0.0, 1.0])

    mask, default_lengths = detect_laminar_phases(signal, method="period", period=2, percentile=50)
    _, corrected_lengths = detect_laminar_phases(
        signal,
        method="period",
        period=2,
        percentile=50,
        drop_final_censored=True,
    )

    np.testing.assert_equal(mask[-1], True)
    np.testing.assert_array_equal(corrected_lengths, default_lengths[:-1])


def test_fit_power_law_mle_recovers_continuous_type_i_exponent():
    rng = np.random.default_rng(2028)
    csn_alpha = 1.5
    samples = (1.0 - rng.random(20_000)) ** (-1.0 / (csn_alpha - 1.0))

    fit = fit_power_law_mle(samples, discrete=False)

    np.testing.assert_equal(fit.discrete, False)
    assert fit.alpha == pytest.approx(-1.5, abs=2.0 * fit.standard_error)
    assert fit.csn_alpha == pytest.approx(1.5, abs=2.0 * fit.standard_error)
    assert fit.x_min >= 1.0
    assert fit.n_tail > 10_000


def test_fit_power_law_mle_drop_final_censored_reduces_truncated_tail_bias():
    rng = np.random.default_rng(0)
    csn_alpha = 1.5
    samples = (1.0 - rng.random(120)) ** (-1.0 / (csn_alpha - 1.0))
    truncated = np.r_[samples, 120.0]

    default_fit = fit_power_law_mle(truncated, discrete=False, min_tail=20)
    explicit_default = fit_power_law_mle(
        truncated,
        discrete=False,
        min_tail=20,
        drop_final_censored=False,
    )
    corrected = fit_power_law_mle(
        truncated,
        discrete=False,
        min_tail=20,
        drop_final_censored=True,
    )

    np.testing.assert_equal(default_fit, explicit_default)
    np.testing.assert_array_less(
        abs(corrected.alpha + csn_alpha),
        abs(default_fit.alpha + csn_alpha),
    )


def test_fit_power_law_mle_discrete_hurwitz_zeta_path_is_finite():
    rng = np.random.default_rng(2029)
    samples = rng.zipf(2.2, size=5000)

    fit = fit_power_law_mle(samples, discrete=True, min_tail=1000)

    np.testing.assert_equal(fit.discrete, True)
    assert fit.csn_alpha > 1.0
    assert fit.alpha < -1.0
    assert fit.n_tail >= 1000
    assert np.isfinite(fit.ks_distance)


def test_fit_power_law_mle_discrete_hurwitz_zeta_path_is_warning_free():
    samples = np.r_[np.ones(500, dtype=np.int64), np.arange(2, 80, dtype=np.int64)]

    with np.errstate(all="raise"):
        fit = fit_power_law_mle(samples, discrete=True, min_tail=20)

    np.testing.assert_equal(fit.discrete, True)
    assert fit.csn_alpha > 1.0
    assert np.isfinite(fit.ks_distance)


def test_powerlaw_gof_does_not_reject_true_power_law():
    rng = np.random.default_rng(2030)
    csn_alpha = 1.5
    samples = (1.0 - rng.random(800)) ** (-1.0 / (csn_alpha - 1.0))
    fit = fit_power_law_mle(samples, discrete=False)

    gof = powerlaw_gof(samples, fit=fit, n_bootstrap=20, rng=2031)

    assert gof.fit == fit
    assert gof.n_bootstrap == 20
    assert 0.0 <= gof.p_value <= 1.0
    assert gof.p_value >= 0.1


def test_powerlaw_alpha_ci_bootstraps_signed_exponent():
    rng = np.random.default_rng(2039)
    csn_alpha = 1.5
    samples = (1.0 - rng.random(1200)) ** (-1.0 / (csn_alpha - 1.0))
    fit = fit_power_law_mle(samples, discrete=False)

    ci = powerlaw_alpha_ci(samples, fit=fit, n_bootstrap=40, rng=2040)

    np.testing.assert_equal(ci.shape, (2,))
    np.testing.assert_array_less(ci[0], fit.alpha)
    np.testing.assert_array_less(fit.alpha, ci[1])
    np.testing.assert_array_less(-1.8, ci[0])
    np.testing.assert_array_less(ci[1], -1.2)


def test_pooled_laminar_lengths_reduce_on_off_exponent_variance():
    seeds = np.arange(2040, 2056, dtype=np.uint32)

    def oracle_factory(seed):
        return on_off_oracle(
            20_000,
            x0=1e-4,
            transverse_lyapunov=0.0,
            noise_scale=0.8,
            seed=seed,
        )

    def alpha_from_lengths(lengths):
        return fit_power_law_mle(lengths, discrete=True).alpha

    single_alphas = np.array(
        [
            alpha_from_lengths(
                pooled_laminar_lengths(
                    oracle_factory,
                    n_runs=1,
                    seed=int(seed),
                    laminar_method="variance",
                    percentile=50.0,
                )
            )
            for seed in seeds
        ],
        dtype=np.float64,
    )
    pooled_alphas = np.array(
        [
            alpha_from_lengths(
                pooled_laminar_lengths(
                    oracle_factory,
                    n_runs=8,
                    seed=int(seed),
                    laminar_method="variance",
                    percentile=50.0,
                )
            )
            for seed in seeds
        ],
        dtype=np.float64,
    )
    first_pool = pooled_laminar_lengths(
        oracle_factory,
        n_runs=8,
        seed=int(seeds[0]),
        laminar_method="variance",
        percentile=50.0,
    )
    second_pool = pooled_laminar_lengths(
        oracle_factory,
        n_runs=8,
        seed=int(seeds[0]),
        laminar_method="variance",
        percentile=50.0,
    )

    np.testing.assert_array_equal(first_pool, second_pool)
    np.testing.assert_array_less(0, first_pool.size)
    np.testing.assert_array_less(np.std(pooled_alphas, ddof=1), 0.5 * np.std(single_alphas, ddof=1))


def test_fit_exponential_recovers_known_rate():
    rng = np.random.default_rng(2032)
    samples = 1.0 + rng.exponential(scale=2.0, size=5000)

    fit = fit_exponential(samples, x_min=1.0)

    assert fit.rate == pytest.approx(0.5, rel=0.06)
    assert fit.scale == pytest.approx(2.0, rel=0.06)
    assert fit.n_tail == samples.size
    assert np.isfinite(fit.log_likelihood)


def test_vuong_comparison_selects_power_law_and_exponential_signs():
    rng = np.random.default_rng(2033)
    power_law = (1.0 - rng.random(8000)) ** (-1.0 / (1.5 - 1.0))
    exponential = 1.0 + rng.exponential(scale=1.5, size=20_000)

    power_law_cmp = compare_powerlaw_exponential(power_law, discrete=False)
    exponential_cmp = compare_powerlaw_exponential(exponential, discrete=False)

    assert power_law_cmp.z > 0.0
    assert power_law_cmp.log_likelihood_ratio > 0.0
    assert exponential_cmp.z < 0.0
    assert exponential_cmp.log_likelihood_ratio < 0.0


def test_intermittency_loglog_fit_is_marked_diagnostic_only():
    x = np.logspace(0.0, 2.0, 40)
    y = x**-1.5

    fit = fit_intermittency_power_law_loglog(x, y, min_points=5)

    assert isinstance(fit, DiagnosticLogLogFit)
    np.testing.assert_equal(fit.diagnostic_only, True)
    assert fit.slope == pytest.approx(-1.5)


def test_mean_laminar_scaling_recovers_type_i_beta_from_oracle_sweep():
    eps_values = np.logspace(-4.0, -2.0, 7)
    mean_lengths = []
    for eps in eps_values:
        orbit = pm_type_i_oracle(20_000, x0=0.0, eps=eps, a=1.0, modulo=False)
        crossing = np.argmax(orbit > 0.5)
        mean_lengths.append(crossing + 1)

    scaling = mean_laminar_scaling(eps_values, np.asarray(mean_lengths, dtype=np.float64))

    np.testing.assert_equal(isinstance(scaling, MeanLaminarScaling), True)
    np.testing.assert_allclose(scaling.beta, 0.5, atol=0.08)
    np.testing.assert_equal(scaling.loglog.diagnostic_only, True)


def test_mean_laminar_scaling_reports_contested_candidate_laws_and_rpd_alpha():
    eps_values = np.logspace(-4.0, -2.0, 8)
    mean_lengths = 2.0 * np.log(1.0 / eps_values) + 1.0

    scaling = mean_laminar_scaling(eps_values, mean_lengths, rpd_alpha=0.25)

    assert isinstance(scaling.inverse_epsilon, CandidateScalingLaw)
    assert isinstance(scaling.logarithmic, CandidateScalingLaw)
    assert scaling.inverse_epsilon.name == "eps^-1"
    assert scaling.logarithmic.name == "log(1/eps)"
    assert scaling.rpd_alpha == pytest.approx(0.25)
    assert scaling.logarithmic.residual_sum_squares < scaling.inverse_epsilon.residual_sum_squares


def test_reinjection_mx_recovers_synthetic_rpd_alpha():
    rng = np.random.default_rng(2034)
    alpha = 0.5
    x_hat = 0.2
    reinjections = x_hat + rng.random(2000) ** (1.0 / (alpha + 1.0))
    series = np.empty(reinjections.size * 3, dtype=np.float64)
    mask = np.zeros(series.size, dtype=bool)
    series[0::3] = 2.0
    series[1::3] = reinjections
    series[2::3] = reinjections + 0.01
    mask[1::3] = True
    mask[2::3] = True

    result = reinjection_Mx(series, mask)

    np.testing.assert_equal(isinstance(result, ReinjectionMx), True)
    np.testing.assert_allclose(result.alpha, alpha, atol=0.08)
    np.testing.assert_allclose(result.x_hat, x_hat, atol=0.04)
    np.testing.assert_array_less(0.99, result.rvalue)


def test_reinjection_mx_ignores_initial_laminar_run():
    series = np.array([10.0, 11.0, 3.0, 0.0, 0.5, 3.0, 1.0, 1.5, 3.0, 2.0, 2.5])
    mask = np.array([True, True, False, True, True, False, True, True, False, True, True])

    result = reinjection_Mx(series, mask)

    np.testing.assert_allclose(result.reinjection_points, [0.0, 1.0, 2.0])


def test_burst_amplitude_distribution_detects_inverse_amplitude_law():
    rng = np.random.default_rng(2035)
    amplitudes = np.exp(rng.uniform(0.0, 20.0, size=5000))
    mask = np.zeros(amplitudes.size, dtype=bool)

    result = burst_amplitude_distribution(amplitudes, mask, min_tail=amplitudes.size)

    np.testing.assert_equal(isinstance(result, BurstAmplitudeDistribution), True)
    np.testing.assert_allclose(result.power_law.alpha, -1.0, atol=0.15)
    np.testing.assert_equal(result.amplitudes.size, amplitudes.size)


def test_burst_amplitude_distribution_bounds_heavy_tailed_fd_bins():
    amplitudes = np.r_[
        np.full(1000, 1e-76),
        np.full(1000, 2e-76),
        [1e-20, 1e-8, 1.0, 1e8, 1e20],
    ]
    mask = np.zeros(amplitudes.size, dtype=bool)

    with np.errstate(all="raise"):
        result = burst_amplitude_distribution(amplitudes, mask, min_tail=20)

    np.testing.assert_equal(isinstance(result, BurstAmplitudeDistribution), True)
    np.testing.assert_array_less(result.bin_edges.size, 514)
    np.testing.assert_equal(np.all(np.isfinite(result.bin_edges)), True)
    np.testing.assert_array_less(0.0, np.diff(result.bin_edges))
    np.testing.assert_equal(np.all(np.isfinite(result.density)), True)


def test_burst_amplitude_distribution_bounds_overflowing_fd_ratio():
    amplitudes = np.r_[
        np.full(1000, 1e-308),
        np.full(1000, 2e-308),
        [1e-200, 1e-100, 1.0, 1e100, 1e308],
    ]
    mask = np.zeros(amplitudes.size, dtype=bool)

    with np.errstate(all="raise"):
        result = burst_amplitude_distribution(amplitudes, mask, min_tail=20)

    np.testing.assert_equal(isinstance(result, BurstAmplitudeDistribution), True)
    np.testing.assert_array_less(result.bin_edges.size, 514)
    np.testing.assert_equal(np.all(np.isfinite(result.bin_edges)), True)
    np.testing.assert_equal(np.all(np.isfinite(result.density)), True)


def test_burst_amplitude_distribution_handles_on_off_oracle_regression():
    on_off = on_off_oracle(
        20_000,
        x0=1e-4,
        transverse_lyapunov=0.0,
        noise_scale=0.8,
        seed=1092158367,
    )
    threshold = np.percentile(np.abs(on_off), 50.0)
    mask = np.abs(on_off) <= threshold

    with np.errstate(all="raise"):
        result = burst_amplitude_distribution(on_off, mask)

    np.testing.assert_equal(isinstance(result, BurstAmplitudeDistribution), True)
    np.testing.assert_array_less(result.bin_edges.size, 514)
    np.testing.assert_equal(np.all(np.isfinite(result.bin_edges)), True)
    np.testing.assert_equal(np.all(np.isfinite(result.density)), True)


def test_laminar_burst_symmetry_separates_on_off_from_type_i_oracle():
    on_off = on_off_oracle(20_000, x0=1e-4, transverse_lyapunov=0.0, noise_scale=0.8, seed=2036)
    on_off_threshold = np.percentile(np.abs(on_off), 50.0)
    on_off_mask = np.abs(on_off) <= on_off_threshold

    type_i = logistic_type_i_oracle(8000, x0=0.2)
    type_i_mask, _ = detect_laminar_phases(type_i, method="period", percentile=10)

    on_off_symmetry = laminar_burst_symmetry(on_off, on_off_mask)
    type_i_symmetry = laminar_burst_symmetry(type_i, type_i_mask)

    np.testing.assert_equal(isinstance(on_off_symmetry, LaminarBurstSymmetry), True)
    np.testing.assert_array_less(0.1, on_off_symmetry.p_value)
    np.testing.assert_array_less(type_i_symmetry.p_value, 1e-6)
    np.testing.assert_array_less(on_off_symmetry.statistic, type_i_symmetry.statistic)


def test_extrema_return_map_uses_successive_peaks():
    t = np.linspace(0.0, 12.0 * np.pi, 4096)
    series = np.sin(t)

    result = extrema_return_map(series, kind="max")

    np.testing.assert_equal(isinstance(result, ExtremaReturnMap), True)
    np.testing.assert_equal(result.points.shape[1], 2)
    np.testing.assert_equal(result.values.size - 1, result.points.shape[0])
    np.testing.assert_allclose(result.points[:, 0], result.values[:-1])
    np.testing.assert_allclose(result.points[:, 1], result.values[1:])


def test_near_diagonal_tangent_channel_extracts_type_i_channel():
    series = logistic_type_i_oracle(8000, x0=0.2)
    extrema = extrema_return_map(series, kind="max")

    channel = near_diagonal_tangent_channel(extrema.points, percentile=10.0)

    np.testing.assert_equal(isinstance(channel, TangentChannel), True)
    np.testing.assert_array_less(20, channel.points.shape[0])
    np.testing.assert_allclose(channel.slope, 1.0, atol=0.05)
    np.testing.assert_array_less(0.95, abs(channel.rvalue))


def test_return_map_reconstruction_reuses_poincare_on_lorenz_1662():
    traj = lorenz_1662_oracle(t_span=(0.0, 5.0), dt=0.01, t_transient=0.0)
    series = traj[:, 0]

    reconstruction = return_map_reconstruction(series, fs=100.0)

    np.testing.assert_equal(isinstance(reconstruction, ReturnMapReconstruction), True)
    np.testing.assert_equal(reconstruction.poincare["section_plane_type"], "signal_delay_pair")
    np.testing.assert_equal(reconstruction.poincare["section_points"].shape[1], 2)
    np.testing.assert_array_less(2, reconstruction.extrema.points.shape[0])
    np.testing.assert_equal(isinstance(reconstruction.tangent_channel, TangentChannel), True)


def test_intermittency_summary_matches_standalone_subroutines():
    series = logistic_type_i_oracle(1500, x0=0.2)
    mask, lengths = detect_laminar_phases(series, method="period", percentile=10.0)
    distribution = laminar_length_distribution(lengths)
    power_law = fit_power_law_mle(lengths)
    gof = powerlaw_gof(lengths, fit=power_law, n_bootstrap=3, rng=77)
    comparison = compare_powerlaw_exponential(lengths)

    summary = intermittency_summary(
        series,
        laminar_method="period",
        percentile=10.0,
        powerlaw_gof_bootstrap=3,
        rng=77,
    )

    np.testing.assert_equal(isinstance(summary, IntermittencySummary), True)
    np.testing.assert_array_equal(summary.laminar_mask, mask)
    np.testing.assert_array_equal(summary.laminar_lengths, lengths)
    np.testing.assert_array_equal(summary.laminar_distribution.values, distribution.values)
    np.testing.assert_array_equal(summary.laminar_distribution.counts, distribution.counts)
    np.testing.assert_allclose(summary.laminar_power_law.alpha, power_law.alpha)
    np.testing.assert_allclose(summary.powerlaw_gof.p_value, gof.p_value)
    np.testing.assert_allclose(summary.family_comparison.z, comparison.z)
    np.testing.assert_array_less(0, summary.recurrence_laminarity.lengths.size)
    np.testing.assert_array_less(0, summary.reinjection.reinjection_points.size)
    np.testing.assert_array_less(0, summary.burst_amplitude.amplitudes.size)
    np.testing.assert_array_less(0, summary.laminar_burst_symmetry.laminar_lengths.size)
    np.testing.assert_array_less(0, summary.return_map.extrema.points.shape[0])


def test_intermittency_module_docstring_carries_signature_caveats():
    import dynachaos.diagnostics.intermittency as intermittency_module

    doc = intermittency_module.__doc__

    np.testing.assert_equal("Signature guide" in doc, True)
    np.testing.assert_equal("Caveats" in doc, True)
    np.testing.assert_equal("not as a theorem or classifier" in doc, True)
