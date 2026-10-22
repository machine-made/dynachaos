import numpy as np
import pytest

from dynachaos.diagnostics.poincare import (
    _auto_delay_from_autocorr,
    _crossing_indices,
    _quality_metrics,
    _spectral_peak_ratio,
    poincare_section,
)


def _periodic_signal(n=4096, dt=0.01):
    t = np.arange(n, dtype=np.float64) * dt
    return np.sin(2.0 * np.pi * 2.0 * t), 1.0 / dt


def test_poincare_section_periodic_signal_returns_planes_and_metrics():
    signal, fs = _periodic_signal()
    out = poincare_section(signal, fs)

    assert out["crossing_times"].size > 20
    assert out["crossing_values"].size == out["crossing_times"].size
    assert out["delay"] >= 1
    assert out["section_points"].ndim == 2
    assert out["section_points"].shape[1] == 2
    assert "signal_derivative" in out["planes"]
    assert out["metrics"]["num_crossings"] == out["crossing_times"].size
    assert out["metrics"]["quality"] in {
        "highly_periodic",
        "periodic",
        "quasi_periodic",
        "chaotic",
        "indeterminate",
        "insufficient_data",
    }


def test_poincare_section_direction_modes_are_supported():
    signal, fs = _periodic_signal()
    up = poincare_section(signal, fs, direction="up")
    down = poincare_section(signal, fs, direction="down")
    both = poincare_section(signal, fs, direction="both")

    assert up["crossing_times"].size > 0
    assert down["crossing_times"].size > 0
    assert both["crossing_times"].size >= up["crossing_times"].size
    assert both["crossing_times"].size >= down["crossing_times"].size


def test_poincare_section_explicit_delay():
    """Exercise the explicit delay= code path."""
    signal, fs = _periodic_signal()
    out = poincare_section(signal, fs, delay=10)

    assert out["delay"] == 10
    assert out["crossing_times"].size > 0
    assert "signal_delay_pair" in out["planes"]
    pair = out["planes"]["signal_delay_pair"]
    assert pair.ndim == 2
    assert pair.shape[1] == 2


def test_poincare_section_handles_short_signal():
    signal = np.array([0.0, 1.0], dtype=np.float64)
    out = poincare_section(signal, fs=100.0)
    assert out["crossing_times"].size == 0
    assert out["section_points"].shape == (0, 2)
    assert out["metrics"]["quality"] == "insufficient_data"


def test_poincare_section_rejects_nonpositive_sampling_frequency():
    with pytest.raises(ValueError, match="fs must be positive"):
        poincare_section(np.array([1.0, 2.0, 3.0]), fs=0.0)


def test_poincare_section_constant_signal_has_no_crossings():
    out = poincare_section(np.ones(10), fs=100.0)
    assert out["crossing_times"].size == 0
    assert out["section_points"].shape == (0, 2)


def test_poincare_section_drops_crossings_with_near_zero_slope():
    # A crossing straddling the eps threshold with a near-zero denominator
    # is filtered by the interpolation "good" mask; if it is the only
    # crossing, the result collapses to the empty-result branch.
    eps = 1e-12
    arr = np.array([-1.0, 0.5 * eps, 1.0000001 * eps, -1.0])
    out = poincare_section(arr, fs=100.0, level=0.0, direction="up", interpolation=True)
    assert out["crossing_times"].size == 0
    assert out["section_points"].shape == (0, 2)


def test_poincare_section_without_interpolation_uses_discrete_samples():
    signal, fs = _periodic_signal()
    out = poincare_section(signal, fs, interpolation=False)
    assert out["crossing_times"].size > 0
    assert out["interpolation_used"] is False
    # Discrete crossing times must land exactly on sample boundaries.
    sample_times = out["crossing_times"] * fs
    np.testing.assert_allclose(sample_times, np.round(sample_times))


def test_poincare_section_disabling_delay_plane_falls_back_to_signal_derivative():
    signal, fs = _periodic_signal()
    # delay >= len(signal) fails the "delay_samples < arr.size" guard, so
    # signal_delay_pair is never built and section_plane_type falls back to
    # signal_derivative (the elif branch).
    out = poincare_section(signal, fs, delay=len(signal))
    assert "signal_delay_pair" not in out["planes"]
    assert "signal_derivative" in out["planes"]
    assert out["section_plane_type"] == "signal_derivative"


def test_auto_delay_from_autocorr_returns_one_for_short_signal():
    assert _auto_delay_from_autocorr(np.array([1.0, 2.0])) == 1


def test_auto_delay_from_autocorr_falls_back_for_constant_signal():
    # A constant signal has zero autocorrelation everywhere (autocorr[0]==0)
    # since the centered signal is identically zero.
    delay = _auto_delay_from_autocorr(np.ones(20))
    assert delay == max(1, 20 // 10)


def test_auto_delay_from_autocorr_falls_back_when_no_local_minimum_found():
    delay = _auto_delay_from_autocorr(np.arange(5.0))
    assert delay == 1


def test_crossing_indices_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction must be one of"):
        _crossing_indices(np.array([-1.0, 1.0]), "sideways", 1e-12)


def test_spectral_peak_ratio_returns_nan_for_short_signal():
    assert np.isnan(_spectral_peak_ratio(np.array([1.0, 2.0])))


def test_spectral_peak_ratio_returns_nan_for_flat_signal():
    # A constant signal has zero spectral power at every frequency, so the
    # primary peak amplitude is exactly zero.
    assert np.isnan(_spectral_peak_ratio(np.ones(16)))


def test_spectral_peak_ratio_is_infinite_for_a_pure_single_tone():
    # A signal made of a single exact Fourier mode has no secondary peak
    # once the primary is removed.
    x = np.array([1.0, 0.0, -1.0, 0.0])
    assert _spectral_peak_ratio(x) == np.inf


def _two_tone_signal(n, ratio):
    t = np.arange(n)
    return np.sin(2 * np.pi * 3 * t / n) + (1.0 / ratio) * np.sin(2 * np.pi * 7 * t / n)


def test_quality_metrics_flags_indeterminate_for_degenerate_crossing_times():
    # Repeated crossing times give zero-mean intervals, so cv is nan and the
    # metric falls back to "indeterminate" regardless of spectral content.
    crossing_times = np.array([1.0, 1.0, 1.0, 1.0])
    metrics = _quality_metrics(crossing_times, np.sin(np.linspace(0, 10, 64)))
    assert metrics["quality"] == "indeterminate"
    assert np.isnan(metrics["coefficient_of_variation"])


def test_quality_metrics_periodic_tier():
    # cv ~ 0.1 (below the 0.2 periodic bound, above the 0.05 highly-periodic
    # bound) with spectral_peak_ratio == 3.5 (in [3, 5)).
    crossing_times = np.array([0.0, 1.0, 2.05, 2.9, 4.05, 5.0])
    metrics = _quality_metrics(crossing_times, _two_tone_signal(4096, ratio=3.5))
    assert metrics["quality"] == "periodic"


def test_quality_metrics_quasi_periodic_tier():
    crossing_times = np.array([0.0, 1.0, 2.3, 3.2, 4.6, 5.5])
    metrics = _quality_metrics(crossing_times, _two_tone_signal(4096, ratio=2.0))
    assert metrics["quality"] == "quasi_periodic"


def test_quality_metrics_chaotic_tier():
    # Highly irregular intervals push cv above the 0.6 quasi-periodic bound.
    crossing_times = np.array([0.0, 1.0, 4.0, 4.3, 9.0, 9.2])
    metrics = _quality_metrics(crossing_times, _two_tone_signal(4096, ratio=2.0))
    assert metrics["quality"] == "chaotic"
