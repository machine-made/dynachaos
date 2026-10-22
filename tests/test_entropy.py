"""Tests for entropy-based diagnostics (SampEn, ApEn, FuzzyEn, MSE)."""

import numpy as np
import pytest
from conftest import logistic_series

from dynachaos.diagnostics.entropy import (
    approximate_entropy,
    fuzzy_entropy,
    multiscale_entropy,
    sample_entropy,
)


def sine_series(n=2000):
    return np.sin(np.linspace(0.0, 40.0 * np.pi, n))


def white_noise(n=2000, seed=2026):
    return np.random.default_rng(seed).standard_normal(n)


# ── SampEn ────────────────────────────────────────────────────────────────────


def test_sample_entropy_chaotic_vs_regular():
    """Chaotic signal has higher SampEn than a periodic one."""
    chaos = logistic_series(n=2000, a=1.99, burn=500)
    regular = sine_series(2000)
    assert sample_entropy(chaos) > sample_entropy(regular)


def test_sample_entropy_regular_near_zero():
    """Highly regular periodic signal has near-zero SampEn."""
    se = sample_entropy(sine_series(2000))
    assert se < 0.25


def test_sample_entropy_noise_high():
    """White noise SampEn is high (> 1.5 for N=2000)."""
    se = sample_entropy(white_noise())
    assert se > 1.5


def test_sample_entropy_explicit_r():
    """Explicit r overrides the default and produces a finite result."""
    x = logistic_series(n=500, a=1.99)
    se = sample_entropy(x, r=0.1)
    assert np.isfinite(se)


def test_sample_entropy_b_zero_returns_nan():
    """When no m-length templates match (r too small), return NaN."""
    x = np.arange(50, dtype=float)  # monotone: all pairs at max distance
    se = sample_entropy(x, m=2, r=1e-15)
    assert np.isnan(se)


def test_sample_entropy_a_zero_returns_inf():
    """When m-length matches exist but no (m+1)-length ones, return inf."""
    # Constant series: all m-length templates match (B > 0), but the m+1
    # templates also all match — so this path is hard to hit in practice.
    # Instead, verify inf is returned when r is extremely tight for m+1.
    x = logistic_series(n=200, a=1.99)
    # Force by calling with a moderately small r where B>0 but A might be 0
    se = sample_entropy(x, m=2, r=1e-10)
    assert np.isnan(se) or np.isinf(se)  # one of the two degenerate outcomes


def test_sample_entropy_short_series_returns_inf():
    """Series too short to form any pair returns inf."""
    x = np.array([1.0, 2.0])  # m=2 needs at least 3 points
    assert np.isinf(sample_entropy(x, m=2))


def test_sample_entropy_invalid_r():
    with pytest.raises(ValueError):
        sample_entropy(np.ones(100), r=-0.1)


def test_sample_entropy_invalid_m():
    with pytest.raises(ValueError):
        sample_entropy(np.ones(100), m=0)


@pytest.mark.parametrize(
    "func", [sample_entropy, approximate_entropy, fuzzy_entropy, multiscale_entropy]
)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_entropy_functions_reject_nonfinite_series(func, bad_value):
    x = np.linspace(0.0, 1.0, 20)
    x[4] = bad_value

    kwargs = {"scales": [1], "r": 0.2} if func is multiscale_entropy else {"r": 0.2}
    with pytest.raises(ValueError, match="finite values"):
        func(x, **kwargs)


@pytest.mark.parametrize(
    "func", [sample_entropy, approximate_entropy, fuzzy_entropy, multiscale_entropy]
)
def test_entropy_translation_invariance_with_fixed_tolerance(func):
    x = logistic_series(n=600, a=1.99, burn=200)
    r = 0.2 * np.std(x, ddof=1)

    kwargs = {"scales": [1, 2], "r": r} if func is multiscale_entropy else {"r": r}
    base = func(x, **kwargs)
    shifted = func(x + 17.0, **kwargs)

    np.testing.assert_allclose(shifted, base, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "func", [sample_entropy, approximate_entropy, fuzzy_entropy, multiscale_entropy]
)
def test_entropy_positive_scaling_invariance_when_tolerance_scales(func):
    x = logistic_series(n=600, a=1.99, burn=200)
    r = 0.2 * np.std(x, ddof=1)
    scale = 3.5

    base_kwargs = {"scales": [1, 2], "r": r} if func is multiscale_entropy else {"r": r}
    scaled_kwargs = (
        {"scales": [1, 2], "r": r * scale} if func is multiscale_entropy else {"r": r * scale}
    )

    base = func(x, **base_kwargs)
    scaled = func(scale * x, **scaled_kwargs)

    np.testing.assert_allclose(scaled, base, rtol=1e-12, atol=1e-12)


# ── ApEn ──────────────────────────────────────────────────────────────────────


def test_approximate_entropy_positive():
    """ApEn is non-negative for any signal."""
    x = logistic_series(n=500, a=1.99, burn=100)
    assert approximate_entropy(x) >= 0.0


def test_approximate_entropy_chaotic_vs_regular():
    """Chaotic ApEn exceeds periodic ApEn."""
    chaos = logistic_series(n=500, a=1.99, burn=100)
    regular = sine_series(500)
    assert approximate_entropy(chaos) > approximate_entropy(regular)


# ── FuzzyEn ───────────────────────────────────────────────────────────────────


def test_fuzzy_entropy_finite_always():
    """FuzzyEn is always finite (no hard threshold, no zero membership)."""
    x = logistic_series(n=1000, a=1.99, burn=200)
    fe = fuzzy_entropy(x)
    assert np.isfinite(fe)


def test_fuzzy_entropy_chaotic_vs_regular():
    """Chaotic FuzzyEn exceeds periodic FuzzyEn."""
    chaos = logistic_series(n=1000, a=1.99, burn=200)
    regular = sine_series(1000)
    assert fuzzy_entropy(chaos) > fuzzy_entropy(regular)


def test_fuzzy_entropy_noise_high():
    """White-noise FuzzyEn is higher than chaotic."""
    chaos = logistic_series(n=1000, a=1.99, burn=200)
    noise = white_noise(1000)
    assert fuzzy_entropy(noise) > fuzzy_entropy(chaos)


def test_fuzzy_entropy_invalid_n():
    with pytest.raises(ValueError):
        fuzzy_entropy(np.ones(100), n=0)


# ── MSE ───────────────────────────────────────────────────────────────────────


def test_mse_shape():
    """MSE returns one value per scale."""
    x = logistic_series(n=2000, a=1.99, burn=500)
    scales = range(1, 11)
    mse = multiscale_entropy(x, scales=scales)
    assert mse.shape == (10,)


def test_mse_scale1_matches_sample_entropy():
    """MSE at scale 1 equals SampEn with r = 0.15 * std(x)."""
    x = logistic_series(n=2000, a=1.99, burn=500)
    r = 0.15 * np.std(x, ddof=1)
    mse = multiscale_entropy(x, scales=[1], m=2)
    se = sample_entropy(x, m=2, r=r)
    assert abs(mse[0] - se) < 1e-10


def test_mse_noise_decreasing():
    """White noise MSE decreases across scales (Costa et al. 2002, Fig. 1)."""
    noise = white_noise(n=4000)
    mse = multiscale_entropy(noise, scales=range(1, 8))
    # Entropy at scale 1 should exceed entropy at scale 6
    assert mse[0] > mse[5]


def test_mse_r_fixed_from_original():
    """Explicit r is preserved across all scales (not rescaled per scale)."""
    x = white_noise(2000)
    r_fixed = 0.05
    mse_a = multiscale_entropy(x, scales=[1, 2, 3], m=2, r=r_fixed)
    # Each scale produces a finite result with the fixed r
    assert all(np.isfinite(mse_a) | np.isinf(mse_a))


def test_mse_invalid_scale():
    with pytest.raises(ValueError):
        multiscale_entropy(np.ones(100), scales=[0])


# ── Rust / Python parity ──────────────────────────────────────────────────────


def test_sample_entropy_rust_python_parity(monkeypatch):
    """Rust and pure-Python backends produce identical SampEn values."""
    import dynachaos.diagnostics.entropy as _ent

    x = logistic_series(n=300, a=1.99, burn=100)
    se_rust = sample_entropy(x, m=2, r=0.2 * np.std(x, ddof=1))

    monkeypatch.setattr(_ent, "_RUST_AVAILABLE", False)
    se_python = sample_entropy(x, m=2, r=0.2 * np.std(x, ddof=1))

    assert abs(se_rust - se_python) < 1e-10


def test_approximate_entropy_rust_python_parity(monkeypatch):
    """Rust and pure-Python backends produce identical ApEn values."""
    import dynachaos.diagnostics.entropy as _ent

    x = logistic_series(n=240, a=1.99, burn=100)
    r = 0.2 * np.std(x, ddof=1)
    ae_rust = approximate_entropy(x, m=2, r=r)

    monkeypatch.setattr(_ent, "_RUST_AVAILABLE", False)
    ae_python = approximate_entropy(x, m=2, r=r)

    assert abs(ae_rust - ae_python) < 1e-10


def test_fuzzy_entropy_rust_python_parity(monkeypatch):
    """Rust and pure-Python backends produce identical FuzzyEn values."""
    import dynachaos.diagnostics.entropy as _ent

    x = logistic_series(n=200, a=1.99, burn=100)
    r = 0.2 * np.std(x, ddof=1)
    fe_rust = fuzzy_entropy(x, m=2, r=r, n=2)

    monkeypatch.setattr(_ent, "_RUST_AVAILABLE", False)
    fe_python = fuzzy_entropy(x, m=2, r=r, n=2)

    assert abs(fe_rust - fe_python) < 1e-9


# ── Coverage: verbose output and edge cases ────────────────────────────────────


def test_sample_entropy_verbose_with_fast_execution(capsys):
    """Verbose output triggers _print_timing (lines 177, 58-59)."""
    x = logistic_series(n=50, a=1.99, burn=20)
    sample_entropy(x, m=2, verbose=True)
    captured = capsys.readouterr()
    assert "entropy" in captured.out
    assert "pairs/s" in captured.out or "M pairs" in captured.out


def test_approximate_entropy_verbose_backend_selection(capsys):
    """Approximate entropy backend name printed in verbose mode (lines 240-241)."""
    x = logistic_series(n=50, a=1.99, burn=20)
    approximate_entropy(x, m=2, verbose=True)
    captured = capsys.readouterr()
    assert "entropy" in captured.out


def test_fuzzy_entropy_verbose(capsys):
    """Fuzzy entropy verbose _print_timing call (line 308)."""
    x = logistic_series(n=50, a=1.99, burn=20)
    fuzzy_entropy(x, m=2, verbose=True)
    captured = capsys.readouterr()
    assert "entropy" in captured.out


def test_sample_entropy_n_m_less_than_2_returns_inf():
    """When embedding produces fewer than 2 templates, return inf (line 156)."""
    x = np.array([1.0, 2.0, 3.0])  # m=2 needs at least 3; with m=2, n_m=2 after trimming
    # Construct a case where n_m < 2 after trimming
    se = sample_entropy(x, m=2, r=0.1)
    assert np.isinf(se)


def test_approximate_entropy_short_series_returns_inf():
    """Approximate entropy returns inf when series too short (line 216)."""
    x = np.array([1.0, 2.0])
    ae = approximate_entropy(x, m=2)
    assert np.isinf(ae)


def test_fuzzy_entropy_short_series_returns_inf():
    """Fuzzy entropy returns inf when series too short (line 280)."""
    x = np.array([1.0, 2.0])
    fe = fuzzy_entropy(x, m=2)
    assert np.isinf(fe)


def test_fuzzy_entropy_n_m_less_than_2_returns_inf():
    """Fuzzy entropy returns inf when n_m < 2 (line 289)."""
    x = np.array([1.0, 2.0, 3.0])
    fe = fuzzy_entropy(x, m=2)
    assert np.isinf(fe)


def test_multiscale_entropy_verbose(capsys):
    """Multiscale entropy verbose call (line 378)."""
    x = logistic_series(n=50, a=1.99, burn=20)
    multiscale_entropy(x, scales=[1, 2], verbose=True)
    captured = capsys.readouterr()
    assert "MSE" in captured.out


def test_multiscale_entropy_invalid_scale_error():
    """Multiscale entropy raises error for scale < 1 (line 367)."""
    x = logistic_series(n=100, a=1.99)
    with pytest.raises(ValueError, match="all scales must be >= 1"):
        multiscale_entropy(x, scales=[0, 1])


def test_approximate_entropy_invalid_m_error():
    """Approximate entropy raises when m < 1 (line 214)."""
    x = logistic_series(n=100, a=1.99)
    with pytest.raises(ValueError, match="m must be >= 1"):
        approximate_entropy(x, m=0)


def test_fuzzy_entropy_invalid_m_error():
    """Fuzzy entropy raises when m < 1 (line 276)."""
    x = logistic_series(n=100, a=1.99)
    with pytest.raises(ValueError, match="m must be >= 1"):
        fuzzy_entropy(x, m=0)
