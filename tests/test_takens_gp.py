"""Ground-truth regression test for the Grassberger-Procaccia stack.

Locks in that the Takens-Theiler estimator and the robust GP protocol recover the
known correlation dimension of canonical systems, and correctly REFUSE to certify
the cases GP cannot resolve at finite N (a 3-torus, white noise). This is the
guard that any future dynachaos change which silently breaks GP is caught.

Literature targets: Lorenz D2~2.05, Rossler ~2.0 (funnel measure biases low),
2-torus =2, 3-torus =3 (unresolvable here), Henon (map, tau=1) ~1.22,
van der Pol limit cycle =1, white noise -> rides D=m (no saturation).
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dynachaos.diagnostics import gp_dimension_robust, takens_theiler_dimension
from dynachaos.diagnostics.embedding import _embed
from dynachaos.diagnostics.gp_protocol import (
    _classify,
    _dc_of_m,
    _dominant_period_samples,
    _is_map_like,
    _saturation,
)

N = 14000
RHO, GOLD = 1.32471795724475, 1.61803398875


# ---------- canonical scalar signals (small N for a fast regression) ----------
def _lorenz(n=N):
    def f(t, s):
        x, y, z = s
        return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]

    te = np.arange(n + 5000) * 0.02
    return solve_ivp(
        f, (0, te[-1]), [1.0, 1.0, 1.0], t_eval=te, method="RK45", rtol=1e-9, atol=1e-11
    ).y[0, 5000:]


def _vdp_wave(n, dt=0.05, mu=2.0):
    def f(t, s):
        a, b = s
        return [b, mu * (1 - a * a) * b - a]

    te = np.linspace(0, dt * n * 3, int(n * 3) + 1)
    return te, solve_ivp(
        f, (0, te[-1]), [1.0, 0.0], t_eval=te, method="RK45", rtol=1e-9, atol=1e-11
    ).y[0]


_TW, _XW = _vdp_wave(N * 2)


def _torus(ratios, n=N):
    t = np.arange(n) * 0.05
    X = np.array([np.interp(t * r, _TW, _XW) for r in ratios])
    s = X.sum(0)
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            s = s + 0.3 * X[i] * X[j]
    return s


def _vdp_cycle(n=N):
    def f(t, s):
        a, b = s
        return [b, 2 * (1 - a * a) * b - a]

    te = np.arange(n + 4000) * 0.03
    return solve_ivp(f, (0, te[-1]), [2.0, 0.0], t_eval=te, method="RK45", rtol=1e-9, atol=1e-11).y[
        0, 4000:
    ]


def _henon(n=N):
    x, y = 0.1, 0.1
    out = np.empty(n + 1000)
    for i in range(n + 1000):
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        out[i] = x
    return out[1000:]


def _norm(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def _tt(sig, tau, m, w):
    d, *_ = takens_theiler_dimension(
        _embed(_norm(sig), m, tau), n_r=60, max_pairs=2_000_000, theiler_window=w
    )
    return d


# -------- direct Takens-Theiler dimension recovery (generous regression bands) --
@pytest.mark.parametrize(
    "name, sig, tau, m, w, lo, hi",
    [
        ("vdp_cycle", _vdp_cycle(), 8, 4, 20, 0.85, 1.15),  # D=1
        ("torus_T2", _torus([1, GOLD]), 35, 6, 80, 1.80, 2.30),  # D=2
        ("lorenz", _lorenz(), 8, 5, 30, 1.85, 2.30),  # D=2.05
        ("henon_map", _henon(), 1, 5, 1, 1.05, 1.40),  # D=1.22, map -> tau=1
    ],
)
def test_takens_recovers_known_dimension(name, sig, tau, m, w, lo, hi):
    d = _tt(sig, tau, m, w)
    assert lo <= d <= hi, f"{name}: D_TT={d:.3f} outside [{lo}, {hi}]"


# -------- robust protocol: certify low-D, refuse to certify the data-wall cases -
def test_protocol_certifies_T2():
    r = gp_dimension_robust(_torus([1, GOLD]), m_max=8, n_segments=6)
    assert r["gp_certifiable"] is True
    assert r["band_class"] == "T2"
    assert 1.8 <= r["D_c"] <= 2.3
    assert r["sigma"] < 0.1


def test_protocol_certifies_lorenz():
    r = gp_dimension_robust(_lorenz(), m_max=8, n_segments=6)
    assert r["gp_certifiable"] is True
    assert r["band_class"] == "T2"


def test_protocol_defers_T3():
    # a 3-torus is unresolvable by GP at this N: must NOT be certified
    r = gp_dimension_robust(_torus([1, RHO, RHO**2]), m_max=10, n_segments=6)
    assert r["gp_certifiable"] is False
    assert "unresolved" in r["band_class"] or "GP cannot certify" in r["band_class"]


def test_protocol_defers_noise():
    # white noise rides D_c = m: no saturation, never certified
    r = gp_dimension_robust(np.random.default_rng(0).standard_normal(N), m_max=10, n_segments=6)
    assert r["gp_certifiable"] is False
    assert r["dcm_slope"] > 0.1  # D_c(m) clearly climbing


def test_henon_map_tau1_recovered_via_is_map():
    r = gp_dimension_robust(_henon(), m_max=8, n_segments=6, is_map=True)
    assert r["tau_used"] == 1
    assert 1.05 <= r["D_c"] <= 1.40


def test_dominant_period_samples_returns_one_for_a_single_sample():
    assert _dominant_period_samples(np.array([1.0])) == 1


def test_is_map_like_is_false_for_a_constant_signal():
    # A constant signal has zero variance, so the lag-1 autocorrelation
    # denominator is 0 and the heuristic must not divide by it.
    assert _is_map_like(np.ones(50)) is False


def test_dc_of_m_returns_nan_when_embedding_too_short_for_reliable_estimate():
    x = np.random.default_rng(0).standard_normal(100)
    out = _dc_of_m(x, tau=5, theiler=1, m_values=[2, 3], max_pairs=1000, norm="chebyshev")
    assert np.all(np.isnan(out))


def test_saturation_returns_nan_with_fewer_than_four_finite_estimates():
    saturated, slope, D_c = _saturation([2, 3, 4, 5], [np.nan, np.nan, np.nan, 2.0])
    assert saturated is False
    assert np.isnan(slope)
    assert D_c == pytest.approx(2.0)


def test_saturation_falls_back_to_full_range_when_upper_half_too_small():
    # With 4 finite points, "upper half >= median" can leave only 2 points,
    # below the 3-point minimum, forcing the fallback to use every point.
    saturated, slope, D_c = _saturation([2, 3, 4, 5], [1.0, 1.5, 2.0, 2.1])
    assert D_c == pytest.approx(np.mean([1.0, 1.5, 2.0, 2.1]))


@pytest.mark.parametrize("D_c", [1.5, 2.4])
def test_classify_reports_ambiguous_band_gap(D_c):
    band_class, certifiable = _classify(D_c, saturated=True)
    assert band_class == "ambiguous (band gap)"
    assert certifiable is True


def test_gp_dimension_robust_falls_back_to_default_m_cao_when_cao_method_raises(monkeypatch):
    import dynachaos.diagnostics.gp_protocol as gp_mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gp_mod, "optimal_dimension", boom)
    x = np.random.default_rng(0).standard_normal(2000)
    r = gp_mod.gp_dimension_robust(x, tau=5, theiler=5, n_segments=3, m_max=6, max_pairs=50_000)
    assert r["m_used"] == 5


def test_gp_dimension_robust_reports_zero_segments_when_all_too_short_to_embed():
    # tau=500, m_cao=3 makes (m_cao-1)*tau=1000 dominate min_seg, but the
    # resulting embed length per segment (~50) is still below the 500-point
    # floor: every segment is skipped, exercising both the per-segment
    # continue and the "too few segments" NaN fallback.
    rng = np.random.default_rng(0)
    n = 5250
    x = rng.standard_normal(n)
    r = gp_dimension_robust(x, tau=500, m_cao=3, theiler=5, n_segments=5, m_max=4, max_pairs=50_000)
    assert r["n_segments_used"] == 0
    assert np.isnan(r["sigma"])
    assert np.isnan(r["ci"][0]) and np.isnan(r["ci"][1])
    assert np.isnan(r["seg_median"])
