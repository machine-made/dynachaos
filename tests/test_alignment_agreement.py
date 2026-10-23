"""Known-answer tests: alignment indices, Lyapunov block errors and diagnostic agreement."""

import numpy as np
import pytest

from dynachaos.diagnostics.agreement import (
    CHAOTIC,
    MARGINAL,
    REGULAR,
    label_from_spectrum,
    roc_auc,
    roc_curve,
    scored_agreement,
)
from dynachaos.diagnostics.lyapunov import lyapunov_spectrum_blocks
from dynachaos.diagnostics.sali_gali import (
    alignment_indices,
    gali,
    log10_sali_from_gali2,
    sali,
)
from dynachaos.diagnostics.zero_one_test import zero_one_statistic
from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

# ---------------------------------------------------------------------------
# ROC area and labels
# ---------------------------------------------------------------------------


def test_auc_perfect_separator_is_one():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    is_chaotic = np.array([False, False, False, True, True, True])
    assert roc_auc(scores, is_chaotic) == 1.0
    assert roc_auc(-scores, is_chaotic) == 0.0


def test_auc_constant_scores_is_one_half():
    assert roc_auc(np.ones(10), np.arange(10) % 2 == 0) == 0.5


def test_auc_random_scores_is_near_one_half():
    rng = np.random.default_rng(7)
    scores = rng.standard_normal(4000)
    assert roc_auc(scores, rng.random(4000) < 0.4) == pytest.approx(0.5, abs=0.03)


def test_auc_matches_pairwise_count_with_ties():
    scores = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 0.0])
    pos = np.array([True, True, False, True, False, False])
    p, n = scores[pos], scores[~pos]
    expected = np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n])
    assert roc_auc(scores, pos) == pytest.approx(expected)


def test_roc_curve_area_equals_auc():
    rng = np.random.default_rng(3)
    scores = np.round(rng.standard_normal(300), 1)  # ties on purpose
    pos = rng.random(300) < 0.5
    fpr, tpr = roc_curve(scores, pos)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0)
    assert np.trapezoid(tpr, fpr) == pytest.approx(roc_auc(scores, pos))


def test_roc_auc_needs_both_classes():
    with pytest.raises(ValueError, match="at least one positive and one negative"):
        roc_auc(np.arange(4.0), np.ones(4, dtype=bool))


@pytest.mark.parametrize(
    ("spectrum", "se", "expected"),
    [
        ([0.5], [0.01], CHAOTIC),
        ([-0.3], [0.01], REGULAR),
        ([0.001], [0.01], MARGINAL),  # 1D map with lambda_1 at zero
        ([1e-6, -2e-6, -0.2, -0.4], [1e-5] * 4, REGULAR),  # 2-torus
        ([1e-6, -5e-3, -0.2, -0.4], [1e-5] * 4, REGULAR),  # partial locking
        ([1e-6, 0.05, -0.2, -0.4], [1e-5] * 4, MARGINAL),  # lambda_2 > 0, lambda_1 ~ 0
        ([0.07, 0.03, -0.01, -0.3], [1e-3] * 4, CHAOTIC),
    ],
)
def test_label_from_spectrum(spectrum, se, expected):
    assert label_from_spectrum(spectrum, se) == expected


def test_scored_agreement_counts_and_threshold():
    labels = np.array([REGULAR, REGULAR, REGULAR, CHAOTIC, CHAOTIC, MARGINAL])
    k01 = np.array([0.02, 0.05, 0.9, 0.95, 0.4, 0.5])
    out = scored_agreement(k01, labels, threshold=0.5)
    assert (out["n_scored"], out["n_marginal"]) == (5, 1)
    assert out["agreement"] == pytest.approx(3 / 5)
    assert list(out["disagree"]) == [2, 4]
    assert list(out["misranked_regular"]) == [2]  # 0.9 > median chaotic 0.675
    assert out["auc"] == pytest.approx((1 + 1 + 0 + 1 + 1 + 1) / 6)


# ---------------------------------------------------------------------------
# 0-1 test on a period-2 orbit
# ---------------------------------------------------------------------------


def _logistic_series(a, n):
    x, out = 0.3, np.empty(n)
    for i in range(n + 2000):
        x = 1.0 - a * x * x
        if i >= 2000:
            out[i - 2000] = x
    return out


def test_zero_one_test_period_two_orbit_is_regular():
    assert zero_one_statistic(_logistic_series(1.1, 5000), n_c=50) < 0.1


def test_zero_one_test_full_chaos_is_chaotic():
    assert zero_one_statistic(_logistic_series(1.9, 5000), n_c=50) > 0.9


# ---------------------------------------------------------------------------
# Lyapunov spectrum with block errors
# ---------------------------------------------------------------------------


def test_block_spectrum_logistic_full_chaos():
    spec, se = lyapunov_spectrum_blocks(
        lambda s: 1.0 - 2.0 * s * s,
        lambda s: np.array([[-4.0 * s[0]]]),
        np.array([0.3]),
        n_iter=40_000,
        n_transient=1000,
    )
    assert abs(spec[0] - np.log(2.0)) < 3 * se[0]
    assert 0.0 < se[0] < 0.01


def test_block_spectrum_sum_equals_mean_log_determinant():
    DB = 2.55
    f = lambda s: coupled_delayed(s, 0.4, DB + 0.1, DB, 5e-3)  # noqa: E731
    jac = lambda s: coupled_delayed_jac(s, 0.4, DB + 0.1, DB, 5e-3)  # noqa: E731
    x0 = np.array([0.5, 0.5, 0.3, 0.3])
    n_iter, n_transient = 20_000, 2000
    spec, se = lyapunov_spectrum_blocks(f, jac, x0, n_iter=n_iter, n_transient=n_transient)
    x = x0
    for _ in range(n_transient):
        x = f(x)
    log_det = 0.0
    for _ in range(n_iter):
        log_det += np.log(abs(np.linalg.det(jac(x))))
        x = f(x)
    assert abs(spec.sum() - log_det / n_iter) < 1e-9


def test_block_spectrum_rejects_one_block():
    with pytest.raises(ValueError, match="n_blocks"):
        lyapunov_spectrum_blocks(lambda s: s, lambda s: np.eye(1), np.zeros(1), 100, 0, 1)


def test_block_spectrum_stops_on_divergence():
    with np.errstate(all="ignore"), pytest.raises(FloatingPointError):
        lyapunov_spectrum_blocks(
            lambda s: s * s + 2.0, lambda s: np.array([[2.0 * s[0]]]), np.array([3.0]), 200, 0, 4
        )


# ---------------------------------------------------------------------------
# Alignment indices
# ---------------------------------------------------------------------------


def _torus_times_contraction(c):
    """Rigid rotation on a 2-torus (angle coordinates) times a contraction by c."""
    omega = np.array([0.31, 0.57])
    f = lambda s: np.array([s[0] + omega[0], s[1] + omega[1], c * s[2]])  # noqa: E731
    jac = lambda s: np.diag([1.0, 1.0, c])  # noqa: E731
    return f, jac


def test_gali_rotation_on_torus_times_contraction():
    c, seed, n = 0.9, 5, 400
    f, jac = _torus_times_contraction(c)
    out = alignment_indices(
        f, jac, np.zeros(3), k=3, n_iter=n, n_transient=0, rng=np.random.default_rng(seed)
    )
    # Reproduce the initial frame: columns of Q from a QR of a Gaussian matrix.
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))[0]
    steps = np.arange(1, n + 1)[:, None]
    v = np.stack([Q[:, :2]] * n)  # (n, 3, 2): angle components stay, radial part shrinks
    v[:, 2, :] *= (c**steps).reshape(n, 1)
    gram = np.einsum("nik,nil->nkl", v, v)
    expected_gali2 = np.sqrt(np.linalg.det(gram) / (gram[:, 0, 0] * gram[:, 1, 1]))
    np.testing.assert_allclose(10.0 ** out[:, 1], expected_gali2, rtol=1e-10)
    # GALI_2 stays at its limit and GALI_3 falls by exactly c per step.
    assert out[-1, 1] == pytest.approx(out[300, 1], abs=1e-12)
    np.testing.assert_allclose(np.diff(out[300:, 2]), np.log10(c), atol=1e-10)


def test_alignment_indices_match_gali_and_sali_before_floor():
    DB = 2.55
    f = lambda s: coupled_delayed(s, 0.4, DB + 0.1, DB, 5e-3)  # noqa: E731
    jac = lambda s: coupled_delayed_jac(s, 0.4, DB + 0.1, DB, 5e-3)  # noqa: E731
    x0 = np.array([0.5, 0.5, 0.3, 0.3])
    n = 40
    full = alignment_indices(f, jac, x0, k=4, n_iter=n, n_transient=500)
    ref4 = gali(f, jac, x0, k=4, n_iter=n, n_transient=500)
    np.testing.assert_allclose(10.0 ** full[:, 3], ref4, rtol=1e-8)
    two = alignment_indices(f, jac, x0, k=2, n_iter=n, n_transient=500)
    ref_sali = sali(f, jac, x0, n_iter=n, n_transient=500)
    np.testing.assert_allclose(10.0 ** log10_sali_from_gali2(two[:, 1]), ref_sali, rtol=1e-7)


def test_gali_decay_rate_equals_lyapunov_gap_sum():
    DB = 2.55
    f = lambda s: coupled_delayed(s, 0.4, DB + 0.1, DB, 5e-3)  # noqa: E731
    jac = lambda s: coupled_delayed_jac(s, 0.4, DB + 0.1, DB, 5e-3)  # noqa: E731
    x0 = np.array([0.5, 0.5, 0.3, 0.3])
    spec, se = lyapunov_spectrum_blocks(f, jac, x0, n_iter=60_000, n_transient=5000)
    out = alignment_indices(f, jac, x0, k=4, n_iter=60_000, n_transient=5000)
    n1, n2 = 10_000, 60_000
    for k in (2, 3, 4):
        measured = (out[n2 - 1, k - 1] - out[n1 - 1, k - 1]) * np.log(10.0) / (n2 - n1)
        predicted = np.sum(spec[1:k] - spec[0])
        assert measured == pytest.approx(predicted, abs=5 * np.hypot(se[0] * k, se[1:k].sum()))


def test_alignment_indices_reject_too_many_vectors():
    with pytest.raises(ValueError, match="k must be"):
        alignment_indices(lambda s: s, lambda s: np.eye(2), np.zeros(2), k=3, n_iter=5)


def test_log10_sali_from_gali2_limits():
    # Orthogonal vectors: GALI_2 = 1 and SALI = sqrt(2).
    assert 10.0 ** log10_sali_from_gali2(0.0) == pytest.approx(np.sqrt(2.0))
    # Aligned vectors: SALI equals GALI_2 and stays resolved below the double-precision floor.
    assert log10_sali_from_gali2(-500.0) == pytest.approx(-500.0)
