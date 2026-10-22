"""Shared test fixtures and helpers."""

import os

import numpy as np
import pytest

# The committed figure caches under figures/ cannot be reproduced elementwise
# anywhere but the machine that produced them.
#
# Most of these sections integrate chaotic maps and coupled map lattices, where
# a last-bit difference in the first arithmetic operation is amplified
# exponentially: on macOS arm64 the recomputed sec12_intermittency cache differs
# from the committed one by a relative 4e4, and sec08_sti differs in 100% of
# elements. That is the systems behaving correctly, not a defect, and no
# tolerance can accommodate it.
#
# The scope of that statement was initially misjudged as "another architecture".
# It is narrower. GitHub's ubuntu-latest runner is also Linux x86_64 and it too
# fails to reproduce sec12_intermittency elementwise, and extracts a different
# number of Lorenz return points than the reference machine did -- a different
# microarchitecture is enough to change which OpenBLAS kernels are selected and
# how the last bits fall. Gating on the architecture therefore asserted
# something false and turned an inherent property of chaotic systems into a CI
# failure.
#
# So elementwise agreement is opt-in, enabled only on the machine that
# regenerates the caches. Everywhere else -- including every CI runner -- the
# pipelines are still executed and their output is still checked against the
# contract that does survive, plus the physics assertions, which hold
# everywhere. See assert_npz_structurally_sound for what survives and for the
# two properties that only *look* portable (exact shape, finiteness).
REFERENCE_PLATFORM_ENV = "DYNACHAOS_REFERENCE_PLATFORM"


def is_reference_platform() -> bool:
    """True only where the committed caches can be reproduced elementwise.

    Set ``DYNACHAOS_REFERENCE_PLATFORM=1`` on the machine that regenerates the
    caches under ``figures/``. There is no autodetect for this: reproducing the
    caches bit-for-bit depends on the CPU, the BLAS build and its kernel
    dispatch, so it is a property of one machine rather than of any platform
    triple that could be sniffed at runtime.
    """
    return os.environ.get(REFERENCE_PLATFORM_ENV) == "1"


requires_reference_platform = pytest.mark.skipif(
    not is_reference_platform(),
    reason=(
        f"committed figure caches reproduce only where {REFERENCE_PLATFORM_ENV}=1; chaotic "
        "sections cannot reproduce them elementwise on any other machine"
    ),
)


def assert_npz_structurally_sound(
    generated, committed, max_leading_ratio=2.0, nan_fraction_tol=0.02
):
    """Assert what stays true off the reference platform: same contract, sane values.

    Elementwise agreement is unattainable off the reference machine (see
    REFERENCE_PLATFORM_ENV), but three things are still genuine contracts and are
    checked here. Two others look like contracts and are not:

    ``shape`` is only partly a contract. Trailing axes are structural -- a
    return map is (N, 2) on every machine and a regression to (N, 3) is real
    breakage. The *leading* axis is often a measurement: ``lorenz_return_points``
    counts Poincare crossings of a chaotic trajectory, so arm64 extracts 276
    where x86_64 extracted 274. That is the Lorenz system behaving correctly.
    So trailing axes must match exactly while the leading axis need only stay
    within ``max_leading_ratio`` of the reference -- a non-degeneracy guard, not
    a precision claim. Real breakage (empty output, wrong pipeline) moves these
    counts by orders of magnitude, not by a percent.

    ``np.isfinite`` is too strong. NaN is a *designed sentinel* in several
    caches: ``coupled_logistic`` builds its phase diagram with
    ``np.full(shape, np.nan)`` and fills only cells with samples
    (``np.divide(..., where=count > 0)``), leaving 15% of sec03_transition's
    grid NaN by construction, which the plotter renders via ``cmap.set_bad``.
    Demanding finiteness therefore fails on a correct array. Infinities are a
    different matter -- nothing here is meant to overflow -- so those are
    rejected outright, and the NaN *fraction* is required to stay near the
    reference so that an all-NaN or never-NaN regression is still caught.
    """
    assert set(generated.files) == set(committed.files)
    for key in generated.files:
        actual = generated[key]
        expected = committed[key]
        assert actual.dtype == expected.dtype, key
        assert actual.ndim == expected.ndim, key
        assert actual.shape[1:] == expected.shape[1:], (
            f"{key}: trailing axes are a structural contract, "
            f"got {actual.shape} against {expected.shape}"
        )
        if expected.shape and expected.shape[0]:
            ratio = actual.shape[0] / expected.shape[0]
            assert 1 / max_leading_ratio <= ratio <= max_leading_ratio, (
                f"{key}: leading axis {actual.shape[0]} is off the reference "
                f"{expected.shape[0]} by more than {max_leading_ratio}x -- too "
                f"large to be architecture jitter in an extracted count"
            )
        if np.issubdtype(actual.dtype, np.floating):
            assert not np.isinf(actual).any(), f"{key}: contains infinities"
            if actual.size and expected.size:
                actual_nan = float(np.isnan(actual).mean())
                expected_nan = float(np.isnan(expected).mean())
                assert abs(actual_nan - expected_nan) <= nan_fraction_tol, (
                    f"{key}: NaN fraction {actual_nan:.4f} against reference "
                    f"{expected_nan:.4f} (tolerance {nan_fraction_tol}). NaN is a "
                    f"valid sentinel here, but the population should not shift."
                )


def assert_threshold_well_separated(distances, eps, min_margin=1e-9):
    """Assert no distance sits close enough to ``eps`` to flip between platforms.

    Tests that pin goldens derived from a thresholded distance matrix are only
    portable while no distance lies near the threshold. Pairwise distances differ
    in their last bits across BLAS implementations and architectures, so a
    distance within ~1e-15 of ``eps`` can fall on either side depending on the
    machine, changing the extracted lines and the pinned values.

    This makes that precondition explicit and enforced rather than accidental:
    if someone retunes the trajectory or the percentile into a degenerate
    configuration, the failure says so instead of appearing later as an
    unexplained mismatch on one platform's CI runner.

    The circle trajectory in TestRecurrenceParity is the cautionary case — its
    distances land exactly on the threshold (margin 0.0), which is why it needed
    quantisation instead of separation.
    """
    distances = np.asarray(distances)
    margin = float(np.min(np.abs(distances - eps)))
    assert margin > min_margin, (
        f"threshold eps={eps!r} is only {margin:.3e} from the nearest pairwise "
        f"distance (need > {min_margin:.0e}). Goldens derived from this "
        f"threshold are not portable across architectures."
    )
    return margin


def logistic_series(n=2000, a=1.99, burn=500):
    """Generate a logistic map time series for testing."""
    x = 0.123456789
    series = np.empty(n)
    for i in range(n + burn):
        x = 1.0 - a * x * x
        if i >= burn:
            series[i - burn] = x
    return series


@pytest.fixture
def chaotic_series():
    """A chaotic logistic map series (a=1.99, 5000 points)."""
    return logistic_series(n=5000, a=1.99, burn=2000)
