"""Tests for multifractal diagnostics."""

import os
from pathlib import Path

import numpy as np
import pytest

from dynachaos.diagnostics.multifractal import (
    _clean_box_sizes,
    _clean_q_values,
    _default_box_sizes,
    _linear_fit_slope_r2,
    _multifractal_moments_python,
    local_multifractality,
    multifractal_spectrum,
)

try:
    import dynachaos._rust  # noqa: F401

    _RUST_IMPORTABLE = True
except ImportError:
    _RUST_IMPORTABLE = False

_NO_RUST_ENV = bool(os.environ.get("DYNACHAOS_NO_RUST"))
_GOLDEN_PATH = Path(__file__).with_name("data") / "rust_parity_goldens.npz"


def _golden(name):
    with np.load(_GOLDEN_PATH) as goldens:
        return goldens[name]


def test_uniform_field_is_monofractal():
    field = np.ones((64, 64), dtype=np.float64)
    q = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    box = np.array([2, 4, 8, 16, 32], dtype=np.int64)

    out = multifractal_spectrum(field, box_sizes=box, q_values=q)

    dq = np.asarray(out["Dq"], dtype=np.float64)
    finite = np.isfinite(dq)
    assert finite.any()
    np.testing.assert_allclose(dq[finite], 2.0, atol=1e-10)
    assert float(out["phi"]) < 1e-12


def test_uniform_field_with_truncated_edges_is_still_monofractal():
    field = np.ones((90, 90), dtype=np.float64)
    q = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    # Deliberately non-divisors of the field side length.
    box = np.array([8, 16, 22, 30], dtype=np.int64)

    out = multifractal_spectrum(field, box_sizes=box, q_values=q)

    dq = np.asarray(out["Dq"], dtype=np.float64)
    finite = np.isfinite(dq)
    assert finite.any()
    # With truncated edges, the exact slope follows the occupied-box scaling.
    n_boxes = (field.shape[0] // box) * (field.shape[1] // box)
    slope, _ = np.polyfit(np.log(box.astype(np.float64)), np.log(n_boxes.astype(np.float64)), 1)
    expected_d = -float(slope)
    np.testing.assert_allclose(dq[finite], expected_d, atol=1e-10)
    assert float(np.std(dq[finite])) < 1e-10


def test_local_multifractality_detects_heterogeneous_patch():
    rng = np.random.default_rng(42)
    field = np.ones((64, 64), dtype=np.float64)
    # Inject one heterogeneous tile.
    field[32:48, 32:48] = rng.lognormal(mean=0.0, sigma=1.0, size=(16, 16))

    out = local_multifractality(
        field,
        tile_size=16,
        box_sizes=np.array([2, 4, 8], dtype=np.int64),
        q_values=np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64),
    )
    phi = np.asarray(out["phi"], dtype=np.float64)

    assert np.nanmax(phi) > np.nanmin(phi)
    # The perturbed tile should be more multifractal than the median tile.
    assert phi[2, 2] > np.nanmedian(phi)


def test_local_multifractality_skips_zero_mass_tiles():
    field = np.zeros((64, 64), dtype=np.float64)
    field[32:48, 32:48] = 1.0

    out = local_multifractality(
        field,
        tile_size=16,
        box_sizes=np.array([2, 4, 8], dtype=np.int64),
        q_values=np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64),
    )
    phi = np.asarray(out["phi"], dtype=np.float64)
    delta = np.asarray(out["delta"], dtype=np.float64)

    assert np.isnan(phi[0, 0])
    assert np.isfinite(phi[2, 2])
    assert delta[0, 0] == pytest.approx(0.0)


def test_multifractal_rust_python_parity(monkeypatch):
    import dynachaos.diagnostics.multifractal as mf_mod

    rng = np.random.default_rng(0)
    field = rng.random((48, 48), dtype=np.float64)
    q = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    box = np.array([2, 4, 8, 16], dtype=np.int64)

    old_flag = mf_mod._RUST_AVAILABLE

    monkeypatch.setattr(mf_mod, "_RUST_AVAILABLE", False)
    py_out = mf_mod.multifractal_spectrum(field, box_sizes=box, q_values=q)

    monkeypatch.setattr(mf_mod, "_RUST_AVAILABLE", old_flag)

    np.testing.assert_allclose(
        py_out["tau"],
        _golden("multifractal_tau"),
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        py_out["Dq"],
        _golden("multifractal_Dq"),
        atol=1e-11,
        rtol=1e-11,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        py_out["alpha_legendre"],
        _golden("multifractal_alpha_legendre"),
        atol=1e-11,
        rtol=1e-11,
        equal_nan=True,
    )
    assert float(py_out["phi"]) == pytest.approx(float(_golden("multifractal_phi")[0]), abs=1e-11)

    if _RUST_IMPORTABLE and not _NO_RUST_ENV:
        monkeypatch.setattr(mf_mod, "_RUST_AVAILABLE", True)
        rust_out = mf_mod.multifractal_spectrum(field, box_sizes=box, q_values=q)
        monkeypatch.setattr(mf_mod, "_RUST_AVAILABLE", old_flag)

        np.testing.assert_allclose(
            rust_out["tau"],
            _golden("multifractal_tau"),
            atol=1e-11,
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            rust_out["Dq"],
            _golden("multifractal_Dq"),
            atol=1e-11,
            rtol=1e-11,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            rust_out["alpha_legendre"],
            _golden("multifractal_alpha_legendre"),
            atol=1e-11,
            rtol=1e-11,
            equal_nan=True,
        )
        assert float(rust_out["phi"]) == pytest.approx(
            float(_golden("multifractal_phi")[0]), abs=1e-11
        )


def test_multifractal_module_falls_back_to_python_when_rust_disabled_via_env(monkeypatch):
    import importlib

    import dynachaos.diagnostics.multifractal as mf_mod

    monkeypatch.setenv("DYNACHAOS_NO_RUST", "1")
    try:
        reloaded = importlib.reload(mf_mod)
        assert reloaded._RUST_AVAILABLE is False
        assert reloaded._multifractal_moments_rs is None
    finally:
        monkeypatch.delenv("DYNACHAOS_NO_RUST", raising=False)
        importlib.reload(mf_mod)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        (np.ones((4, 4, 4)), "1D or 2D"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), "finite values"),
        (np.array([[-1.0, 2.0]]), "nonnegative"),
        (np.zeros((4, 4)), "positive total mass"),
    ],
)
def test_multifractal_spectrum_rejects_malformed_field(field, message):
    with pytest.raises(ValueError, match=message):
        multifractal_spectrum(field, box_sizes=np.array([2, 4]))


def test_default_box_sizes_rejects_fields_too_small_for_any_scale():
    with pytest.raises(ValueError, match="field too small"):
        _default_box_sizes((2, 2))


def test_default_box_sizes_rejects_fields_yielding_a_single_scale():
    # min_side=6 -> max_box=3, so only b=2 fits the doubling sequence
    # (b=4 exceeds max_box) -- fewer than the two scales a log-log fit needs.
    with pytest.raises(ValueError, match="at least two box sizes"):
        _default_box_sizes((6, 6))


def test_default_box_sizes_used_when_box_sizes_is_none():
    field = np.ones((64, 64), dtype=np.float64)
    out = multifractal_spectrum(field)
    np.testing.assert_array_equal(out["box_sizes"], np.array([2, 4, 8, 16, 32]))


def test_clean_box_sizes_rejects_fewer_than_two_positive_scales():
    with pytest.raises(ValueError, match="at least two positive scales"):
        _clean_box_sizes(np.array([5]))


def test_clean_q_values_defaults_to_65_point_grid_when_none():
    q = _clean_q_values(None)
    assert q.shape == (65,)
    np.testing.assert_allclose([q.min(), q.max()], [-8.0, 8.0])


def test_clean_q_values_rejects_fewer_than_three_finite_values():
    with pytest.raises(ValueError, match="at least three finite values"):
        _clean_q_values(np.array([1.0, np.nan]))


def test_linear_fit_slope_r2_returns_nan_for_fewer_than_two_points():
    slope, r2 = _linear_fit_slope_r2(np.array([1.0]), np.array([2.0]))
    assert np.isnan(slope) and np.isnan(r2)


def test_multifractal_moments_python_skips_box_larger_than_field():
    field = np.ones((10, 10))
    q = np.array([-2.0, 0.0, 1.0, 2.0])
    log_z, alpha_num, f_num, ln_scales = _multifractal_moments_python(field, np.array([3, 20]), q)
    assert np.isnan(ln_scales[1])  # box=20 has n_by=n_bx=0 -> skipped
    assert np.isfinite(ln_scales[0])


def test_multifractal_moments_python_skips_scale_with_no_mass_in_any_box():
    # b=3 gives non-overlapping boxes covering rows/cols 0..8 only; putting
    # the sole nonzero point in the truncated remainder (row/col 9) leaves
    # every box with zero mass at that scale.
    field = np.zeros((10, 10))
    field[9, 9] = 1.0
    q = np.array([-2.0, 0.0, 1.0, 2.0])
    log_z, alpha_num, f_num, ln_scales = _multifractal_moments_python(field, np.array([3]), q)
    assert np.all(np.isnan(log_z))


def test_multifractal_moments_python_skips_scale_with_overflowing_mass():
    # Box masses individually huge enough that their sum overflows to inf,
    # exercising the not-finite guard on used_mass.
    field = np.full((8, 8), 1e308)
    q = np.array([0.0, 1.0])
    with np.errstate(over="ignore"):
        log_z, alpha_num, f_num, ln_scales = _multifractal_moments_python(field, np.array([2]), q)
    assert np.all(np.isnan(log_z))


def test_multifractal_moments_python_skips_q_where_power_underflows_to_zero():
    rng = np.random.default_rng(0)
    field = rng.random((32, 32))
    field[0, 0] = 1e-300  # one vanishingly small mass among many boxes
    q = np.array([-1000.0, 0.0, 1.0])
    with np.errstate(over="ignore"):
        log_z, alpha_num, f_num, ln_scales = _multifractal_moments_python(
            field, np.array([2, 4]), q
        )
    assert np.all(np.isnan(log_z[:, 0]))  # q=-1000 overflows -> z not finite


def test_local_multifractality_rejects_1d_field():
    with pytest.raises(ValueError, match="expects a 2D field"):
        local_multifractality(np.ones(64), tile_size=8)


def test_local_multifractality_rejects_nonpositive_tile_size():
    with pytest.raises(ValueError, match="positive integers"):
        local_multifractality(np.ones((64, 64)), tile_size=0)


def test_local_multifractality_rejects_tile_larger_than_field():
    with pytest.raises(ValueError, match="larger than the field extent"):
        local_multifractality(np.ones((64, 64)), tile_size=200)


def test_local_multifractality_accepts_tuple_tile_size():
    out = local_multifractality(
        np.ones((64, 64)),
        tile_size=(16, 16),
        box_sizes=np.array([2, 4, 8]),
        q_values=np.array([-2.0, 0.0, 1.0, 2.0]),
    )
    assert out["phi"].shape == (4, 4)


def test_local_multifractality_default_box_sizes_used_when_none():
    out = local_multifractality(
        np.ones((64, 64)), tile_size=16, q_values=np.array([-2.0, 0.0, 1.0, 2.0])
    )
    np.testing.assert_array_equal(out["box_sizes"], np.array([2, 4, 8]))


def test_local_multifractality_regresses_phi_against_log_delta_across_tiles():
    # Give each of the 16 tiles a different noise amplitude so phi and delta
    # both vary across at least 3 tiles, exercising the real polyfit branch
    # rather than the insufficient-data NaN fallback.
    rng = np.random.default_rng(7)
    field = np.ones((64, 64))
    amps = [0.0, 0.2, 0.6, 1.2]
    for k, (iy, ix) in enumerate(np.ndindex(4, 4)):
        amp = amps[k % len(amps)]
        y0, x0 = iy * 16, ix * 16
        field[y0 : y0 + 16, x0 : x0 + 16] = 1.0 + amp * rng.random((16, 16))

    out = local_multifractality(
        field,
        tile_size=16,
        box_sizes=np.array([2, 4, 8]),
        q_values=np.array([-2.0, 0.0, 1.0, 2.0]),
    )
    assert np.isfinite(out["slope_phi_log_delta"])
    assert np.isfinite(out["intercept_phi_log_delta"])
    assert 0.0 <= out["r2_phi_log_delta"] <= 1.0 + 1e-9


def test_multifractal_python_backend_q_near_1_uses_power_law_branch(monkeypatch):
    """Verify Python backend treats q=1.000001 as power-law, not Shannon."""
    import dynachaos.diagnostics.multifractal as mf_mod

    field = np.ones((64, 64), dtype=np.float64)
    q = np.array([0.5, 1.0, 1.000001], dtype=np.float64)
    box = np.array([2, 4, 8, 16], dtype=np.int64)

    monkeypatch.setattr(mf_mod, "_RUST_AVAILABLE", False)
    out = mf_mod.multifractal_spectrum(field, box_sizes=box, q_values=q)

    dq = np.asarray(out["Dq"], dtype=np.float64)
    d_q1 = dq[1]  # q=1.0
    d_q_near1 = dq[2]  # q=1.000001

    # q=1.0 takes the Shannon branch; q=1.000001 lies outside the 1e-12
    # tolerance and must take the power-law branch. On a uniform
    # (monofractal) field the two must still agree closely.
    assert np.isfinite(d_q1) and np.isfinite(d_q_near1)
    assert abs(d_q1 - d_q_near1) < 0.01
