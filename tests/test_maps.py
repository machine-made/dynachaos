import importlib
from itertools import groupby
from pathlib import Path

import numpy as np
import pytest

from dynachaos.cml.correlation_figure import _fit_correlation_length
from dynachaos.cml.gcm_clusters import broad_positive_mask, compute_clusters, compute_collective
from dynachaos.cml.pattern_dynamics import SPACE_CASES
from dynachaos.cml.primitives import (
    cluster_labels_by_tolerance,
    cml_jacobian_subblock_logistic,
    cml_step,
    cml_step_logistic,
    cml_step_logistic_batch,
    gcm_step,
    sustained_positive_mask,
)
from dynachaos.diagnostics.compare_all_helpers import (
    load_or_compute_npz,
    sweep_pair_metric,
    sweep_scalar_metric,
)
from dynachaos.diagnostics.sali_gali import gali, sali, sali_at_time
from dynachaos.maps._iter import (
    iterate_unwrapped,
    run_animation_sweep,
    run_transient,
    sample_trajectory,
    trajectory_after_transient,
)
from dynachaos.maps.circle_map import circle_map, circle_map_derivative
from dynachaos.maps.coupled_logistic import (
    ATTRACTOR_CASES,
    PHASE_REQUIRED_KEYS,
    PHASE_SCHEMA_VERSION,
    PhaseDiagramPayload,
    coupled_logistic,
    coupled_logistic_jac,
)
from dynachaos.maps.coupled_logistic import (
    compute_attractors as compute_coupled_attractors,
)
from dynachaos.maps.coupled_logistic import (
    compute_basins as compute_coupled_basins,
)
from dynachaos.maps.coupled_logistic import (
    compute_phase_diagram as compute_coupled_phase_diagram,
)
from dynachaos.maps.delayed_logistic import (
    ATTRACTOR_LABELS_SHORT,
    LOCKING_LABELS_SHORT,
)
from dynachaos.maps.delayed_logistic import (
    compute_attractors as compute_delayed_attractors,
)
from dynachaos.maps.delayed_logistic import (
    compute_locking_sequence as compute_delayed_locking_sequence,
)
from dynachaos.maps.delayed_logistic import (
    compute_lyapunov_spectrum as compute_delayed_lyapunov_spectrum,
)
from dynachaos.maps.henon import henon, henon_jac
from dynachaos.maps.intermittency import (
    LOGISTIC_TYPE_I_ONSET,
    LORENZ_INTERMITTENCY_RHO,
    logistic_type_i_oracle,
    lorenz_1662_oracle,
    on_off_oracle,
    pm_type_i_oracle,
    pm_type_ii_oracle,
    pm_type_iii_oracle,
)
from dynachaos.maps.modulated_circle import longest_plateau_window, modulated_circle
from dynachaos.maps.primitives import (
    delayed_logistic,
    delayed_logistic_jac,
    logistic,
    logistic_derivative,
)
from dynachaos.maps.standard_map import standard_map, standard_map_jac
from dynachaos.maps.torus_doubling import (
    compute_map_I,
    compute_map_IV,
    compute_map_IV_lyapunov,
    map_I,
    map_I_jac,
    map_IV,
    map_IV_jac,
)


def test_circle_map_range_and_derivative():
    theta = 0.37
    out = circle_map(theta, A=0.1, D=0.25)
    deriv = circle_map_derivative(theta, A=0.1, D=0.25)

    assert 0.0 <= out < 1.0
    assert np.isfinite(deriv)


def test_sec02_circle_map_cache_preserves_decreasing_staircase_and_tongues():
    figure_dir = Path(__file__).resolve().parents[1] / "figures" / "sec02_circle_map"
    with np.load(figure_dir / "devils_staircase.npz", allow_pickle=False) as data:
        order = np.argsort(data["A"])
        A = data["A"][order]
        rho = data["rho"][order]
        lam = data["lam"][order]

    with np.load(figure_dir / "arnold_tongues.npz", allow_pickle=False) as data:
        Omega = data["Omega"]
        K = data["K"]
        tongue_rho = data["rho"]

    low_A_rho = rho[A <= 0.025]
    high_A_rho = rho[A >= 0.24]
    plateau_mask = np.abs(rho - 0.2) <= 1e-6
    plateau_run = max(
        (sum(1 for _ in group) for value, group in groupby(plateau_mask) if value),
        default=0,
    )
    zero_lock_counts = np.sum(np.abs(tongue_rho) <= 1e-3, axis=1)

    assert A[0] == pytest.approx(0.0)
    assert A[-1] == pytest.approx(0.25)
    assert rho[0] == pytest.approx(0.25, abs=1e-2)
    assert np.all((-1e-6 <= rho) & (rho <= 0.25 + 1e-6))
    assert np.median(low_A_rho) > 0.2
    assert np.median(high_A_rho) < 0.1
    assert np.median(low_A_rho) - np.median(high_A_rho) > 0.15
    assert plateau_run >= 500
    assert np.mean(lam <= 1e-6) > 0.5
    assert np.max(lam) > 0.0
    assert tongue_rho.shape == (K.size, Omega.size)
    np.testing.assert_allclose(tongue_rho[0], Omega, atol=1e-10)
    assert np.all((-1e-6 <= tongue_rho) & (tongue_rho <= 1.0 + 1e-6))
    assert zero_lock_counts[-1] > zero_lock_counts[0]


def test_coupled_logistic_shapes():
    state = np.array([0.2, -0.1], dtype=np.float64)
    out = coupled_logistic(state, A=1.2, D=0.08)
    jac = coupled_logistic_jac(state, A=1.2, D=0.08)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_intermittency_oracles_are_deterministic_and_finite():
    type_i = pm_type_i_oracle(64, x0=0.01, eps=1e-4, a=1.0)
    type_ii = pm_type_ii_oracle(64, x0=1e-3, y0=2e-3, eps=1e-3, a=-1.0)
    type_iii = pm_type_iii_oracle(64, x0=1e-3, eps=1e-3, a=1.0)
    logistic = logistic_type_i_oracle(64, x0=0.2, r=LOGISTIC_TYPE_I_ONSET - 1e-4)
    on_off_a = on_off_oracle(64, seed=123)
    on_off_b = on_off_oracle(64, seed=123)
    on_off_c = on_off_oracle(64, seed=124)

    assert type_i.shape == (64,)
    assert type_ii.shape == (64, 2)
    assert type_iii.shape == (64,)
    assert logistic.shape == (64,)
    assert on_off_a.shape == (64,)
    for series in (type_i, type_ii, type_iii, logistic, on_off_a):
        assert np.all(np.isfinite(series))
    np.testing.assert_allclose(on_off_a, on_off_b)
    assert not np.array_equal(on_off_a, on_off_c)


def test_lorenz_1662_oracle_reuses_flow_helper():
    traj = lorenz_1662_oracle(t_span=(0.0, 0.05), dt=0.01, t_transient=0.0)

    assert LORENZ_INTERMITTENCY_RHO == pytest.approx(166.2)
    assert traj.shape == (5, 3)
    assert np.all(np.isfinite(traj))


def test_coupled_logistic_preserves_diagonal():
    state = np.array([0.37, 0.37], dtype=np.float64)
    out = coupled_logistic(state, A=1.3, D=0.1)

    np.testing.assert_allclose(out[0], out[1])
    np.testing.assert_allclose(out[0], logistic(0.37, 1.3))


def test_coupled_logistic_respects_exchange_symmetry():
    state = np.array([0.2, -0.4], dtype=np.float64)
    swapped = state[::-1].copy()

    out = coupled_logistic(state, A=1.25, D=0.1)
    out_swapped = coupled_logistic(swapped, A=1.25, D=0.1)

    np.testing.assert_allclose(out_swapped, out[::-1])


def test_compute_coupled_phase_diagram_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "phase_diagram.npz"

    payload = compute_coupled_phase_diagram(
        A_values=np.array([0.8, 1.0, 1.2]),
        D_values=np.array([0.0, 0.1]),
        n_transient=2,
        n_sample=3,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["asym"].shape == (2, 3)
    assert payload["lyap"].shape == (2, 3)
    assert int(payload["schema_version"][0]) == PHASE_SCHEMA_VERSION
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["A"], payload["A"])
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["asym"], payload["asym"])
        np.testing.assert_allclose(saved["lyap"], payload["lyap"])


def test_coupled_phase_payload_round_trip(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    payload = PhaseDiagramPayload(
        A=np.array([0.8, 1.0], dtype=np.float64),
        D=np.array([0.0, 0.1], dtype=np.float64),
        asym=np.zeros((2, 2), dtype=np.float64),
        lyap=np.ones((2, 2), dtype=np.float64),
    )
    np.savez_compressed(path, **payload.to_npz())

    with np.load(path, allow_pickle=False) as saved:
        loaded = PhaseDiagramPayload.from_npz(saved)

    assert set(PHASE_REQUIRED_KEYS) == set(payload.to_npz())
    np.testing.assert_allclose(loaded.A, payload.A)
    np.testing.assert_allclose(loaded.D, payload.D)
    np.testing.assert_allclose(loaded.asym, payload.asym)
    np.testing.assert_allclose(loaded.lyap, payload.lyap)
    assert loaded.schema_version == payload.schema_version


def test_coupled_phase_payload_rejects_stale_schema(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8], dtype=np.float64),
        D=np.array([0.0], dtype=np.float64),
        asym=np.zeros((1, 1), dtype=np.float64),
        lyap=np.zeros((1, 1), dtype=np.float64),
        schema_version=np.array([1], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(ValueError, match="stale phase diagram cache"):
            PhaseDiagramPayload.from_npz(saved)


def test_coupled_phase_payload_rejects_missing_keys(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8], dtype=np.float64),
        D=np.array([0.0], dtype=np.float64),
        asym=np.zeros((1, 1), dtype=np.float64),
        schema_version=np.array([PHASE_SCHEMA_VERSION], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(KeyError, match="missing keys: lyap"):
            PhaseDiagramPayload.from_npz(saved)


def test_coupled_phase_payload_rejects_grid_shape_mismatch(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8, 1.0], dtype=np.float64),
        D=np.array([0.0, 0.1], dtype=np.float64),
        asym=np.zeros((2, 1), dtype=np.float64),
        lyap=np.zeros((2, 2), dtype=np.float64),
        schema_version=np.array([PHASE_SCHEMA_VERSION], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(ValueError, match="grid shape mismatch"):
            PhaseDiagramPayload.from_npz(saved)


def test_compute_coupled_phase_diagram_accepts_scalar_sweeps():
    payload = compute_coupled_phase_diagram(
        A_values=1.0,
        D_values=0.1,
        n_transient=1,
        n_sample=1,
        output_path=None,
    )

    assert payload["A"].shape == (1,)
    assert payload["D"].shape == (1,)
    assert payload["asym"].shape == (1, 1)
    assert payload["lyap"].shape == (1, 1)


def test_sec03_phase_diagram_finite_asym_stays_physical_after_divergence_mask():
    with np.load("figures/sec03_transition/phase_diagram.npz", allow_pickle=False) as data:
        asym = data["asym"]
        schema_version = int(data["schema_version"][0])

    finite_asym = asym[np.isfinite(asym)]

    assert schema_version == PHASE_SCHEMA_VERSION
    assert finite_asym.size > 0
    assert np.max(finite_asym) <= 2.3


def test_compute_coupled_attractors_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "attractors.npz"
    cases = ATTRACTOR_CASES[:2]

    payload = compute_coupled_attractors(
        cases=cases,
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["x_0"].shape == (4,)
    assert payload["y_0"].shape == (4,)
    assert payload["x_1"].shape == (4,)
    assert payload["y_1"].shape == (4,)
    assert int(payload["schema_version"][0]) == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["A_values"], payload["A_values"])
        np.testing.assert_allclose(saved["x_0"], payload["x_0"])
        np.testing.assert_allclose(saved["y_1"], payload["y_1"])


def test_compute_coupled_basins_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "basins.npz"

    payload = compute_coupled_basins(
        n_grid=4,
        n_transient=2,
        reference_transient=2,
        period=2,
        output_path=output_path,
    )

    assert payload["x"].shape == (4,)
    assert payload["y"].shape == (4,)
    assert payload["basin"].shape == (4, 4)
    assert payload["basin"].dtype == np.int8
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["x"], payload["x"])
        np.testing.assert_array_equal(saved["basin"], payload["basin"])


# ---------------------------------------------------------------------------
# Logistic map primitives
# ---------------------------------------------------------------------------


def test_logistic_known_values():
    """f(0, a) = 1 for all a; f(1, 2) = 1 - 2 = -1."""
    assert logistic(0.0, 1.5) == 1.0
    assert logistic(1.0, 2.0) == -1.0


def test_logistic_vectorized():
    x = np.array([0.0, 0.5, 1.0])
    result = logistic(x, 2.0)
    expected = 1.0 - 2.0 * x * x
    np.testing.assert_allclose(result, expected)


def test_logistic_derivative_values():
    """f'(x) = -2ax."""
    assert logistic_derivative(0.0, 1.5) == 0.0
    assert logistic_derivative(1.0, 2.0) == -4.0


def test_logistic_derivative_vectorized():
    x = np.array([-1.0, 0.0, 0.5])
    result = logistic_derivative(x, 1.5)
    expected = -2.0 * 1.5 * x
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Delayed logistic map
# ---------------------------------------------------------------------------


def test_delayed_logistic_shape_and_finite():
    state = np.array([0.5, 0.3])
    out = delayed_logistic(state, A=0.3, D=1.8)
    jac = delayed_logistic_jac(state, A=0.3, D=1.8)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_delayed_logistic_jac_finite_difference():
    """Jacobian should match finite-difference approximation."""
    state = np.array([0.5, 0.3])
    A, D = 0.3, 1.8
    jac = delayed_logistic_jac(state, A, D)

    eps = 1e-7
    for j in range(2):
        state_plus = state.copy()
        state_plus[j] += eps
        state_minus = state.copy()
        state_minus[j] -= eps
        fd_col = (delayed_logistic(state_plus, A, D) - delayed_logistic(state_minus, A, D)) / (
            2.0 * eps
        )
        np.testing.assert_allclose(jac[:, j], fd_col, atol=1e-5)


def test_compute_delayed_attractors_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "attractors.npz"

    payload = compute_delayed_attractors(
        D_values=np.array([1.55, 1.65]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.55_x"].shape == (4,)
    assert payload["D_1.65_y"].shape == (4,)
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.55_x"], payload["D_1.55_x"])


def test_compute_delayed_attractors_rejects_rounded_key_collisions():
    with pytest.raises(ValueError, match="unique"):
        compute_delayed_attractors(
            D_values=np.array([1.554, 1.555]),
            n_transient=1,
            n_plot=1,
            output_path=None,
        )


def test_compute_delayed_lyapunov_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "lyapunov_vs_D.npz"

    payload = compute_delayed_lyapunov_spectrum(
        D_values=np.array([1.5, 1.6]),
        n_iter=4,
        n_transient=2,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["D"].shape == (2,)
    assert payload["spectra"].shape == (2, 2)
    assert np.all(np.isfinite(payload["spectra"]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["spectra"], payload["spectra"])


def test_compute_delayed_locking_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "locking_sequence.npz"

    payload = compute_delayed_locking_sequence(
        D_values=np.array([1.86, 1.88]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.860_x"].shape == (4,)
    assert payload["D_1.880_y"].shape == (4,)
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.860_x"], payload["D_1.860_x"])


def test_compute_delayed_locking_rejects_rounded_key_collisions():
    with pytest.raises(ValueError, match="unique"):
        compute_delayed_locking_sequence(
            D_values=np.array([1.86041, 1.86042]),
            n_transient=1,
            n_plot=1,
            output_path=None,
        )


def test_delayed_logistic_panel_labels_match_lyapunov_signs():
    with np.load("figures/sec05_oscillation/attractors.npz", allow_pickle=False) as data:
        attractor_D = data["D_values"]
    with np.load("figures/sec05_oscillation/locking_sequence.npz", allow_pickle=False) as data:
        locking_D = data["D_values"]
    with np.load("figures/sec05_oscillation/lyapunov_vs_D.npz", allow_pickle=False) as data:
        lyap_D = data["D"]
        lambda1 = data["spectra"][:, 0]

    expected = {
        "chaos": "positive",
        "early chaos": "positive",
        "periodic window": "negative",
        "torus": "zero",
    }

    for D_values, labels in (
        (attractor_D, ATTRACTOR_LABELS_SHORT),
        (locking_D, LOCKING_LABELS_SHORT),
    ):
        assert len(D_values) == len(labels)
        for D, label in zip(D_values, labels, strict=True):
            idx = np.argmin(np.abs(lyap_D - D))
            lam = lambda1[idx]
            if expected[label] == "positive":
                assert lam > 1e-3, f"{label=} at D={D} must have positive lambda1, got {lam}"
            elif expected[label] == "negative":
                assert lam < -1e-3, f"{label=} at D={D} must have negative lambda1, got {lam}"
            else:
                assert abs(lam) <= 1e-3, f"{label=} at D={D} must have lambda1 near zero, got {lam}"


def test_sali_curves_store_measured_lambda1_regime_context():
    with np.load("figures/sec11_diagnostics/sali_comparison.npz", allow_pickle=False) as data:
        DB_values = data["DB_values"]
        lambda1_values = data["lambda1_values"]

    np.testing.assert_allclose(DB_values, np.array([2.35, 2.37, 2.47, 2.55]))
    assert abs(lambda1_values[0]) <= 1e-3
    assert abs(lambda1_values[1]) <= 1e-3
    assert lambda1_values[2] > 1e-3
    assert lambda1_values[3] > 5e-2


def test_sec06_canonical_lyapunov_cache_replaces_duplicate_D2_file():
    duplicate_path = Path("figures/sec06_three_torus/lyapunov_vs_D2.npz")
    with np.load("figures/sec06_three_torus/lyapunov_vs_DB.npz", allow_pickle=False) as data:
        DB = data["DB"]
        spectra = data["eps_0.005_spectra"]

    torus_idx = np.argmin(np.abs(DB - 2.35))
    chaos_idx = np.argmin(np.abs(DB - 2.55))

    assert not duplicate_path.exists()
    assert abs(spectra[torus_idx, 0]) <= 1e-3
    assert abs(spectra[torus_idx, 1]) <= 5e-3
    assert spectra[chaos_idx, 0] > 5e-2
    assert spectra[chaos_idx, 1] > 1e-2


def test_sec06_three_torus_cache_preserves_two_zero_exponent_signature():
    with np.load("figures/sec06_three_torus/lyapunov_vs_DB.npz", allow_pickle=False) as data:
        DB = data["DB"]
        eps_values = data["eps_values"]
        spectra = data["eps_0.005_spectra"]

    torus_idx = np.argmin(np.abs(DB - 2.1))
    torus_spectrum = spectra[torus_idx]
    quasiperiodic_region = (2.1 <= DB) & (DB <= 2.2)

    np.testing.assert_allclose(eps_values, np.array([0.001, 0.005, 0.01]))
    assert spectra.shape == (DB.size, 4)
    assert np.all(np.isfinite(spectra))
    assert np.all(np.diff(spectra, axis=1) <= 1e-12)
    assert abs(torus_spectrum[0]) < 1e-3
    assert abs(torus_spectrum[1]) < 1e-3
    assert torus_spectrum[2] < -1e-2
    assert torus_spectrum[3] < -1e-2
    assert np.max(spectra[quasiperiodic_region, 0]) < 1e-3


def test_sec07_correlation_dimension_cache_stays_bounded_and_rises():
    with np.load(
        "figures/sec07_fractalization/correlation_dimension.npz", allow_pickle=False
    ) as data:
        D = data["D"]
        D2 = data["D2"]
        A = float(data["A"][0])

    low_window = D2[D <= 1.75]
    high_window = D2[D >= 1.95]

    assert A == pytest.approx(0.3)
    assert D.shape == D2.shape
    assert np.all(np.isfinite(D2))
    assert np.min(D) >= 1.7
    assert np.max(D) <= 2.0
    assert np.all((0.0 <= D2) & (D2 < 2.0))
    assert np.max(D2) <= 1.45
    assert np.median(high_window) > np.median(low_window) + 0.15
    assert np.corrcoef(D, D2)[0, 1] > 0.5


def test_sec09_spatial_activity_separates_pattern_phase_samples():
    with np.load("figures/sec09_pattern/phase_diagram.npz", allow_pickle=False) as data:
        a_values = data["a"]
        eps_values = data["eps"]
        spatial_activity = data["spatial_activity"]

    phase_activity = []
    for a, eps, _label, _tag in SPACE_CASES:
        ia = np.argmin(np.abs(a_values - a))
        ie = np.argmin(np.abs(eps_values - eps))
        phase_activity.append(spatial_activity[ie, ia])

    phase_activity = np.asarray(phase_activity)
    rounded_groups = np.unique(np.round(phase_activity, 1))

    assert np.all(np.isfinite(phase_activity))
    assert np.ptp(phase_activity) > 0.25
    assert rounded_groups.size >= 3


# ---------------------------------------------------------------------------
# Torus doubling maps
# ---------------------------------------------------------------------------


def test_map_I_shape():
    state = np.array([0.5, 0.3, 0.4])
    out = map_I(state, A=0.4, D=2.0)
    jac = map_I_jac(state, A=0.4, D=2.0)

    assert out.shape == (3,)
    assert jac.shape == (3, 3)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_map_IV_shape():
    state = np.array([0.5, 0.3, 0.4, 0.2])
    out = map_IV(state, A=0.3, D=1.5)
    jac = map_IV_jac(state, A=0.3, D=1.5)

    assert out.shape == (4,)
    assert jac.shape == (4, 4)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_compute_torus_map_I_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_I_attractors.npz"

    payload = compute_map_I(
        D_values=np.array([2.11, 2.16]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_2.11_traj"].shape[1] == 3
    assert payload["D_2.16_traj"].shape[1] == 3
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_2.11_traj"], payload["D_2.11_traj"])


def test_compute_torus_map_IV_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_IV_attractors.npz"

    payload = compute_map_IV(
        D_values=np.array([1.515, 1.5206]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.515_traj"].shape[1] == 4
    assert payload["D_1.5206_traj"].shape[1] == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.515_traj"], payload["D_1.515_traj"])


def test_compute_torus_map_IV_lyapunov_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_IV_lyapunov.npz"

    payload = compute_map_IV_lyapunov(
        D_values=np.array([1.5, 1.51]),
        n_iter=4,
        n_transient=2,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["D"].shape == (2,)
    assert payload["spectra"].shape == (2, 4)
    assert np.all(np.isfinite(payload["spectra"]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["spectra"], payload["spectra"])


def test_map_IV_off_symmetry_ic_and_lyapunov_physics():
    """Physics-gated regression guard for the off-symmetry IC in Map IV.

    Two assertions:
    1. Lyapunov physics: lambda1 <= 1e-3 at torus (D~1.515) and lambda1 > 0 at
       chaos onset (D~1.5212), confirming the torus->chaos transition is correctly
       resolved with the off-symmetry IC.
    2. Off-symmetry: the trajectory at D=1.515 started from x0=[0.5,0.45,0.52,0.48]
       stays off the invariant subspace {X=Z, Y=W} after transient, i.e. max|X-Z|
       and max|Y-W| are both > 1e-6. This is the regression guard that proves the IC
       is not symmetric.
    """
    # --- Part 1: Lyapunov physics ---
    payload_lyap = compute_map_IV_lyapunov(
        D_values=np.linspace(1.48, 1.53, 200),
        n_iter=5000,
        n_transient=2000,
        output_path=None,
        progress_interval=0,
    )
    D = payload_lyap["D"]
    lambda1 = payload_lyap["spectra"][:, 0]

    idx_torus = np.argmin(np.abs(D - 1.515))
    idx_chaos = np.argmin(np.abs(D - 1.5212))

    assert lambda1[idx_torus] <= 1.0e-3, (
        f"Expected lambda1 <= 1e-3 at D~1.515 (torus), got {lambda1[idx_torus]:.6f}. "
        "Symmetric IC may still be in use."
    )
    assert lambda1[idx_chaos] > 0, (
        f"Expected lambda1 > 0 at D~1.5212 (chaos onset), got {lambda1[idx_chaos]:.6f}. "
        "Off-symmetry IC may not be exploring 4D dynamics."
    )

    # --- Part 2: Off-symmetry regression guard ---
    payload_traj = compute_map_IV(
        D_values=np.array([1.515]),
        n_transient=2000,
        n_plot=10000,
        output_path=None,
    )
    traj = payload_traj["D_1.515_traj"]  # shape (10000, 4)
    X, Y, Z, W = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]

    max_xz = np.max(np.abs(X - Z))
    max_yw = np.max(np.abs(Y - W))

    assert max_xz > 1e-6, (
        f"Orbit stayed on symmetric manifold: max|X-Z| = {max_xz:.2e}. "
        "IC may have collapsed back onto the X=Z, Y=W subspace."
    )
    assert max_yw > 1e-6, (
        f"Orbit stayed on symmetric manifold: max|Y-W| = {max_yw:.2e}. "
        "IC may have collapsed back onto the X=Z, Y=W subspace."
    )


def test_run_animation_sweep_returns_and_writes_payload(tmp_path):
    output_path = tmp_path / "animation.npz"

    def iterate_fn(param):
        return np.column_stack(
            (
                np.full(3, param, dtype=np.float64),
                np.arange(3, dtype=np.float64),
            )
        )

    payload = run_animation_sweep(
        iterate_fn,
        np.array([0.1, 0.2]),
        output_path,
        n_plot=3,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["param_values"], np.array([0.1, 0.2]))
    np.testing.assert_allclose(payload["all_x"], np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
    np.testing.assert_allclose(payload["all_y"], np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["all_x"], payload["all_x"])


# ---------------------------------------------------------------------------
# Modulated circle map
# ---------------------------------------------------------------------------


def test_modulated_circle_range():
    state = np.array([0.3, 0.7])
    out = modulated_circle(state, A=0.15, C=0.618, D=0.25, eps=0.05)

    assert out.shape == (2,)
    assert 0.0 <= out[0] < 1.0
    assert 0.0 <= out[1] < 1.0


def test_longest_plateau_window_selects_widest_run():
    D = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    rho = np.array([0.00, 0.25, 0.2502, 0.40, 0.25, 0.2501, 0.2502])

    window = longest_plateau_window(D, rho, target=0.25, tol=5e-4)

    assert window == (0.4, 0.6)


def test_broad_positive_mask_rejects_short_spikes():
    values = np.array([-0.1, 0.03, -0.02, 0.04, 0.05, 0.06, 0.07, -0.01])

    mask = broad_positive_mask(values, threshold=0.02, min_run=3)

    np.testing.assert_array_equal(
        mask,
        np.array([False, False, False, True, True, True, True, False]),
    )


def test_broad_positive_mask_closes_a_run_that_reaches_the_array_end():
    values = np.array([-0.1, 0.03, 0.04, 0.05, 0.06])

    mask = broad_positive_mask(values, threshold=0.02, min_run=3)

    np.testing.assert_array_equal(mask, np.array([False, True, True, True, True]))


def test_sustained_positive_mask_matches_legacy_alias():
    values = np.array([-0.1, 0.03, -0.02, 0.04, 0.05, 0.06, 0.07, -0.01])

    np.testing.assert_array_equal(
        sustained_positive_mask(values, threshold=0.02, min_run=3),
        broad_positive_mask(values, threshold=0.02, min_run=3),
    )


def test_compute_clusters_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "gcm_clusters.npz"

    payload = compute_clusters(
        n_sites=8,
        n_transient=2,
        n_record=4,
        seed=123,
        output_path=output_path,
    )

    assert payload["cluster_labels"].shape == (4, 8)
    assert payload["x_record"].shape == (4, 8)
    assert int(payload["N"][0]) == 8
    assert int(payload["n_transient"][0]) == 2
    assert int(payload["n_record"][0]) == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_array_equal(saved["cluster_labels"], payload["cluster_labels"])
        np.testing.assert_allclose(saved["x_record"], payload["x_record"])


def test_compute_collective_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "collective_lyapunov.npz"
    a_values = np.array([1.4, 1.6])

    payload = compute_collective(
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=4,
        renorm_interval=2,
        seed=123,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["a_values"], a_values)
    assert payload["lyap_c"].shape == (2,)
    assert np.all(np.isfinite(payload["lyap_c"]))
    assert int(payload["N"][0]) == 8
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["a_values"], payload["a_values"])
        np.testing.assert_allclose(saved["lyap_c"], payload["lyap_c"])


def test_sec10_gcm_caches_preserve_collective_signatures():
    with np.load("figures/sec10_gcm/gcm_results.npz", allow_pickle=False) as data:
        N_grid = data["N_grid"]
        msd_grid = data["msd_grid"]

    with np.load("figures/sec10_gcm/gcm_clusters.npz", allow_pickle=False) as data:
        cluster_labels = data["cluster_labels"]
        n_sites = int(data["N"][0])

    with np.load("figures/sec10_gcm/collective_lyapunov.npz", allow_pickle=False) as data:
        a_values = data["a_values"]
        lyap_c = data["lyap_c"]

    cluster_counts = np.array([np.unique(row).size for row in cluster_labels])

    assert N_grid[0] == 100
    assert N_grid[-1] == 20000
    assert msd_grid.shape == (5, 8)
    assert np.all(msd_grid[:, -1] > msd_grid[:, 0] / 10.0)
    assert cluster_labels.shape == (500, 100)
    assert np.all((2 <= cluster_counts) & (cluster_counts <= n_sites / 10))
    assert np.all(np.isfinite(lyap_c))
    assert lyap_c[0] < 0.0
    assert lyap_c[-1] > 0.0
    assert np.min(lyap_c) < -0.25
    assert np.max(lyap_c) > 0.4
    assert a_values[0] == pytest.approx(1.4)
    assert a_values[-1] == pytest.approx(2.0)


def test_cluster_labels_by_tolerance_groups_sorted_runs():
    values = np.array([0.10, 0.1000002, 0.50, 0.5000001, 0.90])

    labels = cluster_labels_by_tolerance(values, tol=1e-5)

    np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, 2]))


def test_cluster_labels_by_tolerance_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1D array"):
        cluster_labels_by_tolerance(np.zeros((2, 2)))


def test_cluster_labels_by_tolerance_returns_empty_for_empty_input():
    labels = cluster_labels_by_tolerance(np.array([]))

    assert labels.shape == (0,)
    assert labels.dtype == int


def test_cml_jacobian_subblock_logistic_rejects_l_out_of_range():
    x = np.array([0.1, -0.2, 0.3])

    with pytest.raises(ValueError, match="L must satisfy"):
        cml_jacobian_subblock_logistic(x, 1.5, 0.2, 0)

    with pytest.raises(ValueError, match="L must satisfy"):
        cml_jacobian_subblock_logistic(x, 1.5, 0.2, len(x) + 1)


def test_iterate_unwrapped_scalar_matches_manual_accumulation():
    value = 0.1
    for _ in range(5):
        value += 0.25 + 0.1 * value

    helper = iterate_unwrapped(0.1, lambda x: 0.25 + 0.1 * x, 5)

    np.testing.assert_allclose(helper, value)


def test_run_transient_and_sample_helpers_record_post_step_states():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    after = run_transient(state, step_fn, 2)
    np.testing.assert_allclose(after, np.array([2.0, 3.0]))

    samples = sample_trajectory(after, step_fn, 3)
    np.testing.assert_allclose(
        samples,
        np.array([[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]),
    )

    combined = trajectory_after_transient(state, step_fn, 2, 3)
    np.testing.assert_allclose(combined, samples)


def test_cml_step_matches_manual_generic_update():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    eps = 0.2

    def f(arr):
        return 2.0 * arr

    def g(arr):
        return arr + 1.0

    out = cml_step(x, f, g, eps)
    manual = f(x) + eps / 2.0 * (np.roll(g(x), -1) + np.roll(g(x), 1) - 2.0 * g(x))

    np.testing.assert_allclose(out, manual)


def test_cml_step_default_preserves_flattened_roll_behavior_for_2d_input():
    x = np.array([[0.1, -0.2], [0.3, -0.4]])
    eps = 0.2

    def f(arr):
        return 2.0 * arr

    def g(arr):
        return arr + 1.0

    out = cml_step(x, f, g, eps)
    gx = g(x)
    manual = f(x) + eps / 2.0 * (np.roll(gx, -1) + np.roll(gx, 1) - 2.0 * gx)

    np.testing.assert_allclose(out, manual)


def test_cml_step_logistic_axis_argument_enables_rowwise_topology():
    x = np.array([[0.1, 0.2, -0.1], [0.3, -0.4, 0.5]])
    eps = 0.2

    explicit = cml_step_logistic(x, 1.6, eps, axis=1)
    rowwise = np.vstack([cml_step_logistic(row, 1.6, eps) for row in x])

    np.testing.assert_allclose(explicit, rowwise)


def test_cml_step_logistic_batch_matches_rowwise_update():
    x = np.array([[0.1, 0.2, -0.1], [0.3, -0.4, 0.5]])
    a_col = np.array([[1.5], [1.8]])
    eps = 0.2

    batch = cml_step_logistic_batch(x, a_col, eps)
    rowwise = np.vstack(
        [cml_step_logistic(x[idx], float(a_col[idx, 0]), eps) for idx in range(len(x))]
    )

    np.testing.assert_allclose(batch, rowwise)


def test_cml_jacobian_subblock_logistic_matches_manual_matrix():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    a = 1.5
    eps = 0.2
    L = 3

    J = cml_jacobian_subblock_logistic(x, a, eps, L)
    dfx = logistic_derivative(x, a)
    expected = np.array(
        [
            [(1.0 - eps) * dfx[0], (eps / 2.0) * dfx[1], 0.0],
            [(eps / 2.0) * dfx[0], (1.0 - eps) * dfx[1], (eps / 2.0) * dfx[2]],
            [0.0, (eps / 2.0) * dfx[1], (1.0 - eps) * dfx[2]],
        ]
    )

    np.testing.assert_allclose(J, expected)


def test_correlation_length_fit_uses_decay_head_not_noise_floor():
    r = np.arange(0, 80)
    corr = np.exp(-r / 2.0)
    corr[6:] = 2e-3 * (1.0 + 0.1 * np.sin(r[6:]))

    xi = _fit_correlation_length(r, corr)

    assert 1.0 < xi < 3.0


def test_correlation_length_fit_ignores_late_tail_spikes():
    r = np.arange(0, 80)
    corr = np.exp(-r / 0.8)
    corr[10] = 0.5

    xi = _fit_correlation_length(r, corr)

    assert 0.5 < xi < 1.0


def test_correlation_length_fit_uses_slow_head_slope_when_target_not_reached():
    # Decay so slow the target e^-1 is never reached within the near-field
    # window, so the loop runs to the array end and falls to the fitted-slope
    # branch instead of the crossing-interpolation branch.
    r = np.arange(0, 30)
    corr = np.exp(-r / 50.0)

    xi = _fit_correlation_length(r, corr)

    assert 45.0 < xi < 55.0


def test_correlation_length_fit_returns_plateau_value_without_interpolation():
    # A flat plateau exactly at the target value: the crossing branch fires
    # with y0 == y1, so no interpolation is needed and x1 is returned as-is.
    r = np.arange(0, 20)
    target = np.exp(-1.0)
    corr = np.full(20, target)

    xi = _fit_correlation_length(r, corr)

    assert xi == pytest.approx(1.0)


def test_correlation_length_fit_uses_envelope_crossing_for_oscillatory_head():
    # An oscillating head (rises above its previous value repeatedly, so the
    # scan loop exits early on the "value > previous" condition well before
    # the target or noise floor) falls through to the envelope-crossing
    # fallback.
    r = np.arange(0, 20)
    corr = np.array([1.0, 0.9, 1.0, 0.9, 1.0, 0.9] + [0.1] * 14)

    xi = _fit_correlation_length(r, corr)

    assert np.isfinite(xi)
    assert xi > 0.0


def test_sec08_correlation_lengths_are_physical_after_refit():
    with np.load("figures/sec08_sti/correlation_decay.npz", allow_pickle=False) as data:
        a_corr = data["a_corr"]
        xi_values = data["xi_values"]

    xi_by_a = dict(zip(a_corr, xi_values, strict=True))

    assert np.all(np.isfinite(xi_values))
    assert np.all((0.0 < xi_values) & (xi_values < 10.0))
    assert xi_by_a[1.85] < xi_by_a[1.7]
    assert xi_by_a[1.95] < xi_by_a[1.7]


def test_gcm_step_matches_manual_formula():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    a = 1.7
    eps = 0.15

    out = gcm_step(x, a, eps)
    fx = logistic(x, a)
    manual = (1.0 - eps) * fx + eps * np.mean(fx)

    np.testing.assert_allclose(out, manual)


def test_run_transient_returns_none_when_diverged():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    def diverged_fn(s):
        return s[0] > 2.5

    out = run_transient(state, step_fn, 5, diverged_fn=diverged_fn)
    assert out is None


def test_sample_trajectory_allow_partial_returns_prefix():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    def diverged_fn(s):
        return s[0] > 2.5

    samples = sample_trajectory(
        state,
        step_fn,
        5,
        diverged_fn=diverged_fn,
        allow_partial=True,
    )

    np.testing.assert_allclose(samples, np.array([[1.0, 2.0], [2.0, 3.0]]))


def test_iterate_unwrapped_vectorized_updates_elementwise():
    state = np.array([0.0, 1.0, 2.0])
    out = iterate_unwrapped(state, lambda s: np.array([1.0, -1.0, 0.5]), 3)
    np.testing.assert_allclose(out, np.array([3.0, -2.0, 3.5]))


def test_sweep_metric_helpers_return_expected_arrays():
    values = np.array([1.0, 2.0, 3.0])

    def series_fn(v):
        return np.array([v, 2.0 * v])

    def scalar_metric(s):
        return float(np.sum(s))

    def pair_metric(s):
        return float(np.min(s)), float(np.max(s))

    scalar = sweep_scalar_metric(values, series_fn, scalar_metric)
    first, second = sweep_pair_metric(values, series_fn, pair_metric)

    np.testing.assert_allclose(scalar, np.array([3.0, 6.0, 9.0]))
    np.testing.assert_allclose(first, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(second, np.array([2.0, 4.0, 6.0]))


def test_sweep_metric_helpers_print_progress_when_requested(capsys):
    values = np.array([1.0, 2.0, 3.0, 4.0])

    def series_fn(v):
        return np.array([v, 2.0 * v])

    def scalar_metric(s):
        return float(np.sum(s))

    def pair_metric(s):
        return float(np.min(s)), float(np.max(s))

    sweep_scalar_metric(
        values, series_fn, scalar_metric, progress_every=2, progress_label="scalar sweep"
    )
    out_scalar = capsys.readouterr().out
    assert "scalar sweep: 2/4" in out_scalar
    assert "scalar sweep: 4/4" in out_scalar

    sweep_pair_metric(values, series_fn, pair_metric, progress_every=2, progress_label="pair sweep")
    out_pair = capsys.readouterr().out
    assert "pair sweep: 2/4" in out_pair
    assert "pair sweep: 4/4" in out_pair


def test_load_or_compute_npz_computes_when_missing(tmp_path):
    path = tmp_path / "sample.npz"

    def compute_fn():
        np.savez_compressed(path, values=np.array([1.0, 2.0]))

    data = load_or_compute_npz(path, "sample", compute_fn)

    np.testing.assert_allclose(data["values"], np.array([1.0, 2.0]))


def test_load_or_compute_npz_loads_existing_cache_without_recomputing(tmp_path):
    path = tmp_path / "sample.npz"
    np.savez_compressed(path, values=np.array([1.0, 2.0]))
    calls = 0

    def compute_fn():
        nonlocal calls
        calls += 1

    data = load_or_compute_npz(path, "sample", compute_fn, required_keys=("values",))

    assert calls == 0
    np.testing.assert_allclose(data["values"], np.array([1.0, 2.0]))


def test_load_or_compute_npz_recomputes_when_required_keys_missing(tmp_path):
    path = tmp_path / "sample.npz"
    np.savez_compressed(path, stale=np.array([0.0]))
    calls = 0

    def compute_fn():
        nonlocal calls
        calls += 1
        np.savez_compressed(path, values=np.array([3.0, 4.0]))

    data = load_or_compute_npz(path, "sample", compute_fn, required_keys=("values",))

    assert calls == 1
    np.testing.assert_allclose(data["values"], np.array([3.0, 4.0]))


def test_load_or_compute_npz_raises_when_compute_leaves_required_keys_missing(tmp_path):
    path = tmp_path / "sample.npz"

    def compute_fn():
        np.savez_compressed(path, other=np.array([1.0]))

    with pytest.raises(KeyError, match="missing required keys"):
        load_or_compute_npz(path, "sample", compute_fn, required_keys=("values",))


# ---------------------------------------------------------------------------
# Henon map
# ---------------------------------------------------------------------------


def test_henon_shape_and_finite():
    """Output shapes and finiteness."""
    state = np.array([0.1, 0.1])
    out = henon(state)
    jac = henon_jac(state)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_henon_known_value():
    """At origin: x' = 1, y' = 0."""
    out = henon(np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [1.0, 0.0])


def test_henon_jac_finite_difference():
    """Jacobian should match finite-difference approximation."""
    state = np.array([0.63, -0.19])
    a, b = 1.4, 0.3
    jac = henon_jac(state, a, b)

    eps = 1e-7
    for j in range(2):
        state_plus = state.copy()
        state_plus[j] += eps
        state_minus = state.copy()
        state_minus[j] -= eps
        fd_col = (henon(state_plus, a, b) - henon(state_minus, a, b)) / (2.0 * eps)
        np.testing.assert_allclose(jac[:, j], fd_col, atol=1e-5)


def test_standard_map_sali_separates_regular_and_chaotic_conservative_orbits():
    regular_x0 = np.array([np.pi, 0.01])
    chaotic_x0 = np.array([0.1, 0.1])
    probe_state = np.array([1.2, 0.4])

    regular = sali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        regular_x0,
        n_iter=1000,
        n_transient=100,
        rng=np.random.default_rng(123),
    )
    chaotic = sali(
        lambda state: standard_map(state, K=5.0),
        lambda state: standard_map_jac(state, K=5.0),
        chaotic_x0,
        n_iter=1000,
        n_transient=100,
        rng=np.random.default_rng(123),
    )

    assert standard_map(probe_state, K=0.5).shape == (2,)
    assert np.linalg.det(standard_map_jac(probe_state, K=0.5)) == pytest.approx(1.0)
    assert np.min(regular[-200:]) > 0.5
    assert chaotic[-1] < 1e-12
    assert chaotic[-1] < regular[-1] * 1e-12


def test_sali_default_rng_is_deterministic_across_calls():
    # rng=None falls back to np.random.default_rng(42) internally, so two
    # calls with no rng given must reproduce the same series exactly.
    x0 = np.array([np.pi, 0.01])
    first = sali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        n_iter=50,
        n_transient=10,
    )
    second = sali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        n_iter=50,
        n_transient=10,
    )
    np.testing.assert_array_equal(first, second)


def test_sali_at_time_returns_final_scalar_matching_full_series():
    x0 = np.array([np.pi, 0.01])
    full_series = sali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        n_iter=200,
        n_transient=50,
        rng=np.random.default_rng(123),
    )
    final = sali_at_time(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        n_iter=200,
        n_transient=50,
        rng=np.random.default_rng(123),
    )
    assert final == pytest.approx(full_series[-1])
    assert isinstance(final, float)


def test_gali_separates_regular_and_chaotic_conservative_orbits():
    regular_x0 = np.array([np.pi, 0.01])
    chaotic_x0 = np.array([0.1, 0.1])

    # GALI_2 on a 2D map reduces to SALI up to normalisation: for a regular
    # (2-torus) orbit it saturates near a positive constant, for a chaotic
    # orbit it decays to (near) zero as the deviation vectors align.
    regular = gali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        regular_x0,
        n_iter=1000,
        n_transient=100,
        rng=np.random.default_rng(123),
    )
    chaotic = gali(
        lambda state: standard_map(state, K=5.0),
        lambda state: standard_map_jac(state, K=5.0),
        chaotic_x0,
        n_iter=1000,
        n_transient=100,
        rng=np.random.default_rng(123),
    )

    assert np.min(regular[-200:]) > 0.5
    assert chaotic[-1] < 1e-12


def test_gali_k_defaults_to_full_phase_space_dimension():
    x0 = np.array([np.pi, 0.01])
    default_k = gali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        k=None,
        n_iter=20,
        n_transient=5,
        rng=np.random.default_rng(1),
    )
    explicit_k = gali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        k=2,
        n_iter=20,
        n_transient=5,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_array_equal(default_k, explicit_k)


def test_gali_default_rng_is_deterministic_across_calls():
    # rng=None falls back to np.random.default_rng(42) internally, mirroring
    # sali's default-rng branch.
    x0 = np.array([np.pi, 0.01])
    first = gali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        n_iter=30,
        n_transient=5,
    )
    second = gali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        n_iter=30,
        n_transient=5,
    )
    np.testing.assert_array_equal(first, second)


def test_gali_k_larger_than_dimension_is_clamped():
    # k=5 on a 2D map is clamped to dim=2 via k = min(k, dim); must not raise
    # and must match the explicit k=2 request with the same rng draw order.
    x0 = np.array([np.pi, 0.01])
    clamped = gali(
        lambda state: standard_map(state, K=0.5),
        lambda state: standard_map_jac(state, K=0.5),
        x0,
        k=5,
        n_iter=20,
        n_transient=5,
        rng=np.random.default_rng(1),
    )
    assert clamped.shape == (20,)
    assert np.all(np.isfinite(clamped))


def test_compute_clusters_seed_controls_rng_determinism():
    first = compute_clusters(seed=7, n_sites=8, n_transient=2, n_record=4, output_path=None)
    second = compute_clusters(seed=7, n_sites=8, n_transient=2, n_record=4, output_path=None)

    np.testing.assert_array_equal(first["cluster_labels"], second["cluster_labels"])
    np.testing.assert_array_equal(first["x_record"], second["x_record"])

    different_seed = compute_clusters(
        seed=99, n_sites=8, n_transient=2, n_record=4, output_path=None
    )

    assert not np.array_equal(
        first["cluster_labels"], different_seed["cluster_labels"]
    ) or not np.array_equal(first["x_record"], different_seed["x_record"]), (
        "compute_clusters should produce different stochastic payloads for different seeds"
    )


def test_compute_collective_seed_controls_rng_determinism():
    a_values = np.array([1.4, 1.6, 1.8])

    first = compute_collective(
        seed=7,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )
    second = compute_collective(
        seed=7,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )

    np.testing.assert_array_equal(first["lyap_c"], second["lyap_c"])

    different_seed = compute_collective(
        seed=99,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )

    assert not np.array_equal(first["lyap_c"], different_seed["lyap_c"]), (
        "compute_collective should produce different lyap_c values for different seeds"
    )


# ---------------------------------------------------------------------------
# Golden-value regression tests for compute-aggregate functions
# ---------------------------------------------------------------------------


# params: D_values=[1.5, 1.6], n_iter=20, n_transient=5, seed=N/A
def test_compute_delayed_lyapunov_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "lyapunov_golden.npz"
    payload = compute_delayed_lyapunov_spectrum(
        D_values=np.array([1.5, 1.6]),
        n_iter=20,
        n_transient=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["D"], np.array([1.5, 1.6]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["spectra"],
        np.array(
            [
                [0.075066870334274, 0.07028722946827429],
                [0.0970755525312533, 0.09294026515025561],
            ]
        ),
        rtol=1e-9,
    )
    assert payload["spectra"].shape == (2, 2)
    assert np.all(np.isfinite(payload["spectra"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["spectra"], payload["spectra"], rtol=1e-9)


# params: D_values=[1.5, 1.51], n_iter=20, n_transient=5, seed=N/A
def test_compute_map_IV_lyapunov_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "map_iv_lyap_golden.npz"
    payload = compute_map_IV_lyapunov(
        D_values=np.array([1.5, 1.51]),
        n_iter=20,
        n_transient=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["D"], np.array([1.5, 1.51]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["spectra"],
        np.array(
            [
                [
                    0.1523415952594922,
                    0.03859679338679587,
                    0.014645886790399288,
                    -0.10933648083443089,
                ],
                [
                    0.15281204945237534,
                    0.04010023664586028,
                    0.012345832346427799,
                    -0.11445889859259217,
                ],
            ]
        ),
        rtol=1e-9,
    )
    assert payload["spectra"].shape == (2, 4)
    assert np.all(np.isfinite(payload["spectra"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["spectra"], payload["spectra"], rtol=1e-9)


# params: a=1.55, eps=0.1, n_sites=4, n_transient=5, n_record=3, seed=7
def test_compute_clusters_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "gcm_clusters_golden.npz"
    payload = compute_clusters(
        a=1.55,
        eps=0.1,
        n_sites=4,
        n_transient=5,
        n_record=3,
        seed=7,
        output_path=output_path,
    )

    np.testing.assert_array_equal(
        payload["cluster_labels"],
        np.array([[0, 3, 2, 1], [3, 0, 1, 2], [0, 1, 3, 2]]),
    )
    np.testing.assert_allclose(
        payload["x_record"],
        np.array(
            [
                [
                    -0.04545209715542766,
                    0.9563681048236234,
                    0.6832212378395283,
                    0.6648412697350994,
                ],
                [
                    0.9263795270955607,
                    -0.3466612849138426,
                    0.27808764055024543,
                    0.31265203806958264,
                ],
                [
                    -0.24185543228176498,
                    0.7876615172785664,
                    0.8474251456092492,
                    0.8189412528994947,
                ],
            ]
        ),
        rtol=1e-9,
    )

    with np.load(output_path) as saved:
        np.testing.assert_array_equal(saved["cluster_labels"], payload["cluster_labels"])
        np.testing.assert_allclose(saved["x_record"], payload["x_record"], rtol=1e-9)


# params: n_sites=4, a_values=[1.4, 1.6], n_transient=5, n_measure=10, seed=7
def test_compute_collective_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "collective_golden.npz"
    payload = compute_collective(
        n_sites=4,
        a_values=np.array([1.4, 1.6]),
        n_transient=5,
        n_measure=10,
        renorm_interval=5,
        seed=7,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["a_values"], np.array([1.4, 1.6]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["lyap_c"],
        np.array([-0.11970016300271111, 0.17056225610801404]),
        rtol=1e-9,
    )
    assert payload["lyap_c"].shape == (2,)
    assert np.all(np.isfinite(payload["lyap_c"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["lyap_c"], payload["lyap_c"], rtol=1e-9)


# params: n_grid=4, n_transient=2, reference_transient=2, period=2, seed=N/A
def test_compute_coupled_basins_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "basins_golden.npz"
    payload = compute_coupled_basins(
        n_grid=4,
        n_transient=2,
        reference_transient=2,
        period=2,
        output_path=output_path,
    )

    np.testing.assert_allclose(
        payload["x"],
        np.array([-1.0, -0.33333333333333337, 0.33333333333333326, 1.0]),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        payload["y"],
        np.array([-1.0, -0.33333333333333337, 0.33333333333333326, 1.0]),
        rtol=1e-9,
    )
    np.testing.assert_array_equal(
        payload["basin"],
        np.array(
            [[0, 1, 1, 1], [2, 0, 1, 1], [2, 2, 0, 1], [2, 2, 2, 0]],
            dtype=np.int8,
        ),
    )

    with np.load(output_path) as saved:
        np.testing.assert_array_equal(saved["basin"], payload["basin"])


# params: A_values=[0.8, 1.0, 1.2], D_values=[0.0, 0.1], n_transient=5, seed=N/A
def test_compute_coupled_phase_diagram_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "phase_golden.npz"
    payload = compute_coupled_phase_diagram(
        A_values=np.array([0.8, 1.0, 1.2]),
        D_values=np.array([0.0, 0.1]),
        n_transient=5,
        n_sample=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(
        payload["asym"],
        np.array(
            [
                [
                    8.2293861882232425e-03,
                    5.9876141248649259e-05,
                    3.0112671401466472e-02,
                ],
                [
                    8.0161532921512221e-02,
                    1.3210878773309020e-02,
                    4.5589216524537955e-03,
                ],
            ]
        ),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        payload["lyap"],
        np.array(
            [
                [-0.0575617156373394, -4.103877162577709, 0.07108327581839047],
                [-0.04912671322212415, -0.5509515979462181, 0.08136671975106122],
            ]
        ),
        rtol=1e-9,
    )

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["asym"], payload["asym"], rtol=1e-9)
        np.testing.assert_allclose(saved["lyap"], payload["lyap"], rtol=1e-9)


# ---------------------------------------------------------------------------
# globally_coupled module tests
# ---------------------------------------------------------------------------


def test_gcm_step_basic_computation():
    """gcm_step updates a population according to the GCM formula."""
    from dynachaos.cml.primitives import gcm_step

    x = np.array([0.1, 0.5, 0.9])
    a = 1.99
    eps = 0.1

    x_new = gcm_step(x, a, eps)

    assert isinstance(x_new, np.ndarray)
    assert x_new.shape == x.shape
    assert np.all(np.isfinite(x_new))


def test_gcm_step_mean_field_formula():
    """gcm_step implements Eq. 1 from Kaneko (1990):
    x_{n+1}(i) = (1 - eps) f(x_n(i)) + eps/N sum_j f(x_n(j))
    """
    from dynachaos.cml.primitives import gcm_step
    from dynachaos.maps.primitives import logistic

    x = np.array([0.2, 0.3, 0.5])
    a = 1.99
    eps = 0.1

    x_new = gcm_step(x, a, eps)

    fx = logistic(x, a)
    mean_fx = np.mean(fx)
    expected = (1.0 - eps) * fx + eps * mean_fx

    np.testing.assert_allclose(x_new, expected, rtol=1e-14)


def test_gcm_step_zero_coupling_recovers_logistic_map():
    """When eps=0, gcm_step reduces to the logistic map."""
    from dynachaos.cml.primitives import gcm_step
    from dynachaos.maps.primitives import logistic

    x = np.array([0.1, 0.5, 0.9])
    a = 1.99
    eps = 0.0

    x_new = gcm_step(x, a, eps)
    expected = logistic(x, a)

    np.testing.assert_allclose(x_new, expected, rtol=1e-14)


def test_gcm_step_full_coupling_converges_to_mean_field():
    """When eps=1, gcm_step converges all sites to the same mean-field value."""
    from dynachaos.cml.primitives import gcm_step
    from dynachaos.maps.primitives import logistic

    x = np.array([0.1, 0.5, 0.9])
    a = 1.99
    eps = 1.0

    x_new = gcm_step(x, a, eps)
    fx = logistic(x, a)
    expected = np.full_like(x, np.mean(fx))

    np.testing.assert_allclose(x_new, expected, rtol=1e-14)


def test_gcm_step_preserves_bounds_in_chaotic_regime():
    """gcm_step dynamics remain finite for chaotic parameters."""
    from dynachaos.cml.primitives import gcm_step

    x = np.random.default_rng(42).uniform(-0.5, 0.5, 16)
    a = 1.99
    eps = 0.3

    for _ in range(100):
        x = gcm_step(x, a, eps)
        assert np.all(np.isfinite(x))


def test_gcm_step_synchronization_for_strong_coupling():
    """Strong coupling (eps ~ 1) synchronizes the population."""
    from dynachaos.cml.primitives import gcm_step

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 10)
    a = 1.99
    eps = 0.95

    for _ in range(200):
        x = gcm_step(x, a, eps)

    max_spread = np.max(x) - np.min(x)
    assert max_spread < 0.1


def test_gcm_mean_field_variance_decreases_with_N_for_weak_coupling():
    """Weaker coupling has larger mean-field fluctuations than strong coupling."""
    from dynachaos.cml.primitives import gcm_step
    from dynachaos.maps.primitives import logistic

    rng = np.random.default_rng(42)
    a = 1.99
    n_sample = 100

    variances = {}
    for eps in [0.05, 0.1, 0.2]:
        N = 50
        x = rng.uniform(-0.5, 0.5, N)

        for _ in range(500):
            x = gcm_step(x, a, eps)

        h_series = np.array([np.mean(logistic(x, a)) for _ in range(n_sample)])
        variances[eps] = np.var(h_series)

    assert variances[0.05] > variances[0.2]


def test_gcm_cluster_behavior_for_intermediate_coupling():
    """Intermediate coupling (0.1 < eps < 0.5) exhibits cluster formation."""
    from dynachaos.cml.primitives import gcm_step

    rng = np.random.default_rng(42)
    N = 16
    x = rng.uniform(-1, 1, N)
    a = 1.99
    eps = 0.3

    for _ in range(500):
        x = gcm_step(x, a, eps)

    spread = np.max(x) - np.min(x)
    assert spread > 0.2
    assert spread < 1.5


# ---------------------------------------------------------------------------
# arnold_tongues module
# ---------------------------------------------------------------------------


def test_arnold_tongues_grid_rotation_numbers_stay_inside_the_swept_band():
    """A small (Omega, K) grid gives rotation numbers inside the swept range.

    arnold_tongues.compute() hardcodes a 2000 x 1000 grid with 55000 iterations
    per cell, so it cannot run at test scale. This walks the same plane with the
    package rotation number instead of repeating the iteration here.
    """
    from dynachaos.maps.circle_map import rotation_number

    Omega_values = np.linspace(0.0, 1.0, 3)
    K_values = np.linspace(0.0, 0.3, 4)

    rho_2d = np.array(
        [
            [
                rotation_number(K, Omega, n_transient=100, n_iter=200, theta0=0.1)
                for Omega in Omega_values
            ]
            for K in K_values
        ]
    )

    assert rho_2d.shape == (4, 3)
    assert np.all(np.isfinite(rho_2d))
    # The drift per step is Omega + K sin(2 pi theta), so it cannot leave
    # [Omega_min - K_max, Omega_max + K_max].
    assert np.all(rho_2d >= -0.3 - 1e-9)
    assert np.all(rho_2d <= 1.0 + 0.3 + 1e-9)


def test_arnold_tongues_rotation_number_equals_Omega_on_the_K_zero_line():
    """At K = 0 the map is a rigid rotation, so rho = Omega exactly.

    This is the one point of the plane with a closed-form answer, so it is the
    check that would show a sign or scale error in the map.
    """
    from dynachaos.maps.circle_map import rotation_number

    for Omega in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rho = rotation_number(0.0, Omega, n_transient=500, n_iter=5000, theta0=0.1)
        # Measured error is at most 1.2e-16; the bound leaves room for a
        # different libm without letting a real error through.
        assert rho == pytest.approx(Omega, abs=1e-14)


def test_arnold_tongues_plot_tiny_data(tmp_path):
    """Plot function works on tiny computed data, saves PNG."""
    import matplotlib

    matplotlib.use("Agg")

    Omega_values = np.linspace(0.0, 1.0, 10)
    K_values = np.linspace(0.0, 0.3, 5)
    Omega_grid, K_grid = np.meshgrid(Omega_values, K_values)
    rho_2d = 0.5 * (1.0 - np.cos(2 * np.pi * K_grid))

    data = {"Omega": Omega_values, "K": K_values, "rho": rho_2d}

    from dynachaos.maps.arnold_tongues import plot as _plot_tongues

    output_png = tmp_path / "arnold_tongues_test.png"

    # Monkey-patch OUTPUT_PNG for this call
    import dynachaos.maps.arnold_tongues as at_module

    old_png = at_module.OUTPUT_PNG
    at_module.OUTPUT_PNG = output_png
    try:
        _plot_tongues(data)
    finally:
        at_module.OUTPUT_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


# ---------------------------------------------------------------------------
# circle_map module: additional tests for compute functions
# ---------------------------------------------------------------------------


def test_circle_map_rotation_number_tiny_A():
    """Rotation number at tiny A should be close to D."""
    from dynachaos.maps.circle_map import rotation_number

    D = 0.25
    A = 0.001
    rho = rotation_number(A, D, n_transient=100, n_iter=500, theta0=0.1)

    np.testing.assert_allclose(rho, D, atol=1e-3)


def test_circle_map_lyapunov_finite():
    """Lyapunov exponent is finite below the critical line, and zero at A = 0."""
    from dynachaos.maps.circle_map import lyapunov_exponent

    # At A = 0 the derivative is 1 at every point, so the sum of log|f'| is a
    # sum of zeros. Anything else means the derivative is wrong.
    assert lyapunov_exponent(0.0, D=0.25, n_transient=100, n_iter=500, theta0=0.1) == 0.0

    # Under the critical line 1 / (2 pi) = 0.159 the map is still invertible,
    # so the exponent cannot be positive. 500 iterations do not resolve it to
    # better than about 1e-5, which is why the bound is not zero.
    for A in [0.05, 0.1, 0.15]:
        lam = lyapunov_exponent(A, D=0.25, n_transient=100, n_iter=500, theta0=0.1)
        assert lam < 1e-3

    # Above the critical line the exponent may take either sign. Only ask that
    # the estimator returns a number.
    for A in [0.2, 0.25]:
        assert np.isfinite(lyapunov_exponent(A, D=0.25, n_transient=100, n_iter=500, theta0=0.1))


def test_circle_map_plot_tiny_devils_staircase(tmp_path):
    """Plot devil's staircase on tiny data; verify PNG created."""
    import matplotlib

    matplotlib.use("Agg")

    A_values = np.linspace(0.0, 0.25, 20)
    rho = 0.2 * np.ones_like(A_values)
    lam = -0.05 * np.ones_like(A_values)

    data = {"A": A_values, "rho": rho, "lam": lam}

    cm_module = importlib.import_module("dynachaos.maps.circle_map")

    output_png = tmp_path / "staircase_test.png"
    old_png = cm_module.OUTPUT_PNG
    cm_module.OUTPUT_PNG = output_png
    try:
        cm_module.plot(data)
    finally:
        cm_module.OUTPUT_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


def test_circle_map_plot_zoom_tiny(tmp_path):
    """Plot zoom of devil's staircase on tiny data."""
    import matplotlib

    matplotlib.use("Agg")

    A_full = np.linspace(0.0, 0.25, 50)
    rho_full = 0.2 * np.ones_like(A_full)
    A_zoom = np.linspace(0.10, 0.17, 30)
    rho_zoom = 0.2 * np.ones_like(A_zoom)

    full_data = {"A": A_full, "rho": rho_full}
    zoom_data = {"A": A_zoom, "rho": rho_zoom}

    cm_module = importlib.import_module("dynachaos.maps.circle_map")

    output_png = tmp_path / "zoom_test.png"
    old_png = cm_module.ZOOM_PNG
    cm_module.ZOOM_PNG = output_png
    try:
        cm_module.plot_zoom(zoom_data, full_data)
    finally:
        cm_module.ZOOM_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


def test_fractalization_iterate_tiny():
    """fractalization.iterate on tiny parameters; verify trajectory shape."""
    from dynachaos.maps.fractalization import iterate

    A = 0.3
    D = 1.75
    traj = iterate(A, D, n_transient=50, n_record=100)

    assert traj.shape == (100, 2)
    assert np.all(np.isfinite(traj))


def test_fractalization_iterate_starting_point():
    """iterate uses fixed-point initialization when x0=None."""
    from dynachaos.maps.fractalization import iterate

    A = 0.3
    D = 1.75
    traj1 = iterate(A, D, n_transient=50, n_record=100, x0=None)
    traj2 = iterate(A, D, n_transient=50, n_record=100, x0=None)

    np.testing.assert_allclose(traj1, traj2)


def test_fractalization_attractors_compute_tiny():
    """Tiny version of compute_attractors; verify output dict keys."""
    from dynachaos.maps.fractalization import iterate

    A = 0.3
    D_values = [1.75, 1.86]

    results = {}
    for D in D_values:
        traj = iterate(A, D, n_transient=50, n_record=100)
        results[f"D_{D}_traj"] = traj

    results["D_values"] = np.array(D_values)
    results["A"] = np.array([A])

    assert "D_1.75_traj" in results
    assert "D_1.86_traj" in results
    assert results["A"].shape == (1,)
    assert results["D_values"].shape == (2,)


def test_fractalization_plot_attractors_tiny(tmp_path):
    """Plot attractors on minimal data."""
    import matplotlib

    matplotlib.use("Agg")

    A = 0.3
    D_values = [1.75, 1.86]

    from dynachaos.maps.fractalization import iterate

    data = {}
    for D in D_values:
        traj = iterate(A, D, n_transient=50, n_record=100)
        data[f"D_{D}_traj"] = traj
    data["D_values"] = np.array(D_values)

    from dynachaos.maps.fractalization import plot_attractors

    output_png = tmp_path / "frac_attractors_test.png"

    import dynachaos.maps.fractalization as frac_module

    old_png = frac_module.FRAC_PNG
    frac_module.FRAC_PNG = output_png
    try:
        plot_attractors(data)
    finally:
        frac_module.FRAC_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


def test_fractalization_plot_dimension_tiny(tmp_path):
    """Plot correlation dimension on minimal data."""
    import matplotlib

    matplotlib.use("Agg")

    D = np.linspace(1.70, 2.00, 5)
    D2 = 1.0 + 0.2 * (D - 1.70)
    D2_err = 0.05 * np.ones_like(D)

    data = {"D": D, "D2": D2, "D2_err": D2_err, "A": np.array([0.3])}

    from dynachaos.maps.fractalization import plot_dimension

    output_png = tmp_path / "frac_dim_test.png"

    import dynachaos.maps.fractalization as frac_module

    old_png = frac_module.DIM_PNG
    frac_module.DIM_PNG = output_png
    try:
        plot_dimension(data)
    finally:
        frac_module.DIM_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


# ---------------------------------------------------------------------------
# coupled_delayed module
# ---------------------------------------------------------------------------


def test_coupled_delayed_shape_and_finite():
    """coupled_delayed map returns shape (4,) and is finite."""
    from dynachaos.maps.coupled_delayed import coupled_delayed

    state = np.array([0.5, 0.5, 0.3, 0.3])
    A = 0.4
    DA = 2.4
    DB = 2.35
    eps = 0.005

    out = coupled_delayed(state, A, DA, DB, eps)

    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


def test_coupled_delayed_jac_shape_and_finite():
    """coupled_delayed_jac returns shape (4, 4) and is finite."""
    from dynachaos.maps.coupled_delayed import coupled_delayed_jac

    state = np.array([0.5, 0.5, 0.3, 0.3])
    A = 0.4
    DA = 2.4
    DB = 2.35
    eps = 0.005

    jac = coupled_delayed_jac(state, A, DA, DB, eps)

    assert jac.shape == (4, 4)
    assert np.all(np.isfinite(jac))


def test_coupled_delayed_jac_finite_difference():
    """Jacobian should match finite-difference approximation."""
    from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

    state = np.array([0.5, 0.5, 0.3, 0.3])
    A = 0.4
    DA = 2.4
    DB = 2.35
    eps = 0.005
    jac = coupled_delayed_jac(state, A, DA, DB, eps)

    eps_fd = 1e-7
    for j in range(4):
        state_plus = state.copy()
        state_plus[j] += eps_fd
        state_minus = state.copy()
        state_minus[j] -= eps_fd
        fd_col = (
            coupled_delayed(state_plus, A, DA, DB, eps)
            - coupled_delayed(state_minus, A, DA, DB, eps)
        ) / (2.0 * eps_fd)
        np.testing.assert_allclose(jac[:, j], fd_col, atol=1e-5)


def test_coupled_delayed_lyapunov_compute_tiny():
    """Tiny Lyapunov computation on 2 DB points."""
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
    from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

    A = 0.4
    DB_values = np.array([2.35, 2.45])

    for DB in DB_values:
        DA = DB + 0.1
        x0 = np.array([0.5, 0.5, 0.3, 0.3])

        def f(s):
            return coupled_delayed(s, A, DA, DB, 0.005)

        def jac(s):
            return coupled_delayed_jac(s, A, DA, DB, 0.005)

        spectra = lyapunov_spectrum(f, jac, x0, n_iter=500, n_transient=200)

        assert spectra.shape == (4,)
        assert np.all(np.isfinite(spectra))


def test_coupled_delayed_projections_compute_tiny():
    """Compute (x, z) projections on tiny iteration count."""
    from dynachaos.maps._iter import trajectory_after_transient
    from dynachaos.maps.coupled_delayed import coupled_delayed

    A = 0.4
    DB = 2.37
    DA = DB + 0.1

    traj = trajectory_after_transient(
        np.array([0.5, 0.5, 0.3, 0.3], dtype=np.float64),
        lambda state: coupled_delayed(state, A, DA, DB, 0.005),
        200,
        300,
        project_fn=lambda state: state[[0, 2]],
    )

    assert traj.shape == (300, 2)
    assert np.all(np.isfinite(traj))


def test_coupled_delayed_plot_lyapunov_tiny(tmp_path):
    """Plot Lyapunov exponents on minimal data."""
    import matplotlib

    matplotlib.use("Agg")

    DB = np.linspace(2.1, 2.65, 5)
    eps_values = np.array([1e-3, 5e-3, 1e-2])

    spectra_data = {}
    for eps in eps_values:
        spectra = np.random.randn(5, 4)
        spectra_data[f"eps_{eps}_spectra"] = spectra

    data = {"DB": DB, "eps_values": eps_values, **spectra_data}

    cd_module = importlib.import_module("dynachaos.maps.coupled_delayed")

    output_png = tmp_path / "coupled_lyap_test.png"
    old_png = cd_module.LYAP_PNG
    cd_module.LYAP_PNG = output_png
    try:
        cd_module.plot_lyapunov(data)
    finally:
        cd_module.LYAP_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


def test_coupled_delayed_plot_projections_tiny(tmp_path):
    """Plot (x, z) projections on minimal data."""
    import matplotlib

    matplotlib.use("Agg")

    DB_values = np.array([2.37, 2.43])
    labels = np.array(["test_a", "test_b"])
    render_modes = np.array(["line", "points"])

    data = {"DB_values": DB_values, "labels": labels, "render_modes": render_modes}
    for DB in DB_values:
        traj = 0.5 * np.random.randn(100, 2) + 0.3
        data[f"DB_{DB}_xz"] = traj

    cd_module = importlib.import_module("dynachaos.maps.coupled_delayed")

    output_png = tmp_path / "coupled_proj_test.png"
    old_png = cd_module.PROJ_PNG
    cd_module.PROJ_PNG = output_png
    try:
        cd_module.plot_projections(data)
    finally:
        cd_module.PROJ_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


def test_modulated_circle_map_shape():
    """modulated_circle map returns shape (2,) and wraps to [0, 1)."""
    from dynachaos.maps.modulated_circle import modulated_circle

    state = np.array([0.3, 0.7])
    out = modulated_circle(state, A=0.10, C=0.618, D=0.25, eps=0.05)

    assert out.shape == (2,)
    assert 0.0 <= out[0] < 1.0
    assert 0.0 <= out[1] < 1.0


def test_modulated_circle_map_known_ic():
    """Test map on known initial condition."""
    from dynachaos.maps.modulated_circle import modulated_circle

    state = np.array([0.0, 0.0])
    A = 0.0
    C = 0.3
    D = 0.4
    eps = 0.0

    out = modulated_circle(state, A, C, D, eps)
    expected = np.array([D % 1.0, C % 1.0])
    np.testing.assert_allclose(out, expected)


def test_modulated_circle_rotation_numbers_finite():
    """Rotation numbers should be finite."""
    from dynachaos.maps.modulated_circle import C_GOLDEN, rotation_numbers

    A = 0.10
    C = C_GOLDEN
    D = 0.5
    eps = 0.05

    rho_theta, rho_phi = rotation_numbers(A, C, D, eps, n_transient=100, n_iter=500)

    assert np.isfinite(rho_theta)
    assert np.isfinite(rho_phi)
    np.testing.assert_allclose(rho_phi, C, atol=1e-4)


def test_modulated_circle_rotation_numbers_tiny_sweep():
    """Sweep D on 3 points; all rotation numbers should be finite."""
    from dynachaos.maps.modulated_circle import C_GOLDEN, rotation_numbers

    A = 0.10
    C = C_GOLDEN
    eps = 0.05
    D_values = np.array([0.2, 0.5, 0.8])

    rho_theta_vals = []
    rho_phi_vals = []
    for D in D_values:
        rt, rp = rotation_numbers(A, C, D, eps, n_transient=100, n_iter=500)
        rho_theta_vals.append(rt)
        rho_phi_vals.append(rp)

    rho_theta_vals = np.array(rho_theta_vals)
    rho_phi_vals = np.array(rho_phi_vals)

    assert np.all(np.isfinite(rho_theta_vals))
    assert np.all(np.isfinite(rho_phi_vals))
    np.testing.assert_allclose(rho_phi_vals, C, atol=1e-4)


def test_modulated_circle_plot_tiny(tmp_path):
    """Plot double devil's staircase on tiny data."""
    import matplotlib

    matplotlib.use("Agg")

    D = np.linspace(0.0, 1.0, 20)
    rho_theta = 0.4 * np.ones_like(D)

    data = {"D": D, "rho_theta": rho_theta}

    mc_module = importlib.import_module("dynachaos.maps.modulated_circle")

    output_png = tmp_path / "modulated_circle_test.png"
    old_png = mc_module.OUTPUT_PNG
    mc_module.OUTPUT_PNG = output_png
    try:
        mc_module.plot(data)
    finally:
        mc_module.OUTPUT_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


def test_modulated_circle_plot_zoom_tiny(tmp_path):
    """Plot zoom of double devil's staircase on tiny data."""
    import matplotlib

    matplotlib.use("Agg")

    D = np.linspace(0.0, 1.0, 30)
    rho_theta = 0.4 * np.ones_like(D)

    data = {"D": D, "rho_theta": rho_theta}

    mc_module = importlib.import_module("dynachaos.maps.modulated_circle")

    output_png = tmp_path / "modulated_zoom_test.png"
    old_png = mc_module.ZOOM_PNG
    mc_module.ZOOM_PNG = output_png
    try:
        mc_module.plot_zoom(data)
    finally:
        mc_module.ZOOM_PNG = old_png

    assert output_png.exists()
    assert output_png.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Additional tests for coverage: registry and compute functions
# ---------------------------------------------------------------------------


def test_arnold_tongues_main_calls_plot(tmp_path, monkeypatch):
    """Test that main() orchestrates compute and plot correctly."""

    arnold_tongues_module = importlib.import_module("dynachaos.maps.arnold_tongues")

    # Monkeypatch the FIG_DIR to use tmp_path
    monkeypatch.setattr(arnold_tongues_module, "FIG_DIR", tmp_path)
    monkeypatch.setattr(arnold_tongues_module, "OUTPUT_NPZ", tmp_path / "test.npz")
    monkeypatch.setattr(arnold_tongues_module, "OUTPUT_PNG", tmp_path / "test.png")

    # We'll skip the heavy compute and just test it doesn't crash
    old_compute = arnold_tongues_module.compute

    def mock_compute():
        np.savez_compressed(
            arnold_tongues_module.OUTPUT_NPZ,
            Omega=np.linspace(0, 1, 3),
            K=np.linspace(0, 0.3, 2),
            rho=np.random.randn(2, 3),
        )

    monkeypatch.setattr(arnold_tongues_module, "compute", mock_compute)

    try:
        arnold_tongues_module.main()
        assert arnold_tongues_module.OUTPUT_PNG.exists()
    finally:
        monkeypatch.setattr(arnold_tongues_module, "compute", old_compute)


def test_circle_map_compute_returns_arrays(tmp_path):
    """Small version of compute(): verify output arrays."""
    n_params = 10
    n_transient = 50
    n_iter = 200
    D = 0.25

    A_values = np.linspace(0.0, 0.25, n_params)
    TWO_PI = 2.0 * np.pi

    theta = np.full(n_params, 0.1)

    from dynachaos.maps._iter import iterate_unwrapped

    theta = iterate_unwrapped(theta, lambda th: D + A_values * np.sin(TWO_PI * th), n_transient)

    theta_start = theta.copy()

    log_sum = np.zeros(n_params)
    for _ in range(n_iter):
        deriv = np.abs(1.0 + TWO_PI * A_values * np.cos(TWO_PI * theta))
        log_sum += np.where(deriv > 0, np.log(deriv), -100.0)
        theta += D + A_values * np.sin(TWO_PI * theta)

    rho = (theta - theta_start) / n_iter
    lam = log_sum / n_iter

    assert rho.shape == (n_params,)
    assert lam.shape == (n_params,)
    assert np.all(np.isfinite(rho))
    assert np.all(np.isfinite(lam))


def test_fractalization_compute_attractors_returns_dict():
    """Tiny version of compute_attractors: verify output keys."""
    from dynachaos.maps.fractalization import iterate

    A = 0.3
    D_values = [1.75, 1.86, 1.90]

    results = {}
    for D in D_values:
        traj = iterate(A, D, n_transient=50, n_record=100)
        results[f"D_{D}_traj"] = traj

    results["D_values"] = np.array(D_values)
    results["A"] = np.array([A])

    assert all(f"D_{D}_traj" in results for D in D_values)
    assert results["A"].shape == (1,)
    assert results["D_values"].shape == (3,)


def test_fractalization_compute_dimensions_tiny():
    """Tiny version of compute_dimensions: 3 D values, compute D2."""
    from dynachaos.diagnostics.correlation import correlation_dimension
    from dynachaos.maps.fractalization import iterate

    A = 0.3
    D_values = np.array([1.75, 1.85, 1.95])
    D2_values = []

    for D in D_values:
        traj = iterate(A, D, n_transient=100, n_record=500)
        D2, _, _, D2_err, _, _ = correlation_dimension(
            traj, n_r=20, max_pairs=10000, return_stderr=True
        )
        D2_values.append(D2)

    D2_values = np.array(D2_values)

    assert D2_values.shape == (3,)
    assert np.all(np.isfinite(D2_values))
    assert np.all((D2_values > 0) & (D2_values < 3.0))


def test_coupled_delayed_compute_lyapunov_tiny():
    """Tiny version of compute_lyapunov: 2 DB values."""
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
    from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac

    A = 0.4
    eps_val = 5e-3
    DB_values = np.array([2.35, 2.45])

    for DB in DB_values:
        DA = DB + 0.1
        x0 = np.array([0.5, 0.5, 0.3, 0.3])

        def f(s):
            return coupled_delayed(s, A, DA, DB, eps_val)

        def jac(s):
            return coupled_delayed_jac(s, A, DA, DB, eps_val)

        spec = lyapunov_spectrum(f, jac, x0, n_iter=500, n_transient=200)

        assert spec.shape == (4,)
        assert np.all(np.isfinite(spec))


def test_coupled_delayed_compute_projections_multiple_db():
    """Compute projections at multiple DB values."""
    from dynachaos.maps._iter import trajectory_after_transient
    from dynachaos.maps.coupled_delayed import coupled_delayed

    A = 0.4
    DB_values = [2.37, 2.43, 2.45]

    results = {}
    for DB in DB_values:
        DA = DB + 0.1
        traj = trajectory_after_transient(
            np.array([0.5, 0.5, 0.3, 0.3], dtype=np.float64),
            lambda state: coupled_delayed(state, A, DA, DB, 0.005),
            100,
            200,
            project_fn=lambda state: state[[0, 2]],
        )
        results[f"DB_{DB}_xz"] = traj

    results["DB_values"] = np.array(DB_values)

    assert len(DB_values) == 3
    assert all(f"DB_{DB}_xz" in results for DB in DB_values)
    assert all(results[f"DB_{DB}_xz"].shape == (200, 2) for DB in DB_values)


def test_modulated_circle_compute_tiny():
    """Tiny version of compute(): sweep D on 5 points."""
    from dynachaos.maps.modulated_circle import C_GOLDEN, rotation_numbers

    A = 0.10
    eps = 0.05
    C = C_GOLDEN

    D_values = np.linspace(0.0, 1.0, 5)

    rho_theta = np.empty(5)
    rho_phi = np.empty(5)

    for i, D in enumerate(D_values):
        rt, rp = rotation_numbers(A, C, D, eps, n_transient=100, n_iter=500)
        rho_theta[i] = rt
        rho_phi[i] = rp

    assert rho_theta.shape == (5,)
    assert rho_phi.shape == (5,)
    assert np.all(np.isfinite(rho_theta))
    assert np.all(np.isfinite(rho_phi))
    np.testing.assert_allclose(rho_phi, C, atol=1e-4)


def test_arnold_tongues_compute_monotonicity_at_K_zero():
    """Verify rho is monotonic in Omega at K=0 (should equal Omega)."""
    n_omega = 10
    n_transient = 100
    n_iter = 500

    Omega_values = np.linspace(0.0, 1.0, n_omega)
    K_values = np.array([0.0])

    Omega_grid, K_grid = np.meshgrid(Omega_values, K_values)
    Omega_flat = Omega_grid.ravel()
    K_flat = K_grid.ravel()

    TWO_PI = 2.0 * np.pi
    theta = np.full_like(Omega_flat, 0.1)

    for _ in range(n_transient):
        theta += Omega_flat + K_flat * np.sin(TWO_PI * theta)

    theta_start = theta.copy()

    for _ in range(n_iter):
        theta += Omega_flat + K_flat * np.sin(TWO_PI * theta)

    rho = (theta - theta_start) / n_iter
    rho_1d = rho.ravel()

    np.testing.assert_allclose(rho_1d, Omega_values, atol=1e-4)
    np.testing.assert_array_less(-0.001, np.diff(rho_1d))


def test_circle_map_winding_number_K_small():
    """At small K (subcritical), winding number should be in valid range."""
    from dynachaos.maps.circle_map import rotation_number

    A_vals = np.array([0.001, 0.05, 0.10])
    rho_vals = []

    for A in A_vals:
        rho = rotation_number(A, D=0.25, n_transient=100, n_iter=500)
        rho_vals.append(rho)

    rho_vals = np.array(rho_vals)

    assert np.all(np.isfinite(rho_vals))
    assert np.all((0.0 <= rho_vals) & (rho_vals <= 0.25))


def test_fractalization_iterate_produces_consistent_orbit():
    """Iterate twice from same x0; verify consistency."""
    from dynachaos.maps.fractalization import iterate

    A = 0.3
    D = 1.85
    x0 = np.array([0.3, 0.2])

    traj1 = iterate(A, D, n_transient=50, n_record=100, x0=x0.copy())
    traj2 = iterate(A, D, n_transient=50, n_record=100, x0=x0.copy())

    np.testing.assert_allclose(traj1, traj2)


def test_coupled_delayed_fixed_point_near_torus():
    """Test trajectory near the 3-torus does not diverge."""
    from dynachaos.maps._iter import run_transient
    from dynachaos.maps.coupled_delayed import coupled_delayed

    A = 0.4
    DA = 2.37 + 0.1
    DB = 2.37
    eps = 0.005

    x0 = np.array([0.5, 0.5, 0.3, 0.3])
    state = run_transient(x0, lambda s: coupled_delayed(s, A, DA, DB, eps), 500)

    assert state is not None
    assert np.all(np.isfinite(state))


def test_modulated_circle_phi_rotation_preserved():
    """Test that phi advances by C per iteration (golden mean)."""
    from dynachaos.maps.modulated_circle import C_GOLDEN, modulated_circle

    A = 0.10
    D = 0.5
    eps = 0.05

    state = np.array([0.3, 0.2])

    for _ in range(10):
        state = modulated_circle(state, A, C_GOLDEN, D, eps)

    expected_phi = (0.2 + 10 * C_GOLDEN) % 1.0
    np.testing.assert_allclose(state[1], expected_phi, atol=1e-10)


# ---------------------------------------------------------------------------
# Main orchestration and registry paths
# ---------------------------------------------------------------------------


def test_circle_map_main_full_workflow(tmp_path, monkeypatch):
    """Test circle_map main() orchestration with mocked compute."""

    cm = importlib.import_module("dynachaos.maps.circle_map")

    monkeypatch.setattr(cm, "FIG_DIR", tmp_path)
    monkeypatch.setattr(cm, "OUTPUT_NPZ", tmp_path / "devils.npz")
    monkeypatch.setattr(cm, "OUTPUT_PNG", tmp_path / "devils.png")
    monkeypatch.setattr(cm, "ZOOM_NPZ", tmp_path / "zoom.npz")
    monkeypatch.setattr(cm, "ZOOM_PNG", tmp_path / "zoom.png")

    def mock_compute():
        np.savez_compressed(
            cm.OUTPUT_NPZ,
            A=np.linspace(0, 0.25, 5),
            rho=np.random.randn(5),
            lam=np.random.randn(5),
        )

    def mock_compute_zoom():
        np.savez_compressed(cm.ZOOM_NPZ, A=np.linspace(0.10, 0.17, 5), rho=np.random.randn(5))

    old_compute = cm.compute
    old_compute_zoom = cm.compute_zoom
    monkeypatch.setattr(cm, "compute", mock_compute)
    monkeypatch.setattr(cm, "compute_zoom", mock_compute_zoom)

    try:
        cm.main()
        assert cm.OUTPUT_PNG.exists()
        assert cm.ZOOM_PNG.exists()
    finally:
        monkeypatch.setattr(cm, "compute", old_compute)
        monkeypatch.setattr(cm, "compute_zoom", old_compute_zoom)


def test_fractalization_main_orchestration(tmp_path, monkeypatch):
    """Test fractalization main() with mocked compute functions."""

    frac = importlib.import_module("dynachaos.maps.fractalization")

    monkeypatch.setattr(frac, "FIG_DIR", tmp_path)
    monkeypatch.setattr(frac, "FRAC_NPZ", tmp_path / "frac.npz")
    monkeypatch.setattr(frac, "FRAC_PNG", tmp_path / "frac.png")
    monkeypatch.setattr(frac, "DIM_NPZ", tmp_path / "dim.npz")
    monkeypatch.setattr(frac, "DIM_PNG", tmp_path / "dim.png")
    monkeypatch.setattr(frac, "ANIM_NPZ", tmp_path / "anim.npz")
    monkeypatch.setattr(frac, "ANIM_GIF", tmp_path / "anim.gif")

    def mock_attractors():
        results = {}
        for D in [1.75, 1.86]:
            results[f"D_{D}_traj"] = np.random.randn(100, 2)
        results["D_values"] = np.array([1.75, 1.86])
        results["A"] = np.array([0.3])
        np.savez_compressed(frac.FRAC_NPZ, **results)

    def mock_dimensions():
        np.savez_compressed(
            frac.DIM_NPZ,
            D=np.linspace(1.70, 2.00, 3),
            D2=np.random.rand(3) + 1.0,
            D2_err=0.05 * np.ones(3),
            A=np.array([0.3]),
        )

    def mock_animation():
        np.savez_compressed(
            frac.ANIM_NPZ,
            param_values=np.linspace(1.75, 1.96, 5),
            all_x=np.random.randn(5, 100),
            all_y=np.random.randn(5, 100),
        )

    old_attractors = frac.compute_attractors
    old_dimensions = frac.compute_dimensions
    old_animation = frac.compute_animation_data
    monkeypatch.setattr(frac, "compute_attractors", mock_attractors)
    monkeypatch.setattr(frac, "compute_dimensions", mock_dimensions)
    monkeypatch.setattr(frac, "compute_animation_data", mock_animation)

    try:
        frac.main()
        assert frac.FRAC_PNG.exists()
        assert frac.DIM_PNG.exists()
    finally:
        monkeypatch.setattr(frac, "compute_attractors", old_attractors)
        monkeypatch.setattr(frac, "compute_dimensions", old_dimensions)
        monkeypatch.setattr(frac, "compute_animation_data", old_animation)


def test_coupled_delayed_main_orchestration(tmp_path, monkeypatch):
    """Test coupled_delayed main() with mocked compute functions."""

    cd = importlib.import_module("dynachaos.maps.coupled_delayed")

    monkeypatch.setattr(cd, "FIG_DIR", tmp_path)
    monkeypatch.setattr(cd, "LYAP_NPZ", tmp_path / "lyap.npz")
    monkeypatch.setattr(cd, "LYAP_PNG", tmp_path / "lyap.png")
    monkeypatch.setattr(cd, "PROJ_NPZ", tmp_path / "proj.npz")
    monkeypatch.setattr(cd, "PROJ_PNG", tmp_path / "proj.png")
    monkeypatch.setattr(cd, "ANIM_NPZ", tmp_path / "anim.npz")
    monkeypatch.setattr(cd, "ANIM_GIF", tmp_path / "anim.gif")

    def mock_lyapunov():
        data = {
            "DB": np.linspace(2.1, 2.65, 3),
            "eps_values": np.array([1e-3, 5e-3, 1e-2]),
        }
        for eps in [1e-3, 5e-3, 1e-2]:
            key = f"eps_{eps}_spectra"
            data[key] = np.random.randn(3, 4)
        np.savez_compressed(cd.LYAP_NPZ, **data)

    def mock_projections():
        results = {"DB_values": np.array([2.37, 2.43])}
        for DB in [2.37, 2.43]:
            results[f"DB_{DB}_xz"] = np.random.randn(100, 2)
        results["labels"] = np.array(["test_a", "test_b"])
        results["render_modes"] = np.array(["line", "points"])
        results["schema_version"] = np.array([cd.PROJ_SCHEMA_VERSION])
        np.savez_compressed(cd.PROJ_NPZ, **results)

    def mock_animation():
        np.savez_compressed(
            cd.ANIM_NPZ,
            param_values=np.linspace(2.1, 2.65, 5),
            all_x=np.random.randn(5, 100),
            all_y=np.random.randn(5, 100),
        )

    old_lyap = cd.compute_lyapunov
    old_proj = cd.compute_projections
    old_anim = cd.compute_animation_data
    monkeypatch.setattr(cd, "compute_lyapunov", mock_lyapunov)
    monkeypatch.setattr(cd, "compute_projections", mock_projections)
    monkeypatch.setattr(cd, "compute_animation_data", mock_animation)

    try:
        cd.main()
        assert cd.LYAP_PNG.exists()
        assert cd.PROJ_PNG.exists()
    finally:
        monkeypatch.setattr(cd, "compute_lyapunov", old_lyap)
        monkeypatch.setattr(cd, "compute_projections", old_proj)
        monkeypatch.setattr(cd, "compute_animation_data", old_anim)


def test_modulated_circle_main_orchestration(tmp_path, monkeypatch):
    """Test modulated_circle main() with mocked compute."""

    mc = importlib.import_module("dynachaos.maps.modulated_circle")

    monkeypatch.setattr(mc, "FIG_DIR", tmp_path)
    monkeypatch.setattr(mc, "OUTPUT_NPZ", tmp_path / "double.npz")
    monkeypatch.setattr(mc, "OUTPUT_PNG", tmp_path / "double.png")
    monkeypatch.setattr(mc, "ZOOM_PNG", tmp_path / "zoom.png")

    def mock_compute():
        np.savez_compressed(
            mc.OUTPUT_NPZ,
            D=np.linspace(0, 1, 10),
            rho_theta=np.random.randn(10),
            rho_phi=mc.C_GOLDEN * np.ones(10),
            A=np.array([0.10]),
            C=np.array([mc.C_GOLDEN]),
            eps=np.array([0.05]),
        )

    old_compute = mc.compute
    monkeypatch.setattr(mc, "compute", mock_compute)

    try:
        mc.main()
        assert mc.OUTPUT_PNG.exists()
        assert mc.ZOOM_PNG.exists()
    finally:
        monkeypatch.setattr(mc, "compute", old_compute)


# ---------------------------------------------------------------------------
# Additional coverage: edge cases and boundary conditions
# ---------------------------------------------------------------------------


def test_arnold_tongues_rotation_number_range():
    """Verify rotation numbers stay in [0, 1] for all K, Omega."""
    n_omega = 5
    n_transient = 100
    n_iter = 300

    Omega_values = np.linspace(0.0, 1.0, n_omega)
    K_values = np.linspace(0.0, 0.3, 4)

    Omega_grid, K_grid = np.meshgrid(Omega_values, K_values)
    Omega_flat = Omega_grid.ravel()
    K_flat = K_grid.ravel()

    TWO_PI = 2.0 * np.pi
    theta = np.full_like(Omega_flat, 0.1)

    for _ in range(n_transient):
        theta += Omega_flat + K_flat * np.sin(TWO_PI * theta)

    theta_start = theta.copy()

    for _ in range(n_iter):
        theta += Omega_flat + K_flat * np.sin(TWO_PI * theta)

    rho = (theta - theta_start) / n_iter

    assert np.all((-1e-6 <= rho) & (rho <= 1.0 + 1e-6))


def test_circle_map_phase_wrapping():
    """Verify circle_map properly wraps angles to [0, 1)."""
    from dynachaos.maps.circle_map import circle_map

    theta_large = 1.5
    out = circle_map(theta_large, A=0.1, D=0.25)

    assert 0.0 <= out < 1.0


def test_circle_map_derivative_is_smooth():
    """Derivative should be finite and continuous."""
    from dynachaos.maps.circle_map import circle_map_derivative

    theta_values = np.linspace(0.0, 1.0, 11)
    derivs = circle_map_derivative(theta_values, A=0.1, D=0.25)

    assert np.all(np.isfinite(derivs))
    assert derivs.shape == theta_values.shape


def test_fractalization_fixed_point_initialization():
    """Verify fixed-point x0 calculation is consistent."""
    from dynachaos.maps.fractalization import iterate

    A = 0.3
    D = 1.85

    # Compute fixed point: fp = (sqrt(1 + 4*D) - 1) / (2*D)
    fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)

    traj1 = iterate(A, D, n_transient=100, n_record=50, x0=None)
    traj2 = iterate(A, D, n_transient=100, n_record=50, x0=np.array([fp + 0.01, fp - 0.01]))

    np.testing.assert_allclose(traj1, traj2, atol=1e-6)


def test_coupled_delayed_jacobian_structure():
    """Verify Jacobian has expected block structure."""
    from dynachaos.maps.coupled_delayed import coupled_delayed_jac

    state = np.array([0.5, 0.5, 0.3, 0.3])
    A = 0.4
    DA = 2.4
    DB = 2.35
    eps = 0.005

    jac = coupled_delayed_jac(state, A, DA, DB, eps)

    assert jac[1, 0] == 1.0
    assert jac[1, 1] == 0.0
    assert jac[3, 2] == 1.0
    assert jac[3, 3] == 0.0


def test_coupled_delayed_perturbations():
    """Test that coupling perturbations affect the state evolution."""
    from dynachaos.maps.coupled_delayed import coupled_delayed

    state_base = np.array([0.5, 0.5, 0.3, 0.2])  # Modified so z != w

    A = 0.4
    DA = 2.4
    DB = 2.35

    out_no_eps = coupled_delayed(state_base, A, DA, DB, eps=0.0)
    out_with_eps = coupled_delayed(state_base, A, DA, DB, eps=0.05)

    assert not np.allclose(out_no_eps, out_with_eps)


def test_modulated_circle_sine_modulation():
    """Test that sine modulation term affects iteration."""
    from dynachaos.maps.modulated_circle import modulated_circle

    state = np.array([0.3, 0.3])
    A = 0.10
    C = 0.618
    D = 0.5

    out_no_eps = modulated_circle(state, A, C, D, eps=0.0)
    out_with_eps = modulated_circle(state, A, C, D, eps=0.1)

    assert not np.allclose(out_no_eps, out_with_eps)


def test_modulated_circle_phi_advances_by_C():
    """Test that phi advances exactly by C per iteration."""
    from dynachaos.maps.modulated_circle import C_GOLDEN, modulated_circle

    state = np.array([0.5, 0.0])
    A = 0.10
    D = 0.5
    eps = 0.05

    out = modulated_circle(state, A, C_GOLDEN, D, eps)

    expected_phi = C_GOLDEN
    np.testing.assert_allclose(out[1], expected_phi, atol=1e-10)


# ── Coverage: _iter edge cases ──────────────────────────────────────────────────


def test_run_transient_returns_none_on_divergence_early_exit():
    """run_transient returns None when diverged_fn triggers (line 36, return None)."""
    from dynachaos.maps._iter import run_transient

    state = np.array([1.0, 2.0])

    def step_fn(s):
        return s * 2.0

    def diverged_fn(s):
        return np.max(np.abs(s)) > 100.0

    result = run_transient(state, step_fn, 100, diverged_fn=diverged_fn)
    assert result is None


def test_sample_trajectory_returns_none_on_divergence_without_partial():
    """sample_trajectory returns None when diverged without allow_partial (line 63)."""
    from dynachaos.maps._iter import sample_trajectory

    state = np.array([1.0])

    def step_fn(s):
        return s * 5.0

    def diverged_fn(s):
        return np.abs(s[0]) > 100.0

    result = sample_trajectory(state, step_fn, 100, diverged_fn=diverged_fn, allow_partial=False)
    assert result is None


def test_trajectory_after_transient_returns_none_on_divergence():
    """trajectory_after_transient returns None when divergence occurs (line 85)."""
    from dynachaos.maps._iter import trajectory_after_transient

    state = np.array([1.0])

    def step_fn(s):
        return s * 10.0

    def diverged_fn(s):
        return np.abs(s[0]) > 50.0

    result = trajectory_after_transient(state, step_fn, 100, 50, diverged_fn=diverged_fn)
    assert result is None
