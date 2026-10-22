"""Coverage tests for the cml figure modules: pattern_dynamics, spatiotemporal,
correlation_figure, comoving_figure, gcm_clusters, globally_coupled.

The compute() entry points in several of these modules are hardcoded to
paper-scale iteration counts (tens of thousands of CML steps per parameter
combination) and are not reachable within the test time budget -- see the
per-module notes below. Where a module exposes a computation that runs in a
fraction of a second at its own hardcoded scale (spatiotemporal.compute,
pattern_dynamics.compute_space_amplitude, correlation_figure.compute_correlations,
gcm_clusters.compute_clusters/compute_collective), it is exercised directly.
Plot functions are exercised with tiny synthetic payloads shaped like the
real .npz schema, matching the monkeypatch-the-module-constant pattern used
throughout test_maps.py (see test_arnold_tongues_main_calls_plot).
"""

import numpy as np
import pytest
from matplotlib.axes import Axes

from dynachaos.cml import (
    comoving_figure,
    correlation_figure,
    gcm_clusters,
    globally_coupled,
    pattern_dynamics,
    spatiotemporal,
)

# ---------------------------------------------------------------------------
# spatiotemporal
# ---------------------------------------------------------------------------


def test_model_functions_match_hand_computed_values():
    c = (np.sqrt(5) - 1.0) / 2.0
    # Below the kink c, model A is the quadratic branch x + x^2 + a.
    np.testing.assert_allclose(spatiotemporal.model_A_f(np.array([0.0]), a=-0.01), [-0.01])
    # Above the kink, model A is the linear branch -3(x-c)+1+a.
    above = c + 0.1
    np.testing.assert_allclose(spatiotemporal.model_A_f(np.array([above]), a=0.0), [1.0 - 0.3])

    np.testing.assert_allclose(spatiotemporal.model_B_f(np.array([0.0]), A=0.2, C=0.55), [0.55])
    np.testing.assert_allclose(spatiotemporal.model_B_g(np.array([0.25])), [1.0])

    np.testing.assert_allclose(spatiotemporal.model_C_f(np.array([0.0]), A=1.752), [1.0])
    np.testing.assert_allclose(spatiotemporal.model_C_f(np.array([1.0]), A=1.752), [1.0 - 1.752])


def test_simulate_cml_returns_requested_spacetime_shape():
    spacetime = spatiotemporal.simulate_cml(
        spatiotemporal.model_C_f,
        spatiotemporal.model_C_f,
        eps=0.2,
        N=16,
        n_transient=5,
        n_record=7,
    )

    assert spacetime.shape == (7, 16)
    assert np.all(np.isfinite(spacetime))


def test_spatiotemporal_compute_writes_all_nine_series(monkeypatch, tmp_path):
    npz_path = tmp_path / "spacetime_diagrams.npz"
    monkeypatch.setattr(spatiotemporal, "FIG_DIR", tmp_path)
    monkeypatch.setattr(spatiotemporal, "STI_NPZ", npz_path)

    spatiotemporal.compute()

    with np.load(npz_path) as data:
        keys = set(data.files)
    expected = {f"{m}_eps_{e}" for m, e in [("A", 0.06), ("A", 0.07), ("A", 0.08)]}
    expected |= {f"{m}_eps_{e}" for m, e in [("B", 0.02), ("B", 0.024), ("B", 0.03)]}
    expected |= {f"{m}_eps_{e}" for m, e in [("C", 0.16), ("C", 0.20), ("C", 0.30)]}
    assert keys == expected


def test_spatiotemporal_plot_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "spacetime_diagrams.png"
    monkeypatch.setattr(spatiotemporal, "STI_PNG", png_path)

    rng = np.random.default_rng(0)
    data = {
        f"{m}_eps_{e}": rng.uniform(-1, 1, (5, 8))
        for m, evals in [
            ("A", [0.06, 0.07, 0.08]),
            ("B", [0.02, 0.024, 0.03]),
            ("C", [0.16, 0.20, 0.30]),
        ]
        for e in evals
    }

    spatiotemporal.plot(data)

    assert png_path.stat().st_size > 0


def test_spatiotemporal_main_computes_then_plots(monkeypatch, tmp_path):
    npz_path = tmp_path / "spacetime_diagrams.npz"
    png_path = tmp_path / "spacetime_diagrams.png"
    monkeypatch.setattr(spatiotemporal, "FIG_DIR", tmp_path)
    monkeypatch.setattr(spatiotemporal, "STI_NPZ", npz_path)
    monkeypatch.setattr(spatiotemporal, "STI_PNG", png_path)

    spatiotemporal.main()

    assert npz_path.exists()
    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# pattern_dynamics
#
# compute_phase_diagram() sweeps 200 x 160 (a, eps) points, each with a 5000
# step transient plus a 2000 step sampling window run on a vectorized (200, N)
# array -- src/dynachaos/cml/pattern_dynamics.py:48-112. That is not reachable
# in a test-scale budget; the plotting path is exercised below with a tiny
# synthetic grid shaped like its output instead.
# ---------------------------------------------------------------------------


def test_pattern_dynamics_compute_space_amplitude_writes_all_cases(monkeypatch, tmp_path):
    npz_path = tmp_path / "space_amplitude.npz"
    monkeypatch.setattr(pattern_dynamics, "FIG_DIR", tmp_path)
    monkeypatch.setattr(pattern_dynamics, "SPACE_NPZ", npz_path)

    pattern_dynamics.compute_space_amplitude()

    with np.load(npz_path) as data:
        for a, eps, _label, _tag in pattern_dynamics.SPACE_CASES:
            snap = data[f"a_{a}_eps_{eps}_snap"]
            assert snap.shape == (12, 100)
            assert np.all(np.isfinite(snap))
        np.testing.assert_array_equal(
            data["schema_version"], [pattern_dynamics.SPACE_SCHEMA_VERSION]
        )


def test_pattern_dynamics_plot_space_amplitude_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "space_amplitude.png"
    monkeypatch.setattr(pattern_dynamics, "SPACE_PNG", png_path)

    rng = np.random.default_rng(1)
    results = {}
    for a, eps, label, tag in pattern_dynamics.SPACE_CASES:
        results[f"a_{a}_eps_{eps}_snap"] = rng.uniform(-1, 1, (12, 20))
        results[f"a_{a}_eps_{eps}_label"] = np.array([label])
        results[f"a_{a}_eps_{eps}_tag"] = np.array([tag])
    results["params"] = np.array([(a, eps) for a, eps, _, _ in pattern_dynamics.SPACE_CASES])

    pattern_dynamics.plot_space_amplitude(results)

    assert png_path.stat().st_size > 0


def test_pattern_dynamics_plot_phase_diagram_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "phase_diagram.png"
    monkeypatch.setattr(pattern_dynamics, "PHASE_PNG", png_path)

    a = np.linspace(1.5, 2.0, 6)
    eps = np.linspace(0.0, 0.4, 5)
    rng = np.random.default_rng(2)
    data = {
        "a": a,
        "eps": eps,
        "spatial_activity": rng.uniform(0.0, 1.0, (5, 6)),
    }

    pattern_dynamics.plot_phase_diagram(data)

    assert png_path.stat().st_size > 0


def test_pattern_dynamics_main_computes_both_caches_when_missing(monkeypatch, tmp_path):
    phase_npz = tmp_path / "phase_diagram.npz"
    phase_png = tmp_path / "phase_diagram.png"
    space_npz = tmp_path / "space_amplitude.npz"
    space_png = tmp_path / "space_amplitude.png"
    monkeypatch.setattr(pattern_dynamics, "FIG_DIR", tmp_path)
    monkeypatch.setattr(pattern_dynamics, "PHASE_NPZ", phase_npz)
    monkeypatch.setattr(pattern_dynamics, "PHASE_PNG", phase_png)
    monkeypatch.setattr(pattern_dynamics, "SPACE_NPZ", space_npz)
    monkeypatch.setattr(pattern_dynamics, "SPACE_PNG", space_png)

    def fake_compute_phase_diagram():
        a = np.linspace(1.5, 2.0, 4)
        eps = np.linspace(0.0, 0.4, 3)
        np.savez_compressed(
            phase_npz,
            a=a,
            eps=eps,
            lam=np.zeros((3, 4)),
            spatial_activity=np.ones((3, 4)),
            schema_version=np.array([pattern_dynamics.PHASE_SCHEMA_VERSION]),
        )

    monkeypatch.setattr(pattern_dynamics, "compute_phase_diagram", fake_compute_phase_diagram)

    pattern_dynamics.main()

    assert phase_npz.exists()
    assert phase_png.stat().st_size > 0
    assert space_npz.exists()
    assert space_png.stat().st_size > 0


def test_pattern_dynamics_main_recomputes_stale_phase_cache(monkeypatch, tmp_path):
    phase_npz = tmp_path / "phase_diagram.npz"
    phase_png = tmp_path / "phase_diagram.png"
    space_npz = tmp_path / "space_amplitude.npz"
    space_png = tmp_path / "space_amplitude.png"
    monkeypatch.setattr(pattern_dynamics, "FIG_DIR", tmp_path)
    monkeypatch.setattr(pattern_dynamics, "PHASE_NPZ", phase_npz)
    monkeypatch.setattr(pattern_dynamics, "PHASE_PNG", phase_png)
    monkeypatch.setattr(pattern_dynamics, "SPACE_NPZ", space_npz)
    monkeypatch.setattr(pattern_dynamics, "SPACE_PNG", space_png)

    # A schema_version below the current one takes the KeyError/stale-cache
    # recompute branch (pattern_dynamics.py:311-314) instead of the
    # FileNotFoundError branch.
    a = np.linspace(1.5, 2.0, 4)
    eps = np.linspace(0.0, 0.4, 3)
    np.savez_compressed(
        phase_npz,
        a=a,
        eps=eps,
        lam=np.zeros((3, 4)),
        spatial_activity=np.ones((3, 4)),
        schema_version=np.array([0]),
    )

    def fake_compute_phase_diagram():
        np.savez_compressed(
            phase_npz,
            a=a,
            eps=eps,
            lam=np.zeros((3, 4)),
            spatial_activity=np.full((3, 4), 2.0),
            schema_version=np.array([pattern_dynamics.PHASE_SCHEMA_VERSION]),
        )

    monkeypatch.setattr(pattern_dynamics, "compute_phase_diagram", fake_compute_phase_diagram)

    # A valid, current-schema space-amplitude cache also pre-exists, so the
    # space half of main() takes the "Loaded" branch (pattern_dynamics.py:
    # 318-322) instead of recomputing.
    pattern_dynamics.compute_space_amplitude()
    with np.load(space_npz) as data:
        assert int(data["schema_version"][0]) == pattern_dynamics.SPACE_SCHEMA_VERSION

    pattern_dynamics.main()

    with np.load(phase_npz) as data:
        np.testing.assert_array_equal(
            data["schema_version"], [pattern_dynamics.PHASE_SCHEMA_VERSION]
        )
    assert space_npz.exists()


def test_pattern_dynamics_main_recomputes_stale_space_cache(monkeypatch, tmp_path):
    phase_npz = tmp_path / "phase_diagram.npz"
    phase_png = tmp_path / "phase_diagram.png"
    space_npz = tmp_path / "space_amplitude.npz"
    space_png = tmp_path / "space_amplitude.png"
    monkeypatch.setattr(pattern_dynamics, "FIG_DIR", tmp_path)
    monkeypatch.setattr(pattern_dynamics, "PHASE_NPZ", phase_npz)
    monkeypatch.setattr(pattern_dynamics, "PHASE_PNG", phase_png)
    monkeypatch.setattr(pattern_dynamics, "SPACE_NPZ", space_npz)
    monkeypatch.setattr(pattern_dynamics, "SPACE_PNG", space_png)

    def fake_compute_phase_diagram():
        a = np.linspace(1.5, 2.0, 4)
        eps = np.linspace(0.0, 0.4, 3)
        np.savez_compressed(
            phase_npz,
            a=a,
            eps=eps,
            lam=np.zeros((3, 4)),
            spatial_activity=np.ones((3, 4)),
            schema_version=np.array([pattern_dynamics.PHASE_SCHEMA_VERSION]),
        )

    monkeypatch.setattr(pattern_dynamics, "compute_phase_diagram", fake_compute_phase_diagram)

    # A stale space-amplitude cache (schema_version below current) takes the
    # KeyError recompute branch (pattern_dynamics.py:327-330).
    pattern_dynamics.compute_space_amplitude()
    with np.load(space_npz) as data:
        stale = {k: data[k] for k in data.files}
    stale["schema_version"] = np.array([0])
    np.savez_compressed(space_npz, **stale)

    pattern_dynamics.main()

    with np.load(space_npz) as data:
        np.testing.assert_array_equal(
            data["schema_version"], [pattern_dynamics.SPACE_SCHEMA_VERSION]
        )


# ---------------------------------------------------------------------------
# correlation_figure
#
# compute_lyapunov_density() sweeps 4 a values x 8 subsystem sizes, each with
# a 5000 step transient plus a 20000 step Jacobian-propagated measurement
# (src/dynachaos/cml/correlation_figure.py:92-145). Measured at ~9.4s for the
# real parameters, over the per-test 5s budget, so it is left uncovered; the
# lighter compute_correlations() (~0.7s) is exercised directly instead.
# ---------------------------------------------------------------------------


def test_correlation_figure_compute_correlations_normalizes_to_one_at_zero_lag():
    a_values, r_vals, all_corr = correlation_figure.compute_correlations()

    np.testing.assert_allclose(a_values, [1.5, 1.7, 1.85, 1.95])
    assert r_vals[0] == 0
    np.testing.assert_allclose(all_corr[:, 0], 1.0)
    assert np.all(np.isfinite(all_corr))


def test_correlation_figure_compute_combines_correlations_and_fits_xi(monkeypatch, tmp_path):
    npz_path = tmp_path / "correlation_decay.npz"
    monkeypatch.setattr(correlation_figure, "FIG_DIR", tmp_path)
    monkeypatch.setattr(correlation_figure, "CORR_NPZ", npz_path)

    def fake_lyapunov_density():
        a_values = np.array([1.5, 1.7, 1.85, 1.95])
        L_values = np.array([10, 20])
        density = np.full((4, 2), 0.01)
        return a_values, L_values, density

    monkeypatch.setattr(correlation_figure, "compute_lyapunov_density", fake_lyapunov_density)

    correlation_figure.compute()

    with np.load(npz_path) as data:
        assert data["all_corr"].shape[0] == 4
        assert data["xi_values"].shape == (4,)
        assert data["density"].shape == (4, 2)


def test_correlation_figure_plot_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "correlation_decay.png"
    monkeypatch.setattr(correlation_figure, "CORR_PNG", png_path)

    r_vals = np.arange(0, 20)
    a_corr = np.array([1.5, 1.7, 1.85, 1.95])
    all_corr = np.exp(-r_vals[None, :] / np.array([[2.0], [4.0], [8.0], [16.0]]))
    xi_values = np.array([2.0, 4.0, 8.0, np.nan])
    a_lyap = a_corr
    L_vals = np.array([10, 20, 40])
    density = np.tile(np.linspace(0.1, 0.05, 3), (4, 1))

    data = {
        "a_corr": a_corr,
        "r_vals": r_vals,
        "all_corr": all_corr,
        "xi_values": xi_values,
        "a_lyap": a_lyap,
        "L_vals": L_vals,
        "density": density,
    }

    correlation_figure.plot(data)

    assert png_path.stat().st_size > 0


def test_correlation_figure_main_computes_when_cache_missing(monkeypatch, tmp_path):
    npz_path = tmp_path / "correlation_decay.npz"
    png_path = tmp_path / "correlation_decay.png"
    monkeypatch.setattr(correlation_figure, "FIG_DIR", tmp_path)
    monkeypatch.setattr(correlation_figure, "CORR_NPZ", npz_path)
    monkeypatch.setattr(correlation_figure, "CORR_PNG", png_path)

    def fake_compute():
        r_vals = np.arange(0, 10)
        a_vals = np.array([1.5, 1.7])
        all_corr = np.exp(-r_vals[None, :] / np.array([[2.0], [3.0]]))
        np.savez_compressed(
            npz_path,
            a_corr=a_vals,
            r_vals=r_vals,
            all_corr=all_corr,
            xi_values=np.array([2.0, 3.0]),
            a_lyap=a_vals,
            L_vals=np.array([10, 20]),
            density=np.full((2, 2), 0.01),
        )

    monkeypatch.setattr(correlation_figure, "compute", fake_compute)

    correlation_figure.main()

    assert npz_path.exists()
    assert png_path.stat().st_size > 0


def test_correlation_figure_main_plots_from_cache_when_present(monkeypatch, tmp_path):
    npz_path = tmp_path / "correlation_decay.npz"
    png_path = tmp_path / "correlation_decay.png"
    monkeypatch.setattr(correlation_figure, "FIG_DIR", tmp_path)
    monkeypatch.setattr(correlation_figure, "CORR_NPZ", npz_path)
    monkeypatch.setattr(correlation_figure, "CORR_PNG", png_path)

    r_vals = np.arange(0, 10)
    a_vals = np.array([1.5, 1.7])
    all_corr = np.exp(-r_vals[None, :] / np.array([[2.0], [3.0]]))
    np.savez_compressed(
        npz_path,
        a_corr=a_vals,
        r_vals=r_vals,
        all_corr=all_corr,
        xi_values=np.array([2.0, 3.0]),
        a_lyap=a_vals,
        L_vals=np.array([10, 20]),
        density=np.full((2, 2), 0.01),
    )

    # The cache already exists, so main() takes the "Loaded" branch
    # (correlation_figure.py:354) rather than calling compute().
    correlation_figure.main()

    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# comoving_figure
#
# compute() propagates a two-copy Benettin scheme on an N=500 lattice across
# 301 co-moving velocities, each with a 20000 step transient plus a 100000
# step measurement (src/dynachaos/cml/comoving_figure.py:37-76) -- not
# reachable at test scale. plot() and main()'s cache-miss branch are
# exercised with a fast synthetic replacement for compute().
# ---------------------------------------------------------------------------


def test_comoving_figure_plot_writes_png_and_marks_crossings(monkeypatch, tmp_path):
    # plot() writes to the module constant. Without this redirection the test
    # overwrites the committed paper figure in figures/sec08_sti/.
    png_path = tmp_path / "comoving_lyapunov.png"
    monkeypatch.setattr(comoving_figure, "FIG_DIR", tmp_path)
    monkeypatch.setattr(comoving_figure, "OUTPUT_PNG", png_path)

    v_values = np.linspace(-1.5, 1.5, 61)
    a_values = np.array([1.70, 1.85])
    lam_170 = -v_values * 0.2
    # Below the -9.5 validity floor at the first two points, so the adjacent
    # pair (invalid, valid) hits the "continue" skip when scanning for zero
    # crossings (comoving_figure.py:132-134).
    lam_170[:2] = -20.0
    data = {
        "v_values": v_values,
        "a_values": a_values,
        # This curve vanishes exactly on the grid point v = 0, so the product
        # of two neighbours is zero and the scan records no crossing.
        "lambda_a1.70": lam_170,
        # This one has its root at v = 0.05 / 0.3 = 1/6, between the grid
        # points 0.15 and 0.20, so the scan does record a crossing.
        "lambda_a1.85": -v_values * 0.3 + 0.05,
    }

    # The crossing label is the only visible evidence that the scan ran, and
    # plot() closes the figure before returning. Record the text as it is drawn.
    labels: list[str] = []
    draw_text = Axes.text

    def record_text(self, x, y, s, *args, **kwargs):
        labels.append(s)
        return draw_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", record_text)

    comoving_figure.plot(data)

    assert png_path.stat().st_size > 0
    # lambda is linear in v here, so the linear interpolation between the two
    # neighbouring grid points must return the exact root 1/6 = 0.1667.
    assert any(r"v_p(a=1.85) \approx 0.17" in label for label in labels)
    assert not any("a=1.70" in label for label in labels)


def test_comoving_figure_main_computes_when_cache_missing(monkeypatch, tmp_path):
    npz_path = tmp_path / "comoving_lyapunov.npz"
    png_path = tmp_path / "comoving_lyapunov.png"
    monkeypatch.setattr(comoving_figure, "FIG_DIR", tmp_path)
    monkeypatch.setattr(comoving_figure, "OUTPUT_NPZ", npz_path)
    monkeypatch.setattr(comoving_figure, "OUTPUT_PNG", png_path)

    def fake_compute():
        v_values = np.linspace(-1.5, 1.5, 11)
        np.savez_compressed(
            npz_path,
            v_values=v_values,
            a_values=np.array([1.70]),
            eps=np.array([0.3]),
            N=np.array([500]),
            **{"lambda_a1.70": -v_values * 0.1},
        )

    monkeypatch.setattr(comoving_figure, "compute", fake_compute)

    comoving_figure.main()

    assert npz_path.exists()
    assert png_path.stat().st_size > 0


def test_comoving_figure_main_plots_from_cache_when_present(monkeypatch, tmp_path):
    npz_path = tmp_path / "comoving_lyapunov.npz"
    png_path = tmp_path / "comoving_lyapunov.png"
    monkeypatch.setattr(comoving_figure, "FIG_DIR", tmp_path)
    monkeypatch.setattr(comoving_figure, "OUTPUT_NPZ", npz_path)
    monkeypatch.setattr(comoving_figure, "OUTPUT_PNG", png_path)

    v_values = np.linspace(-1.5, 1.5, 11)
    np.savez_compressed(
        npz_path,
        v_values=v_values,
        a_values=np.array([1.70]),
        eps=np.array([0.3]),
        N=np.array([500]),
        **{"lambda_a1.70": -v_values * 0.1},
    )

    # The cache already exists, so main() takes the "Loaded" branch
    # (comoving_figure.py:186) rather than calling compute().
    comoving_figure.main()

    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# gcm_clusters
# ---------------------------------------------------------------------------


def test_gcm_clusters_detect_clusters_matches_shared_helper():
    values = np.array([0.10, 0.1000002, 0.50])

    np.testing.assert_array_equal(
        gcm_clusters.detect_clusters(values, tol=1e-5),
        gcm_clusters.cluster_labels_by_tolerance(values, tol=1e-5),
    )


def test_gcm_clusters_compute_collective_defaults_a_values_when_omitted():
    # a_values=None takes the np.linspace(1.4, 2.0, 100) default
    # (src/dynachaos/cml/gcm_clusters.py:137); n_transient/n_measure are
    # overridden to zero cost so the 100-value sweep still runs in well
    # under a second.
    payload = gcm_clusters.compute_collective(
        n_sites=4,
        n_transient=0,
        n_measure=1,
        renorm_interval=1,
        seed=1,
        output_path=None,
        progress_interval=0,
    )

    assert payload["a_values"].shape == (100,)
    np.testing.assert_allclose(payload["a_values"][0], 1.4)
    np.testing.assert_allclose(payload["a_values"][-1], 2.0)


def test_gcm_clusters_compute_collective_zero_perturbation_reports_zero_lyapunov():
    payload = gcm_clusters.compute_collective(
        n_sites=4,
        a_values=np.array([1.5]),
        n_transient=0,
        n_measure=4,
        renorm_interval=1,
        h_delta=0.0,
        seed=1,
        output_path=None,
        progress_interval=0,
    )

    # h_delta=0 means the two copies never separate, so n_renorm stays 0 and
    # the fallback branch (src/dynachaos/cml/gcm_clusters.py:191) reports 0.
    assert payload["lyap_c"][0] == 0.0


def test_gcm_clusters_compute_collective_writes_progress_checkpoints(tmp_path):
    output_path = tmp_path / "collective_lyapunov.npz"

    gcm_clusters.compute_collective(
        n_sites=4,
        a_values=np.array([1.4, 1.6]),
        n_transient=1,
        n_measure=2,
        renorm_interval=1,
        seed=3,
        output_path=output_path,
        progress_interval=1,
    )

    with np.load(output_path) as saved:
        # The final checkpoint (progress_interval=1 fires every a) leaves both
        # entries in the cache.
        assert saved["a_values"].shape == (2,)


def test_gcm_clusters_plot_clusters_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "gcm_clusters.png"
    monkeypatch.setattr(gcm_clusters, "CLUSTER_PNG", png_path)

    payload = gcm_clusters.compute_clusters(
        n_sites=8, n_transient=2, n_record=4, seed=5, output_path=None
    )

    gcm_clusters.plot_clusters(payload)

    assert png_path.stat().st_size > 0


def test_gcm_clusters_plot_collective_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "collective_lyapunov.png"
    monkeypatch.setattr(gcm_clusters, "COLL_PNG", png_path)

    payload = gcm_clusters.compute_collective(
        n_sites=8,
        a_values=np.linspace(1.4, 2.0, 6),
        n_transient=1,
        n_measure=2,
        renorm_interval=1,
        seed=7,
        output_path=None,
        progress_interval=0,
    )

    gcm_clusters.plot_collective(payload)

    assert png_path.stat().st_size > 0


def test_gcm_clusters_plot_collective_fills_sustained_positive_band(monkeypatch, tmp_path):
    png_path = tmp_path / "collective_lyapunov.png"
    monkeypatch.setattr(gcm_clusters, "COLL_PNG", png_path)

    # A run of >= min_run=4 positive entries triggers the shaded
    # sustained-lambda_c > 0 fill_between branch (gcm_clusters.py:329-341).
    a_values = np.linspace(1.4, 2.0, 8)
    lyap_c = np.array([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.2, -0.1])
    payload = {
        "a_values": a_values,
        "lyap_c": lyap_c,
        "eps": np.array([0.1]),
        "N": np.array([8]),
    }

    gcm_clusters.plot_collective(payload)

    assert png_path.stat().st_size > 0


def test_gcm_clusters_main_computes_and_plots_when_cache_missing(monkeypatch, tmp_path):
    cluster_npz = tmp_path / "gcm_clusters.npz"
    cluster_png = tmp_path / "gcm_clusters.png"
    coll_npz = tmp_path / "collective_lyapunov.npz"
    coll_png = tmp_path / "collective_lyapunov.png"
    monkeypatch.setattr(gcm_clusters, "FIG_DIR", tmp_path)
    monkeypatch.setattr(gcm_clusters, "CLUSTER_NPZ", cluster_npz)
    monkeypatch.setattr(gcm_clusters, "CLUSTER_PNG", cluster_png)
    monkeypatch.setattr(gcm_clusters, "COLL_NPZ", coll_npz)
    monkeypatch.setattr(gcm_clusters, "COLL_PNG", coll_png)

    original_compute_clusters = gcm_clusters.compute_clusters
    original_compute_collective = gcm_clusters.compute_collective

    def fake_compute_clusters(**_kwargs):
        return original_compute_clusters(
            n_sites=8, n_transient=2, n_record=4, seed=1, output_path=None
        )

    def fake_compute_collective(**_kwargs):
        return original_compute_collective(
            n_sites=8,
            a_values=np.array([1.4, 1.6]),
            n_transient=1,
            n_measure=2,
            renorm_interval=1,
            seed=1,
            output_path=None,
            progress_interval=0,
        )

    monkeypatch.setattr(gcm_clusters, "compute_clusters", fake_compute_clusters)
    monkeypatch.setattr(gcm_clusters, "compute_collective", fake_compute_collective)

    gcm_clusters.main()

    assert cluster_png.stat().st_size > 0
    assert coll_png.stat().st_size > 0


def test_gcm_clusters_main_plots_from_cache_when_both_files_exist(monkeypatch, tmp_path):
    cluster_npz = tmp_path / "gcm_clusters.npz"
    cluster_png = tmp_path / "gcm_clusters.png"
    coll_npz = tmp_path / "collective_lyapunov.npz"
    coll_png = tmp_path / "collective_lyapunov.png"
    monkeypatch.setattr(gcm_clusters, "FIG_DIR", tmp_path)
    monkeypatch.setattr(gcm_clusters, "CLUSTER_NPZ", cluster_npz)
    monkeypatch.setattr(gcm_clusters, "CLUSTER_PNG", cluster_png)
    monkeypatch.setattr(gcm_clusters, "COLL_NPZ", coll_npz)
    monkeypatch.setattr(gcm_clusters, "COLL_PNG", coll_png)

    gcm_clusters.compute_clusters(
        n_sites=8, n_transient=2, n_record=4, seed=1, output_path=cluster_npz
    )
    gcm_clusters.compute_collective(
        n_sites=8,
        a_values=np.array([1.4, 1.6]),
        n_transient=1,
        n_measure=2,
        renorm_interval=1,
        seed=1,
        output_path=coll_npz,
        progress_interval=0,
    )

    # Both caches already exist, so main() takes the "Loaded" branch
    # (gcm_clusters.py:369-370, 378-379) rather than recomputing.
    gcm_clusters.main()

    assert cluster_png.stat().st_size > 0
    assert coll_png.stat().st_size > 0


# ---------------------------------------------------------------------------
# globally_coupled
#
# compute() runs 100000-step mean-field samples for 5 lattice sizes, then an
# 8 x 5 (N, a) grid each with a 5000 step transient plus a 50000 step sample
# (src/dynachaos/cml/globally_coupled.py:35-92) -- not reachable at test
# scale. plot_msd/plot_distribution are exercised with tiny synthetic
# payloads shaped like the real schema.
# ---------------------------------------------------------------------------


def test_globally_coupled_plot_msd_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "gcm_msd.png"
    monkeypatch.setattr(globally_coupled, "GCM_PNG", png_path)

    N_grid = np.array([100, 200, 500, 1000])
    a_for_msd = np.array([1.80, 1.85, 1.92, 1.95, 1.99])
    msd_grid = np.outer(1.0 / (a_for_msd - 1.5), 1.0 / N_grid)

    data = {
        "N_grid": N_grid,
        "msd_grid": msd_grid,
        "a_for_msd": a_for_msd,
        "eps": np.array([0.1]),
    }

    globally_coupled.plot_msd(data)

    assert png_path.stat().st_size > 0


def test_globally_coupled_plot_distribution_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "gcm_distribution.png"
    monkeypatch.setattr(globally_coupled, "DIST_PNG", png_path)

    rng = np.random.default_rng(4)
    N_values = np.array([100, 400, 1000])
    data = {"N_values": N_values, "a": np.array([1.99]), "eps": np.array([0.1])}
    for N in N_values:
        h = rng.normal(0.3, 0.05 / np.sqrt(N), size=200)
        data[f"N_{N}_h"] = h
        data[f"N_{N}_mean"] = np.array([h.mean()])
        data[f"N_{N}_msd"] = np.array([h.var()])

    globally_coupled.plot_distribution(data)

    assert png_path.stat().st_size > 0


def test_globally_coupled_main_plots_from_cache(monkeypatch, tmp_path):
    npz_path = tmp_path / "gcm_results.npz"
    gcm_png = tmp_path / "gcm_msd.png"
    dist_png = tmp_path / "gcm_distribution.png"
    monkeypatch.setattr(globally_coupled, "FIG_DIR", tmp_path)
    monkeypatch.setattr(globally_coupled, "GCM_NPZ", npz_path)
    monkeypatch.setattr(globally_coupled, "GCM_PNG", gcm_png)
    monkeypatch.setattr(globally_coupled, "DIST_PNG", dist_png)

    rng = np.random.default_rng(9)
    N_values = np.array([100, 400])
    payload = {"N_values": N_values, "a": np.array([1.99]), "eps": np.array([0.1])}
    for N in N_values:
        h = rng.normal(0.3, 0.05, size=50)
        payload[f"N_{N}_h"] = h
        payload[f"N_{N}_mean"] = np.array([h.mean()])
        payload[f"N_{N}_msd"] = np.array([h.var()])
    payload["a_for_msd"] = np.array([1.80, 1.99])
    payload["N_grid"] = np.array([100, 400])
    payload["msd_grid"] = np.array([[0.02, 0.01], [0.05, 0.03]])
    np.savez_compressed(npz_path, **payload)

    globally_coupled.main()

    assert gcm_png.stat().st_size > 0
    assert dist_png.stat().st_size > 0


def test_globally_coupled_main_computes_when_cache_missing(monkeypatch, tmp_path):
    npz_path = tmp_path / "gcm_results.npz"
    gcm_png = tmp_path / "gcm_msd.png"
    dist_png = tmp_path / "gcm_distribution.png"
    monkeypatch.setattr(globally_coupled, "FIG_DIR", tmp_path)
    monkeypatch.setattr(globally_coupled, "GCM_NPZ", npz_path)
    monkeypatch.setattr(globally_coupled, "GCM_PNG", gcm_png)
    monkeypatch.setattr(globally_coupled, "DIST_PNG", dist_png)

    def fake_compute():
        rng = np.random.default_rng(11)
        N_values = np.array([100, 400])
        payload = {"N_values": N_values, "a": np.array([1.99]), "eps": np.array([0.1])}
        for N in N_values:
            h = rng.normal(0.3, 0.05, size=30)
            payload[f"N_{N}_h"] = h
            payload[f"N_{N}_mean"] = np.array([h.mean()])
            payload[f"N_{N}_msd"] = np.array([h.var()])
        payload["a_for_msd"] = np.array([1.80, 1.99])
        payload["N_grid"] = np.array([100, 400])
        payload["msd_grid"] = np.array([[0.02, 0.01], [0.05, 0.03]])
        np.savez_compressed(npz_path, **payload)

    monkeypatch.setattr(globally_coupled, "compute", fake_compute)

    globally_coupled.main()

    assert npz_path.exists()
    assert gcm_png.stat().st_size > 0
    assert dist_png.stat().st_size > 0


# ---------------------------------------------------------------------------
# spatiotemporal edge cases
# ---------------------------------------------------------------------------


def _flat_spacetime_payload(value=1.0):
    """Return a payload with all nine series constant at one value."""
    return {
        f"{model}_eps_{eps}": np.full((10, 20), value)
        for model, eps_vals in [
            ("A", [0.06, 0.07, 0.08]),
            ("B", [0.02, 0.024, 0.03]),
            ("C", [0.16, 0.20, 0.30]),
        ]
        for eps in eps_vals
    }


def test_spatiotemporal_plot_widens_the_colour_range_on_a_flat_field(monkeypatch, tmp_path):
    # A constant field puts the 1st and 99th percentile on the same value, so
    # vmax - vmin is zero and imshow would get an empty colour range. plot()
    # opens it to +/- 0.5 around the value (spatiotemporal.py:160-162).
    png_path = tmp_path / "spacetime_diagrams.png"
    monkeypatch.setattr(spatiotemporal, "STI_PNG", png_path)

    ranges = []
    draw_image = Axes.imshow

    def record_imshow(self, X, *args, **kwargs):
        ranges.append((kwargs["vmin"], kwargs["vmax"]))
        return draw_image(self, X, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", record_imshow)

    spatiotemporal.plot(_flat_spacetime_payload(value=1.0))

    assert png_path.stat().st_size > 0
    assert len(ranges) == 9
    for vmin, vmax in ranges:
        assert (vmin, vmax) == (0.5, 1.5)


def test_spatiotemporal_main_plots_from_cache_without_computing(monkeypatch, tmp_path):
    npz_path = tmp_path / "spacetime_diagrams.npz"
    png_path = tmp_path / "spacetime_diagrams.png"
    monkeypatch.setattr(spatiotemporal, "FIG_DIR", tmp_path)
    monkeypatch.setattr(spatiotemporal, "STI_NPZ", npz_path)
    monkeypatch.setattr(spatiotemporal, "STI_PNG", png_path)
    np.savez_compressed(npz_path, **_flat_spacetime_payload(value=0.25))

    def fail_if_called():
        raise AssertionError("main() recomputed although the cache was present")

    monkeypatch.setattr(spatiotemporal, "compute", fail_if_called)

    spatiotemporal.main()

    assert png_path.stat().st_size > 0


# ── Coverage: comoving_lyapunov edge cases ────────────────────────────────────


def test_comoving_lyapunov_x_init_shape_validation():
    """comoving_lyapunov._initial_state validates x_init shape (line 50)."""
    from dynachaos.diagnostics.comoving_lyapunov import _initial_state

    with pytest.raises(ValueError, match="x_init must have shape"):
        _initial_state(N=10, x_init=np.array([1.0, 2.0]))


def test_comoving_lyapunov_default_x_init():
    """comoving_lyapunov uses default x_init when None (line 99)."""
    from dynachaos.diagnostics.comoving_lyapunov import _initial_state

    x_init = _initial_state(N=5, x_init=None)
    assert x_init.shape == (5,)
    assert np.all(np.isfinite(x_init))


def test_comoving_lyapunov_x_init_validation():
    """comoving_lyapunov validates x_init shape (line 50)."""
    from dynachaos.diagnostics.comoving_lyapunov import comoving_lyapunov_spectrum_logistic

    with pytest.raises(ValueError, match="x_init must have shape"):
        comoving_lyapunov_spectrum_logistic(
            a=1.8,
            eps=0.1,
            N=10,
            v_values=[0.0],
            n_iter=5,
            n_transient=2,
            x_init=np.array([1.0, 2.0]),
        )
