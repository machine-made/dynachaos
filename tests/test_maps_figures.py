"""Coverage tests for the maps plotting functions: delayed_logistic,
torus_doubling, coupled_logistic.

Each module's compute_* functions accept tunable n_transient/n_plot/n_iter
and an explicit output_path, and are already exercised at paper scale by
test_maps.py. Here they are called at test scale (tiny counts, output_path
in tmp_path) purely to produce a payload dict shaped like the real .npz
schema, which is then fed to the plot_* function with the PNG path
monkeypatched to tmp_path -- the same monkeypatch-the-module-constant
pattern used throughout test_maps.py.

Animation/GIF paths (compute_*_animation, make_*_animation_gif) and each
module's __main__ guard are not exercised: the animation sweeps are
hardcoded to 200 D values with a 20000-step transient each
(delayed_logistic.py:458-466, torus_doubling.py:385-397/417-429,
coupled_logistic.py:729-...), well over the test time budget, and GIF
encoding is out of scope per the coverage-campaign brief.
"""

import sys

import numpy as np

# dynachaos.maps re-exports functions named identically to their defining
# submodules (e.g. the package attribute "coupled_logistic" is the map
# function, not the module), so `from dynachaos.maps import coupled_logistic`
# would bind the function instead of the module. Pull the actual submodule
# objects out of sys.modules after importing them.
import dynachaos.maps.coupled_logistic  # noqa: F401
import dynachaos.maps.delayed_logistic  # noqa: F401
import dynachaos.maps.torus_doubling  # noqa: F401

coupled_logistic = sys.modules["dynachaos.maps.coupled_logistic"]
delayed_logistic = sys.modules["dynachaos.maps.delayed_logistic"]
torus_doubling = sys.modules["dynachaos.maps.torus_doubling"]

# ---------------------------------------------------------------------------
# delayed_logistic
# ---------------------------------------------------------------------------


def test_delayed_logistic_plot_attractors_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "attractors.png"
    monkeypatch.setattr(delayed_logistic, "ATTR_PNG", png_path)

    payload = delayed_logistic.compute_attractors(n_transient=2, n_plot=5, output_path=None)

    delayed_logistic.plot_attractors(payload)

    assert png_path.stat().st_size > 0


def test_delayed_logistic_plot_lyapunov_writes_png_with_convergence_inset(monkeypatch, tmp_path):
    png_path = tmp_path / "lyapunov_vs_D.png"
    monkeypatch.setattr(delayed_logistic, "LYAP_PNG", png_path)

    # D values inside [1.85, 2.15] so the convergence-band inset branch
    # (delayed_logistic.py:332-351) is exercised too.
    payload = delayed_logistic.compute_lyapunov_spectrum(
        D_values=np.linspace(1.85, 2.15, 4),
        n_iter=20,
        n_transient=5,
        output_path=None,
        progress_interval=0,
    )

    delayed_logistic.plot_lyapunov(payload)

    assert png_path.stat().st_size > 0


def test_delayed_logistic_plot_locking_sequence_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "locking_sequence.png"
    monkeypatch.setattr(delayed_logistic, "LOCK_PNG", png_path)

    payload = delayed_logistic.compute_locking_sequence(n_transient=2, n_plot=5, output_path=None)

    delayed_logistic.plot_locking_sequence(payload)

    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# torus_doubling
# ---------------------------------------------------------------------------


def test_torus_doubling_plot_map_I_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "map_I_attractors.png"
    monkeypatch.setattr(torus_doubling, "MAP1_PNG", png_path)

    payload = torus_doubling.compute_map_I(n_transient=2, n_plot=5, output_path=None)

    torus_doubling.plot_map_I(payload)

    assert png_path.stat().st_size > 0


def test_torus_doubling_plot_map_IV_writes_png_with_zoom_inset(monkeypatch, tmp_path):
    png_path = tmp_path / "map_IV_attractors.png"
    monkeypatch.setattr(torus_doubling, "MAP4_PNG", png_path)

    payload = torus_doubling.compute_map_IV(n_transient=2, n_plot=30, output_path=None)

    torus_doubling.plot_map_IV(payload)

    assert png_path.stat().st_size > 0


def test_torus_doubling_plot_lyapunov_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "map_IV_lyapunov.png"
    monkeypatch.setattr(torus_doubling, "LYAP_PNG", png_path)

    payload = torus_doubling.compute_map_IV_lyapunov(
        D_values=np.linspace(1.48, 1.53, 4),
        n_iter=20,
        n_transient=5,
        output_path=None,
        progress_interval=0,
    )

    torus_doubling.plot_lyapunov(payload)

    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# coupled_logistic
# ---------------------------------------------------------------------------


def test_coupled_logistic_plot_phase_diagram_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "phase_diagram.png"
    monkeypatch.setattr(coupled_logistic, "PHASE_PNG", png_path)

    payload = coupled_logistic.compute_phase_diagram(
        A_values=np.array([0.8, 1.0, 1.2]),
        D_values=np.array([0.0, 0.1]),
        n_transient=2,
        n_sample=3,
        output_path=None,
        progress_interval=0,
    )

    coupled_logistic.plot_phase_diagram(payload)

    assert png_path.stat().st_size > 0


def test_coupled_logistic_plot_attractors_writes_png(monkeypatch, tmp_path):
    png_path = tmp_path / "attractors.png"
    monkeypatch.setattr(coupled_logistic, "ATTR_PNG", png_path)

    payload = coupled_logistic.compute_attractors(n_transient=2, n_plot=5, output_path=None)

    coupled_logistic.plot_attractors(payload)

    assert png_path.stat().st_size > 0


def test_coupled_logistic_plot_basins_writes_png_and_marks_divergence(monkeypatch, tmp_path):
    png_path = tmp_path / "basins.png"
    monkeypatch.setattr(coupled_logistic, "BASIN_PNG", png_path)

    payload = coupled_logistic.compute_basins(
        n_grid=6,
        n_transient=2,
        reference_transient=2,
        period=2,
        output_path=None,
    )
    # Force at least one diverged cell so the "Diverged" legend-patch branch
    # (coupled_logistic.py:702-706) is exercised too.
    payload["basin"][0, 0] = -1

    coupled_logistic.plot_basins(payload)

    assert png_path.stat().st_size > 0
