import pytest

import dynachaos
from dynachaos.io.paths import output_root, section_dir


def test_package_imports():
    assert hasattr(dynachaos, "__all__")


def test_submodule_imports():
    from dynachaos.diagnostics.permutation import permutation_entropy
    from dynachaos.maps.circle_map import circle_map

    assert callable(circle_map)
    assert callable(permutation_entropy)


def test_output_path_helpers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert output_root() == tmp_path / "figures"
    assert section_dir("sec02_circle_map") == tmp_path / "figures" / "sec02_circle_map"


def test_output_root_honors_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom_output"
    monkeypatch.setenv("DYNACHAOS_OUTPUT_ROOT", str(override))
    assert output_root() == override.expanduser().resolve()


# ---------------------------------------------------------------------------
# Re-export smoke tests
# ---------------------------------------------------------------------------


def test_maps_reexports():
    from dynachaos.maps import circle_map, delayed_logistic, logistic

    assert all(callable(f) for f in [logistic, delayed_logistic, circle_map])


def test_diagnostics_reexports():
    from dynachaos.diagnostics import correlation_integral, lyapunov_exponent_1d

    assert all(callable(f) for f in [correlation_integral, lyapunov_exponent_1d])


def test_cml_reexports():
    from dynachaos.cml import cml_step, gcm_step

    assert all(callable(f) for f in [cml_step, gcm_step])


def test_safe_load_reexport():
    from dynachaos.io import safe_load

    assert callable(safe_load)


def test_viz_reexports():
    pytest.importorskip("matplotlib")
    from dynachaos.viz import poincare_section_plot, return_map_plot

    assert callable(poincare_section_plot)
    assert callable(return_map_plot)


def test_no_rust_env_returns_false_without_importing():
    from dynachaos import _ensure_rust_backend

    def failing_importer(name):
        raise AssertionError(f"unexpected import: {name}")

    assert (
        _ensure_rust_backend(env={"DYNACHAOS_NO_RUST": "1"}, import_module=failing_importer)
        is False
    )


def test_rust_backend_available():
    from dynachaos import _ensure_rust_backend

    assert _ensure_rust_backend(env={}) is True


def test_missing_rust_backend_raises_runtime_error():
    from dynachaos import _ensure_rust_backend

    def failing_importer(name):
        raise ImportError(name)

    with pytest.raises(RuntimeError, match="maturin develop"):
        _ensure_rust_backend(env={}, import_module=failing_importer)
