import numpy as np
import pytest

from dynachaos.cml import sti_spine_figure
from dynachaos.pipelines.registry import get_section


def test_sti_spine_compute_writes_golden_cache(tmp_path):
    output_path = tmp_path / "sti_spine.npz"

    payload = sti_spine_figure.compute(output_path)

    assert output_path.exists()
    assert tuple(payload) == sti_spine_figure.REQUIRED_KEYS
    np.testing.assert_array_equal(payload["schema_version"], [2])
    np.testing.assert_array_equal(payload["seed"], [20260602])
    np.testing.assert_equal(payload["source_file"][0], "src/dynachaos/cml/sti_spine_figure.py")
    np.testing.assert_allclose(payload["model_a_parameter"], [-0.01])
    np.testing.assert_allclose(payload["display_eps"], [0.08])
    np.testing.assert_equal(payload["spacetime"].shape, (480, 512))
    np.testing.assert_equal(payload["turbulent_mask"].shape, (480, 512))
    np.testing.assert_equal(payload["sweep_eps"].shape, (9,))
    np.testing.assert_equal(payload["turbulent_fraction"].shape, (9,))
    # Kaneko (1985) model-A STI with a=-0.01: sub-critical coupling is laminar
    # (no bursts), super-critical coupling is burst-dominated, with a sharp onset
    # in between (the eps sweep brackets the transition).
    turbulent_fraction = payload["turbulent_fraction"]
    np.testing.assert_array_less(turbulent_fraction[0], 0.05)
    np.testing.assert_array_less(0.5, turbulent_fraction[-1])
    np.testing.assert_array_less(0.7, turbulent_fraction.max())
    np.testing.assert_array_less(20, payload["laminar_cluster_sizes"].size)
    np.testing.assert_array_less(0.0, payload["cluster_decay_rate"])


def test_sti_spine_plot_writes_png(tmp_path):
    payload = sti_spine_figure.compute(None, n_sites=128, n_transient=200, n_record=120)
    output_path = tmp_path / "sti_spine.png"

    result = sti_spine_figure.plot(payload, output_path)

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_sti_spine_laminar_cluster_runs_wrap_periodic_rows():
    mask = np.array([[True, True, False, True], [False, True, True, False]])

    lengths = sti_spine_figure._periodic_run_lengths(mask)

    np.testing.assert_array_equal(np.sort(lengths), [2, 3])


def test_sti_spine_laminar_cluster_runs_reject_non_spacetime_mask():
    with pytest.raises(ValueError, match="2D spacetime"):
        sti_spine_figure._periodic_run_lengths(np.array([True, False]))


def test_sti_spine_registry_contract_matches_module_keys():
    spec = get_section("sec12_intermittency")
    contract_keys = spec.required_npz_keys("sti_spine.npz")

    assert "dynachaos.cml.sti_spine_figure" in spec.modules
    assert "sti_spine.npz" in spec.cache_files
    assert "sti_spine.npz" in spec.output_files
    assert "sti_spine.png" in spec.output_files
    assert tuple(contract_keys) == sti_spine_figure.REQUIRED_KEYS
