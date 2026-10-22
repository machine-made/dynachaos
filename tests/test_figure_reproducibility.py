import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import assert_npz_structurally_sound, is_reference_platform

from dynachaos.io.paths import safe_load
from dynachaos.pipelines.registry import get_section
from dynachaos.pipelines.runner import run_section

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_FIGURES = REPO_ROOT / "figures"
FAST_REPRODUCIBILITY_SECTIONS = (
    "sec03_transition",
    "sec08_sti",
    "sec09_pattern",
    "sec10_gcm",
    "sec11_diagnostics",
)
HEAVY_REPRODUCIBILITY_SECTIONS = (
    "sec02_circle_map",
    "sec04_doubling",
    "sec05_oscillation",
    "sec06_three_torus",
    "sec07_fractalization",
)


def _assert_npz_matches(generated_path: Path, committed_path: Path):
    """Compare a regenerated cache against the committed one.

    Elementwise equality is asserted only on the architecture that produced the
    caches. These sections integrate chaotic maps, so on another architecture a
    last-bit difference is amplified into an O(1) one and elementwise agreement
    is unattainable by construction (see conftest.REFERENCE_PLATFORM_ENV). There the
    contract itself is still checked -- see assert_npz_structurally_sound for
    what survives the architecture change and what only looks like it does.
    """
    with safe_load(generated_path) as generated, safe_load(committed_path) as committed:
        if not is_reference_platform():
            assert_npz_structurally_sound(generated, committed)
            return
        assert set(generated.files) == set(committed.files)
        for key in generated.files:
            actual = generated[key]
            expected = committed[key]
            assert actual.shape == expected.shape, key
            assert actual.dtype == expected.dtype, key
            if np.issubdtype(actual.dtype, np.floating):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_array_equal(actual, expected)


def _prepare_section_dependencies(section_id: str, tmp_path: Path):
    if section_id == "sec11_diagnostics":
        dependency = Path("sec06_three_torus") / "lyapunov_vs_DB.npz"
        target = tmp_path / dependency
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(COMMITTED_FIGURES / dependency, target)


def _assert_section_reproduces(section_id: str, tmp_path: Path):
    _prepare_section_dependencies(section_id, tmp_path)
    run_section(section_id, output_root=tmp_path, profile="paper", recompute=True)
    spec = get_section(section_id)
    for cache_file in spec.cache_files:
        _assert_npz_matches(
            tmp_path / section_id / cache_file,
            COMMITTED_FIGURES / section_id / cache_file,
        )


@pytest.mark.parametrize("section_id", FAST_REPRODUCIBILITY_SECTIONS)
def test_fast_figure_caches_reproduce_committed(section_id, tmp_path):
    _assert_section_reproduces(section_id, tmp_path)


@pytest.mark.skipif(
    not os.environ.get("DYNACHAOS_SLOW"),
    reason="heavy section recompute (minutes each); set DYNACHAOS_SLOW=1 to run",
)
@pytest.mark.parametrize("section_id", HEAVY_REPRODUCIBILITY_SECTIONS)
def test_heavy_figure_caches_reproduce_committed(section_id, tmp_path):
    _assert_section_reproduces(section_id, tmp_path)
