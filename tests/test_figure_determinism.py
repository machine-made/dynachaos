from pathlib import Path

import numpy as np
from conftest import assert_npz_structurally_sound, is_reference_platform

from dynachaos.io.paths import safe_load
from dynachaos.pipelines.registry import get_section
from dynachaos.pipelines.runner import run_section

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_FIGURES = REPO_ROOT / "figures"
DETERMINISM_SECTIONS = ("sec12_intermittency",)


def _assert_npz_matches(generated_path: Path, committed_path: Path):
    with safe_load(generated_path) as generated, safe_load(committed_path) as committed:
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


def _assert_npz_structurally_sound(generated_path: Path, committed_path: Path):
    """Check what remains true off the reference platform: same contract, sane values."""
    with safe_load(generated_path) as generated, safe_load(committed_path) as committed:
        assert_npz_structurally_sound(generated, committed)


def test_representative_figure_caches_recompute_deterministically(tmp_path):
    """Recompute chaotic sections and compare against the committed caches.

    Elementwise equality only holds on the architecture that produced the
    caches (see conftest.REFERENCE_PLATFORM_ENV). Everywhere else the pipeline is
    still run and its output contract still verified.
    """
    check = _assert_npz_matches if is_reference_platform() else _assert_npz_structurally_sound
    for section_id in DETERMINISM_SECTIONS:
        run_section(section_id, output_root=tmp_path, profile="paper", recompute=True)
        spec = get_section(section_id)
        for cache_file in spec.cache_files:
            check(
                tmp_path / section_id / cache_file,
                COMMITTED_FIGURES / section_id / cache_file,
            )
