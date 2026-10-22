import re
from pathlib import Path

from dynachaos.pipelines.registry import SECTION_SPECS, list_sections

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_ROOT = REPO_ROOT / "figures"
PAPER_TEX = REPO_ROOT / "paper" / "main.tex"
# sec04_doubling/map_IV_attractors.png is an alternative attractor-gallery view that
# the manuscript intentionally does not cite. The five sec12_intermittency figures are
# now cited in main.tex (STI spine + the temporal-intermittency spotlight), so they are
# no longer orphans.
ORPHAN_IMAGE_ALLOWLIST = {
    "sec04_doubling/map_IV_attractors.png",
}


def _registry_outputs_by_relative_path():
    outputs = {}
    for section_id in list_sections():
        spec = SECTION_SPECS[section_id]
        for output_file in spec.output_files:
            outputs[f"{section_id}/{output_file}"] = FIGURES_ROOT / section_id / output_file
    return outputs


def _test_sources_text():
    texts = []
    for path in sorted((REPO_ROOT / "tests").glob("test*.py")):
        if path.name == Path(__file__).name:
            continue
        texts.append(path.read_text(encoding="utf-8"))
    return "\n".join(texts)


def test_registry_outputs_exist_and_npz_outputs_have_contracts():
    missing_outputs = []
    missing_contracts = []
    empty_contracts = []

    for section_id in list_sections():
        spec = SECTION_SPECS[section_id]
        contracts = {contract.file_name: contract.required_keys for contract in spec.npz_contracts}
        for output_file in spec.output_files:
            if (FIGURES_ROOT / section_id / output_file).suffix not in {".npz", ".png", ".gif"}:
                continue
            if not (FIGURES_ROOT / section_id / output_file).exists():
                missing_outputs.append(f"{section_id}/{output_file}")
            if output_file.endswith(".npz"):
                required_keys = contracts.get(output_file)
                if required_keys is None:
                    missing_contracts.append(f"{section_id}/{output_file}")
                elif not required_keys:
                    empty_contracts.append(f"{section_id}/{output_file}")

    assert not missing_outputs, f"Missing registry outputs on disk: {missing_outputs}"
    assert not missing_contracts, f"NPZ outputs without NpzContract: {missing_contracts}"
    assert not empty_contracts, f"NPZ outputs with empty NpzContract: {empty_contracts}"


def test_each_registry_section_has_cache_test_coverage():
    test_text = _test_sources_text()
    uncovered_sections = []

    for section_id in list_sections():
        spec = SECTION_SPECS[section_id]
        covered = [cache_file for cache_file in spec.cache_files if cache_file in test_text]
        if not covered:
            uncovered_sections.append(section_id)

    assert not uncovered_sections, f"Sections with no cache filename in tests: {uncovered_sections}"


def test_manuscript_figure_references_match_registry_and_disk_when_present():
    if not PAPER_TEX.exists():
        return

    registry_outputs = _registry_outputs_by_relative_path()
    registry_images = {
        rel_path for rel_path in registry_outputs if Path(rel_path).suffix in {".png", ".gif"}
    }
    tex = PAPER_TEX.read_text(encoding="utf-8")
    tex_refs = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex))

    refs_missing_from_registry = tex_refs - registry_images
    refs_missing_on_disk = [
        rel_path for rel_path in sorted(tex_refs) if not (FIGURES_ROOT / rel_path).exists()
    ]
    registry_images_not_in_tex = registry_images - tex_refs

    assert not refs_missing_from_registry, (
        f"Manuscript figure refs missing from registry: {sorted(refs_missing_from_registry)}"
    )
    assert not refs_missing_on_disk, (
        f"Manuscript figure refs missing on disk: {refs_missing_on_disk}"
    )
    assert registry_images_not_in_tex == ORPHAN_IMAGE_ALLOWLIST
