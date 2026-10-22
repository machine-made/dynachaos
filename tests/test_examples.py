import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
RECIPES = EXAMPLES / "recipes"


RELIABILITY_FIELDS = {
    "method_name",
    "backend",
    "parameters",
    "data_length",
    "data_shape",
    "sampling_downsampling_note",
    "validity_warnings",
    "unresolved_verdicts",
    "scale_evidence",
    "schema_version",
}


def _repo_src() -> Path:
    return ROOT / "src"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    return env


def _copy_recipe(tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(RECIPES / name, destination)
    return destination


def _run_recipe(recipe_dir: Path, config_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dynachaos.cli", "analyze", config_name],
        cwd=recipe_dir,
        check=False,
        capture_output=True,
        text=True,
        env=_cli_env(),
        timeout=30,
    )


def _load_outputs(recipe_dir: Path, output_dir: str) -> tuple[dict, dict, str]:
    output_root = recipe_dir / output_dir
    results_path = output_root / "results.json"
    metadata_path = output_root / "metadata.json"
    summary_path = output_root / "summary.md"
    assert results_path.exists()
    assert metadata_path.exists()
    assert summary_path.exists()
    return (
        json.loads(results_path.read_text(encoding="utf-8")),
        json.loads(metadata_path.read_text(encoding="utf-8")),
        summary_path.read_text(encoding="utf-8"),
    )


def test_external_signal_recipe_smoke_outputs_and_reliability(tmp_path):
    recipe_dir = _copy_recipe(tmp_path, "external_signal")

    proc = _run_recipe(recipe_dir, "external_signal_recipe.jsonc")

    assert proc.returncode == 0, proc.stderr
    assert "results\t" in proc.stdout
    results, metadata, summary = _load_outputs(recipe_dir, "outputs/external_signal_recipe")
    assert set(results["diagnostics"]) == {
        "permutation_entropy",
        "correlation_dimension",
        "rqa_streaming",
    }
    assert metadata["input"]["kind"] == "file"
    assert metadata["input"]["path"].endswith("external_signal_fixture.npy")
    assert set(metadata["reliability"]) == set(results["diagnostics"])
    for record in metadata["reliability"].values():
        assert set(record) == RELIABILITY_FIELDS
        assert record["data_length"] > 0
        assert record["schema_version"] == "1.0"
    assert "Results JSON: `results.json`" in summary
    assert "Metadata JSON: `metadata.json`" in summary


def test_long_signal_streaming_recipe_smoke_stays_in_streaming_path(tmp_path):
    recipe_dir = _copy_recipe(tmp_path, "long_signal_streaming")

    proc = _run_recipe(recipe_dir, "long_signal_streaming_recipe.jsonc")

    assert proc.returncode == 0, proc.stderr
    results, metadata, summary = _load_outputs(recipe_dir, "outputs/long_signal_streaming_recipe")
    assert set(results["diagnostics"]) == {"permutation_entropy", "rqa_streaming"}
    assert "rqa_dense" not in results["diagnostics"]
    assert metadata["scale_cost"]["signal_length_N"] == 160
    rqa_record = metadata["reliability"]["rqa_streaming"]
    assert rqa_record["method_name"] == "rqa_from_trajectory"
    assert rqa_record["data_length"] == 159
    assert "without dense recurrence matrix" in rqa_record["sampling_downsampling_note"].lower()
    assert "Results JSON: `results.json`" in summary


def test_examples_readme_documents_recipe_commands_and_legacy_status():
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    external_doc = (RECIPES / "external_signal" / "README.md").read_text(encoding="utf-8")
    long_doc = (RECIPES / "long_signal_streaming" / "README.md").read_text(encoding="utf-8")

    assert "dynachaos analyze external_signal_recipe.jsonc" in readme
    assert "dynachaos analyze long_signal_streaming_recipe.jsonc" in readme
    assert "Legacy/internal benchmark scripts" in readme
    assert "metadata.json" in external_doc
    assert "reliability" in external_doc
    assert "rqa_streaming" in long_doc
    assert "dense-recurrence memory cap" in long_doc
