import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dynachaos.workflow import WorkflowError, _json_safe, _require_mapping, run_workflow

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


def _repo_src() -> Path:
    return ROOT / "src"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_src())
    return env


def _write_config(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _minimal_config(input_name: str, diagnostic: str) -> str:
    return (
        f'{{"input": {{"path": "{input_name}"}}, "output": {{"dir": "out"}}, '
        f'"diagnostics": [{{"name": "{diagnostic}"}}]}}'
    )


def test_workflow_fixture_writes_stable_outputs(tmp_path):
    shutil.copy(DATA / "workflow_signal.npy", tmp_path / "workflow_signal.npy")
    cfg = tmp_path / "workflow_fixture.jsonc"
    cfg.write_text((DATA / "workflow_fixture.jsonc").read_text(encoding="utf-8"), encoding="utf-8")

    paths = run_workflow(cfg)

    assert paths["output_dir"] == tmp_path / "workflow_output"
    assert paths["results"].exists()
    assert paths["metadata"].exists()
    assert paths["summary"].exists()
    results = json.loads(paths["results"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert set(results["diagnostics"]) == {"permutation_entropy", "rqa_streaming"}
    assert "value" in results["diagnostics"]["permutation_entropy"]
    assert "RR" in results["diagnostics"]["rqa_streaming"]["stats"]
    assert metadata["scale_cost"]["signal_length_N"] == 64
    assert "reliability" in metadata
    assert "rqa_streaming" in metadata["reliability"]
    assert "Results JSON: `results.json`" in paths["summary"].read_text(encoding="utf-8")


def test_workflow_generated_benchmark_signal(tmp_path):
    cfg = _write_config(
        tmp_path / "generated.jsonc",
        """
        {
          "input": {"generated": {"name": "logistic", "n": 32, "seed": 7, "a": 1.9}},
          "output": {"dir": "out"},
          "diagnostics": [{"name": "permutation_entropy", "d": 3}]
        }
        """,
    )

    paths = run_workflow(cfg)

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["input"]["kind"] == "generated"
    assert metadata["scale_cost"]["signal_length_N"] == 32


def test_workflow_missing_input_is_explicit(tmp_path):
    cfg = _write_config(
        tmp_path / "missing.jsonc",
        _minimal_config("missing.npy", "permutation_entropy"),
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "input file does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing input did not fail")


def test_workflow_invalid_shape_is_explicit(tmp_path):
    shutil.copy(DATA / "workflow_bad_shape.npy", tmp_path / "bad.npy")
    cfg = _write_config(
        tmp_path / "bad_shape.jsonc",
        _minimal_config("bad.npy", "permutation_entropy"),
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "input signal must be 1D" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid shape did not fail")


def test_workflow_unsupported_diagnostic_is_explicit(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "unsupported.jsonc",
        _minimal_config("x.npy", "not_a_diagnostic"),
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "unsupported diagnostic name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported diagnostic did not fail")


def test_workflow_dense_rqa_scale_envelope_is_explicit(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "too_dense.jsonc",
        """
        {
          "input": {"path": "x.npy"},
          "output": {"dir": "out"},
          "scale_limits": {"dense_rqa_max_bytes": 128},
          "diagnostics": [{"name": "rqa_dense", "embedding": {"d": 1}, "eps": 0.1}]
        }
        """,
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        msg = str(exc)
        assert "8*N^2" in msg
        assert "rqa_streaming" in msg
        assert "allow_dense_rqa_beyond_envelope" in msg
    else:  # pragma: no cover
        raise AssertionError("oversized dense RQA did not fail")


def test_workflow_rejects_non_boolean_envelope_override(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "string_override.jsonc",
        """
        {
          "input": {"path": "x.npy"},
          "output": {"dir": "out"},
          "scale_limits": {"dense_rqa_max_bytes": 128,
                           "allow_dense_rqa_beyond_envelope": "false"},
          "diagnostics": [{"name": "rqa_dense", "embedding": {"d": 1}, "eps": 0.1}]
        }
        """,
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "must be the JSON boolean" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("string envelope override did not fail")


def test_workflow_rejects_both_path_and_generated(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "dual_input.jsonc",
        """
        {
          "input": {"path": "x.npy",
                    "generated": {"name": "logistic", "n": 32}},
          "output": {"dir": "out"},
          "diagnostics": [{"name": "permutation_entropy"}]
        }
        """,
    )

    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "not both" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("dual input source did not fail")


def test_get_figure_theme_defaults_and_honors_env_override(monkeypatch):
    from dynachaos.config import DEFAULT_FIGURE_THEME, get_figure_theme

    monkeypatch.delenv("DYNACHAOS_THEME", raising=False)
    assert get_figure_theme() == DEFAULT_FIGURE_THEME

    monkeypatch.setenv("DYNACHAOS_THEME", "custom_theme")
    assert get_figure_theme() == "custom_theme"


def test_strip_jsonc_handles_escaped_backslash_and_quote_inside_strings():
    from dynachaos.config import strip_jsonc

    # A literal backslash followed by a quote inside a JSON string must not
    # be mistaken for an escaped closing quote (the escaped=True/False
    # bookkeeping in strip_jsonc's string-scanning loop).
    text = r'{"a": "back\\slash", "b": "quote\"inside"}'
    stripped = strip_jsonc(text)
    parsed = json.loads(stripped)
    assert parsed == {"a": "back\\slash", "b": 'quote"inside'}


def test_strip_jsonc_removes_block_comments():
    from dynachaos.config import strip_jsonc

    text = '{\n  "a": 1, /* block\n comment spanning lines */ "b": 2\n}'
    parsed = json.loads(strip_jsonc(text))
    assert parsed == {"a": 1, "b": 2}


def test_load_jsonc_rejects_non_object_top_level(tmp_path):
    from dynachaos.config import load_jsonc

    path = tmp_path / "array.jsonc"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        load_jsonc(path)


def test_json_safe_converts_numpy_scalars_dicts_and_nan_floats():
    payload = {
        "arr": np.array([1.0, 2.0]),
        "scalar": np.float64(3.5),
        "nested": {"nan": float("nan"), "list": (1, 2.0)},
    }
    safe = _json_safe(payload)
    assert safe["arr"] == [1.0, 2.0]
    assert safe["scalar"] == 3.5
    assert safe["nested"]["nan"] is None
    assert safe["nested"]["list"] == [1, 2.0]


def test_require_mapping_rejects_non_dict_value():
    with pytest.raises(WorkflowError, match="must be an object"):
        _require_mapping([1, 2, 3], "diagnostics[]")


def test_workflow_unreadable_input_file_reports_os_error(tmp_path):
    # A directory in place of the .npy file makes np.load raise OSError
    # (IsADirectoryError), exercising the "could not read" branch distinct
    # from the "does not exist" branch.
    (tmp_path / "x.npy").mkdir()
    cfg = _write_config(
        tmp_path / "unreadable.jsonc",
        _minimal_config("x.npy", "permutation_entropy"),
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "could not read input file" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("directory input did not fail")


def test_workflow_npz_multi_array_requires_explicit_key(tmp_path):
    np.savez(tmp_path / "multi.npz", a=np.linspace(0.0, 1.0, 16), b=np.linspace(1.0, 2.0, 16))
    cfg = _write_config(
        tmp_path / "multi.jsonc",
        _minimal_config("multi.npz", "permutation_entropy"),
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "npz_key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ambiguous npz input did not fail")


def test_workflow_npz_unknown_key_is_explicit(tmp_path):
    np.savez(tmp_path / "multi.npz", a=np.linspace(0.0, 1.0, 16), b=np.linspace(1.0, 2.0, 16))
    cfg = _write_config(
        tmp_path / "badkey.jsonc",
        '{"input": {"path": "multi.npz", "npz_key": "nope"}, "output": {"dir": "out"}, '
        '"diagnostics": [{"name": "permutation_entropy"}]}',
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "does not contain array 'nope'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown npz key did not fail")


def test_workflow_npz_single_array_auto_selects_sole_key(tmp_path):
    np.savez(tmp_path / "single.npz", only=np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "single.jsonc",
        _minimal_config("single.npz", "permutation_entropy"),
    )
    paths = run_workflow(cfg)
    assert paths["results"].exists()


def test_workflow_npz_with_explicit_key_succeeds(tmp_path):
    np.savez(tmp_path / "multi.npz", a=np.linspace(0.0, 1.0, 32), b=np.linspace(1.0, 2.0, 32))
    cfg = _write_config(
        tmp_path / "goodkey.jsonc",
        '{"input": {"path": "multi.npz", "npz_key": "a"}, "output": {"dir": "out"}, '
        '"diagnostics": [{"name": "permutation_entropy"}]}',
    )
    paths = run_workflow(cfg)
    assert paths["results"].exists()


def test_workflow_generated_unsupported_name_is_explicit(tmp_path):
    cfg = _write_config(
        tmp_path / "bad_generated.jsonc",
        '{"input": {"generated": {"name": "not_logistic", "n": 16}}, "output": {"dir": "out"}, '
        '"diagnostics": [{"name": "permutation_entropy"}]}',
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "unsupported generated benchmark signal" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported generated signal did not fail")


def test_workflow_generated_rejects_nonpositive_n(tmp_path):
    cfg = _write_config(
        tmp_path / "zero_n.jsonc",
        '{"input": {"generated": {"name": "logistic", "n": 0}}, "output": {"dir": "out"}, '
        '"diagnostics": [{"name": "permutation_entropy"}]}',
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "generated.n must be a positive integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("n=0 did not fail")


def test_workflow_generated_signal_honors_explicit_x0(tmp_path):
    cfg = _write_config(
        tmp_path / "explicit_x0.jsonc",
        '{"input": {"generated": {"name": "logistic", "n": 16, "a": 1.9, "x0": 0.3}}, '
        '"output": {"dir": "out"}, "diagnostics": [{"name": "permutation_entropy"}]}',
    )
    paths = run_workflow(cfg)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["input"]["x0"] == 0.3


def test_workflow_input_missing_path_and_generated_is_explicit(tmp_path):
    cfg = _write_config(
        tmp_path / "no_input.jsonc",
        '{"input": {}, "output": {"dir": "out"}, "diagnostics": [{"name": "permutation_entropy"}]}',
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "input must set either path or generated" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty input did not fail")


def test_workflow_rejects_empty_signal(tmp_path):
    np.save(tmp_path / "empty.npy", np.array([], dtype=np.float64))
    cfg = _write_config(
        tmp_path / "empty.jsonc", _minimal_config("empty.npy", "permutation_entropy")
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "input signal must be non-empty" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty signal did not fail")


def test_workflow_rejects_nonfinite_signal(tmp_path):
    np.save(tmp_path / "nanit.npy", np.array([1.0, np.nan, 2.0]))
    cfg = _write_config(
        tmp_path / "nanit.jsonc", _minimal_config("nanit.npy", "permutation_entropy")
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "input signal must contain only finite values" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-finite signal did not fail")


def test_workflow_diagnostic_entry_missing_name_is_explicit(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "noname.jsonc",
        '{"input": {"path": "x.npy"}, "output": {"dir": "out"}, "diagnostics": [{}]}',
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "diagnostic entry is missing name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("nameless diagnostic did not fail")


def test_workflow_correlation_dimension_diagnostic_with_stderr(tmp_path):
    t = np.linspace(0.0, 2.0 * np.pi, 600, endpoint=False)
    np.save(tmp_path / "circle.npy", np.sin(t) + 1e-6 * np.cos(3 * t))
    cfg = _write_config(
        tmp_path / "corrdim.jsonc",
        '{"input": {"path": "circle.npy"}, "output": {"dir": "out"}, '
        '"diagnostics": [{"name": "correlation_dimension", "embedding": {"d": 2, "tau": 5}, '
        '"n_r": 20}]}',
    )
    paths = run_workflow(cfg)
    results = json.loads(paths["results"].read_text(encoding="utf-8"))
    out = results["diagnostics"]["correlation_dimension"]
    assert "D2" in out and "D2_stderr" in out
    assert "r_values" in out and "local_slopes" in out


def test_workflow_correlation_dimension_diagnostic_without_stderr(tmp_path):
    t = np.linspace(0.0, 2.0 * np.pi, 600, endpoint=False)
    np.save(tmp_path / "circle2.npy", np.sin(t) + 1e-6 * np.cos(3 * t))
    cfg = _write_config(
        tmp_path / "corrdim_nostd.jsonc",
        '{"input": {"path": "circle2.npy"}, "output": {"dir": "out"}, '
        '"diagnostics": [{"name": "correlation_dimension", "embedding": {"d": 2, "tau": 5}, '
        '"n_r": 20, "return_stderr": false}]}',
    )
    paths = run_workflow(cfg)
    results = json.loads(paths["results"].read_text(encoding="utf-8"))
    out = results["diagnostics"]["correlation_dimension"]
    assert "D2" in out
    assert "D2_stderr" not in out


def test_workflow_dense_rqa_succeeds_within_scale_envelope(tmp_path):
    np.save(tmp_path / "small.npy", np.sin(np.linspace(0.0, 20.0, 64)))
    cfg = _write_config(
        tmp_path / "dense_ok.jsonc",
        '{"input": {"path": "small.npy"}, "output": {"dir": "out"}, '
        '"diagnostics": [{"name": "rqa_dense", "embedding": {"d": 1}, "eps": 0.5}]}',
    )
    paths = run_workflow(cfg)
    results = json.loads(paths["results"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert "RR" in results["diagnostics"]["rqa_dense"]["stats"]
    assert metadata["reliability"]["rqa_dense"]["backend"] == "dense recurrence_matrix + rqa"


def test_run_workflow_rejects_missing_config_file(tmp_path):
    missing_cfg = tmp_path / "does_not_exist.jsonc"
    try:
        run_workflow(missing_cfg)
    except WorkflowError as exc:
        assert "config file does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing config file did not fail")


def test_run_workflow_requires_output_dir(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "no_output.jsonc",
        '{"input": {"path": "x.npy"}, "diagnostics": [{"name": "permutation_entropy"}]}',
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "output.dir" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing output.dir did not fail")


def test_run_workflow_rejects_empty_diagnostics_list(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "empty_diag.jsonc",
        '{"input": {"path": "x.npy"}, "output": {"dir": "out"}, "diagnostics": []}',
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "diagnostics must be a non-empty list" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty diagnostics list did not fail")


def test_run_workflow_rejects_duplicate_diagnostic_names(tmp_path):
    np.save(tmp_path / "x.npy", np.linspace(0.0, 1.0, 16))
    cfg = _write_config(
        tmp_path / "dup_diag.jsonc",
        '{"input": {"path": "x.npy"}, "output": {"dir": "out"}, '
        '"diagnostics": [{"name": "permutation_entropy"}, {"name": "permutation_entropy"}]}',
    )
    try:
        run_workflow(cfg)
    except WorkflowError as exc:
        assert "duplicate diagnostic name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate diagnostic name did not fail")


def test_cli_analyze_reports_user_error_without_traceback(tmp_path):
    cfg = _write_config(
        tmp_path / "missing.jsonc",
        _minimal_config("missing.npy", "permutation_entropy"),
    )

    proc = subprocess.run(
        [sys.executable, "-m", "dynachaos.cli", "analyze", str(cfg)],
        check=False,
        capture_output=True,
        text=True,
        env=_cli_env(),
        timeout=30,
    )

    assert proc.returncode == 2
    assert "input file does not exist" in proc.stderr
    assert "Traceback" not in proc.stderr
