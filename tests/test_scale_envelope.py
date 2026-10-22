import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "benchmarks" / "scale_envelope.jsonc"
SCRIPT = ROOT / "benchmarks" / "scale_envelope.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("scale_envelope", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


# Peak RSS is unavailable on Windows: there is no resource module and no
# /proc, so the benchmark reports 0 rather than failing. Requiring a positive
# value there asserts the availability of a facility the platform lacks.
PEAK_RSS_AVAILABLE = sys.platform != "win32"


def test_strip_jsonc_preserves_comment_markers_inside_strings():
    module = _load_script_module()
    text = r"""
    {
      "uri": "file://example/path", // line comment
      "pattern": "not /* a comment */ here",
      /* block comment */
      "escaped": "quote: \" // still string"
    }
    """

    data = json.loads(module._strip_jsonc(text))

    assert data["uri"] == "file://example/path"
    assert data["pattern"] == "not /* a comment */ here"
    assert data["escaped"] == 'quote: " // still string'


def test_scale_envelope_ci_end_to_end(tmp_path):
    module = _load_script_module()
    cfg = module._load_config(CONFIG)
    json_out = tmp_path / "scale_envelope.json"
    md_out = tmp_path / "scale_envelope.md"
    cfg["output"] = {"json": json_out.as_posix(), "markdown": md_out.as_posix()}
    cfg["rqa"]["stop_predicted_bytes"] = 3 * 8 * 150 * 150
    cfg_path = tmp_path / "scale_envelope.jsonc"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(SCRIPT), str(cfg_path)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert json_out.exists()
    assert md_out.exists()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["mode"] == "ci"
    for key in ["machine", "signals", "gp", "rqa"]:
        assert key in data
    for key in ["cpu_model", "python", "numpy"]:
        assert key in data["machine"]
    # Host total memory is not recorded: this report is committed publicly and
    # each case already reports its own peak RSS.
    assert "ram_bytes" not in data["machine"]

    gp_rows = data["gp"]
    assert gp_rows
    required_gp = {
        "N",
        "embedding_dim",
        "delay",
        "radius_count",
        "backend",
        "wall_time_s_p50",
        "peak_rss_bytes",
    }
    measured_gp = [row for row in gp_rows if not row.get("skipped")]
    assert measured_gp
    for row in measured_gp:
        assert required_gp <= row.keys()
        assert row["backend"] in {"rust", "python"}
        assert row["wall_time_s_p50"] >= 0.0
        assert row["peak_rss_bytes"] > 0 if PEAK_RSS_AVAILABLE else row["peak_rss_bytes"] >= 0

    # Cross-backend parity is only defined when both backends actually ran.
    # Under DYNACHAOS_NO_RUST=1 there is no second backend to compare against,
    # so the benchmark emits no parity block — absence is correct there, and
    # asserting it unconditionally is what made the pure-Python CI job fail.
    backends_run = {row["backend"] for row in measured_gp}
    if backends_run == {"rust", "python"}:
        for row in measured_gp:
            assert "parity_vs_other_backend" in row, (
                f"both backends ran, so row N={row['N']} must carry parity"
            )
            parity = row["parity_vs_other_backend"]
            assert parity["max_abs_delta_logC"] <= 1e-12
            assert parity["max_abs_delta_slope"] <= 1e-12
    else:
        assert not any("parity_vs_other_backend" in row for row in measured_gp), (
            f"only {backends_run} ran, so no row should claim cross-backend parity"
        )

    rqa_rows = data["rqa"]
    assert rqa_rows
    skipped_rqa = [row for row in rqa_rows if row.get("skipped")]
    assert skipped_rqa
    assert skipped_rqa[0]["skipped"] is True
    measured_rqa = [row for row in rqa_rows if not row.get("skipped")]
    assert measured_rqa
    for row in measured_rqa:
        for key in [
            "N",
            "wall_time_s_p50",
            "peak_rss_bytes",
            "predicted_dense_distance_bytes",
            "predicted_peak_with_temporaries_bytes",
            "dense_rqa_temporaries_multiplier",
            "configured_impractical_threshold_N",
            "rqa",
        ]:
            assert key in row
        assert row["predicted_dense_distance_bytes"] == 8 * row["N"] * row["N"]
        assert (
            row["predicted_peak_with_temporaries_bytes"]
            == 3 * row["predicted_dense_distance_bytes"]
        )
        assert row["wall_time_s_p50"] >= 0.0
        assert row["peak_rss_bytes"] > 0 if PEAK_RSS_AVAILABLE else row["peak_rss_bytes"] >= 0
        for key in ["RR", "DET", "LAM", "L", "TT", "ENTR", "Lmax"]:
            assert key in row["rqa"]

    md = md_out.read_text(encoding="utf-8")
    assert "Hardware and software" in md
    assert "Caveats" in md
    assert "3x temporary-array multiplier" in md
    assert "Dense recurrence/RQA configured impracticality threshold" in md
