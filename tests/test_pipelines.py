import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from dynachaos.pipelines import runner
from dynachaos.pipelines.registry import get_figure, get_section, list_sections
from dynachaos.pipelines.runner import run_section, validate_section_cache, validate_section_outputs


def test_section_registry_contains_all_expected_sections():
    sections = list_sections()
    assert sections == (
        "sec02_circle_map",
        "sec03_transition",
        "sec04_doubling",
        "sec05_oscillation",
        "sec06_three_torus",
        "sec07_fractalization",
        "sec08_sti",
        "sec09_pattern",
        "sec10_gcm",
        "sec11_diagnostics",
        "sec12_intermittency",
    )


def test_registry_covers_all_includegraphics_targets():
    paper_tex_path = Path("paper/main.tex")
    if not paper_tex_path.exists():
        pytest.skip("paper sources are not part of the package repository")
    paper_tex = paper_tex_path.read_text(encoding="utf-8")
    refs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", paper_tex)
    include_targets = set(refs)

    declared_pngs = set()
    for section_id in list_sections():
        spec = get_section(section_id)
        for rel in spec.output_files:
            if rel.endswith(".png"):
                declared_pngs.add(f"{section_id}/{rel}")

    missing = include_targets - declared_pngs
    assert not missing, f"Missing includegraphics coverage for: {sorted(missing)}"


def test_section_spec_cache_and_output_paths_resolve_under_section_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNACHAOS_OUTPUT_ROOT", str(tmp_path))
    spec = get_section("sec02_circle_map")

    cache_paths = spec.cache_paths()
    output_paths = spec.output_paths()

    assert cache_paths == tuple(tmp_path / "sec02_circle_map" / name for name in spec.cache_files)
    assert output_paths == tuple(tmp_path / "sec02_circle_map" / name for name in spec.output_files)


def test_get_figure_returns_none_for_unregistered_png():
    assert get_figure("sec02_circle_map", "not_a_registered_figure.png") is None


def test_get_figure_returns_matching_spec_for_registered_png():
    spec = get_section("sec02_circle_map")
    assert spec.figures, "sec02_circle_map must register at least one figure"
    known_png = spec.figures[0].png

    figure = get_figure("sec02_circle_map", known_png)

    assert figure is not None
    assert figure.png == known_png


def test_smoke_profile_requires_precomputed_cache(tmp_path):
    with pytest.raises(RuntimeError, match="Section sec02_circle_map is missing expected artifact"):
        run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")


def test_smoke_profile_rejects_malformed_cache_before_running_module(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    (section_dir / "devils_staircase.npz").write_text("not an npz", encoding="utf-8")

    def fail_run(*args, **kwargs):
        raise AssertionError("smoke profile should validate caches before running modules")

    monkeypatch.setattr(runner, "_run_module", fail_run)

    with pytest.raises(RuntimeError, match="Section sec02_circle_map has malformed NPZ artifact"):
        run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")


def test_smoke_profile_rejects_missing_required_cache_keys_before_running_module(
    tmp_path, monkeypatch
):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0])

    def fail_run(*args, **kwargs):
        raise AssertionError("smoke profile should validate caches before running modules")

    monkeypatch.setattr(runner, "_run_module", fail_run)

    with pytest.raises(RuntimeError, match="missing required NPZ keys: lam"):
        run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")


def test_smoke_profile_accepts_valid_cached_outputs_without_recomputation(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    calls = []

    def record_run(module_name, output_root, profile):
        calls.append((module_name, output_root, profile))

    monkeypatch.setattr(runner, "_run_module", record_run)

    outputs = run_section("sec02_circle_map", output_root=tmp_path, profile="smoke")

    assert [path.name for path in outputs] == list(get_section("sec02_circle_map").output_files)
    assert calls == [
        ("dynachaos.maps.circle_map", tmp_path.resolve(), "smoke"),
        ("dynachaos.maps.arnold_tongues", tmp_path.resolve(), "smoke"),
    ]


def test_run_section_writes_opt_in_timing_ledger(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    clock = iter([10.0, 10.5, 20.0, 21.25])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(runner, "get_rss_mb", lambda: 123.4567)
    monkeypatch.setattr(runner, "_run_module", lambda module_name, output_root, profile: None)

    ledger_path = tmp_path / "perf" / "sections.jsonl"
    run_section(
        "sec02_circle_map",
        output_root=tmp_path,
        profile="smoke",
        timing_ledger=ledger_path,
    )

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert events == [
        {
            "cache_state": "validated",
            "module": "dynachaos.maps.circle_map",
            "peak_rss_mb": 123.457,
            "profile": "smoke",
            "section_id": "sec02_circle_map",
            "wall_time_s": 0.5,
        },
        {
            "cache_state": "validated",
            "module": "dynachaos.maps.arnold_tongues",
            "peak_rss_mb": 123.457,
            "profile": "smoke",
            "section_id": "sec02_circle_map",
            "wall_time_s": 1.25,
        },
    ]


def test_run_section_timing_ledger_can_come_from_env(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    clock = iter([1.0, 1.25, 2.0, 2.75])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(runner, "get_rss_mb", lambda: 50.0)
    monkeypatch.setattr(runner, "_run_module", lambda module_name, output_root, profile: None)

    ledger_path = tmp_path / "timing.jsonl"
    monkeypatch.setenv("DYNACHAOS_TIMING_LEDGER", str(ledger_path))
    run_section("sec02_circle_map", output_root=tmp_path, profile="paper")

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [event["cache_state"] for event in events] == ["not_checked", "not_checked"]
    assert [event["wall_time_s"] for event in events] == [0.25, 0.75]


def test_run_section_without_timing_ledger_does_not_touch_timer(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    def fail_timer():
        raise AssertionError("default runs should not collect timing")

    monkeypatch.delenv("DYNACHAOS_TIMING_LEDGER", raising=False)
    monkeypatch.setattr(runner.time, "perf_counter", fail_timer)
    monkeypatch.setattr(runner, "_run_module", lambda module_name, output_root, profile: None)

    run_section("sec02_circle_map", output_root=tmp_path, profile="paper")


def test_section_validators_can_check_cache_and_outputs_without_running_modules(tmp_path):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    cache_paths = validate_section_cache("sec02_circle_map", output_root=tmp_path)
    output_paths = validate_section_outputs("sec02_circle_map", output_root=tmp_path)

    assert [path.name for path in cache_paths] == list(get_section("sec02_circle_map").cache_files)
    assert [path.name for path in output_paths] == list(
        get_section("sec02_circle_map").output_files
    )


def test_section_cache_validator_rejects_malformed_npz(tmp_path):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    (section_dir / "devils_staircase.npz").write_bytes(b"not a valid npz archive")
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])

    with pytest.raises(RuntimeError, match="malformed NPZ artifact"):
        validate_section_cache("sec02_circle_map", output_root=tmp_path)


def test_validate_artifact_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hi", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported extension: .txt"):
        runner._validate_artifact(path, required_keys=(), section_id="sec02")


def test_validate_artifact_raises_assertion_for_unrecognized_status(monkeypatch, tmp_path):
    # _classify_artifact only ever returns known statuses; this exercises the
    # defensive final branch in _validate_artifact by forcing an impossible
    # status through monkeypatching.
    monkeypatch.setattr(runner, "_classify_artifact", lambda path, **kw: ("bogus", "??"))
    with pytest.raises(AssertionError, match="unknown artifact status: bogus"):
        runner._validate_artifact(tmp_path / "x.npz", required_keys=(), section_id="sec02")


def test_inspect_section_artifacts_reports_cache_and_output_roles(tmp_path):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    for png_name in ("devils_staircase.png", "arnold_tongues.png", "staircase_zoom.png"):
        (section_dir / png_name).write_bytes(b"png")

    results = runner.inspect_section_artifacts("sec02_circle_map", output_root=tmp_path)

    roles = {r.role for r in results}
    assert roles == {"cache", "output"}
    assert all(r.status == "ok" for r in results)


def test_section_output_validator_rejects_empty_figure(tmp_path):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    (section_dir / "devils_staircase.png").write_bytes(b"")
    (section_dir / "arnold_tongues.png").write_bytes(b"png")
    (section_dir / "staircase_zoom.png").write_bytes(b"png")

    with pytest.raises(RuntimeError, match="devils_staircase.png.*empty"):
        validate_section_outputs("sec02_circle_map", output_root=tmp_path)


# ---------------------------------------------------------------------------
# T2: schema_version contract for attractors.npz
# ---------------------------------------------------------------------------


def test_smoke_gate_rejects_attractors_missing_schema_version(tmp_path):
    """Regression: attractors.npz without schema_version must fail smoke gate (T2).

    Before the fix, schema_version was absent from the NpzContract, so an old-format
    cache (no schema_version key) would pass validate_section_cache even though
    main() would reject and recompute it.
    """
    section_dir = tmp_path / "sec03_transition"
    section_dir.mkdir()
    np.savez_compressed(
        section_dir / "phase_diagram.npz",
        A=[1.0],
        D=[0.1],
        asym=[0.0],
        lyap=[0.1],
        schema_version=[4],
    )
    # attractors.npz missing the schema_version key (old pre-v4 cache format)
    np.savez_compressed(
        section_dir / "attractors.npz",
        A_values=[1.0],
        labels=["sync"],
        initial_states=[[0.5, 0.5]],
        x_limits=[0.0, 1.0],
        y_limits=[0.0, 1.0],
        D=[0.1],
    )
    np.savez_compressed(
        section_dir / "basins.npz",
        x=[0.5],
        y=[0.5],
        basin=[0],
        A=[1.0],
        D=[0.1],
    )
    with pytest.raises(RuntimeError, match="missing required NPZ keys.*schema_version"):
        validate_section_cache("sec03_transition", output_root=tmp_path)


# ---------------------------------------------------------------------------
# T1: child RSS measurement
# ---------------------------------------------------------------------------


def test_run_module_returns_child_peak_rss_mb(monkeypatch, tmp_path):
    """_run_module returns peak RSS of the child process via os.wait4 (T1 unit test).

    Uses a monkeypatched Popen and wait4 to inject a known rusage and verify
    the KB-to-MB conversion matches get_rss_mb's convention.
    """
    import os as _os

    from dynachaos.pipelines import runner as _runner

    if not hasattr(_os, "wait4"):
        pytest.skip("os.wait4 not available on this platform")

    # ru_maxrss units are platform-dependent: bytes on macOS, KiB on Linux.
    # Build the fake value in the units the running platform actually reports,
    # so this asserts the conversion rather than the developer's platform.
    _MIB = 1024 * 1024
    _RSS_RAW_256_MIB = 256 * _MIB if sys.platform == "darwin" else 256 * 1024

    class FakeRusage:
        ru_maxrss = _RSS_RAW_256_MIB

    class FakePopen:
        pid = 9999
        returncode = None

        def __init__(self, cmd, env=None):
            pass

    def fake_wait4(pid, options):
        return (pid, 0, FakeRusage())  # exit_status=0 → clean exit

    monkeypatch.setattr(_runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(_runner.os, "wait4", fake_wait4)
    if hasattr(_os, "waitstatus_to_exitcode"):
        monkeypatch.setattr(_runner.os, "waitstatus_to_exitcode", lambda x: 0)

    rss = _runner._run_module("dummy.module", tmp_path, "paper")

    assert rss == pytest.approx(256.0, abs=0.01)


def test_run_module_raises_when_wait4_reports_nonzero_exit(monkeypatch, tmp_path):
    import os as _os

    from dynachaos.pipelines import runner as _runner

    if not hasattr(_os, "wait4"):
        pytest.skip("os.wait4 not available on this platform")

    class FakeRusage:
        ru_maxrss = 1024

    class FakePopen:
        pid = 9999
        returncode = None

        def __init__(self, cmd, env=None):
            pass

    monkeypatch.setattr(_runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(_runner.os, "wait4", lambda pid, opts: (pid, 5 << 8, FakeRusage()))
    if hasattr(_os, "waitstatus_to_exitcode"):
        monkeypatch.setattr(_runner.os, "waitstatus_to_exitcode", lambda x: x >> 8)

    with pytest.raises(RuntimeError, match=r"exit 5"):
        _runner._run_module("dummy.module", tmp_path, "paper")


def test_run_module_falls_back_to_wifexited_without_waitstatus_helper(monkeypatch, tmp_path):
    import os as _os

    from dynachaos.pipelines import runner as _runner

    if not hasattr(_os, "wait4"):
        pytest.skip("os.wait4 not available on this platform")

    class FakeRusage:
        ru_maxrss = 1024

    class FakePopen:
        pid = 9999
        returncode = None

        def __init__(self, cmd, env=None):
            pass

    monkeypatch.setattr(_runner.subprocess, "Popen", FakePopen)
    # Clean exit (status 0) with no waitstatus_to_exitcode: falls through to
    # the WIFEXITED/WEXITSTATUS branch.
    monkeypatch.setattr(_runner.os, "wait4", lambda pid, opts: (pid, 0, FakeRusage()))
    monkeypatch.delattr(_runner.os, "waitstatus_to_exitcode", raising=False)

    rss = _runner._run_module("dummy.module", tmp_path, "paper")
    assert rss is not None


def test_run_module_reports_negative_signal_exit_code(monkeypatch, tmp_path):
    import os as _os
    import signal

    from dynachaos.pipelines import runner as _runner

    if not hasattr(_os, "wait4"):
        pytest.skip("os.wait4 not available on this platform")

    class FakeRusage:
        ru_maxrss = 1024

    class FakePopen:
        pid = 9999
        returncode = None

        def __init__(self, cmd, env=None):
            pass

    monkeypatch.setattr(_runner.subprocess, "Popen", FakePopen)
    # A raw signal-terminated status (low byte == signal number, no
    # waitstatus_to_exitcode) exercises the WTERMSIG branch.
    monkeypatch.setattr(
        _runner.os, "wait4", lambda pid, opts: (pid, int(signal.SIGKILL), FakeRusage())
    )
    monkeypatch.delattr(_runner.os, "waitstatus_to_exitcode", raising=False)

    with pytest.raises(RuntimeError, match=r"exit -9"):
        _runner._run_module("dummy.module", tmp_path, "paper")


def test_run_module_without_wait4_uses_simple_wait(monkeypatch, tmp_path):
    from dynachaos.pipelines import runner as _runner

    class FakePopen:
        returncode = 0

        def __init__(self, cmd, env=None):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(_runner.subprocess, "Popen", FakePopen)
    monkeypatch.delattr(_runner.os, "wait4", raising=False)

    rss = _runner._run_module("dummy.module", tmp_path, "paper")
    assert rss is None


def test_run_module_without_wait4_raises_on_nonzero_returncode(monkeypatch, tmp_path):
    from dynachaos.pipelines import runner as _runner

    class FakePopen:
        returncode = 7

        def __init__(self, cmd, env=None):
            pass

        def wait(self):
            return 7

    monkeypatch.setattr(_runner.subprocess, "Popen", FakePopen)
    monkeypatch.delattr(_runner.os, "wait4", raising=False)

    with pytest.raises(RuntimeError, match=r"exit 7"):
        _runner._run_module("dummy.module", tmp_path, "paper")


def test_repo_src_dir_returns_none_when_no_checkout_found(monkeypatch):
    from pathlib import Path as _Path

    from dynachaos.pipelines import runner as _runner

    monkeypatch.setattr(_Path, "is_dir", lambda self: False)
    assert _runner._repo_src_dir() is None


def test_runner_env_appends_to_existing_pythonpath(monkeypatch, tmp_path):
    from dynachaos.pipelines import runner as _runner

    monkeypatch.setenv("PYTHONPATH", "/some/existing/path")
    monkeypatch.setattr(_runner, "_repo_src_dir", lambda: Path("/repo/src"))

    env = _runner._runner_env(tmp_path, "paper")

    assert env["PYTHONPATH"].startswith(f"{Path('/repo/src')}{_runner.os.pathsep}")
    assert "/some/existing/path" in env["PYTHONPATH"]


def test_runner_env_sets_pythonpath_when_none_was_set(monkeypatch, tmp_path):
    from dynachaos.pipelines import runner as _runner

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(_runner, "_repo_src_dir", lambda: Path("/repo/src"))

    env = _runner._runner_env(tmp_path, "paper")

    assert env["PYTHONPATH"] == str(Path("/repo/src"))


def test_runner_env_pythonpath_uses_platform_path_separator(monkeypatch, tmp_path):
    """The joined PYTHONPATH follows the platform's os.pathsep."""
    from dynachaos.pipelines import runner as _runner

    monkeypatch.setenv("PYTHONPATH", "/some/existing/path")
    monkeypatch.setattr(_runner, "_repo_src_dir", lambda: Path("/repo/src"))
    monkeypatch.setattr(_runner.os, "pathsep", ";")

    env = _runner._runner_env(tmp_path, "paper")

    assert env["PYTHONPATH"] == f"{Path('/repo/src')}{_runner.os.pathsep}/some/existing/path"


def test_run_section_rejects_invalid_profile(tmp_path):
    with pytest.raises(ValueError, match="profile must be one of"):
        run_section("sec02_circle_map", output_root=tmp_path, profile="bogus")


def test_run_section_recompute_deletes_existing_outputs_before_running(tmp_path, monkeypatch):
    section_dir = tmp_path / "sec02_circle_map"
    section_dir.mkdir()
    np.savez_compressed(section_dir / "devils_staircase.npz", A=[1.0], rho=[0.0], lam=[0.0])
    np.savez_compressed(section_dir / "arnold_tongues.npz", Omega=[0.0], K=[1.0], rho=[0.0])
    np.savez_compressed(section_dir / "staircase_zoom.npz", A=[1.0], rho=[0.0])
    stale_png = section_dir / "devils_staircase.png"
    stale_png.write_bytes(b"stale")

    def fake_run_module(module_name, output_root, profile):
        # Simulate the module regenerating its declared output artifacts
        # with content that satisfies each artifact's NPZ contract.
        spec = get_section("sec02_circle_map")
        for name in spec.output_files:
            path = section_dir / name
            if name.endswith(".npz"):
                keys = spec.required_npz_keys(name)
                np.savez_compressed(path, **{key: [0.0] for key in keys})
            else:
                path.write_bytes(b"fresh")
        return None

    monkeypatch.setattr(runner, "_run_module", fake_run_module)
    run_section("sec02_circle_map", output_root=tmp_path, profile="paper", recompute=True)

    assert stale_png.read_bytes() == b"fresh"


def test_run_all_runs_every_registered_section(tmp_path, monkeypatch):
    from dynachaos.pipelines.registry import list_sections

    ran_sections = []

    def fake_run_section(section_id, *, output_root, profile, recompute, timing_ledger):
        ran_sections.append(section_id)
        return [output_root / section_id / "dummy.png"]

    monkeypatch.setattr(runner, "run_section", fake_run_section)
    results = runner.run_all(output_root=tmp_path, profile="paper")

    assert ran_sections == list(list_sections())
    assert set(results) == set(list_sections())


@pytest.mark.skipif(not hasattr(__import__("os"), "wait4"), reason="requires POSIX wait4")
def test_ledger_records_child_rss_not_orchestrator(tmp_path, monkeypatch):
    """Ledger peak_rss_mb reflects child allocation, not orchestrator RSS (T1 integration).

    Real subprocesses are spawned via a monkeypatched _run_module at two different
    allocation sizes, and the ledgered value must track the difference between
    them rather than report a constant.

    The child writes to every page. ``np.zeros`` is backed by ``calloc``, which
    returns lazily-mapped pages that are never resident until touched, so a
    zeros-only child does not reliably raise peak RSS on any platform.
    """
    import os as _os

    from dynachaos.pipelines import runner as _runner
    from dynachaos.utils.system import get_rss_mb

    section_dir = tmp_path / "sec03_transition"
    section_dir.mkdir()
    np.savez_compressed(
        section_dir / "phase_diagram.npz",
        A=[1.0],
        D=[0.1],
        asym=[0.0],
        lyap=[0.1],
        schema_version=[4],
    )
    np.savez_compressed(
        section_dir / "attractors.npz",
        A_values=[1.0],
        labels=["sync"],
        initial_states=[[0.5, 0.5]],
        x_limits=[0.0, 1.0],
        y_limits=[0.0, 1.0],
        D=[0.1],
        schema_version=[4],
    )
    np.savez_compressed(
        section_dir / "basins.npz",
        x=[0.5],
        y=[0.5],
        basin=[0],
        A=[1.0],
        D=[0.1],
    )
    for png in ("phase_diagram.png", "attractors.png", "basins.png"):
        (section_dir / png).write_bytes(b"png")

    def make_allocating_module(alloc_mib):
        """Build a fake _run_module whose child allocates ``alloc_mib`` and reports its RSS."""

        def allocating_module(module_name, output_root, profile):
            import subprocess as _sp
            import sys as _sys

            proc = _sp.Popen(
                [
                    _sys.executable,
                    "-c",
                    # ones() writes every element, faulting the pages in; zeros()
                    # would be lazily mapped and never become resident.
                    f"import numpy as np; x = np.ones({alloc_mib} * 1024 * 1024 // 8); "
                    "x[::4096] += 1",
                ],
            )
            pid, status, rusage = _os.wait4(proc.pid, 0)
            divisor = 1024 * 1024 if _sys.platform == "darwin" else 1024
            return rusage.ru_maxrss / divisor

        return allocating_module

    def ledger_rss(alloc_mib, ledger_name):
        monkeypatch.setattr(_runner, "_run_module", make_allocating_module(alloc_mib))
        ledger_path = tmp_path / ledger_name
        run_section(
            "sec03_transition", output_root=tmp_path, profile="paper", timing_ledger=ledger_path
        )
        events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
        assert len(events) == 1
        return events[0]["peak_rss_mb"]

    # Measure the ledger against two different child allocations. Comparing a
    # single measurement against the orchestrator's own RSS -- the obvious
    # approach, and what this test used to do -- cannot work: getrusage(
    # RUSAGE_SELF).ru_maxrss is a high-water mark over the whole process
    # lifetime, so inside a pytest session it accumulates whatever the heaviest
    # earlier test allocated. It climbs from ~72 MB to ~130 MB on Linux and to
    # ~465 MB on macOS purely as a function of test ordering. Any threshold
    # against a moving reference is really a threshold against test-suite
    # history.
    #
    # The difference between two child allocations is the honest measurement:
    # were the ledger recording the orchestrator, both runs would report the
    # same number and the delta would collapse to zero.
    small_mib, large_mib = 100, 300
    rss_small = ledger_rss(small_mib, "timing_small.jsonl")
    rss_large = ledger_rss(large_mib, "timing_large.jsonl")

    # Each child faults in its full array, so its peak RSS must clear that
    # allocation. A value below this means nothing resembling the child was
    # measured.
    assert rss_small >= 50.0, (
        f"Expected child RSS >= 50 MB (numpy + {small_mib} MB alloc), got {rss_small:.1f} MB"
    )

    observed_delta = rss_large - rss_small
    expected_delta = large_mib - small_mib

    # The delta is only *observable* while the parent stays smaller than the
    # children. Popen fork+execs, and ru_maxrss is a high-water mark that
    # already covers the window between fork and exec, during which the child
    # still shares the parent's address space. A child's reported peak is
    # therefore effectively max(P, its own post-exec peak), for parent RSS P.
    # Writing S and L for the two child peaks, the observed delta is
    #
    #     P <= S      ->  L - S   (full delta)
    #     S <  P <  L ->  L - P   (partially masked)
    #     P >= L      ->  0       (fully masked)
    #
    # so it stays within the +/-50% band below only while P < L - (L - S)/2.
    # On a CI runner with a ~362 MB pytest process, both a 100 MB and a 300 MB
    # child reported exactly 362.5 MB and the delta collapsed to 0.0.
    #
    # That is a property of the measurement rather than a defect in the ledger,
    # so the delta is asserted only where it can be seen. It does also mean the
    # production peak_rss_mb figures inherit the orchestrator's footprint as a
    # floor -- worth knowing when reading the timing ledger.
    orchestrator_rss = get_rss_mb()
    masking_threshold = large_mib - expected_delta / 2
    if orchestrator_rss >= masking_threshold:
        pytest.skip(
            f"orchestrator RSS {orchestrator_rss:.0f} MB is at or above the "
            f"{masking_threshold:.0f} MB masking threshold; fork inheritance "
            f"compresses the child delta below what this test can resolve"
        )
    assert 0.5 * expected_delta < observed_delta < 1.5 * expected_delta, (
        f"Ledger reported {rss_small:.1f} MB for a {small_mib} MB child and "
        f"{rss_large:.1f} MB for a {large_mib} MB one — a delta of "
        f"{observed_delta:.1f} MB where ~{expected_delta} MB was expected. The "
        f"ledger does not track the child's allocation; it may be recording the "
        f"orchestrator's own rusage."
    )


# ---------------------------------------------------------------------------
# Profile helpers (pipelines.profile module)
# ---------------------------------------------------------------------------


def test_current_profile_defaults_to_paper(monkeypatch):
    """current_profile() returns 'paper' when DYNACHAOS_PROFILE is not set."""
    monkeypatch.delenv("DYNACHAOS_PROFILE", raising=False)
    from dynachaos.pipelines.profile import current_profile

    assert current_profile() == "paper"


def test_current_profile_returns_paper_when_set():
    """current_profile() returns 'paper' when explicitly set."""
    import os

    from dynachaos.pipelines.profile import current_profile

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "paper"
        assert current_profile() == "paper"
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val


def test_current_profile_returns_smoke_when_set():
    """current_profile() returns 'smoke' when explicitly set."""
    import os

    from dynachaos.pipelines.profile import current_profile

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "smoke"
        assert current_profile() == "smoke"
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val


def test_current_profile_normalizes_whitespace_and_case():
    """current_profile() strips whitespace and lowercases the value."""
    import os

    from dynachaos.pipelines.profile import current_profile

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "  SMOKE  "
        assert current_profile() == "smoke"
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val


def test_current_profile_rejects_invalid_profile_name():
    """current_profile() falls back to 'paper' for invalid profile names."""
    import os

    from dynachaos.pipelines.profile import current_profile

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "invalid_profile"
        assert current_profile() == "paper"
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val


def test_is_smoke_returns_true_when_smoke_is_active():
    """is_smoke() returns True when current_profile() == 'smoke'."""
    import os

    from dynachaos.pipelines.profile import is_smoke

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "smoke"
        assert is_smoke() is True
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val


def test_is_smoke_returns_false_when_paper_is_active():
    """is_smoke() returns False when current_profile() == 'paper'."""
    import os

    from dynachaos.pipelines.profile import is_smoke

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "paper"
        assert is_smoke() is False
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val


def test_choose_returns_paper_value_when_paper_active():
    """choose(paper, smoke) returns paper when profile is 'paper'."""
    import os

    from dynachaos.pipelines.profile import choose

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "paper"
        result = choose(paper=100, smoke=50)
        assert result == 100
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val


def test_choose_returns_smoke_value_when_smoke_active():
    """choose(paper, smoke) returns smoke when profile is 'smoke'."""
    import os

    from dynachaos.pipelines.profile import choose

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "smoke"
        result = choose(paper=100, smoke=50)
        assert result == 50
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val


def test_choose_works_with_strings():
    """choose() works with non-numeric types (e.g., strings)."""
    import os

    from dynachaos.pipelines.profile import choose

    old_val = os.environ.get("DYNACHAOS_PROFILE")
    try:
        os.environ["DYNACHAOS_PROFILE"] = "smoke"
        result = choose(paper="large_run", smoke="quick_run")
        assert result == "quick_run"
    finally:
        if old_val is None:
            os.environ.pop("DYNACHAOS_PROFILE", None)
        else:
            os.environ["DYNACHAOS_PROFILE"] = old_val
