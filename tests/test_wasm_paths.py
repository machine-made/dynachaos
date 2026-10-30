"""Checks on the path remapping in ``scripts/build_wasm.py``.

rustc writes absolute source paths into a WebAssembly module. A module built
in a home folder then carries that folder name to every reader of the
published file. These tests pin the two halves of the fix: the scanner that
finds a local path in a built file, and the flags that keep one out.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # A dataclass looks its own module up in sys.modules.
        sys.modules[name] = module
        old = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = old
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


@pytest.fixture(scope="module")
def paths_module():
    return _load("check_wasm_paths")


@pytest.fixture(scope="module")
def build_module():
    return _load("build_wasm")


@pytest.mark.parametrize(
    "marker",
    [b"/home/", b"/Users/", b"C:\\Users", b"C:/Users"],
)
def test_the_scanner_finds_each_home_folder_marker(tmp_path, paths_module, marker):
    module = tmp_path / "m.wasm"
    module.write_bytes(b"\x00asm\x01\x00\x00\x00" + marker + b"someone/src/lib.rs\x00")
    leaks = paths_module.find_local_paths([tmp_path])
    assert [(leak.path, leak.marker.encode(), leak.count) for leak in leaks] == [
        (module, marker, 1)
    ]


def test_the_scanner_passes_a_path_free_build(tmp_path, paths_module):
    (tmp_path / "m.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00./rust/core/src/maps.rs\x00")
    (tmp_path / "m.js").write_text("export default 1;\n")
    assert paths_module.find_local_paths([tmp_path]) == []


def test_the_scanner_reads_nested_files_and_counts_each_hit(tmp_path, paths_module):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "m.wasm").write_bytes(b"/home/x/a.rs\x00/home/x/b.rs\x00")
    (leak,) = paths_module.find_local_paths([tmp_path])
    assert leak.path == nested / "m.wasm"
    assert leak.count == 2


def test_a_missing_path_is_an_error_not_a_pass(tmp_path, paths_module):
    with pytest.raises(FileNotFoundError):
        paths_module.find_local_paths([tmp_path / "absent"])
    assert paths_module.main([str(tmp_path / "absent")]) == 2


def test_the_command_exits_one_on_a_leak_and_zero_on_a_clean_build(tmp_path, paths_module):
    (tmp_path / "m.wasm").write_bytes(b"/Users/x/a.rs")
    assert paths_module.main([str(tmp_path)]) == 1
    (tmp_path / "m.wasm").write_bytes(b"./rust/core/src/a.rs")
    assert paths_module.main([str(tmp_path)]) == 0


def test_the_repository_root_is_remapped_after_the_home_folder(build_module):
    pairs = build_module.remap_prefixes()
    sources = [src for src, _ in pairs]
    root = str(build_module.PROJECT_ROOT)
    assert root in sources
    # rustc applies the last matching pair, so the narrow root must come after
    # the wide home folder that usually contains it.
    assert sources.index(root) > sources.index(str(Path.home()))
    assert dict(pairs)[root] == "."


def test_no_pair_remaps_the_filesystem_root(build_module, monkeypatch):
    # Path.home is patched, not $HOME: Windows reads USERPROFILE instead.
    root = Path(Path.cwd().anchor)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: root))
    sources = [src for src, _ in build_module.remap_prefixes()]
    assert str(root) not in sources
    assert str(root.resolve()) not in sources


def test_the_build_keeps_the_callers_flags_and_adds_the_remap(build_module, monkeypatch):
    monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    monkeypatch.setenv("RUSTFLAGS", "-C target-feature=+simd128")
    env = build_module.build_environment()
    flags = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags[:2] == ["-C", "target-feature=+simd128"]
    assert f"--remap-path-prefix={build_module.PROJECT_ROOT}=." in flags
    assert "RUSTFLAGS" not in env


def test_the_build_keeps_encoded_flags_and_a_path_with_a_space(build_module, monkeypatch, tmp_path):
    spaced = tmp_path / "my home"
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "-C\x1fopt-level=3")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: spaced))
    flags = build_module.build_environment()["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags[:2] == ["-C", "opt-level=3"]
    # One argument per pair, even when the folder name holds a space.
    assert f"--remap-path-prefix={spaced}=~" in flags
