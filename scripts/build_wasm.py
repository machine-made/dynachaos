"""Build the WebAssembly bundle the gallery page loads.

The kernels in ``rust/core`` are compiled for ``wasm32-unknown-unknown`` and
wrapped by ``wasm-bindgen`` for the browser. The output goes to ``site/wasm/``
next to the rest of the generated site.

The script writes ``site/wasm/manifest.json``. It records the file names, their
SHA-256 digests and their sizes. The page build reads the digest and appends it
to the script URL, which is how ``app.js`` already busts its cache.

Run it locally with::

    uv run python scripts/build_wasm.py

The ``wasm-bindgen`` command line tool must have the same version as the
``wasm-bindgen`` crate. A different version writes glue that does not match the
module. This script reads the version from ``rust/wasm/Cargo.lock`` and
installs the matching tool if necessary.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from check_wasm_paths import find_local_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WASM_CRATE = PROJECT_ROOT / "rust" / "wasm"
CARGO_LOCK = WASM_CRATE / "Cargo.lock"
OUT_DIR = PROJECT_ROOT / "site" / "wasm"
TARGET = "wasm32-unknown-unknown"

# Warn above this compressed size, fail above the hard limit. A visitor on a
# slow link waits for this file before a live figure can draw.
WARN_GZIP_BYTES = 300 * 1024
FAIL_GZIP_BYTES = 1024 * 1024


def wasm_bindgen_version() -> str:
    """Return the wasm-bindgen version that the crate resolves to.

    The version comes from the lock file, not the manifest, because the
    manifest gives a range and the tool must match the resolved version.
    """
    text = CARGO_LOCK.read_text(encoding="utf-8")
    match = re.search(r'name = "wasm-bindgen"\nversion = "([^"]+)"', text)
    if match is None:
        raise SystemExit(f"no wasm-bindgen entry in {CARGO_LOCK}")
    return match.group(1)


def installed_version(tool: str) -> str | None:
    """Return the version of an installed tool, or None when it is absent."""
    if shutil.which(tool) is None:
        return None
    result = subprocess.run([tool, "--version"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    parts = result.stdout.split()
    return parts[-1] if parts else None


def ensure_wasm_bindgen(required: str, *, allow_install: bool) -> None:
    """Make sure the wasm-bindgen tool matches the crate version."""
    current = installed_version("wasm-bindgen")
    if current == required:
        return
    command = ["cargo", "install", "wasm-bindgen-cli", "--version", required, "--locked"]
    if not allow_install:
        found = current or "not installed"
        raise SystemExit(
            f"wasm-bindgen {required} is required but {found} was found. "
            f"Install it with: {' '.join(command)}"
        )
    print(f"installing wasm-bindgen {required} (found: {current or 'nothing'})")
    subprocess.run(command, check=True)


def remap_prefixes() -> list[tuple[str, str]]:
    """Return the (local prefix, replacement) pairs that hide the build machine.

    rustc writes absolute source paths into the module: the panic location of
    every ``unwrap`` and index in ``rust/core``, which is a path dependency
    outside the ``rust/wasm`` workspace, and the paths of registry crates. A
    published module must not carry the home folder of whoever built it.

    The pairs run from the widest prefix to the narrowest. rustc applies the
    last pair that matches, so the repository root wins over the home folder
    that contains it. Both the given and the resolved form of each prefix are
    listed, because a symlinked checkout shows up as either.
    """
    home = Path.home()
    cargo_home = Path(os.environ.get("CARGO_HOME") or home / ".cargo")
    rustup_home = Path(os.environ.get("RUSTUP_HOME") or home / ".rustup")
    wanted = [
        (home, "~"),
        (rustup_home, "/rustup"),
        (cargo_home, "/cargo"),
        (PROJECT_ROOT, "."),
    ]
    pairs: list[tuple[str, str]] = []
    for prefix, replacement in wanted:
        for form in (Path(os.path.abspath(prefix)), prefix.resolve()):
            pair = (str(form), replacement)
            # A prefix of "/" would rewrite every path, so it is never a pair.
            if len(pair[0]) > 1 and pair not in pairs:
                pairs.append(pair)
    return pairs


def build_environment() -> dict[str, str]:
    """Return the environment for ``cargo build``, with path remapping added.

    The flags go in ``CARGO_ENCODED_RUSTFLAGS`` so a path with a space in it
    survives. Flags the caller already set in ``CARGO_ENCODED_RUSTFLAGS`` or
    ``RUSTFLAGS`` are kept. The repository ``.cargo/config.toml`` only sets
    flags for x86_64 and aarch64 targets, so nothing is lost for wasm32.
    """
    env = dict(os.environ)
    if "CARGO_ENCODED_RUSTFLAGS" in env:
        flags = [f for f in env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") if f]
    else:
        flags = shlex.split(env.get("RUSTFLAGS", ""))
    flags += [f"--remap-path-prefix={src}={dst}" for src, dst in remap_prefixes()]
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    env.pop("RUSTFLAGS", None)
    return env


def check_no_local_paths(directory: Path) -> None:
    """Stop the build when a built file still carries a local path."""
    leaks = find_local_paths([directory])
    if leaks:
        for leak in leaks:
            print(f"local path in {leak.path}: {leak.marker} ({leak.count} times)", file=sys.stderr)
        raise SystemExit(f"{directory} carries a local path; see scripts/build_wasm.py")


def build_module() -> Path:
    """Compile the wasm crate and return the path of the built module."""
    subprocess.run(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(WASM_CRATE / "Cargo.toml"),
            "--target",
            TARGET,
            "--release",
        ],
        env=build_environment(),
        check=True,
    )
    module = WASM_CRATE / "target" / TARGET / "release" / "dynachaos_wasm.wasm"
    if not module.exists():
        raise SystemExit(f"cargo reported success but {module} is missing")
    return module


def run_bindgen(module: Path) -> None:
    """Write the browser glue and the trimmed module into the site directory."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    subprocess.run(
        [
            "wasm-bindgen",
            "--target",
            "web",
            "--out-dir",
            str(OUT_DIR),
            str(module),
        ],
        check=True,
    )


def digest(path: Path) -> str:
    """Return the first ten hex characters of the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def write_manifest() -> dict[str, object]:
    """Record the built files and return the manifest."""
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(OUT_DIR.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        payload = path.read_bytes()
        files[path.name] = {
            "sha256": digest(path),
            "bytes": len(payload),
            "gzip_bytes": len(gzip.compress(payload)),
        }
    manifest: dict[str, object] = {
        "wasm_bindgen": wasm_bindgen_version(),
        "target": TARGET,
        "files": files,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def check_size(manifest: dict[str, object]) -> int:
    """Report the compressed size and return a non-zero code when too large."""
    files = manifest["files"]
    if not isinstance(files, dict):
        raise SystemExit("manifest is malformed")
    total_gzip = 0
    for name, entry in sorted(files.items()):
        if not isinstance(entry, dict):
            continue
        gzip_bytes = int(entry["gzip_bytes"])
        total_gzip += gzip_bytes
        print(f"  {name}: {int(entry['bytes']):,} bytes, {gzip_bytes:,} gzipped")
    print(f"total gzipped: {total_gzip:,} bytes")

    if total_gzip > FAIL_GZIP_BYTES:
        print(f"FAIL: bundle is above the {FAIL_GZIP_BYTES:,} byte limit", file=sys.stderr)
        return 1
    if total_gzip > WARN_GZIP_BYTES:
        print(f"warning: bundle is above the {WARN_GZIP_BYTES:,} byte guide", file=sys.stderr)
    return 0


def main() -> int:
    """Build the bundle and report its size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="fail instead of installing a matching wasm-bindgen",
    )
    args = parser.parse_args()

    required = wasm_bindgen_version()
    ensure_wasm_bindgen(required, allow_install=not args.no_install)
    module = build_module()
    run_bindgen(module)
    check_no_local_paths(OUT_DIR)
    manifest = write_manifest()
    print(f"wrote {OUT_DIR}")
    return check_size(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
