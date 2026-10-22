"""Build the consumable WebAssembly package that chaos-atlas vendors.

The package is the ``wasm-bindgen`` output for ``--target web`` with
TypeScript declarations, plus a ``package.json`` stamped with the source
commit, a README recording the build provenance, and a byte copy of the
repository LICENSE. It is written to ``pkg/`` at the repository root. That
directory is gitignored: the package is a build artifact, and CI builds it
and uploads it.

Run it with::

    uv run python scripts/build_wasm_pkg.py

The script takes no options. It reuses the helpers in
``scripts/build_wasm.py``: the wasm-bindgen version check and the module
build are the same ones the site bundle uses, so the package and the site
can never disagree about which tool built the module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from build_wasm import (
    TARGET,
    WASM_CRATE,
    build_module,
    ensure_wasm_bindgen,
    wasm_bindgen_version,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_ROOT / "pkg"
PKG_NAME = "@openfluids/dynachaos-wasm"


def crate_version() -> str:
    """Return the version of the rust/wasm crate."""
    manifest = tomllib.loads((WASM_CRATE / "Cargo.toml").read_text(encoding="utf-8"))
    return manifest["package"]["version"]


def git_head() -> str:
    """Return the commit the package is built from."""
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def run_bindgen(module: Path) -> None:
    """Write the web-target glue, declarations and module into pkg/."""
    if PKG_DIR.exists():
        shutil.rmtree(PKG_DIR)
    PKG_DIR.mkdir(parents=True)
    subprocess.run(
        [
            "wasm-bindgen",
            "--target",
            "web",
            "--typescript",
            "--out-dir",
            str(PKG_DIR),
            str(module),
        ],
        check=True,
    )


def write_package_json(commit: str, bindgen: str) -> None:
    """Write the package manifest with the build provenance.

    The package declares no dependencies: the glue is self-contained and the
    consumer only needs a runtime that can instantiate a wasm module.
    """
    manifest = {
        "name": PKG_NAME,
        "version": crate_version(),
        "description": (
            "The dynachaos compute kernels compiled to WebAssembly: "
            "the same Rust code the published figures ran on."
        ),
        "type": "module",
        "main": "dynachaos_wasm.js",
        "types": "dynachaos_wasm.d.ts",
        "license": "Apache-2.0",
        "dynachaos": {"commit": commit, "wasm_bindgen": bindgen},
    }
    (PKG_DIR / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_readme(commit: str, bindgen: str) -> None:
    """Write the README recording what the package is and where it came from."""
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = f"""# {PKG_NAME}

The dynachaos compute kernels compiled to WebAssembly: the same Rust code
the published figures ran on, wrapped by wasm-bindgen for the browser. This
package is what chaos-atlas copies into `vendor/dynachaos-wasm`. See
`docs/wasm-vendoring.md` in the source repository for the recipe.

## Provenance

- Source: openfluids/dynachaos commit `{commit}`
- Built: {built}
- wasm-bindgen: {bindgen}
- Target: `{TARGET}`, `--target web` with TypeScript declarations

## Contents

- `dynachaos_wasm.js` — the ES-module glue. Initialise it with `initSync`
  and the `.wasm` bytes (or the default async init) before calling any
  export.
- `dynachaos_wasm_bg.wasm` — the compiled kernels.
- `dynachaos_wasm.d.ts`, `dynachaos_wasm_bg.wasm.d.ts` — TypeScript
  declarations.
- `LICENSE` — Apache-2.0, copied unchanged from the source repository.

Every export clamps its inputs and returns one flat `Float64Array`. The
layout of each result is documented on the export in
`rust/wasm/src/lib.rs` of the source commit, and the contract is described
in `docs/wasm-architecture.md`.
"""
    (PKG_DIR / "README.md").write_text(text, encoding="utf-8")


def copy_license() -> None:
    """Copy the repository LICENSE into the package unchanged."""
    shutil.copyfile(PROJECT_ROOT / "LICENSE", PKG_DIR / "LICENSE")


def main() -> int:
    """Build the package and list what was written."""
    bindgen = wasm_bindgen_version()
    ensure_wasm_bindgen(bindgen, allow_install=True)
    module = build_module()
    run_bindgen(module)
    commit = git_head()
    write_package_json(commit, bindgen)
    write_readme(commit, bindgen)
    copy_license()
    for path in sorted(PKG_DIR.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size:,} bytes")
    print(f"wrote {PKG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
