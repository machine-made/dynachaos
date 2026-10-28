"""Fail when a built WebAssembly file carries a path from the build machine.

rustc embeds absolute source paths in a module: panic locations and debug
info. When the module is built in a home folder, the folder name ends up in a
file that is published. ``scripts/build_wasm.py`` remaps those paths while it
builds. This check proves that the remap held, on the files themselves.

It scans every file under the given paths, the module and the glue alike, for
the start of a user's home folder on Linux, macOS and Windows.

Run it with::

    uv run python scripts/check_wasm_paths.py site/wasm pkg

The exit code is 1 when any file carries a marker, and 2 when a path to scan
does not exist. A directory that is missing is an error, not a pass: a check
that scans nothing proves nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Each marker opens the home folder of a user. The root folder, ``/root``, and
# CI runner folders also start with one of these on most systems, so the check
# fails there too, which is the point: no local path ships.
MARKERS: tuple[bytes, ...] = (b"/home/", b"/Users/", b"C:\\Users", b"C:/Users")


@dataclass(frozen=True)
class Leak:
    """One marker found in one file."""

    path: Path
    marker: str
    count: int


def find_local_paths(paths: Iterable[Path]) -> list[Leak]:
    """Return every marker found in the files under ``paths``.

    A path that is a file is scanned as it is. A path that is a directory is
    scanned recursively. A path that does not exist raises ``FileNotFoundError``.
    """
    leaks: list[Leak] = []
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(root)
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for file in files:
            data = file.read_bytes()
            for marker in MARKERS:
                count = data.count(marker)
                if count:
                    leaks.append(Leak(file, marker.decode(), count))
    return leaks


def main(argv: list[str]) -> int:
    """Scan the given paths and report what is found."""
    if not argv:
        print("usage: check_wasm_paths.py PATH [PATH ...]", file=sys.stderr)
        return 2
    try:
        leaks = find_local_paths([Path(a) for a in argv])
    except FileNotFoundError as error:
        print(f"no such path: {error}", file=sys.stderr)
        return 2
    for leak in leaks:
        print(f"{leak.path}: {leak.marker} x{leak.count}")
    if leaks:
        print("FAIL: a built file carries a local path", file=sys.stderr)
        return 1
    print(f"ok: no local path in {', '.join(argv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
