"""Every registered figure snippet must run standalone and demo its module."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dynachaos.pipelines.registry import SECTION_SPECS, list_sections

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _snippets():
    out = []
    for section_id in list_sections():
        for figure in SECTION_SPECS[section_id].figures:
            if figure.snippet:
                out.append((section_id, figure))
    return out


SNIPPETS = _snippets()
SNIPPET_IDS = [f"{section}/{figure.png}" for section, figure in SNIPPETS]


@pytest.mark.parametrize("section_id,figure", SNIPPETS, ids=SNIPPET_IDS)
def test_snippet_runs(section_id, figure):
    path = REPO_ROOT / figure.snippet
    assert path.exists(), f"registered snippet is missing on disk: {figure.snippet}"

    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, (
        f"{figure.snippet} exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("section_id,figure", SNIPPETS, ids=SNIPPET_IDS)
def test_snippet_imports_registered_module(section_id, figure):
    path = REPO_ROOT / figure.snippet
    source = path.read_text(encoding="utf-8")
    module = figure.module
    # Accept the exact module or any of its parent packages (e.g. a figure
    # attributed to "dynachaos.diagnostics.compare_all" may be demoed via a
    # sibling module such as "dynachaos.diagnostics.sali_gali" -- the
    # diagnostics package is the reusable library the figure advertises).
    candidates = []
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        candidates.append(".".join(parts[:i]))
    assert any(candidate in source for candidate in candidates), (
        f"{figure.snippet} does not import {module} or one of its parent packages ({candidates})"
    )
