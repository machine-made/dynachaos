"""Central configuration for dynachaos."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Single source of truth for figure theme across all scripts.
DEFAULT_FIGURE_THEME = "signature"

# Optional runtime override for quick comparisons without editing code.
FIGURE_THEME_ENV_VAR = "DYNACHAOS_THEME"


def get_figure_theme() -> str:
    """Return active figure theme ID."""
    return os.environ.get(FIGURE_THEME_ENV_VAR, DEFAULT_FIGURE_THEME)


def strip_jsonc(text: str) -> str:
    """Remove JSONC comments without touching comment markers in strings."""
    out: list[str] = []
    i = 0
    in_string = False
    escaped = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(i + 2, len(text))
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_jsonc(path: str | Path) -> dict[str, Any]:
    """Load a JSONC object from ``path`` using the project comment-stripping convention."""
    data = json.loads(strip_jsonc(Path(path).read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ValueError("JSONC config must contain an object")
    return data
