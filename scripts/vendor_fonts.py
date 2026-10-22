"""Regenerate the vendored WOFF2 webfonts from a local TeX Live installation.

The interactive paper is typeset in TeX Gyre Pagella, converted from the
upstream OpenType files to WOFF2. Only the container format changes; the glyph
outlines are untouched. The results are committed under ``assets/fonts/`` because
CI runners have no TeX installation, so they cannot be regenerated there.

Usage::

    uv run --with fonttools --with brotli python scripts/vendor_fonts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets" / "fonts"

SEARCH_ROOTS = (
    Path("/usr/share/texmf/fonts/opentype/public"),
    Path("/usr/local/texlive"),
    Path("/usr/share/texlive/texmf-dist/fonts/opentype/public"),
    Path("/opt/homebrew/share/texmf/fonts/opentype/public"),
)

FACES = {
    "pagella-regular": "texgyrepagella-regular.otf",
    "pagella-italic": "texgyrepagella-italic.otf",
    "pagella-bold": "texgyrepagella-bold.otf",
    "pagella-bolditalic": "texgyrepagella-bolditalic.otf",
    "pagella-math": "texgyrepagella-math.otf",
}


def find_otf(filename: str) -> Path:
    """Locate an upstream OpenType file across the usual TeX Live layouts."""
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for candidate in root.rglob(filename):
            return candidate
    raise SystemExit(
        f"could not find {filename}. Install TeX Gyre (Debian/Ubuntu: "
        f"'apt install tex-gyre tex-gyre-math', or a full TeX Live) and retry."
    )


def main() -> None:
    try:
        from fontTools.ttLib import TTFont
    except ModuleNotFoundError:
        raise SystemExit(
            "fonttools and brotli are required. Run this via:\n"
            "  uv run --with fonttools --with brotli python scripts/vendor_fonts.py"
        ) from None

    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, filename in FACES.items():
        src = find_otf(filename)
        font = TTFont(src)
        font.flavor = "woff2"
        dst = OUT / f"{name}.woff2"
        font.save(dst)
        size = dst.stat().st_size
        total += size
        print(f"  {size / 1024:7.1f} KB  {dst.name}  <- {src}")

    print(f"  {'-' * 7}")
    print(f"  {total / 1024:7.1f} KB  total")

    licence = OUT / "GUST-FONT-LICENSE.txt"
    if not licence.exists():
        print(
            f"\nWARNING: {licence} is missing. The GUST Font License must be "
            "redistributed alongside these files.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
