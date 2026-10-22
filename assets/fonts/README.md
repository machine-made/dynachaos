# Vendored webfonts

The interactive paper is typeset in **TeX Gyre Pagella** (a Palatino cut) with
**TeX Gyre Pagella Math** for the MathML. They come from the TeX ecosystem the
manuscript itself was written in, which is why they were chosen over a generic
web serif.

| File | Face |
|---|---|
| `pagella-regular.woff2` | TeX Gyre Pagella Regular |
| `pagella-italic.woff2` | TeX Gyre Pagella Italic |
| `pagella-bold.woff2` | TeX Gyre Pagella Bold |
| `pagella-bolditalic.woff2` | TeX Gyre Pagella Bold Italic |
| `pagella-math.woff2` | TeX Gyre Pagella Math |

These are format conversions (OpenType to WOFF2) of the upstream fonts. The
glyph outlines are unmodified; only the container format differs. Regenerate
them from a TeX Live install with:

```bash
uv run --with fonttools --with brotli python scripts/vendor_fonts.py
```

## Licence and attribution

Copyright (C) 2007-2018 Bogusław Jackowski and Janusz M. Nowacki.
Licensed under the GUST Font License (GFL), reproduced in
`GUST-FONT-LICENSE.txt`. Upstream:
<http://www.gust.org.pl/projects/e-foundry/tex-gyre/>

They are redistributed here under the GFL and are **not** covered by this
project's Apache-2.0 licence.

### On the GFL renaming clause

The GUST Font License asks — *"requested, but not legally required"* — that
derived works rename the fonts. These files are container-format conversions
with unmodified glyph outlines, so they are distributed under the original
names: calling them something else would misrepresent whose typeface a reader
is looking at. The conversion is disclosed above and the upstream source is
named for every face.
