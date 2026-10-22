"""Build the interactive paper at ``site/index.html``.

Two modes, and the split matters:

**Import** (local, occasional) --- ``--manuscript <path-to.tex>`` runs pandoc on
a LaTeX source that lives *outside* this repository, converts it to HTML with
native MathML and resolved citations, and caches the result as
``web/paper-body.html``. Requires pandoc.

**Build** (default, and what CI runs) --- reads the cached
``web/paper-body.html`` and assembles the finished page. Requires no pandoc, no
LaTeX, and no manuscript source. This repository therefore holds a website, not
a manuscript: no ``.tex``, no ``.bib``, no PDF.

Equations are native MathML, so they need no JavaScript and no math library;
``TeX Gyre Pagella Math`` makes them match the body text.
"""

# ruff: noqa: E501 -- inlined HTML templates are kept in their authored form.

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The registry import must also work under a bare interpreter with no
# installed dynachaos -- CI builds the page without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_shell import CSS, JS  # noqa: E402

from dynachaos.pipelines.registry import get_figure, get_section  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
BODY = WEB / "paper-body.html"
META = WEB / "paper-meta.json"
SITE = REPO / "site"
FONTS_SRC = REPO / "assets" / "fonts"
LIVE_SRC = REPO / "site-src" / "live"

FIG_RE = re.compile(r"<figure\b.*?</figure>", re.S)
PROGRAM_ARC_SLOT = "<!--PROGRAM-ARC-->"
IMG_RE = re.compile(r'<img[^>]*src="(?:figures/)?([a-z0-9_]+)/([a-z0-9_]+)\.png"[^>]*/?>', re.I)
CAP_RE = re.compile(r"<figcaption>(.*?)</figcaption>", re.S)
SEC_RE = re.compile(r'<section id="([^"]+)" class="([^"]*)"[^>]*>')
HEAD_RE = re.compile(r"<(h[12])([^>]*)>(.*?)</\1>", re.S)
TABLE_RE = re.compile(r"<table\b.*?</table>", re.S)


ANCHORED_ENVS = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|table\*?)\}(.*?)\\end\{\1\}", re.S
)
# LaTeX numbers every display environment whose name has no star. The page must
# use the same rule, or its numbers drift away from the manuscript's.
NUMBERED_ENVS = frozenset({"equation", "align", "gather", "multline"})
LABEL_RE = re.compile(r"\\label\{((?:eq|tab):[^}]+)\}")
ID_ATTR = re.compile(r'\sid="([^"]+)"')


COLSPEC_NOISE = re.compile(r"[@><!]\{(?:[^{}]|\{[^{}]*\})*\}")
COLSPEC_SIZED = re.compile(r"[pmb]\{[^{}]*\}")
COLSPEC_STAR = re.compile(r"\*\{(\d+)\}\{([^{}]*)\}")


def count_columns(spec: str) -> int:
    """Count real columns in a LaTeX column specification."""
    spec = COLSPEC_STAR.sub(lambda m: m.group(2) * int(m.group(1)), spec)
    spec = COLSPEC_NOISE.sub("", spec)
    spec = COLSPEC_SIZED.sub("l", spec)
    return sum(1 for ch in spec if ch in "lcrXY")


def brace_group(text: str, start: int) -> tuple[str, int]:
    """Read one balanced ``{...}`` group beginning at ``start``.

    Column specifications nest braces (``>{\\raggedright}p{2.2cm}``), so a
    non-greedy regex stops at the first closing brace and captures nonsense.
    """
    if start >= len(text) or text[start] != "{":
        return "", start
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return "", start


def tabularx_to_tabular(tex: str) -> tuple[str, int]:
    """Rewrite ``tabularx`` environments as plain ``tabular``.

    Pandoc converts ``tabular`` correctly but leaves ``tabularx`` untouched, so
    the column specification and every ``&`` separator land on the page as
    prose. The X/Y stretch columns carry no meaning on the web -- the table is
    laid out by CSS -- so collapsing them to ``l`` loses nothing.
    """
    out, pos, count = [], 0, 0
    marker = "\\begin{tabularx}"
    while True:
        i = tex.find(marker, pos)
        if i < 0:
            break
        j = i + len(marker)
        _, j = brace_group(tex, j)  # width argument
        spec, j = brace_group(tex, j)  # column specification
        end = tex.find("\\end{tabularx}", j)
        if not spec or end < 0:
            out.append(tex[pos : i + len(marker)])
            pos = i + len(marker)
            continue
        cols = count_columns(spec)
        if not cols:
            out.append(tex[pos:j])
            pos = j
            continue
        count += 1
        out.append(tex[pos:i])
        body = tex[j:end]
        out.append(f"\\begin{{tabular}}{{{'l' * cols}}}{body}\\end{{tabular}}")
        pos = end + len("\\end{tabularx}")
    out.append(tex[pos:])
    return "".join(out), count


TAGGED_ENV = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?)\}(.*?)\\end\{\1\}", re.S)
TAG_RE = re.compile(r"\\tag\{(.*?)\}\s*")
TAG_SYMBOLS = {"\\star": "\u2605", "\\dagger": "\u2020", "\\ddagger": "\u2021", "\\ast": "*"}


def tag_text(raw: str) -> str:
    """Turn a LaTeX ``\\tag`` argument into the text it should display."""
    out = raw.strip().strip("$").strip()
    for macro, glyph in TAG_SYMBOLS.items():
        out = out.replace(macro, glyph)
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    return out.replace("{", "").replace("}", "").strip()


def extract_tags(tex: str) -> tuple[str, list[str]]:
    """Pull ``\\tag{...}`` out of display maths and anchor each one.

    Pandoc cannot convert ``\\tag`` and drops the marker, so the manuscript's
    starred preview equations arrive with no marker at all. Removing the tag
    lets the maths convert, and the anchor lets the marker be put back beside
    the equation afterwards.
    """
    tags: list[str] = []

    def repl(match: re.Match[str]) -> str:
        body = match.group(2)
        found = TAG_RE.search(body)
        if not found:
            return match.group(0)
        index = len(tags)
        tags.append(tag_text(found.group(1)))
        cleaned = TAG_RE.sub("", body)
        return (
            f"\\hypertarget{{eqtag-{index}}}{{}}"
            f"\\begin{{{match.group(1)}}}{cleaned}\\end{{{match.group(1)}}}"
        )

    return TAGGED_ENV.sub(repl, tex), tags


def anchor_labels(tex: str) -> tuple[str, int]:
    """Insert ``\\hypertarget`` before every equation LaTeX numbers, and every table.

    Pandoc drops ``\\label`` inside display math, and misses it on some table
    environments, so ``\\eqref`` and table cross-references would dangle.
    Emitting an explicit hypertarget gives each one a real anchor. Duplicates
    (where pandoc *did* emit an id) are removed afterwards by
    :func:`dedupe_ids`. Operates on an in-memory copy; the manuscript on disk is
    never touched.

    An unlabelled equation gets an anchor too, and this is what keeps the page
    honest. LaTeX numbers every unstarred display environment, labelled or not.
    Anchoring only the labelled ones made the page skip the others and count
    1, 2, 3 over a shorter list, so its numbers slid below the manuscript's:
    the coupled-map lattice was (13) in the PDF and (11) on the page. The
    generated ids use the ``eq:`` prefix that :func:`number_equations` looks
    for, and are positional, so a shared link keeps working.
    """
    count = 0
    unlabelled = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count, unlabelled
        env = match.group(1)
        is_table = env.startswith("table")
        label = LABEL_RE.search(match.group(2))
        if not is_table and env not in NUMBERED_ENVS:
            # A starred environment shows no number, so anchoring it here would
            # let :func:`number_equations` give it one the manuscript does not.
            return match.group(0)
        if label:
            count += 1
            return f"\\hypertarget{{{label.group(1)}}}{{}}{match.group(0)}"
        if is_table:
            # An unlabelled table is never cross-referenced.
            return match.group(0)
        unlabelled += 1
        count += 1
        return f"\\hypertarget{{eq:unlabelled-{unlabelled}}}{{}}{match.group(0)}"

    return ANCHORED_ENVS.sub(repl, tex), count


def dedupe_ids(body: str) -> tuple[str, int]:
    """Drop repeated ``id`` attributes, keeping the first occurrence of each.

    A duplicate id makes the second target unreachable, so this keeps the
    earliest anchor -- which is the injected hypertarget, sitting immediately
    before the object it names.
    """
    seen: set[str] = set()
    removed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal removed
        value = match.group(1)
        if value in seen:
            removed += 1
            return ""
        seen.add(value)
        return match.group(0)

    return ID_ATTR.sub(repl, body), removed


ACCENTS = {
    ("'", "e"): "é", ("'", "E"): "É", ("`", "e"): "è", ("`", "E"): "È",
    ("^", "e"): "ê", ("^", "E"): "Ê", ('"', "e"): "ë", ("'", "a"): "á",
    ("`", "a"): "à", ("^", "a"): "â", ('"', "a"): "ä", ("'", "o"): "ó",
    ("^", "o"): "ô", ('"', "o"): "ö", ("'", "i"): "í", ("^", "i"): "î",
    ('"', "i"): "ï", ("'", "u"): "ú", ("^", "u"): "û", ('"', "u"): "ü",
    ("~", "n"): "ñ", ("'", "c"): "ć", ("~", "a"): "ã", ("~", "o"): "õ",
}  # fmt: skip

ACCENT_RE = re.compile(r"\\(['`^\"~=.])\{?([a-zA-Z])\}?")


def extract_meta(tex: str) -> dict[str, str]:
    """Pull title, byline, affiliation and a lede from the manuscript preamble.

    Cached to ``web/paper-meta.json`` at import so the build never needs the
    LaTeX source. Anything not found is left blank rather than invented --- an
    author list is not something to guess at. The lede is the manuscript's
    ``\\sitelede`` macro when it defines one, else the abstract's claim.
    """

    def clean(raw: str) -> str:
        raw = ACCENT_RE.sub(lambda m: ACCENTS.get((m.group(1), m.group(2)), m.group(2)), raw)
        raw = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", raw)
        raw = raw.replace(r"\&", "&").replace("~", " ").replace("--", "\u2013")
        raw = re.sub(r"\\[a-zA-Z]+", "", raw)
        raw = raw.replace("{", "").replace("}", "")
        return " ".join(raw.split()).strip()

    def find(pattern: str) -> str:
        m = re.search(pattern, tex, re.S)
        return clean(m.group(1)) if m else ""

    abstract = find(r"\\begin\{abstract\}(.*?)\\end\{abstract\}")
    sentences = [x for x in re.split(r"(?<=\.)\s+", abstract) if x]

    # Lead with the claim, not the background: prefer the sentence that states
    # what was investigated over the one describing prior work.
    start = 0
    for i, sentence in enumerate(sentences):
        if re.match(r"\s*We\b", sentence):
            start = i
            break
    lede = ""
    for sentence in sentences[start:]:
        if lede and len(lede) + len(sentence) > 320:
            break
        lede += (" " if lede else "") + sentence

    return {
        "title": find(r"\\title\{(.+?)\}\s*\n"),
        "authors": find(r"\\author\{(.+?)\}"),
        "affil": find(r"\\affil[^{]*\{(.+?)\}"),
        "lede": find(r"\\newcommand\{\\sitelede\}\{(.+?)\}\s*\n") or lede or abstract[:320],
    }


def run_pandoc(source: Path) -> tuple[str, list[str]]:
    """Convert a LaTeX manuscript to an HTML fragment with MathML and citations."""
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found on PATH; install it to use --manuscript")
    if not source.exists():
        raise SystemExit(f"manuscript not found: {source}")
    bib = source.parent / "references.bib"

    raw_tex = source.read_text(encoding="utf-8")
    raw_tex, n_tabx = tabularx_to_tabular(raw_tex)
    raw_tex, eq_tags = extract_tags(raw_tex)
    if eq_tags:
        print(f"  preserved {len(eq_tags)} tagged equations: {' '.join(eq_tags)}")
    if n_tabx:
        print(f"  rewrote {n_tabx} tabularx environments as tabular")
    patched, n_anchor = anchor_labels(raw_tex)
    staged = Path(tempfile.mkdtemp(prefix="dynachaos-paper-")) / source.name
    staged.write_text(patched, encoding="utf-8")
    print(f"  anchored {n_anchor} labelled equations and tables")

    cmd = [
        "pandoc", str(staged), "-f", "latex", "-t", "html5",
        "--mathml", "--citeproc", "--section-divs", "--wrap=none",
        # every author-year links to its own bibliography entry, as hyperref does
        "-M", "link-citations=true", "-M", "link-bibliography=true",
        "--resource-path", str(source.parent),
    ]  # fmt: skip
    if bib.exists():
        cmd += ["--bibliography", str(bib)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"pandoc failed:\n{proc.stderr}")
    for line in proc.stderr.splitlines():
        if line.startswith("[WARNING]"):
            print(f"  pandoc: {line}")
    return proc.stdout, eq_tags


HYPERTARGET_RE = re.compile(r'<(span|div) id="((?:eq|tab):[^"]+)"[^>]*>\s*</\1>')
BLOCK_MATH_RE = re.compile(r'<math display="block".*?</math>', re.S)
ANNOTATION_RE = re.compile(r"<annotation\b[^>]*>.*?</annotation>", re.S)
XREF_RE = re.compile(
    r'(<a href="#([^"]+)"[^>]*data-reference-type="(?:eqref|ref)"[^>]*>)\[([^\]]+)\]</a>'
)


def apply_eq_tags(body: str, tags: list[str]) -> tuple[str, int]:
    """Put each recovered ``\\tag`` marker beside its equation."""
    if not tags:
        return body, 0
    out, pos, done = [], 0, 0
    for match in re.finditer(r'<span id="eqtag-(\d+)"[^>]*></span>', body):
        index = int(match.group(1))
        if index >= len(tags):
            continue
        eq = BLOCK_MATH_RE.search(body, match.end())
        if not eq or eq.start() - match.end() > 1200:
            continue
        done += 1
        out.append(body[pos : eq.start()])
        out.append(f'<div class="eqn">{eq.group(0)}<span class="eqno">({tags[index]})</span></div>')
        pos = eq.end()
    out.append(body[pos:])
    return "".join(out), done


def number_equations(body: str) -> tuple[str, dict[str, str], int]:
    """Number labelled display equations and show the number beside each.

    Pandoc leaves ``\\eqref`` as a literal ``[eq:foo]`` and gives display maths no
    number at all, so the prose reads "the delayed logistic map [eq:delayed_logistic]".
    Number them here and hand back the mapping so the references can be rewritten.
    """
    numbers: dict[str, str] = {}
    count = 0

    def tag(match: re.Match[str]) -> str:
        nonlocal count
        label = match.group(2)
        if label.startswith("tab:"):
            return match.group(0)
        count += 1
        numbers[label] = str(count)
        return f'<span id="{label}" data-eqno="{count}"></span>'

    body = HYPERTARGET_RE.sub(tag, body)

    # Attach the number to the display equation that follows each anchor.
    out, pos = [], 0
    for m in re.finditer(r'<span id="((?:eq):[^"]+)" data-eqno="(\d+)"></span>', body):
        eq = BLOCK_MATH_RE.search(body, m.end())
        if not eq or eq.start() - m.end() > 1200:
            continue
        out.append(body[pos : eq.start()])
        out.append(f'<div class="eqn">{eq.group(0)}<span class="eqno">({m.group(2)})</span></div>')
        pos = eq.end()
    out.append(body[pos:])
    return "".join(out), numbers, count


def anchor_unnumbered_equations(body: str) -> tuple[str, int]:
    """Give the display equations nobody numbered a stable anchor.

    Run this after ``apply_eq_tags`` and ``number_equations``, so the only
    block maths left bare are the ones the manuscript never labelled. Those get
    an id -- so a reader can link to one and the page can preview it -- but
    deliberately no visible number: numbering them would shift every subsequent
    equation number and the web version would stop agreeing with the source.

    The ids are part of the page's public surface once a reader has shared a
    link, so they are positional and stable rather than content-derived.
    """
    out, pos, count = [], 0, 0
    for eq in BLOCK_MATH_RE.finditer(body):
        # Anything already wrapped carries its number and its own anchor.
        if body.rfind('<div class="eqn">', 0, eq.start()) > body.rfind("</div>", 0, eq.start()):
            continue
        count += 1
        out.append(body[pos : eq.start()])
        out.append(f'<div class="eqn eqn-bare" id="eq-u{count}">{eq.group(0)}</div>')
        pos = eq.end()
    out.append(body[pos:])
    return "".join(out), count


TABLE_CAP_RE = re.compile(r'(<table id="(tab:[^"]+)"[^>]*>\s*<caption>)(.*?)(</caption>)', re.S)


APPENDIX_SEC_RE = re.compile(r'<section id="(app:[^"]+)" class="([^"]*)"')


def is_appendix(sec_id: str, cls: str) -> bool:
    """True for a top-level section that ``\\appendix`` numbers with a letter."""
    return sec_id.startswith("app:") and "level1" in cls and "unnumbered" not in cls


def appendix_letters(body: str) -> list[tuple[int, str, str]]:
    """Find where each appendix starts, and the letter LaTeX gives it.

    ``\\appendix`` restarts top-level numbering as A, B, C. The manuscript
    labels those sections ``app:``, and pandoc keeps the id, so the prefix is
    the signal. Returns ``(start offset, id, letter)`` in document order.
    """
    out: list[tuple[int, str, str]] = []
    for match in APPENDIX_SEC_RE.finditer(body):
        if not is_appendix(match.group(1), match.group(2)):
            continue
        out.append((match.start(), match.group(1), chr(ord("A") + len(out))))
    return out


def number_tables(body: str) -> tuple[str, dict[str, str]]:
    """Number the tables in reading order so table references resolve.

    The number shown beside each caption comes from this same map, so a table
    citation and its caption cannot disagree -- the same fix ``renumber_figure_refs``
    applies to figures.
    """
    appendices = appendix_letters(body)
    numbers: dict[str, str] = {}
    main_count = 0
    within: dict[str, int] = {}
    for match in HYPERTARGET_RE.finditer(body):
        if not match.group(2).startswith("tab:"):
            continue
        # A table inside an appendix is numbered A1, B1 and so on, restarting
        # in each appendix, which is what LaTeX does after \\appendix.
        letter = next((ltr for start, _, ltr in reversed(appendices) if start < match.start()), "")
        if letter:
            within[letter] = within.get(letter, 0) + 1
            numbers[match.group(2)] = f"{letter}{within[letter]}"
        else:
            main_count += 1
            numbers[match.group(2)] = str(main_count)

    def number_caption(match: re.Match[str]) -> str:
        number = numbers.get(match.group(2))
        if not number:
            return match.group(0)
        return f'{match.group(1)}<span class="num">Table {number}.</span> {match.group(3)}{match.group(4)}'

    body = TABLE_CAP_RE.sub(number_caption, body)
    return body, numbers


FIGREF_RE = re.compile(r'(<a href="#(fig:[^"]+)"[^>]*data-reference-type="ref"[^>]*>)([^<]*)</a>')
SECREF_RE = re.compile(
    r'(<a href="#((?:sec|app|tab):[^"]+)"[^>]*data-reference-type="ref"[^>]*>)([^<]*)</a>'
)


def renumber_figure_refs(body: str) -> tuple[str, int, int]:
    """Rewrite every figure reference to the number its figure actually shows.

    Pandoc resolved these against the *manuscript's* figure order at import
    time, but this page numbers figures in *page* order and deliberately moves
    the programme-arc figure to the end -- which slid every other figure by one.
    The result was 35 references that named a figure and linked to its
    neighbour.

    Equations and tables already avoid this because ``resolve_refs`` rewrites
    them from the numbers this build assigned. Figures were simply never added
    to that map, so nothing compared the two. Deriving the text from
    ``data-fignum`` here means the caption and the sentence cannot disagree
    again, whatever order the page puts figures in.
    """
    # The id and data-fignum attributes appear in either order on the tag, so
    # read them per figure rather than assuming one regex ordering.
    shown: dict[str, str] = {}
    for tag in re.findall(r"<figure[^>]*>", body):
        num = re.search(r'data-fignum="(\d+)"', tag)
        fid = re.search(r'id="(fig:[^"]+)"', tag)
        if num and fid:
            shown[fid.group(1)] = num.group(1)

    changed = missing = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed, missing
        open_tag, target, text = match.group(1), match.group(2), match.group(3)
        number = shown.get(target)
        if number is None:
            missing += 1
            return match.group(0)
        if text.strip() != number:
            changed += 1
        return f"{open_tag}{number}</a>"

    return FIGREF_RE.sub(repl, body), changed, missing


def resolve_refs(body: str, numbers: dict[str, str]) -> tuple[str, int]:
    """Replace literal ``[eq:foo]`` reference text with the real number."""
    fixed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal fixed
        open_tag, target = match.group(1), match.group(2)
        number = numbers.get(target)
        if not number:
            return match.group(0)
        fixed += 1
        label = f"({number})" if target.startswith("eq:") else number
        return f"{open_tag}{label}</a>"

    return XREF_RE.sub(repl, body), fixed


# Transcribed verbatim from the manuscript's tabularx overview figure, which
# pandoc cannot convert (it emits the raw cell separators as prose). Content is
# the manuscript's, not invented; only the presentation is ours.
PROGRAM_ARC = (
    ("1982", "Circle map", "phase locking"),
    ("1983\u201384", "Torus instabilities", "oscillation, doubling"),
    ("1983", "Coupled maps", "symmetry breaking"),
    ("1985", "CML", "spatial extension"),
    ("1989\u201390", "GCM", "mean-field"),
    ("1994\u201398", "Milnor / biology", "applications"),
)


def program_arc(fig_id: str, caption: str, number: int) -> str:
    """Render the research-arc overview as a timeline rather than a table."""
    cells = "".join(
        f'<li style="--i:{i}"><span class="yr">{year}</span>'
        f'<span class="topic">{topic}</span><span class="mech">{mech}</span></li>'
        for i, (year, topic, mech) in enumerate(PROGRAM_ARC)
    )
    return (
        f'<figure id="{fig_id}" class="arc" data-fignum="{number}">'
        f'<ol class="arc-track">{cells}</ol>'
        f'<figcaption><span class="num">Figure {number}.</span> {caption}</figcaption>'
        f"</figure>"
    )


OPACITY_RULE = re.compile(r"([^{}]+)\{([^{}]*opacity:\s*([01])[^{}]*)\}")


def specificity(selector: str) -> tuple[int, int, int]:
    """Rough CSS specificity: (ids, classes, elements)."""
    sel = selector.strip()
    return (
        sel.count("#"),
        sel.count(".") + sel.count("[") + sel.count(":"),
        len(re.findall(r"(?:^|[\s>+~])[a-z]+", sel)),
    )


def check_reveal_rules(css: str) -> list[str]:
    """Fail the build if something is hidden with no stronger rule to show it.

    A rule that sets ``opacity:0`` needs a more specific rule setting it back to
    1, or the element never appears. Getting this wrong blanks the page while
    every other check still passes, so it is verified rather than trusted.

    Hiders revealed by a CSS animation, and keyframe steps themselves, are not
    the failure this is looking for.
    """
    KEYFRAME_STEP = re.compile(r"^(from|to|\d+%)$")
    hiders: list[tuple[str, tuple[int, int, int]]] = []
    showers: list[tuple[str, tuple[int, int, int]]] = []

    for match in OPACITY_RULE.finditer(css):
        declarations = match.group(2)
        for selector in match.group(1).split(","):
            selector = selector.strip().rsplit("}", 1)[-1].strip()
            if not selector or selector.startswith("@") or KEYFRAME_STEP.match(selector):
                continue
            if match.group(3) == "1":
                showers.append((selector, specificity(selector)))
            elif "animation" not in declarations:
                hiders.append((selector, specificity(selector)))

    problems = []
    for selector, spec in hiders:
        # the rightmost simple selector is what the reveal rule must also target
        key = re.split(r"[\s>+~]", selector.split(":")[0])[-1]
        if not key:
            continue
        if any(key in other and other_spec > spec for other, other_spec in showers):
            continue
        problems.append(selector)
    return problems


def strip_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


_ANNOTATION_RE = re.compile(r"<annotation[^>]*>.*?</annotation>", re.S)
_XREF_LABEL_RE = re.compile(r"\[(?:eq|fig|tab|sec|app):[^\]]*\]")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_ALT_MAX = 200


def build_alt_text(caption: str) -> str:
    """Build a short, clean alt string from a figure caption.

    The caption HTML carries MathML with a raw-LaTeX ``<annotation>`` fallback
    (so every symbol would otherwise be spoken twice) and pandoc's unresolved
    ``[eq:label]``/``[fig:label]`` cross-reference placeholders. Both are
    stripped before the text is cut at a sentence boundary near ``_ALT_MAX``
    characters -- never mid-word, and never mid-sentence when the sentence
    itself is short enough to keep whole.
    """
    text = _ANNOTATION_RE.sub("", caption)
    text = strip_tags(text)
    text = _XREF_LABEL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    if len(text) <= _ALT_MAX:
        return text
    window = text[:_ALT_MAX]
    last_end = None
    for m in _SENTENCE_END_RE.finditer(window):
        last_end = m
    if last_end:
        return window[: last_end.end()].strip()
    cut = window.rsplit(" ", 1)[0].rstrip(".,;: ")
    return cut + "…"


_INDEX_HEADING_TAGS = {"h2", "h3", "h4"}
_INDEX_TEXT_TAGS = {"p", "figcaption", "caption"}
_INDEX_TRUNCATE_AT = 300
_INDEX_BYTE_BUDGET = 150_000


class _SearchIndexParser(HTMLParser):
    """Walk the assembled body collecting headings and paragraph/caption text.

    A regex pass cannot track "nearest ancestor id" or "which heading a
    paragraph currently falls under" across nested tags, so this uses a real
    (if lenient) parse instead. Folded ``<details>`` are not special-cased --
    everything inside one is walked exactly like everything else, which is the
    whole point: a browser's own Ctrl+F cannot see in there, so this index has
    to. Text inside a MathML ``<annotation>`` (the raw-LaTeX fallback) is
    skipped rather than concatenated, or every symbol would be indexed twice.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.id_stack: list[str] = [""]
        self.section = ""
        self.units: list[dict[str, str]] = []
        self._capture: list[list[str]] = []
        self._annotation_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._open(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._open(tag, attrs)
        self._close(tag)

    def _open(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        self.id_stack.append(attrs_d.get("id") or self.id_stack[-1])
        if tag in ("annotation", "annotation-xml"):
            self._annotation_depth += 1
        if tag in _INDEX_HEADING_TAGS or tag in _INDEX_TEXT_TAGS:
            self._capture.append([])

    def handle_endtag(self, tag: str) -> None:
        self._close(tag)

    def _close(self, tag: str) -> None:
        if (tag in _INDEX_HEADING_TAGS or tag in _INDEX_TEXT_TAGS) and self._capture:
            text = re.sub(r"\s+", " ", "".join(self._capture.pop())).strip()
            is_heading = tag in _INDEX_HEADING_TAGS
            section = text if is_heading else self.section
            if is_heading:
                self.section = text
            if text:
                unit_id = self.id_stack[-1]
                self.units.append({"id": unit_id, "section": section, "tag": tag, "text": text})
        if tag in ("annotation", "annotation-xml") and self._annotation_depth:
            self._annotation_depth -= 1
        if len(self.id_stack) > 1:
            self.id_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._annotation_depth or not self._capture:
            return
        self._capture[-1].append(data)


def build_search_index(body: str) -> list[dict[str, str]]:
    """Extract searchable units (headings, paragraphs, captions) from the body.

    Runs on the final assembled body -- after figures, tables and back matter
    are folded -- so ids are the page's own anchors and folded ``<details>``
    content is included, not skipped.
    """
    parser = _SearchIndexParser()
    parser.feed(body)
    parser.close()
    return parser.units


def search_index_json(units: list[dict[str, str]]) -> tuple[str, bool]:
    """Serialise the search index, truncating unit text if it runs too large.

    Relevance for a jump-to-section search does not need a whole paragraph, so
    once the raw JSON would exceed roughly 150 KB, each unit's text is cut to
    its first ~300 characters rather than shipping the full page twice over.
    """

    def dump(units: list[dict[str, str]]) -> str:
        return json.dumps(units, ensure_ascii=False, separators=(",", ":"))

    raw = dump(units)
    truncated = False
    if len(raw.encode("utf-8")) > _INDEX_BYTE_BUDGET:
        truncated = True
        units = [
            {
                **u,
                "text": (
                    u["text"][:_INDEX_TRUNCATE_AT] + "…"
                    if len(u["text"]) > _INDEX_TRUNCATE_AT
                    else u["text"]
                ),
            }
            for u in units
        ]
        raw = dump(units)
    # a paragraph's own text could contain "</script" and prematurely close
    # the block this gets embedded in
    raw = raw.replace("</", "<\\/")
    return raw, truncated


# ---------------------------------------------------------------------------
# figure snippets: hand-rolled server-side syntax highlighting (no pygments --
# the page stays zero-dependency and the build stays stdlib+repo-only)
# ---------------------------------------------------------------------------

_PY_KEYWORDS = (
    "False|None|True|and|as|assert|async|await|break|class|continue|def|del|"
    "elif|else|except|finally|for|from|global|if|import|in|is|lambda|"
    "nonlocal|not|or|pass|raise|return|try|while|with|yield"
)
_SNIPPET_TOKEN_RE = re.compile(
    r"(?P<comment>#[^\n]*)"
    r"|(?P<string>'''(?:.|\n)*?'''|\"\"\"(?:.|\n)*?\"\"\"|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\")"
    r"|(?P<fn>(?<=def\s)[A-Za-z_]\w*|(?<=class\s)[A-Za-z_]\w*)"
    r"|(?P<mod>(?<=import\s)[A-Za-z_][\w.]*|(?<=from\s)[A-Za-z_][\w.]*)"
    r"|(?P<kw>\b(?:" + _PY_KEYWORDS + r")\b)"
    r"|(?P<num>\b\d+\.?\d*(?:[eE][+-]?\d+)?\b)"
    r"|(?P<other>.)",
    re.S,
)


def highlight_python(source: str) -> str:
    """Tokenize a Python snippet into HTML spans, without any third-party lexer."""
    out = []
    for m in _SNIPPET_TOKEN_RE.finditer(source):
        text = html.escape(m.group())
        kind = m.lastgroup
        out.append(text if kind == "other" else f'<span class="tok-{kind}">{text}</span>')
    return "".join(out)


def figure_code_block(section: str, name: str) -> tuple[str, bool]:
    """Render the "how this figure was computed" panel for one figure.

    Returns the HTML (empty string when the figure has no registered snippet)
    and whether a snippet was found, so the caller can gate the page stats.
    """
    figure = get_figure(section, f"{name}.png")
    if figure is None or not figure.snippet:
        return "", False
    snippet_path = REPO / figure.snippet
    if not snippet_path.exists():
        return "", False
    source = snippet_path.read_text(encoding="utf-8")
    footer = ""
    if figure.npz:
        keys = get_section(section).required_npz_keys(figure.npz)
        key_list = ", ".join(keys) if keys else "no contract keys"
        footer = f'<p class="fig-code-footer">Artifact: <code>{figure.npz}</code> ({key_list})</p>'
    return (
        '<details class="fig-code">'
        "<summary>How this figure was computed</summary>"
        f'<pre><code class="fig-snippet">{highlight_python(source)}</code></pre>'
        f'<div class="fig-code-actions"><button type="button" class="fig-code-copy">copy</button></div>'
        f"{footer}"
        "</details>"
    ), True


LIVE_FIGURES = {
    "fig:arnold_tongues": "arnold_tongues",
    "fig:devils_staircase": "devils_staircase",
    "fig:delayed_logistic_attractors": "delayed_logistic_attractors",
    "fig:map_I_attractors": "torus_doubling_attractors",
    "fig:double_staircase": "double_staircase",
    "fig:spacetime_diagrams": "spacetime_diagrams",
}


def figure_block(
    fig_id: str, section: str, name: str, caption: str, dims: dict[str, tuple[int, int]]
) -> str:
    """Render one figure.

    The static image is always present and is what the reader sees first; the
    interactive chart is fetched only when asked for, so no reader pays for a
    payload they never open. A live-capable figure gets an interact button
    even when its JSON is missing; live mode then computes the view in-browser.
    """
    thumb = SITE / "thumbs" / section / f"{name}.webp"
    data = SITE / "data" / section / f"{name}.json"
    src = f"thumbs/{section}/{name}.webp" if thumb.exists() else f"full/{section}/{name}.png"
    size = ""
    if (section, name) in dims:
        w, h = dims[(section, name)]
        size = f' width="{w}" height="{h}"'
    alt = html.escape(build_alt_text(caption), quote=True)
    live_kind = LIVE_FIGURES.get(fig_id, "")

    acts = ['<button type="button" class="act-zoom">enlarge</button>']
    attrs = f' id="{fig_id}"'
    badge = ""
    if data.exists() or live_kind:
        acts.insert(0, '<button type="button" class="act-interact">interact</button>')
        attrs += ' data-state="static"'
        if live_kind:
            attrs += f' data-live="{html.escape(live_kind, quote=True)}"'
        if data.exists():
            acts.insert(1, '<button type="button" class="act-copylink">copy link</button>')
            attrs += f' data-src="data/{section}/{name}.json"'
            badge = (
                '<button type="button" class="fig-badge" hidden'
                ' aria-label="Figure modified from published defaults. Reset to published values">'
                "modified: reset</button>"
            )

    code_html, has_snippet = figure_code_block(section, name)
    if has_snippet:
        acts.insert(0, '<button type="button" class="act-code">code</button>')

    return (
        f"<figure{attrs}>"
        f'<div class="fig-head"><span class="name">{section}/{name}</span>{badge}'
        f'<span class="acts">{"".join(acts)}</span></div>'
        f'<div class="fig-body">'
        f'<img src="{src}" data-full="full/{section}/{name}.png" alt="{alt}"'
        f' loading="lazy" decoding="async"{size}>'
        f"</div>"
        f"<figcaption>{caption}</figcaption>{code_html}</figure>"
    )


def get_image_size(path: Path) -> tuple[int, int] | None:
    try:
        data = path.read_bytes()
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            return struct.unpack(">II", data[16:24])
        if data.startswith(b"RIFF") and len(data) >= 30 and data[8:12] == b"WEBP":
            tag = data[12:16]
            if tag == b"VP8 ":
                w = (data[26] | (data[27] << 8)) & 0x3FFF
                h = (data[28] | (data[29] << 8)) & 0x3FFF
                return (w, h)
            elif tag == b"VP8L" and len(data) >= 25:
                b0, b1, b2, b3 = data[21:25]
                w = 1 + (((b1 & 0x3F) << 8) | b0)
                h = 1 + (((b3 & 0xF) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
                return (w, h)
            elif tag == b"VP8X" and len(data) >= 30:
                w = 1 + int.from_bytes(data[24:27], "little")
                h = 1 + int.from_bytes(data[27:30], "little")
                return (w, h)
    except Exception:
        pass
    return None


def thumb_dims() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for base in (SITE / "thumbs", SITE / "full"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in (".webp", ".png"):
                sz = get_image_size(path)
                if sz and (path.parent.name, path.stem) not in out:
                    out[(path.parent.name, path.stem)] = sz
    return out


def transform(
    body: str, eq_tags: list[str] | None = None
) -> tuple[str, list[tuple[int, str, str, str]]]:
    """Rewrite pandoc output into the page's own figure, table and heading shapes."""
    dims = thumb_dims()
    stats: dict[str, object] = {
        "interactive": 0,
        "static": 0,
        "arc": 0,
        "dropped": [],
        "snippets_present": 0,
        "snippets_missing": [],
    }

    def do_figure(match: re.Match[str]) -> str:
        block = match.group(0)
        fid = re.search(r'<figure id="([^"]+)"', block)
        fig_id = fid.group(1) if fid else ""
        img = IMG_RE.search(block)
        cap = CAP_RE.search(block)
        caption = cap.group(1).strip() if cap else ""
        if not img:
            # pandoc cannot convert the tabularx overview; it emits raw cell
            # separators as prose. Render it as a purpose-built timeline instead.
            if fig_id == "fig:program_map":
                stats["arc"] += 1
                stats["arc_caption"] = caption
                return PROGRAM_ARC_SLOT
            stats["dropped"].append(fig_id or "(unlabelled)")
            return ""
        section, name = img.group(1), img.group(2)
        if (SITE / "data" / section / f"{name}.json").exists():
            stats["interactive"] += 1
        else:
            stats["static"] += 1
        figure = get_figure(section, f"{name}.png")
        if figure is not None and figure.snippet:
            stats["snippets_present"] += 1
        else:
            stats["snippets_missing"].append(f"{section}/{name}")
        return figure_block(fig_id, section, name, caption, dims)

    body = FIG_RE.sub(do_figure, body)
    body = TABLE_RE.sub(lambda m: f'<div class="table-wrap">{m.group(0)}</div>', body)

    # Number the figures so a reader can name what they are looking at, and so
    # that a number cited from this page means the same figure in the
    # manuscript. The page shows the programme-arc figure last for design
    # reasons, but the manuscript calls it Figure 1, so its number is reserved
    # here and the figures that follow it in the manuscript start at 2. Page
    # order and manuscript order agree once that one figure is set aside.
    arc_fignum = 1
    fig_no = arc_fignum

    def number_figure(match: re.Match[str]) -> str:
        nonlocal fig_no
        block = match.group(0)
        fig_no += 1
        return block.replace(
            "<figcaption>", f'<figcaption><span class="num">Figure {fig_no}.</span> ', 1
        ).replace("<figure ", f'<figure data-fignum="{fig_no}" ', 1)

    body = FIG_RE.sub(number_figure, body)
    if PROGRAM_ARC_SLOT in body:
        arc_caption = stats.get("arc_caption")
        if not arc_caption:
            raise SystemExit(
                "could not find the fig:program_map caption in the manuscript body "
                "to build the timeline figure"
            )
        body = body.replace(
            PROGRAM_ARC_SLOT, program_arc("fig:program_map", arc_caption, arc_fignum), 1
        )

    # Keep pandoc's ids verbatim: they are the manuscript's own \label anchors,
    # and every internal cross-reference in the text points at them.
    body = SEC_RE.sub(lambda m: f'<section id="{m.group(1)}" class="{m.group(2)} reveal">', body)

    # Number sections from the heading tree, skipping the ones the manuscript
    # marks unnumbered, then demote level1 -> h2 and level2 -> h3 so the
    # shell's type scale applies.
    nav: list[tuple[int, str, str, str]] = []
    top = sub = n_app = 0
    # After \appendix the manuscript numbers top-level sections A, B, C, and
    # its subsections A.1, B.1. The page has to do the same or a reader
    # following "see B" from the paper lands on a section called 12.
    letter = ""
    for m in re.finditer(r'<section id="([^"]+)" class="([^"]*)"', body):
        sec_id, cls = m.group(1), m.group(2)
        tail = body[m.end() : m.end() + 4000]
        h = HEAD_RE.search(tail)
        if not h:
            continue
        level = 1 if "level1" in cls else 2
        number = ""
        if "unnumbered" not in cls:
            if level == 1:
                sub = 0
                if is_appendix(sec_id, cls):
                    n_app += 1
                    letter = chr(ord("A") + n_app - 1)
                    number = letter
                else:
                    letter = ""
                    top += 1
                    number = str(top)
            else:
                sub += 1
                number = f"{letter or top}.{sub}"
        nav.append((level, sec_id, strip_tags(h.group(3)), number))

    numbers = iter(n for *_, n in nav)
    # nav was built by walking sections in document order and pairing each with
    # the next heading found in its tail (see the loop just above): the same
    # order HEAD_RE.sub below visits headings in, so this second iterator can
    # ride along with `numbers` and hand demote() the enclosing section's id
    # for the copy-link anchor without re-parsing anything.
    heading_ids = iter(sec_id for _, sec_id, _, _ in nav)

    def demote(match: re.Match[str]) -> str:
        tag, attrs, text = match.group(1), match.group(2), match.group(3)
        new = "h2" if tag == "h1" else "h3"
        number = next(numbers, "")
        label = f'<span class="secno">{number}</span> ' if number else ""
        sec_id = next(heading_ids, "")
        anchor = (
            f' <a class="head-anchor" href="#{sec_id}" aria-label="Copy link to this section">#</a>'
            if sec_id
            else ""
        )
        return f"<{new}{attrs}>{label}{text}{anchor}</{new}>"

    body = HEAD_RE.sub(demote, body)

    shown_secs = {sec_id: number for _, sec_id, _, number in nav if number}
    body, n_tagged = apply_eq_tags(body, eq_tags or [])
    body, eq_numbers, n_eq = number_equations(body)
    body, n_bare = anchor_unnumbered_equations(body)
    body, tab_numbers = number_tables(body)
    body, n_fixed = resolve_refs(body, {**eq_numbers, **tab_numbers})

    # Pandoc resolved section and table references at import time and does not
    # act on \appendix, so it wrote "12" where the manuscript says "B" and "3"
    # where it says "A1". resolve_refs cannot correct these: it only rewrites
    # the references pandoc left as literal text. Rewrite them from the numbers
    # this build assigned, the way renumber_figure_refs does for figures.
    shown_refs = {**shown_secs, **tab_numbers}

    def fix_ref(match: re.Match[str]) -> str:
        number = shown_refs.get(match.group(2))
        return match.group(0) if number is None else f"{match.group(1)}{number}</a>"

    body = SECREF_RE.sub(fix_ref, body)
    body, n_figref, n_figref_missing = renumber_figure_refs(body)
    # The raw-LaTeX annotation duplicates every equation as plain text when a
    # browser does not render MathML; drop it rather than rely on a UA stylesheet.
    n_ann = len(ANNOTATION_RE.findall(body))
    body = ANNOTATION_RE.sub("", body)
    body = body.replace(' xmlns="http://www.w3.org/1998/Math/MathML"', "")
    body = body.replace('<math display="inline">', "<math>")
    body = re.sub(r"<semantics>\s*", "", body)
    body = re.sub(r"\s*</semantics>", "", body)
    body, folded = fold_back_matter(body)
    print(
        f"  figures: {stats['interactive']} interactive, {stats['static']} static "
        f"(numbered 1-{fig_no})"
    )
    if stats["dropped"]:
        print(f"  DROPPED (pandoc could not convert): {', '.join(stats['dropped'])}")
    n_total_figs = stats["snippets_present"] + len(stats["snippets_missing"])
    print(f"  figure snippets: {stats['snippets_present']} of {n_total_figs}")
    if stats["snippets_missing"]:
        print(f"  WARNING: figures without a code snippet: {', '.join(stats['snippets_missing'])}")
    print(f"  sections numbered: {sum(1 for *_, n in nav if n)} of {len(nav)}")
    print(
        f"  equations numbered: {n_eq}; tagged: {n_tagged}; "
        f"anchored unnumbered: {n_bare}; references resolved: {n_fixed}"
    )
    print(f"  figure references renumbered to match captions: {n_figref}")
    if n_figref_missing:
        print(f"  WARNING: {n_figref_missing} figure references point at no figure on the page")
    print(f"  latex annotations stripped: {n_ann}")
    print(f"  back matter folded: {folded}")
    return body, nav


# Reference material the reader should be able to skip: it interrupts the
# argument but must stay on the page and stay linkable.
BACK_MATTER = ("sec:glossary", "app:provenance", "app:repro_index", "app:assumptions")


def fold_back_matter(body: str) -> tuple[str, int]:
    """Wrap reference sections in a collapsed ``<details>``.

    The section keeps its id and stays in the document, so every cross-reference
    into it still resolves; browsers open a closed ``<details>`` automatically
    when a link targets something inside it.
    """
    folded = 0
    for sec_id in BACK_MATTER:
        pattern = re.compile(
            rf'(<section id="{re.escape(sec_id)}"[^>]*>)\s*<h2([^>]*)>(.*?)</h2>', re.S
        )
        match = pattern.search(body)
        if not match:
            continue
        folded += 1
        body = pattern.sub(
            lambda m: (
                f"{m.group(1)}<details class='backmatter'>"
                f"<summary><h2{m.group(2)}>{m.group(3)}</h2></summary>"
            ),
            body,
            count=1,
        )
        end = body.index("</section>", body.index(f'id="{sec_id}"'))
        body = body[:end] + "</details>" + body[end:]

    # The reader has just been told which manuscript figure each one reproduces;
    # this is where the numbering is most likely to be compared.
    note = (
        '<p class="fig-note">Figure numbers on this page are the manuscript\'s own, so a '
        "number cited from here means the same figure in the paper.</p>"
    )
    marker = '<table id="tab:repro_index"'
    if marker not in body:
        # Without the note a reader has no way to know that a figure number
        # cited from this page is the paper's own -- fail the build rather than
        # ship that silently.
        raise SystemExit(
            "fold_back_matter: reproduction-index table not found; "
            "the figure-numbering note has nowhere to go"
        )
    body = body.replace(marker, note + marker, 1)
    return body, folded


def build_nav(nav: list[tuple[int, str, str, str]]) -> str:
    """Nested contents: top-level entries always visible, children on demand."""
    out = ['<nav class="spine" aria-label="Contents"><ul class="toc">']
    open_sub = False
    for level, sec_id, title, number in nav:
        label = f"<i>{number}</i> {html.escape(title)}" if number else html.escape(title)
        if level == 1:
            if open_sub:
                out.append("</ul></li>")
                open_sub = False
            out.append(f'<li class="top"><a href="#{sec_id}">{label}</a><ul class="sub">')
            open_sub = True
        else:
            if not open_sub:
                out.append('<li class="top"><ul class="sub">')
                open_sub = True
            out.append(f'<li><a href="#{sec_id}">{label}</a></li>')
    if open_sub:
        out.append("</ul></li>")
    out.append("</ul></nav>")
    return "".join(out)


def count_figures(body: str) -> int:
    """Count the figures in the manuscript body. Each one has a figcaption."""
    count = len(CAP_RE.findall(body))
    if not count:
        raise SystemExit("found no <figcaption> to count")
    return count


def count_diagnostic_spotlights(body: str) -> int:
    """Count the 'Diagnostic spotlight' subsections the manuscript actually has."""
    count = len(re.findall(r"Diagnostic spotlight", body))
    if not count:
        raise SystemExit("found no 'Diagnostic spotlight' subsections to count")
    return count


def hero(meta: dict[str, str], body: str) -> str:
    stats = "".join(
        f"<li><b>{b}</b><span>{lab}</span></li>"
        for b, lab in (
            (str(count_figures(body)), "figures recomputed"),
            (str(count_diagnostic_spotlights(body)), "diagnostic spotlights"),
        )
    )
    return f"""<header class="hero">
<canvas id="bifurcation" aria-hidden="true"></canvas>
<canvas id="hero-motes" aria-hidden="true"></canvas>
<div class="hero-inner">
<p class="eyebrow">Recomputed with dynachaos</p>
<h1>{html.escape(meta["title"])}</h1>
<p class="byline">{html.escape(meta["authors"])}<span class="affil">{html.escape(meta["affil"])}</span></p>
<p class="lede">{html.escape(meta["lede"])}</p>
<p class="hero-links"><a href="gallery.html">all figures as a gallery</a></p>
<ul class="stats">{stats}</ul>
</div>
<div class="legend">
<span><span class="swatch" style="background:var(--locked)"></span><b>&lambda; &lt; 0</b> &nbsp;mode-locked</span>
<span><span class="swatch" style="background:var(--torus)"></span><b>&lambda; &asymp; 0</b> &nbsp;quasiperiodic</span>
<span><span class="swatch" style="background:var(--chaotic)"></span><b>&lambda; &gt; 0</b> &nbsp;chaotic</span>
<span style="color:var(--ink-low)">above: logistic attractor, computed live in your browser</span>
</div>
</header>"""


CONTROLS = """<div class="progress" aria-hidden="true"></div>
<div class="controls" role="group" aria-label="Reading controls">
<button type="button" class="c-rail" title="Toggle contents sidebar">contents</button>
<span class="sep"></span>
<button type="button" class="c-smaller" title="Smaller text" aria-label="Smaller text">A&minus;</button>
<button type="button" class="c-larger" title="Larger text" aria-label="Larger text">A+</button>
<span class="sep"></span>
<button type="button" class="c-fig" title="Toggle larger figures">figures</button>
<button type="button" class="c-theme" title="Cycle theme: auto, light, dark">auto</button>
<span class="sep"></span>
<button type="button" class="c-search" title="Search this page (/)">search</button>
<button type="button" class="c-help" title="Keyboard shortcuts (?)">keys</button>
</div>"""

SEARCH_OVERLAY = """<div class="search-overlay" role="dialog" aria-modal="true" aria-label="Search this page">
<div class="search-panel">
<input type="text" class="search-input" placeholder="Search this page&hellip;" aria-label="Search this page" autocomplete="off" spellcheck="false">
<ul class="search-results" role="listbox" aria-label="Search results"></ul>
<p class="search-empty">No matches.</p>
</div>
</div>"""

# Rows are rendered by JS from a single source array (see SHORTCUTS in the
# main script) so this table cannot list a key that nothing actually wires up.
HELP_OVERLAY = """<div class="help-overlay" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
<div class="help-panel">
<div class="help-head"><h2>Keyboard shortcuts</h2>
<button type="button" class="help-close" aria-label="Close">close</button></div>
<table class="help-table"><thead><tr><th>Key</th><th>Action</th></tr></thead>
<tbody></tbody></table>
</div>
</div>"""

TOC_BACKDROP = '<div class="toc-backdrop" hidden></div>'


# Applied before the ~24 KB inline <style>, so a reader with a saved theme or
# font size never sees the default flash while the rest of the page parses.
# Same localStorage keys and attribute names as the controls IIFE in JS below
# -- that script re-applies them harmlessly once it runs, it never fights this.
PREF_SCRIPT = """<script>(function(){"use strict";try{
var d=document.documentElement;
var g=function(k){return localStorage.getItem(k);};
var th=g("dc-theme");
if(th)d.setAttribute("data-theme",th);
d.setAttribute("data-size",g("dc-size")||"l");
d.setAttribute("data-rail",g("dc-rail")||"on");
d.setAttribute("data-figsize",g("dc-fig")||"short");
}catch(e){}})();</script>"""


def minify_css(css: str) -> str:
    css = re.sub(r"/\*[\s\S]*?\*/", "", css)
    css = re.sub(r"\s*([\{\}\:\;\,\>\+\~])\s*", r"\1", css)
    css = re.sub(r";\}", "}", css)
    css = re.sub(r"\s+", " ", css)
    return css.strip()


def minify_js(js_code: str) -> str:
    if shutil.which("bun"):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fin:
            fin.write(js_code)
            fin.flush()
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fout:
                res = subprocess.run(
                    ["bun", "build", fin.name, "--minify", "--outfile", fout.name],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    return Path(fout.name).read_text(encoding="utf-8").strip()
    lines = []
    for line in js_code.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    return text.strip()


def minify_html(html_str: str) -> str:
    tokens = re.split(r"(<pre\b[^>]*>.*?</pre>|<code\b[^>]*>.*?</code>)", html_str, flags=re.S)
    for i in range(0, len(tokens), 2):
        # Keep one space before an inline tag: pandoc puts U+00A0 between an
        # italic term and its citation span, and dropping it glues them.
        tokens[i] = re.sub(
            r">\s+<(?!/?(?:a|span|em|strong|code|sub|sup|i|b|math)\b)", "><", tokens[i]
        )
        tokens[i] = re.sub(r"\s+", " ", tokens[i])
    return "".join(tokens).strip()


def assemble(body: str, nav: str, meta: dict[str, str], js_hash: str = "") -> str:
    min_css = minify_css(CSS)
    script_tag = (
        f'<script src="app.js?v={js_hash}" defer></script>'
        if js_hash
        else '<script src="app.js" defer></script>'
    )
    raw_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(meta["title"])}</title>
<meta name="description" content="{html.escape(meta["lede"][:180])}">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌀</text></svg>">
<link rel="preload" href="fonts/pagella-bold.woff2" as="font" type="font/woff2" crossorigin fetchpriority="high">
<link rel="preload" href="fonts/pagella-regular.woff2" as="font" type="font/woff2" crossorigin>
{PREF_SCRIPT}
<style>{min_css}</style>
{script_tag}
</head>
<body>
<a class="skip-link" href="#main-content">Skip to content</a>
{CONTROLS}
{hero(meta, body)}
<div class="shell">
{nav}
<main id="main-content">
<article>
{body}
</article>
</main>
</div>
<footer class="foot">
<p>Every figure is recomputed by the same public commands:</p>
<pre><code>pip install dynachaos
dynachaos list
dynachaos run all</code></pre>
<p><a href="gallery.html">Figure index</a> &middot;
<a href="https://github.com/openfluids/dynachaos">dynachaos on GitHub</a></p>
</footer>
<div class="lb" role="dialog" aria-modal="true" aria-label="Figure viewer">
<button class="lb-close" type="button">close</button>
<img alt="" width="900" height="600">
<p class="lb-cap"></p>
<p class="lb-actions"><a class="lb-goto" href="#" data-fig-target="">go to figure in text</a></p>
<p class="lb-announce sr-only" aria-live="polite"></p>
</div>
{SEARCH_OVERLAY}
{HELP_OVERLAY}
{TOC_BACKDROP}
</body>
</html>"""
    return minify_html(raw_html)


def copy_and_subset_fonts(body: str, src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    chars = sorted(set(body))
    unicodes_str = ",".join(f"U+{ord(c):04X}" for c in chars if ord(c) < 0x2500) or "U+0020-007E"

    count = 0
    for face in src_dir.glob("*.woff2"):
        dst = dst_dir / face.name
        if face.stem == "pagella-bold":
            face_unicodes = "U+0020-007E,U+00A0-00FF,U+2010-2027"
        elif face.stem in ("pagella-regular", "pagella-italic", "pagella-bolditalic"):
            face_unicodes = f"{unicodes_str},U+0020-007E,U+00A0-00FF,U+2010-2027"
        else:
            face_unicodes = None

        if face_unicodes:
            try:
                from fontTools.subset import main as subset_main

                with tempfile.NamedTemporaryFile(suffix=".woff2", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                args = [
                    str(face),
                    f"--unicodes={face_unicodes}",
                    "--flavor=woff2",
                    "--layout-features=*",
                    f"--output-file={tmp_path}",
                ]
                subset_main(args)
                if tmp_path.exists() and tmp_path.stat().st_size > 0:
                    shutil.copy2(tmp_path, dst)
                    count += 1
                    continue
            except Exception:
                pass
        shutil.copy2(face, dst)
        count += 1
    return count


def copy_live_runtime(src_dir: Path, dst_dir: Path) -> int:
    """Copy the hand-written live-figure JS into the generated site.

    These files stay as ESM and are not folded into app.js: a worker needs
    its own URL, and the scheduler is imported by node tests. Missing source
    is a no-op so a checkout without site-src still builds the paper page.
    """
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src_dir.glob("*.js")):
        shutil.copy2(path, dst_dir / path.name)
        count += 1
    return count


def write_cml_x0(dst_dir: Path) -> None:
    """Write the live CML figure's initial field to ``site/live/cml-x0.json``.

    The paper draws ``x0 = default_rng(42).uniform(0, 1, N)``
    (``cml/spatiotemporal.py`` ``simulate_cml``), and PCG64 cannot be
    reproduced cheaply in the browser, so the page fetches the field
    instead. ``uniform(0, 1, N)`` consumes the first N of these 512 draws —
    one 64-bit draw per double — so the file's first 200 values are the
    published figure's x0 exactly, and the site-count control slices to
    the chosen N. JSON round-trips f64 exactly.
    """
    import numpy as np

    x0 = np.random.default_rng(42).uniform(0, 1, 512)
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "cml-x0.json").write_text(
        json.dumps(x0.tolist(), separators=(",", ":")), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manuscript", type=Path, help="LaTeX source to import (local only)")
    args = ap.parse_args()

    if args.manuscript:
        WEB.mkdir(parents=True, exist_ok=True)
        tex = args.manuscript.read_text(encoding="utf-8")
        raw, eq_tags = run_pandoc(args.manuscript)
        BODY.write_text(raw, encoding="utf-8")
        meta = extract_meta(tex)
        meta["eq_tags"] = eq_tags
        missing = [k for k, v in meta.items() if not v]
        if missing:
            print(f"  WARNING: could not parse from the manuscript: {', '.join(missing)}")
        META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"  imported {len(raw):,} bytes -> {BODY.relative_to(REPO)}")
        print(f"  title: {meta['title'][:70]}")

    if not BODY.exists():
        raise SystemExit(
            f"{BODY.relative_to(REPO)} is missing. Import it once with:\n"
            f"  uv run --extra viz python scripts/build_paper.py --manuscript <path/to/main.tex>"
        )

    SITE.mkdir(parents=True, exist_ok=True)
    fonts_dst = SITE / "fonts"

    if not META.exists():
        raise SystemExit(f"{META.relative_to(REPO)} is missing. Import it once with --manuscript.")
    meta = json.loads(META.read_text(encoding="utf-8"))

    hidden = check_reveal_rules(CSS)
    if hidden:
        raise SystemExit(
            "CSS would hide these with no stronger rule to reveal them, which "
            "blanks the page:\n  " + "\n  ".join(hidden)
        )

    body, nav = transform(BODY.read_text(encoding="utf-8"), meta.get("eq_tags"))
    body, dropped = dedupe_ids(body)
    if dropped:
        print(f"  removed {dropped} duplicate id attributes")

    copy_and_subset_fonts(body, FONTS_SRC, fonts_dst)
    n_live = copy_live_runtime(LIVE_SRC, SITE / "live")
    write_cml_x0(SITE / "live")

    index_units = build_search_index(body)
    index_json, index_truncated = search_index_json(index_units)
    (SITE / "search-index.json").write_text(index_json, encoding="utf-8")

    min_js = minify_js(JS)
    js_hash = hashlib.sha256(min_js.encode("utf-8")).hexdigest()[:10]
    (SITE / "app.js").write_text(min_js, encoding="utf-8")

    page = assemble(body, build_nav(nav), meta, js_hash)
    out = SITE / "index.html"
    out.write_text(page, encoding="utf-8")

    print(f"  sections in nav: {len(nav)}")
    print(f"  MathML nodes:    {page.count('<math')}")
    print(f"  references:      {body.count('csl-entry')}")
    print(f"  fonts copied:    {len(list(fonts_dst.glob('*.woff2')))}")
    print(f"  live runtime:    {n_live} files")
    print(
        f"  search index:    {len(index_units)} units, "
        f"{len(index_json.encode('utf-8')):,} bytes"
        + (" (truncated to fit the byte budget)" if index_truncated else "")
    )
    print(f"  wrote {out.relative_to(REPO)} ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
