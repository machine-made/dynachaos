"""Checks on the page assembly in ``scripts/build_paper.py``.

These exist because a whole class of defect here is invisible: the build
succeeds, the page renders, every link resolves -- and the text still names the
wrong figure. 52 references pointed at a figure one place off from the one they
named, and survived every build, because nothing compared the number in the
sentence against the number in the caption.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_paper.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_paper", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def _ref(target: str, text: str) -> str:
    return f'<a href="#{target}" data-reference-type="ref" data-reference="{target}">{text}</a>'


def test_figure_refs_take_the_number_the_caption_shows():
    module = _load()
    # The programme-arc figure is numbered last on the page but was numbered
    # first in the manuscript -- the exact shift that caused the original bug.
    body = (
        _ref("fig:arnold", "2")
        + _ref("fig:arc", "1")
        + '<figure data-fignum="1" id="fig:arnold"></figure>'
        + '<figure id="fig:arc" class="arc" data-fignum="37"></figure>'
    )

    out, changed, missing = module.renumber_figure_refs(body)

    assert changed == 2
    assert missing == 0
    assert _ref("fig:arnold", "1") in out
    assert _ref("fig:arc", "37") in out


def test_figure_refs_are_left_alone_when_they_already_agree():
    module = _load()
    body = _ref("fig:a", "4") + '<figure data-fignum="4" id="fig:a"></figure>'

    out, changed, missing = module.renumber_figure_refs(body)

    assert (changed, missing) == (0, 0)
    assert out == body


def test_a_reference_to_a_figure_that_is_not_on_the_page_is_reported():
    """Silently leaving it alone is right -- but it must be counted, so the
    build can say so rather than shipping a link to nothing."""
    module = _load()
    body = _ref("fig:absent", "9")

    out, changed, missing = module.renumber_figure_refs(body)

    assert (changed, missing) == (0, 1)
    assert out == body


def test_table_captions_get_the_same_number_the_ref_map_assigns():
    """Prose cites "Table N" via the numbers map ``number_tables`` returns; the
    caption must show the identical number or a reader sees a citation and a
    caption that disagree, the same bug the figure numbering test guards."""
    module = _load()
    body = (
        '<div id="tab:notation"></div><table id="tab:notation">'
        "<caption>Global notation.</caption></table>"
        '<div id="tab:atlas"></div><table id="tab:atlas">'
        "<caption>Mechanistic atlas of Kaneko-style systems.</caption></table>"
    )

    out, numbers = module.number_tables(body)

    assert numbers == {"tab:notation": "1", "tab:atlas": "2"}
    assert '<caption><span class="num">Table 1.</span> Global notation.' in out
    assert '<caption><span class="num">Table 2.</span> Mechanistic atlas' in out


def test_build_alt_text_strips_annotations_and_labels_and_cuts_at_a_sentence():
    """The alt builder must not speak MathML twice, must not leak pandoc's
    unresolved ``[eq:...]`` placeholders, and must never cut mid-word."""
    module = _load()
    caption = (
        "Lyapunov spectrum of the delayed logistic map "
        '<a href="#eq:delayed_logistic" data-reference-type="eqref" '
        'data-reference="eq:delayed_logistic">[eq:delayed_logistic]</a> at '
        '<math display="inline"><semantics><mrow><mi>&alpha;</mi><mo>=</mo>'
        '<mn>0.3</mn></mrow><annotation encoding="application/x-tex">'
        "\\alpha = 0.3</annotation></semantics></math> versus D. "
        "The vertical dashed line marks the onset of chaos in the sweep, "
        "well past the two hundred character mark this sentence is padded "
        "out to reach so the cutoff logic actually has to make a choice "
        "here between keeping the first sentence or the second one."
    )

    alt = module.build_alt_text(caption)

    assert "[eq:" not in alt
    assert "\\alpha" not in alt
    assert "annotation" not in alt
    # the doubled math text is gone; only the rendered symbol remains once
    assert alt.count("0.3") <= 1
    assert alt.endswith((".", "…", "!", "?"))
    assert not alt.endswith(("li", "trac", "wedg"))  # no mid-word cuts
    assert len(alt) <= 203  # ~200 chars plus a little slack for the ellipsis


def test_build_alt_text_keeps_a_short_caption_whole():
    module = _load()
    caption = "Devil's staircase of the circle map."

    alt = module.build_alt_text(caption)

    assert alt == "Devil's staircase of the circle map."


def test_search_index_covers_folded_details_and_strips_annotations():
    """The whole point of the index: a browser's own Ctrl+F cannot see inside
    a closed <details>, so the build-time index must -- and it must not speak
    an equation's raw-LaTeX annotation as if it were prose."""
    module = _load()
    body = (
        '<section id="sec:results"><h2>Results</h2>'
        '<details class="backmatter"><summary><h2>Provenance</h2></summary>'
        "<p>Every run is recomputed at "
        '<math display="inline"><semantics><mrow><mi>&alpha;</mi><mo>=</mo>'
        '<mn>0.3</mn></mrow><annotation encoding="application/x-tex">'
        "\\alpha = 0.3</annotation></semantics></math> from fresh seeds.</p>"
        "</details></section>"
    )

    units = module.build_search_index(body)
    para = next(u for u in units if "Every run" in u["text"])

    assert "\\alpha" not in para["text"]
    assert para["text"].count("0.3") == 1
    # the id resolves to the ancestor section's real anchor, not a made-up one
    assert para["id"] == "sec:results"
    # indexed under the nearest heading actually seen, inside the details
    assert para["section"] == "Provenance"
    headings = {u["text"] for u in units if u["tag"] in ("h2", "h3", "h4")}
    assert headings == {"Results", "Provenance"}


def test_search_index_json_is_embeddable_in_a_script_tag():
    module = _load()
    units = [{"id": "sec:a", "section": "A", "tag": "p", "text": "contains </script> literally"}]

    raw, truncated = module.search_index_json(units)
    parsed = json.loads(raw.replace("<\\/", "</"))

    assert truncated is False
    assert "</script" not in raw
    assert parsed[0]["text"] == "contains </script> literally"


def test_unnumbered_display_equations_get_an_anchor_but_no_number():
    """Only starred display maths reaches this step, and LaTeX gives those no
    number. The equations LaTeX does number are anchored earlier, in
    anchor_labels, whether or not the manuscript labelled them."""
    module = _load()
    numbered = '<div class="eqn"><math display="block">a</math><span class="eqno">(1)</span></div>'
    body = numbered + '<p>text</p><math display="block">b</math>'

    out, count = module.anchor_unnumbered_equations(body)

    assert count == 1
    assert 'id="eq-u1"' in out
    # exactly one visible number, still the one that was already there
    assert len(re.findall(r'class="eqno"', out)) == 1
    assert numbered in out


def test_the_numbering_note_lands_before_the_reproduction_index():
    """The page keeps the manuscript's figure numbers; shipping without the note
    that says so would leave a cited number unexplained, so its absence must
    fail the build rather than pass silently."""
    module = _load()
    body = '<table id="tab:repro_index"><caption>Reproduction index.</caption></table>'

    out, _ = module.fold_back_matter(body)

    note_at = out.index("Figure numbers on this page")
    assert note_at < out.index('<table id="tab:repro_index"')
    assert "the manuscript's own" in out


def test_a_missing_reproduction_index_fails_the_build():
    import pytest

    module = _load()
    with pytest.raises(SystemExit):
        module.fold_back_matter("<p>no table here</p>")


def _hero_body() -> str:
    return "<figure><figcaption>Circle map staircase.</figcaption></figure>Diagnostic spotlight"


def _hero_html(module) -> str:
    return module.hero(
        {"title": "T", "authors": "A", "affil": "F", "lede": "L"},
        _hero_body(),
    )


def test_the_hero_legend_stays_a_caption_and_not_a_control_panel():
    """Check the legend keeps its plain caption and grows no buttons."""
    html = _hero_html(_load())

    assert "logistic attractor, computed live in your browser" in html
    assert "<button" not in html
    for gone in ("hero-shock", "hero-hud", "hero-actions", "hero-cobweb"):
        assert gone not in html


def test_the_hero_text_block_keeps_its_own_typography():
    """Check the hero type rules survive; losing them drops the text to body style."""
    css = re.sub(r"\s+", "", _load().CSS)

    # .lede stays larger than body copy, .stats stays a row of figures rather
    # than a bulleted list, and the entrance animation the JS asks for exists.
    assert ".lede{" in css
    lede = css.split(".lede{", 1)[1].split("}", 1)[0]
    assert "font-size:clamp(" in lede and "max-width:48ch" in lede

    assert ".byline{" in css and ".byline.affil{" in css
    stats = css.split(".stats{", 1)[1].split("}", 1)[0]
    assert "display:flex" in stats and "list-style:none" in stats
    assert ".statsb{" in css and ".statsspan{" in css
    assert ".hero-rise{" in css and "@keyframeshero-rise{" in css


def test_hero_offscreen_pause_does_not_cancel_the_frame_loop():
    """Check leaving the plate clears the flags and lets the loop end by itself."""
    js = _load().JS
    match = re.search(r"new IntersectionObserver\((.*?)\)\.observe", js, re.S)
    assert match, "IntersectionObserver is missing"
    assert "cancelAnimationFrame" not in match.group(1)


def test_every_unstarred_display_equation_is_anchored_even_without_a_label():
    """LaTeX numbers an equation whether or not the manuscript labelled it.
    Anchoring only the labelled ones made the page count over a shorter list,
    so its numbers slid below the manuscript's: the coupled-map lattice showed
    (11) here against (13) in the paper."""
    module = _load()
    tex = (
        "\\begin{equation}a\\label{eq:one}\\end{equation}"
        "\\begin{equation}b\\end{equation}"
        "\\begin{equation}c\\label{eq:three}\\end{equation}"
    )

    out, count = module.anchor_labels(tex)

    assert count == 3
    assert "\\hypertarget{eq:one}" in out
    assert "\\hypertarget{eq:unlabelled-1}" in out
    assert "\\hypertarget{eq:three}" in out


def test_a_starred_equation_is_never_anchored():
    """A starred environment shows no number. Anchoring it would let
    ``number_equations`` give it one the manuscript does not have."""
    module = _load()
    tex = "\\begin{equation*}a\\end{equation*}"

    out, count = module.anchor_labels(tex)

    assert count == 0
    assert "\\hypertarget" not in out


def test_the_multline_environment_is_anchored():
    """The co-moving Lyapunov equation is a multline. It was missing from the
    environment list, so it reached the page with no anchor and no number."""
    module = _load()
    tex = "\\begin{multline}a\\label{eq:comoving}\\end{multline}"

    out, count = module.anchor_labels(tex)

    assert count == 1
    assert "\\hypertarget{eq:comoving}" in out


def test_tables_inside_an_appendix_are_lettered():
    """LaTeX restarts numbering at ``\\appendix``, so the provenance table is
    A1, not 3. The page called it 3 and the prose cited 3, agreeing with each
    other and with nothing in the paper."""
    module = _load()
    body = (
        '<div id="tab:notation"></div><table id="tab:notation">'
        "<caption>Global notation.</caption></table>"
        '<section id="app:provenance" class="level1">'
        '<div id="tab:provenance"></div><table id="tab:provenance">'
        "<caption>Equation provenance.</caption></table></section>"
        '<section id="app:repro_index" class="level1">'
        '<div id="tab:repro_index"></div><table id="tab:repro_index">'
        "<caption>Reproduction index.</caption></table></section>"
    )

    out, numbers = module.number_tables(body)

    assert numbers == {"tab:notation": "1", "tab:provenance": "A1", "tab:repro_index": "B1"}
    assert '<caption><span class="num">Table A1.</span> Equation provenance.' in out


def test_appendix_letters_follow_document_order():
    """The letter comes from the position of the appendix, so a new appendix
    inserted between two others takes its letter from where it sits."""
    module = _load()
    body = (
        '<section id="app:first" class="level1"></section>'
        '<section id="sec:body" class="level1"></section>'
        '<section id="app:second" class="level1"></section>'
    )

    found = module.appendix_letters(body)

    assert [(sec_id, letter) for _, sec_id, letter in found] == [
        ("app:first", "A"),
        ("app:second", "B"),
    ]


def test_arnold_tongues_is_live_even_without_chart_json(tmp_path, monkeypatch):
    """The live figure must offer interact without a JSON export; otherwise a
    checkout that has not run export_figure_data.py ships a static PNG only."""
    module = _load()
    monkeypatch.setattr(module, "SITE", tmp_path)
    html = module.figure_block(
        "fig:arnold_tongues", "sec02_circle_map", "arnold_tongues", "Arnold tongues.", {}
    )
    assert 'id="fig:arnold_tongues"' in html
    assert 'data-live="arnold_tongues"' in html
    assert 'class="act-interact"' in html
    assert "data-src" not in html
    assert 'data-state="static"' in html


def test_arnold_tongues_keeps_json_src_as_a_reduced_data_fallback(tmp_path, monkeypatch):
    """prefers-reduced-data falls back to the JSON chart when it exists, so the
    live figure still carries data-src; it must not grow a second interact button."""
    module = _load()
    monkeypatch.setattr(module, "SITE", tmp_path)
    data = tmp_path / "data" / "sec02_circle_map"
    data.mkdir(parents=True)
    (data / "arnold_tongues.json").write_text("{}")
    html = module.figure_block(
        "fig:arnold_tongues", "sec02_circle_map", "arnold_tongues", "Arnold tongues.", {}
    )
    assert 'data-live="arnold_tongues"' in html
    assert 'data-src="data/sec02_circle_map/arnold_tongues.json"' in html
    assert html.count("act-interact") == 1


def test_a_json_figure_that_is_not_live_is_unchanged(tmp_path, monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "SITE", tmp_path)
    data = tmp_path / "data" / "sec02_circle_map"
    data.mkdir(parents=True)
    (data / "devil_staircase.json").write_text("{}")
    html = module.figure_block(
        "fig:devil_staircase", "sec02_circle_map", "devil_staircase", "Devil's staircase.", {}
    )
    assert "data-live" not in html
    assert 'data-src="data/sec02_circle_map/devil_staircase.json"' in html
    assert 'class="act-interact"' in html


def test_a_static_figure_still_has_no_interact_button(tmp_path, monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "SITE", tmp_path)
    html = module.figure_block(
        "fig:attractors", "sec03_transition", "attractors", "Attractors.", {}
    )
    assert "data-live" not in html
    assert "data-src" not in html
    assert "act-interact" not in html
    assert 'class="act-zoom"' in html


def test_the_assembled_page_numbers_everything_the_way_the_manuscript_does():
    """The whole-page check the unit tests above cannot make.

    Each unit test holds one helper to its contract. None of them can see the
    assembled page, which is where the defect actually appeared: a sentence
    naming one figure linked to the figure captioned with the next number, and
    every build stayed green.

    The three assertions live in one test on purpose. The page is built here,
    into the ignored ``site/`` directory that CI writes anyway, and a separate
    test reading that file would race the build under ``pytest -n auto``: a
    clean checkout has no ``site/index.html`` at all.
    """
    build = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True)
    assert build.returncode == 0, f"the build failed:\n{build.stderr}"
    page = (ROOT / "site" / "index.html").read_text(encoding="utf-8")

    shown: dict[str, str] = {}
    for tag in re.findall(r"<figure[^>]*>", page):
        num = re.search(r'data-fignum="(\d+)"', tag)
        fid = re.search(r'id="(fig:[^"]+)"', tag)
        if num and fid:
            shown[fid.group(1)] = num.group(1)
    assert len(shown) == 37, f"expected 37 numbered figures, found {len(shown)}"

    # Every figure reference must name the number its figure displays.
    disagreements = [
        (target, text, shown[target])
        for target, text in re.findall(
            r'href="#(fig:[^"]+)"[^>]*data-reference-type="ref"[^>]*>([^<]{1,8})</a>', page
        )
        if target in shown and shown[target] != text.strip()
    ]
    assert not disagreements, f"reference text disagrees with the figure it names: {disagreements}"

    # The programme arc is shown last for design reasons, but the manuscript
    # calls it Figure 1. Numbering it by page position slid every other figure
    # one below the paper.
    assert shown.get("fig:program_map") == "1"

    # LaTeX restarts numbering at \appendix. A reader told to see Appendix B
    # must not find a section called 12.
    for sec_id, letter in (("app:provenance", "A"), ("app:repro_index", "B")):
        head = page.index(f'id="{sec_id}"')
        heading = page[head : page.index("</h", head)]
        assert f'<span class="secno">{letter}</span>' in heading, sec_id


def _body_text() -> str:
    """Return the cached manuscript body with tags stripped and U+00A0 folded to a space."""
    raw = (ROOT / "web" / "paper-body.html").read_text(encoding="utf-8")
    raw = re.sub(r"<math.*?</math>", "[m]", raw, flags=re.S)
    return re.sub(r"<[^>]+>", "", raw).replace(" ", " ")


def test_the_body_has_no_em_dash_and_no_doubled_author_citation():
    """Check the two prose defects of the referee pass stay fixed."""
    raw = (ROOT / "web" / "paper-body.html").read_text(encoding="utf-8")
    assert "—" not in raw
    doubled = re.findall(r"([A-Z][a-z]+)[  ]<span class=\"citation\"[^>]*>\(<a [^>]*>\1 \d{4}", raw)
    assert doubled == []


def test_the_body_has_no_plain_text_section_reference():
    """Check every 'Section N' is a cross-reference link the builder renumbers."""
    raw = (ROOT / "web" / "paper-body.html").read_text(encoding="utf-8")
    assert re.findall(r"Section[  ]\d+(?:\.\d+)?", raw) == []


def test_the_body_has_no_banned_phrase():
    """Check the phrases the referee pass removed do not come back."""
    text = _body_text()
    for phrase in (
        "honest negative",
        "honest non-power",
        "striking",
        "remarkable",
        "ground-breaking",
        "substrate",
        "flagship",
        "fundamental paradigm",
        "In summary",
        "Taken together",
        "deep links",
        "remains elusive",
        "stated sharply",
    ):
        assert phrase not in text, phrase


def test_minify_keeps_the_space_before_an_inline_tag_only():
    """Check minification glues block tags but not an italic term to its citation."""
    minify = _load().minify_html
    assert (
        minify('</em> <span class="citation">(x)</span>')
        == '</em> <span class="citation">(x)</span>'
    )
    assert minify("<li>a</li>\n<li>b</li>") == "<li>a</li><li>b</li>"
    assert minify("</span> <math><mi>x</mi></math>") == "</span> <math><mi>x</mi></math>"


def test_the_lede_comes_from_the_sitelede_macro_when_the_manuscript_defines_one():
    """Check the page lede is the hand-written macro, not a sentence cut from the abstract."""
    extract_meta = _load().extract_meta
    tex = (
        "\\newcommand{\\sitelede}{Kaneko\u2019s papers, recomputed.}\n"
        "\\begin{abstract}\nOld work. We investigate whether it holds.\n\\end{abstract}\n"
    )
    assert extract_meta(tex)["lede"] == "Kaneko\u2019s papers, recomputed."
    assert extract_meta(tex.split("\n", 1)[1])["lede"] == "We investigate whether it holds."
