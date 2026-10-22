import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pytest

from dynachaos.config import DEFAULT_FIGURE_THEME
from dynachaos.utils.style import (
    CMAP_DIVERGING,
    CMAP_SEQUENTIAL,
    CMAP_SPACETIME,
    COLOR_CYCLE,
    COLORS,
    MARKER_CYCLE,
    apply_axes_polish,
    available_themes,
    color_for,
    figure_spec,
    finalize_legend,
    marker_for,
    panel_label,
    reference_line,
    save_theme_previews,
    series_style,
    setup,
    system_color,
    theme_description,
)


def test_swiss_style_contract():
    setup()

    assert "red" in COLORS
    assert "black" in COLORS
    assert len(COLOR_CYCLE) >= 6
    assert len(MARKER_CYCLE) == len(COLOR_CYCLE)

    assert matplotlib.rcParams["axes.grid"] is False
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["font.family"][0] == "serif"

    assert CMAP_DIVERGING
    assert CMAP_SEQUENTIAL
    assert CMAP_SPACETIME


def test_style_cycles_repeat_consistently():
    n = len(COLOR_CYCLE)
    for i in range(2 * n):
        assert color_for(i) == COLOR_CYCLE[i % n]
        assert marker_for(i) == MARKER_CYCLE[i % n]


def test_style_preview_render(tmp_path):
    setup()

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = [0, 1, 2, 3, 4]
    for idx in range(4):
        y = [v + idx for v in x]
        sty = series_style(idx)
        ax.plot(x, y, label=f"series {idx + 1}", **sty)

    ax.set_title("Swiss Style Preview")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    out = Path(tmp_path) / "swiss_style_preview.png"
    fig.savefig(out)
    plt.close(fig)

    assert out.exists()
    assert out.stat().st_size > 0


def test_theme_registry_contract():
    themes = available_themes()
    assert set(themes) == {
        "editorial-grid",
        "zurich-transit",
        "alpine-modern",
        "bauhaus-pop",
        "signature",
    }
    for theme in themes:
        assert theme_description(theme)
        setup(theme)
        assert color_for(0, theme=theme)
        assert marker_for(0, theme=theme)


def test_default_theme_is_signature():
    assert DEFAULT_FIGURE_THEME == "signature"


def test_env_override_changes_global_theme(monkeypatch):
    monkeypatch.setenv("DYNACHAOS_THEME", "zurich-transit")
    setup()
    assert color_for(0) == color_for(0, theme="zurich-transit")
    assert marker_for(0) == marker_for(0, theme="zurich-transit")
    setup("editorial-grid")


def test_save_theme_previews_renders_all(tmp_path):
    paths = save_theme_previews(tmp_path)
    assert len(paths) == len(available_themes())
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_figure_spec_contract():
    for kind in ("single", "double", "grid"):
        spec = figure_spec(kind)
        assert spec.kind == kind
        assert spec.figsize[0] > 0
        assert spec.figsize[1] > 0
        assert spec.label_size > 0
        assert spec.title_size > 0
        assert spec.tick_size > 0
        assert spec.legend_size > 0

    with pytest.raises(ValueError):
        figure_spec("poster")


def test_axes_polish_and_legend_helpers():
    setup()
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], label="series")
    ax.set_title("Test")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    spec = apply_axes_polish(ax, kind="single")
    legend = finalize_legend(ax, kind="single", loc="upper left")

    assert legend is not None
    assert legend.get_texts()
    assert ax.get_title(loc="left") == "Test"
    assert ax._left_title.get_fontsize() == spec.title_size
    assert ax.xaxis.get_label().get_fontsize() == spec.label_size
    plt.close(fig)


def test_apply_axes_polish_finds_title_set_at_a_different_loc():
    # The title was set at "left", but title_loc requests "right": the
    # requested location is empty, so apply_axes_polish falls back to
    # scanning left/center/right for whichever one actually holds text
    # (style.py:389-394).
    setup()
    fig, ax = plt.subplots()
    ax.set_title("Left title", loc="left")

    spec = apply_axes_polish(ax, kind="single", title_loc="right")

    assert ax._left_title.get_fontsize() == spec.title_size
    plt.close(fig)


def test_finalize_legend_returns_none_without_handles():
    setup()
    fig, ax = plt.subplots()

    legend = finalize_legend(ax, kind="single")

    assert legend is None
    plt.close(fig)


def test_system_color_rejects_unknown_system():
    with pytest.raises(KeyError, match="Unknown system"):
        system_color("not-a-real-system")


def test_theme_description_rejects_unknown_theme():
    with pytest.raises(ValueError, match="Unknown theme"):
        theme_description("not-a-real-theme")


def test_panel_label_falls_back_to_upper_left_for_unknown_loc():
    setup()
    fig, ax = plt.subplots()

    text = panel_label(ax, "(a)", loc="middle")

    assert text.get_ha() == "left"
    assert text.get_va() == "top"
    plt.close(fig)


def test_reference_line_annotates_vertical_line_label():
    setup()
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])

    line = reference_line(ax, 0.5, axis="x", label="onset")

    assert line is not None
    texts = [t.get_text() for t in ax.texts]
    assert " onset" in texts
    plt.close(fig)


def test_plot_modules_avoid_hardcoded_fontsize_numbers():
    repo = Path(__file__).resolve().parents[1]
    plot_dirs = [
        repo / "src/dynachaos/maps",
        repo / "src/dynachaos/cml",
        repo / "src/dynachaos/diagnostics",
    ]
    files: list[Path] = []
    for folder in plot_dirs:
        files.extend(sorted(folder.glob("*.py")))

    allowlist: dict[str, tuple[str, ...]] = {
        "compare_all.py": (
            "fontsize=spec.tick_size",
            "fontsize=spec.title_size",
        ),
    }

    pattern = re.compile(r"fontsize\s*=\s*[0-9]")
    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        stripped = text
        for token in allowlist.get(path.name, ()):
            stripped = stripped.replace(token, "")
        if pattern.search(stripped):
            offenders.append(str(path))

    assert not offenders, "Hardcoded numeric fontsize found in plotting modules:\n" + "\n".join(
        offenders
    )


def test_no_hardcoded_color_literals_outside_style_module():
    repo = Path(__file__).resolve().parents[1]
    py_files = sorted((repo / "src/dynachaos").rglob("*.py"))
    py_files = [p for p in py_files if p.name != "style.py"]

    patterns = [
        re.compile(
            r"\b(?:color|facecolor|edgecolor|markerfacecolor|markeredgecolor|cmap)\s*=\s*['\"][^'\"]+['\"]"
        ),
        re.compile(r"[,(\s]c\s*=\s*['\"][^'\"]+['\"]"),
        re.compile(r"['\"](?:k|b|r|w)[-.:]"),
        re.compile(r"#[0-9A-Fa-f]{3,8}"),
    ]

    offenders: list[str] = []
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        if any(p.search(text) for p in patterns):
            offenders.append(str(path))

    assert not offenders, "Hardcoded color literals detected outside style.py:\n" + "\n".join(
        offenders
    )
