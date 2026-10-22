"""Swiss-inspired plotting style system for dynachaos.

Design principles encoded from International Typographic / Swiss Style:
- modular structure and visible hierarchy
- asymmetry and left-weighted composition
- strong sans-serif typography
- disciplined use of whitespace
- restrained but intentional color accents
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dynachaos.config import (
    DEFAULT_FIGURE_THEME,
    FIGURE_THEME_ENV_VAR,
    get_figure_theme,
)

# Column widths in inches for figure sizing
SINGLE_COL = 3.4
DOUBLE_COL = 7.0


@dataclass(frozen=True)
class ThemeSpec:
    """Visual design token bundle for one plotting theme."""

    name: str
    description: str
    colors: dict[str, str]
    color_cycle: tuple[str, ...]
    marker_cycle: tuple[str, ...]
    cmap_diverging: str
    cmap_sequential: str
    cmap_spacetime: str
    font_family: str = "sans-serif"
    font_list: tuple[str, ...] = (
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "Nimbus Sans",
        "DejaVu Sans",
    )
    mathtext_fontset: str = "dejavusans"


@dataclass(frozen=True)
class FigureSpec:
    """Shared size/typography tokens for a figure layout class."""

    kind: str
    figsize: tuple[float, float]
    label_size: float
    title_size: float
    tick_size: float
    legend_size: float
    title_pad: float


_THEMES: dict[str, ThemeSpec] = {
    "editorial-grid": ThemeSpec(
        name="editorial-grid",
        description=(
            "Classic Swiss editorial voice: neutral field, black-first hierarchy, "
            "red signal accent."
        ),
        colors={
            "black": "#111111",
            "offwhite": "#FFFFFF",
            "grid": "#D9D9D2",
            "red": "#E10600",
            "blue": "#0057B8",
            "green": "#008A5C",
            "orange": "#C96A00",
            "purple": "#5E4FA2",
            "brown": "#7A4B2A",
            "pink": "#C73765",
            "grey": "#6E6E6E",
            "yellow": "#C9A227",
            "cyan": "#007C91",
        },
        color_cycle=(
            "#111111",
            "#E10600",
            "#0057B8",
            "#008A5C",
            "#C96A00",
            "#6E6E6E",
            "#7A4B2A",
            "#007C91",
        ),
        marker_cycle=("o", "s", "^", "D", "v", "P", "X", "*"),
        cmap_diverging="RdBu_r",
        cmap_sequential="cividis",
        cmap_spacetime="magma",
    ),
    "zurich-transit": ThemeSpec(
        name="zurich-transit",
        description=(
            "Wayfinding-inspired Swiss transit look: high legibility, stronger chromatic "
            "coding, crisp geometry."
        ),
        colors={
            "black": "#0F1419",
            "offwhite": "#FFFFFF",
            "grid": "#D6DDE3",
            "red": "#D90429",
            "blue": "#1D4ED8",
            "green": "#0F766E",
            "orange": "#EA580C",
            "purple": "#6D28D9",
            "brown": "#7C2D12",
            "pink": "#BE185D",
            "grey": "#64748B",
            "yellow": "#CA8A04",
            "cyan": "#0E7490",
        },
        color_cycle=(
            "#0F1419",
            "#1D4ED8",
            "#D90429",
            "#0F766E",
            "#EA580C",
            "#6D28D9",
            "#64748B",
            "#0E7490",
        ),
        marker_cycle=("s", "o", "^", "D", "v", "X", "P", "*"),
        cmap_diverging="coolwarm",
        cmap_sequential="viridis",
        cmap_spacetime="inferno",
    ),
    "alpine-modern": ThemeSpec(
        name="alpine-modern",
        description=(
            "Swiss landscape abstraction: glacier blues + sunrise warm accents with generous "
            "negative space."
        ),
        colors={
            "black": "#1F2933",
            "offwhite": "#FFFFFF",
            "grid": "#DDD8CC",
            "red": "#E76F51",
            "blue": "#264653",
            "green": "#2A9D8F",
            "orange": "#F4A261",
            "purple": "#5B4B8A",
            "brown": "#9C6644",
            "pink": "#B83B5E",
            "grey": "#7D8597",
            "yellow": "#E9C46A",
            "cyan": "#3A86A8",
        },
        color_cycle=(
            "#1F2933",
            "#264653",
            "#2A9D8F",
            "#E76F51",
            "#F4A261",
            "#5B4B8A",
            "#7D8597",
            "#3A86A8",
        ),
        marker_cycle=("^", "o", "s", "D", "v", "P", "X", "*"),
        cmap_diverging="PuOr",
        cmap_sequential="plasma",
        cmap_spacetime="magma",
    ),
    "bauhaus-pop": ThemeSpec(
        name="bauhaus-pop",
        description=(
            "Constructivist/Bauhaus echo with bold primary accents and punchy poster contrast."
        ),
        colors={
            "black": "#101010",
            "offwhite": "#FFFFFF",
            "grid": "#D4D0C6",
            "red": "#D62828",
            "blue": "#003049",
            "green": "#2A9D8F",
            "orange": "#F77F00",
            "purple": "#7B2CBF",
            "brown": "#8D5524",
            "pink": "#D81B60",
            "grey": "#5C5C5C",
            "yellow": "#FCBF49",
            "cyan": "#0096C7",
        },
        color_cycle=(
            "#101010",
            "#D62828",
            "#003049",
            "#F77F00",
            "#FCBF49",
            "#2A9D8F",
            "#7B2CBF",
            "#0096C7",
        ),
        marker_cycle=("D", "s", "o", "^", "v", "P", "X", "*"),
        cmap_diverging="seismic",
        cmap_sequential="turbo",
        cmap_spacetime="rocket",
    ),
    "signature": ThemeSpec(
        name="signature",
        description=(
            "dynachaos signature: cool slate ink, single vermilion signal accent, "
            "CB-safe per-system colours, Computer Modern paper font."
        ),
        colors={
            "black": "#2E3440",
            "offwhite": "#FFFFFF",
            "grid": "#E3E7EC",
            "red": "#D1495B",
            "blue": "#3D6FB4",
            "green": "#2A9D8F",
            "orange": "#E9A23B",
            "purple": "#6A4C93",
            "brown": "#8D5A3C",
            "pink": "#C73765",
            "grey": "#AEB6C2",
            "yellow": "#E9C46A",
            "cyan": "#138086",
        },
        color_cycle=(
            "#2E3440",
            "#D1495B",
            "#3D6FB4",
            "#2A9D8F",
            "#E9A23B",
            "#6A4C93",
            "#138086",
            "#AEB6C2",
        ),
        marker_cycle=("o", "s", "^", "D", "v", "P", "X", "*"),
        cmap_diverging="RdBu_r",
        cmap_sequential="viridis",
        cmap_spacetime="viridis",
        font_family="serif",
        font_list=("Latin Modern Roman", "CMU Serif", "DejaVu Serif"),
        mathtext_fontset="cm",
    ),
}

DEFAULT_THEME = DEFAULT_FIGURE_THEME

_FIGURE_SPECS: dict[str, FigureSpec] = {
    "single": FigureSpec(
        kind="single",
        figsize=(SINGLE_COL, 2.65),
        label_size=8.8,
        title_size=8.8,
        tick_size=7.8,
        legend_size=6.8,
        title_pad=3.2,
    ),
    "double": FigureSpec(
        kind="double",
        figsize=(DOUBLE_COL, 3.1),
        label_size=9.0,
        title_size=9.1,
        tick_size=7.9,
        legend_size=6.9,
        title_pad=3.3,
    ),
    "grid": FigureSpec(
        kind="grid",
        figsize=(DOUBLE_COL, 4.55),
        label_size=8.4,
        title_size=8.5,
        tick_size=7.2,
        legend_size=6.5,
        title_pad=3.0,
    ),
}


# Mutable exports kept for backward compatibility with existing figure modules.
COLORS: dict[str, str] = {}
COLOR_CYCLE: list[str] = []
MARKER_CYCLE: list[str] = []
CMAP_DIVERGING = "RdBu_r"
CMAP_SEQUENTIAL = "cividis"
CMAP_SPACETIME = "magma"


# Semantic color mappings for common system types
SYSTEM_COLORS = {
    "logistic": "#3D6FB4",
    "delayed_logistic": "#D1495B",
    "circle": "#2A9D8F",
    "coupled_logistic": "#E9A23B",
    "gcm": "#6A4C93",
    "cml": "#138086",
}
LYAP_COLORS = ("#2E3440", "#D1495B", "#3D6FB4")  # lambda_1, lambda_2, lambda_3


def system_color(name: str) -> str:
    """Return canonical color for a dynamical system by name.

    Parameters
    ----------
    name : str
        System type: 'logistic', 'delayed_logistic', 'circle', 'coupled_logistic', 'gcm',
        'cml'.

    Returns
    -------
    str
        Hex color string.

    Raises
    ------
    KeyError
        If name is not in SYSTEM_COLORS.
    """
    if name not in SYSTEM_COLORS:
        valid = ", ".join(sorted(SYSTEM_COLORS.keys()))
        raise KeyError(f"Unknown system '{name}'. Valid systems: {valid}")
    return SYSTEM_COLORS[name]


def lyap_color(index: int) -> str:
    """Return canonical color for a Lyapunov exponent by index.

    Parameters
    ----------
    index : int
        Exponent index (0, 1, 2, ...). Wraps cyclically.

    Returns
    -------
    str
        Hex color string.
    """
    return LYAP_COLORS[index % len(LYAP_COLORS)]


def available_themes() -> tuple[str, ...]:
    """Return available Swiss-style theme IDs."""
    return tuple(_THEMES.keys())


def theme_description(theme: str) -> str:
    """Return one-line description for a theme."""
    return _resolve_theme(theme).description


def figure_spec(kind: str = "double") -> FigureSpec:
    """Return a shared figure layout/typography specification."""
    if kind not in _FIGURE_SPECS:
        valid = ", ".join(sorted(_FIGURE_SPECS))
        raise ValueError(f"Unknown figure spec '{kind}'. Available specs: {valid}")
    return _FIGURE_SPECS[kind]


def apply_axes_polish(
    ax,
    *,
    kind: str = "double",
    title_loc: str | None = None,
    grid: bool | None = None,
    equal: bool = False,
) -> FigureSpec:
    """Apply consistent typography polish to an axis."""
    spec = figure_spec(kind)

    ax.xaxis.label.set_size(spec.label_size)
    ax.yaxis.label.set_size(spec.label_size)
    ax.tick_params(axis="both", which="both", labelsize=spec.tick_size)
    if grid is not None:
        ax.grid(grid)
    if equal:
        ax.set_aspect("equal", adjustable="box")

    active_loc = (
        mpl.rcParams.get("axes.titlelocation", "center") if title_loc is None else title_loc
    )
    title_text = ax.get_title(loc=active_loc)
    if not title_text:
        for loc in ("left", "center", "right"):
            title_text = ax.get_title(loc=loc)
            if title_text:
                active_loc = loc
                break

    if title_text:
        kwargs: dict[str, object] = {
            "fontsize": spec.title_size,
            "pad": spec.title_pad,
        }
        kwargs["loc"] = active_loc
        ax.set_title(title_text, **kwargs)

    return spec


def finalize_legend(ax, *, kind: str = "double", **kwargs):
    """Create a legend with class-consistent defaults when handles exist."""
    handles, _labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    spec = figure_spec(kind)
    defaults = {
        "fontsize": spec.legend_size,
        "frameon": False,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def _resolve_theme(theme: str | None) -> ThemeSpec:
    theme_id = get_figure_theme() if theme is None else theme
    if theme_id not in _THEMES:
        valid = ", ".join(available_themes())
        raise ValueError(
            f"Unknown theme '{theme_id}'. Available themes: {valid}. "
            f"Set {FIGURE_THEME_ENV_VAR} or dynachaos.config.DEFAULT_FIGURE_THEME."
        )
    return _THEMES[theme_id]


def _set_theme_exports(spec: ThemeSpec) -> None:
    """Update backward-compatible module globals to a theme spec."""
    COLORS.clear()
    COLORS.update(spec.colors)

    COLOR_CYCLE.clear()
    COLOR_CYCLE.extend(spec.color_cycle)

    MARKER_CYCLE.clear()
    MARKER_CYCLE.extend(spec.marker_cycle)

    global CMAP_DIVERGING, CMAP_SEQUENTIAL, CMAP_SPACETIME
    CMAP_DIVERGING = spec.cmap_diverging
    CMAP_SEQUENTIAL = spec.cmap_sequential
    CMAP_SPACETIME = spec.cmap_spacetime


def marker_for(index: int, theme: str | None = None) -> str:
    """Return canonical marker for a series index."""
    spec = _resolve_theme(theme)
    return spec.marker_cycle[index % len(spec.marker_cycle)]


def color_for(index: int, theme: str | None = None) -> str:
    """Return canonical color for a series index."""
    spec = _resolve_theme(theme)
    return spec.color_cycle[index % len(spec.color_cycle)]


def series_style(index: int, theme: str | None = None) -> dict[str, object]:
    """Return canonical series kwargs (color + marker + line weight)."""
    spec = _resolve_theme(theme)
    return {
        "color": spec.color_cycle[index % len(spec.color_cycle)],
        "marker": spec.marker_cycle[index % len(spec.marker_cycle)],
        "markersize": 4.2,
        "markerfacecolor": spec.colors["offwhite"],
        "markeredgewidth": 0.9,
        "linewidth": 1.4,
    }


def setup(theme: str | None = None) -> None:
    """Apply Swiss-style rcParams globally.

    Parameters
    ----------
    theme : str | None
        Theme ID from ``available_themes()``. ``None`` uses default theme.
    """
    spec = _resolve_theme(theme)
    _set_theme_exports(spec)

    plt.style.use("default")

    params = {
        # Canvas and export
        "figure.figsize": (7, 5),
        "figure.dpi": 150,
        "figure.facecolor": spec.colors["offwhite"],
        "axes.facecolor": spec.colors["offwhite"],
        "savefig.facecolor": spec.colors["offwhite"],
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Typography (left hierarchy)
        "font.family": spec.font_family,
        "font.size": 9.2,
        "mathtext.fontset": spec.mathtext_fontset,
        # Axes / structure
        "axes.linewidth": 1.0,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.edgecolor": spec.colors["black"],
        "axes.labelcolor": spec.colors["black"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=spec.color_cycle),
        # Grid lines (modular rhythm)
        "grid.color": spec.colors["grid"],
        "grid.linewidth": 0.45,
        "grid.alpha": 0.55,
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "xtick.color": spec.colors["black"],
        "ytick.color": spec.colors["black"],
        # Lines and markers
        "lines.linewidth": 1.2,
        "lines.markersize": 3.8,
        "lines.markeredgewidth": 0.9,
        # Text / legend
        "text.color": spec.colors["black"],
        "legend.fontsize": 9,
        "legend.frameon": False,
    }
    if spec.font_family == "serif":
        params["font.serif"] = list(spec.font_list)
    else:
        params["font.sans-serif"] = list(spec.font_list)
    # Ensure fallback for the other family
    params.setdefault("font.serif", ["CMU Serif", "DejaVu Serif"])
    params.setdefault(
        "font.sans-serif",
        ["Helvetica Neue", "Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
    )
    mpl.rcParams.update(params)


def render_theme_preview(theme: str, output_path: Path) -> Path:
    """Render one asymmetrical Swiss-layout style preview image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup(theme)

    x = np.linspace(0.0, 8.0, 280)
    fig = plt.figure(figsize=(8.0, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.0, 1.0], hspace=0.25, wspace=0.32)

    ax_left = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1:])
    ax_bottom = fig.add_subplot(gs[1, 1:])

    # Left panel: marker+line hierarchy
    for i in range(5):
        y = np.sin(x * (0.45 + 0.12 * i)) * (1.0 - i * 0.08) + i * 0.45
        sty = series_style(i, theme=theme)
        ax_left.plot(x, y, markevery=26, label=f"S{i + 1}", **sty)
    ax_left.set_title("Series Grammar", loc="left")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")
    ax_left.legend(ncol=3, loc="upper right")

    # Top-right: dense scatter accent
    rng = np.random.default_rng(42)
    px = rng.normal(0.0, 1.0, 1700)
    py = 0.75 * px + rng.normal(0.0, 0.55, 1700)
    ax_top.scatter(px, py, s=8, c=color_for(1, theme=theme), alpha=0.38, linewidths=0)
    ax_top.set_title("Accent + Neutral Field", loc="left")
    ax_top.set_xlabel("feature A")
    ax_top.set_ylabel("feature B")

    # Bottom-right: colormap sweep check
    grid_x = np.linspace(-2.0, 2.0, 140)
    grid_y = np.linspace(-2.0, 2.0, 90)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = np.sin(X * 1.3) * np.cos(Y * 0.9)
    ax_bottom.imshow(Z, cmap=_resolve_theme(theme).cmap_sequential, aspect="auto", origin="lower")
    ax_bottom.set_title("Field Colormap", loc="left")
    ax_bottom.set_xlabel("i")
    ax_bottom.set_ylabel("j")

    fig.suptitle(
        f"dynachaos Swiss Theme: {theme}",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_theme_previews(output_dir: Path | str, themes: Iterable[str] | None = None) -> list[Path]:
    """Render preview images for all or selected themes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chosen = list(available_themes()) if themes is None else list(themes)
    paths: list[Path] = []
    for theme in chosen:
        spec = _resolve_theme(theme)
        filename = f"{spec.name}.png"
        paths.append(render_theme_preview(spec.name, output_dir / filename))
    return paths


def panel_label(ax, letter, *, loc="upper left", **kwargs):
    """Draw a bold panel tag like "(a)" in the given axis corner.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    letter : str
        Label text, typically a single letter or short string.
    loc : str, default "upper left"
        Axes-fraction location: 'upper left', 'upper right', 'lower left', 'lower right'.
    **kwargs
        Forwarded to ax.text() (e.g., fontsize, color, fontweight).

    Returns
    -------
    matplotlib.text.Text
        The text object.
    """
    # Inset slightly from the corner so an "upper" label clears a left-aligned
    # title (which sits just above the axes) instead of colliding with it.
    loc_map = {
        "upper left": (0.035, 0.92),
        "upper right": (0.965, 0.92),
        "lower left": (0.035, 0.08),
        "lower right": (0.965, 0.08),
    }
    if loc not in loc_map:
        loc = "upper left"
    xy = loc_map[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"

    defaults = {
        "fontsize": figure_spec().title_size,
        "fontweight": "bold",
        "ha": ha,
        "va": va,
        "transform": ax.transAxes,
        # Subtle offwhite backing keeps the tag legible over data/fields.
        "bbox": {
            "facecolor": COLORS["offwhite"],
            "alpha": 0.6,
            "pad": 1.2,
            "edgecolor": "none",
        },
    }
    defaults.update(kwargs)
    return ax.text(xy[0], xy[1], letter, **defaults)


def reference_line(ax, value=0.0, *, axis="y", label=None, **kwargs):
    """Draw a grey dashed reference line on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    value : float, default 0.0
        Position of the line in data coordinates.
    axis : str, default "y"
        Axis to draw on: "x" (axvline) or "y" (axhline).
    label : str | None
        Optional inline annotation text.
    **kwargs
        Forwarded to axvline/axhline.

    Returns
    -------
    matplotlib.lines.Line2D
        The line object.
    """
    defaults = {
        "color": COLORS["grey"],
        "ls": "--",
        "lw": 1.0,
        "alpha": 0.7,
    }
    defaults.update(kwargs)

    if axis.lower() == "x":
        line = ax.axvline(value, **defaults)
    else:
        line = ax.axhline(value, **defaults)

    if label:
        if axis.lower() == "x":
            ax.text(value, ax.get_ylim()[1], f" {label}", fontsize=6, va="top", ha="left")
        else:
            ax.text(
                ax.get_xlim()[1],
                value,
                f" {label}",
                fontsize=6,
                va="bottom",
                ha="right",
            )

    return line


def add_field_colorbar(fig, mappable, ax, *, label=None, **kwargs):
    """Add a consistently-sized colorbar with small tick labels.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    mappable : matplotlib.cm.ScalarMappable
        Scalar mappable (usually from an imshow/contourf).
    ax : matplotlib.axes.Axes
        Reference axis for colorbar placement.
    label : str | None
        Colorbar label.
    **kwargs
        Forwarded to fig.colorbar().

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar object.
    """
    defaults = {
        "ax": ax,
        "fraction": 0.046,
        "pad": 0.03,
    }
    defaults.update(kwargs)
    cbar = fig.colorbar(mappable, **defaults)
    cbar.ax.tick_params(labelsize=6)
    if label:
        cbar.set_label(label, fontsize=7)
    return cbar


def annotate_on_field(ax, x, y, text, **kwargs):
    """Draw semi-opaque text on a field (e.g., image) for legibility.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    x, y : float
        Data coordinates.
    text : str
        Text content.
    **kwargs
        Forwarded to ax.text() (e.g., fontsize, color).

    Returns
    -------
    matplotlib.text.Text
        The text object.
    """
    defaults = {
        "ha": "center",
        "va": "center",
        "bbox": {
            "facecolor": COLORS["offwhite"],
            "alpha": 0.75,
            "pad": 2,
            "edgecolor": "none",
        },
        "fontsize": 7,
    }
    defaults.update(kwargs)
    return ax.text(x, y, text, **defaults)


# Initialize exported constants with configured theme.
_set_theme_exports(_resolve_theme(None))
