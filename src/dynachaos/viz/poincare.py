"""Generic Poincare section plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dynachaos.utils.style import CMAP_SEQUENTIAL, COLORS, apply_axes_polish, color_for


def _select_plane(section: dict[str, object], plane: str) -> tuple[str, np.ndarray]:
    planes = section.get("planes", {})
    if not isinstance(planes, dict):
        planes = {}

    if plane != "auto":
        selected = np.asarray(planes.get(plane, np.empty((0, 2))), dtype=np.float64)
        return plane, selected

    if "signal_delay_pair" in planes:
        selected_name = "signal_delay_pair"
    elif "signal_derivative" in planes:
        selected_name = "signal_derivative"
    elif planes:
        selected_name = next(iter(planes))
    else:
        selected_name = str(section.get("section_plane_type", "unavailable"))

    selected = np.asarray(
        planes.get(selected_name, section.get("section_points", np.empty((0, 2)))),
        dtype=np.float64,
    )
    if selected.ndim == 1:
        selected = selected.reshape(-1, 2)
    if selected.shape[-1] != 2:
        selected = np.empty((0, 2), dtype=np.float64)
    return selected_name, selected


def _axis_labels(plane_name: str) -> tuple[str, str]:
    labels = {
        "signal_delay_pair": ("x(t - delay)", "x(t + delay)"),
        "signal_derivative": ("x(t)", "dx/dt(t)"),
        "derivative_second": ("dx/dt(t)", "d2x/dt2(t)"),
    }
    return labels.get(plane_name, ("coord_1", "coord_2"))


def poincare_section_plot(
    section: dict[str, object],
    *,
    ax: plt.Axes | None = None,
    plane: str = "auto",
    title: str | None = None,
    show_metrics: bool = True,
    kind: str = "single",
    point_alpha: float = 0.72,
) -> plt.Axes:
    """Plot a Poincare section result produced by diagnostics.poincare_section.

    Parameters
    ----------
    section : dict
        Output from ``dynachaos.diagnostics.poincare_section``.
    ax : matplotlib Axes or None
        Target axes; created when omitted.
    plane : str
        Plane key to plot (default ``"auto"``).
    title : str or None
        Optional axes title.
    show_metrics : bool
        Annotate quality metrics text box when True.
    kind : {"single", "double", "grid"}
        Style token class passed to ``apply_axes_polish``.
    point_alpha : float
        Point alpha for scatter layer.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    plane_name, points = _select_plane(section, plane=plane)
    xlab, ylab = _axis_labels(plane_name)

    if points.size > 0:
        n = points.shape[0]
        order = np.linspace(0.0, 1.0, n, dtype=np.float64)
        point_size = float(np.clip(2000.0 / max(n, 1), 4.0, 20.0))
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=order,
            cmap=CMAP_SEQUENTIAL,
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient crossings for section",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=color_for(0),
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title, loc="left")
    apply_axes_polish(ax, kind=kind, title_loc="left")

    if show_metrics:
        metrics = section.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metrics_text = (
            f"Crossings: {int(metrics.get('num_crossings', 0))}\n"
            f"Mean period: {float(metrics.get('mean_period', np.nan)):.3f}s\n"
            f"CV: {float(metrics.get('coefficient_of_variation', np.nan)):.3f}\n"
            f"Spectral ratio: {float(metrics.get('spectral_peak_ratio', np.nan)):.3f}\n"
            f"Section: {plane_name}"
        )
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            color=color_for(0),
            bbox=dict(
                boxstyle="round",
                facecolor=COLORS["offwhite"],
                edgecolor=COLORS["grid"],
                alpha=0.92,
            ),
        )

    return ax


__all__ = ["poincare_section_plot"]
