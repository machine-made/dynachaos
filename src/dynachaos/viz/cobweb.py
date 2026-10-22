"""Cobweb (staircase) diagrams for 1D iterated maps."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from dynachaos.utils.style import color_for


def cobweb_diagram(
    f: Callable,
    x0: float,
    n_iter: int = 50,
    x_range: tuple[float, float] = (0.0, 1.0),
    ax: plt.Axes | None = None,
    n_curve: int = 500,
) -> plt.Axes:
    """Plot a cobweb diagram for a 1D map.

    Parameters
    ----------
    f : callable
        The map ``f(x) -> x``.
    x0 : float
        Initial condition.
    n_iter : int
        Number of cobweb iterates to draw.
    x_range : tuple
        (xmin, xmax) for the curve and diagonal.
    ax : matplotlib Axes or None
        Axes to plot on; created if None.
    n_curve : int
        Number of points for the map curve.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    map_color = color_for(0)
    web_color = color_for(1)

    # Plot f(x) and y=x
    xs = np.linspace(x_range[0], x_range[1], n_curve)
    try:
        ys = f(xs)
    except (TypeError, IndexError):
        ys = np.array([f(xi) for xi in xs])
    ax.plot(xs, ys, color=map_color, lw=1.5, label="$f(x)$")
    ax.plot(
        xs, xs, ls="--", color=ax.spines["bottom"].get_edgecolor(), lw=0.8, alpha=0.5, label="$y=x$"
    )

    # Cobweb
    x = x0
    for _ in range(n_iter):
        y = f(x)
        ax.plot([x, x], [x, y], color=web_color, lw=0.6, alpha=0.7)
        ax.plot([x, y], [y, y], color=web_color, lw=0.6, alpha=0.7)
        x = y

    ax.set_xlabel("$x_n$")
    ax.set_ylabel("$x_{n+1}$")
    ax.set_xlim(x_range)
    ax.set_ylim(x_range)
    ax.set_aspect("equal")
    return ax
