"""Return map (Poincaré first-return) plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from dynachaos.utils.style import color_for


def return_map_plot(
    series: npt.NDArray[np.float64],
    lag: int = 1,
    ax: plt.Axes | None = None,
    **scatter_kw,
) -> plt.Axes:
    """Plot a return map (x_n vs x_{n+lag}).

    Parameters
    ----------
    series : ndarray
        1D time series.
    lag : int
        Lag for the return map (default 1).
    ax : matplotlib Axes or None
        Axes to plot on; created if None.
    **scatter_kw
        Extra keyword arguments passed to ``ax.scatter``.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    defaults = {"s": 0.5, "c": color_for(0), "alpha": 0.3}
    defaults.update(scatter_kw)

    x_n = series[:-lag]
    x_next = series[lag:]
    ax.scatter(x_n, x_next, **defaults)

    ax.set_xlabel("$x_n$")
    ax.set_ylabel(f"$x_{{n+{lag}}}$")
    ax.set_aspect("equal")
    return ax
