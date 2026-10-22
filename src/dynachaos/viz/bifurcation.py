"""Bifurcation diagram computation and plotting."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from dynachaos.utils.style import color_for


def bifurcation_diagram(
    f: Callable,
    param_values: npt.NDArray[np.float64],
    x0: float = 0.5,
    n_transient: int = 500,
    n_record: int = 200,
    ax: plt.Axes | None = None,
    **scatter_kw,
) -> plt.Axes:
    """Compute and plot a bifurcation diagram for a 1D map.

    Parameters
    ----------
    f : callable
        ``f(x, p)`` — the parameterised map (must accept scalars).
    param_values : ndarray
        1D array of parameter values to sweep.
    x0 : float
        Initial condition.
    n_transient : int
        Transient iterates to discard at each parameter.
    n_record : int
        Iterates to record at each parameter.
    ax : matplotlib Axes or None
        Axes to plot on; created if None.
    **scatter_kw
        Extra keyword arguments passed to ``ax.scatter``.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    defaults = {"s": 0.02, "c": color_for(0), "lw": 0, "alpha": 0.3}
    defaults.update(scatter_kw)

    n_params = len(param_values)
    all_p = np.repeat(param_values, n_record)
    all_x = np.empty(n_params * n_record, dtype=np.float64)

    for k, p in enumerate(param_values):
        x = float(x0)
        for _ in range(n_transient):
            x = f(x, p)
        block = all_x[k * n_record : (k + 1) * n_record]
        for j in range(n_record):
            block[j] = x
            x = f(x, p)

    ax.scatter(all_p, all_x, **defaults)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("$x$")
    return ax
