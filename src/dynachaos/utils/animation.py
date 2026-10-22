"""Generic animation helpers for 2D attractor GIFs.

Provides compute_animation_sweep() for parameter-sweep caching and
make_attractor_gif() for rendering GIFs with fixed axis limits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_animation_sweep(
    iterate_fn,
    param_values,
    output_path: Path,
    *,
    n_plot: int = 5_000,
    progress_interval: int = 50,
):
    """Sweep a parameter and cache 2D attractor projections.

    Parameters
    ----------
    iterate_fn : callable
        ``iterate_fn(param) -> ndarray`` of shape ``(n_plot, 2)``.
        Caller provides a closure that iterates the map at one parameter
        value, discards transients, and returns the 2D projection to plot.
    param_values : array-like
        Parameter values to sweep (one per animation frame).
    output_path : Path
        Where to save the ``.npz`` cache.
    n_plot : int
        Expected number of points per frame (used for pre-allocation).
    progress_interval : int
        Print progress and save incremental checkpoint every this many frames.

    Returns
    -------
    dict
        Keys ``param_values``, ``all_x``, ``all_y``.
    """
    param_values = np.asarray(param_values, dtype=np.float64)
    n_frames = len(param_values)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_x = np.empty((n_frames, n_plot))
    all_y = np.empty((n_frames, n_plot))

    for i, p in enumerate(param_values):
        traj = iterate_fn(p)
        if traj is None:
            # Diverged — fill with NaN so the frame is blank
            all_x[i] = np.nan
            all_y[i] = np.nan
        else:
            actual = len(traj)
            if actual >= n_plot:
                all_x[i] = traj[:n_plot, 0]
                all_y[i] = traj[:n_plot, 1]
            else:
                # Fewer points than expected — pad with NaN
                all_x[i, :actual] = traj[:, 0]
                all_x[i, actual:] = np.nan
                all_y[i, :actual] = traj[:, 1]
                all_y[i, actual:] = np.nan

        if progress_interval and (i + 1) % progress_interval == 0:
            print(f"  Animation: {i + 1}/{n_frames}")
            np.savez_compressed(
                output_path,
                param_values=param_values[: i + 1],
                all_x=all_x[: i + 1],
                all_y=all_y[: i + 1],
            )

    np.savez_compressed(output_path, param_values=param_values, all_x=all_x, all_y=all_y)
    print(f"Saved {output_path}")
    return {"param_values": param_values, "all_x": all_x, "all_y": all_y}


def make_attractor_gif(
    param_values,
    all_x,
    all_y,
    output_path: Path,
    *,
    title_template: str = "{param_name} = {param_value}",
    param_name: str = "D",
    param_fmt: str = ".3f",
    xlabel: str = "$x$",
    ylabel: str = "$y$",
    fps: int = 15,
    dpi: int = 100,
    figsize: tuple[float, float] = (4.5, 4.0),
    point_size: float = 0.1,
    alpha: float = 0.4,
):
    """Render a GIF from precomputed 2D trajectories.

    Parameters
    ----------
    param_values : ndarray, shape (n_frames,)
    all_x, all_y : ndarray, shape (n_frames, n_plot)
    output_path : Path
    title_template : str
        Format string with ``{param_name}`` and ``{param_value}`` placeholders.
    param_name : str
        Name shown in the title (e.g. ``"D"``, ``r"$D_2$"``).
    param_fmt : str
        Format specifier for the parameter value.
    xlabel, ylabel : str
    fps, dpi : int
    figsize : tuple
    point_size, alpha : float

    Returns
    -------
    Path
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from dynachaos.utils.style import COLORS, figure_spec, setup

    setup()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed axis limits with 5% padding (ignore NaN and Inf)
    pad = 0.05
    finite_x = all_x[np.isfinite(all_x)]
    finite_y = all_y[np.isfinite(all_y)]
    xmin, xmax = finite_x.min(), finite_x.max()
    ymin, ymax = finite_y.min(), finite_y.max()
    x_range = xmax - xmin
    y_range = ymax - ymin
    xlim = (xmin - pad * x_range, xmax + pad * x_range)
    ylim = (ymin - pad * y_range, ymax + pad * y_range)

    spec = figure_spec("single")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    scatter = ax.scatter([], [], s=point_size, c=COLORS["black"], alpha=alpha)
    title_obj = ax.set_title("", fontsize=spec.title_size)

    def update(frame):
        scatter.set_offsets(np.column_stack([all_x[frame], all_y[frame]]))
        pv = format(param_values[frame], param_fmt)
        title_obj.set_text(title_template.format(param_name=param_name, param_value=pv))
        return scatter, title_obj

    anim = FuncAnimation(fig, update, frames=len(param_values), blit=True)
    anim.save(str(output_path), dpi=dpi, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path
