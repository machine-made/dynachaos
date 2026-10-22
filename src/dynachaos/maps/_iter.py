"""Shared iteration helpers for low-dimensional map scripts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np


def iterate_unwrapped(state0, increment_fn: Callable, n_steps: int):
    """Advance an unwrapped state by repeatedly adding an increment."""
    state = np.array(state0, dtype=np.float64, copy=True)
    if state.ndim == 0:
        value = float(state)
        for _ in range(n_steps):
            value += float(increment_fn(value))
        return value

    for _ in range(n_steps):
        state += np.asarray(increment_fn(state), dtype=np.float64)
    return state


def run_transient(
    state0,
    step_fn: Callable,
    n_transient: int,
    diverged_fn: Callable | None = None,
):
    """Iterate a transient segment and return final state or None."""
    state = np.array(state0, dtype=np.float64, copy=True)
    for _ in range(n_transient):
        state = np.asarray(step_fn(state), dtype=np.float64)
        if diverged_fn is not None and diverged_fn(state):
            return None
    return state


def sample_trajectory(
    state0,
    step_fn: Callable,
    n_record: int,
    *,
    project_fn: Callable | None = None,
    diverged_fn: Callable | None = None,
    allow_partial: bool = False,
):
    """Record a trajectory segment after repeated map iterations."""
    state = np.array(state0, dtype=np.float64, copy=True)
    sample0 = np.asarray(
        project_fn(state) if project_fn is not None else state,
        dtype=np.float64,
    )
    shape = (n_record,) if sample0.ndim == 0 else (n_record, *sample0.shape)
    traj = np.empty(shape, dtype=np.float64)

    for i in range(n_record):
        state = np.asarray(step_fn(state), dtype=np.float64)
        if diverged_fn is not None and diverged_fn(state):
            if allow_partial:
                return traj[:i]
            return None
        sample = np.asarray(
            project_fn(state) if project_fn is not None else state,
            dtype=np.float64,
        )
        traj[i] = sample
    return traj


def trajectory_after_transient(
    state0,
    step_fn: Callable,
    n_transient: int,
    n_record: int,
    *,
    project_fn: Callable | None = None,
    diverged_fn: Callable | None = None,
    allow_partial: bool = False,
):
    """Run transient then record trajectory points."""
    state = run_transient(state0, step_fn, n_transient, diverged_fn=diverged_fn)
    if state is None:
        return None
    return sample_trajectory(
        state,
        step_fn,
        n_record,
        project_fn=project_fn,
        diverged_fn=diverged_fn,
        allow_partial=allow_partial,
    )


def run_animation_sweep(
    iterate_fn: Callable,
    param_values,
    output_npz: Path,
    *,
    n_plot: int,
    progress_interval: int | None = None,
):
    """Thin wrapper around shared animation-sweep computation."""
    from dynachaos.utils.animation import compute_animation_sweep

    kwargs = {"n_plot": n_plot}
    if progress_interval is not None:
        kwargs["progress_interval"] = progress_interval
    return compute_animation_sweep(iterate_fn, param_values, output_npz, **kwargs)
