#!/usr/bin/env python3
"""Shared helpers for diagnostics.compare_all."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dynachaos.io.paths import load_or_compute_npz
from dynachaos.maps.primitives import delayed_logistic, logistic

__all__ = [
    "delayed_logistic_series",
    "delayed_logistic_trajectory",
    "load_or_compute_npz",
    "logistic_series",
    "sweep_pair_metric",
    "sweep_scalar_metric",
]


def _delayed_logistic_state_after_transient(D, A, n_transient):
    """Return delayed-logistic state after transient iterations."""
    fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
    state = np.array([fp + 0.01, fp - 0.01])
    for _ in range(n_transient):
        state = delayed_logistic(state, A, D)
    return state


def logistic_series(a, n_transient=5000, n_record=10_000):
    """Scalar time series from the logistic map f(x) = 1 - a x^2."""
    x = 0.1
    for _ in range(n_transient):
        x = logistic(x, a)
    series = np.empty(n_record)
    for i in range(n_record):
        x = logistic(x, a)
        series[i] = x
    return series


def delayed_logistic_series(D, A=0.3, n_transient=10_000, n_record=10_000):
    """Scalar time series (x component) from the delayed logistic map."""
    state = _delayed_logistic_state_after_transient(D, A, n_transient)
    series = np.empty(n_record)
    for i in range(n_record):
        state = delayed_logistic(state, A, D)
        series[i] = state[0]
    return series


def delayed_logistic_trajectory(D, A=0.3, n_transient=10_000, n_record=2000):
    """2D trajectory from the delayed logistic map."""
    state = _delayed_logistic_state_after_transient(D, A, n_transient)

    trajectory = np.empty((n_record, 2))
    for i in range(n_record):
        state = delayed_logistic(state, A, D)
        trajectory[i] = state
    return trajectory


def sweep_scalar_metric(
    values,
    series_fn: Callable[[float], np.ndarray],
    metric_fn: Callable[[np.ndarray], float],
    *,
    progress_every: int | None = None,
    progress_label: str | None = None,
):
    """Evaluate a scalar metric over a parameter sweep."""
    metrics = np.empty(len(values))
    for i, value in enumerate(values):
        metrics[i] = metric_fn(series_fn(value))
        if (
            progress_every is not None
            and progress_label is not None
            and (i + 1) % progress_every == 0
        ):
            print(f"  {progress_label}: {i + 1}/{len(values)}")
    return metrics


def sweep_pair_metric(
    values,
    series_fn: Callable[[float], np.ndarray],
    metric_fn: Callable[[np.ndarray], tuple[float, float]],
    *,
    progress_every: int | None = None,
    progress_label: str | None = None,
):
    """Evaluate a pair-valued metric over a parameter sweep."""
    first = np.empty(len(values))
    second = np.empty(len(values))
    for i, value in enumerate(values):
        first[i], second[i] = metric_fn(series_fn(value))
        if (
            progress_every is not None
            and progress_label is not None
            and (i + 1) % progress_every == 0
        ):
            print(f"  {progress_label}: {i + 1}/{len(values)}")
    return first, second
