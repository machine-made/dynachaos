"""Private validation helpers for diagnostics modules."""

from __future__ import annotations

import numpy as np


def finite_series_1d(x, *, name="x"):
    """Return a flattened finite float64 series."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def finite_trajectory(X, *, name="X"):
    """Return a finite non-empty 2D trajectory, promoting 1D input to a column."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1D or 2D trajectory")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def positive_int(value, name):
    """Return a positive integer, rejecting bools and lossy conversions."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value_int != value or value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def nonnegative_int(value, name):
    """Return a non-negative integer, rejecting bools and lossy conversions."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value_int != value or value_int < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value_int


def finite_positive_scalar(value, *, name):
    """Return a finite positive float."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def finite_nonnegative_scalar(value, *, name):
    """Return a finite non-negative float."""
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def sorted_nonnegative_radius_grid(r_values, *, name="r_values"):
    """Return a finite sorted 1D non-negative radius grid."""
    radii = np.asarray(r_values, dtype=np.float64)
    if radii.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(radii)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(radii < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if np.any(np.diff(radii) < 0.0):
        raise ValueError(f"{name} must be sorted in ascending order")
    return radii


def square_bool_matrix(matrix, *, name="R", symmetric=False):
    """Return a non-empty square bool matrix, optionally requiring symmetry."""
    arr = np.asarray(matrix, dtype=bool)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if symmetric and not np.array_equal(arr, arr.T):
        raise ValueError(f"{name} must be symmetric")
    return arr
