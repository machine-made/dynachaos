//! Python bindings for recurrence line extraction.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Count consecutive `true` run lengths in a one-dimensional boolean mask.
///
/// Parameters
/// ----------
/// mask : numpy.ndarray, shape (N,), dtype bool
///     Boolean mask to scan.
/// min_length : int
///     Minimum run length to record.
///
/// Returns
/// -------
/// numpy.ndarray of int64
///     Run lengths that meet the minimum threshold.
#[pyfunction]
#[pyo3(signature = (mask, min_length))]
pub fn count_line_lengths<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray1<'py, bool>,
    min_length: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let arr = mask.as_array();
    let owned;
    let slice = match arr.as_slice() {
        Some(values) => values,
        None => {
            owned = arr.iter().copied().collect::<Vec<_>>();
            owned.as_slice()
        }
    };
    let lengths =
        dynachaos_core::count_line_lengths(slice, min_length).map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, lengths))
}

/// Extract diagonal line lengths from the upper triangle of a recurrence matrix.
///
/// Scans every super-diagonal k = 1, 2, ..., N-1 and records the length of
/// each consecutive run of `true` values that meets the minimum threshold.
///
/// Parameters
/// ----------
/// r : numpy.ndarray, shape (N, N), dtype bool
///     The recurrence matrix.
/// l_min : int
///     Minimum diagonal line length to record (default 2).
///
/// Returns
/// -------
/// numpy.ndarray of int64
///     Array of diagonal line lengths (each ≥ l_min).
#[pyfunction]
#[pyo3(signature = (r, l_min = 2))]
pub fn diagonal_lines<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, bool>,
    l_min: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let lengths = dynachaos_core::diagonal_lines(r.as_array(), l_min).map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, lengths))
}

/// Extract vertical line lengths from a recurrence matrix.
///
/// Scans each column j and records consecutive runs of `true` values.
///
/// Parameters
/// ----------
/// r : numpy.ndarray, shape (N, N), dtype bool
///     The recurrence matrix.
/// v_min : int
///     Minimum vertical line length to record (default 2).
///
/// Returns
/// -------
/// numpy.ndarray of int64
///     Array of vertical line lengths (each ≥ v_min).
#[pyfunction]
#[pyo3(signature = (r, v_min = 2))]
pub fn vertical_lines<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, bool>,
    v_min: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let lengths = dynachaos_core::vertical_lines(r.as_array(), v_min).map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, lengths))
}
