//! Python bindings for approximate and fuzzy entropy kernels.

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Count approximate-entropy template matches for every template row.
///
/// Parameters
/// ----------
/// traj : numpy.ndarray of float64, shape (N, d)
///     Embedded trajectory templates. Must be C-contiguous.
/// r : float
///     Positive match tolerance. Matches use Chebyshev distance <= r.
///
/// Returns
/// -------
/// numpy.ndarray of int64, shape (N,)
///     counts[i] = number of rows j with max(abs(traj[i] - traj[j])) <= r.
///     Self-matches are included, matching ApEn (Pincus, 1991).
#[pyfunction]
pub fn apen_counts<'py>(
    py: Python<'py>,
    traj: PyReadonlyArray2<'py, f64>,
    r: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let traj_arr = traj.as_array();
    let shape = traj_arr.shape();
    let n_pts = *shape
        .first()
        .ok_or_else(|| PyValueError::new_err("traj must be two-dimensional"))?;
    let dim = *shape
        .get(1)
        .ok_or_else(|| PyValueError::new_err("traj must be two-dimensional"))?;
    let traj_slice = traj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("traj must be C-contiguous"))?;

    let counts =
        dynachaos_core::apen_counts(traj_slice, n_pts, dim, r).map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, counts))
}

/// Compute the total fuzzy-membership sum over all valid template pairs.
///
/// Parameters
/// ----------
/// traj : numpy.ndarray of float64, shape (N, d)
///     Embedded trajectory templates. Must be C-contiguous.
/// r : float
///     Positive tolerance scale in the fuzzy membership.
/// n : int
///     Fuzzy exponent in `exp(-(d / r)^n)`.
/// theiler_window : int
///     Minimum temporal separation: only pairs with `j > i + theiler_window`.
///
/// Returns
/// -------
/// float
///     Sum of fuzzy memberships over all valid upper-triangle pairs.
#[pyfunction]
#[pyo3(signature = (traj, r, n, theiler_window = 0))]
pub fn fuzzy_entropy_sum(
    traj: PyReadonlyArray2<'_, f64>,
    r: f64,
    n: u32,
    theiler_window: usize,
) -> PyResult<f64> {
    let traj_arr = traj.as_array();
    let n_pts = traj_arr.shape()[0];
    let dim = traj_arr.shape()[1];
    let traj_slice = traj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("traj must be C-contiguous"))?;

    dynachaos_core::fuzzy_entropy_sum(traj_slice, n_pts, dim, r, n, theiler_window)
        .map_err(crate::core_to_py)
}
