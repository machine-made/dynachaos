//! Python bindings for the Grassberger-Procaccia correlation integral.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Count pairs within each distance threshold (exact all-pairs).
///
/// Parameters
/// ----------
/// traj : numpy.ndarray of float64, shape (N, d)
///     Trajectory points (already embedded if needed).  Must be C-contiguous.
/// r_values : numpy.ndarray of float64, shape (n_r,)
///     Distance thresholds (must be sorted ascending).
/// theiler_window : int
///     Minimum temporal separation: only pairs with |i-j| > w are counted.
/// use_chebyshev : bool
///     If True, use Chebyshev (max-norm); else Euclidean.
///
/// Returns
/// -------
/// numpy.ndarray of int64, shape (n_r,)
///     counts[k] = number of valid pairs with dist < r_values[k].
#[pyfunction]
#[pyo3(signature = (traj, r_values, theiler_window = 0, use_chebyshev = true))]
pub fn correlation_counts<'py>(
    py: Python<'py>,
    traj: PyReadonlyArray2<'py, f64>,
    r_values: PyReadonlyArray1<'py, f64>,
    theiler_window: usize,
    use_chebyshev: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let r_slice = r_values.as_slice()?;
    if r_slice.is_empty() {
        return Ok(PyArray1::from_vec(py, Vec::new()));
    }

    let traj_arr = traj.as_array();
    let n = traj_arr.shape()[0];
    let dim = traj_arr.shape()[1];
    let traj_slice = traj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("traj must be C-contiguous"))?;

    let counts = dynachaos_core::correlation_counts(
        traj_slice,
        n,
        dim,
        r_slice,
        theiler_window,
        use_chebyshev,
    )
    .map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, counts))
}
