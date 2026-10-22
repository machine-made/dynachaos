//! Python bindings for multifractal partition moments.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compute multifractal canonical moments for a 2D nonnegative measure field.
///
/// Parameters
/// ----------
/// field : numpy.ndarray of float64, shape (ny, nx)
///     Nonnegative measure field.
/// box_sizes : numpy.ndarray of int64, shape (n_scales,)
///     Box side lengths. Each scale uses non-overlapping boxes and truncates
///     edge remainders.
/// q_values : numpy.ndarray of float64, shape (n_q,)
///     Moment orders q.
///
/// Returns
/// -------
/// tuple
///     (log_z, alpha_num, f_num, ln_scales), where:
///     - log_z: ndarray, shape (n_scales, n_q)
///     - alpha_num: ndarray, shape (n_scales, n_q)
///     - f_num: ndarray, shape (n_scales, n_q)
///     - ln_scales: ndarray, shape (n_scales,)
#[pyfunction]
#[pyo3(signature = (field, box_sizes, q_values))]
pub fn multifractal_moments<'py>(
    py: Python<'py>,
    field: PyReadonlyArray2<'py, f64>,
    box_sizes: PyReadonlyArray1<'py, i64>,
    q_values: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    #![allow(clippy::type_complexity)]
    let arr = field.as_array();
    let ny = arr.shape()[0];
    let nx = arr.shape()[1];
    let field_slice = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("field must be C-contiguous"))?;

    let moments = dynachaos_core::multifractal_moments(
        field_slice,
        ny,
        nx,
        box_sizes.as_slice()?,
        q_values.as_slice()?,
    )
    .map_err(crate::core_to_py)?;

    Ok((
        PyArray2::from_owned_array(py, moments.log_z),
        PyArray2::from_owned_array(py, moments.alpha_num),
        PyArray2::from_owned_array(py, moments.f_num),
        PyArray1::from_vec(py, moments.ln_scales),
    ))
}
