//! Python bindings for the Gottwald–Melbourne 0-1 test.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Evaluate the 0-1 test statistic for each frequency in `c_values`.
///
/// Parameters
/// ----------
/// phi : numpy.ndarray of float64, shape (N,)
///     Scalar time series.
/// c_values : numpy.ndarray of float64, shape (n_c,)
///     Frequencies to evaluate. The caller draws them; the kernel only
///     evaluates, so both backends see the same values.
/// n_cut : int
///     Number of lags used in the MSD regression, in [2, N].
///
/// Returns
/// -------
/// numpy.ndarray of float64, shape (n_c,)
///     One K per c value, in the same order.
#[pyfunction]
#[pyo3(signature = (phi, c_values, n_cut))]
pub fn zero_one_k<'py>(
    py: Python<'py>,
    phi: PyReadonlyArray1<'py, f64>,
    c_values: PyReadonlyArray1<'py, f64>,
    n_cut: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let phi_slice = phi.as_slice()?;
    let c_slice = c_values.as_slice()?;
    let k_values = py
        .detach(|| dynachaos_core::zero_one_k(phi_slice, c_slice, n_cut))
        .map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, k_values))
}
