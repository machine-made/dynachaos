//! Python bindings for co-moving Lyapunov kernels.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Specialized co-moving Lyapunov spectrum for logistic CML with g=f.
///
/// The Python caller owns RNG/initial-state construction so tests can compare
/// exactly against the existing generic callable implementation.
#[pyfunction]
#[pyo3(signature = (x_init, v_values, a, eps, n_iter, n_transient))]
pub fn comoving_lyapunov_logistic<'py>(
    py: Python<'py>,
    x_init: PyReadonlyArray1<'py, f64>,
    v_values: PyReadonlyArray1<'py, f64>,
    a: f64,
    eps: f64,
    n_iter: usize,
    n_transient: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lambda_v = dynachaos_core::comoving_lyapunov_logistic(
        x_init.as_slice()?,
        v_values.as_slice()?,
        a,
        eps,
        n_iter,
        n_transient,
    )
    .map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, lambda_v))
}
