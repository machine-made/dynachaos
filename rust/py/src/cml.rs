//! Python bindings for the coupled-map-lattice kernels.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Logistic CML subsystem Jacobian for sites 0..L-1.
///
/// Returns the row-major flattened L x L block used by the Python caller.
#[pyfunction]
#[pyo3(signature = (x, a, eps, L))]
#[allow(non_snake_case)]
pub fn cml_jacobian_logistic<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: f64,
    eps: f64,
    L: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let jacobian = dynachaos_core::cml_jacobian_logistic(x.as_slice()?, a, eps, L)
        .map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, jacobian))
}

/// CML space-time tile: the field of a coupled-map lattice after a transient.
///
/// Mirrors `simulate_cml` in `dynachaos.cml.spatiotemporal`: `model` selects
/// the lattice — 0 is model (A) (piecewise map), 1 is model (B) (circle map
/// coupled through `sin(2 pi u)`), 2 is model (C) (logistic map) — and the
/// kernel iterates `n_transient` periodic CML steps from `x0`, then records
/// the state after each of the next `n_record` steps.
///
/// Returns an array of shape `(n_record, n_sites)`, row-major by time then
/// site, the same layout `simulate_cml` fills.
#[pyfunction]
#[pyo3(signature = (model, eps, n_transient, n_record, x0))]
pub fn cml_spacetime_tile<'py>(
    py: Python<'py>,
    model: u8,
    eps: f64,
    n_transient: usize,
    n_record: usize,
    x0: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n_sites = x0.as_slice()?.len();
    let flat =
        dynachaos_core::cml_spacetime_tile(model, eps, n_transient, n_record, x0.as_slice()?)
            .map_err(crate::core_to_py)?;
    let tile = Array2::from_shape_vec((n_record, n_sites), flat)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, tile))
}
