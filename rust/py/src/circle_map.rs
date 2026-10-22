//! Python bindings for the sine circle map kernels.

use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::prelude::*;

/// Rotation numbers over a rectangular tile of the (Omega, K) plane.
///
/// The map is `theta -> theta + Omega + K * sin(2 * pi * theta)`, iterated
/// without reduction modulo 1, which is the convention
/// `dynachaos.maps.arnold_tongues` uses.
///
/// Returns an array of shape `(n_K, n_omega)`: K varies down the rows and
/// Omega along the columns, matching the `rho` array the reproduction
/// pipeline stores.
#[pyfunction]
#[pyo3(signature = (
    omega_min,
    omega_max,
    n_omega,
    k_min,
    k_max,
    n_k,
    n_transient = 5000,
    n_iter = 50_000,
    theta0 = 0.1,
))]
#[allow(clippy::too_many_arguments)]
pub fn rotation_number_tile<'py>(
    py: Python<'py>,
    omega_min: f64,
    omega_max: f64,
    n_omega: usize,
    k_min: f64,
    k_max: f64,
    n_k: usize,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rho = py
        .detach(|| {
            dynachaos_core::rotation_number_tile(
                omega_min,
                omega_max,
                n_omega,
                k_min,
                k_max,
                n_k,
                n_transient,
                n_iter,
                theta0,
            )
        })
        .map_err(crate::core_to_py)?;
    let grid = Array2::from_shape_vec((n_k, n_omega), rho)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, grid))
}
