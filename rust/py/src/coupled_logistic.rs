//! Python bindings for coupled logistic map kernels.

use numpy::ndarray::Array3;
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute basin labels for a fixed two-site coupled logistic map.
///
/// Python keeps ownership of reference-orbit construction, x/y grids, payload
/// writing, and plotting. This kernel only accelerates the grid transient and
/// classification loop.
#[pyfunction]
#[pyo3(signature = (x_values, y_values, A, D, n_transient, ref_a))]
#[allow(non_snake_case)]
pub fn coupled_logistic_basin_grid<'py>(
    py: Python<'py>,
    x_values: PyReadonlyArray1<'py, f64>,
    y_values: PyReadonlyArray1<'py, f64>,
    A: f64,
    D: f64,
    n_transient: usize,
    ref_a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let basin = dynachaos_core::coupled_logistic_basin_grid(
        x_values.as_slice()?,
        y_values.as_slice()?,
        A,
        D,
        n_transient,
        ref_a.as_array(),
    )
    .map_err(crate::core_to_py)?;
    Ok(PyArray2::from_owned_array(py, basin))
}

/// Coupled-logistic phase tile: `(asym, lyap)` for every `(D, A)` pair.
///
/// Mirrors `compute_phase_diagram` in `dynachaos.maps.coupled_logistic`. The
/// tangent step runs before the map step. `|x| > 10` or `|y| > 10` sets the
/// state to NaN, and a cell with no valid sample is NaN.
///
/// Returns an array of shape `(n_D, n_A, 2)`: `[..., 0]` is `<|x - y|>`,
/// `[..., 1]` is the largest-tangent log growth. Pass `x0 = 0.1`, `y0 = 0.2`
/// to match the Python sweep's hardcoded start.
#[pyfunction]
#[pyo3(signature = (a_values, d_values, n_transient, n_sample, x0, y0))]
pub fn coupled_logistic_phase_tile<'py>(
    py: Python<'py>,
    a_values: PyReadonlyArray1<'py, f64>,
    d_values: PyReadonlyArray1<'py, f64>,
    n_transient: usize,
    n_sample: usize,
    x0: f64,
    y0: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let a = a_values.as_slice()?;
    let d = d_values.as_slice()?;
    let n_a = a.len();
    let n_d = d.len();
    let flat = dynachaos_core::coupled_logistic_phase_tile(a, d, n_transient, n_sample, x0, y0)
        .map_err(crate::core_to_py)?;
    let tile = Array3::from_shape_vec((n_d, n_a, 2), flat)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyArray3::from_owned_array(py, tile))
}

/// Coupled-logistic attractor tile: one `(x, y)` trajectory per `A`.
///
/// Mirrors `compute_attractors` in `dynachaos.maps.coupled_logistic`, which
/// passes no divergence check to `trajectory_after_transient`. This kernel
/// likewise records every sample, including non-finite values.
///
/// Returns an array of shape `(n_A, n_plot, 2)`.
#[pyfunction]
#[pyo3(signature = (a_values, D, n_transient, n_plot, state0))]
#[allow(non_snake_case)]
pub fn coupled_logistic_attractor_tile<'py>(
    py: Python<'py>,
    a_values: PyReadonlyArray1<'py, f64>,
    D: f64,
    n_transient: usize,
    n_plot: usize,
    state0: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let a = a_values.as_slice()?;
    let n_a = a.len();
    let flat = dynachaos_core::coupled_logistic_attractor_tile(
        a,
        D,
        n_transient,
        n_plot,
        state0.as_slice()?,
    )
    .map_err(crate::core_to_py)?;
    let tile = Array3::from_shape_vec((n_a, n_plot, 2), flat)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyArray3::from_owned_array(py, tile))
}
