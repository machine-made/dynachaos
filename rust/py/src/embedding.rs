//! Python bindings for Cao embedding-dimension selector helpers.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Select embedding dimension from a Cao E1(d) curve.
///
/// This mirrors the Python selector logic:
/// 1) choose onset of a stable near-1 plateau (forward window),
/// 2) fallback to first near-one crossing,
/// 3) fallback to closest value to 1.
#[pyfunction]
#[pyo3(signature = (
    e1,
    near_one_lower = 0.95,
    near_one_upper = 1.05,
    saturation_tol = 0.02,
    plateau_span = 3,
    smoothing_window = 1,
    min_dim = 2,
    max_dim = None
))]
#[allow(clippy::too_many_arguments)]
pub fn select_dimension_cao(
    e1: PyReadonlyArray1<'_, f64>,
    near_one_lower: f64,
    near_one_upper: f64,
    saturation_tol: f64,
    plateau_span: usize,
    smoothing_window: usize,
    min_dim: usize,
    max_dim: Option<usize>,
) -> PyResult<usize> {
    Ok(dynachaos_core::select_dimension_cao(
        e1.as_slice()?,
        near_one_lower,
        near_one_upper,
        saturation_tol,
        plateau_span,
        smoothing_window,
        min_dim,
        max_dim,
    ))
}
