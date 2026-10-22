//! Python bindings for average mutual information.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute the Average Mutual Information I(τ) for τ = 1..tau_max.
///
/// Uses uniform histogram binning with `n_bins` bins per axis.
///
/// Parameters
/// ----------
/// x : numpy.ndarray of float64, shape (N,)
///     Scalar time series.
/// tau_max : int
///     Maximum delay.
/// n_bins : int
///     Number of histogram bins (default 64).
///
/// Returns
/// -------
/// numpy.ndarray of float64, shape (tau_max,)
///     Mutual information values I(1), I(2), ..., I(tau_max).
#[pyfunction]
#[pyo3(signature = (x, tau_max, n_bins = 64))]
pub fn ami_histogram<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    tau_max: usize,
    n_bins: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mi_values =
        dynachaos_core::ami_histogram(x.as_slice()?, tau_max, n_bins).map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, mi_values))
}
