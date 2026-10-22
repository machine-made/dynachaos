//! Python bindings for the ordinal pattern distribution.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute the ordinal pattern distribution of a time series.
///
/// Parameters
/// ----------
/// x : numpy.ndarray of float64, shape (N,)
///     Scalar time series.
/// d : int
///     Embedding dimension (pattern length), typically 3–7.
/// tau : int
///     Time delay between successive elements.
///
/// Returns
/// -------
/// counts : numpy.ndarray of int64, shape (d!,)
///     Raw counts for each ordinal pattern (indexed by Lehmer code).
/// n_windows : int
///     Total number of windows analysed.
#[pyfunction]
#[pyo3(signature = (x, d = 5, tau = 1))]
pub fn ordinal_distribution<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    d: usize,
    tau: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, i64)> {
    let (counts, n_windows) =
        dynachaos_core::ordinal_distribution(x.as_slice()?, d, tau).map_err(crate::core_to_py)?;
    Ok((PyArray1::from_vec(py, counts), n_windows))
}
