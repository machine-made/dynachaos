//! Python bindings for synthetic intermittency oracle kernels.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (n, x0, eps, a, modulo))]
pub fn pm_type_i_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    eps: f64,
    a: f64,
    modulo: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out = dynachaos_core::pm_type_i_oracle(n, x0, eps, a, modulo).map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (n, x0, y0, eps, a, theta))]
pub fn pm_type_ii_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    y0: f64,
    eps: f64,
    a: f64,
    theta: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr =
        dynachaos_core::pm_type_ii_oracle(n, x0, y0, eps, a, theta).map_err(crate::core_to_py)?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
#[pyo3(signature = (n, x0, eps, a))]
pub fn pm_type_iii_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    eps: f64,
    a: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out = dynachaos_core::pm_type_iii_oracle(n, x0, eps, a).map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (driver, x0, transverse_lyapunov, noise_scale))]
pub fn on_off_oracle<'py>(
    py: Python<'py>,
    driver: PyReadonlyArray1<'py, f64>,
    x0: f64,
    transverse_lyapunov: f64,
    noise_scale: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out =
        dynachaos_core::on_off_oracle(driver.as_slice()?, x0, transverse_lyapunov, noise_scale)
            .map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (n, x0, y0, eps))]
pub fn on_off_skew_logistic_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    y0: f64,
    eps: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr =
        dynachaos_core::on_off_skew_logistic_oracle(n, x0, y0, eps).map_err(crate::core_to_py)?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
#[pyo3(signature = (n, x0, r))]
pub fn logistic_type_i_oracle<'py>(
    py: Python<'py>,
    n: usize,
    x0: f64,
    r: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out = dynachaos_core::logistic_type_i_oracle(n, x0, r).map_err(crate::core_to_py)?;
    Ok(PyArray1::from_vec(py, out))
}
