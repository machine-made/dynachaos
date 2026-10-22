use dynachaos_core::CoreError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod ami;
mod circle_map;
mod cml;
mod comoving;
mod correlation_gp;
mod coupled_logistic;
mod embedding;
mod entropy;
mod intermittency;
mod maps;
mod multifractal;
mod permutation;
mod recurrence;
mod zero_one;

pub(crate) fn core_to_py(err: CoreError) -> PyErr {
    match err {
        CoreError::InvalidArgument(message) => PyValueError::new_err(message),
        CoreError::Runtime(message) => PyRuntimeError::new_err(message),
    }
}

/// Rust-accelerated backends for dynachaos.
///
/// This module is optional — all algorithms have pure-Python fallbacks.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recurrence::diagonal_lines, m)?)?;
    m.add_function(wrap_pyfunction!(recurrence::vertical_lines, m)?)?;
    m.add_function(wrap_pyfunction!(permutation::ordinal_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(ami::ami_histogram, m)?)?;
    // Cao/FNN nearest-neighbor statistics remain in Python/SciPy; cKDTree is faster.
    m.add_function(wrap_pyfunction!(embedding::select_dimension_cao, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_gp::correlation_counts, m)?)?;
    m.add_function(wrap_pyfunction!(circle_map::rotation_number_tile, m)?)?;
    m.add_function(wrap_pyfunction!(cml::cml_jacobian_logistic, m)?)?;
    m.add_function(wrap_pyfunction!(comoving::comoving_lyapunov_logistic, m)?)?;
    m.add_function(wrap_pyfunction!(
        coupled_logistic::coupled_logistic_basin_grid,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        coupled_logistic::coupled_logistic_phase_tile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        coupled_logistic::coupled_logistic_attractor_tile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(recurrence::count_line_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::apen_counts, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::fuzzy_entropy_sum, m)?)?;
    m.add_function(wrap_pyfunction!(multifractal::multifractal_moments, m)?)?;
    m.add_function(wrap_pyfunction!(intermittency::pm_type_i_oracle, m)?)?;
    m.add_function(wrap_pyfunction!(intermittency::pm_type_ii_oracle, m)?)?;
    m.add_function(wrap_pyfunction!(intermittency::pm_type_iii_oracle, m)?)?;
    m.add_function(wrap_pyfunction!(intermittency::on_off_oracle, m)?)?;
    m.add_function(wrap_pyfunction!(
        intermittency::on_off_skew_logistic_oracle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(intermittency::logistic_type_i_oracle, m)?)?;
    m.add_function(wrap_pyfunction!(zero_one::zero_one_k, m)?)?;
    m.add_function(wrap_pyfunction!(maps::delayed_logistic_attractor_tile, m)?)?;
    m.add_function(wrap_pyfunction!(maps::torus_doubling_attractor_tile, m)?)?;
    m.add_function(wrap_pyfunction!(maps::modulated_circle_rotation_tile, m)?)?;
    m.add_function(wrap_pyfunction!(cml::cml_spacetime_tile, m)?)?;
    m.add_function(wrap_pyfunction!(maps::circle_map_lyapunov_sum, m)?)?;
    m.add_function(wrap_pyfunction!(maps::delayed_logistic_lyapunov_tile, m)?)?;
    m.add_function(wrap_pyfunction!(maps::torus_doubling_lyapunov_tile, m)?)?;
    m.add_function(wrap_pyfunction!(maps::coupled_delayed_lyapunov_tile, m)?)?;
    m.add_function(wrap_pyfunction!(maps::coupled_delayed_projection_tile, m)?)?;
    Ok(())
}
