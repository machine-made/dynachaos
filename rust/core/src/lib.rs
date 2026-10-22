//! Pure-Rust kernels for dynachaos.
//!
//! Callers pass slices or ndarray views. The Python extension wraps these
//! functions for `dynachaos._rust`.

pub mod ami;
pub mod circle_map;
pub mod cml;
pub mod comoving;
pub mod correlation_gp;
pub mod coupled_logistic;
pub mod embedding;
pub mod entropy;
mod fft;
pub mod intermittency;
pub mod lyapunov;
pub mod maps;
pub mod multifractal;
pub mod permutation;
pub mod recurrence;
pub mod zero_one;

pub use ami::ami_histogram;
pub use circle_map::{
    ExitKind, rotation_number_full, rotation_number_point, rotation_number_tile,
    rotation_number_with_exit,
};
pub use cml::{cml_jacobian_logistic, cml_spacetime_tile};
pub use comoving::comoving_lyapunov_logistic;
pub use correlation_gp::correlation_counts;
pub use coupled_logistic::{
    coupled_logistic_attractor_tile, coupled_logistic_basin_grid, coupled_logistic_phase_tile,
    coupled_logistic_reference_orbit,
};
pub use embedding::select_dimension_cao;
pub use entropy::{apen_counts, fuzzy_entropy_sum};
pub use intermittency::{
    logistic_type_i_oracle, on_off_oracle, on_off_skew_logistic_oracle, pm_type_i_oracle,
    pm_type_ii_oracle, pm_type_iii_oracle,
};
pub use lyapunov::{
    circle_map_lyapunov_sum, coupled_delayed_lyapunov_tile, delayed_logistic_lyapunov_tile,
    torus_doubling_lyapunov_tile,
};
pub use maps::{
    coupled_delayed_projection_tile, delayed_logistic_attractor_tile,
    modulated_circle_rotation_tile, torus_doubling_attractor_tile,
};
pub use multifractal::{MultifractalMoments, multifractal_moments};
pub use permutation::ordinal_distribution;
pub use recurrence::{count_line_lengths, diagonal_lines, vertical_lines};
pub use zero_one::zero_one_k;

use std::fmt;

/// Raised when a kernel rejects its input or cannot build its output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// Raised when a caller argument is invalid.
    InvalidArgument(String),
    /// Raised when an output array cannot be formed.
    Runtime(String),
}

impl CoreError {
    /// Build an invalid-argument error from a message.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    /// Build a runtime error from a message.
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) | Self::Runtime(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for CoreError {}
