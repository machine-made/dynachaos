//! Print per-cell exit kinds for the wasm parity tile.
//!
//! One line per cell: `kind` is 0 (locked), 1 (bounded-error stop), or 2
//! (exhausted). Locked lines also print `p` and `q`. Keep the params in step
//! with `rust/wasm/examples/tile_reference.rs`.

use dynachaos_core::{ExitKind, rotation_number_with_exit};

mod params {
    pub const OMEGA_MIN: f64 = 0.0;
    pub const OMEGA_MAX: f64 = 1.0;
    pub const N_OMEGA: usize = 64;
    pub const K_MIN: f64 = 0.0;
    pub const K_MAX: f64 = 0.3;
    pub const N_K: usize = 64;
    pub const N_TRANSIENT: usize = 200;
    pub const N_ITER: usize = 500;
    pub const THETA0: f64 = 0.1;
}

fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    let mut values: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    values[n - 1] = stop;
    values
}

fn main() {
    let omega_values = linspace(params::OMEGA_MIN, params::OMEGA_MAX, params::N_OMEGA);
    let k_values = linspace(params::K_MIN, params::K_MAX, params::N_K);

    for &k in &k_values {
        for &omega in &omega_values {
            let (_, kind, _) = rotation_number_with_exit(
                omega,
                k,
                params::N_TRANSIENT,
                params::N_ITER,
                params::THETA0,
            );
            match kind {
                ExitKind::Locked { p, q } => println!("0 {p} {q}"),
                ExitKind::BoundedError => println!("1"),
                ExitKind::Exhausted => println!("2"),
            }
        }
    }
}
