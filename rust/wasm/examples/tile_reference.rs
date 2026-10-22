//! Print a fixed rotation-number tile as raw bit patterns.
//!
//! The parity check compares this native output against the same tile computed
//! by the WebAssembly build. Values are printed as the hexadecimal bit pattern
//! of each f64, so the comparison is exact and a difference can be measured in
//! units in the last place. Printing decimal text would hide a one-bit
//! disagreement behind the formatter.
//!
//! Build this WITHOUT `-C target-cpu=native`. The repository enables that flag
//! for local native builds, and it lets the compiler fuse a multiply and an add
//! into one instruction that `wasm32` has no equivalent for. Comparing a
//! natively tuned build against wasm would measure the compiler, not the port.

use dynachaos_wasm::rotation_number_tile;

fn main() {
    let tile = rotation_number_tile(
        params::OMEGA_MIN,
        params::OMEGA_MAX,
        params::N_OMEGA,
        params::K_MIN,
        params::K_MAX,
        params::N_K,
        params::N_TRANSIENT,
        params::N_ITER,
        params::THETA0,
    );
    for value in tile {
        println!("{:016x}", value.to_bits());
    }
}

/// The tile both sides must agree on. Keep in step with `parity_tile.mjs`.
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
