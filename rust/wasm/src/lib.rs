//! Browser surface over the dynachaos kernels.
//!
//! Every export here obeys the contract in `docs/wasm-architecture.md`, which
//! exists because a web page is a hostile caller: its arguments arrive from a
//! slider, a URL or a stranger.
//!
//! 1. Clamp every input into a documented range. Never trust, never reject.
//! 2. Cap every loop. A frozen tab is worse than a coarse picture.
//! 3. Return one flat `Vec<f64>` whose layout is written in the doc comment.
//!
//! The kernels themselves live in `dynachaos-core` and are the same code the
//! published figures were computed with. This crate only converts and guards.
//! Twenty-five exports: `rotation_number_tile` for the picture,
//! `rotation_number_point` for the quoted readout, `zero_one_k` for the
//! 0-1 test for chaos, `correlation_counts`, `apen_counts`,
//! `fuzzy_entropy_sum`, `ordinal_distribution`, `diagonal_lines`,
//! `vertical_lines`, `multifractal_moments`, `ami_histogram` and
//! `select_dimension_cao` for the diagnostics panel,
//! `delayed_logistic_attractor_tile`, `torus_doubling_attractor_tile`,
//! `modulated_circle_rotation_tile` and `cml_spacetime_tile` for the
//! map-attractor and space-time live figures,
//! `circle_map_lyapunov_sum`, `delayed_logistic_lyapunov_tile`,
//! `torus_doubling_lyapunov_tile` and `coupled_delayed_lyapunov_tile`
//! for the Lyapunov panels, and `coupled_logistic_phase_tile`,
//! `coupled_logistic_attractor_tile`, `coupled_logistic_basin_grid`,
//! `coupled_delayed_projection_tile` and `fractalization_attractor_tile`
//! exported for later map panels (no page calls them yet).

use ndarray::Array2;
use wasm_bindgen::prelude::*;

/// Largest tile side the browser may ask for, in cells.
const MAX_SIDE: usize = 512;
/// Largest transient the browser may ask for, in steps.
const MAX_TRANSIENT: usize = 20_000;
/// Largest measured stretch the browser may ask for, in steps.
const MAX_ITER: usize = 200_000;
/// Total step budget for one call: cells x (transient + iter).
///
/// A worker computes one tile per frame, so this is the ceiling on how long a
/// single call may take. At roughly 2 ns per step this is about half a second
/// of work on one core. `n_iter` is reduced to fit, and the value actually
/// used is reported in the returned header so the caller can label the result.
const MAX_STEPS: u64 = 250_000_000;

/// Number of leading `f64` values that describe the tile before the data.
const HEADER: usize = 4;

/// Largest series the browser may hand to `zero_one_k`, in samples.
const MAX_ZERO_ONE_N: usize = 20_000;
/// Largest c ensemble the browser may ask for.
const MAX_ZERO_ONE_C: usize = 100;
/// Number of leading `f64` values that describe a `zero_one_k` result.
const ZERO_ONE_HEADER: usize = 3;

/// Largest embedded trajectory the browser may hand to a diagnostics kernel,
/// in rows.
const MAX_DIAG_N: usize = 20_000;
/// Largest embedding dimension a diagnostics kernel may be asked to use.
const MAX_DIAG_DIM: usize = 32;
/// Largest radius list `correlation_counts` may be asked to evaluate.
const MAX_DIAG_R: usize = 256;
/// Total work budget for one diagnostics call: pairs x dim terms.
///
/// One pair term is one coordinate subtraction and comparison (or one
/// membership evaluation for the fuzzy sum), so this is the ceiling on how
/// long a single call may take. The trajectory is shortened to fit, and the
/// row count actually used is reported in the returned header.
const MAX_DIAG_TERMS: u64 = 200_000_000;
/// Largest fuzzy exponent `fuzzy_entropy_sum` may be asked to use.
const MAX_FUZZY_N: u32 = 16;
/// Largest pattern length `ordinal_distribution` may be asked to use.
///
/// The output carries `d!` counts, so `d` is capped at 8 (40320 values)
/// rather than the kernel's 10 (3.6 million) to keep the flat result small
/// enough to post back to a page.
const MAX_ORDINAL_D: usize = 8;
/// Number of leading `f64` values that describe a `correlation_counts`
/// result, before the radii actually used.
const CORRELATION_HEADER: usize = 5;
/// Number of leading `f64` values that describe an `apen_counts` result.
const APEN_HEADER: usize = 3;
/// Number of leading `f64` values that describe a `fuzzy_entropy_sum` result.
const FUZZY_HEADER: usize = 5;
/// Number of leading `f64` values that describe an `ordinal_distribution`
/// result.
const ORDINAL_HEADER: usize = 4;
/// Largest recurrence-matrix side the browser may hand to a line kernel.
///
/// The mask arrives as `side * side` bytes and the scan is `O(side^2)`, so
/// the side is capped to keep the matrix interactive: at 1024 the mask is
/// one megabyte and the scan is about a million cell visits.
const MAX_RQA_SIDE: usize = 1024;
/// Largest field side `multifractal_moments` may be asked to use, per axis.
const MAX_MF_SIDE: usize = 512;
/// Largest box-size list `multifractal_moments` may be asked to evaluate.
const MAX_MF_BOXES: usize = 64;
/// Largest q list `multifractal_moments` may be asked to evaluate.
const MAX_MF_Q: usize = 64;
/// Largest delay `ami_histogram` may be asked to evaluate.
const MAX_AMI_TAU: usize = 512;
/// Largest bin count `ami_histogram` may be asked to use.
const MAX_AMI_BINS: usize = 512;
/// Largest E1 curve `select_dimension_cao` may be asked to read.
const MAX_CAO_N: usize = 256;
/// Number of leading `f64` values that describe a `diagonal_lines` or
/// `vertical_lines` result.
const LINES_HEADER: usize = 3;
/// Number of leading `f64` values that describe a `multifractal_moments`
/// result.
const MULTIFRACTAL_HEADER: usize = 4;
/// Number of leading `f64` values that describe an `ami_histogram` result.
const AMI_HEADER: usize = 3;
/// Largest D list `delayed_logistic_attractor_tile` may be asked to sweep.
const MAX_MAP_D_LIST: usize = 64;
/// Largest recorded run a map-attractor tile may be asked for, in samples.
const MAX_MAP_PLOT: usize = 4096;
/// Largest magnitude a map-attractor initial-state component may carry.
const MAP_STATE_ABS: f64 = 4.0;
/// Smallest delayed-logistic D the browser may ask for.
///
/// The paper sweeps D over [1.4, 3.5] for the attractor animation
/// (`src/dynachaos/maps/delayed_logistic.py:461`).
const DELAYED_D_MIN: f64 = 1.4;
/// Largest delayed-logistic D the browser may ask for.
const DELAYED_D_MAX: f64 = 3.5;
/// Smallest torus-doubling D the browser may ask for.
///
/// The paper sweeps map I over [1.9, 2.25] and map IV over [1.48, 1.53]
/// (`src/dynachaos/maps/torus_doubling.py:388,420`); the union is the
/// documented range.
const TORUS_D_MIN: f64 = 1.48;
/// Largest torus-doubling D the browser may ask for.
const TORUS_D_MAX: f64 = 2.25;
/// Golden-mean inverse, the modulation frequency the paper's modulated-circle
/// figure uses (`src/dynachaos/maps/modulated_circle.py:27`).
const C_GOLDEN: f64 = 0.6180339887498949;
/// Largest recorded field a CML space-time tile may be asked for, in rows.
const MAX_CML_RECORD: usize = 2048;
/// Fallback coupling for `cml_spacetime_tile` when `eps` is not finite: the
/// middle value of each model's paper sweep
/// (`src/dynachaos/cml/spatiotemporal.py:102,107,112`).
const CML_EPS_FALLBACK: [f64; 3] = [0.07, 0.024, 0.2];
/// Largest D list a Lyapunov tile may sweep. The coupled-delayed tile is
/// tighter: see `MAX_LYAP_DB`.
const MAX_LYAP_D: usize = 512;
/// Largest `D_B` list `coupled_delayed_lyapunov_tile` may sweep.
const MAX_LYAP_DB: usize = 128;
/// Circle-map amplitude cap from the kernel list. The staircase sweeps
/// `A` over `[0, 0.25]` (`src/dynachaos/maps/circle_map.py:109`).
const CIRCLE_A_MAX: f64 = 0.25;
/// Delayed-logistic Lyapunov sweep (`src/dynachaos/maps/delayed_logistic.py:162`).
const DELAYED_LYAP_D_MIN: f64 = 1.3;
/// Largest delayed-logistic D the Lyapunov tile may ask for.
const DELAYED_LYAP_D_MAX: f64 = 2.5;
/// Coupled-delayed Lyapunov sweep (`src/dynachaos/maps/coupled_delayed.py:93`).
const COUPLED_DB_MIN: f64 = 2.1;
/// Largest coupled-delayed `D_B` the Lyapunov tile may ask for.
const COUPLED_DB_MAX: f64 = 2.65;
/// Smallest coupling the coupled-delayed Lyapunov figure uses.
const COUPLED_EPS_MIN: f64 = 1e-3;
/// Largest coupling the coupled-delayed Lyapunov figure uses.
const COUPLED_EPS_MAX: f64 = 1e-2;
/// Middle coupling of the published epsilon trio.
const COUPLED_EPS_PAPER: f64 = 5e-3;
/// Coupled-logistic phase diagram sweeps A over [0.5, 1.65]
/// (`src/dynachaos/maps/coupled_logistic.py:180`).
const LOGISTIC_A_MIN: f64 = 0.5;
/// Largest coupled-logistic A the phase, attractor and basin tiles may ask for.
const LOGISTIC_A_MAX: f64 = 1.65;
/// Coupled-logistic phase diagram sweeps D over [0, 0.3]
/// (`src/dynachaos/maps/coupled_logistic.py:184`).
const LOGISTIC_D_MIN: f64 = 0.0;
/// Largest coupled-logistic D those tiles may ask for.
const LOGISTIC_D_MAX: f64 = 0.3;
/// Basin figure parameter (`src/dynachaos/maps/coupled_logistic.py:381`).
const BASIN_A_PAPER: f64 = 1.35344;
/// Gallery coupling (`src/dynachaos/maps/coupled_logistic.py:268`).
const LOGISTIC_D_PAPER: f64 = 0.1;
/// Phase-diagram and animation start (`src/dynachaos/maps/coupled_logistic.py:194`).
const PHASE_X0: f64 = 0.1;
/// Second component of that start.
const PHASE_Y0: f64 = 0.2;
/// Reference-orbit start (`src/dynachaos/maps/coupled_logistic.py:403`).
const BASIN_X_REF: f64 = 0.1;
/// Second component of the reference-orbit start.
const BASIN_Y_REF: f64 = 0.6;
/// Largest phase-diagram sample count from the kernel list.
const MAX_PHASE_SAMPLE: usize = 200_000;
/// Largest A list `coupled_logistic_attractor_tile` may sweep.
const MAX_ATTRACTOR_A: usize = 64;
/// Largest `D_B` list `coupled_delayed_projection_tile` may sweep.
const MAX_PROJECTION_DB: usize = 32;
/// Largest D list `fractalization_attractor_tile` may sweep.
const MAX_FRACTAL_D: usize = 32;
/// Largest basin period from the kernel list.
const MAX_BASIN_PERIOD: usize = 64;
/// Largest reference-orbit transient. The paper uses 500_000; the kernel
/// list caps the browser at 50_000.
const MAX_REFERENCE_TRANSIENT: usize = 50_000;
/// Fractalization animation sweeps D over [1.75, 1.96]
/// (`src/dynachaos/maps/fractalization.py:250`).
const FRACTAL_D_MIN: f64 = 1.75;
/// Largest fractalization D the browser may ask for.
const FRACTAL_D_MAX: f64 = 1.96;

/// Rotation numbers of the sine circle map over a tile of the (Omega, K) plane.
///
/// The map is `theta -> theta + Omega + K * sin(2 * pi * theta)`, iterated
/// without reduction modulo 1. It is the map
/// `dynachaos.maps.arnold_tongues` sweeps to draw the Arnold tongues, and in
/// this convention it stops being invertible at `K = 1 / (2 * pi)`, about
/// 0.159.
///
/// # Returned layout
///
/// One flat array of `4 + n_k * n_omega` values:
///
/// - `[0]` = `n_omega` actually used, after clamping.
/// - `[1]` = `n_k` actually used, after clamping.
/// - `[2]` = `n_transient` actually used.
/// - `[3]` = `n_iter` actually used, which may be lower than requested when
///   the step budget binds.
/// - `[4 ..]` = the rotation numbers, row-major, `n_k` rows of `n_omega`
///   values. K varies down the rows, Omega along the columns, matching the
///   `rho` array the reproduction pipeline stores.
///
/// Read the header rather than assuming the values you passed were honoured.
/// Every input is clamped, so a request outside the documented range comes
/// back as the nearest allowed one instead of an error.
///
/// # Clamping
///
/// - `n_omega`, `n_k`: 1 to 512 cells.
/// - `n_transient`: 0 to 20000 steps.
/// - `n_iter`: 1 to 200000 steps, reduced further to respect the step budget.
/// - `omega_min`, `omega_max`, `k_min`, `k_max`, `theta0`: a non-finite value
///   falls back to the value the published figure uses.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn rotation_number_tile(
    omega_min: f64,
    omega_max: f64,
    n_omega: usize,
    k_min: f64,
    k_max: f64,
    n_k: usize,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> Vec<f64> {
    let omega_min = finite_or(omega_min, 0.0);
    let omega_max = finite_or(omega_max, 1.0);
    let k_min = finite_or(k_min, 0.0);
    let k_max = finite_or(k_max, 0.3);
    let theta0 = finite_or(theta0, 0.1);

    let n_omega = n_omega.clamp(1, MAX_SIDE);
    let n_k = n_k.clamp(1, MAX_SIDE);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(n_omega, n_k, n_transient, n_iter);

    // The kernel only fails on arguments this function has already ruled out,
    // so an error here would be a bug in the clamping above. Report it as an
    // empty tile rather than trapping and killing the worker.
    let Ok(rho) = dynachaos_core::rotation_number_tile(
        omega_min,
        omega_max,
        n_omega,
        k_min,
        k_max,
        n_k,
        n_transient,
        n_iter,
        theta0,
    ) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(HEADER + rho.len());
    out.push(n_omega as f64);
    out.push(n_k as f64);
    out.push(n_transient as f64);
    out.push(n_iter as f64);
    out.extend_from_slice(&rho);
    out
}

/// Rotation number of one (Omega, K) point, lock detection on, display stop off.
///
/// This is the value the live figure quotes on hover. Locked orbits return the
/// exact rational; unlocked orbits run the full `n_iter` average. The tile
/// export keeps the display-tolerance stop; this one does not.
///
/// # Returned layout
///
/// One `f64`: `[0]` is the rotation number after clamping.
///
/// # Clamping
///
/// - `n_transient`: 0 to 20000 steps.
/// - `n_iter`: 1 to 200000 steps, reduced further to respect the step budget
///   for a single cell.
/// - `omega`, `k`, `theta0`: a non-finite value falls back to the published
///   figure's defaults (0, 0, 0.1).
#[wasm_bindgen]
pub fn rotation_number_point(
    omega: f64,
    k: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> Vec<f64> {
    let omega = finite_or(omega, 0.0);
    let k = finite_or(k, 0.0);
    let theta0 = finite_or(theta0, 0.1);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(1, 1, n_transient, n_iter);
    vec![dynachaos_core::rotation_number_point(
        omega,
        k,
        n_transient,
        n_iter,
        theta0,
    )]
}

/// The 0-1 test for chaos: one K per frequency in `c_values`.
///
/// This is the same estimator `dynachaos.diagnostics.zero_one_test` runs in
/// Python: per frequency `c` it builds the translation variables
/// `p_n = Σ φ_j cos(jc)`, `q_n = Σ φ_j sin(jc)`, forms the mean-square
/// displacement from the autocovariance identity, and returns the Pearson
/// correlation of the lag with `D`. The caller draws the frequencies; the
/// kernel only evaluates them.
///
/// # Returned layout
///
/// One flat array of `3 + n_c` values:
///
/// - `[0]` = `N` actually used, after truncation.
/// - `[1]` = `n_c` actually used, after truncation.
/// - `[2]` = `n_cut` actually used, after clamping to `[2, N]`.
/// - `[3 ..]` = one K per c value kept, in the order they were passed.
/// Read the header rather than assuming the values you passed were honoured.
/// An oversized request is cut, not refused. A request that cannot produce a
/// statistic at all — a non-finite sample or frequency, fewer than three
/// samples, or no frequencies — returns an empty array, never a trap.
///
/// # Clamping
///
/// - `phi`: truncated to 20000 samples; a non-finite sample anywhere in the
///   kept prefix returns an empty array.
/// - `c_values`: truncated to 100 frequencies; a non-finite frequency
///   anywhere in the kept prefix returns an empty array.
/// - `n_cut`: clamped to `[2, N]`. The kernel forms the autocovariance by
///   FFT, so a call costs `n_c` transforms of a power-of-two length below
///   `4N` and needs no further work budget.
#[wasm_bindgen]
pub fn zero_one_k(phi: &[f64], c_values: &[f64], n_cut: usize) -> Vec<f64> {
    let n = phi.len().min(MAX_ZERO_ONE_N);
    let n_c = c_values.len().min(MAX_ZERO_ONE_C);
    if n < 3 || n_c == 0 {
        return Vec::new();
    }
    let phi = &phi[..n];
    let c_values = &c_values[..n_c];
    if phi.iter().any(|v| !v.is_finite()) || c_values.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }

    let n_cut = n_cut.clamp(2, n);

    // The kernel only fails on arguments this function has already ruled out,
    // so an error here would be a bug in the clamping above. Report it as an
    // empty result rather than trapping and killing the worker.
    let Ok(k_values) = dynachaos_core::zero_one_k(phi, c_values, n_cut) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(ZERO_ONE_HEADER + k_values.len());
    out.push(n as f64);
    out.push(n_c as f64);
    out.push(n_cut as f64);
    out.extend_from_slice(&k_values);
    out
}

/// Correlation integral counts: pairs within each distance threshold.
///
/// This is the same estimator `dynachaos.diagnostics.correlation_integral`
/// runs in Python: `traj` is a row-major embedded trajectory of `n_pts` rows
/// of `dim` coordinates, and `counts[k]` is the number of pairs with
/// `|i - j| > theiler_window` and distance below `r_values[k]`. The kernel
/// requires the thresholds sorted ascending, so this export sorts a copy and
/// returns the radii it actually used; the counts line up with that list,
/// not with the caller's order.
///
/// # Returned layout
///
/// One flat array of `5 + n_r + n_r` values:
///
/// - `[0]` = `n_pts` actually used, after truncation.
/// - `[1]` = `dim` actually used, after clamping.
/// - `[2]` = `n_r` actually used, after truncation.
/// - `[3]` = `theiler_window` actually used, after clamping.
/// - `[4]` = `use_chebyshev` actually used: 1 for the max norm, 0 for
///   Euclidean.
/// - `[5 .. 5 + n_r]` = the radii actually used, sorted ascending.
/// - `[5 + n_r ..]` = one count per radius, in the same order.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — fewer than one complete
/// row, no radii, or a non-finite entry anywhere in the kept trajectory or
/// radius prefix — returns an empty array, never a trap.
///
/// # Clamping
///
/// - `traj`: `n_pts = traj.len() / dim` complete rows, truncated to 20000;
///   a non-finite entry anywhere in the kept prefix returns an empty array.
/// - `dim`: clamped to `[1, 32]`.
/// - `r_values`: truncated to 256 entries; a non-finite entry in the kept
///   prefix returns an empty array; negative radii are clamped to 0 and the
///   list is sorted ascending.
/// - `theiler_window`: clamped to `n_pts - 1`.
/// - `n_pts` is reduced further until `pairs x dim` fits the 2e8 pair-term
///   budget; the value actually used is reported in the header.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn correlation_counts(
    traj: &[f64],
    dim: usize,
    r_values: &[f64],
    theiler_window: usize,
    use_chebyshev: bool,
) -> Vec<f64> {
    let dim = dim.clamp(1, MAX_DIAG_DIM);
    let mut n_pts = (traj.len() / dim).min(MAX_DIAG_N);
    if n_pts == 0 {
        return Vec::new();
    }
    let theiler_window = theiler_window.min(n_pts - 1);
    n_pts = fit_diag_budget(n_pts, dim, theiler_window, false);
    let theiler_window = theiler_window.min(n_pts - 1);
    let traj = &traj[..n_pts * dim];
    if traj.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }

    let mut r_used: Vec<f64> = r_values[..r_values.len().min(MAX_DIAG_R)].to_vec();
    if r_used.is_empty() || r_used.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    for r in &mut r_used {
        *r = r.max(0.0);
    }
    r_used.sort_by(f64::total_cmp);

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok(counts) = dynachaos_core::correlation_counts(
        traj,
        n_pts,
        dim,
        &r_used,
        theiler_window,
        use_chebyshev,
    ) else {
        return Vec::new();
    };

    let n_r = r_used.len();
    let mut out = Vec::with_capacity(CORRELATION_HEADER + n_r + counts.len());
    out.push(n_pts as f64);
    out.push(dim as f64);
    out.push(n_r as f64);
    out.push(theiler_window as f64);
    out.push(if use_chebyshev { 1.0 } else { 0.0 });
    out.extend_from_slice(&r_used);
    out.extend(counts.iter().map(|&c| c as f64));
    out
}

/// Approximate-entropy template matches: one count per template row.
///
/// This is the same estimator `dynachaos.diagnostics.approximate_entropy`
/// runs in Python: `traj` is a row-major embedded trajectory of `n_pts` rows
/// of `dim` coordinates, and `counts[i]` is the number of rows `j` with
/// Chebyshev distance `max(abs(traj[i] - traj[j])) <= r`, self-matches
/// included as Pincus (1991) defines them.
///
/// # Returned layout
///
/// One flat array of `3 + n_pts` values:
///
/// - `[0]` = `n_pts` actually used, after truncation.
/// - `[1]` = `dim` actually used, after clamping.
/// - `[2]` = `r` actually used, after the fallback below.
/// - `[3 ..]` = one match count per template row, in row order.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — fewer than one complete
/// row, a non-finite entry anywhere in the kept trajectory, or a tolerance
/// that stays non-positive after the fallback — returns an empty array,
/// never a trap.
///
/// # Clamping
///
/// - `traj`: `n_pts = traj.len() / dim` complete rows, truncated to 20000;
///   a non-finite entry anywhere in the kept prefix returns an empty array.
/// - `dim`: clamped to `[1, 32]`.
/// - `r`: a non-finite or non-positive value falls back to `0.2 * std`
///   (ddof = 1) over all coordinates of the kept trajectory. The Python
///   diagnostics take the same fraction of the scalar series, which gives a
///   slightly different value.
/// - `n_pts` is reduced further until `n_pts^2 x dim` fits the 2e8
///   pair-term budget; the value actually used is reported in the header.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn apen_counts(traj: &[f64], dim: usize, r: f64) -> Vec<f64> {
    let dim = dim.clamp(1, MAX_DIAG_DIM);
    let mut n_pts = (traj.len() / dim).min(MAX_DIAG_N);
    if n_pts == 0 {
        return Vec::new();
    }
    n_pts = fit_diag_budget(n_pts, dim, 0, true);
    let traj = &traj[..n_pts * dim];
    if traj.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let Some(r) = tolerance_or_default(r, traj) else {
        return Vec::new();
    };

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok(counts) = dynachaos_core::apen_counts(traj, n_pts, dim, r) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(APEN_HEADER + counts.len());
    out.push(n_pts as f64);
    out.push(dim as f64);
    out.push(r);
    out.extend(counts.iter().map(|&c| c as f64));
    out
}

/// Fuzzy-entropy membership sum over the valid template pairs.
///
/// This is the same accumulator `dynachaos.diagnostics.fuzzy_entropy` runs
/// in Python: `traj` is a row-major embedded trajectory of `n_pts` rows of
/// `dim` coordinates, and the result is the sum of `exp(-(d / r)^n)` over
/// the pairs with `j > i + theiler_window`, Chebyshev distance `d`. The
/// Python diagnostic mean-centres its templates before calling the kernel;
/// this export sums over the trajectory it is given, so centre first if
/// that is the comparison you want.
///
/// # Returned layout
///
/// One flat array of `6` values:
///
/// - `[0]` = `n_pts` actually used, after truncation.
/// - `[1]` = `dim` actually used, after clamping.
/// - `[2]` = `r` actually used, after the fallback below.
/// - `[3]` = `n` actually used, after clamping.
/// - `[4]` = `theiler_window` actually used, after clamping.
/// - `[5]` = the membership sum.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — fewer than one complete
/// row, a non-finite entry anywhere in the kept trajectory, or a tolerance
/// that stays non-positive after the fallback — returns an empty array,
/// never a trap.
///
/// # Clamping
///
/// - `traj`: `n_pts = traj.len() / dim` complete rows, truncated to 20000;
///   a non-finite entry anywhere in the kept prefix returns an empty array.
/// - `dim`: clamped to `[1, 32]`.
/// - `r`: a non-finite or non-positive value falls back to `0.2 * std`
///   (ddof = 1) over all coordinates of the kept trajectory. The Python
///   diagnostics take the same fraction of the scalar series, which gives a
///   slightly different value.
/// - `n`: clamped to `[1, 16]`.
/// - `theiler_window`: clamped to `n_pts - 1`.
/// - `n_pts` is reduced further until `pairs x dim` fits the 2e8 pair-term
///   budget; the value actually used is reported in the header.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn fuzzy_entropy_sum(
    traj: &[f64],
    dim: usize,
    r: f64,
    n: u32,
    theiler_window: usize,
) -> Vec<f64> {
    let dim = dim.clamp(1, MAX_DIAG_DIM);
    let mut n_pts = (traj.len() / dim).min(MAX_DIAG_N);
    if n_pts == 0 {
        return Vec::new();
    }
    let theiler_window = theiler_window.min(n_pts - 1);
    n_pts = fit_diag_budget(n_pts, dim, theiler_window, false);
    let theiler_window = theiler_window.min(n_pts - 1);
    let traj = &traj[..n_pts * dim];
    if traj.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let Some(r) = tolerance_or_default(r, traj) else {
        return Vec::new();
    };
    let n = n.clamp(1, MAX_FUZZY_N);

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok(sum) = dynachaos_core::fuzzy_entropy_sum(traj, n_pts, dim, r, n, theiler_window) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(FUZZY_HEADER + 1);
    out.push(n_pts as f64);
    out.push(dim as f64);
    out.push(r);
    out.push(n as f64);
    out.push(theiler_window as f64);
    out.push(sum);
    out
}

/// Ordinal pattern distribution: raw Lehmer-code counts of length `d!`.
///
/// This is the same estimator `dynachaos.diagnostics.ordinal_distribution`
/// runs in Python: each sliding window of `d` samples spaced `tau` apart is
/// encoded as the Lehmer code of its argsort permutation, and `counts[k]` is
/// the number of windows that landed on code `k`.
///
/// # Returned layout
///
/// One flat array of `4 + d!` values:
///
/// - `[0]` = `N` actually used, after truncation.
/// - `[1]` = `d` actually used, after clamping.
/// - `[2]` = `tau` actually used, after clamping.
/// - `[3]` = `n_windows` actually analysed.
/// - `[4 ..]` = `d!` counts in Lehmer-code order.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — a non-finite sample
/// anywhere in the kept prefix, or a series too short for a single window —
/// returns an empty array, never a trap.
///
/// # Clamping
///
/// - `x`: truncated to 20000 samples; a non-finite sample anywhere in the
///   kept prefix returns an empty array.
/// - `d`: clamped to `[2, 8]` (the kernel accepts 10, but `10!` counts will
///   not fit a browser message).
/// - `tau`: clamped to `[1, N]`.
/// - The sample cap is the work budget: at most 20000 windows of at most 8
///   samples each, sized for interactive use.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn ordinal_distribution(x: &[f64], d: usize, tau: usize) -> Vec<f64> {
    let n = x.len().min(MAX_DIAG_N);
    if n == 0 {
        return Vec::new();
    }
    let x = &x[..n];
    if x.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let d = d.clamp(2, MAX_ORDINAL_D);
    let tau = tau.clamp(1, n);

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok((counts, n_windows)) = dynachaos_core::ordinal_distribution(x, d, tau) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(ORDINAL_HEADER + counts.len());
    out.push(n as f64);
    out.push(d as f64);
    out.push(tau as f64);
    out.push(n_windows as f64);
    out.extend(counts.iter().map(|&c| c as f64));
    out
}

/// Diagonal line lengths of a recurrence matrix (determinism runs).
///
/// This is the same estimator `dynachaos.diagnostics.recurrence` runs in
/// Python: `mask` is a row-major `u8` recurrence matrix of `side` rows and
/// `side` columns where any nonzero byte counts as recurrent, and the result
/// is the length of every run of recurrent cells along the super-diagonals
/// `k = 1 .. side` that meets `l_min`. The kernel is the native
/// `diagonal_lines` itself: this export only rebuilds the mask as an
/// `Array2<bool>` and hands the kernel its view, so the browser runs the
/// same scan the paper used; `scripts/check_wasm_diagnostics.py` checks
/// the two agree exactly.
///
/// # Returned layout
///
/// One flat array of `3 + n_lines` values:
///
/// - `[0]` = `side` actually used, after clamping.
/// - `[1]` = `l_min` actually used, after clamping.
/// - `[2]` = `n_lines`, the number of line lengths returned.
/// - `[3 ..]` = the line lengths, in super-diagonal order `k = 1 .. side`.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — a mask shorter than
/// `side * side`, or a side that clamps below 2 — returns an empty array,
/// never a trap.
///
/// # Clamping
///
/// - `side`: clamped to `[2, 1024]` so the `side * side` mask stays
///   interactive (at 1024 the mask is one megabyte and the scan is about a
///   million cell visits).
/// - `l_min`: clamped to `[1, side]`.
/// - `mask`: the first `side * side` bytes are used; a nonzero byte is
///   recurrent. The `O(side^2)` scan is the work budget.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn diagonal_lines(mask: &[u8], side: usize, l_min: usize) -> Vec<f64> {
    let side = side.clamp(2, MAX_RQA_SIDE);
    if mask.len() < side * side {
        return Vec::new();
    }
    let l_min = l_min.clamp(1, side);
    let mask = &mask[..side * side];

    // Building the matrix only fails on a shape mismatch and the kernel only
    // fails on a zero minimum or a non-square matrix, all ruled out by the
    // clamps above, so an error here would be a bug in the clamping. Report
    // it as an empty result rather than trapping and killing the worker.
    let Ok(r) = Array2::from_shape_vec(
        (side, side),
        mask.iter().map(|&b| b != 0).collect::<Vec<bool>>(),
    ) else {
        return Vec::new();
    };
    let Ok(lengths) = dynachaos_core::diagonal_lines(r.view(), l_min) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(LINES_HEADER + lengths.len());
    out.push(side as f64);
    out.push(l_min as f64);
    out.push(lengths.len() as f64);
    out.extend(lengths.iter().map(|&v| v as f64));
    out
}

/// Vertical line lengths of a recurrence matrix (laminarity runs).
///
/// This is the same estimator `dynachaos.diagnostics.recurrence` runs in
/// Python: `mask` is a row-major `u8` recurrence matrix of `side` rows and
/// `side` columns where any nonzero byte counts as recurrent, and the result
/// is the length of every run of recurrent cells down each column that meets
/// `v_min`. The kernel is the native `vertical_lines` itself: this export
/// only rebuilds the mask as an `Array2<bool>` and hands the kernel its
/// view, so the browser runs the same scan the paper used;
/// `scripts/check_wasm_diagnostics.py` checks the two agree exactly.
///
/// # Returned layout
///
/// One flat array of `3 + n_lines` values:
///
/// - `[0]` = `side` actually used, after clamping.
/// - `[1]` = `v_min` actually used, after clamping.
/// - `[2]` = `n_lines`, the number of line lengths returned.
/// - `[3 ..]` = the line lengths, in column order `j = 0 .. side`.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — a mask shorter than
/// `side * side`, or a side that clamps below 2 — returns an empty array,
/// never a trap.
///
/// # Clamping
///
/// - `side`: clamped to `[2, 1024]` so the `side * side` mask stays
///   interactive (at 1024 the mask is one megabyte and the scan is about a
///   million cell visits).
/// - `v_min`: clamped to `[1, side]`.
/// - `mask`: the first `side * side` bytes are used; a nonzero byte is
///   recurrent. The `O(side^2)` scan is the work budget.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn vertical_lines(mask: &[u8], side: usize, v_min: usize) -> Vec<f64> {
    let side = side.clamp(2, MAX_RQA_SIDE);
    if mask.len() < side * side {
        return Vec::new();
    }
    let v_min = v_min.clamp(1, side);
    let mask = &mask[..side * side];

    // Building the matrix only fails on a shape mismatch and the kernel only
    // fails on a zero minimum or a non-square matrix, all ruled out by the
    // clamps above, so an error here would be a bug in the clamping. Report
    // it as an empty result rather than trapping and killing the worker.
    let Ok(r) = Array2::from_shape_vec(
        (side, side),
        mask.iter().map(|&b| b != 0).collect::<Vec<bool>>(),
    ) else {
        return Vec::new();
    };
    let Ok(lengths) = dynachaos_core::vertical_lines(r.view(), v_min) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(LINES_HEADER + lengths.len());
    out.push(side as f64);
    out.push(v_min as f64);
    out.push(lengths.len() as f64);
    out.extend(lengths.iter().map(|&v| v as f64));
    out
}

/// Multifractal partition moments over dyadic box scales.
///
/// This is the same estimator `dynachaos.diagnostics.multifractal` runs in
/// Python: `field` is a row-major nonnegative measure field of `ny` rows and
/// `nx` columns, and for each box size `r` and moment order `q` the kernel
/// returns `log_z = ln(sum p^q)`, `alpha_num = sum mu ln p` and
/// `f_num = sum mu ln mu`, plus `ln(r)` per scale. Boxes do not overlap and
/// edge remainders are truncated, exactly as the native kernel does.
///
/// # Returned layout
///
/// One flat array of `4 + 3 * n_scales * n_q + n_scales` values:
///
/// - `[0]` = `ny` actually used, after clamping.
/// - `[1]` = `nx` actually used, after clamping.
/// - `[2]` = `n_scales` actually used, after truncation.
/// - `[3]` = `n_q` actually used, after truncation.
/// - `[4 .. 4 + n_scales * n_q]` = `log_z`, row-major `(n_scales, n_q)`.
/// - then `alpha_num`, row-major `(n_scales, n_q)`.
/// - then `f_num`, row-major `(n_scales, n_q)`.
/// - then `ln_scales`, `n_scales` values.
///
/// A skipped scale or a non-finite `q` leaves `NaN` in its slot, matching the
/// native kernel. Read the header rather than assuming the values you passed
/// were honoured. A request that cannot produce a result at all — a field
/// shorter than `ny * nx`, a non-finite or negative entry, or a non-positive
/// total mass — returns an empty array, never a trap.
///
/// # Clamping
///
/// - `ny`, `nx`: clamped to `[1, 512]` so the `ny * nx` field stays
///   interactive.
/// - `box_sizes`: truncated to 64 entries; each is rounded toward zero and a
///   non-positive or too-large size leaves `NaN` in its row.
/// - `q_values`: truncated to 64 entries; a non-finite `q` leaves `NaN` in
///   its column.
/// - The `O(ny * nx)` box accumulation is the work budget.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn multifractal_moments(
    field: &[f64],
    ny: usize,
    nx: usize,
    box_sizes: &[f64],
    q_values: &[f64],
) -> Vec<f64> {
    let ny = ny.clamp(1, MAX_MF_SIDE);
    let nx = nx.clamp(1, MAX_MF_SIDE);
    if field.len() < ny * nx {
        return Vec::new();
    }
    let field = &field[..ny * nx];

    let n_scales = box_sizes.len().min(MAX_MF_BOXES);
    let n_q = q_values.len().min(MAX_MF_Q);
    if n_scales == 0 || n_q == 0 {
        return Vec::new();
    }
    let box_sizes: Vec<i64> = box_sizes[..n_scales]
        .iter()
        .map(|&b| if b.is_finite() { b as i64 } else { 0 })
        .collect();
    let q_values = &q_values[..n_q];

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok(m) = dynachaos_core::multifractal_moments(field, ny, nx, &box_sizes, q_values) else {
        return Vec::new();
    };

    let block = n_scales * n_q;
    let mut out = Vec::with_capacity(MULTIFRACTAL_HEADER + 3 * block + n_scales);
    out.push(ny as f64);
    out.push(nx as f64);
    out.push(n_scales as f64);
    out.push(n_q as f64);
    out.extend(m.log_z.iter().copied());
    out.extend(m.alpha_num.iter().copied());
    out.extend(m.f_num.iter().copied());
    out.extend(m.ln_scales.iter().copied());
    out
}

/// Average mutual information `I(tau)` for `tau = 1 ..= tau_max`.
///
/// This is the same estimator `dynachaos.diagnostics` runs in Python: `x` is
/// a scalar series and the result is the delayed mutual information at each
/// lag, estimated with a uniform `n_bins`-by-`n_bins` histogram as Fraser and
/// Swinney (1986) describe.
///
/// # Returned layout
///
/// One flat array of `3 + tau_max` values:
///
/// - `[0]` = `N` actually used, after truncation.
/// - `[1]` = `tau_max` actually used, after clamping.
/// - `[2]` = `n_bins` actually used, after clamping.
/// - `[3 ..]` = `I(1) .. I(tau_max)`, in lag order.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — fewer than two samples,
/// or a non-finite sample anywhere in the kept prefix — returns an empty
/// array, never a trap.
///
/// # Clamping
///
/// - `x`: truncated to 20000 samples; a non-finite sample anywhere in the
///   kept prefix returns an empty array.
/// - `tau_max`: clamped to `[1, 512]`.
/// - `n_bins`: clamped to `[1, 512]`.
/// - The `tau_max` histogram passes over `N` samples are the work budget.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
pub fn ami_histogram(x: &[f64], tau_max: usize, n_bins: usize) -> Vec<f64> {
    let n = x.len().min(MAX_DIAG_N);
    if n < 2 {
        return Vec::new();
    }
    let x = &x[..n];
    if x.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let tau_max = tau_max.clamp(1, MAX_AMI_TAU);
    let n_bins = n_bins.clamp(1, MAX_AMI_BINS);

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty result rather than trapping and killing the worker.
    let Ok(mi) = dynachaos_core::ami_histogram(x, tau_max, n_bins) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(AMI_HEADER + mi.len());
    out.push(n as f64);
    out.push(tau_max as f64);
    out.push(n_bins as f64);
    out.extend(mi);
    out
}

/// Cao's embedding-dimension selector from an E1(d) curve.
///
/// This is the same selector `dynachaos.diagnostics` runs in Python: `e1` is
/// the E1 curve indexed by dimension, and the result is the chosen embedding
/// dimension — the onset of a stable near-1 plateau, else the first near-one
/// crossing, else the closest value to 1.
///
/// # Returned layout
///
/// One flat array of `4` values:
///
/// - `[0]` = `N`, the E1 length actually used, after truncation.
/// - `[1]` = `min_dim` actually used, after clamping.
/// - `[2]` = `max_dim` actually used: `0` when the caller left it automatic,
///   else the clamped bound.
/// - `[3]` = the selected dimension.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A request that cannot produce a result at all — an empty E1 curve —
/// returns an empty array, never a trap.
///
/// # Clamping
///
/// - `e1`: truncated to 256 entries; a non-finite entry is skipped by the
///   selector, matching the native kernel.
/// - `near_one_lower`, `near_one_upper`, `saturation_tol`: a non-finite
///   value falls back to the published defaults (0.95, 1.05, 0.02).
/// - `plateau_span`: clamped to `[2, N]`; `smoothing_window` to `[1, N]`;
///   `min_dim` to `[1, N]`; `max_dim` to `[min_dim, N]` when set.
/// - The single `O(N)` pass is the work budget.
///
/// The browser is for exploration; the Python package is for production runs.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn select_dimension_cao(
    e1: &[f64],
    near_one_lower: f64,
    near_one_upper: f64,
    saturation_tol: f64,
    plateau_span: usize,
    smoothing_window: usize,
    min_dim: usize,
    max_dim: f64,
) -> Vec<f64> {
    let n = e1.len().min(MAX_CAO_N);
    if n == 0 {
        return Vec::new();
    }
    let e1 = &e1[..n];
    let near_one_lower = finite_or(near_one_lower, 0.95);
    let near_one_upper = finite_or(near_one_upper, 1.05);
    let saturation_tol = finite_or(saturation_tol, 0.02);
    let plateau_span = plateau_span.clamp(2, n);
    let smoothing_window = smoothing_window.clamp(1, n);
    let min_dim = min_dim.clamp(1, n);
    let max_dim_opt = if max_dim.is_finite() && max_dim >= 1.0 {
        Some((max_dim as usize).clamp(min_dim, n))
    } else {
        None
    };

    let dim = dynachaos_core::select_dimension_cao(
        e1,
        near_one_lower,
        near_one_upper,
        saturation_tol,
        plateau_span,
        smoothing_window,
        min_dim,
        max_dim_opt,
    );

    vec![
        n as f64,
        min_dim as f64,
        max_dim_opt.map_or(0.0, |d| d as f64),
        dim as f64,
    ]
}
/// Delayed-logistic attractor tile: one `(x, y)` trajectory per D value.
///
/// This is the map `dynachaos.maps.delayed_logistic` iterates
/// (`delayed_logistic`, `src/dynachaos/maps/delayed_logistic.py:82`):
/// `x' = A * x + (1 - A) * (1 - D * y * y)`, `y' = x`. The export sweeps `n_D`
/// evenly spaced D values from `d_min` to `d_max` and, for each, iterates
/// `n_transient` steps from `state0` then records `n_plot` states.
///
/// # Returned layout
///
/// One flat array of `4 + n_D * n_plot * 2` values:
///
/// - `[0]` = `n_D` actually used, after clamping.
/// - `[1]` = `n_plot` actually used, after clamping and the step budget.
/// - `[2]` = `n_transient` actually used.
/// - `[3]` = the state dimension, always 2.
/// - `[4 ..]` = the samples, D-major: `n_D` blocks of `n_plot` `(x, y)`
///   pairs each.
///
/// A D whose orbit diverges (any `|state| > 1e10`) yields a block of `NaN`
/// pairs, matching the `None` the Python pipeline stores for that D. Read the
/// header rather than assuming the values you passed were honoured. A kernel
/// error — impossible after the clamps below — returns an empty array,
/// never a trap.
///
/// # Clamping
///
/// - `a`: clamped to `[0, 1]`; a non-finite value falls back to 0.3.
/// - `d_min`, `d_max`: clamped to `[1.4, 3.5]`, the range the paper sweeps;
///   non-finite values fall back to the full range.
/// - `n_D`: clamped to `[1, 64]`; `n_transient` to `[0, 20000]`; `n_plot` to
///   `[1, 4096]`, reduced further to respect the step budget.
/// - `state0`: the first two entries are used; a missing or non-finite
///   component falls back to 0.5, then every component clamps to `[-4, 4]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn delayed_logistic_attractor_tile(
    a: f64,
    d_min: f64,
    d_max: f64,
    n_d: usize,
    n_transient: usize,
    n_plot: usize,
    state0: &[f64],
) -> Vec<f64> {
    let a = finite_or(a, 0.3).clamp(0.0, 1.0);
    let d_min = finite_or(d_min, DELAYED_D_MIN).clamp(DELAYED_D_MIN, DELAYED_D_MAX);
    let d_max = finite_or(d_max, DELAYED_D_MAX).clamp(DELAYED_D_MIN, DELAYED_D_MAX);
    let n_d = n_d.clamp(1, MAX_MAP_D_LIST);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_plot = n_plot.clamp(1, MAX_MAP_PLOT);
    let n_plot = fit_step_budget(n_d, 1, n_transient, n_plot);
    let start = map_state::<2>(state0);

    // Same spacing the kernel list fixes: d_min + (d_max - d_min) * k / (n - 1).
    let d_values: Vec<f64> = (0..n_d)
        .map(|k| d_min + (d_max - d_min) * (k as f64) / ((n_d - 1).max(1) as f64))
        .collect();

    // The kernel only fails on arguments this function has already ruled out,
    // so an error here would be a bug in the clamping above. Report it as an
    // empty tile rather than trapping and killing the worker.
    let Ok(samples) =
        dynachaos_core::delayed_logistic_attractor_tile(a, &d_values, n_transient, n_plot, &start)
    else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(HEADER + samples.len());
    out.push(n_d as f64);
    out.push(n_plot as f64);
    out.push(n_transient as f64);
    out.push(2.0);
    out.extend_from_slice(&samples);
    out
}

/// Torus-doubling attractor tile: one trajectory of map I or map IV.
///
/// These are the maps `dynachaos.maps.torus_doubling` iterates
/// (`src/dynachaos/maps/torus_doubling.py:46,63`): with
/// `L_D(u) = 1 - D * u * u`, map I (`map_kind` 1) steps `(X, Y, Z)` to
/// `(A * X + (1 - A) * L_D(Y), Z, X)` and map IV (`map_kind` 4) steps
/// `(X, Y, Z, W)` to `(A * X + (1 - A) * L_D(Y), Z, A * Z + (1 - A) * L_D(W), X)`.
/// The export iterates `n_transient` steps from `state0`, then records up to
/// `n_plot` states.
///
/// # Returned layout
///
/// One flat array of `4 + n_produced * dim` values:
///
/// - `[0]` = `map_kind` actually used, after snapping to `{1, 4}`.
/// - `[1]` = the state dimension: 3 for map I, 4 for map IV.
/// - `[2]` = `n_transient` actually used.
/// - `[3]` = `n_produced`, the number of samples actually recorded.
/// - `[4 ..]` = the samples, `dim` values each, in iteration order.
///
/// Divergence (any `|state| > 1e10`) stops the record and returns the samples
/// produced so far — possibly zero when the transient itself diverged —
/// exactly as `iterate_map` does in Python. Read the header rather than
/// assuming `n_plot` samples came back. A kernel error — impossible after
/// the clamps below — returns an empty array, never a trap.
///
/// # Clamping
///
/// - `map_kind`: snapped to the nearer of `{1, 4}`: 2 and below is map I,
///   3 and above is map IV.
/// - `a`: clamped to `[0, 1]`; a non-finite value falls back to 0.4.
/// - `d`: clamped to `[1.48, 2.25]`, the union of the paper's map-I and
///   map-IV sweeps; a non-finite value falls back to 1.9.
/// - `n_transient`: clamped to `[0, 20000]`; `n_plot` to `[1, 4096]`,
///   reduced further to respect the step budget.
/// - `state0`: the first `dim` entries are used; a missing or non-finite
///   component falls back to 0.5, then every component clamps to `[-4, 4]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn torus_doubling_attractor_tile(
    map_kind: usize,
    a: f64,
    d: f64,
    n_transient: usize,
    n_plot: usize,
    state0: &[f64],
) -> Vec<f64> {
    let map_kind = if map_kind <= 2 { 1 } else { 4 };
    let dim = if map_kind == 1 { 3 } else { 4 };
    let a = finite_or(a, 0.4).clamp(0.0, 1.0);
    let d = finite_or(d, 1.9).clamp(TORUS_D_MIN, TORUS_D_MAX);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_plot = n_plot.clamp(1, MAX_MAP_PLOT);
    let n_plot = fit_step_budget(1, 1, n_transient, n_plot);

    // The kernel only fails on arguments this function has already ruled out,
    // so an error here would be a bug in the clamping above. Report it as an
    // empty tile rather than trapping and killing the worker.
    let samples = if map_kind == 1 {
        let start = map_state::<3>(state0);
        dynachaos_core::torus_doubling_attractor_tile(1, a, d, n_transient, n_plot, &start)
    } else {
        let start = map_state::<4>(state0);
        dynachaos_core::torus_doubling_attractor_tile(4, a, d, n_transient, n_plot, &start)
    };
    let Ok(samples) = samples else {
        return Vec::new();
    };

    let n_produced = samples.len() / dim;
    let mut out = Vec::with_capacity(HEADER + samples.len());
    out.push(map_kind as f64);
    out.push(dim as f64);
    out.push(n_transient as f64);
    out.push(n_produced as f64);
    out.extend_from_slice(&samples);
    out
}

/// Modulated-circle rotation numbers over a `D` sweep.
///
/// This is the map `dynachaos.maps.modulated_circle` iterates
/// (`src/dynachaos/maps/modulated_circle.py:35`): `theta` advances by
/// `A * sin(2 pi theta) + D + eps * sin(2 pi phi)` while `phi` advances by
/// `C`, both unwrapped. For each `D` in the sweep the export iterates
/// `n_transient` steps from `(theta0, phi0)`, then returns the mean
/// increments over the next `n_iter` steps — the pair `rotation_numbers`
/// computes (`src/dynachaos/maps/modulated_circle.py:48`).
///
/// # Returned layout
///
/// One flat array of `4 + n_d * 2` values:
///
/// - `[0]` = `n_d` actually used, after clamping.
/// - `[1]` = `n_transient` actually used.
/// - `[2]` = `n_iter` actually used, which may be lower than requested when
///   the step budget binds.
/// - `[3]` = 2, the pair width.
/// - `[4 ..]` = the `(rho_theta, rho_phi)` pairs, D-major.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A kernel error — impossible after the clamps below — returns an empty
/// array, never a trap.
///
/// # Clamping
///
/// - `n_d`: 1 to 512 values; `n_transient`: 0 to 20000 steps; `n_iter`: 1 to
///   200000 steps, reduced further to respect the step budget.
/// - `a`, `c`, `eps`, `d_min`, `d_max`, `theta0`, `phi0`: clamped to
///   `[0, 1]`; a non-finite value falls back to the paper's setting (0.1,
///   the golden-mean inverse, 0.05, the sweep endpoints 0 and 1, and 0.1 for
///   the phases).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn modulated_circle_rotation_tile(
    a: f64,
    c: f64,
    d_min: f64,
    d_max: f64,
    n_d: usize,
    eps: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
    phi0: f64,
) -> Vec<f64> {
    let a = finite_or(a, 0.1).clamp(0.0, 1.0);
    let c = finite_or(c, C_GOLDEN).clamp(0.0, 1.0);
    let d_min = finite_or(d_min, 0.0).clamp(0.0, 1.0);
    let d_max = finite_or(d_max, 1.0).clamp(0.0, 1.0);
    let eps = finite_or(eps, 0.05).clamp(0.0, 1.0);
    let n_d = n_d.clamp(1, MAX_SIDE);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(n_d, 1, n_transient, n_iter);
    let state0 = [
        finite_or(theta0, 0.1).clamp(0.0, 1.0),
        finite_or(phi0, 0.1).clamp(0.0, 1.0),
    ];

    // Same spacing the kernel list fixes: d_min + (d_max - d_min) * k / (n - 1).
    let d_values: Vec<f64> = (0..n_d)
        .map(|k| d_min + (d_max - d_min) * (k as f64) / ((n_d - 1).max(1) as f64))
        .collect();

    // The kernel only fails on arguments this function has already ruled out,
    // so an error here would be a bug in the clamping above. Report it as an
    // empty tile rather than trapping and killing the worker.
    let Ok(pairs) = dynachaos_core::modulated_circle_rotation_tile(
        a,
        c,
        &d_values,
        eps,
        n_transient,
        n_iter,
        &state0,
    ) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(HEADER + pairs.len());
    out.push(n_d as f64);
    out.push(n_transient as f64);
    out.push(n_iter as f64);
    out.push(2.0);
    out.extend_from_slice(&pairs);
    out
}

/// CML space-time tile: the field of a coupled-map lattice after a transient.
///
/// These are the three models `dynachaos.cml.spatiotemporal` simulates
/// (`src/dynachaos/cml/spatiotemporal.py:42-68`): `model` 0 is (A), the
/// piecewise Kaneko map; `model` 1 is (B), the circle map coupled through
/// `sin(2 pi u)`; `model` 2 is (C), the logistic map `1 - 1.752 u^2`. The
/// export iterates `n_transient` periodic CML steps from `x0`, then records
/// the state after each of the next `n_record` steps.
///
/// # Returned layout
///
/// One flat array of `4 + n_record * n_sites` values:
///
/// - `[0]` = `model` actually used, after snapping to `{0, 1, 2}`.
/// - `[1]` = `n_sites` actually used.
/// - `[2]` = `n_transient` actually used.
/// - `[3]` = `n_record` actually used, which may be lower than requested when
///   the step budget binds.
/// - `[4 ..]` = the field, row-major by time then site: entry
///   `4 + t * n_sites + i` is site `i` of the `t`-th recorded row.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A kernel error — impossible after the clamps below — returns an empty
/// array, never a trap.
///
/// # Clamping
///
/// - `model`: snapped to `{0, 1, 2}`; anything above 2 is model (C).
/// - `eps`: clamped to `[0, 1]`; a non-finite value falls back to the middle
///   of the model's paper sweep (0.07, 0.024, 0.2).
/// - `n_sites`: 2 to 512 sites; `n_transient`: 0 to 20000 steps; `n_record`:
///   1 to 2048 rows, reduced further to respect the step budget.
/// - `x0`: the first `n_sites` entries are used; a missing or non-finite
///   entry falls back to 0.5, then every entry clamps to `[-1, 2]`. The
///   bound is wider than every observed orbit of the three models (A dips
///   to -0.11, B to -0.05, C to -0.75), so a play chunk continues from the
///   window's last row unchanged; a start far outside the invariant range
///   can diverge, and the page never supplies one.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn cml_spacetime_tile(
    model: usize,
    eps: f64,
    n_sites: usize,
    n_transient: usize,
    n_record: usize,
    x0: &[f64],
) -> Vec<f64> {
    let model = model.min(2) as u8;
    let eps = finite_or(eps, CML_EPS_FALLBACK[model as usize]).clamp(0.0, 1.0);
    let n_sites = n_sites.clamp(2, MAX_SIDE);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_record = n_record.clamp(1, MAX_CML_RECORD);
    let n_record = fit_step_budget(n_sites, 1, n_transient, n_record);
    let start = cml_state(n_sites, x0);

    // The kernel only fails on arguments this function has already ruled out,
    // so an error here would be a bug in the clamping above. Report it as an
    // empty tile rather than trapping and killing the worker.
    let Ok(field) = dynachaos_core::cml_spacetime_tile(model, eps, n_transient, n_record, &start)
    else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(HEADER + field.len());
    out.push(model as f64);
    out.push(n_sites as f64);
    out.push(n_transient as f64);
    out.push(n_record as f64);
    out.extend_from_slice(&field);
    out
}

/// Lyapunov exponent of the circle map at one `(A, D)`.
///
/// This is `dynachaos.maps.circle_map.lyapunov_exponent`
/// (`src/dynachaos/maps/circle_map.py:75`): both the transient and the
/// measured stretch use the wrapped map
/// `theta' = (theta + D + A sin(2 pi theta)) mod 1`, and a zero derivative
/// contributes `-100` instead of `log(0)`. The devils-staircase Lyapunov
/// panel draws `lambda(A)` at `D = 0.25`.
///
/// # Returned layout
///
/// One flat array of 3 values:
///
/// - `[0]` = `n_transient` actually used.
/// - `[1]` = `n_iter` actually used, which may be lower than requested when
///   the step budget binds.
/// - `[2]` = `lambda`, the exponent.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A kernel error — impossible after the clamps below — returns an empty
/// array, never a trap.
///
/// # Clamping
///
/// - `a`: clamped to `[0, 0.25]`, the staircase sweep; a non-finite value
///   falls back to 0.
/// - `d`: clamped to `[0, 1]`; a non-finite value falls back to 0.25, the
///   paper's D.
/// - `theta0`: clamped to `[0, 1]`; a non-finite value falls back to 0.1.
/// - `n_transient`: clamped to `[0, 20000]`; `n_iter` to `[1, 200000]`,
///   reduced further to respect the step budget.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn circle_map_lyapunov_sum(
    a: f64,
    d: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> Vec<f64> {
    let a = finite_or(a, 0.0).clamp(0.0, CIRCLE_A_MAX);
    let d = finite_or(d, 0.25).clamp(0.0, 1.0);
    let theta0 = finite_or(theta0, 0.1).clamp(0.0, 1.0);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(1, 1, n_transient, n_iter);

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty tile rather than trapping and killing the worker.
    let Ok(lambda) = dynachaos_core::circle_map_lyapunov_sum(a, d, n_transient, n_iter, theta0)
    else {
        return Vec::new();
    };

    vec![n_transient as f64, n_iter as f64, lambda]
}

/// Delayed-logistic Lyapunov tile: one 2-spectrum per `D`.
///
/// This is `lyapunov_spectrum` (`src/dynachaos/diagnostics/lyapunov.py:76`)
/// run on `delayed_logistic` (`src/dynachaos/maps/delayed_logistic.py:82`)
/// and its Jacobian. The Lyapunov figure sweeps `D` over `[1.3, 2.5]`
/// (`src/dynachaos/maps/delayed_logistic.py:162`).
///
/// # Returned layout
///
/// One flat array of `4 + n_D * 2` values:
///
/// - `[0]` = `n_D` actually used.
/// - `[1]` = `n_transient` actually used.
/// - `[2]` = `n_iter` actually used, which may be lower than requested when
///   the step budget binds.
/// - `[3]` = 2, the spectrum width.
/// - `[4 ..]` = the spectra, D-major: two exponents per `D`, largest first.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A kernel error — impossible after the clamps below — returns an empty
/// array, never a trap.
///
/// # Clamping
///
/// - `a`: clamped to `[0, 1]`; a non-finite value falls back to 0.3.
/// - `d_min`, `d_max`: clamped to `[1.3, 2.5]`, the Lyapunov sweep; a
///   non-finite value falls back to the matching endpoint.
/// - `n_D`: clamped to `[1, 512]`; `n_transient` to `[0, 20000]`; `n_iter`
///   to `[1, 200000]`, reduced further to respect the step budget.
/// - `state0`: the first two entries are used; a missing or non-finite
///   component falls back to 0.5, then every component clamps to `[-4, 4]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn delayed_logistic_lyapunov_tile(
    a: f64,
    d_min: f64,
    d_max: f64,
    n_d: usize,
    n_transient: usize,
    n_iter: usize,
    state0: &[f64],
) -> Vec<f64> {
    let a = finite_or(a, 0.3).clamp(0.0, 1.0);
    let d_min = finite_or(d_min, DELAYED_LYAP_D_MIN).clamp(DELAYED_LYAP_D_MIN, DELAYED_LYAP_D_MAX);
    let d_max = finite_or(d_max, DELAYED_LYAP_D_MAX).clamp(DELAYED_LYAP_D_MIN, DELAYED_LYAP_D_MAX);
    let n_d = n_d.clamp(1, MAX_LYAP_D);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(n_d, 1, n_transient, n_iter);
    let start = map_state::<2>(state0);

    let d_values: Vec<f64> = (0..n_d)
        .map(|k| d_min + (d_max - d_min) * (k as f64) / ((n_d - 1).max(1) as f64))
        .collect();

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty tile rather than trapping and killing the worker.
    let Ok(spectra) =
        dynachaos_core::delayed_logistic_lyapunov_tile(a, &d_values, n_transient, n_iter, &start)
    else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(HEADER + spectra.len());
    out.push(n_d as f64);
    out.push(n_transient as f64);
    out.push(n_iter as f64);
    out.push(2.0);
    out.extend_from_slice(&spectra);
    out
}

/// Torus-doubling Lyapunov tile: one spectrum per `D`, map I or map IV.
///
/// This is `lyapunov_spectrum` run on `map_I` or `map_IV` and its Jacobian
/// (`src/dynachaos/maps/torus_doubling.py:46,63`). The paper sweeps `D` over
/// `[1.48, 1.53]` for map IV and `[1.9, 2.25]` for map I; the export accepts
/// their union `[1.48, 2.25]`.
///
/// # Returned layout
///
/// One flat array of `4 + n_D * dim` values:
///
/// - `[0]` = `n_D` actually used.
/// - `[1]` = `n_transient` actually used.
/// - `[2]` = `n_iter` actually used, which may be lower than requested when
///   the step budget binds.
/// - `[3]` = the spectrum width: 3 for map I, 4 for map IV.
/// - `[4 ..]` = the spectra, D-major: `dim` exponents per `D`, largest first.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A kernel error — impossible after the clamps below — returns an empty
/// array, never a trap.
///
/// # Clamping
///
/// - `map_kind`: snapped to the nearer of `{1, 4}`: 2 and below is map I,
///   3 and above is map IV.
/// - `a`: clamped to `[0, 1]`; a non-finite value falls back to 0.4.
/// - `d_min`, `d_max`: clamped to `[1.48, 2.25]`, the union of the paper's
///   map-I and map-IV sweeps; a non-finite value falls back to the matching
///   endpoint.
/// - `n_D`: clamped to `[1, 512]`; `n_transient` to `[0, 20000]`; `n_iter`
///   to `[1, 200000]`, reduced further to respect the step budget.
/// - `state0`: the first `dim` entries are used; a missing or non-finite
///   component falls back to 0.5, then every component clamps to `[-4, 4]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn torus_doubling_lyapunov_tile(
    map_kind: usize,
    a: f64,
    d_min: f64,
    d_max: f64,
    n_d: usize,
    n_transient: usize,
    n_iter: usize,
    state0: &[f64],
) -> Vec<f64> {
    let map_kind = if map_kind <= 2 { 1 } else { 4 };
    let dim = if map_kind == 1 { 3 } else { 4 };
    let a = finite_or(a, 0.4).clamp(0.0, 1.0);
    let d_min = finite_or(d_min, TORUS_D_MIN).clamp(TORUS_D_MIN, TORUS_D_MAX);
    let d_max = finite_or(d_max, TORUS_D_MAX).clamp(TORUS_D_MIN, TORUS_D_MAX);
    let n_d = n_d.clamp(1, MAX_LYAP_D);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(n_d, 1, n_transient, n_iter);

    let d_values: Vec<f64> = (0..n_d)
        .map(|k| d_min + (d_max - d_min) * (k as f64) / ((n_d - 1).max(1) as f64))
        .collect();

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty tile rather than trapping and killing the worker.
    let spectra = if map_kind == 1 {
        let start = map_state::<3>(state0);
        dynachaos_core::torus_doubling_lyapunov_tile(1, a, &d_values, n_transient, n_iter, &start)
    } else {
        let start = map_state::<4>(state0);
        dynachaos_core::torus_doubling_lyapunov_tile(4, a, &d_values, n_transient, n_iter, &start)
    };
    let Ok(spectra) = spectra else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(HEADER + spectra.len());
    out.push(n_d as f64);
    out.push(n_transient as f64);
    out.push(n_iter as f64);
    out.push(dim as f64);
    out.extend_from_slice(&spectra);
    out
}

/// Coupled-delayed Lyapunov tile: one 4-spectrum per `D_B`.
///
/// This is `lyapunov_spectrum` run on `coupled_delayed` and its Jacobian
/// (`src/dynachaos/maps/coupled_delayed.py:52`), with `D_A = D_B + 0.1` as
/// `compute_lyapunov` fixes it. The paper sweeps `D_B` over `[2.1, 2.65]`
/// at `eps` in `{1e-3, 5e-3, 1e-2}`.
///
/// # Returned layout
///
/// One flat array of `4 + n_DB * 4` values:
///
/// - `[0]` = `n_DB` actually used.
/// - `[1]` = `n_transient` actually used.
/// - `[2]` = `n_iter` actually used, which may be lower than requested when
///   the step budget binds.
/// - `[3]` = 4, the spectrum width.
/// - `[4 ..]` = the spectra, `D_B`-major: four exponents per `D_B`, largest
///   first.
///
/// Read the header rather than assuming the values you passed were honoured.
/// A kernel error — impossible after the clamps below — returns an empty
/// array, never a trap.
///
/// # Clamping
///
/// - `a`: clamped to `[0, 1]`; a non-finite value falls back to 0.4.
/// - `db_min`, `db_max`: clamped to `[2.1, 2.65]`, the paper sweep; a
///   non-finite value falls back to the matching endpoint.
/// - `eps`: clamped to `[1e-3, 1e-2]`, the paper's epsilon trio; a
///   non-finite value falls back to 5e-3.
/// - `n_DB`: clamped to `[1, 128]`; `n_transient` to `[0, 20000]`; `n_iter`
///   to `[1, 200000]`, reduced further to respect the step budget.
/// - `state0`: the first four entries are used; a missing or non-finite
///   component falls back to 0.5, then every component clamps to `[-4, 4]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn coupled_delayed_lyapunov_tile(
    a: f64,
    db_min: f64,
    db_max: f64,
    n_db: usize,
    eps: f64,
    n_transient: usize,
    n_iter: usize,
    state0: &[f64],
) -> Vec<f64> {
    let a = finite_or(a, 0.4).clamp(0.0, 1.0);
    let db_min = finite_or(db_min, COUPLED_DB_MIN).clamp(COUPLED_DB_MIN, COUPLED_DB_MAX);
    let db_max = finite_or(db_max, COUPLED_DB_MAX).clamp(COUPLED_DB_MIN, COUPLED_DB_MAX);
    let n_db = n_db.clamp(1, MAX_LYAP_DB);
    let eps = finite_or(eps, COUPLED_EPS_PAPER).clamp(COUPLED_EPS_MIN, COUPLED_EPS_MAX);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_iter = n_iter.clamp(1, MAX_ITER);
    let n_iter = fit_step_budget(n_db, 1, n_transient, n_iter);
    let start = map_state::<4>(state0);

    let db_values: Vec<f64> = (0..n_db)
        .map(|k| db_min + (db_max - db_min) * (k as f64) / ((n_db - 1).max(1) as f64))
        .collect();

    // The kernel only fails on arguments this function has already ruled
    // out, so an error here would be a bug in the clamping above. Report it
    // as an empty tile rather than trapping and killing the worker.
    let Ok(spectra) = dynachaos_core::coupled_delayed_lyapunov_tile(
        a,
        &db_values,
        eps,
        n_transient,
        n_iter,
        &start,
    ) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(HEADER + spectra.len());
    out.push(n_db as f64);
    out.push(n_transient as f64);
    out.push(n_iter as f64);
    out.push(4.0);
    out.extend_from_slice(&spectra);
    out
}

/// Coupled-logistic phase tile: asymmetry and largest-tangent growth on an (A, D) grid.
///
/// Mirrors `compute_phase_diagram` (`src/dynachaos/maps/coupled_logistic.py:161`).
/// The tangent step runs before the map step. `|x| > 10` or `|y| > 10` sets
/// that state to NaN, and a cell with no valid sample is NaN.
///
/// # Returned layout
///
/// `[n_A, n_D, n_transient, n_sample, asym_0, lyap_0, ...]`, row-major in D
/// then A, two values per cell. The header reports the values actually used.
/// A kernel error — impossible after the clamps below — returns an empty
/// array, never a trap.
///
/// # Clamping
///
/// - `a_min`, `a_max`: clamped to `[0.5, 1.65]`; a non-finite value falls back
///   to the matching endpoint.
/// - `d_min`, `d_max`: clamped to `[0, 0.3]`; a non-finite value falls back to
///   the matching endpoint.
/// - `n_A`, `n_D`: clamped to `[1, 512]`; `n_transient` to `[0, 20000]`;
///   `n_sample` to `[1, 200000]`, then both counts shrink until
///   `n_A * n_D * (n_transient + n_sample) <= MAX_STEPS`.
/// - `x0`, `y0`: a non-finite value falls back to `(0.1, 0.2)`, then each
///   component clamps to `[-1, 1]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn coupled_logistic_phase_tile(
    a_min: f64,
    a_max: f64,
    n_a: usize,
    d_min: f64,
    d_max: f64,
    n_d: usize,
    n_transient: usize,
    n_sample: usize,
    x0: f64,
    y0: f64,
) -> Vec<f64> {
    let a_min = finite_or(a_min, LOGISTIC_A_MIN).clamp(LOGISTIC_A_MIN, LOGISTIC_A_MAX);
    let a_max = finite_or(a_max, LOGISTIC_A_MAX).clamp(LOGISTIC_A_MIN, LOGISTIC_A_MAX);
    let d_min = finite_or(d_min, LOGISTIC_D_MIN).clamp(LOGISTIC_D_MIN, LOGISTIC_D_MAX);
    let d_max = finite_or(d_max, LOGISTIC_D_MAX).clamp(LOGISTIC_D_MIN, LOGISTIC_D_MAX);
    let n_a = n_a.clamp(1, MAX_SIDE);
    let n_d = n_d.clamp(1, MAX_SIDE);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_sample = n_sample.clamp(1, MAX_PHASE_SAMPLE);
    let (n_transient, n_sample) = fit_pair_budget(n_a, n_d, n_transient, n_sample);
    let x0 = logistic_coord(x0, PHASE_X0);
    let y0 = logistic_coord(y0, PHASE_Y0);
    let a_values = swept_values(a_min, a_max, n_a);
    let d_values = swept_values(d_min, d_max, n_d);
    let Ok(samples) = dynachaos_core::coupled_logistic_phase_tile(
        &a_values,
        &d_values,
        n_transient,
        n_sample,
        x0,
        y0,
    ) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(HEADER + samples.len());
    out.push(n_a as f64);
    out.push(n_d as f64);
    out.push(n_transient as f64);
    out.push(n_sample as f64);
    out.extend_from_slice(&samples);
    out
}

/// Coupled-logistic attractor tile: one `(x, y)` trajectory per `A` at fixed `D`.
///
/// Mirrors `compute_attractors` (`src/dynachaos/maps/coupled_logistic.py:265`).
/// Python passes no `diverged_fn`. This kernel does the same: it records every
/// requested sample, including non-finite values, and does not apply a
/// magnitude gate.
///
/// # Returned layout
///
/// `[n_A, n_plot, n_transient, 2, x_0, y_0, ...]`, A-major `(x, y)` pairs.
/// A kernel error returns an empty array, never a trap.
///
/// # Clamping
///
/// - `a_min`, `a_max`: clamped to `[0.5, 1.65]`; a non-finite value falls back
///   to the matching endpoint.
/// - `d`: clamped to `[0, 0.3]`; a non-finite value falls back to `0.1`.
/// - `n_A`: clamped to `[1, 64]`; `n_transient` to `[0, 20000]`; `n_plot` to
///   `[1, 4096]`, then the pair shrinks to respect the step budget.
/// - `x0`, `y0`: a non-finite value falls back to `(0.1, 0.2)`, then each
///   component clamps to `[-1, 1]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn coupled_logistic_attractor_tile(
    a_min: f64,
    a_max: f64,
    n_a: usize,
    d: f64,
    n_transient: usize,
    n_plot: usize,
    x0: f64,
    y0: f64,
) -> Vec<f64> {
    let a_min = finite_or(a_min, LOGISTIC_A_MIN).clamp(LOGISTIC_A_MIN, LOGISTIC_A_MAX);
    let a_max = finite_or(a_max, LOGISTIC_A_MAX).clamp(LOGISTIC_A_MIN, LOGISTIC_A_MAX);
    let n_a = n_a.clamp(1, MAX_ATTRACTOR_A);
    let d = finite_or(d, LOGISTIC_D_PAPER).clamp(LOGISTIC_D_MIN, LOGISTIC_D_MAX);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_plot = n_plot.clamp(1, MAX_MAP_PLOT);
    let (n_transient, n_plot) = fit_pair_budget(n_a, 1, n_transient, n_plot);
    let x0 = logistic_coord(x0, PHASE_X0);
    let y0 = logistic_coord(y0, PHASE_Y0);
    let a_values = swept_values(a_min, a_max, n_a);
    let Ok(samples) = dynachaos_core::coupled_logistic_attractor_tile(
        &a_values,
        d,
        n_transient,
        n_plot,
        &[x0, y0],
    ) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(HEADER + samples.len());
    out.push(n_a as f64);
    out.push(n_plot as f64);
    out.push(n_transient as f64);
    out.push(2.0);
    out.extend_from_slice(&samples);
    out
}

/// Coupled-logistic basin grid, including the reference orbit the Python caller builds.
///
/// The reference orbit is `_find_reference_orbit`
/// (`src/dynachaos/maps/coupled_logistic.py:308`): `reference_transient` steps
/// from `(x_ref, y_ref)`, then `period` recorded states, with no divergence
/// check. The grid classification then calls the existing
/// `coupled_logistic_basin_grid`, which marks `|x| > 100` or `|y| > 100` as
/// label `-1`.
///
/// # Returned layout
///
/// `[n_x, n_y, n_transient, period, label_0, ...]`, row-major in y then x.
/// Labels are `-1, 0, 1, 2` as f64. A kernel error returns an empty array.
///
/// # Clamping
///
/// - `a`: clamped to `[0.5, 1.65]`; a non-finite value falls back to `1.35344`.
/// - `d`: clamped to `[0, 0.3]`; a non-finite value falls back to `0.1`.
/// - `x_min`, `x_max`, `y_min`, `y_max`, `x_ref`, `y_ref`: clamped to `[-1, 1]`.
///   A non-finite range endpoint falls back to the paper domain `[-1, 1]`; a
///   non-finite reference component falls back to `(0.1, 0.6)`.
/// - `n_x`, `n_y`: clamped to `[1, 512]`; `n_transient` to `[0, 20000]`;
///   `reference_transient` to `[0, 50000]`; `period` to `[1, 64]`. The pair
///   `(n_transient, reference_transient)` then shrinks until
///   `n_x * n_y * n_transient + reference_transient <= MAX_STEPS`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn coupled_logistic_basin_grid(
    a: f64,
    d: f64,
    x_min: f64,
    x_max: f64,
    n_x: usize,
    y_min: f64,
    y_max: f64,
    n_y: usize,
    n_transient: usize,
    reference_transient: usize,
    period: usize,
    x_ref: f64,
    y_ref: f64,
) -> Vec<f64> {
    let a = finite_or(a, BASIN_A_PAPER).clamp(LOGISTIC_A_MIN, LOGISTIC_A_MAX);
    let d = finite_or(d, LOGISTIC_D_PAPER).clamp(LOGISTIC_D_MIN, LOGISTIC_D_MAX);
    let x_min = logistic_coord(x_min, -1.0);
    let x_max = logistic_coord(x_max, 1.0);
    let y_min = logistic_coord(y_min, -1.0);
    let y_max = logistic_coord(y_max, 1.0);
    let n_x = n_x.clamp(1, MAX_SIDE);
    let n_y = n_y.clamp(1, MAX_SIDE);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let reference_transient = reference_transient.min(MAX_REFERENCE_TRANSIENT);
    let period = period.clamp(1, MAX_BASIN_PERIOD);
    let (n_transient, reference_transient) =
        fit_basin_budget(n_x, n_y, n_transient, reference_transient);
    let x_ref = logistic_coord(x_ref, BASIN_X_REF);
    let y_ref = logistic_coord(y_ref, BASIN_Y_REF);
    let x_values = swept_values(x_min, x_max, n_x);
    let y_values = swept_values(y_min, y_max, n_y);
    let Ok(orbit) = dynachaos_core::coupled_logistic_reference_orbit(
        a,
        d,
        x_ref,
        y_ref,
        reference_transient,
        period,
    ) else {
        return Vec::new();
    };
    let Ok(ref_a) = Array2::from_shape_vec((period, 2), orbit) else {
        return Vec::new();
    };
    let Ok(basin) = dynachaos_core::coupled_logistic_basin_grid(
        &x_values,
        &y_values,
        a,
        d,
        n_transient,
        ref_a.view(),
    ) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(HEADER + basin.len());
    out.push(n_x as f64);
    out.push(n_y as f64);
    out.push(n_transient as f64);
    out.push(period as f64);
    out.extend(basin.iter().copied().map(|label| label as f64));
    out
}

/// Coupled-delayed projection tile: one `(x, z)` trajectory per `D_B`.
///
/// Mirrors `compute_projections` (`src/dynachaos/maps/coupled_delayed.py:123`)
/// with `D_A = D_B + 0.1`. Python passes no `diverged_fn`. This kernel does
/// the same: it records every requested `(x, z)` sample, including non-finite
/// values, and does not apply a magnitude gate.
///
/// # Returned layout
///
/// `[n_DB, n_plot, n_transient, 2, x_0, z_0, ...]`, D-major projected pairs.
/// A kernel error returns an empty array, never a trap.
///
/// # Clamping
///
/// - `a`: clamped to `[0, 1]`; a non-finite value falls back to `0.4`.
/// - `db_min`, `db_max`: clamped to `[2.1, 2.65]`; a non-finite value falls
///   back to the matching endpoint.
/// - `eps`: clamped to `[1e-3, 1e-2]`; a non-finite value falls back to `5e-3`.
/// - `n_DB`: clamped to `[1, 32]`; `n_transient` to `[0, 20000]`; `n_plot` to
///   `[1, 4096]`, then the pair shrinks to respect the step budget.
/// - `state0`: the first four entries are used; a missing or non-finite
///   component falls back to `0.5`, then every component clamps to `[-4, 4]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn coupled_delayed_projection_tile(
    a: f64,
    db_min: f64,
    db_max: f64,
    n_db: usize,
    eps: f64,
    n_transient: usize,
    n_plot: usize,
    state0: &[f64],
) -> Vec<f64> {
    let a = finite_or(a, 0.4).clamp(0.0, 1.0);
    let db_min = finite_or(db_min, COUPLED_DB_MIN).clamp(COUPLED_DB_MIN, COUPLED_DB_MAX);
    let db_max = finite_or(db_max, COUPLED_DB_MAX).clamp(COUPLED_DB_MIN, COUPLED_DB_MAX);
    let n_db = n_db.clamp(1, MAX_PROJECTION_DB);
    let eps = finite_or(eps, COUPLED_EPS_PAPER).clamp(COUPLED_EPS_MIN, COUPLED_EPS_MAX);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_plot = n_plot.clamp(1, MAX_MAP_PLOT);
    let (n_transient, n_plot) = fit_pair_budget(n_db, 1, n_transient, n_plot);
    let start = map_state::<4>(state0);
    let db_values = swept_values(db_min, db_max, n_db);
    let Ok(samples) = dynachaos_core::coupled_delayed_projection_tile(
        a,
        &db_values,
        eps,
        n_transient,
        n_plot,
        &start,
    ) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(HEADER + samples.len());
    out.push(n_db as f64);
    out.push(n_plot as f64);
    out.push(n_transient as f64);
    out.push(2.0);
    out.extend_from_slice(&samples);
    out
}

/// Fractalization attractor tile: the delayed logistic map under this figure's clamps.
///
/// `maps.fractalization.iterate` (`src/dynachaos/maps/fractalization.py:49`) is
/// the delayed logistic map. This export does not reimplement it: it calls
/// `delayed_logistic_attractor_tile` after applying the fractalization clamps
/// (`n_D <= 32`, D in `[1.75, 1.96]`). Python `iterate` passes no
/// `diverged_fn`. The reused kernel instead treats any `|component| > 1e10`
/// as divergence and fills that `D` block with NaN.
///
/// # Returned layout
///
/// `[n_D, n_plot, n_transient, 2, x_0, y_0, ...]`, D-major `(x, y)` pairs.
/// A kernel error returns an empty array, never a trap.
///
/// # Clamping
///
/// - `a`: clamped to `[0, 1]`; a non-finite value falls back to `0.3`.
/// - `d_min`, `d_max`: clamped to `[1.75, 1.96]`; a non-finite value falls
///   back to the matching endpoint.
/// - `n_D`: clamped to `[1, 32]`; `n_transient` to `[0, 20000]`; `n_plot` to
///   `[1, 4096]`, then the pair shrinks to respect the step budget.
/// - `state0`: the first two entries are used; a missing or non-finite
///   component falls back to `0.5`, then every component clamps to `[-4, 4]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn fractalization_attractor_tile(
    a: f64,
    d_min: f64,
    d_max: f64,
    n_d: usize,
    n_transient: usize,
    n_plot: usize,
    state0: &[f64],
) -> Vec<f64> {
    let a = finite_or(a, 0.3).clamp(0.0, 1.0);
    let d_min = finite_or(d_min, FRACTAL_D_MIN).clamp(FRACTAL_D_MIN, FRACTAL_D_MAX);
    let d_max = finite_or(d_max, FRACTAL_D_MAX).clamp(FRACTAL_D_MIN, FRACTAL_D_MAX);
    let n_d = n_d.clamp(1, MAX_FRACTAL_D);
    let n_transient = n_transient.min(MAX_TRANSIENT);
    let n_plot = n_plot.clamp(1, MAX_MAP_PLOT);
    let (n_transient, n_plot) = fit_pair_budget(n_d, 1, n_transient, n_plot);
    let start = map_state::<2>(state0);
    let d_values = swept_values(d_min, d_max, n_d);
    let Ok(samples) =
        dynachaos_core::delayed_logistic_attractor_tile(a, &d_values, n_transient, n_plot, &start)
    else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(HEADER + samples.len());
    out.push(n_d as f64);
    out.push(n_plot as f64);
    out.push(n_transient as f64);
    out.push(2.0);
    out.extend_from_slice(&samples);
    out
}

/// Build a clamped `n_sites`-entry CML initial field.
///
/// A missing or non-finite entry falls back to 0.5, then every entry clamps
/// to `[-1, 2]`. `simulate_cml` draws `x0` from `[0, 1]`
/// (`src/dynachaos/cml/spatiotemporal.py:74`), but the live figure also
/// feeds the window's last row back in to continue the run, and the three
/// models' orbits leave `[0, 1]` (A dips to -0.11, B to -0.05, C to -0.75).
/// `[-1, 2]` covers every observed orbit while still bounding a start far
/// outside the invariant range, which can diverge.
fn cml_state(n_sites: usize, x0: &[f64]) -> Vec<f64> {
    let mut start = vec![0.5_f64; n_sites];
    for (slot, &v) in start.iter_mut().zip(x0.iter()) {
        *slot = finite_or(v, 0.5).clamp(-1.0, 2.0);
    }
    start
}

/// Build a clamped `DIM`-component map initial state.
///
/// A missing or non-finite component falls back to 0.5, then every component
/// clamps to `[-MAP_STATE_ABS, MAP_STATE_ABS]`.
fn map_state<const DIM: usize>(state0: &[f64]) -> [f64; DIM] {
    let mut start = [0.5; DIM];
    for (slot, &v) in start.iter_mut().zip(state0.iter()) {
        *slot = finite_or(v, 0.5).clamp(-MAP_STATE_ABS, MAP_STATE_ABS);
    }
    start
}

/// Reduce `n_pts` until `pairs x dim` fits the diagnostics pair-term budget.
///
/// With `self_pairs` the kernel compares every row to every row (`n_pts^2`
/// pairs, ApEn); otherwise it compares the upper triangle beyond the Theiler
/// window. The result is at least 1: a statistic over a single row is poor
/// but honest, and the header reports what was used.
fn fit_diag_budget(n_pts: usize, dim: usize, theiler_window: usize, self_pairs: bool) -> usize {
    let terms = |m: usize| -> u64 {
        let pairs = if self_pairs {
            (m as u64).saturating_mul(m as u64)
        } else {
            let n_eff = (m as u64).saturating_sub(theiler_window as u64 + 1);
            n_eff.saturating_mul(n_eff + 1) / 2
        };
        pairs.saturating_mul(dim as u64)
    };
    if terms(n_pts) <= MAX_DIAG_TERMS {
        return n_pts;
    }
    // Solve pairs(m) = budget for m, then walk down to the exact integer fit.
    let budget_pairs = MAX_DIAG_TERMS / (dim as u64).max(1);
    let mut m = if self_pairs {
        (budget_pairs as f64).sqrt() as usize
    } else {
        let n_eff = ((1.0 + 8.0 * budget_pairs as f64).sqrt() - 1.0) / 2.0;
        n_eff as usize + theiler_window + 1
    };
    while terms(m) > MAX_DIAG_TERMS {
        m -= 1;
    }
    m.min(n_pts).max(1)
}

/// Return `r` when it is finite and positive, otherwise `0.2 * std`, with
/// ddof = 1, over all coordinates of the kept trajectory. The Python diagnostics
/// take the same fraction of the scalar series; an embedding repeats most
/// samples `dim` times, so the two values differ slightly. `None` when even
/// the fallback cannot produce a positive tolerance.
fn tolerance_or_default(r: f64, traj: &[f64]) -> Option<f64> {
    if r.is_finite() && r > 0.0 {
        return Some(r);
    }
    let n = traj.len();
    if n < 2 {
        return None;
    }
    let mean = traj.iter().sum::<f64>() / n as f64;
    let variance = traj.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
    let fallback = 0.2 * variance.sqrt();
    if fallback.is_finite() && fallback > 0.0 {
        Some(fallback)
    } else {
        None
    }
}

/// Return `value` when it is finite, otherwise `fallback`.
fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

/// Reduce `n_iter` until the call fits the step budget.
///
/// The transient is charged too, because it costs the same per step. The
/// result is at least 1: a tile computed from a single measured step is poor
/// but honest, and the header reports what was used.
fn fit_step_budget(n_omega: usize, n_k: usize, n_transient: usize, n_iter: usize) -> usize {
    let cells = (n_omega as u64) * (n_k as u64);
    let requested = cells.saturating_mul(n_transient as u64 + n_iter as u64);
    if requested <= MAX_STEPS {
        return n_iter;
    }
    let affordable = MAX_STEPS / cells.max(1);
    let measured = affordable.saturating_sub(n_transient as u64);
    measured.max(1) as usize
}

/// Evenly spaced samples, matching `min + (max - min) * k / (n - 1)`.
fn swept_values(min: f64, max: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| min + (max - min) * (k as f64) / ((n - 1).max(1) as f64))
        .collect()
}

/// Clamp one coupled-logistic coordinate to the paper domain `[-1, 1]`.
fn logistic_coord(value: f64, fallback: f64) -> f64 {
    finite_or(value, fallback).clamp(-1.0, 1.0)
}

/// Shrink transient and measured steps until `cells * (transient + measured)`
/// fits `MAX_STEPS`. The measured count stays at least 1; the transient shrinks
/// only when it alone cannot leave room for that step.
fn fit_pair_budget(
    n_rows: usize,
    n_cols: usize,
    n_transient: usize,
    n_measure: usize,
) -> (usize, usize) {
    let cells = (n_rows as u64).saturating_mul(n_cols as u64).max(1);
    let transient = n_transient as u64;
    let measure = n_measure as u64;
    if cells.saturating_mul(transient.saturating_add(measure)) <= MAX_STEPS {
        return (n_transient, n_measure);
    }
    let affordable = MAX_STEPS / cells;
    if transient.saturating_add(1) > affordable {
        return (affordable.saturating_sub(1) as usize, 1);
    }
    (n_transient, (affordable - transient) as usize)
}

/// Shrink the basin transient, then the reference transient, until
/// `n_x * n_y * n_transient + reference_transient <= MAX_STEPS`.
fn fit_basin_budget(
    n_x: usize,
    n_y: usize,
    n_transient: usize,
    n_reference: usize,
) -> (usize, usize) {
    let cells = (n_x as u64).saturating_mul(n_y as u64).max(1);
    let mut reference = n_reference as u64;
    if reference > MAX_STEPS {
        reference = MAX_STEPS;
    }
    let transient = n_transient as u64;
    if cells.saturating_mul(transient).saturating_add(reference) <= MAX_STEPS {
        return (n_transient, reference as usize);
    }
    let affordable = (MAX_STEPS - reference) / cells;
    (affordable as usize, reference as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_reports_the_values_actually_used() {
        let out = rotation_number_tile(0.0, 1.0, 3, 0.0, 0.3, 2, 100, 200, 0.1);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 100.0);
        assert_eq!(out[3], 200.0);
        assert_eq!(out.len(), HEADER + 6);
    }

    #[test]
    fn oversized_requests_are_clamped_not_refused() {
        let out = rotation_number_tile(0.0, 1.0, 100_000, 0.0, 0.3, 100_000, 0, 1, 0.1);
        assert_eq!(out[0], MAX_SIDE as f64);
        assert_eq!(out[1], MAX_SIDE as f64);
        assert_eq!(out.len(), HEADER + MAX_SIDE * MAX_SIDE);
    }

    #[test]
    fn a_huge_iteration_count_is_cut_to_the_step_budget() {
        let out = rotation_number_tile(0.0, 1.0, 256, 0.0, 0.3, 256, 0, MAX_ITER, 0.1);
        let used = out[3] as u64;
        assert!(used >= 1);
        assert!(used < MAX_ITER as u64);
        assert!((256u64 * 256) * used <= MAX_STEPS);
    }

    #[test]
    fn non_finite_arguments_fall_back_to_the_published_range() {
        let out = rotation_number_tile(f64::NAN, f64::INFINITY, 2, 0.0, 0.3, 1, 10, 10, f64::NAN);
        // Omega fell back to 0..1 and theta0 to 0.1, so the call still ran.
        assert_eq!(out.len(), HEADER + 2);
        assert!(out[HEADER..].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_zero_coupling_row_still_returns_omega() {
        // Same analytic check the core crate makes, through the clamped
        // browser entry point: with K = 0 the drift per step is Omega.
        let out = rotation_number_tile(0.0, 0.5, 3, 0.0, 0.0, 1, 200, 500, 0.1);
        let tile = &out[HEADER..];
        for (value, expected) in tile.iter().zip([0.0, 0.25, 0.5]) {
            assert!((value - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn a_locked_point_returns_the_exact_rational() {
        let out = rotation_number_point(0.05, 0.12, 200, 2000, 0.1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn a_non_finite_point_still_returns_a_number() {
        let out = rotation_number_point(f64::NAN, f64::NAN, 10, 10, f64::NAN);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    fn zero_one_phi(n: usize) -> Vec<f64> {
        // A deterministic chaotic-ish series: logistic map at r = 4.
        let mut x = 0.123456789f64;
        (0..n)
            .map(|_| {
                x = 4.0 * x * (1.0 - x);
                x
            })
            .collect()
    }

    #[test]
    fn zero_one_header_reports_the_values_actually_used() {
        let phi = zero_one_phi(500);
        let c = [0.7, 1.1, 1.9];
        let out = zero_one_k(&phi, &c, 50);
        assert_eq!(out[0], 500.0);
        assert_eq!(out[1], 3.0);
        assert_eq!(out[2], 50.0);
        assert_eq!(out.len(), ZERO_ONE_HEADER + 3);
        assert!(out[ZERO_ONE_HEADER..].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_one_oversized_requests_are_cut_not_refused() {
        let phi = zero_one_phi(MAX_ZERO_ONE_N + 5000);
        let c = vec![0.9; MAX_ZERO_ONE_C + 50];
        let out = zero_one_k(&phi, &c, MAX_ZERO_ONE_N + 5000);
        assert_eq!(out[0], MAX_ZERO_ONE_N as f64);
        assert_eq!(out[1], MAX_ZERO_ONE_C as f64);
        // n_cut clamps to [2, N] only: the FFT cost is n_c transforms, so
        // the full window at the N cap is kept, not cut.
        assert_eq!(out[2], MAX_ZERO_ONE_N as f64);
    }

    #[test]
    fn zero_one_n_cut_below_two_reports_two() {
        let phi = zero_one_phi(100);
        let c = [0.9];
        for n_cut in [0usize, 1] {
            let out = zero_one_k(&phi, &c, n_cut);
            assert_eq!(out[2], 2.0, "n_cut {n_cut} should clamp up to 2");
            assert_eq!(out.len(), ZERO_ONE_HEADER + 1);
        }
    }

    #[test]
    fn zero_one_n_cut_is_clamped_into_the_series() {
        let phi = zero_one_phi(50);
        let c = [0.9];
        let out = zero_one_k(&phi, &c, 10_000);
        assert_eq!(out[2], 50.0);
        assert_eq!(out.len(), ZERO_ONE_HEADER + 1);
    }

    #[test]
    fn zero_one_non_finite_input_returns_an_empty_result() {
        let mut phi = zero_one_phi(100);
        phi[40] = f64::NAN;
        assert!(zero_one_k(&phi, &[0.9], 10).is_empty());

        let phi = zero_one_phi(100);
        assert!(zero_one_k(&phi, &[0.9, f64::INFINITY], 10).is_empty());
    }

    #[test]
    fn zero_one_unusable_requests_return_an_empty_result() {
        let phi = zero_one_phi(100);
        assert!(zero_one_k(&phi, &[], 10).is_empty());
        assert!(zero_one_k(&phi[..2], &[0.9], 10).is_empty());
        assert!(zero_one_k(&phi[..1], &[0.9], 10).is_empty());
    }

    #[test]
    fn zero_one_a_zero_series_returns_exact_zeros() {
        let phi = vec![0.0f64; 100];
        let out = zero_one_k(&phi, &[0.7, 1.1, 1.9], 10);
        assert_eq!(out.len(), ZERO_ONE_HEADER + 3);
        assert!(out[ZERO_ONE_HEADER..].iter().all(|&v| v == 0.0));
    }

    /// A deterministic embedded trajectory: logistic map at r = 4, tiled
    /// into `dim`-wide rows.
    fn diag_traj(n_pts: usize, dim: usize) -> Vec<f64> {
        let mut x = 0.123456789f64;
        (0..n_pts * dim)
            .map(|_| {
                x = 4.0 * x * (1.0 - x);
                x
            })
            .collect()
    }

    #[test]
    fn correlation_header_reports_the_values_actually_used() {
        let traj = diag_traj(200, 2);
        let r = [0.05, 0.1, 0.2];
        let out = correlation_counts(&traj, 2, &r, 3, true);
        assert_eq!(out[0], 200.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 3.0);
        assert_eq!(out[3], 3.0);
        assert_eq!(out[4], 1.0);
        assert_eq!(out.len(), CORRELATION_HEADER + 3 + 3);
        // The radii actually used come back sorted, then the counts.
        assert_eq!(&out[5..8], &[0.05, 0.1, 0.2]);
        let counts = &out[CORRELATION_HEADER + 3..];
        assert!(counts.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn correlation_radii_are_sorted_and_clamped_not_refused() {
        let traj = diag_traj(50, 1);
        let out = correlation_counts(&traj, 1, &[0.3, -1.0, 0.1], 0, true);
        assert_eq!(out[2], 3.0);
        assert_eq!(&out[5..8], &[0.0, 0.1, 0.3]);
    }

    #[test]
    fn correlation_pair_work_is_cut_to_the_budget() {
        // 20000 rows at dim 2 is 4e8 pair terms, over the 2e8 budget, so
        // n_pts must come back reduced.
        let traj = diag_traj(MAX_DIAG_N, 2);
        let out = correlation_counts(&traj, 2, &[0.5], 0, true);
        let used = out[0] as u64;
        assert!(used >= 1);
        assert!(used < MAX_DIAG_N as u64);
        let pairs = used * (used - 1) / 2;
        assert!(pairs * 2 <= MAX_DIAG_TERMS);
    }

    #[test]
    fn correlation_unusable_requests_return_an_empty_result() {
        let traj = diag_traj(100, 2);
        // Fewer than one complete row.
        assert!(correlation_counts(&traj[..1], 2, &[0.1], 0, true).is_empty());
        // No radii.
        assert!(correlation_counts(&traj, 2, &[], 0, true).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 2);
        bad[7] = f64::NAN;
        assert!(correlation_counts(&bad, 2, &[0.1], 0, true).is_empty());
        // A non-finite radius in the kept prefix.
        assert!(correlation_counts(&traj, 2, &[0.1, f64::INFINITY], 0, true).is_empty());
    }

    #[test]
    fn apen_header_reports_the_values_actually_used() {
        let traj = diag_traj(150, 2);
        let out = apen_counts(&traj, 2, 0.15);
        assert_eq!(out[0], 150.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.15);
        assert_eq!(out.len(), APEN_HEADER + 150);
        // Self-matches are included, so every row matches at least itself.
        assert!(out[APEN_HEADER..].iter().all(|&c| c >= 1.0));
    }

    #[test]
    fn apen_pair_work_is_cut_to_the_budget() {
        // 20000 rows is 4e8 pair terms at dim 1, over the 2e8 budget, so
        // n_pts must come back reduced.
        let traj = diag_traj(MAX_DIAG_N, 1);
        let out = apen_counts(&traj, 1, 0.2);
        let used = out[0] as u64;
        assert!(used >= 1);
        assert!(used < MAX_DIAG_N as u64);
        assert!(used * used <= MAX_DIAG_TERMS);
    }

    #[test]
    fn apen_a_non_positive_tolerance_falls_back() {
        let traj = diag_traj(100, 1);
        let out = apen_counts(&traj, 1, -1.0);
        assert!(!out.is_empty());
        let used_r = out[2];
        assert!(used_r > 0.0);
        assert!(used_r.is_finite());
    }

    #[test]
    fn apen_unusable_requests_return_an_empty_result() {
        let traj = diag_traj(100, 2);
        // Fewer than one complete row.
        assert!(apen_counts(&traj[..1], 2, 0.1).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 2);
        bad[5] = f64::NAN;
        assert!(apen_counts(&bad, 2, 0.1).is_empty());
        // A constant series leaves the fallback tolerance at zero.
        let flat = vec![0.5f64; 200];
        assert!(apen_counts(&flat, 1, f64::NAN).is_empty());
    }

    #[test]
    fn fuzzy_header_reports_the_values_actually_used() {
        let traj = diag_traj(120, 2);
        let out = fuzzy_entropy_sum(&traj, 2, 0.2, 2, 1);
        assert_eq!(out[0], 120.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.2);
        assert_eq!(out[3], 2.0);
        assert_eq!(out[4], 1.0);
        assert_eq!(out.len(), FUZZY_HEADER + 1);
        assert!(out[FUZZY_HEADER] > 0.0);
        assert!(out[FUZZY_HEADER].is_finite());
    }

    #[test]
    fn fuzzy_exponent_is_clamped_not_refused() {
        let traj = diag_traj(60, 1);
        let out = fuzzy_entropy_sum(&traj, 1, 0.2, 1000, 0);
        assert_eq!(out[3], MAX_FUZZY_N as f64);
    }

    #[test]
    fn fuzzy_pair_work_is_cut_to_the_budget() {
        // Same shape as the correlation budget cut: 20000 rows at dim 2 is
        // 4e8 pair terms, over the 2e8 budget.
        let traj = diag_traj(MAX_DIAG_N, 2);
        let out = fuzzy_entropy_sum(&traj, 2, 0.2, 2, 0);
        let used = out[0] as u64;
        assert!(used >= 1);
        assert!(used < MAX_DIAG_N as u64);
        let pairs = used * (used - 1) / 2;
        assert!(pairs * 2 <= MAX_DIAG_TERMS);
    }

    #[test]
    fn fuzzy_unusable_requests_return_an_empty_result() {
        let traj = diag_traj(100, 2);
        // Fewer than one complete row.
        assert!(fuzzy_entropy_sum(&traj[..1], 2, 0.1, 2, 0).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 2);
        bad[9] = f64::INFINITY;
        assert!(fuzzy_entropy_sum(&bad, 2, 0.1, 2, 0).is_empty());
        // A constant series leaves the fallback tolerance at zero.
        let flat = vec![0.5f64; 200];
        assert!(fuzzy_entropy_sum(&flat, 1, 0.0, 2, 0).is_empty());
    }

    #[test]
    fn ordinal_header_reports_the_values_actually_used() {
        let x = diag_traj(500, 1);
        let out = ordinal_distribution(&x, 3, 2);
        assert_eq!(out[0], 500.0);
        assert_eq!(out[1], 3.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 496.0); // 500 - (3 - 1) * 2 windows
        assert_eq!(out.len(), ORDINAL_HEADER + 6); // 3! = 6 patterns
        let total: f64 = out[ORDINAL_HEADER..].iter().sum();
        assert_eq!(total, out[3]);
    }

    #[test]
    fn ordinal_d_and_tau_are_clamped_not_refused() {
        let x = diag_traj(500, 1);
        let out = ordinal_distribution(&x, 20, 0);
        assert_eq!(out[1], MAX_ORDINAL_D as f64);
        assert_eq!(out[2], 1.0);
        assert_eq!(out.len(), ORDINAL_HEADER + 40320); // 8! patterns
    }

    #[test]
    fn ordinal_the_sample_cap_is_the_budget() {
        let x = diag_traj(MAX_DIAG_N + 5000, 1);
        let out = ordinal_distribution(&x, 3, 1);
        assert_eq!(out[0], MAX_DIAG_N as f64);
    }

    #[test]
    fn ordinal_unusable_requests_return_an_empty_result() {
        let x = diag_traj(100, 1);
        // Too short for a single window at d = 3, tau = 200.
        assert!(ordinal_distribution(&x, 3, 200).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 1);
        bad[42] = f64::NAN;
        assert!(ordinal_distribution(&bad, 3, 1).is_empty());
        // An empty series.
        assert!(ordinal_distribution(&[], 3, 1).is_empty());
    }

    #[test]
    fn ordinal_a_monotone_series_lands_in_one_pattern() {
        let x: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let out = ordinal_distribution(&x, 3, 2);
        // An ascending window always sorts to the identity permutation, so
        // every window lands in a single Lehmer code.
        let nonzero = out[ORDINAL_HEADER..].iter().filter(|&&c| c > 0.0).count();
        assert_eq!(nonzero, 1);
        assert_eq!(out[ORDINAL_HEADER], 496.0);
    }

    #[test]
    fn fit_diag_budget_leaves_small_requests_alone() {
        assert_eq!(fit_diag_budget(100, 2, 0, false), 100);
        assert_eq!(fit_diag_budget(100, 2, 0, true), 100);
    }

    #[test]
    fn fit_diag_budget_counts_self_pairs_for_apen() {
        // ApEn compares every row to every row: n^2 terms at dim 1.
        let used = fit_diag_budget(MAX_DIAG_N, 1, 0, true);
        assert!((used as u64) * (used as u64) <= MAX_DIAG_TERMS);
        assert!(used < MAX_DIAG_N);
    }

    #[test]
    fn fit_diag_budget_counts_valid_pairs_for_theiler() {
        // At dim 2 the budget binds below 20000 rows. A Theiler window
        // removes pairs, so more rows fit it.
        let plain = fit_diag_budget(MAX_DIAG_N, 2, 0, false);
        let windowed = fit_diag_budget(MAX_DIAG_N, 2, 100, false);
        assert!(plain < MAX_DIAG_N);
        assert!(windowed > plain);
    }

    #[test]
    fn fit_diag_budget_keeps_every_row_that_fits() {
        for (theiler_window, self_pairs) in [(0, false), (100, false), (0, true)] {
            let used = fit_diag_budget(MAX_DIAG_N, 2, theiler_window, self_pairs);
            // The fitted count fits, and one more row would not.
            assert_eq!(fit_diag_budget(used, 2, theiler_window, self_pairs), used);
            assert!(fit_diag_budget(used + 1, 2, theiler_window, self_pairs) < used + 1);
        }
    }

    #[test]
    fn diagnostics_dim_is_clamped_not_refused() {
        let traj = diag_traj(40, MAX_DIAG_DIM);
        // Dimension 0 reads the trajectory as one coordinate per row.
        assert_eq!(correlation_counts(&traj, 0, &[0.1], 0, true)[1], 1.0);
        assert_eq!(apen_counts(&traj, 0, 0.1)[1], 1.0);
        assert_eq!(fuzzy_entropy_sum(&traj, 0, 0.1, 2, 0)[1], 1.0);
        // A dimension above the cap is cut to the cap.
        let cap = MAX_DIAG_DIM as f64;
        assert_eq!(correlation_counts(&traj, 1000, &[0.1], 0, true)[1], cap);
        assert_eq!(apen_counts(&traj, 1000, 0.1)[1], cap);
        assert_eq!(fuzzy_entropy_sum(&traj, 1000, 0.1, 2, 0)[1], cap);
    }

    #[test]
    fn diagnostics_theiler_window_is_clamped_not_refused() {
        let traj = diag_traj(30, 1);
        assert_eq!(correlation_counts(&traj, 1, &[0.1], 1000, true)[3], 29.0);
        assert_eq!(fuzzy_entropy_sum(&traj, 1, 0.1, 2, 1000)[4], 29.0);
    }

    #[test]
    fn diagnostics_row_cap_binds_when_a_theiler_window_frees_the_budget() {
        // At dim 1 a Theiler window of 10000 leaves few enough pairs that
        // 25000 rows fit the pair budget, so only the row cap stops them.
        let traj = diag_traj(MAX_DIAG_N + 5000, 1);
        let cap = MAX_DIAG_N as f64;
        assert_eq!(correlation_counts(&traj, 1, &[0.1], 10_000, true)[0], cap);
        assert_eq!(fuzzy_entropy_sum(&traj, 1, 0.1, 2, 10_000)[0], cap);
    }

    #[test]
    fn tolerance_fallback_is_a_fifth_of_the_sample_std() {
        // Mean 2.5, squared deviations sum to 5, ddof = 1 gives variance 5/3.
        let expected = 0.2 * (5.0f64 / 3.0).sqrt();
        assert_eq!(
            tolerance_or_default(f64::NAN, &[1.0, 2.0, 3.0, 4.0]),
            Some(expected)
        );
        assert_eq!(tolerance_or_default(0.3, &[1.0, 2.0, 3.0, 4.0]), Some(0.3));
    }

    /// A deterministic recurrence mask: a logistic-map orbit thresholded so
    /// about a third of the cells are recurrent, with a few nonzero bytes
    /// that are not 0/1 to exercise the "any nonzero byte" rule.
    fn rqa_mask(side: usize) -> Vec<u8> {
        let mut x = 0.123456789f64;
        (0..side * side)
            .map(|i| {
                x = 4.0 * x * (1.0 - x);
                if x > 0.66 {
                    // Every third recurrent cell carries a byte above 1.
                    if i % 3 == 0 { 7 } else { 1 }
                } else {
                    0
                }
            })
            .collect()
    }

    #[test]
    fn diagonal_header_reports_the_values_actually_used() {
        let mask = rqa_mask(40);
        let out = diagonal_lines(&mask, 40, 2);
        assert_eq!(out[0], 40.0);
        assert_eq!(out[1], 2.0);
        let n_lines = out[2] as usize;
        assert_eq!(out.len(), LINES_HEADER + n_lines);
        // Every reported line meets l_min.
        assert!(out[LINES_HEADER..].iter().all(|&v| v >= 2.0));
    }

    #[test]
    fn diagonal_matches_the_native_kernel() {
        // The same mask through the pyo3 path is the parity oracle; here we
        // check the traversal against count_line_lengths applied by hand.
        let side = 25;
        let mask = rqa_mask(side);
        let out = diagonal_lines(&mask, side, 2);
        let mut expected: Vec<i64> = Vec::new();
        let mut buf = Vec::with_capacity(side);
        for k in 1..side {
            buf.clear();
            for i in 0..(side - k) {
                buf.push(mask[i * side + (i + k)] != 0);
            }
            expected.extend(dynachaos_core::count_line_lengths(&buf, 2).unwrap());
        }
        let got: Vec<i64> = out[LINES_HEADER..].iter().map(|&v| v as i64).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn diagonal_side_is_capped_not_refused() {
        // A side above the cap clamps to the cap, so the mask only needs the
        // capped number of cells.
        let mask = rqa_mask(MAX_RQA_SIDE);
        let out = diagonal_lines(&mask, MAX_RQA_SIDE + 500, 2);
        assert_eq!(out[0], MAX_RQA_SIDE as f64);
    }

    #[test]
    fn diagonal_l_min_is_clamped_into_the_side() {
        let mask = rqa_mask(30);
        let out = diagonal_lines(&mask, 30, 0);
        assert_eq!(out[1], 1.0);
        let out = diagonal_lines(&mask, 30, 10_000);
        assert_eq!(out[1], 30.0);
    }

    #[test]
    fn diagonal_unusable_requests_return_an_empty_result() {
        let mask = rqa_mask(20);
        // A mask shorter than side * side cannot produce a matrix.
        assert!(diagonal_lines(&mask[..100], 20, 2).is_empty());
        // A side below 2 clamps up to 2, not refused: a 2x2 scan still runs.
        let tiny = rqa_mask(4);
        assert_eq!(diagonal_lines(&tiny, 0, 2)[0], 2.0);
        assert_eq!(diagonal_lines(&tiny, 1, 2)[0], 2.0);
    }

    #[test]
    fn vertical_header_reports_the_values_actually_used() {
        let mask = rqa_mask(40);
        let out = vertical_lines(&mask, 40, 2);
        assert_eq!(out[0], 40.0);
        assert_eq!(out[1], 2.0);
        let n_lines = out[2] as usize;
        assert_eq!(out.len(), LINES_HEADER + n_lines);
        assert!(out[LINES_HEADER..].iter().all(|&v| v >= 2.0));
    }

    #[test]
    fn vertical_matches_the_native_kernel() {
        let side = 25;
        let mask = rqa_mask(side);
        let out = vertical_lines(&mask, side, 2);
        let mut expected: Vec<i64> = Vec::new();
        let mut buf = Vec::with_capacity(side);
        for j in 0..side {
            buf.clear();
            for i in 0..side {
                buf.push(mask[i * side + j] != 0);
            }
            expected.extend(dynachaos_core::count_line_lengths(&buf, 2).unwrap());
        }
        let got: Vec<i64> = out[LINES_HEADER..].iter().map(|&v| v as i64).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn line_exports_scan_the_right_direction_on_an_asymmetric_mask() {
        // The two tests above copy the export's traversal, so they cannot
        // catch a wrong direction. This mask is not symmetric, and the
        // expected lengths are written by hand:
        //   1 1 0 0
        //   1 0 1 0
        //   1 0 0 1
        //   0 0 0 0
        // Super-diagonal k = 1 is (0,1), (1,2), (2,3): one line of 3. The
        // other super-diagonals are empty. Column 0 holds a line of 3;
        // columns 1, 2 and 3 hold one cell each.
        let mask: [u8; 16] = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0];
        let diag = diagonal_lines(&mask, 4, 1);
        assert_eq!(&diag[LINES_HEADER..], &[3.0]);
        let vert = vertical_lines(&mask, 4, 1);
        assert_eq!(&vert[LINES_HEADER..], &[3.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn vertical_side_is_capped_not_refused() {
        let mask = rqa_mask(MAX_RQA_SIDE);
        let out = vertical_lines(&mask, MAX_RQA_SIDE + 500, 2);
        assert_eq!(out[0], MAX_RQA_SIDE as f64);
    }

    #[test]
    fn vertical_v_min_is_clamped_into_the_side() {
        let mask = rqa_mask(30);
        let out = vertical_lines(&mask, 30, 0);
        assert_eq!(out[1], 1.0);
        let out = vertical_lines(&mask, 30, 10_000);
        assert_eq!(out[1], 30.0);
    }

    #[test]
    fn vertical_unusable_requests_return_an_empty_result() {
        let mask = rqa_mask(20);
        // A mask shorter than side * side cannot produce a matrix.
        assert!(vertical_lines(&mask[..100], 20, 2).is_empty());
        // A side below 2 clamps up to 2, not refused: a 2x2 scan still runs.
        let tiny = rqa_mask(4);
        assert_eq!(vertical_lines(&tiny, 0, 2)[0], 2.0);
        assert_eq!(vertical_lines(&tiny, 1, 2)[0], 2.0);
    }

    /// A deterministic nonnegative field for the multifractal export.
    fn mf_field(ny: usize, nx: usize) -> Vec<f64> {
        let mut x = 0.123456789f64;
        (0..ny * nx)
            .map(|_| {
                x = 4.0 * x * (1.0 - x);
                x + 0.5
            })
            .collect()
    }

    #[test]
    fn multifractal_header_reports_the_values_actually_used() {
        let field = mf_field(8, 8);
        let out = multifractal_moments(&field, 8, 8, &[2.0, 4.0], &[1.0, 2.0]);
        assert_eq!(out[0], 8.0);
        assert_eq!(out[1], 8.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 2.0);
        // 4 + 3 * 2 * 2 + 2 = 18 values.
        assert_eq!(out.len(), MULTIFRACTAL_HEADER + 3 * 4 + 2);
        // ln_scales sit at the tail: ln(2), ln(4).
        assert!((out[16] - 2.0f64.ln()).abs() < 1e-12);
        assert!((out[17] - 4.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn multifractal_sides_are_capped_not_refused() {
        let field = mf_field(MAX_MF_SIDE, MAX_MF_SIDE);
        let out =
            multifractal_moments(&field, MAX_MF_SIDE + 100, MAX_MF_SIDE + 100, &[2.0], &[1.0]);
        assert_eq!(out[0], MAX_MF_SIDE as f64);
        assert_eq!(out[1], MAX_MF_SIDE as f64);
    }

    #[test]
    fn multifractal_box_and_q_lists_are_truncated_not_refused() {
        let field = mf_field(8, 8);
        let boxes = vec![2.0; MAX_MF_BOXES + 10];
        let qs = vec![1.0; MAX_MF_Q + 10];
        let out = multifractal_moments(&field, 8, 8, &boxes, &qs);
        assert_eq!(out[2], MAX_MF_BOXES as f64);
        assert_eq!(out[3], MAX_MF_Q as f64);
    }

    #[test]
    fn multifractal_unusable_requests_return_an_empty_result() {
        let field = mf_field(8, 8);
        // A field shorter than ny * nx cannot produce a result.
        assert!(multifractal_moments(&field[..10], 8, 8, &[2.0], &[1.0]).is_empty());
        // No box sizes or no q values.
        assert!(multifractal_moments(&field, 8, 8, &[], &[1.0]).is_empty());
        assert!(multifractal_moments(&field, 8, 8, &[2.0], &[]).is_empty());
        // A non-finite or negative field entry.
        let mut bad = mf_field(8, 8);
        bad[5] = f64::NAN;
        assert!(multifractal_moments(&bad, 8, 8, &[2.0], &[1.0]).is_empty());
        let mut neg = mf_field(8, 8);
        neg[3] = -1.0;
        assert!(multifractal_moments(&neg, 8, 8, &[2.0], &[1.0]).is_empty());
        // A zero-mass field.
        let zero = vec![0.0f64; 64];
        assert!(multifractal_moments(&zero, 8, 8, &[2.0], &[1.0]).is_empty());
    }

    #[test]
    fn ami_header_reports_the_values_actually_used() {
        let x = diag_traj(500, 1);
        let out = ami_histogram(&x, 10, 16);
        assert_eq!(out[0], 500.0);
        assert_eq!(out[1], 10.0);
        assert_eq!(out[2], 16.0);
        assert_eq!(out.len(), AMI_HEADER + 10);
        assert!(out[AMI_HEADER..].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ami_tau_and_bins_are_clamped_not_refused() {
        let x = diag_traj(200, 1);
        let out = ami_histogram(&x, 0, 0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 1.0);
        let out = ami_histogram(&x, MAX_AMI_TAU + 100, MAX_AMI_BINS + 100);
        assert_eq!(out[1], MAX_AMI_TAU as f64);
        assert_eq!(out[2], MAX_AMI_BINS as f64);
    }

    #[test]
    fn ami_the_sample_cap_is_the_budget() {
        let x = diag_traj(MAX_DIAG_N + 5000, 1);
        let out = ami_histogram(&x, 5, 8);
        assert_eq!(out[0], MAX_DIAG_N as f64);
    }

    #[test]
    fn ami_unusable_requests_return_an_empty_result() {
        let x = diag_traj(100, 1);
        // Fewer than two samples.
        assert!(ami_histogram(&x[..1], 5, 8).is_empty());
        // A non-finite sample in the kept prefix.
        let mut bad = diag_traj(100, 1);
        bad[7] = f64::NAN;
        assert!(ami_histogram(&bad, 5, 8).is_empty());
    }

    #[test]
    fn cao_header_reports_the_values_actually_used() {
        // A plateau at 1 from dimension 3 on selects dimension 3.
        let e1 = [0.4, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0];
        let out = select_dimension_cao(&e1, 0.95, 1.05, 0.02, 3, 1, 2, f64::NAN);
        assert_eq!(out[0], 7.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.0); // max_dim left automatic
        assert_eq!(out[3], 3.0);
    }

    #[test]
    fn cao_parameters_are_clamped_not_refused() {
        let e1 = [0.4, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0];
        // Non-finite tolerances fall back to the published defaults.
        let out = select_dimension_cao(&e1, f64::NAN, f64::INFINITY, f64::NAN, 3, 1, 2, f64::NAN);
        assert_eq!(out[3], 3.0);
        // min_dim clamps into the curve; a max_dim below min_dim clamps up.
        let out = select_dimension_cao(&e1, 0.95, 1.05, 0.02, 3, 1, 0, 1.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 1.0);
        // A max_dim above the curve length clamps down to N.
        let out = select_dimension_cao(&e1, 0.95, 1.05, 0.02, 3, 1, 2, 10_000.0);
        assert_eq!(out[2], 7.0);
    }

    #[test]
    fn cao_the_curve_cap_is_the_budget() {
        let e1 = vec![1.0f64; MAX_CAO_N + 100];
        let out = select_dimension_cao(&e1, 0.95, 1.05, 0.02, 3, 1, 2, f64::NAN);
        assert_eq!(out[0], MAX_CAO_N as f64);
    }

    #[test]
    fn cao_unusable_requests_return_an_empty_result() {
        assert!(select_dimension_cao(&[], 0.95, 1.05, 0.02, 3, 1, 2, f64::NAN).is_empty());
    }

    #[test]
    fn delayed_header_reports_the_values_actually_used() {
        let out = delayed_logistic_attractor_tile(0.3, 1.55, 2.16, 3, 200, 64, &[0.4, 0.35]);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 64.0);
        assert_eq!(out[2], 200.0);
        assert_eq!(out[3], 2.0);
        assert_eq!(out.len(), HEADER + 3 * 64 * 2);
    }

    #[test]
    fn delayed_counts_are_clamped_not_refused() {
        let out = delayed_logistic_attractor_tile(
            0.3,
            1.55,
            2.16,
            MAX_MAP_D_LIST + 10,
            MAX_TRANSIENT + 10_000,
            MAX_MAP_PLOT + 10,
            &[0.4, 0.35],
        );
        assert_eq!(out[0], MAX_MAP_D_LIST as f64);
        assert_eq!(out[1], MAX_MAP_PLOT as f64);
        assert_eq!(out[2], MAX_TRANSIENT as f64);
        assert_eq!(out.len(), HEADER + MAX_MAP_D_LIST * MAX_MAP_PLOT * 2);
    }

    #[test]
    fn delayed_a_is_clamped_to_the_paper_range() {
        // A = -2 clamps to 0, so x' = 1 - D * y * y and y' = x. With
        // n_transient = 0 the first sample is (1 - d * y0^2, x0); d_min = 0
        // clamps to DELAYED_D_MIN = 1.4.
        let out = delayed_logistic_attractor_tile(-2.0, 0.0, 0.0, 1, 0, 1, &[0.4, 0.35]);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 1.0);
        let expected_x = 1.0 - DELAYED_D_MIN * 0.35 * 0.35;
        assert_eq!(out[HEADER], expected_x);
        assert_eq!(out[HEADER + 1], 0.4);
    }

    #[test]
    fn delayed_d_endpoints_are_clamped_to_the_paper_range() {
        // d_max = 99 clamps to DELAYED_D_MAX = 3.5. With no transient and
        // two D values, the second block exposes the clamped endpoint directly.
        let out = delayed_logistic_attractor_tile(0.3, 1.4, 99.0, 2, 0, 1, &[0.4, 0.35]);
        let expected_x = 0.3 * 0.4 + (1.0 - 0.3) * (1.0 - DELAYED_D_MAX * 0.35 * 0.35);
        assert_eq!(out.len(), HEADER + 4);
        assert_eq!(out[HEADER + 2], expected_x);
        assert_eq!(out[HEADER + 3], 0.4);
    }

    #[test]
    fn delayed_state_is_clamped_and_padded() {
        // Missing components fall back to 0.5; oversized ones clamp to 4.
        let padded = delayed_logistic_attractor_tile(0.3, 1.55, 1.55, 1, 20, 8, &[]);
        let default_state = delayed_logistic_attractor_tile(0.3, 1.55, 1.55, 1, 20, 8, &[0.5, 0.5]);
        assert_eq!(padded, default_state);
        let clamped = delayed_logistic_attractor_tile(0.3, 1.55, 1.55, 1, 0, 1, &[99.0, -99.0]);
        let at_cap = delayed_logistic_attractor_tile(0.3, 1.55, 1.55, 1, 0, 1, &[4.0, -4.0]);
        assert_eq!(clamped, at_cap);

        let nan_state = delayed_logistic_attractor_tile(0.3, 1.55, 1.55, 1, 20, 8, &[f64::NAN]);
        assert_eq!(nan_state, default_state);
    }

    #[test]
    fn delayed_divergence_marks_one_block_nan() {
        // D = 3.0 is inside the clamped range and diverges; the first block
        // stays finite, the second is all NaN.
        let out = delayed_logistic_attractor_tile(0.3, 1.55, 3.0, 2, 2000, 8, &[0.4, 0.35]);
        assert_eq!(out.len(), HEADER + 2 * 8 * 2);
        assert!(out[HEADER..HEADER + 16].iter().all(|v| v.is_finite()));
        assert!(out[HEADER + 16..].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn torus_header_reports_the_values_actually_used() {
        let out = torus_doubling_attractor_tile(1, 0.4, 2.19, 200, 64, &[0.5, 0.5, 0.5]);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 3.0);
        assert_eq!(out[2], 200.0);
        assert_eq!(out[3], 64.0);
        assert_eq!(out.len(), HEADER + 64 * 3);
    }

    #[test]
    fn torus_map_kind_snaps_to_one_or_four() {
        for (kind, dim) in [(0usize, 3.0), (2, 3.0), (3, 4.0), (usize::MAX, 4.0)] {
            let state = [0.5, 0.45, 0.52, 0.48];
            let out = torus_doubling_attractor_tile(kind, 0.4, 2.19, 10, 4, &state);
            assert_eq!(out[0], if dim == 3.0 { 1.0 } else { 4.0 });
            assert_eq!(out[1], dim);
        }
    }

    #[test]
    fn torus_counts_are_clamped_not_refused() {
        let out = torus_doubling_attractor_tile(
            1,
            0.4,
            2.19,
            MAX_TRANSIENT + 10_000,
            MAX_MAP_PLOT + 10,
            &[0.5, 0.5, 0.5],
        );
        assert_eq!(out[2], MAX_TRANSIENT as f64);
        assert!(out[3] <= MAX_MAP_PLOT as f64);
        // n_plot = 0 clamps up to 1, so a bounded orbit still yields a sample.
        let out = torus_doubling_attractor_tile(1, 0.4, 2.19, 10, 0, &[0.5, 0.5, 0.5]);
        assert_eq!(out[3], 1.0);
        assert_eq!(out.len(), HEADER + 3);
    }

    #[test]
    fn torus_a_and_d_are_clamped_to_the_paper_ranges() {
        // A = -2 clamps to 0, so X' = L_D(Y) and Y' = Z; with n_transient = 0
        // the first sample is (1 - d * y0^2, z0, x0). d = 99 clamps to
        // TORUS_D_MAX = 2.25.
        let out = torus_doubling_attractor_tile(1, -2.0, 99.0, 0, 1, &[0.4, 0.35, 0.3]);
        assert_eq!(out[3], 1.0);
        let expected_x = 1.0 - TORUS_D_MAX * 0.35 * 0.35;
        assert_eq!(out[HEADER], expected_x);
        assert_eq!(out[HEADER + 1], 0.3);
        assert_eq!(out[HEADER + 2], 0.4);
    }

    #[test]
    fn torus_state_is_clamped_and_padded() {
        let padded = torus_doubling_attractor_tile(1, 0.4, 2.19, 20, 8, &[]);
        let default_state = torus_doubling_attractor_tile(1, 0.4, 2.19, 20, 8, &[0.5, 0.5, 0.5]);
        assert_eq!(padded, default_state);

        let clamped = torus_doubling_attractor_tile(1, 0.4, 2.19, 0, 1, &[99.0, -99.0, 0.5]);
        let at_cap = torus_doubling_attractor_tile(1, 0.4, 2.19, 0, 1, &[4.0, -4.0, 0.5]);
        assert_eq!(clamped, at_cap);

        // Extra entries beyond the map dimension are ignored.
        let long = torus_doubling_attractor_tile(1, 0.4, 2.19, 20, 8, &[0.5, 0.5, 0.5, 99.0]);
        assert_eq!(long, default_state);
    }

    #[test]
    fn torus_divergence_returns_partial_samples() {
        // Map IV at D = 2.12 escapes 1212 steps into the record after a
        // 2000-step transient, so the header reports 1212 samples, not 4096.
        let out = torus_doubling_attractor_tile(
            4,
            0.3,
            2.12,
            2000,
            MAX_MAP_PLOT,
            &[0.5, 0.45, 0.52, 0.48],
        );
        assert_eq!(out[3], 1212.0);
        assert_eq!(out.len(), HEADER + 1212 * 4);
        assert!(out[HEADER..].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn modulated_circle_header_reports_the_values_actually_used() {
        let out =
            modulated_circle_rotation_tile(0.1, C_GOLDEN, 0.0, 1.0, 5, 0.05, 100, 500, 0.1, 0.1);
        assert_eq!(out[0], 5.0);
        assert_eq!(out[1], 100.0);
        assert_eq!(out[2], 500.0);
        assert_eq!(out[3], 2.0);
        assert_eq!(out.len(), HEADER + 5 * 2);
    }

    #[test]
    fn modulated_circle_counts_are_clamped_not_refused() {
        // The D-list clamp with small counts stays cheap; the step clamps
        // with n_d = 1 stay inside the budget.
        let out = modulated_circle_rotation_tile(
            0.1,
            C_GOLDEN,
            0.0,
            1.0,
            MAX_SIDE + 10,
            0.05,
            10,
            50,
            0.1,
            0.1,
        );
        assert_eq!(out[0], MAX_SIDE as f64);
        assert_eq!(out.len(), HEADER + MAX_SIDE * 2);
        let out = modulated_circle_rotation_tile(
            0.1,
            C_GOLDEN,
            0.0,
            1.0,
            1,
            0.05,
            MAX_TRANSIENT + 10_000,
            MAX_ITER + 10,
            0.1,
            0.1,
        );
        assert_eq!(out[1], MAX_TRANSIENT as f64);
        assert_eq!(out[2], MAX_ITER as f64);
    }
    #[test]
    fn modulated_circle_parameters_are_clamped_to_unit_interval() {
        // A = -2 clamps to 0, eps = 99 clamps to 1, and D endpoints outside
        // [0, 1] clamp in. With n_transient = 0 and n_iter = 1 the returned
        // rho_theta is the increment at state0 itself, so the clamps are
        // visible in the value, not just the header.
        let out =
            modulated_circle_rotation_tile(-2.0, C_GOLDEN, -1.0, 99.0, 1, 99.0, 0, 1, 0.1, 0.1);
        let expected = 0.0 + 1.0 * (2.0 * std::f64::consts::PI * 0.1).sin();
        assert_eq!(out[HEADER], expected);
        // The same request with the parameters already inside the ranges
        // returns the same number.
        let inside =
            modulated_circle_rotation_tile(0.0, C_GOLDEN, 0.0, 1.0, 1, 1.0, 0, 1, 0.1, 0.1);
        assert_eq!(out, inside);
    }

    #[test]
    fn modulated_circle_c_is_clamped_to_unit_interval() {
        // C = 2.0 clamps to 1.0: rho_phi is the mean increment of phi, so the
        // clamp is visible in the value itself.
        let out = modulated_circle_rotation_tile(0.1, 2.0, 0.3, 0.3, 1, 0.05, 10, 50, 0.1, 0.1);
        let at_cap = modulated_circle_rotation_tile(0.1, 1.0, 0.3, 0.3, 1, 0.05, 10, 50, 0.1, 0.1);
        assert_eq!(out, at_cap);
        assert!(out[HEADER + 1] <= 1.0);
    }

    #[test]
    fn modulated_circle_non_finite_arguments_fall_back() {
        let fallback = modulated_circle_rotation_tile(
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            3,
            f64::NAN,
            50,
            200,
            f64::NAN,
            f64::NAN,
        );
        let explicit =
            modulated_circle_rotation_tile(0.1, C_GOLDEN, 0.0, 1.0, 3, 0.05, 50, 200, 0.1, 0.1);
        assert_eq!(fallback, explicit);
    }

    #[test]
    fn modulated_circle_phases_are_clamped_to_unit_interval() {
        // Non-integer out-of-range phases: 7.3 clamps to 1.0 and -3.4 to 0.0,
        // so the request equals the clamped one. (Integer values would not
        // distinguish the clamp: sin(2 pi k) is the same at every integer.)
        let clamped =
            modulated_circle_rotation_tile(0.1, C_GOLDEN, 0.3, 0.3, 1, 0.05, 50, 200, 7.3, -3.4);
        let at_cap =
            modulated_circle_rotation_tile(0.1, C_GOLDEN, 0.3, 0.3, 1, 0.05, 50, 200, 1.0, 0.0);
        assert_eq!(clamped, at_cap);
    }

    #[test]
    fn cml_header_reports_the_values_actually_used() {
        let x0 = [0.1, 0.4, 0.7, 0.2];
        let out = cml_spacetime_tile(2, 0.2, 4, 10, 8, &x0);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 4.0);
        assert_eq!(out[2], 10.0);
        assert_eq!(out[3], 8.0);
        assert_eq!(out.len(), HEADER + 8 * 4);
    }

    #[test]
    fn cml_model_snaps_to_a_b_or_c() {
        let x0 = [0.1, 0.4, 0.7, 0.2];
        for (model, expected) in [(0usize, 0.0), (1, 1.0), (2, 2.0), (usize::MAX, 2.0)] {
            let out = cml_spacetime_tile(model, 0.2, 4, 10, 4, &x0);
            assert_eq!(out[0], expected);
        }
    }

    #[test]
    fn cml_counts_are_clamped_not_refused() {
        let x0 = [0.1, 0.4, 0.7, 0.2];
        let out = cml_spacetime_tile(2, 0.2, 1, MAX_TRANSIENT + 10_000, MAX_CML_RECORD + 10, &x0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], MAX_TRANSIENT as f64);
        assert_eq!(out[3], MAX_CML_RECORD as f64);
        assert_eq!(out.len(), HEADER + MAX_CML_RECORD * 2);
    }

    #[test]
    fn cml_eps_is_clamped_and_falls_back_per_model() {
        let x0 = [0.1, 0.4, 0.7, 0.2];
        // eps = 99 clamps to 1: the same request at eps = 1 returns the same
        // field.
        let clamped = cml_spacetime_tile(2, 99.0, 4, 10, 4, &x0);
        let at_cap = cml_spacetime_tile(2, 1.0, 4, 10, 4, &x0);
        assert_eq!(clamped, at_cap);
        // A non-finite eps falls back to the middle of the model's paper
        // sweep: 0.07 for (A), 0.024 for (B), 0.2 for (C).
        for (model, eps) in [(0usize, 0.07), (1, 0.024), (2, 0.2)] {
            let fallback = cml_spacetime_tile(model, f64::NAN, 4, 10, 4, &x0);
            let explicit = cml_spacetime_tile(model, eps, 4, 10, 4, &x0);
            assert_eq!(fallback, explicit);
        }
    }

    #[test]
    fn cml_state_is_padded_truncated_and_clamped() {
        // Fewer entries than n_sites pad with 0.5; more are truncated.
        let padded = cml_spacetime_tile(2, 0.2, 4, 10, 4, &[0.1, 0.4]);
        let explicit = cml_spacetime_tile(2, 0.2, 4, 10, 4, &[0.1, 0.4, 0.5, 0.5]);
        assert_eq!(padded, explicit);
        let truncated = cml_spacetime_tile(2, 0.2, 2, 10, 4, &[0.1, 0.4, 0.7, 0.2]);
        let short = cml_spacetime_tile(2, 0.2, 2, 10, 4, &[0.1, 0.4]);
        assert_eq!(truncated, short);
        // Entries outside [-1, 2] clamp in; non-finite entries fall back
        // to 0.5. Asserting cml_state directly pins the bounds: through
        // the export a narrower clamp would produce the same field on
        // both sides and stay invisible.
        assert_eq!(
            cml_state(4, &[-5.0, 9.0, f64::NAN, 0.2]),
            [-1.0, 2.0, 0.5, 0.2]
        );
    }

    #[test]
    fn circle_lyapunov_header_reports_the_values_actually_used() {
        let out = circle_map_lyapunov_sum(0.1, 0.25, 200, 5000, 0.1);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], 200.0);
        assert_eq!(out[1], 5000.0);
        assert!(out[2].is_finite());
    }

    #[test]
    fn circle_lyapunov_a_is_clamped_to_the_staircase_sweep() {
        // A = 0.9 clamps to CIRCLE_A_MAX = 0.25.
        let out = circle_map_lyapunov_sum(0.9, 0.25, 200, 1000, 0.1);
        let capped = circle_map_lyapunov_sum(CIRCLE_A_MAX, 0.25, 200, 1000, 0.1);
        assert_eq!(out, capped);
        let below = circle_map_lyapunov_sum(-1.0, 0.25, 200, 1000, 0.1);
        let at_zero = circle_map_lyapunov_sum(0.0, 0.25, 200, 1000, 0.1);
        assert_eq!(below, at_zero);
    }

    #[test]
    fn circle_lyapunov_d_is_clamped_to_unit_interval() {
        // d = 9 clamps to 1.
        let out = circle_map_lyapunov_sum(0.1, 9.0, 200, 1000, 0.1);
        let capped = circle_map_lyapunov_sum(0.1, 1.0, 200, 1000, 0.1);
        assert_eq!(out, capped);
        let out = circle_map_lyapunov_sum(0.1, -4.0, 200, 1000, 0.1);
        let floored = circle_map_lyapunov_sum(0.1, 0.0, 200, 1000, 0.1);
        assert_eq!(out, floored);
    }

    #[test]
    fn circle_lyapunov_theta0_and_counts_are_clamped() {
        let out = circle_map_lyapunov_sum(0.1, 0.25, 200, 1000, 7.3);
        let capped = circle_map_lyapunov_sum(0.1, 0.25, 200, 1000, 1.0);
        assert_eq!(out, capped);
        let out = circle_map_lyapunov_sum(0.1, 0.25, MAX_TRANSIENT + 10_000, MAX_ITER + 10, 0.1);
        assert_eq!(out[0], MAX_TRANSIENT as f64);
        assert_eq!(out[1], MAX_ITER as f64);
    }

    #[test]
    fn circle_lyapunov_iteration_count_respects_the_step_budget() {
        // Both clamps are independently observable: the count clamps up to
        // MAX_ITER, and the step budget cannot exceed MAX_STEPS.
        let out = circle_map_lyapunov_sum(0.1, 0.25, 0, usize::MAX, 0.1);
        assert_eq!(out[1], MAX_ITER as f64);
        let out = circle_map_lyapunov_sum(0.1, 0.25, MAX_TRANSIENT, MAX_ITER, 0.1);
        assert!(out[0] as u64 + out[1] as u64 <= MAX_STEPS);
    }

    #[test]
    fn delayed_lyapunov_header_reports_the_values_actually_used() {
        let out = delayed_logistic_lyapunov_tile(0.3, 1.55, 1.9, 4, 200, 5000, &[0.4, 0.35]);
        assert_eq!(out[0], 4.0);
        assert_eq!(out[1], 200.0);
        assert_eq!(out[2], 5000.0);
        assert_eq!(out[3], 2.0);
        assert_eq!(out.len(), HEADER + 4 * 2);
    }

    #[test]
    fn delayed_lyapunov_a_is_clamped_to_the_paper_range() {
        // A = -2 clamps to 0: J is [[0, -2*D*y], [1, 0]]. With n_transient =
        // 0 and n_iter = 1 the spectrum is log|diag R| of that J, which is
        // [0, log|2*D*y0|].
        let out = delayed_logistic_lyapunov_tile(
            -2.0,
            DELAYED_LYAP_D_MIN,
            DELAYED_LYAP_D_MIN,
            1,
            0,
            1,
            &[0.4, 0.35],
        );
        let capped = delayed_logistic_lyapunov_tile(
            0.0,
            DELAYED_LYAP_D_MIN,
            DELAYED_LYAP_D_MIN,
            1,
            0,
            1,
            &[0.4, 0.35],
        );
        assert_eq!(out, capped);
        let expected_second = (-2.0 * DELAYED_LYAP_D_MIN * 0.35_f64).abs().ln();
        assert_eq!(out[HEADER], 0.0);
        assert_eq!(out[HEADER + 1], expected_second);
    }

    #[test]
    fn delayed_lyapunov_d_endpoints_are_clamped_to_the_paper_range() {
        // d_max = 99 clamps to DELAYED_LYAP_D_MAX = 2.5. A two-point sweep
        // runs the endpoints: block 0 at d_min = 1.6, block 1 at the clamp.
        let out = delayed_logistic_lyapunov_tile(0.3, 1.6, 99.0, 2, 0, 1, &[0.4, 0.35]);
        assert_eq!(out[0], 2.0);
        let first = delayed_logistic_lyapunov_tile(0.3, 1.6, 1.6, 1, 0, 1, &[0.4, 0.35]);
        let second = delayed_logistic_lyapunov_tile(
            0.3,
            DELAYED_LYAP_D_MAX,
            DELAYED_LYAP_D_MAX,
            1,
            0,
            1,
            &[0.4, 0.35],
        );
        assert_eq!(&out[HEADER..HEADER + 2], &first[HEADER..]);
        assert_eq!(&out[HEADER + 2..], &second[HEADER..]);
    }

    #[test]
    fn delayed_lyapunov_counts_are_clamped_not_refused() {
        let out = delayed_logistic_lyapunov_tile(
            0.3,
            1.55,
            1.9,
            MAX_LYAP_D + 10,
            MAX_TRANSIENT + 10_000,
            MAX_ITER + 10,
            &[0.4, 0.35],
        );
        assert_eq!(out[0], MAX_LYAP_D as f64);
        assert_eq!(out[1], MAX_TRANSIENT as f64);
        assert_eq!(out[2], MAX_ITER as f64);
    }

    #[test]
    fn delayed_lyapunov_state_is_clamped_and_padded() {
        let padded = delayed_logistic_lyapunov_tile(0.3, 1.55, 1.55, 1, 20, 8, &[]);
        let default_state = delayed_logistic_lyapunov_tile(0.3, 1.55, 1.55, 1, 20, 8, &[0.5, 0.5]);
        assert_eq!(padded, default_state);
        let clamped = delayed_logistic_lyapunov_tile(0.3, 1.55, 1.55, 1, 20, 8, &[99.0, -99.0]);
        let at_cap = delayed_logistic_lyapunov_tile(0.3, 1.55, 1.55, 1, 20, 8, &[4.0, -4.0]);
        assert_eq!(clamped, at_cap);
        let nan_state = delayed_logistic_lyapunov_tile(0.3, 1.55, 1.55, 1, 20, 8, &[f64::NAN]);
        assert_eq!(nan_state, default_state);
    }

    #[test]
    fn torus_lyapunov_header_reports_the_values_actually_used() {
        let out = torus_doubling_lyapunov_tile(1, 0.4, 2.19, 2.19, 1, 200, 5000, &[0.5, 0.5, 0.5]);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 200.0);
        assert_eq!(out[2], 5000.0);
        assert_eq!(out[3], 3.0);
        assert_eq!(out.len(), HEADER + 3);
    }

    #[test]
    fn torus_lyapunov_map_kind_snaps_to_one_or_four() {
        let state = [0.5, 0.45, 0.52, 0.48];
        for (kind, dim) in [(0usize, 3.0), (2, 3.0), (3, 4.0), (usize::MAX, 4.0)] {
            let out = torus_doubling_lyapunov_tile(kind, 0.4, 2.19, 2.19, 1, 10, 4, &state);
            assert_eq!(out[3], dim);
        }
    }

    #[test]
    fn torus_lyapunov_a_and_d_are_clamped_to_the_paper_ranges() {
        // A = -2 clamps to 0; d = 99 clamps to TORUS_D_MAX = 2.25. With
        // n_transient = 0 and n_iter = 1, map I's Jacobian is
        // [[0, -2*D*y0, 0], [0, 0, 1], [1, 0, 0]]. Its QR diagonal is
        // [-1, -2*D*y0, -1], so the one-step spectrum is
        // [log(2*D*y0), 0, ~0], sorted descending.
        let out = torus_doubling_lyapunov_tile(1, -2.0, 99.0, 99.0, 1, 0, 1, &[0.4, 0.35, 0.3]);
        let capped = torus_doubling_lyapunov_tile(
            1,
            0.0,
            TORUS_D_MAX,
            TORUS_D_MAX,
            1,
            0,
            1,
            &[0.4, 0.35, 0.3],
        );
        assert_eq!(out, capped);
        let first = (-2.0 * TORUS_D_MAX * 0.35_f64).abs().ln();
        assert_eq!(out[HEADER], first);
        assert_eq!(out[HEADER + 1], 0.0);
        assert!(out[HEADER + 2] <= 0.0);
    }

    #[test]
    fn torus_lyapunov_counts_are_clamped_not_refused() {
        let out = torus_doubling_lyapunov_tile(
            1,
            0.4,
            2.19,
            2.19,
            MAX_LYAP_D + 10,
            MAX_TRANSIENT + 10_000,
            MAX_ITER + 10,
            &[0.5, 0.5, 0.5],
        );
        assert_eq!(out[0], MAX_LYAP_D as f64);
        assert_eq!(out[1], MAX_TRANSIENT as f64);
        assert_eq!(out[2], MAX_ITER as f64);
    }

    #[test]
    fn torus_lyapunov_state_is_clamped_and_padded() {
        let padded = torus_doubling_lyapunov_tile(1, 0.4, 2.19, 2.19, 1, 20, 8, &[]);
        let default_state =
            torus_doubling_lyapunov_tile(1, 0.4, 2.19, 2.19, 1, 20, 8, &[0.5, 0.5, 0.5]);
        assert_eq!(padded, default_state);
        let clamped =
            torus_doubling_lyapunov_tile(1, 0.4, 2.19, 2.19, 1, 0, 1, &[99.0, -99.0, 0.5]);
        let at_cap = torus_doubling_lyapunov_tile(1, 0.4, 2.19, 2.19, 1, 0, 1, &[4.0, -4.0, 0.5]);
        assert_eq!(clamped, at_cap);
        let long =
            torus_doubling_lyapunov_tile(1, 0.4, 2.19, 2.19, 1, 20, 8, &[0.5, 0.5, 0.5, 99.0]);
        assert_eq!(long, default_state);
    }

    #[test]
    fn coupled_lyapunov_header_reports_the_values_actually_used() {
        let out = coupled_delayed_lyapunov_tile(
            0.4,
            2.3,
            2.3,
            1,
            0.005,
            200,
            5000,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 200.0);
        assert_eq!(out[2], 5000.0);
        assert_eq!(out[3], 4.0);
        assert_eq!(out.len(), HEADER + 4);
    }

    #[test]
    fn coupled_lyapunov_db_endpoints_are_clamped_to_the_paper_range() {
        // db_max = 99 clamps to COUPLED_DB_MAX = 2.65. A two-point sweep
        // runs both endpoints; block 1 lands on the clamped value.
        let out =
            coupled_delayed_lyapunov_tile(0.4, 2.3, 99.0, 2, 0.005, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        assert_eq!(out[0], 2.0);
        let second = coupled_delayed_lyapunov_tile(
            0.4,
            COUPLED_DB_MAX,
            COUPLED_DB_MAX,
            1,
            0.005,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(&out[HEADER + 4..], &second[HEADER..]);
    }

    #[test]
    fn coupled_lyapunov_a_eps_and_counts_are_clamped() {
        // A = -2 clamps to 0.
        let out =
            coupled_delayed_lyapunov_tile(-2.0, 2.3, 2.3, 1, 0.005, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        let capped =
            coupled_delayed_lyapunov_tile(0.0, 2.3, 2.3, 1, 0.005, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        assert_eq!(out, capped);
        // eps = 9 clamps to COUPLED_EPS_MAX = 1e-2.
        let out = coupled_delayed_lyapunov_tile(0.4, 2.3, 2.3, 1, 9.0, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        let capped = coupled_delayed_lyapunov_tile(
            0.4,
            2.3,
            2.3,
            1,
            COUPLED_EPS_MAX,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(out, capped);
        // eps = 0 clamps up to COUPLED_EPS_MIN = 1e-3.
        let out = coupled_delayed_lyapunov_tile(0.4, 2.3, 2.3, 1, 0.0, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        let floored = coupled_delayed_lyapunov_tile(
            0.4,
            2.3,
            2.3,
            1,
            COUPLED_EPS_MIN,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(out, floored);
        let out = coupled_delayed_lyapunov_tile(
            0.4,
            2.3,
            2.3,
            MAX_LYAP_DB + 10,
            0.005,
            MAX_TRANSIENT + 10_000,
            MAX_ITER + 10,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(out[0], MAX_LYAP_DB as f64);
        assert_eq!(out[1], MAX_TRANSIENT as f64);
        assert_eq!(out[2], MAX_ITER as f64);
    }

    #[test]
    fn coupled_lyapunov_state_is_clamped_and_padded() {
        // map_state pads missing components to 0.5; a short slice is not the
        // published state (0.5, 0.5, 0.3, 0.3), it is (0.5, 0.5, 0.3, 0.5).
        let padded = coupled_delayed_lyapunov_tile(0.4, 2.3, 2.3, 1, 0.005, 20, 8, &[]);
        let all_half =
            coupled_delayed_lyapunov_tile(0.4, 2.3, 2.3, 1, 0.005, 20, 8, &[0.5, 0.5, 0.5, 0.5]);
        let published =
            coupled_delayed_lyapunov_tile(0.4, 2.3, 2.3, 1, 0.005, 20, 8, &[0.5, 0.5, 0.3, 0.3]);
        assert_eq!(padded, all_half);
        assert_ne!(padded, published);
        let clamped = coupled_delayed_lyapunov_tile(
            0.4,
            2.3,
            2.3,
            1,
            0.005,
            0,
            1,
            &[99.0, -99.0, 99.0, -99.0],
        );
        let at_cap =
            coupled_delayed_lyapunov_tile(0.4, 2.3, 2.3, 1, 0.005, 0, 1, &[4.0, -4.0, 4.0, -4.0]);
        assert_eq!(clamped, at_cap);
    }

    #[test]
    fn phase_header_reports_the_values_actually_used() {
        let out = coupled_logistic_phase_tile(0.8, 1.2, 3, 0.0, 0.2, 2, 5, 4, 0.1, 0.2);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 5.0);
        assert_eq!(out[3], 4.0);
        assert_eq!(out.len(), HEADER + 3 * 2 * 2);
    }

    #[test]
    fn phase_a_endpoints_are_clamped_to_the_paper_range() {
        let low = coupled_logistic_phase_tile(-2.0, 0.8, 1, 0.1, 0.1, 1, 0, 1, 0.1, 0.2);
        let at_min =
            coupled_logistic_phase_tile(LOGISTIC_A_MIN, 0.8, 1, 0.1, 0.1, 1, 0, 1, 0.1, 0.2);
        assert_eq!(low, at_min);
        let high = coupled_logistic_phase_tile(0.8, 9.0, 2, 0.1, 0.1, 1, 0, 1, 0.1, 0.2);
        let at_max =
            coupled_logistic_phase_tile(0.8, LOGISTIC_A_MAX, 2, 0.1, 0.1, 1, 0, 1, 0.1, 0.2);
        assert_eq!(high, at_max);
    }

    #[test]
    fn phase_a_non_finite_falls_back_to_the_paper_endpoints() {
        let out = coupled_logistic_phase_tile(f64::NAN, f64::NAN, 2, 0.1, 0.1, 1, 0, 1, 0.1, 0.2);
        let fallback = coupled_logistic_phase_tile(
            LOGISTIC_A_MIN,
            LOGISTIC_A_MAX,
            2,
            0.1,
            0.1,
            1,
            0,
            1,
            0.1,
            0.2,
        );
        assert_eq!(out, fallback);
    }

    #[test]
    fn phase_d_endpoints_are_clamped_to_the_paper_range() {
        let low = coupled_logistic_phase_tile(1.1, 1.1, 1, -1.0, 0.1, 1, 0, 1, 0.1, 0.2);
        let at_min =
            coupled_logistic_phase_tile(1.1, 1.1, 1, LOGISTIC_D_MIN, 0.1, 1, 0, 1, 0.1, 0.2);
        assert_eq!(low, at_min);
        let high = coupled_logistic_phase_tile(1.1, 1.1, 1, 0.1, 9.0, 2, 0, 1, 0.1, 0.2);
        let at_max =
            coupled_logistic_phase_tile(1.1, 1.1, 1, 0.1, LOGISTIC_D_MAX, 2, 0, 1, 0.1, 0.2);
        assert_eq!(high, at_max);
    }

    #[test]
    fn phase_d_non_finite_falls_back_to_the_paper_endpoints() {
        let out = coupled_logistic_phase_tile(1.1, 1.1, 1, f64::NAN, f64::NAN, 2, 0, 1, 0.1, 0.2);
        let fallback = coupled_logistic_phase_tile(
            1.1,
            1.1,
            1,
            LOGISTIC_D_MIN,
            LOGISTIC_D_MAX,
            2,
            0,
            1,
            0.1,
            0.2,
        );
        assert_eq!(out, fallback);
    }

    #[test]
    fn phase_grid_sides_are_clamped() {
        let out = coupled_logistic_phase_tile(1.1, 1.2, 0, 0.0, 0.1, MAX_SIDE + 10, 0, 1, 0.1, 0.2);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], MAX_SIDE as f64);
        assert_eq!(out.len(), HEADER + MAX_SIDE * 2);
    }

    #[test]
    fn phase_counts_are_clamped() {
        let out = coupled_logistic_phase_tile(
            1.1,
            1.1,
            1,
            0.1,
            0.1,
            1,
            MAX_TRANSIENT + 10_000,
            0,
            0.1,
            0.2,
        );
        assert_eq!(out[2], MAX_TRANSIENT as f64);
        assert_eq!(out[3], 1.0);
        let wide = coupled_logistic_phase_tile(
            1.1,
            1.1,
            1,
            0.1,
            0.1,
            1,
            0,
            MAX_PHASE_SAMPLE + 10,
            0.1,
            0.2,
        );
        assert_eq!(wide[3], MAX_PHASE_SAMPLE as f64);
    }

    #[test]
    fn phase_state_is_clamped_to_the_unit_interval() {
        let high = coupled_logistic_phase_tile(1.1, 1.1, 1, 0.1, 0.1, 1, 0, 1, 5.0, -5.0);
        let capped = coupled_logistic_phase_tile(1.1, 1.1, 1, 0.1, 0.1, 1, 0, 1, 1.0, -1.0);
        assert_eq!(high, capped);
        let missing =
            coupled_logistic_phase_tile(1.1, 1.1, 1, 0.1, 0.1, 1, 0, 1, f64::NAN, f64::NAN);
        let fallback =
            coupled_logistic_phase_tile(1.1, 1.1, 1, 0.1, 0.1, 1, 0, 1, PHASE_X0, PHASE_Y0);
        assert_eq!(missing, fallback);
    }

    #[test]
    fn phase_sample_count_respects_the_step_budget() {
        let n = 200;
        let out = coupled_logistic_phase_tile(0.8, 1.2, n, 0.0, 0.2, n, 0, 10_000, 0.1, 0.2);
        let used = out[3] as u64;
        assert!(used < 10_000);
        assert!((n as u64) * (n as u64) * used <= MAX_STEPS);
    }

    #[test]
    fn attractor_header_reports_the_values_actually_used() {
        let out = coupled_logistic_attractor_tile(1.0, 1.3, 4, 0.1, 6, 5, 0.1, 0.2);
        assert_eq!(out[0], 4.0);
        assert_eq!(out[1], 5.0);
        assert_eq!(out[2], 6.0);
        assert_eq!(out[3], 2.0);
        assert_eq!(out.len(), HEADER + 4 * 5 * 2);
    }

    #[test]
    fn attractor_a_endpoints_are_clamped_to_the_paper_range() {
        let low = coupled_logistic_attractor_tile(-3.0, 1.1, 1, 0.1, 0, 1, 0.1, 0.2);
        let at_min = coupled_logistic_attractor_tile(LOGISTIC_A_MIN, 1.1, 1, 0.1, 0, 1, 0.1, 0.2);
        assert_eq!(low, at_min);
        let high = coupled_logistic_attractor_tile(1.1, 9.0, 2, 0.1, 0, 1, 0.1, 0.2);
        let at_max = coupled_logistic_attractor_tile(1.1, LOGISTIC_A_MAX, 2, 0.1, 0, 1, 0.1, 0.2);
        assert_eq!(high, at_max);
        let missing = coupled_logistic_attractor_tile(f64::NAN, f64::NAN, 2, 0.1, 0, 1, 0.1, 0.2);
        let fallback =
            coupled_logistic_attractor_tile(LOGISTIC_A_MIN, LOGISTIC_A_MAX, 2, 0.1, 0, 1, 0.1, 0.2);
        assert_eq!(missing, fallback);
    }

    #[test]
    fn attractor_d_is_clamped_and_non_finite_falls_back() {
        let high = coupled_logistic_attractor_tile(1.1, 1.1, 1, 4.0, 0, 1, 0.1, 0.2);
        let capped = coupled_logistic_attractor_tile(1.1, 1.1, 1, LOGISTIC_D_MAX, 0, 1, 0.1, 0.2);
        assert_eq!(high, capped);
        let low = coupled_logistic_attractor_tile(1.1, 1.1, 1, -1.0, 0, 1, 0.1, 0.2);
        let floored = coupled_logistic_attractor_tile(1.1, 1.1, 1, LOGISTIC_D_MIN, 0, 1, 0.1, 0.2);
        assert_eq!(low, floored);
        let missing = coupled_logistic_attractor_tile(1.1, 1.1, 1, f64::NAN, 0, 1, 0.1, 0.2);
        let fallback =
            coupled_logistic_attractor_tile(1.1, 1.1, 1, LOGISTIC_D_PAPER, 0, 1, 0.1, 0.2);
        assert_eq!(missing, fallback);
    }

    #[test]
    fn attractor_counts_are_clamped() {
        let out = coupled_logistic_attractor_tile(
            1.0,
            1.2,
            MAX_ATTRACTOR_A + 10,
            0.1,
            MAX_TRANSIENT + 5,
            0,
            0.1,
            0.2,
        );
        assert_eq!(out[0], MAX_ATTRACTOR_A as f64);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], MAX_TRANSIENT as f64);
        let wide =
            coupled_logistic_attractor_tile(1.1, 1.1, 1, 0.1, 0, MAX_MAP_PLOT + 10, 0.1, 0.2);
        assert_eq!(wide[1], MAX_MAP_PLOT as f64);
    }

    #[test]
    fn attractor_state_is_clamped_to_the_unit_interval() {
        let high = coupled_logistic_attractor_tile(1.1, 1.1, 1, 0.1, 0, 1, 8.0, -8.0);
        let capped = coupled_logistic_attractor_tile(1.1, 1.1, 1, 0.1, 0, 1, 1.0, -1.0);
        assert_eq!(high, capped);
        let missing = coupled_logistic_attractor_tile(1.1, 1.1, 1, 0.1, 0, 1, f64::NAN, f64::NAN);
        let fallback = coupled_logistic_attractor_tile(1.1, 1.1, 1, 0.1, 0, 1, PHASE_X0, PHASE_Y0);
        assert_eq!(missing, fallback);
    }

    #[test]
    fn basin_header_reports_the_values_actually_used() {
        let out = coupled_logistic_basin_grid(
            1.35344, 0.1, -1.0, 1.0, 3, -0.5, 0.5, 2, 1, 4, 3, 0.1, 0.6,
        );
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 3.0);
        assert_eq!(out.len(), HEADER + 6);
        assert!(
            out[HEADER..]
                .iter()
                .all(|label| { *label == -1.0 || *label == 0.0 || *label == 1.0 || *label == 2.0 })
        );
    }

    #[test]
    fn basin_a_and_d_are_clamped() {
        let high_a =
            coupled_logistic_basin_grid(9.0, 0.1, -1.0, 1.0, 2, -1.0, 1.0, 2, 0, 1, 1, 0.1, 0.6);
        let a_cap = coupled_logistic_basin_grid(
            LOGISTIC_A_MAX,
            0.1,
            -1.0,
            1.0,
            2,
            -1.0,
            1.0,
            2,
            0,
            1,
            1,
            0.1,
            0.6,
        );
        assert_eq!(high_a, a_cap);
        let missing_a = coupled_logistic_basin_grid(
            f64::NAN,
            0.1,
            -1.0,
            1.0,
            2,
            -1.0,
            1.0,
            2,
            0,
            1,
            1,
            0.1,
            0.6,
        );
        let a_fallback = coupled_logistic_basin_grid(
            BASIN_A_PAPER,
            0.1,
            -1.0,
            1.0,
            2,
            -1.0,
            1.0,
            2,
            0,
            1,
            1,
            0.1,
            0.6,
        );
        assert_eq!(missing_a, a_fallback);
        let high_d =
            coupled_logistic_basin_grid(1.2, 4.0, -1.0, -1.0, 1, 0.5, 0.5, 1, 3, 1, 1, 0.1, 0.6);
        let d_cap = coupled_logistic_basin_grid(
            1.2,
            LOGISTIC_D_MAX,
            -1.0,
            -1.0,
            1,
            0.5,
            0.5,
            1,
            3,
            1,
            1,
            0.1,
            0.6,
        );
        assert_eq!(high_d, d_cap);
        let missing_d = coupled_logistic_basin_grid(
            1.2,
            f64::NAN,
            -1.0,
            1.0,
            2,
            -1.0,
            1.0,
            2,
            0,
            1,
            1,
            0.1,
            0.6,
        );
        let d_fallback = coupled_logistic_basin_grid(
            1.2,
            LOGISTIC_D_PAPER,
            -1.0,
            1.0,
            2,
            -1.0,
            1.0,
            2,
            0,
            1,
            1,
            0.1,
            0.6,
        );
        assert_eq!(missing_d, d_fallback);
    }

    #[test]
    fn basin_ranges_and_reference_state_are_clamped() {
        let wide =
            coupled_logistic_basin_grid(1.2, 0.1, -4.0, 4.0, 2, -4.0, 4.0, 2, 0, 1, 1, 3.0, -3.0);
        let capped =
            coupled_logistic_basin_grid(1.2, 0.1, -1.0, 1.0, 2, -1.0, 1.0, 2, 0, 1, 1, 1.0, -1.0);
        assert_eq!(wide, capped);
        let missing = coupled_logistic_basin_grid(
            1.2,
            0.1,
            f64::NAN,
            f64::NAN,
            2,
            f64::NAN,
            f64::NAN,
            2,
            0,
            1,
            1,
            f64::NAN,
            f64::NAN,
        );
        let fallback = coupled_logistic_basin_grid(
            1.2,
            0.1,
            -1.0,
            1.0,
            2,
            -1.0,
            1.0,
            2,
            0,
            1,
            1,
            BASIN_X_REF,
            BASIN_Y_REF,
        );
        assert_eq!(missing, fallback);
    }

    #[test]
    fn basin_counts_and_period_are_clamped() {
        let out = coupled_logistic_basin_grid(
            1.2,
            0.1,
            -1.0,
            1.0,
            0,
            -1.0,
            1.0,
            MAX_SIDE + 8,
            MAX_TRANSIENT + 10,
            0,
            0,
            0.1,
            0.6,
        );
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], MAX_SIDE as f64);
        assert_eq!(out[2], MAX_TRANSIENT as f64);
        assert_eq!(out[3], 1.0);
        let period = coupled_logistic_basin_grid(
            1.2,
            0.1,
            -1.0,
            1.0,
            1,
            -1.0,
            1.0,
            1,
            0,
            1,
            MAX_BASIN_PERIOD + 20,
            0.1,
            0.6,
        );
        assert_eq!(period[3], MAX_BASIN_PERIOD as f64);
    }

    #[test]
    fn basin_reference_transient_is_clamped() {
        // period = 3 so a one-step shift of the reference window changes the
        // recorded set, and therefore some labels. 50_001 must clamp to 50_000.
        let over = coupled_logistic_basin_grid(
            1.35344, 0.1, -1.0, 1.0, 17, -1.0, 1.0, 17, 0, 50_001, 3, 0.1, 0.6,
        );
        let capped = coupled_logistic_basin_grid(
            1.35344, 0.1, -1.0, 1.0, 17, -1.0, 1.0, 17, 0, 50_000, 3, 0.1, 0.6,
        );
        assert_eq!(over.len(), HEADER + 17 * 17);
        assert_eq!(over, capped);
    }

    #[test]
    fn basin_transient_respects_the_step_budget() {
        let n = 200;
        let out = coupled_logistic_basin_grid(
            1.35344, 0.1, -1.0, 1.0, n, -1.0, 1.0, n, 10_000, 0, 1, 0.1, 0.6,
        );
        let used = out[2] as u64;
        assert!(used < 10_000);
        assert!((n as u64) * (n as u64) * used <= MAX_STEPS);
    }

    #[test]
    fn projection_header_reports_the_values_actually_used() {
        let out =
            coupled_delayed_projection_tile(0.4, 2.2, 2.5, 3, 5e-3, 4, 5, &[0.5, 0.5, 0.3, 0.3]);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 5.0);
        assert_eq!(out[2], 4.0);
        assert_eq!(out[3], 2.0);
        assert_eq!(out.len(), HEADER + 3 * 5 * 2);
    }

    #[test]
    fn projection_a_is_clamped() {
        let high =
            coupled_delayed_projection_tile(3.0, 2.3, 2.3, 1, 5e-3, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        let capped =
            coupled_delayed_projection_tile(1.0, 2.3, 2.3, 1, 5e-3, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        assert_eq!(high, capped);
        let missing = coupled_delayed_projection_tile(
            f64::NAN,
            2.3,
            2.3,
            1,
            5e-3,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        let fallback =
            coupled_delayed_projection_tile(0.4, 2.3, 2.3, 1, 5e-3, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        assert_eq!(missing, fallback);
    }

    #[test]
    fn projection_db_endpoints_are_clamped() {
        let high =
            coupled_delayed_projection_tile(0.4, 2.3, 9.0, 2, 5e-3, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        let second = coupled_delayed_projection_tile(
            0.4,
            COUPLED_DB_MAX,
            COUPLED_DB_MAX,
            1,
            5e-3,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(&high[HEADER + 2..], &second[HEADER..]);
        let low =
            coupled_delayed_projection_tile(0.4, 0.0, 2.3, 1, 5e-3, 0, 1, &[0.5, 0.5, 0.3, 0.3]);
        let floored = coupled_delayed_projection_tile(
            0.4,
            COUPLED_DB_MIN,
            2.3,
            1,
            5e-3,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(low, floored);
        let missing = coupled_delayed_projection_tile(
            0.4,
            f64::NAN,
            f64::NAN,
            1,
            5e-3,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        let fallback = coupled_delayed_projection_tile(
            0.4,
            COUPLED_DB_MIN,
            COUPLED_DB_MAX,
            1,
            5e-3,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(missing, fallback);
    }

    #[test]
    fn projection_eps_is_clamped() {
        let high =
            coupled_delayed_projection_tile(0.4, 2.3, 2.3, 1, 1.0, 1, 1, &[0.5, 0.5, 0.3, 0.3]);
        let capped = coupled_delayed_projection_tile(
            0.4,
            2.3,
            2.3,
            1,
            COUPLED_EPS_MAX,
            1,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(high, capped);
        let low =
            coupled_delayed_projection_tile(0.4, 2.3, 2.3, 1, 0.0, 1, 1, &[0.5, 0.5, 0.3, 0.3]);
        let floored = coupled_delayed_projection_tile(
            0.4,
            2.3,
            2.3,
            1,
            COUPLED_EPS_MIN,
            1,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(low, floored);
        let missing = coupled_delayed_projection_tile(
            0.4,
            2.3,
            2.3,
            1,
            f64::NAN,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        let fallback = coupled_delayed_projection_tile(
            0.4,
            2.3,
            2.3,
            1,
            COUPLED_EPS_PAPER,
            0,
            1,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(missing, fallback);
    }

    #[test]
    fn projection_counts_are_clamped() {
        let out = coupled_delayed_projection_tile(
            0.4,
            2.2,
            2.5,
            MAX_PROJECTION_DB + 8,
            5e-3,
            MAX_TRANSIENT + 4,
            0,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(out[0], MAX_PROJECTION_DB as f64);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], MAX_TRANSIENT as f64);
        let wide = coupled_delayed_projection_tile(
            0.4,
            2.3,
            2.3,
            1,
            5e-3,
            0,
            MAX_MAP_PLOT + 3,
            &[0.5, 0.5, 0.3, 0.3],
        );
        assert_eq!(wide[1], MAX_MAP_PLOT as f64);
    }

    #[test]
    fn projection_state_is_clamped_and_padded() {
        let padded = coupled_delayed_projection_tile(0.4, 2.3, 2.3, 1, 5e-3, 0, 1, &[]);
        let default_state =
            coupled_delayed_projection_tile(0.4, 2.3, 2.3, 1, 5e-3, 0, 1, &[0.5, 0.5, 0.5, 0.5]);
        assert_eq!(padded, default_state);
        let high = coupled_delayed_projection_tile(
            0.4,
            2.3,
            2.3,
            1,
            5e-3,
            0,
            1,
            &[99.0, -99.0, 99.0, -99.0],
        );
        let capped =
            coupled_delayed_projection_tile(0.4, 2.3, 2.3, 1, 5e-3, 0, 1, &[4.0, -4.0, 4.0, -4.0]);
        assert_eq!(high, capped);
    }

    #[test]
    fn fractalization_reuses_the_delayed_logistic_kernel() {
        let fractal = fractalization_attractor_tile(0.3, 1.8, 1.9, 2, 5, 4, &[0.4, 0.35]);
        let delayed = delayed_logistic_attractor_tile(0.3, 1.8, 1.9, 2, 5, 4, &[0.4, 0.35]);
        assert_eq!(fractal, delayed);
    }

    #[test]
    fn fractalization_d_endpoints_use_its_own_range() {
        // 1.4 is inside the delayed-logistic sweep and outside fractalization.
        let low = fractalization_attractor_tile(0.3, 1.4, 1.4, 1, 0, 1, &[0.4, 0.35]);
        let capped =
            fractalization_attractor_tile(0.3, FRACTAL_D_MIN, FRACTAL_D_MIN, 1, 0, 1, &[0.4, 0.35]);
        assert_eq!(low, capped);
        let delayed = delayed_logistic_attractor_tile(0.3, 1.4, 1.4, 1, 0, 1, &[0.4, 0.35]);
        assert_ne!(low, delayed);
        let high = fractalization_attractor_tile(0.3, 1.8, 9.0, 2, 0, 1, &[0.4, 0.35]);
        let at_max = fractalization_attractor_tile(0.3, 1.8, FRACTAL_D_MAX, 2, 0, 1, &[0.4, 0.35]);
        assert_eq!(high, at_max);
    }

    #[test]
    fn fractalization_a_and_counts_are_clamped() {
        let high = fractalization_attractor_tile(-2.0, 1.8, 1.8, 1, 0, 1, &[0.4, 0.35]);
        let capped = fractalization_attractor_tile(0.0, 1.8, 1.8, 1, 0, 1, &[0.4, 0.35]);
        assert_eq!(high, capped);
        let missing = fractalization_attractor_tile(f64::NAN, 1.8, 1.8, 1, 0, 1, &[0.4, 0.35]);
        let fallback = fractalization_attractor_tile(0.3, 1.8, 1.8, 1, 0, 1, &[0.4, 0.35]);
        assert_eq!(missing, fallback);
        let counts = fractalization_attractor_tile(
            0.3,
            1.8,
            1.9,
            MAX_FRACTAL_D + 8,
            MAX_TRANSIENT + 3,
            0,
            &[0.4, 0.35],
        );
        assert_eq!(counts[0], MAX_FRACTAL_D as f64);
        assert_eq!(counts[1], 1.0);
        assert_eq!(counts[2], MAX_TRANSIENT as f64);
        let wide =
            fractalization_attractor_tile(0.3, 1.8, 1.8, 1, 0, MAX_MAP_PLOT + 2, &[0.4, 0.35]);
        assert_eq!(wide[1], MAX_MAP_PLOT as f64);
    }

    #[test]
    fn fractalization_state_is_clamped_and_padded() {
        let padded = fractalization_attractor_tile(0.3, 1.8, 1.8, 1, 0, 1, &[]);
        let default_state = fractalization_attractor_tile(0.3, 1.8, 1.8, 1, 0, 1, &[0.5, 0.5]);
        assert_eq!(padded, default_state);
        let high = fractalization_attractor_tile(0.3, 1.8, 1.8, 1, 0, 1, &[99.0, -99.0]);
        let capped = fractalization_attractor_tile(0.3, 1.8, 1.8, 1, 0, 1, &[4.0, -4.0]);
        assert_eq!(high, capped);
        let nan_state = fractalization_attractor_tile(0.3, 1.8, 1.8, 1, 0, 1, &[f64::NAN, 0.4]);
        let nan_fallback = fractalization_attractor_tile(0.3, 1.8, 1.8, 1, 0, 1, &[0.5, 0.4]);
        assert_eq!(nan_state, nan_fallback);
    }

    #[test]
    fn fractalization_d_non_finite_falls_back_to_its_own_endpoints() {
        let out = fractalization_attractor_tile(0.3, f64::NAN, f64::NAN, 2, 0, 1, &[0.4, 0.35]);
        let fallback =
            fractalization_attractor_tile(0.3, FRACTAL_D_MIN, FRACTAL_D_MAX, 2, 0, 1, &[0.4, 0.35]);
        assert_eq!(out, fallback);
    }
}
