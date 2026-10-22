//! Coupled logistic map kernels.

use ndarray::{Array2, ArrayView2};

use crate::CoreError;

/// Compute basin labels for a fixed two-site coupled logistic map.
///
/// The caller owns reference-orbit construction, grids, and plotting.
/// This kernel runs the grid transient and classifies each point.
pub fn coupled_logistic_basin_grid(
    x_values: &[f64],
    y_values: &[f64],
    a: f64,
    d: f64,
    n_transient: usize,
    ref_a: ArrayView2<f64>,
) -> Result<Array2<i8>, CoreError> {
    if ref_a.ndim() != 2 || ref_a.shape()[1] != 2 {
        return Err(CoreError::invalid_argument(
            "ref_a must have shape (period, 2)",
        ));
    }

    let nx = x_values.len();
    let ny = y_values.len();
    let period = ref_a.shape()[0];
    let basin_len = nx
        .checked_mul(ny)
        .ok_or_else(|| CoreError::invalid_argument("basin grid is too large"))?;

    let ref_pairs: Vec<(f64, f64)> = (0..period)
        .map(|k| (ref_a[[k, 0]], ref_a[[k, 1]]))
        .collect();

    let mut basin = Vec::with_capacity(basin_len);
    for &y0 in y_values {
        basin.extend(basin_row(x_values, y0, a, d, n_transient, &ref_pairs));
    }

    Array2::from_shape_vec((ny, nx), basin)
        .map_err(|e| CoreError::runtime(format!("shape error basin: {e}")))
}

/// One iteration of the coupled Kaneko logistic map.
///
/// ```text
/// x' = (1 - A * x * x) + D * (y - x)
/// y' = (1 - A * y * y) + D * (x - y)
/// ```
///
/// Same operation order as `coupled_logistic`
/// (`src/dynachaos/maps/coupled_logistic.py:142`). No product uses `mul_add`.
#[inline]
fn coupled_step(x: f64, y: f64, a: f64, d: f64) -> (f64, f64) {
    let x_new = logistic(x, a) + d * (y - x);
    let y_new = logistic(y, a) + d * (x - y);
    (x_new, y_new)
}

#[inline]
fn logistic(x: f64, a: f64) -> f64 {
    1.0 - a * x * x
}

fn basin_row(
    x_values: &[f64],
    y0: f64,
    a: f64,
    d: f64,
    n_transient: usize,
    ref_pairs: &[(f64, f64)],
) -> Vec<i8> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        x_values
            .par_iter()
            .map(|&x0| basin_point(x0, y0, a, d, n_transient, ref_pairs))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        x_values
            .iter()
            .map(|&x0| basin_point(x0, y0, a, d, n_transient, ref_pairs))
            .collect()
    }
}

fn basin_point(
    x0: f64,
    y0: f64,
    a: f64,
    d: f64,
    n_transient: usize,
    ref_pairs: &[(f64, f64)],
) -> i8 {
    let mut x = x0;
    let mut y = y0;
    for _ in 0..n_transient {
        let (new_x, new_y) = coupled_step(x, y, a, d);
        if new_x.abs() > 100.0 || new_y.abs() > 100.0 {
            x = f64::NAN;
            y = f64::NAN;
        } else {
            x = new_x;
            y = new_y;
        }
    }

    if x.is_nan() {
        return -1;
    }

    let mut dist_a = f64::INFINITY;
    let mut dist_b = f64::INFINITY;
    for &(ax, ay) in ref_pairs {
        let da = squared_distance(x, y, ax, ay);
        let db = squared_distance(x, y, ay, ax);
        if da < dist_a {
            dist_a = da;
        }
        if db < dist_b {
            dist_b = db;
        }
    }

    if dist_a < dist_b {
        1
    } else if dist_b < dist_a {
        2
    } else {
        0
    }
}

#[inline]
fn squared_distance(x: f64, y: f64, ref_x: f64, ref_y: f64) -> f64 {
    let dx = x - ref_x;
    let dy = y - ref_y;
    dx * dx + dy * dy
}

/// Magnitude past which a phase-diagram orbit is marked divergent.
///
/// `compute_phase_diagram` replaces the state with NaN when `|x| > 10` or
/// `|y| > 10` (`src/dynachaos/maps/coupled_logistic.py:203`).
const PHASE_DIVERGENCE: f64 = 10.0;

/// Apply the phase-diagram divergence mask: `|x| > 10` or `|y| > 10` becomes NaN.
///
/// A NaN component fails the comparison, so an already-NaN state stays NaN,
/// matching `np.where(mask, nan, state)`.
#[inline]
fn phase_mask(x: f64, y: f64) -> (f64, f64) {
    if x.abs() > PHASE_DIVERGENCE || y.abs() > PHASE_DIVERGENCE {
        (f64::NAN, f64::NAN)
    } else {
        (x, y)
    }
}

/// One phase-diagram cell: symmetry-breaking order parameter and largest-tangent growth.
///
/// The tangent step runs before the map step, as
/// `compute_phase_diagram` does (`src/dynachaos/maps/coupled_logistic.py:215`).
/// A sample with no finite positive tangent norm, or a post-step state that
/// the divergence mask has cleared, does not enter its sum. A cell with no
/// valid sample is NaN.
fn phase_cell(a: f64, d: f64, n_transient: usize, n_sample: usize, x0: f64, y0: f64) -> (f64, f64) {
    let mut x = x0;
    let mut y = y0;
    for _ in 0..n_transient {
        let (x_new, y_new) = coupled_step(x, y, a, d);
        (x, y) = phase_mask(x_new, y_new);
    }

    let mut sum_absdiff = 0.0;
    let mut count_absdiff = 0.0;
    let mut lyap_sum = 0.0;
    let mut lyap_count = 0.0;
    let mut vx = 1.0 / 2.0_f64.sqrt();
    let mut vy = 1.0 / 2.0_f64.sqrt();
    for _ in 0..n_sample {
        // Jacobian of `coupled_logistic_jac`: diag is `-2 A u - D`, off-diag is `D`.
        let j11 = -2.0 * a * x - d;
        let j22 = -2.0 * a * y - d;
        let tx = j11 * vx + d * vy;
        let ty = d * vx + j22 * vy;
        let tnorm = (tx * tx + ty * ty).sqrt();
        let valid_tan = x.is_finite() && y.is_finite() && tnorm.is_finite() && tnorm > 0.0;
        if valid_tan {
            lyap_sum += tnorm.ln();
            lyap_count += 1.0;
            vx = tx / tnorm;
            vy = ty / tnorm;
        }

        let (x_new, y_new) = coupled_step(x, y, a, d);
        (x, y) = phase_mask(x_new, y_new);
        if x.is_finite() && y.is_finite() {
            sum_absdiff += (x - y).abs();
            count_absdiff += 1.0;
        }
    }

    let asym = if count_absdiff > 0.0 {
        sum_absdiff / count_absdiff
    } else {
        f64::NAN
    };
    let lyap = if lyap_count > 0.0 {
        lyap_sum / lyap_count
    } else {
        f64::NAN
    };
    (asym, lyap)
}

/// Coupled-logistic phase tile: `(asym, lyap)` for every `(D, A)` pair.
///
/// Mirrors `compute_phase_diagram` (`src/dynachaos/maps/coupled_logistic.py:161`).
/// For each `D` the kernel starts every `A` from `(x0, y0)`, discards
/// `n_transient` steps, then accumulates `n_sample` diagnostics. The tangent
/// vector starts at `(1/sqrt(2), 1/sqrt(2))` and is stepped *before* the map,
/// matching the Python loop. `|x| > 10` or `|y| > 10` sets that state to NaN;
/// a cell with no valid sample is NaN rather than a divide-by-zero.
///
/// The result is flat and D-major: `d_values.len()` rows of `a_values.len()`
/// `(asym, lyap)` pairs, so `out[(j * n_A + i) * 2]` is the asymmetry at
/// `d_values[j]`, `a_values[i]`. Python hardcodes the start `(0.1, 0.2)`;
/// pass that pair to reproduce the paper grid.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when either grid is empty, `n_sample` is
/// zero, or `x0`, `y0`, or any grid value is not finite.
pub fn coupled_logistic_phase_tile(
    a_values: &[f64],
    d_values: &[f64],
    n_transient: usize,
    n_sample: usize,
    x0: f64,
    y0: f64,
) -> Result<Vec<f64>, CoreError> {
    if a_values.is_empty() || d_values.is_empty() {
        return Err(CoreError::invalid_argument(
            "A and D grids must not be empty",
        ));
    }
    if n_sample == 0 {
        return Err(CoreError::invalid_argument("n_sample must be at least 1"));
    }
    if !x0.is_finite() || !y0.is_finite() {
        return Err(CoreError::invalid_argument("x0 and y0 must be finite"));
    }
    if a_values.iter().any(|a| !a.is_finite()) || d_values.iter().any(|d| !d.is_finite()) {
        return Err(CoreError::invalid_argument("A and D grids must be finite"));
    }

    let mut out = Vec::with_capacity(d_values.len() * a_values.len() * 2);
    for &d in d_values {
        for &a in a_values {
            let (asym, lyap) = phase_cell(a, d, n_transient, n_sample, x0, y0);
            out.push(asym);
            out.push(lyap);
        }
    }
    Ok(out)
}

/// Coupled-logistic attractor tile: one `(x, y)` trajectory per `A`.
///
/// Mirrors `compute_attractors` (`src/dynachaos/maps/coupled_logistic.py:265`),
/// which calls `trajectory_after_transient` on [`coupled_step`] at fixed `D`.
/// Python passes no `diverged_fn`, so it has no divergence check. This kernel
/// does the same: it records every post-step sample for the requested length,
/// including non-finite values, and does not apply a magnitude gate.
///
/// The result is flat and A-major: `a_values.len()` blocks of `n_plot`
/// `(x, y)` pairs. `state0` is the initial state for every `A`. The paper
/// gallery uses `D = 0.1` and the per-panel states in `ATTRACTOR_CASES`.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `a_values` is empty, `n_plot` is zero,
/// `state0` does not hold exactly two entries, or `D`, an `A`, or a state
/// component is not finite.
pub fn coupled_logistic_attractor_tile(
    a_values: &[f64],
    d: f64,
    n_transient: usize,
    n_plot: usize,
    state0: &[f64],
) -> Result<Vec<f64>, CoreError> {
    if a_values.is_empty() {
        return Err(CoreError::invalid_argument("a_values must not be empty"));
    }
    if n_plot == 0 {
        return Err(CoreError::invalid_argument("n_plot must be at least 1"));
    }
    if state0.len() != 2 {
        return Err(CoreError::invalid_argument(
            "state0 must hold exactly 2 components",
        ));
    }
    if !d.is_finite() || a_values.iter().any(|a| !a.is_finite()) {
        return Err(CoreError::invalid_argument("A and D must be finite"));
    }
    if state0.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument("state0 must be finite"));
    }

    let mut out = Vec::with_capacity(a_values.len() * n_plot * 2);
    for &a in a_values {
        let mut x = state0[0];
        let mut y = state0[1];
        for _ in 0..n_transient {
            (x, y) = coupled_step(x, y, a, d);
        }
        for _ in 0..n_plot {
            (x, y) = coupled_step(x, y, a, d);
            out.push(x);
            out.push(y);
        }
    }
    Ok(out)
}

/// Reference orbit used by the basin classifier.
///
/// Mirrors `_find_reference_orbit` (`src/dynachaos/maps/coupled_logistic.py:308`):
/// iterate `n_transient` steps of [`coupled_step`] from `(x0, y0)`, then
/// record `period` post-step states. Python passes no `diverged_fn`, so this
/// helper does not apply a magnitude gate; a non-finite state is recorded as
/// produced. The grid classification in [`coupled_logistic_basin_grid`] still
/// marks `|x| > 100` or `|y| > 100` as diverged.
///
/// The result is flat: `period` `(x, y)` pairs, in iteration order.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `period` is zero or `A`, `D`, `x0`, or
/// `y0` is not finite.
pub fn coupled_logistic_reference_orbit(
    a: f64,
    d: f64,
    x0: f64,
    y0: f64,
    n_transient: usize,
    period: usize,
) -> Result<Vec<f64>, CoreError> {
    if period == 0 {
        return Err(CoreError::invalid_argument("period must be at least 1"));
    }
    if !a.is_finite() || !d.is_finite() || !x0.is_finite() || !y0.is_finite() {
        return Err(CoreError::invalid_argument(
            "A, D, x0 and y0 must be finite",
        ));
    }

    let mut x = x0;
    let mut y = y0;
    for _ in 0..n_transient {
        (x, y) = coupled_step(x, y, a, d);
    }
    let mut out = Vec::with_capacity(period * 2);
    for _ in 0..period {
        (x, y) = coupled_step(x, y, a, d);
        out.push(x);
        out.push(y);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values produced once from the Python modules at small N and
    // pasted as literals; the tests do not call Python.

    /// `compute_phase_diagram` at A = [1.1, 1.25], D = [0.0, 0.1], 20 transient
    /// and 8 samples from (0.1, 0.2). D-major `(asym, lyap)` pairs.
    const PHASE_SMALL: [f64; 8] = [
        1.1524449370914347e-6,
        -0.45813844380398705,
        0.05340044492304369,
        -0.021723630347748213,
        9.8879238130678e-16,
        -0.45814198120856403,
        0.00010443188671842296,
        -0.00435970632819202,
    ];

    /// First eight samples of the A = 1.10, D = 0.1 attractor from (0.1, 0.6)
    /// after a 20-step transient. `ATTRACTOR_CASES` panel 0.
    const ATTRACTOR_A_1_10: [f64; 16] = [
        0.9178807160025972,
        -0.2066455449113604,
        -0.03920813578178034,
        1.0654800067365113,
        1.1087778085491984,
        -0.35924122348258913,
        -0.4991289548074579,
        1.0048422208889844,
        0.8763544326897403,
        -0.2610757953388617,
        0.041460176332687504,
        1.038766394799859,
        1.0978397810030258,
        -0.2866698071087637,
        -0.4642283620392279,
        1.0480534226726255,
    ];

    /// `_find_reference_orbit(1.35344, 0.1, 0.1, 0.6, n_transient=8, period=4)`.
    const REFERENCE_ORBIT: [f64; 8] = [
        0.22601185961994902,
        -0.09058510776079458,
        0.8992047464511848,
        1.0205538259018143,
        -0.08221495767497404,
        -0.42178342213866044,
        0.9568948475557361,
        0.7931781716206624,
    ];

    #[test]
    fn phase_tile_matches_python_exactly() {
        let out = coupled_logistic_phase_tile(&[1.1, 1.25], &[0.0, 0.1], 20, 8, 0.1, 0.2).unwrap();
        assert_eq!(out, PHASE_SMALL);
    }

    #[test]
    fn phase_tile_is_d_major() {
        let out = coupled_logistic_phase_tile(&[1.1, 1.25], &[0.0, 0.1], 20, 8, 0.1, 0.2).unwrap();
        assert_eq!(out.len(), 2 * 2 * 2);
        // Second row is D = 0.1. Its first pair is the A = 1.1 cell.
        assert_eq!(out[4], 9.8879238130678e-16);
        assert_eq!(out[5], -0.45814198120856403);
    }

    #[test]
    fn phase_divergence_marks_the_cell_nan() {
        // A = 10 from (1, 0.2) exceeds |x| > 10 on the second step, so a
        // 2-step transient leaves no valid sample. The neighbouring A = 1.1
        // cell stays finite.
        let out = coupled_logistic_phase_tile(&[1.1, 10.0], &[0.0], 2, 4, 1.0, 0.2).unwrap();
        assert!(out[0].is_finite() && out[1].is_finite());
        assert!(out[2].is_nan() && out[3].is_nan());
    }

    #[test]
    fn phase_rejects_invalid_input() {
        assert!(coupled_logistic_phase_tile(&[], &[0.1], 1, 1, 0.1, 0.2).is_err());
        assert!(coupled_logistic_phase_tile(&[1.1], &[], 1, 1, 0.1, 0.2).is_err());
        assert!(coupled_logistic_phase_tile(&[1.1], &[0.1], 1, 0, 0.1, 0.2).is_err());
        assert!(coupled_logistic_phase_tile(&[f64::NAN], &[0.1], 1, 1, 0.1, 0.2).is_err());
        assert!(coupled_logistic_phase_tile(&[1.1], &[0.1], 1, 1, f64::NAN, 0.2).is_err());
    }

    #[test]
    fn attractor_tile_matches_python_exactly() {
        let out = coupled_logistic_attractor_tile(&[1.10], 0.1, 20, 8, &[0.1, 0.6]).unwrap();
        assert_eq!(out, ATTRACTOR_A_1_10);
    }

    #[test]
    fn attractor_tile_is_a_major() {
        let out = coupled_logistic_attractor_tile(&[1.10, 1.25], 0.1, 20, 4, &[0.1, 0.6]).unwrap();
        assert_eq!(out.len(), 2 * 4 * 2);
        assert_eq!(&out[..8], &ATTRACTOR_A_1_10[..8]);
        // Python first pair at A = 1.25.
        assert_eq!(out[8], 0.8240478749648047);
        assert_eq!(out[9], -0.33665629859016794);
    }

    #[test]
    fn attractor_records_non_finite_without_a_divergence_gate() {
        // A = 1e200, D = 0, from (1, 0). Step 1 is finite (`-1e200`, `1`).
        // Step 2 overflows. Python records both; a whole-block NaN gate would
        // wipe the finite sample too.
        let out = coupled_logistic_attractor_tile(&[1e200], 0.0, 0, 2, &[1.0, 0.0]).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out[0].is_finite());
        assert!(!out[2].is_finite());
    }

    #[test]
    fn attractor_rejects_invalid_input() {
        assert!(coupled_logistic_attractor_tile(&[], 0.1, 1, 1, &[0.1, 0.2]).is_err());
        assert!(coupled_logistic_attractor_tile(&[1.1], 0.1, 1, 0, &[0.1, 0.2]).is_err());
        assert!(coupled_logistic_attractor_tile(&[1.1], 0.1, 1, 1, &[0.1]).is_err());
        assert!(coupled_logistic_attractor_tile(&[1.1], f64::NAN, 1, 1, &[0.1, 0.2]).is_err());
        assert!(coupled_logistic_attractor_tile(&[1.1], 0.1, 1, 1, &[0.1, f64::NAN]).is_err());
    }

    #[test]
    fn reference_orbit_matches_python_exactly() {
        let out = coupled_logistic_reference_orbit(1.35344, 0.1, 0.1, 0.6, 8, 4).unwrap();
        assert_eq!(out, REFERENCE_ORBIT);
    }

    #[test]
    fn reference_orbit_rejects_invalid_input() {
        assert!(coupled_logistic_reference_orbit(1.35, 0.1, 0.1, 0.6, 1, 0).is_err());
        assert!(coupled_logistic_reference_orbit(f64::NAN, 0.1, 0.1, 0.6, 1, 2).is_err());
    }
}
