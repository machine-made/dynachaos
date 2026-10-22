//! Sine circle map rotation numbers over a rectangular (Omega, K) tile.
//!
//! The map is the one the reproduction pipeline uses
//! (`src/dynachaos/maps/arnold_tongues.py`, `src/dynachaos/maps/circle_map.py`):
//!
//! ```text
//! theta_{n+1} = theta_n + Omega + K * sin(2 * pi * theta_n)
//! ```
//!
//! Note the sine term carries `K` directly, not `K / (2 * pi)`. In this
//! convention the map loses invertibility at `K = 1 / (2 * pi)`, about 0.159,
//! which is the critical line the Arnold-tongue figure draws.
//!
//! The iteration is deliberately not reduced modulo 1. The rotation number is
//! the mean drift of the unwrapped angle, so wrapping would destroy it.

use crate::CoreError;

/// How a single-cell rotation-number computation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitKind {
    /// The unwrapped orbit closed with winding `p` over period `q`.
    Locked { p: i64, q: usize },
    /// The running estimate stabilised within half the display tolerance.
    BoundedError,
    /// The configured iteration budget was used.
    Exhausted,
}

const LOCK_TOLERANCE: f64 = 1e-12;
const MAX_LOCK_PERIOD: usize = 32;
/// Periods up to this are tested at every step; longer ones are swept.
const LOCK_SCAN_ALWAYS: usize = 1;
/// One period in this many is tested per step above `LOCK_SCAN_ALWAYS`.
const LOCK_SCAN_STRIDE: usize = 32;
const MIN_BOUNDED_ERROR_STEPS: usize = 50;
/// The bounded-error test is evaluated on every this-many-th step.
const BOUNDED_CHECK_EVERY: usize = 8;
/// Invertibility threshold: above this K the C/m tail is not a bound.
const K_CRITICAL: f64 = 1.0 / std::f64::consts::TAU;
/// Half the colour resolution (1/256): stop when dyadic windows agree below this.
const HALF_DISPLAY_TOLERANCE: f64 = 0.5 / 256.0;

struct PendingLock {
    p: i64,
    q: usize,
    detect_step: usize,
}

#[cfg(test)]
fn same_rational(a: (i64, usize), b: (i64, usize)) -> bool {
    a.0 * b.1 as i64 == b.0 * a.1 as i64
}
fn gcd_usize(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd_usize(b, a % b) }
}

fn reduce_rational(p: i64, q: usize) -> (i64, usize) {
    let g = gcd_usize(p.unsigned_abs() as usize, q);
    let p = p / g as i64;
    let q = q / g;
    (p, q)
}

/// Values equally spaced from `start` to `stop`, `stop` included.
///
/// This reproduces `numpy.linspace`: each value is `start + i * step`, and the
/// last value is forced to `stop` so the endpoint is exact. A single point
/// returns `start`, as numpy does.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    let mut values: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    values[n - 1] = stop;
    values
}

/// Look for a closed orbit, testing a rotating subset of the possible periods.
///
/// Once an orbit is on a cycle of period `q`, `theta_n - theta_{n-q}` is the
/// same integer at every step, so the closure test does not have to try every
/// period every time. Testing one period in `LOCK_SCAN_STRIDE` costs a quarter
/// of the work and finds the same lock at most `LOCK_SCAN_STRIDE - 1` steps
/// later. Short periods carry most of the tongues, so they are tested every
/// step and the stride applies only above `LOCK_SCAN_ALWAYS`.
fn detect_lock(theta: f64, ring: &[f64; 32], step: usize) -> Option<(i64, usize)> {
    let max_q = MAX_LOCK_PERIOD.min(step - 1);
    let mut q = 1;
    while q <= max_q {
        let theta_n = ring[(step - q - 1) % 32];
        let delta = theta - theta_n;
        let p = delta.round() as i64;
        if (delta - p as f64).abs() < LOCK_TOLERANCE {
            return Some((p, q));
        }
        q += if q < LOCK_SCAN_ALWAYS {
            1
        } else {
            LOCK_SCAN_STRIDE
        };
    }
    // Sweep the strided periods across steps so every one is still reached.
    let phase = step % LOCK_SCAN_STRIDE;
    if phase != 0 {
        let mut q = LOCK_SCAN_ALWAYS + phase;
        while q <= max_q {
            let theta_n = ring[(step - q - 1) % 32];
            let delta = theta - theta_n;
            let p = delta.round() as i64;
            if (delta - p as f64).abs() < LOCK_TOLERANCE {
                return Some((p, q));
            }
            q += LOCK_SCAN_STRIDE;
        }
    }
    None
}

/// Advance pending-lock state. While waiting for confirmation (`step <
/// detect_step + q`), ignore intermediate detections. At the confirmation
/// step, require `theta - theta_{step-q}` to match the stored winding `p`.
/// Store the raw `(p, q)` from [`detect_lock`]; reduce only on return.
fn update_pending_lock(
    pending_lock: &mut Option<PendingLock>,
    step: usize,
    theta: f64,
    ring: &[f64; 32],
    detected: Option<(i64, usize)>,
) -> Option<(i64, usize)> {
    if let Some(pending) = pending_lock.as_ref() {
        if step < pending.detect_step + pending.q {
            return None;
        }
        if step == pending.detect_step + pending.q {
            let theta_n = ring[(step - pending.q - 1) % 32];
            let delta = theta - theta_n;
            if (delta - pending.p as f64).abs() < LOCK_TOLERANCE {
                let confirmed = reduce_rational(pending.p, pending.q);
                *pending_lock = None;
                return Some(confirmed);
            }
            *pending_lock = None;
            if let Some((p, q)) = detected {
                *pending_lock = Some(PendingLock {
                    p,
                    q,
                    detect_step: step,
                });
            }
            return None;
        }
        *pending_lock = None;
    }

    if let Some((p, q)) = detected {
        *pending_lock = Some(PendingLock {
            p,
            q,
            detect_step: step,
        });
    }
    None
}

/// Core iteration with independently switchable lock and display-stop exits.
///
/// Returns rotation number, exit kind, and the number of map steps (sine
/// evaluations) performed. The tile path turns both exits on. The quoted
/// single-point path keeps lock detection and turns the display stop off.
/// [`rotation_number_full`] turns both off.
fn rotation_number_compute(
    omega: f64,
    k: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
    allow_lock: bool,
    allow_display_stop: bool,
) -> (f64, ExitKind, usize) {
    const TWO_PI: f64 = std::f64::consts::TAU;
    let mut theta = theta0;
    let mut ring = [0.0_f64; 32];
    let mut step = 0usize;
    let mut map_steps = 0usize;
    let mut pending_lock: Option<PendingLock> = None;

    let mut advance = |theta: &mut f64| {
        *theta += omega + k * (TWO_PI * *theta).sin();
        map_steps += 1;
    };
    for _ in 0..n_transient {
        advance(&mut theta);
        step += 1;
        ring[(step - 1) % 32] = theta;
    }

    let theta_start = theta;
    let mut measure_thetas: Vec<f64> = if allow_display_stop {
        Vec::with_capacity(n_iter)
    } else {
        Vec::new()
    };

    for m in 1..=n_iter {
        advance(&mut theta);
        step += 1;

        if allow_lock {
            let detected = detect_lock(theta, &ring, step);
            if let Some((p, q)) =
                update_pending_lock(&mut pending_lock, step, theta, &ring, detected)
            {
                let rho_mean = (theta - theta_start) / m as f64;
                let exact = p as f64 / q as f64;
                let lock_mean_tol = (1.0 / n_iter as f64).min(HALF_DISPLAY_TOLERANCE);
                if m >= MIN_BOUNDED_ERROR_STEPS && (rho_mean - exact).abs() < lock_mean_tol {
                    return (exact, ExitKind::Locked { p, q }, map_steps);
                }
                pending_lock = None;
            }
        }

        if allow_display_stop {
            measure_thetas.push(theta);
        }
        ring[(step - 1) % 32] = theta;
        // Above K_c the map is non-invertible; the running-mean tail is not a
        // trustworthy bound, so only lock detection can end the measure phase.
        // The estimate moves like C/m, so testing it every step buys nothing and
        // costs two divisions each time. Testing every BOUNDED_CHECK_EVERY steps
        // delays a stop by at most that many steps out of several hundred.
        if allow_display_stop
            && k <= K_CRITICAL
            && m >= MIN_BOUNDED_ERROR_STEPS
            && m % BOUNDED_CHECK_EVERY == 0
        {
            let rho_m = (theta - theta_start) / m as f64;
            let half_m = m / 2;
            let rho_half = (measure_thetas[half_m - 1] - theta_start) / half_m as f64;
            if (rho_m - rho_half).abs() < HALF_DISPLAY_TOLERANCE {
                let mut estimates = [rho_m, rho_half, 0.0, 0.0, 0.0];
                let mut count = 2usize;
                for divisor in [4usize, 8, 16] {
                    let window = m / divisor;
                    if window >= MIN_BOUNDED_ERROR_STEPS {
                        estimates[count] =
                            (measure_thetas[window - 1] - theta_start) / window as f64;
                        count += 1;
                    }
                }
                // m and m/2 alone can agree while the mean is still biased; four
                // dyadic windows must agree (worst |early-full|=4.0e-3 with m/2 only).
                if count >= 4 {
                    let mut worst = 0.0_f64;
                    for i in 0..count {
                        for j in (i + 1)..count {
                            worst = worst.max((estimates[i] - estimates[j]).abs());
                        }
                    }
                    if worst < HALF_DISPLAY_TOLERANCE {
                        return (rho_m, ExitKind::BoundedError, map_steps);
                    }
                }
            }
        }
    }

    let rho = (theta - theta_start) / n_iter as f64;
    (rho, ExitKind::Exhausted, map_steps)
}

/// Rotation number of one (Omega, K) cell with early exit.
///
/// `theta` runs `n_transient` steps to settle onto the attractor, then
/// `n_iter` more steps unless the orbit locks or the estimate stabilises.
#[inline]
fn rotation_number(omega: f64, k: f64, n_transient: usize, n_iter: usize, theta0: f64) -> f64 {
    rotation_number_compute(omega, k, n_transient, n_iter, theta0, true, true).0
}

/// Same as [`rotation_number`] but also reports how the iteration ended and how
/// many map steps were taken. Exposed for tests and the wasm parity helper.
pub fn rotation_number_with_exit(
    omega: f64,
    k: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> (f64, ExitKind, usize) {
    rotation_number_compute(omega, k, n_transient, n_iter, theta0, true, true)
}

/// Full fixed-count rotation number with no early exit.
pub fn rotation_number_full(
    omega: f64,
    k: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> f64 {
    rotation_number_compute(omega, k, n_transient, n_iter, theta0, false, false).0
}

/// Rotation number of one (Omega, K) point with lock detection and no display stop.
///
/// Locked orbits return the exact rational `p / q`. Unlocked orbits run the
/// full `n_iter` average, matching [`rotation_number_full`]. The tile path
/// keeps the display-tolerance stop; this is the value a reader can quote.
pub fn rotation_number_point(
    omega: f64,
    k: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> f64 {
    rotation_number_compute(omega, k, n_transient, n_iter, theta0, true, false).0
}

/// Rotation numbers over a rectangular tile of the (Omega, K) plane.
///
/// The tile has `n_k` rows and `n_omega` columns. Omega runs from `omega_min`
/// to `omega_max` along each row; K runs from `k_min` to `k_max` down the
/// rows. Both ends are included, as in `numpy.linspace`.
///
/// Returns one `Vec<f64>` of `n_k * n_omega` values in row-major order: the
/// cell at row `i` (K index) and column `j` (Omega index) is at
/// `i * n_omega + j`. This matches the `rho` array the reproduction pipeline
/// stores, which has shape `(n_K, n_omega)`.
///
/// Every cell is independent, so the parallel and sequential builds return
/// identical values; there is no reduction whose order could differ.
///
/// This function validates its arguments and reports a message on bad input.
/// It does not clamp: clamping ranges for untrusted callers belongs in the
/// browser binding, which has to accept whatever a slider sends, while the
/// Python caller must still be able to ask for a paper-scale tile.
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
) -> Result<Vec<f64>, CoreError> {
    if n_omega < 1 || n_k < 1 {
        return Err(CoreError::invalid_argument("n_omega and n_k must be >= 1"));
    }
    if n_iter < 1 {
        return Err(CoreError::invalid_argument("n_iter must be >= 1"));
    }
    for (name, value) in [
        ("omega_min", omega_min),
        ("omega_max", omega_max),
        ("k_min", k_min),
        ("k_max", k_max),
        ("theta0", theta0),
    ] {
        if !value.is_finite() {
            return Err(CoreError::invalid_argument(format!(
                "{name} must be finite"
            )));
        }
    }
    let cells = n_k
        .checked_mul(n_omega)
        .ok_or_else(|| CoreError::invalid_argument("n_k * n_omega overflows"))?;

    let omega_values = linspace(omega_min, omega_max, n_omega);
    let k_values = linspace(k_min, k_max, n_k);
    let mut rho = vec![0.0_f64; cells];

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        rho.par_chunks_mut(n_omega)
            .zip(k_values.par_iter())
            .for_each(|(row, &k)| {
                for (cell, &omega) in row.iter_mut().zip(omega_values.iter()) {
                    *cell = rotation_number(omega, k, n_transient, n_iter, theta0);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (row, &k) in rho.chunks_mut(n_omega).zip(k_values.iter()) {
            for (cell, &omega) in row.iter_mut().zip(omega_values.iter()) {
                *cell = rotation_number(omega, k, n_transient, n_iter, theta0);
            }
        }
    }

    Ok(rho)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values come from the Python pipeline, measured at
    // n_transient=2000, n_iter=5000, theta0=0.1.
    const N_TRANSIENT: usize = 2000;
    const N_ITER: usize = 5000;
    const THETA0: f64 = 0.1;

    const DISPLAY_TOLERANCE: f64 = 1.0 / 256.0;

    fn single(omega: f64, k: f64) -> f64 {
        let tile =
            rotation_number_tile(omega, omega, 1, k, k, 1, N_TRANSIENT, N_ITER, THETA0).unwrap();
        tile[0]
    }

    #[test]
    fn without_coupling_the_rotation_number_is_omega() {
        // K = 0 leaves theta += Omega, so the drift per step is Omega. The
        // tolerance is not zero because 5000 unwrapped additions accumulate
        // rounding; Python gives 0.29999999999996246 for the same input.
        assert!((single(0.3, 0.0) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn zero_tongue_locks_the_rotation_number_to_zero() {
        // Omega = 0 with K > 0 drives theta to the fixed point at 0.5, so the
        // angle stops drifting: this is the 0/1 Arnold tongue.
        assert_eq!(single(0.0, 0.2), 0.0);
    }

    #[test]
    fn half_tongue_locks_the_rotation_number_to_one_half() {
        // The 1/2 tongue: a period-2 cycle advancing half a turn per step.
        assert!((single(0.5, 0.2) - 0.5).abs() < 1e-12);
        assert!((single(0.5, 0.05) - 0.5).abs() < 1e-12);

        let (rho, kind, _) = rotation_number_with_exit(0.5, 0.2, N_TRANSIENT, N_ITER, THETA0);
        assert!((rho - 0.5).abs() < 1e-12);
        match kind {
            ExitKind::Locked { p, q } => assert!(same_rational((p, q), (1, 2))),
            other => panic!("expected Locked 1/2, got {other:?}"),
        }

        let (rho_third, kind_third, _) =
            rotation_number_with_exit(1.0 / 3.0, 0.2, N_TRANSIENT, N_ITER, THETA0);
        assert!((rho_third - 1.0 / 3.0).abs() < 1e-12);
        match kind_third {
            ExitKind::Locked { p, q } => assert!(same_rational((p, q), (1, 3))),
            other => panic!("expected Locked 1/3, got {other:?}"),
        }
    }

    #[test]
    fn tile_layout_is_row_major_with_k_down_the_rows() {
        // Two K rows, three Omega columns. Row 0 is K = 0, where every cell
        // must equal its own Omega.
        let tile = rotation_number_tile(0.0, 0.5, 3, 0.0, 0.2, 2, 200, 500, THETA0).unwrap();
        assert_eq!(tile.len(), 6);
        for (j, expected) in [0.0, 0.25, 0.5].iter().enumerate() {
            assert!((tile[j] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn bad_arguments_are_reported() {
        assert!(rotation_number_tile(0.0, 1.0, 0, 0.0, 0.3, 4, 10, 10, 0.1).is_err());
        assert!(rotation_number_tile(0.0, 1.0, 4, 0.0, 0.3, 4, 10, 0, 0.1).is_err());
        assert!(rotation_number_tile(f64::NAN, 1.0, 4, 0.0, 0.3, 4, 10, 10, 0.1).is_err());
    }

    #[test]
    fn linspace_matches_numpy_including_the_endpoint() {
        assert_eq!(linspace(0.0, 1.0, 1), vec![0.0]);
        assert_eq!(linspace(0.0, 1.0, 3), vec![0.0, 0.5, 1.0]);
        // The endpoint is exact even when the step does not divide evenly.
        let values = linspace(0.0, 0.3, 1000);
        assert_eq!(values[999], 0.3);
    }

    #[test]
    fn early_exit_matches_full_within_display_tolerance() {
        const N_OMEGA: usize = 48;
        const N_K: usize = 24;
        const N_TRANSIENT_GRID: usize = 200;
        const N_ITER_GRID: usize = 2000;
        const THETA0_GRID: f64 = 0.1;

        let omega_values = linspace(0.0, 1.0, N_OMEGA);
        let k_values = linspace(0.0, 0.3, N_K);

        let mut worst_all = 0.0_f64;
        let mut worst_locked = 0.0_f64;
        let mut worst_locked_correction = 0.0_f64;

        for &k in &k_values {
            for &omega in &omega_values {
                let (rho_early, kind, _) =
                    rotation_number_with_exit(omega, k, N_TRANSIENT_GRID, N_ITER_GRID, THETA0_GRID);
                let rho_full =
                    rotation_number_full(omega, k, N_TRANSIENT_GRID, N_ITER_GRID, THETA0_GRID);
                let diff = (rho_early - rho_full).abs();
                worst_all = worst_all.max(diff);

                if let ExitKind::Locked { p, q } = kind {
                    worst_locked = worst_locked.max(diff);
                    let exact = p as f64 / q as f64;
                    worst_locked_correction = worst_locked_correction.max((exact - rho_full).abs());
                }

                assert!(
                    diff <= DISPLAY_TOLERANCE,
                    "omega={omega}, k={k}: |early-full|={diff} > {DISPLAY_TOLERANCE}"
                );
            }
        }

        let locked_o1m_bound = 1.0 / N_ITER_GRID as f64;
        assert!(
            worst_locked <= locked_o1m_bound,
            "worst locked early-vs-full error {worst_locked} exceeds O(1/m) bound {locked_o1m_bound}"
        );

        eprintln!("WORST_ALL={worst_all:.6e}");
        eprintln!("WORST_LOCKED={worst_locked:.6e}");
        eprintln!("WORST_LOCKED_CORRECTION={worst_locked_correction:.6e}");
    }

    #[test]
    fn iteration_gain_on_base_view() {
        const N_OMEGA: usize = 240;
        const N_K: usize = 72;
        const N_TRANSIENT_VIEW: usize = 200;
        const N_ITER_VIEW: usize = 2000;
        const THETA0_VIEW: f64 = 0.1;

        let omega_values = linspace(0.0, 1.0, N_OMEGA);
        let k_values = linspace(0.0, 0.3, N_K);
        let cells = N_OMEGA * N_K;
        let iterations_before = cells * (N_TRANSIENT_VIEW + N_ITER_VIEW);
        let mut iterations_after = 0usize;
        let mut worst_all = 0.0_f64;
        let mut worst_locked = 0.0_f64;

        for &k in &k_values {
            for &omega in &omega_values {
                let (rho_early, kind, steps) =
                    rotation_number_with_exit(omega, k, N_TRANSIENT_VIEW, N_ITER_VIEW, THETA0_VIEW);
                iterations_after += steps;
                let rho_full =
                    rotation_number_full(omega, k, N_TRANSIENT_VIEW, N_ITER_VIEW, THETA0_VIEW);
                let diff = (rho_early - rho_full).abs();
                worst_all = worst_all.max(diff);
                if let ExitKind::Locked { .. } = kind {
                    worst_locked = worst_locked.max(diff);
                }
            }
        }

        let fraction = iterations_after as f64 / iterations_before as f64;
        eprintln!("ITERATIONS_BEFORE={iterations_before}");
        eprintln!("ITERATIONS_AFTER={iterations_after}");
        eprintln!("WORST_ALL={worst_all:.6e}");
        eprintln!("WORST_LOCKED={worst_locked:.6e}");
        eprintln!(
            "ITERATION_FRACTION={:.4} ({:.1}%)",
            fraction,
            fraction * 100.0
        );
    }

    #[test]
    fn point_kernel_locked_cell_is_exact_rational() {
        const N_TRANSIENT_LIVE: usize = 200;
        const N_ITER_LIVE: usize = 2000;
        let rho = rotation_number_point(0.05, 0.12, N_TRANSIENT_LIVE, N_ITER_LIVE, THETA0);
        assert_eq!(rho, 0.0);
        assert_eq!((rho - 0.0).abs(), 0.0);

        let half = rotation_number_point(0.5, 0.2, N_TRANSIENT, N_ITER, THETA0);
        assert_eq!(half, 0.5);
        assert_eq!((half - 0.5).abs(), 0.0);
    }

    #[test]
    fn point_kernel_unlocked_cell_matches_full() {
        const N_TRANSIENT_LIVE: usize = 200;
        const N_ITER_LIVE: usize = 2000;
        let omega = 0.08;
        let k = 0.03;
        let (_, kind, _) =
            rotation_number_with_exit(omega, k, N_TRANSIENT_LIVE, N_ITER_LIVE, THETA0);
        assert!(
            !matches!(kind, ExitKind::Locked { .. }),
            "expected an unlocked cell at (0.08, 0.03), got {kind:?}"
        );
        let point = rotation_number_point(omega, k, N_TRANSIENT_LIVE, N_ITER_LIVE, THETA0);
        let full = rotation_number_full(omega, k, N_TRANSIENT_LIVE, N_ITER_LIVE, THETA0);
        let diff = (point - full).abs();
        assert!(
            diff <= 1e-12,
            "unlocked point vs full: |{point} - {full}| = {diff} > 1e-12"
        );
    }
}
