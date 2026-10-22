//! Lyapunov exponent and spectra for the low-dimensional maps.
//!
//! `circle_map_lyapunov_sum` mirrors `lyapunov_exponent`
//! (`src/dynachaos/maps/circle_map.py:75`). The three tiles mirror
//! `lyapunov_spectrum` (`src/dynachaos/diagnostics/lyapunov.py:76`): discard
//! the transient, then at every step form `Q = J(x) @ Q`, step `x = f(x)`,
//! replace `Q` by the Q factor of its QR, and accumulate `log|diag R|` with
//! the `1e-300` guard. The spectrum is that sum divided by `n_iter`, sorted
//! descending. One Householder QR serves the 2-, 3- and 4-dimensional maps.
//! It uses the LAPACK reflector sign (`beta = -sign(alpha) * hypot`) so the
//! frame stays on the same branch as `np.linalg.qr`; the exponents are still
//! compared with a tolerance, not bit for bit.
//!
//! No product is evaluated with `mul_add`. Map steps use the same operation
//! order as the Python line they mirror, so the state orbit of the three
//! polynomial maps is bit-identical to Python.

use crate::CoreError;
use crate::maps::{COUPLED_DA_OFFSET, coupled_delayed_step};

/// Floor used when a circle-map derivative is zero.
///
/// `lyapunov_exponent` adds this instead of `log(0)`
/// (`src/dynachaos/maps/circle_map.py:85`).
const ZERO_DERIVATIVE_LOG: f64 = -100.0;

/// Replacement for a zero singular value before the logarithm.
///
/// `lyapunov_spectrum` uses `np.where(diag > 0, diag, 1e-300)`
/// (`src/dynachaos/diagnostics/lyapunov.py:139`).
const SINGULAR_VALUE_FLOOR: f64 = 1e-300;

fn require_positive_iter(n_iter: usize) -> Result<(), CoreError> {
    if n_iter == 0 {
        Err(CoreError::invalid_argument("n_iter must be at least 1"))
    } else {
        Ok(())
    }
}

fn require_finite(name: &str, value: f64) -> Result<(), CoreError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(CoreError::invalid_argument(format!(
            "{name} must be finite"
        )))
    }
}

fn require_finite_slice(name: &str, values: &[f64]) -> Result<(), CoreError> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(CoreError::invalid_argument(format!(
            "{name} must contain only finite values"
        )))
    } else {
        Ok(())
    }
}

fn identity<const N: usize>() -> [[f64; N]; N] {
    let mut q = [[0.0; N]; N];
    let mut i = 0;
    while i < N {
        q[i][i] = 1.0;
        i += 1;
    }
    q
}

fn sort_descending<const N: usize>(values: &mut [f64; N]) {
    let mut i = 0;
    while i < N {
        let mut best = i;
        let mut j = i + 1;
        while j < N {
            if values[j] > values[best] {
                best = j;
            }
            j += 1;
        }
        values.swap(i, best);
        i += 1;
    }
}

/// Column norm of `r[k+1:, k]`, folded with `hypot` so a large entry cannot
/// overflow the sum of squares. An empty tail (the last column) is `0`.
fn subcolumn_norm<const N: usize>(r: &[[f64; N]; N], k: usize) -> f64 {
    let mut norm = 0.0_f64;
    let mut i = k + 1;
    while i < N {
        norm = norm.hypot(r[i][k]);
        i += 1;
    }
    norm
}

/// LAPACK-style Householder QR of a square matrix.
///
/// Returns `(Q, diag R)` for `A = Q R`. The reflector is
/// `H = I - tau v v^T` with `v[k] = 1` and
/// `beta = -sign(alpha) * hypot(alpha, ||x||)`, the sign `DLARFG` uses
/// (`sign(+0) = +1`). A column that is already zero below the pivot gets
/// `tau = 0` and keeps its pivot, including an exact zero. Subdiagonal
/// entries of `R` are not returned; the caller only needs `|diag R|`.
fn householder_qr<const N: usize>(a: &[[f64; N]; N]) -> ([[f64; N]; N], [f64; N]) {
    let mut r = *a;
    let mut q = identity::<N>();
    let mut diag = [0.0; N];
    let mut k = 0;
    while k < N {
        let alpha = r[k][k];
        let xnorm = subcolumn_norm(&r, k);
        let mut v = [0.0; N];
        let (beta, tau) = if xnorm == 0.0 {
            (alpha, 0.0)
        } else {
            let beta = -alpha.signum() * alpha.hypot(xnorm);
            let tau = (beta - alpha) / beta;
            let scale = 1.0 / (alpha - beta);
            v[k] = 1.0;
            let mut i = k + 1;
            while i < N {
                v[i] = r[i][k] * scale;
                i += 1;
            }
            (beta, tau)
        };
        diag[k] = beta;
        if tau != 0.0 {
            apply_reflector_left(&mut r, k, tau, &v);
            apply_reflector_right(&mut q, k, tau, &v);
        }
        r[k][k] = beta;
        let mut i = k + 1;
        while i < N {
            r[i][k] = 0.0;
            i += 1;
        }
        k += 1;
    }
    (q, diag)
}

/// `R[k:, k:] = (I - tau v v^T) R[k:, k:]`, with `v` zero above row `k`.
fn apply_reflector_left<const N: usize>(r: &mut [[f64; N]; N], k: usize, tau: f64, v: &[f64; N]) {
    let mut j = k;
    while j < N {
        let mut dot = 0.0;
        let mut i = k;
        while i < N {
            dot += v[i] * r[i][j];
            i += 1;
        }
        i = k;
        while i < N {
            r[i][j] -= tau * dot * v[i];
            i += 1;
        }
        j += 1;
    }
}

/// `Q = Q (I - tau v v^T)`. Reflectors accumulate on the right because
/// `A = H_0 H_1 ... H_{n-1} R`.
fn apply_reflector_right<const N: usize>(q: &mut [[f64; N]; N], k: usize, tau: f64, v: &[f64; N]) {
    let mut w = [0.0; N];
    let mut i = 0;
    while i < N {
        let mut acc = 0.0;
        let mut j = k;
        while j < N {
            acc += q[i][j] * v[j];
            j += 1;
        }
        w[i] = acc;
        i += 1;
    }
    i = 0;
    while i < N {
        let mut j = k;
        while j < N {
            q[i][j] -= tau * w[i] * v[j];
            j += 1;
        }
        i += 1;
    }
}

fn matmul<const N: usize>(left: &[[f64; N]; N], right: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    let mut row = 0;
    while row < N {
        let mut col = 0;
        while col < N {
            let mut acc = 0.0;
            let mut k = 0;
            while k < N {
                acc += left[row][k] * right[k][col];
                k += 1;
            }
            out[row][col] = acc;
            col += 1;
        }
        row += 1;
    }
    out
}

/// One Benettin step-sum, matching the body of `lyapunov_spectrum` at
/// `reorth_interval = 1` (`src/dynachaos/diagnostics/lyapunov.py:129`).
fn accumulate_spectrum<const N: usize>(
    mut state: [f64; N],
    n_transient: usize,
    n_iter: usize,
    step: impl Fn([f64; N]) -> [f64; N],
    jac: impl Fn([f64; N]) -> [[f64; N]; N],
) -> [f64; N] {
    for _ in 0..n_transient {
        state = step(state);
    }
    let mut q = identity::<N>();
    let mut log_sums = [0.0; N];
    for _ in 0..n_iter {
        let jacobian = jac(state);
        let product = matmul(&jacobian, &q);
        state = step(state);
        let (q_next, diag) = householder_qr(&product);
        q = q_next;
        let mut i = 0;
        while i < N {
            let singular = if diag[i].abs() > 0.0 {
                diag[i].abs()
            } else {
                SINGULAR_VALUE_FLOOR
            };
            log_sums[i] += singular.ln();
            i += 1;
        }
    }
    let mut spectrum = [0.0; N];
    let mut i = 0;
    while i < N {
        spectrum[i] = log_sums[i] / n_iter as f64;
        i += 1;
    }
    sort_descending(&mut spectrum);
    spectrum
}

/// One wrapped step of the Kaneko circle map.
///
/// ```text
/// theta' = (theta + D + A * sin(2 * pi * theta)) mod 1
/// ```
///
/// Same expression and operation order as `circle_map`
/// (`src/dynachaos/maps/circle_map.py:41`). Python's `% 1.0` is a floored
/// modulo, so this uses `rem_euclid`.
#[inline]
fn circle_map_step(theta: f64, a: f64, d: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (theta + d + a * (two_pi * theta).sin()).rem_euclid(1.0)
}

/// Derivative `1 + 2 * pi * A * cos(2 * pi * theta)`.
///
/// Same expression and operation order as `circle_map_derivative`
/// (`src/dynachaos/maps/circle_map.py:46`). `D` does not appear.
#[inline]
fn circle_map_derivative(theta: f64, a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    1.0 + (two_pi * a) * (two_pi * theta).cos()
}

/// Lyapunov exponent of the circle map at one `(A, D)`.
///
/// Formula, mirroring `lyapunov_exponent`
/// (`src/dynachaos/maps/circle_map.py:75`):
///
/// ```text
/// theta <- (theta + D + A * sin(2 * pi * theta)) mod 1
/// lambda = mean(log|1 + 2 * pi * A * cos(2 * pi * theta)|)
/// ```
///
/// A zero derivative contributes `-100` instead of `log(0)`, the floor at
/// `circle_map.py:85`. The transient is the wrapped map, as
/// `run_transient` + `circle_map` does there — not the unwrapped sum the
/// staircase figure accumulates in `compute`. `theta0` is the initial state.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `n_iter` is zero or `a`, `d`, or
/// `theta0` is not finite.
pub fn circle_map_lyapunov_sum(
    a: f64,
    d: f64,
    n_transient: usize,
    n_iter: usize,
    theta0: f64,
) -> Result<f64, CoreError> {
    require_positive_iter(n_iter)?;
    require_finite("A", a)?;
    require_finite("D", d)?;
    require_finite("theta0", theta0)?;

    let mut theta = theta0;
    for _ in 0..n_transient {
        theta = circle_map_step(theta, a, d);
    }
    let mut log_sum = 0.0;
    for _ in 0..n_iter {
        let deriv = circle_map_derivative(theta, a).abs();
        if deriv > 0.0 {
            log_sum += deriv.ln();
        } else {
            log_sum += ZERO_DERIVATIVE_LOG;
        }
        theta = circle_map_step(theta, a, d);
    }
    Ok(log_sum / n_iter as f64)
}

/// Delayed logistic map, one iteration of `(x, y)`.
///
/// ```text
/// x' = A * x + (1 - A) * (1 - D * y * y)
/// y' = x
/// ```
///
/// Same operation order as `delayed_logistic`
/// (`src/dynachaos/maps/delayed_logistic.py:82`).
#[inline]
fn delayed_logistic_step(state: [f64; 2], a: f64, d: f64) -> [f64; 2] {
    let [x, y] = state;
    let x_new = a * x + (1.0 - a) * (1.0 - d * y * y);
    [x_new, x]
}

/// Jacobian `[[A, -2 (1 - A) D y], [1, 0]]`.
///
/// Same expression as `delayed_logistic_jac`
/// (`src/dynachaos/maps/delayed_logistic.py:90`).
#[inline]
fn delayed_logistic_jac(state: [f64; 2], a: f64, d: f64) -> [[f64; 2]; 2] {
    let [_x, y] = state;
    let dy = -2.0 * (1.0 - a) * d * y;
    [[a, dy], [1.0, 0.0]]
}

/// Delayed-logistic Lyapunov tile: one 2-spectrum per `D`.
///
/// For each `d` in `d_values` the kernel runs [`accumulate_spectrum`] on
/// [`delayed_logistic_step`] / [`delayed_logistic_jac`] from `state0`. The
/// result is flat and D-major, two exponents per `D`, largest first:
/// `out[k * 2 + j]` is exponent `j` at `d_values[k]`.
///
/// Initial state is the caller-supplied `state0`, the same convention as
/// `delayed_logistic_attractor_tile`. The Python sweep at
/// `src/dynachaos/maps/delayed_logistic.py:170` instead rebuilds
/// `fp(D) ± 0.01` with `fp = (sqrt(1 + 4 D) - 1) / (2 D)` for every `D`;
/// pass that vector to reproduce one paper orbit.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `d_values` is empty, `n_iter` is zero,
/// `state0` does not hold exactly two entries, or `a`, any `d`, or a state
/// component is not finite.
pub fn delayed_logistic_lyapunov_tile(
    a: f64,
    d_values: &[f64],
    n_transient: usize,
    n_iter: usize,
    state0: &[f64],
) -> Result<Vec<f64>, CoreError> {
    if d_values.is_empty() {
        return Err(CoreError::invalid_argument("d_values must not be empty"));
    }
    require_positive_iter(n_iter)?;
    if state0.len() != 2 {
        return Err(CoreError::invalid_argument(
            "state0 must hold exactly 2 components",
        ));
    }
    require_finite("A", a)?;
    require_finite_slice("D", d_values)?;
    require_finite_slice("state0", state0)?;

    let start = [state0[0], state0[1]];
    let mut out = Vec::with_capacity(d_values.len() * 2);
    for &d in d_values {
        let spectrum = accumulate_spectrum(
            start,
            n_transient,
            n_iter,
            |state| delayed_logistic_step(state, a, d),
            |state| delayed_logistic_jac(state, a, d),
        );
        out.extend_from_slice(&spectrum);
    }
    Ok(out)
}

/// Kaneko logistic `1 - D u^2` (`src/dynachaos/maps/primitives.py:6`).
#[inline]
fn logistic(u: f64, d: f64) -> f64 {
    1.0 - d * u * u
}

/// Torus-doubling map (I), one iteration of `(X, Y, Z)`.
///
/// ```text
/// X' = A * X + (1 - A) * L_D(Y)
/// Y' = Z
/// Z' = X
/// ```
///
/// with `L_D(u) = 1 - D u^2`. Same operation order as `map_I`
/// (`src/dynachaos/maps/torus_doubling.py:46`).
#[inline]
fn torus_map_i_step(state: [f64; 3], a: f64, d: f64) -> [f64; 3] {
    let [x, y, z] = state;
    let x_new = a * x + (1.0 - a) * logistic(y, d);
    [x_new, z, x]
}

/// Jacobian of map (I) (`src/dynachaos/maps/torus_doubling.py:55`).
#[inline]
fn torus_map_i_jac(state: [f64; 3], a: f64, d: f64) -> [[f64; 3]; 3] {
    let [_x, y, _z] = state;
    let dy = (1.0 - a) * (-2.0 * d * y);
    [[a, dy, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
}

/// Torus-doubling map (IV), one iteration of `(X, Y, Z, W)`.
///
/// ```text
/// X' = A * X + (1 - A) * L_D(Y)
/// Y' = Z
/// Z' = A * Z + (1 - A) * L_D(W)
/// W' = X
/// ```
///
/// Same operation order as `map_IV` (`src/dynachaos/maps/torus_doubling.py:63`).
#[inline]
fn torus_map_iv_step(state: [f64; 4], a: f64, d: f64) -> [f64; 4] {
    let [x, y, z, w] = state;
    let x_new = a * x + (1.0 - a) * logistic(y, d);
    let z_new = a * z + (1.0 - a) * logistic(w, d);
    [x_new, z, z_new, x]
}

/// Jacobian of map (IV) (`src/dynachaos/maps/torus_doubling.py:73`).
#[inline]
fn torus_map_iv_jac(state: [f64; 4], a: f64, d: f64) -> [[f64; 4]; 4] {
    let [_x, y, _z, w] = state;
    let dy = (1.0 - a) * (-2.0 * d * y);
    let dw = (1.0 - a) * (-2.0 * d * w);
    [
        [a, dy, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, a, dw],
        [1.0, 0.0, 0.0, 0.0],
    ]
}

/// Torus-doubling Lyapunov tile: one spectrum per `D`, map I or map IV.
///
/// `map_kind` 1 is map (I), a 3-spectrum; `map_kind` 4 is map (IV), a
/// 4-spectrum. For each `d` the kernel runs [`accumulate_spectrum`] from
/// `state0`. The result is flat and D-major, `dim` exponents per `D`,
/// largest first.
///
/// Initial state is the caller-supplied `state0`. The Python sweeps pass
/// `(0.5, 0.5, 0.5)` for map I (`src/dynachaos/maps/torus_doubling.py:119`)
/// and `(0.5, 0.45, 0.52, 0.48)` for map IV
/// (`src/dynachaos/maps/torus_doubling.py:176`); pass that vector to
/// reproduce a paper orbit.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `map_kind` is not 1 or 4, `d_values`
/// is empty, `n_iter` is zero, `state0` does not match the map dimension,
/// or `a`, any `d`, or a state component is not finite.
pub fn torus_doubling_lyapunov_tile(
    map_kind: u8,
    a: f64,
    d_values: &[f64],
    n_transient: usize,
    n_iter: usize,
    state0: &[f64],
) -> Result<Vec<f64>, CoreError> {
    let dim = match map_kind {
        1 => 3,
        4 => 4,
        _ => {
            return Err(CoreError::invalid_argument(
                "map_kind must be 1 (map I) or 4 (map IV)",
            ));
        }
    };
    if d_values.is_empty() {
        return Err(CoreError::invalid_argument("d_values must not be empty"));
    }
    require_positive_iter(n_iter)?;
    if state0.len() != dim {
        return Err(CoreError::invalid_argument(format!(
            "state0 must hold exactly {dim} components for map_kind {map_kind}"
        )));
    }
    require_finite("A", a)?;
    require_finite_slice("D", d_values)?;
    require_finite_slice("state0", state0)?;

    if map_kind == 1 {
        let start = [state0[0], state0[1], state0[2]];
        let mut out = Vec::with_capacity(d_values.len() * 3);
        for &d in d_values {
            let spectrum = accumulate_spectrum(
                start,
                n_transient,
                n_iter,
                |state| torus_map_i_step(state, a, d),
                |state| torus_map_i_jac(state, a, d),
            );
            out.extend_from_slice(&spectrum);
        }
        Ok(out)
    } else {
        let start = [state0[0], state0[1], state0[2], state0[3]];
        let mut out = Vec::with_capacity(d_values.len() * 4);
        for &d in d_values {
            let spectrum = accumulate_spectrum(
                start,
                n_transient,
                n_iter,
                |state| torus_map_iv_step(state, a, d),
                |state| torus_map_iv_jac(state, a, d),
            );
            out.extend_from_slice(&spectrum);
        }
        Ok(out)
    }
}

/// Jacobian of the coupled delayed map
/// (`src/dynachaos/maps/coupled_delayed.py:67`).
#[inline]
fn coupled_delayed_jac(state: [f64; 4], a: f64, da: f64, db: f64, eps: f64) -> [[f64; 4]; 4] {
    let [_x, y, _z, w] = state;
    [
        [a, da * (1.0 - 2.0 * y), eps, -eps],
        [1.0, 0.0, 0.0, 0.0],
        [-eps, eps, a, db * (1.0 - 2.0 * w)],
        [0.0, 0.0, 1.0, 0.0],
    ]
}

/// Coupled-delayed Lyapunov tile: one 4-spectrum per `D_B`.
///
/// `D_A = D_B + 0.1` for every entry, as `compute_lyapunov` fixes it
/// (`src/dynachaos/maps/coupled_delayed.py:100`). For each `db` the kernel
/// runs [`accumulate_spectrum`] on [`coupled_delayed_step`] /
/// [`coupled_delayed_jac`] from `state0`. The result is flat and D-major,
/// four exponents per `D_B`, largest first.
///
/// Initial state is the caller-supplied `state0`. The Python sweep passes
/// `(0.5, 0.5, 0.3, 0.3)` (`src/dynachaos/maps/coupled_delayed.py:101`);
/// pass that vector to reproduce a paper orbit.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `db_values` is empty, `n_iter` is
/// zero, `state0` does not hold exactly four entries, or `a`, `eps`, any
/// `db`, or a state component is not finite.
pub fn coupled_delayed_lyapunov_tile(
    a: f64,
    db_values: &[f64],
    eps: f64,
    n_transient: usize,
    n_iter: usize,
    state0: &[f64],
) -> Result<Vec<f64>, CoreError> {
    if db_values.is_empty() {
        return Err(CoreError::invalid_argument("db_values must not be empty"));
    }
    require_positive_iter(n_iter)?;
    if state0.len() != 4 {
        return Err(CoreError::invalid_argument(
            "state0 must hold exactly 4 components",
        ));
    }
    require_finite("A", a)?;
    require_finite("eps", eps)?;
    require_finite_slice("DB", db_values)?;
    require_finite_slice("state0", state0)?;

    let start = [state0[0], state0[1], state0[2], state0[3]];
    let mut out = Vec::with_capacity(db_values.len() * 4);
    for &db in db_values {
        let da = db + COUPLED_DA_OFFSET;
        let spectrum = accumulate_spectrum(
            start,
            n_transient,
            n_iter,
            |state| coupled_delayed_step(state, a, da, db, eps),
            |state| coupled_delayed_jac(state, a, da, db, eps),
        );
        out.extend_from_slice(&spectrum);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: &[f64], expected: &[f64]) {
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected) {
            assert!(
                (g - e).abs() <= 1e-12,
                "got {g:.16e}, expected {e:.16e}, diff {:.3e}",
                (g - e).abs()
            );
        }
    }

    #[test]
    fn circle_sum_matches_python_literal() {
        // lyapunov_exponent(0.1, 0.25, n_transient=5, n_iter=8, theta0=0.1)
        let got = circle_map_lyapunov_sum(0.1, 0.25, 5, 8, 0.1).unwrap();
        assert_close(&[got], &[-0.019117325131734163]);
    }

    #[test]
    fn circle_zero_derivative_uses_the_python_floor() {
        // A = 1/(2 pi), theta0 = 0.5 makes the derivative exactly 0.
        let a = 1.0 / (2.0 * std::f64::consts::PI);
        let got = circle_map_lyapunov_sum(a, 0.0, 0, 1, 0.5).unwrap();
        assert_eq!(got, -100.0);
    }

    #[test]
    fn delayed_spectrum_matches_python_literal() {
        // lyapunov_spectrum at A=0.3, D=1.55, n_transient=4, n_iter=6,
        // state0=(0.4, 0.35).
        let got = delayed_logistic_lyapunov_tile(0.3, &[1.55], 4, 6, &[0.4, 0.35]).unwrap();
        assert_close(&got, &[-0.03429366148235978, -0.1129286522696946]);
    }

    #[test]
    fn torus_spectra_match_python_literals() {
        let map_i = torus_doubling_lyapunov_tile(1, 0.4, &[2.19], 3, 5, &[0.5, 0.5, 0.5]).unwrap();
        assert_close(
            &map_i,
            &[
                0.09509416374314716,
                0.08776636581766876,
                0.046244624248652014,
            ],
        );
        let map_iv =
            torus_doubling_lyapunov_tile(4, 0.3, &[1.5212], 3, 5, &[0.5, 0.45, 0.52, 0.48])
                .unwrap();
        assert_close(
            &map_iv,
            &[
                0.11904624608858176,
                0.08329115757085308,
                0.07586257818925221,
                0.04304271726480864,
            ],
        );
    }

    #[test]
    fn coupled_spectrum_matches_python_literal() {
        // DB=2.3 so DA=2.4, eps=5e-3, n_transient=2, n_iter=4,
        // state0=(0.5, 0.5, 0.3, 0.3).
        let got =
            coupled_delayed_lyapunov_tile(0.4, &[2.3], 5e-3, 2, 4, &[0.5, 0.5, 0.3, 0.3]).unwrap();
        assert_close(
            &got,
            &[
                0.08813072555802139,
                0.05564281443940726,
                -0.04791923276835684,
                -0.30555516798505555,
            ],
        );
    }

    #[test]
    fn tile_is_d_major() {
        let out = delayed_logistic_lyapunov_tile(0.3, &[1.55, 1.9], 4, 6, &[0.4, 0.35]).unwrap();
        assert_eq!(out.len(), 4);
        let first = delayed_logistic_lyapunov_tile(0.3, &[1.55], 4, 6, &[0.4, 0.35]).unwrap();
        assert_eq!(&out[..2], first.as_slice());
    }

    #[test]
    fn invalid_input_is_rejected() {
        assert!(circle_map_lyapunov_sum(0.1, 0.25, 0, 0, 0.1).is_err());
        assert!(circle_map_lyapunov_sum(f64::NAN, 0.25, 0, 1, 0.1).is_err());
        assert!(circle_map_lyapunov_sum(0.1, f64::INFINITY, 0, 1, 0.1).is_err());
        assert!(circle_map_lyapunov_sum(0.1, 0.25, 0, 1, f64::NAN).is_err());

        assert!(delayed_logistic_lyapunov_tile(0.3, &[], 0, 1, &[0.4, 0.35]).is_err());
        assert!(delayed_logistic_lyapunov_tile(0.3, &[1.55], 0, 0, &[0.4, 0.35]).is_err());
        assert!(delayed_logistic_lyapunov_tile(0.3, &[1.55], 0, 1, &[0.4]).is_err());
        assert!(delayed_logistic_lyapunov_tile(f64::NAN, &[1.55], 0, 1, &[0.4, 0.35]).is_err());
        assert!(delayed_logistic_lyapunov_tile(0.3, &[f64::INFINITY], 0, 1, &[0.4, 0.35]).is_err());
        assert!(delayed_logistic_lyapunov_tile(0.3, &[1.55], 0, 1, &[0.4, f64::NAN]).is_err());

        assert!(torus_doubling_lyapunov_tile(2, 0.4, &[2.19], 0, 1, &[0.5; 3]).is_err());
        assert!(torus_doubling_lyapunov_tile(1, 0.4, &[], 0, 1, &[0.5; 3]).is_err());
        assert!(torus_doubling_lyapunov_tile(1, 0.4, &[2.19], 0, 0, &[0.5; 3]).is_err());
        assert!(torus_doubling_lyapunov_tile(1, 0.4, &[2.19], 0, 1, &[0.5; 4]).is_err());
        assert!(torus_doubling_lyapunov_tile(4, 0.3, &[1.5], 0, 1, &[0.5; 3]).is_err());
        assert!(torus_doubling_lyapunov_tile(4, f64::NAN, &[1.5], 0, 1, &[0.5; 4]).is_err());

        assert!(coupled_delayed_lyapunov_tile(0.4, &[], 0.005, 0, 1, &[0.5; 4]).is_err());
        assert!(coupled_delayed_lyapunov_tile(0.4, &[2.3], 0.005, 0, 0, &[0.5; 4]).is_err());
        assert!(coupled_delayed_lyapunov_tile(0.4, &[2.3], 0.005, 0, 1, &[0.5; 3]).is_err());
        assert!(coupled_delayed_lyapunov_tile(0.4, &[2.3], f64::NAN, 0, 1, &[0.5; 4]).is_err());
        assert!(coupled_delayed_lyapunov_tile(f64::NAN, &[2.3], 0.005, 0, 1, &[0.5; 4]).is_err());
    }
}
