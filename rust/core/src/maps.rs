//! Low-dimensional map-iteration kernels: delayed logistic, torus doubling,
//! modulated circle, and the coupled-delayed `(x, z)` projection.
//!
//! Each kernel iterates a map past a transient and records the states that
//! follow, mirroring `trajectory_after_transient` in
//! `src/dynachaos/maps/_iter.py`. The map bodies below evaluate the same
//! expression in the same operation order as the Python line they mirror, so a
//! trajectory is bit-identical to the Python reference. The logistic primitive
//! is the Kaneko form `1 - a * x * x` (`src/dynachaos/maps/primitives.py:6`),
//! not `r * x * (1 - x)`.
//!
//! Divergence follows the Python rule where Python has one: a state with any
//! `|component| > 1e10` stops the computation. The delayed logistic kernel drops
//! the whole block for that parameter value (`allow_partial = false`, as
//! `src/dynachaos/maps/delayed_logistic.py:113` requests); the torus kernels
//! keep the samples recorded so far (`allow_partial = true`,
//! `src/dynachaos/maps/torus_doubling.py:99`). The coupled-delayed projection
//! has no such check in Python; see [`coupled_delayed_projection_tile`].

use crate::CoreError;

/// Any state component beyond this magnitude counts as diverged.
///
/// This is the threshold `diverged_fn` uses in both Python modules
/// (`np.any(np.abs(state) > 1e10)`).
const DIVERGENCE: f64 = 1e10;

/// Run `n_transient` steps, then record the next `n_plot` states.
///
/// Mirrors `trajectory_after_transient` (`src/dynachaos/maps/_iter.py:72`):
/// the transient is iterated with the divergence check after every step, and
/// a diverged transient returns `None`. During the record phase a diverged
/// state returns `None` when `allow_partial` is false, or the `i` samples
/// recorded so far when it is true. The recorded samples are the states
/// *after* each step, exactly as `sample_trajectory` fills `traj[i]`.
fn transient_then_record<const DIM: usize>(
    mut state: [f64; DIM],
    n_transient: usize,
    n_plot: usize,
    allow_partial: bool,
    step: impl Fn([f64; DIM]) -> [f64; DIM],
) -> Option<Vec<f64>> {
    for _ in 0..n_transient {
        state = step(state);
        if diverged(&state) {
            return None;
        }
    }
    let mut out = Vec::with_capacity(n_plot * DIM);
    for _ in 0..n_plot {
        state = step(state);
        if diverged(&state) {
            return if allow_partial { Some(out) } else { None };
        }
        out.extend_from_slice(&state);
    }
    Some(out)
}

#[inline]
fn diverged<const DIM: usize>(state: &[f64; DIM]) -> bool {
    state.iter().any(|v| v.abs() > DIVERGENCE)
}

/// Kaneko logistic primitive `f(u) = 1 - a * u * u`.
///
/// Same expression and operation order as `logistic` in
/// `src/dynachaos/maps/primitives.py:6`.
#[inline]
fn logistic(u: f64, a: f64) -> f64 {
    1.0 - a * u * u
}

/// Delayed logistic map, one iteration of `state = (x, y) -> (x', y')`.
///
/// Formula (parameters named as in the paper):
///
/// ```text
/// x' = A * x + (1 - A) * (1 - D * y * y)
/// y' = x
/// ```
///
/// Mirrors `delayed_logistic` (`src/dynachaos/maps/delayed_logistic.py:82`)
/// with the same operation order, so the orbit is bit-identical.
#[inline]
fn delayed_logistic_step(state: [f64; 2], a: f64, d: f64) -> [f64; 2] {
    let [x, y] = state;
    let x_new = a * x + (1.0 - a) * (1.0 - d * y * y);
    let y_new = x;
    [x_new, y_new]
}

/// Torus-doubling map (I), one iteration of `state = (X, Y, Z)`.
///
/// Formula, with `L_D(u) = 1 - D * u * u` the Kaneko logistic:
///
/// ```text
/// X' = A * X + (1 - A) * L_D(Y)
/// Y' = Z
/// Z' = X
/// ```
///
/// Mirrors `map_I` (`src/dynachaos/maps/torus_doubling.py:46`) with the same
/// operation order, so the orbit is bit-identical.
#[inline]
fn torus_map_i_step(state: [f64; 3], a: f64, d: f64) -> [f64; 3] {
    let [x, y, z] = state;
    let x_new = a * x + (1.0 - a) * logistic(y, d);
    let y_new = z;
    let z_new = x;
    [x_new, y_new, z_new]
}

/// Torus-doubling map (IV), one iteration of `state = (X, Y, Z, W)`.
///
/// Formula, with `L_D(u) = 1 - D * u * u` the Kaneko logistic:
///
/// ```text
/// X' = A * X + (1 - A) * L_D(Y)
/// Y' = Z
/// Z' = A * Z + (1 - A) * L_D(W)
/// W' = X
/// ```
///
/// Mirrors `map_IV` (`src/dynachaos/maps/torus_doubling.py:63`) with the same
/// operation order, so the orbit is bit-identical.
#[inline]
fn torus_map_iv_step(state: [f64; 4], a: f64, d: f64) -> [f64; 4] {
    let [x, y, z, w] = state;
    let x_new = a * x + (1.0 - a) * logistic(y, d);
    let y_new = z;
    let z_new = a * z + (1.0 - a) * logistic(w, d);
    let w_new = x;
    [x_new, y_new, z_new, w_new]
}

/// Delayed-logistic attractor tile: one `(x, y)` trajectory per `D` value.
///
/// For each `d` in `d_values` the kernel starts from `state0 = (x0, y0)`,
/// iterates `n_transient` steps of [`delayed_logistic_step`], then records
/// `n_plot` states. The result is flat, D-major: `d_values.len()` blocks of
/// `n_plot` `(x, y)` pairs each, so `out[k * n_plot * 2 + i * 2 + j]` is
/// coordinate `j` of sample `i` at `d_values[k]`.
///
/// A `d` whose orbit diverges (any `|state| > 1e10`, during the transient or
/// the record) yields a block of `NaN` pairs. This mirrors
/// `compute_attractor` returning `None` for that `D`
/// (`src/dynachaos/maps/delayed_logistic.py:101`): the tile keeps a fixed
/// layout and marks the hole instead of shifting the blocks.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `d_values` is empty, `n_plot` is zero,
/// `state0` does not hold exactly two entries, or `a`, any `d`, or a state
/// component is not finite.
pub fn delayed_logistic_attractor_tile(
    a: f64,
    d_values: &[f64],
    n_transient: usize,
    n_plot: usize,
    state0: &[f64],
) -> Result<Vec<f64>, CoreError> {
    if d_values.is_empty() {
        return Err(CoreError::invalid_argument("d_values must not be empty"));
    }
    if n_plot == 0 {
        return Err(CoreError::invalid_argument("n_plot must be at least 1"));
    }
    if state0.len() != 2 {
        return Err(CoreError::invalid_argument(
            "state0 must hold exactly 2 components",
        ));
    }
    if !a.is_finite() {
        return Err(CoreError::invalid_argument("A must be finite"));
    }
    if d_values.iter().any(|d| !d.is_finite()) {
        return Err(CoreError::invalid_argument("every D must be finite"));
    }
    if state0.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument("state0 must be finite"));
    }

    let start = [state0[0], state0[1]];
    let block = n_plot * 2;
    let mut out = Vec::with_capacity(d_values.len() * block);
    for &d in d_values {
        match transient_then_record(start, n_transient, n_plot, false, |s| {
            delayed_logistic_step(s, a, d)
        }) {
            Some(samples) => out.extend_from_slice(&samples),
            None => out.extend(std::iter::repeat_n(f64::NAN, block)),
        }
    }
    Ok(out)
}

/// Torus-doubling attractor tile: one trajectory of map I or map IV.
///
/// `map_kind` selects the map: `1` is map (I), a 3D state `(X, Y, Z)` stepped
/// by [`torus_map_i_step`]; `4` is map (IV), a 4D state `(X, Y, Z, W)` stepped
/// by [`torus_map_iv_step`]. `state0` must hold exactly that many components.
/// The kernel iterates `n_transient` steps, then records states until
/// `n_plot` samples are written or the orbit diverges.
///
/// The result is flat: `n_produced * dim` values, `dim` per sample, in
/// iteration order. Divergence (any `|state| > 1e10`) stops the record and
/// returns the samples produced so far — possibly zero when the transient
/// itself diverged — exactly as `iterate_map` does with
/// `allow_partial = true` (`src/dynachaos/maps/torus_doubling.py:91`).
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `map_kind` is not 1 or 4, `n_plot` is
/// zero, `state0` does not match the map dimension, or `a`, `d`, or a state
/// component is not finite.
pub fn torus_doubling_attractor_tile(
    map_kind: u8,
    a: f64,
    d: f64,
    n_transient: usize,
    n_plot: usize,
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
    if n_plot == 0 {
        return Err(CoreError::invalid_argument("n_plot must be at least 1"));
    }
    if state0.len() != dim {
        return Err(CoreError::invalid_argument(format!(
            "state0 must hold exactly {dim} components for map_kind {map_kind}"
        )));
    }
    if !a.is_finite() || !d.is_finite() {
        return Err(CoreError::invalid_argument("A and D must be finite"));
    }
    if state0.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument("state0 must be finite"));
    }

    let samples = match map_kind {
        1 => {
            let start = [state0[0], state0[1], state0[2]];
            transient_then_record(start, n_transient, n_plot, true, |s| {
                torus_map_i_step(s, a, d)
            })
        }
        _ => {
            let start = [state0[0], state0[1], state0[2], state0[3]];
            transient_then_record(start, n_transient, n_plot, true, |s| {
                torus_map_iv_step(s, a, d)
            })
        }
    };
    // `allow_partial` makes the record phase infallible; `None` here means the
    // transient diverged, which the Python caller reports as no samples.
    Ok(samples.unwrap_or_default())
}

/// Modulated-circle rotation numbers: one `(rho_theta, rho_phi)` pair per `D`.
///
/// The map is Kaneko's modulated circle (Eq. 3.1):
///
/// ```text
/// theta' = theta + A * sin(2 * pi * theta) + D + eps * sin(2 * pi * phi)
/// phi'   = phi + C
/// ```
///
/// For each `d` in `d_values` the kernel starts from `state0 = (theta0, phi0)`
/// and accumulates the *unwrapped* increments — `theta` and `phi` are never
/// reduced modulo 1 — exactly as `rotation_numbers` does
/// (`src/dynachaos/maps/modulated_circle.py:48`, using `iterate_unwrapped` in
/// `src/dynachaos/maps/_iter.py:11`). After `n_transient` steps it records the
/// state, iterates `n_iter` more, and returns the mean increments
/// `(theta_end - theta_start) / n_iter` and `(phi_end - phi_start) / n_iter`.
/// The increment is evaluated in the same operation order as the Python line,
/// so the pair is bit-identical to the Python reference.
///
/// The result is flat, D-major: `d_values.len()` pairs, so
/// `out[k * 2]` is `rho_theta` and `out[k * 2 + 1]` is `rho_phi` at
/// `d_values[k]`.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `d_values` is empty, `n_iter` is zero,
/// `state0` does not hold exactly two entries, or `a`, `c`, `eps`, any `d`,
/// or a state component is not finite.
pub fn modulated_circle_rotation_tile(
    a: f64,
    c: f64,
    d_values: &[f64],
    eps: f64,
    n_transient: usize,
    n_iter: usize,
    state0: &[f64],
) -> Result<Vec<f64>, CoreError> {
    if d_values.is_empty() {
        return Err(CoreError::invalid_argument("d_values must not be empty"));
    }
    if n_iter == 0 {
        return Err(CoreError::invalid_argument("n_iter must be at least 1"));
    }
    if state0.len() != 2 {
        return Err(CoreError::invalid_argument(
            "state0 must hold exactly 2 components (theta0, phi0)",
        ));
    }
    if !a.is_finite() || !c.is_finite() || !eps.is_finite() {
        return Err(CoreError::invalid_argument("A, C and eps must be finite"));
    }
    if d_values.iter().any(|d| !d.is_finite()) {
        return Err(CoreError::invalid_argument("every D must be finite"));
    }
    if state0.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument("state0 must be finite"));
    }

    let increment = |state: [f64; 2], d: f64| -> [f64; 2] {
        let [theta, phi] = state;
        let dtheta = a * (2.0 * std::f64::consts::PI * theta).sin()
            + d
            + eps * (2.0 * std::f64::consts::PI * phi).sin();
        [dtheta, c]
    };

    let mut out = Vec::with_capacity(d_values.len() * 2);
    for &d in d_values {
        let mut state = [state0[0], state0[1]];
        for _ in 0..n_transient {
            let step = increment(state, d);
            state[0] += step[0];
            state[1] += step[1];
        }
        let start = state;
        for _ in 0..n_iter {
            let step = increment(state, d);
            state[0] += step[0];
            state[1] += step[1];
        }
        out.push((state[0] - start[0]) / n_iter as f64);
        out.push((state[1] - start[1]) / n_iter as f64);
    }
    Ok(out)
}

/// Paper offset `D_A = D_B + 0.1` (`src/dynachaos/maps/coupled_delayed.py:135`).
pub(crate) const COUPLED_DA_OFFSET: f64 = 0.1;

/// One iteration of the 4D coupled delayed logistic map.
///
/// ```text
/// x' = A x + D_A y (1 - y) + eps (z - w)
/// y' = x
/// z' = A z + D_B w (1 - w) + eps (y - x)
/// w' = z
/// ```
///
/// Same operation order as `coupled_delayed`
/// (`src/dynachaos/maps/coupled_delayed.py:52`). No product uses `mul_add`.
#[inline]
pub(crate) fn coupled_delayed_step(
    state: [f64; 4],
    a: f64,
    da: f64,
    db: f64,
    eps: f64,
) -> [f64; 4] {
    let [x, y, z, w] = state;
    let h1 = z - w;
    let h2 = y - x;
    let x_new = a * x + da * y * (1.0 - y) + eps * h1;
    let z_new = a * z + db * w * (1.0 - w) + eps * h2;
    [x_new, x, z_new, z]
}

/// Coupled-delayed projection tile: one `(x, z)` trajectory per `D_B`.
///
/// Mirrors `compute_projections` (`src/dynachaos/maps/coupled_delayed.py:123`).
/// `D_A = D_B + 0.1` for every entry. For each `db` the kernel iterates
/// [`coupled_delayed_step`] `n_transient` times from `state0`, then records
/// `n_plot` projected `(x, z)` pairs — coordinates 0 and 2 of the post-step
/// state, the same selection as `project_fn=lambda state: state[[0, 2]]`.
///
/// Python passes no `diverged_fn`, so it has no divergence check. This kernel
/// does the same: it records every requested sample, including non-finite
/// values, and does not apply a magnitude gate. (The Lyapunov tile in
/// `lyapunov.rs` is a different kernel and is not reused here.)
///
/// The result is flat and D-major: `db_values.len()` blocks of `n_plot`
/// `(x, z)` pairs. The paper sweep starts from `(0.5, 0.5, 0.3, 0.3)` at
/// `A = 0.4`, `eps = 5e-3`.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `db_values` is empty, `n_plot` is zero,
/// `state0` does not hold exactly four entries, or `A`, `eps`, a `D_B`, or a
/// state component is not finite.
pub fn coupled_delayed_projection_tile(
    a: f64,
    db_values: &[f64],
    eps: f64,
    n_transient: usize,
    n_plot: usize,
    state0: &[f64],
) -> Result<Vec<f64>, CoreError> {
    if db_values.is_empty() {
        return Err(CoreError::invalid_argument("db_values must not be empty"));
    }
    if n_plot == 0 {
        return Err(CoreError::invalid_argument("n_plot must be at least 1"));
    }
    if state0.len() != 4 {
        return Err(CoreError::invalid_argument(
            "state0 must hold exactly 4 components",
        ));
    }
    if !a.is_finite() || !eps.is_finite() {
        return Err(CoreError::invalid_argument("A and eps must be finite"));
    }
    if db_values.iter().any(|db| !db.is_finite()) {
        return Err(CoreError::invalid_argument("every D_B must be finite"));
    }
    if state0.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument("state0 must be finite"));
    }

    let start = [state0[0], state0[1], state0[2], state0[3]];
    let mut out = Vec::with_capacity(db_values.len() * n_plot * 2);
    for &db in db_values {
        let da = db + COUPLED_DA_OFFSET;
        let mut state = start;
        for _ in 0..n_transient {
            state = coupled_delayed_step(state, a, da, db, eps);
        }
        for _ in 0..n_plot {
            state = coupled_delayed_step(state, a, da, db, eps);
            out.push(state[0]);
            out.push(state[2]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values produced once from the Python modules
    // (`trajectory_after_transient` / `iterate_map`) at small N and pasted as
    // literals; the tests do not call Python. The Rust maps evaluate the same
    // expression in the same order, so equality is exact.

    /// First eight (x, y) samples of the delayed logistic map at A = 0.3,
    /// D = 1.55 from state0 = (0.4, 0.35) after a 20-step transient.
    const DELAYED_D_1_55: [f64; 16] = [
        0.15399350646670135,
        0.7210566987713561,
        0.18208185425530526,
        0.15399350646670135,
        0.7288948662397992,
        0.18208185425530526,
        0.8826965850827202,
        0.7288948662397992,
        0.3883617927814688,
        0.8826965850827202,
        -0.028872750694174484,
        0.3883617927814688,
        0.527693177721454,
        -0.028872750694174484,
        0.8574034585465131,
        0.527693177721454,
    ];

    /// First eight samples of map (I) at A = 0.4, D = 2.19 from
    /// state0 = (0.5, 0.5, 0.5) after a 20-step transient.
    const MAP_I_D_2_19: [f64; 24] = [
        0.38908632913180996,
        0.7392388509439576,
        0.6896311418057928,
        0.037567592181869336,
        0.6896311418057928,
        0.38908632913180996,
        -0.009899683964599237,
        0.38908632913180996,
        0.037567592181869336,
        0.39711606904047125,
        0.037567592181869336,
        -0.009899683964599237,
        0.7569919479033894,
        -0.009899683964599237,
        0.39711606904047125,
        0.9026680022435807,
        0.39711606904047125,
        0.7569919479033894,
        0.7538478605081669,
        0.7569919479033894,
        0.9026680022435807,
        0.14856877692686069,
        0.9026680022435807,
        0.7538478605081669,
    ];

    /// First eight samples of map (IV) at A = 0.3, D = 1.5212 from
    /// state0 = (0.5, 0.45, 0.52, 0.48) after a 20-step transient.
    const MAP_IV_D_1_5212: [f64; 32] = [
        0.30109345123122205,
        0.48089910059388197,
        0.06389495741852494,
        0.6592355241657535,
        0.544068936226674,
        0.06389495741852494,
        0.25639809957873705,
        0.30109345123122205,
        0.8588734017320518,
        0.25639809957873705,
        0.6803839463475813,
        0.544068936226674,
        0.8876594543943093,
        0.6803839463475813,
        0.5889108228197848,
        0.8588734017320518,
        0.47335963900200795,
        0.5889108228197848,
        0.09117962399320012,
        0.8876594543943093,
        0.47270438779925594,
        0.09117962399320012,
        -0.11167540444194093,
        0.47335963900200795,
        0.8329585306549983,
        -0.11167540444194093,
        0.42789937431761293,
        0.47270438779925594,
        0.936607517925359,
        0.42789937431761293,
        0.5904319124748301,
        0.8329585306549983,
    ];

    #[test]
    fn delayed_logistic_tile_matches_python_exactly() {
        let out = delayed_logistic_attractor_tile(0.3, &[1.55], 20, 8, &[0.4, 0.35]).unwrap();
        assert_eq!(out, DELAYED_D_1_55);
    }

    #[test]
    fn delayed_logistic_tile_is_d_major() {
        // Two D values: the second block must be the D = 2.16 orbit, whose
        // first pair Python computes as (0.38245579820297315, 0.5329508278461266).
        let out = delayed_logistic_attractor_tile(0.3, &[1.55, 2.16], 20, 8, &[0.4, 0.35]).unwrap();
        assert_eq!(out.len(), 2 * 8 * 2);
        assert_eq!(&out[..16], &DELAYED_D_1_55);
        assert_eq!(out[16], 0.38245579820297315);
        assert_eq!(out[17], 0.5329508278461266);
    }

    #[test]
    fn delayed_logistic_divergence_marks_one_block_nan() {
        // Python: D = 3.0 escapes at step 46, so compute_attractor is None.
        let out =
            delayed_logistic_attractor_tile(0.3, &[1.55, 3.0], 2000, 8, &[0.4, 0.35]).unwrap();
        assert_eq!(out.len(), 2 * 8 * 2);
        assert!(out[..16].iter().all(|v| v.is_finite()));
        assert!(out[16..].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn delayed_logistic_rejects_invalid_input() {
        assert!(delayed_logistic_attractor_tile(0.3, &[], 20, 8, &[0.4, 0.35]).is_err());
        assert!(delayed_logistic_attractor_tile(0.3, &[1.55], 20, 0, &[0.4, 0.35]).is_err());
        assert!(delayed_logistic_attractor_tile(0.3, &[1.55], 20, 8, &[0.4]).is_err());
        assert!(delayed_logistic_attractor_tile(f64::NAN, &[1.55], 20, 8, &[0.4, 0.35]).is_err());
        assert!(
            delayed_logistic_attractor_tile(0.3, &[f64::INFINITY], 20, 8, &[0.4, 0.35]).is_err()
        );
        assert!(delayed_logistic_attractor_tile(0.3, &[1.55], 20, 8, &[0.4, f64::NAN]).is_err());
    }

    #[test]
    fn torus_map_i_tile_matches_python_exactly() {
        let out = torus_doubling_attractor_tile(1, 0.4, 2.19, 20, 8, &[0.5, 0.5, 0.5]).unwrap();
        assert_eq!(out, MAP_I_D_2_19);
    }

    #[test]
    fn torus_map_iv_tile_matches_python_exactly() {
        let out =
            torus_doubling_attractor_tile(4, 0.3, 1.5212, 20, 8, &[0.5, 0.45, 0.52, 0.48]).unwrap();
        assert_eq!(out, MAP_IV_D_1_5212);
    }

    #[test]
    fn torus_record_divergence_returns_partial_samples() {
        // Python: map I at D = 2.8 escapes 281 steps into the record after a
        // 500-step transient, so iterate_map returns a (281, 3) trajectory.
        let out = torus_doubling_attractor_tile(1, 0.4, 2.8, 500, 4096, &[0.5, 0.5, 0.5]).unwrap();
        assert_eq!(out.len(), 281 * 3);
        assert!(out.iter().all(|v| v.is_finite()));

        // Map IV at D = 2.12 escapes 1212 steps into the record.
        let out = torus_doubling_attractor_tile(4, 0.3, 2.12, 2000, 4096, &[0.5, 0.45, 0.52, 0.48])
            .unwrap();
        assert_eq!(out.len(), 1212 * 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn torus_transient_divergence_returns_no_samples() {
        // Python: map I at D = 2.8 escapes at step 782, inside a 2000-step
        // transient, so iterate_map returns None.
        let out = torus_doubling_attractor_tile(1, 0.4, 2.8, 2000, 4096, &[0.5, 0.5, 0.5]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn torus_rejects_invalid_input() {
        assert!(torus_doubling_attractor_tile(2, 0.4, 2.19, 20, 8, &[0.5; 3]).is_err());
        assert!(torus_doubling_attractor_tile(1, 0.4, 2.19, 20, 0, &[0.5; 3]).is_err());
        assert!(torus_doubling_attractor_tile(1, 0.4, 2.19, 20, 8, &[0.5; 4]).is_err());
        assert!(torus_doubling_attractor_tile(4, 0.3, 1.5, 20, 8, &[0.5; 3]).is_err());
        assert!(torus_doubling_attractor_tile(1, f64::NAN, 2.19, 20, 8, &[0.5; 3]).is_err());
        assert!(torus_doubling_attractor_tile(1, 0.4, f64::INFINITY, 20, 8, &[0.5; 3]).is_err());
        assert!(
            torus_doubling_attractor_tile(4, 0.3, 1.5, 20, 8, &[0.5, 0.5, f64::NAN, 0.5]).is_err()
        );
    }

    #[test]
    fn modulated_circle_matches_python_exactly() {
        // Expected values produced once from `rotation_numbers`
        // (`src/dynachaos/maps/modulated_circle.py:48`) at A = 0.1,
        // C = (sqrt(5) - 1) / 2, eps = 0.05, n_transient = 50, n_iter = 200,
        // state0 = (0.1, 0.1); the test does not call Python. The Rust kernel
        // evaluates the same increment in the same order, so equality is
        // exact — including rho_phi, which is C plus accumulated rounding
        // drift, not C itself.
        let c = (5.0_f64.sqrt() - 1.0) / 2.0;
        let out = modulated_circle_rotation_tile(0.1, c, &[0.3, 0.7], 0.05, 50, 200, &[0.1, 0.1])
            .unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 0.2890118498900553);
        assert_eq!(out[1], 0.6180339887498952);
        assert_eq!(out[2], 0.7107426428675732);
        assert_eq!(out[3], 0.6180339887498952);
    }

    #[test]
    fn modulated_circle_keeps_phases_unwrapped() {
        // With n_transient = 0 and n_iter = 1, rho_theta is the increment
        // evaluated at state0 itself: A * sin(2 pi theta0) + D + eps * sin(2 pi
        // phi0). A kernel that reduced the phases modulo 1 would still pass
        // this check; the exact-match test above is the one that pins the
        // unwrapped accumulation over hundreds of steps.
        let c = (5.0_f64.sqrt() - 1.0) / 2.0;
        let out = modulated_circle_rotation_tile(0.1, c, &[0.3], 0.05, 0, 1, &[0.1, 0.1]).unwrap();
        let expected = 0.1 * (2.0 * std::f64::consts::PI * 0.1).sin()
            + 0.3
            + 0.05 * (2.0 * std::f64::consts::PI * 0.1).sin();
        assert_eq!(out[0], expected);
        assert_eq!(out[1], c);
    }

    #[test]
    fn modulated_circle_rejects_invalid_input() {
        let c = (5.0_f64.sqrt() - 1.0) / 2.0;
        assert!(modulated_circle_rotation_tile(0.1, c, &[], 0.05, 50, 200, &[0.1, 0.1]).is_err());
        assert!(modulated_circle_rotation_tile(0.1, c, &[0.3], 0.05, 50, 0, &[0.1, 0.1]).is_err());
        assert!(modulated_circle_rotation_tile(0.1, c, &[0.3], 0.05, 50, 200, &[0.1]).is_err());
        assert!(
            modulated_circle_rotation_tile(f64::NAN, c, &[0.3], 0.05, 50, 200, &[0.1, 0.1])
                .is_err()
        );
        assert!(
            modulated_circle_rotation_tile(0.1, c, &[0.3], f64::INFINITY, 50, 200, &[0.1, 0.1])
                .is_err()
        );
        assert!(
            modulated_circle_rotation_tile(0.1, c, &[f64::NAN], 0.05, 50, 200, &[0.1, 0.1])
                .is_err()
        );
        assert!(
            modulated_circle_rotation_tile(0.1, c, &[0.3], 0.05, 50, 200, &[0.1, f64::NAN])
                .is_err()
        );
    }

    /// First eight `(x, z)` samples at A = 0.4, D_B = 2.37, eps = 5e-3 from
    /// (0.5, 0.5, 0.3, 0.3) after a 20-step transient. `PROJECTION_CASES` panel 0.
    const PROJECTION_DB_2_37: [f64; 16] = [
        0.9079264445850574,
        0.8452492484777323,
        0.6563897866332581,
        0.8854591014629967,
        0.4692391229644689,
        0.6654443306035374,
        0.7436848948843963,
        0.5074819217651929,
        0.911846953966259,
        0.729249310973097,
        0.8366732683853395,
        0.8832262435005075,
        0.5339829456932967,
        0.8216104314296215,
        0.5508133417931551,
        0.5745938458985095,
    ];

    #[test]
    fn projection_tile_matches_python_exactly() {
        let out = coupled_delayed_projection_tile(0.4, &[2.37], 5e-3, 20, 8, &[0.5, 0.5, 0.3, 0.3])
            .unwrap();
        assert_eq!(out, PROJECTION_DB_2_37);
    }

    #[test]
    fn projection_tile_is_d_major_and_records_x_z() {
        let out =
            coupled_delayed_projection_tile(0.4, &[2.37, 2.43], 5e-3, 20, 8, &[0.5, 0.5, 0.3, 0.3])
                .unwrap();
        assert_eq!(out.len(), 2 * 8 * 2);
        assert_eq!(&out[..16], &PROJECTION_DB_2_37);
        // A swap of (x, z) would put z in the even slot. The first sample's x
        // is the value Python records in column 0.
        assert_eq!(out[0], 0.9079264445850574);
        assert_eq!(out[1], 0.8452492484777323);
    }

    #[test]
    fn projection_rejects_invalid_input() {
        assert!(
            coupled_delayed_projection_tile(0.4, &[], 5e-3, 1, 1, &[0.5, 0.5, 0.3, 0.3]).is_err()
        );
        assert!(
            coupled_delayed_projection_tile(0.4, &[2.37], 5e-3, 1, 0, &[0.5, 0.5, 0.3, 0.3])
                .is_err()
        );
        assert!(
            coupled_delayed_projection_tile(0.4, &[2.37], 5e-3, 1, 1, &[0.5, 0.5, 0.3]).is_err()
        );
        assert!(
            coupled_delayed_projection_tile(f64::NAN, &[2.37], 5e-3, 1, 1, &[0.5, 0.5, 0.3, 0.3])
                .is_err()
        );
        assert!(
            coupled_delayed_projection_tile(0.4, &[2.37], 5e-3, 1, 1, &[0.5, 0.5, 0.3, f64::NAN])
                .is_err()
        );
    }
}
