//! Coupled-map lattice kernels.

use crate::CoreError;

/// Compute the logistic CML subsystem Jacobian for sites 0..l-1.
///
/// Returns the row-major flattened l x l block. `x` is the full lattice.
pub fn cml_jacobian_logistic(x: &[f64], a: f64, eps: f64, l: usize) -> Result<Vec<f64>, CoreError> {
    let n = x.len();
    if l < 1 || l > n {
        return Err(CoreError::invalid_argument(format!(
            "L must satisfy 1 <= L <= N (got L={l}, N={n})"
        )));
    }

    let matrix_len = l
        .checked_mul(l)
        .ok_or_else(|| CoreError::invalid_argument("L is too large for an L x L matrix"))?;
    let mut jacobian = vec![0.0_f64; matrix_len];
    let diagonal_scale = 1.0 - eps;
    let off_diagonal_scale = eps / 2.0;

    for i in 0..l {
        let row = i * l;
        jacobian[row + i] = diagonal_scale * logistic_derivative(x[i], a);

        let i_left = if i == 0 { n - 1 } else { i - 1 };
        if i_left < l {
            jacobian[row + i_left] = off_diagonal_scale * logistic_derivative(x[i_left], a);
        }

        let i_right = (i + 1) % n;
        if i_right < l {
            jacobian[row + i_right] = off_diagonal_scale * logistic_derivative(x[i_right], a);
        }
    }

    Ok(jacobian)
}

#[inline]
fn logistic_derivative(x: f64, a: f64) -> f64 {
    -2.0 * a * x
}

/// The piecewise map's breakpoint `c = (sqrt(5) - 1) / 2` for model (A).
///
/// Same value as `c` in `model_A_f` (`src/dynachaos/cml/spatiotemporal.py:51`).
const MODEL_A_C: f64 = 0.6180339887498949;

/// Local map of CML model (A): the piecewise Kaneko (1985) map.
///
/// ```text
/// f(u) = u + u * u - 0.01          when u < c
/// f(u) = -3.0 * (u - c) + 1.0 - 0.01  otherwise
/// ```
///
/// Same expression and operation order as `model_A_f`
/// (`src/dynachaos/cml/spatiotemporal.py:42`), whose `a = -0.01` is written
/// as a subtraction here.
#[inline]
fn model_a_f(u: f64) -> f64 {
    if u < MODEL_A_C {
        u + u * u - 0.01
    } else {
        -3.0 * (u - MODEL_A_C) + 1.0 - 0.01
    }
}

/// Local map of CML model (B): the circle map `f(u) = (u + 0.2 * sin(2 pi u) +
/// 0.55) mod 1`.
///
/// Same expression and operation order as `model_B_f`
/// (`src/dynachaos/cml/spatiotemporal.py:55`). The Python `% 1.0` is a floored
/// modulo, so this uses `rem_euclid`, not Rust's truncating `%`: for a
/// negative argument the two differ (`-0.23 % 1.0` is `-0.23` in Rust but
/// `0.77` in Python).
#[inline]
fn model_b_f(u: f64) -> f64 {
    (u + 0.2 * (2.0 * std::f64::consts::PI * u).sin() + 0.55).rem_euclid(1.0)
}

/// Coupling map of CML model (B): `g(u) = sin(2 * pi * u)`.
///
/// Same expression as `model_B_g` (`src/dynachaos/cml/spatiotemporal.py:60`).
#[inline]
fn model_b_g(u: f64) -> f64 {
    (2.0 * std::f64::consts::PI * u).sin()
}

/// Local map of CML model (C): the logistic map `f(u) = 1 - 1.752 * u * u`.
///
/// Same expression and operation order as `model_C_f`
/// (`src/dynachaos/cml/spatiotemporal.py:65`), which calls `logistic`
/// (`src/dynachaos/maps/primitives.py:6`) with `a = 1.752`.
#[inline]
fn model_c_f(u: f64) -> f64 {
    1.0 - 1.752 * u * u
}

/// The local map `f` of the selected model.
#[inline]
fn cml_local_map(model: u8, u: f64) -> f64 {
    match model {
        0 => model_a_f(u),
        1 => model_b_f(u),
        _ => model_c_f(u),
    }
}

/// The coupling map `g` of the selected model: `model_B_g` for model (B),
/// the local map itself for models (A) and (C).
#[inline]
fn cml_coupling_map(model: u8, u: f64) -> f64 {
    match model {
        1 => model_b_g(u),
        _ => cml_local_map(model, u),
    }
}

/// One periodic CML step of `x` into `next`, using `gx` as scratch.
///
/// Mirrors `cml_step` (`src/dynachaos/cml/primitives.py:27`):
///
/// ```text
/// next[i] = f(x[i]) + eps / 2.0 * (g(x[i+1]) + g(x[i-1]) - 2.0 * g(x[i]))
/// ```
///
/// with periodic boundary conditions (`np.roll` on the 1-D lattice), so the
/// left neighbour of site 0 is site `n - 1`. The update is evaluated in the
/// same operation order as the Python line, so the field is bit-identical.
fn cml_step_into(model: u8, x: &[f64], eps: f64, gx: &mut [f64], next: &mut [f64]) {
    let n = x.len();
    for (i, &u) in x.iter().enumerate() {
        gx[i] = cml_coupling_map(model, u);
    }
    for (i, &u) in x.iter().enumerate() {
        let gl = gx[(i + 1) % n];
        let gr = gx[(i + n - 1) % n];
        next[i] = cml_local_map(model, u) + eps / 2.0 * (gl + gr - 2.0 * gx[i]);
    }
}

/// CML space-time tile: the field of a coupled-map lattice after a transient.
///
/// `model` selects the lattice: `0` is model (A) (piecewise map, `f = g =
/// model_a_f`), `1` is model (B) (circle map `model_b_f` coupled through
/// `model_b_g`), `2` is model (C) (logistic map, `f = g = model_c_f`). These
/// are the three models `simulate_cml` runs in
/// `src/dynachaos/cml/spatiotemporal.py:70`.
///
/// The kernel iterates `n_transient` steps of [`cml_step_into`] from `x0`,
/// then records the state after each of the next `n_record` steps. The result
/// is flat, row-major by time then site: `out[t * n_sites + i]` is site `i`
/// of the `t`-th recorded field, exactly as `spacetime[t] = x` fills the
/// `(n_record, N)` array in Python.
///
/// # Errors
///
/// `CoreError::InvalidArgument` when `model` is not 0, 1 or 2, `x0` is empty,
/// `n_record` is zero, or `eps` or an `x0` entry is not finite.
pub fn cml_spacetime_tile(
    model: u8,
    eps: f64,
    n_transient: usize,
    n_record: usize,
    x0: &[f64],
) -> Result<Vec<f64>, CoreError> {
    if model > 2 {
        return Err(CoreError::invalid_argument(
            "model must be 0 (A), 1 (B) or 2 (C)",
        ));
    }
    if x0.is_empty() {
        return Err(CoreError::invalid_argument("x0 must not be empty"));
    }
    if n_record == 0 {
        return Err(CoreError::invalid_argument("n_record must be at least 1"));
    }
    if !eps.is_finite() {
        return Err(CoreError::invalid_argument("eps must be finite"));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument("every x0 entry must be finite"));
    }

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut gx = vec![0.0_f64; n];
    let mut next = vec![0.0_f64; n];
    for _ in 0..n_transient {
        cml_step_into(model, &x, eps, &mut gx, &mut next);
        std::mem::swap(&mut x, &mut next);
    }
    let mut out = Vec::with_capacity(n_record * n);
    for _ in 0..n_record {
        cml_step_into(model, &x, eps, &mut gx, &mut next);
        std::mem::swap(&mut x, &mut next);
        out.extend_from_slice(&x);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values produced once from `simulate_cml`
    // (`src/dynachaos/cml/spatiotemporal.py:70`) at n_transient = 10,
    // n_record = 4 on the eight-site lattice below, and pasted as literals;
    // the tests do not call Python. The Rust step evaluates the same
    // expression in the same order, so equality is exact.
    const X0: [f64; 8] = [0.1, 0.4, 0.7, 0.2, 0.9, 0.3, 0.6, 0.05];

    /// Model (A) at eps = 0.07, four recorded rows of eight sites.
    const MODEL_A_EPS_0_07: [f64; 32] = [
        0.5722600102033311,
        0.9152806832906797,
        0.2812773050352615,
        0.2341773318981848,
        0.5916762144483211,
        0.7115712669748102,
        0.17566147510761795,
        0.3298712738283996,
        0.8459027410801728,
        0.13478647372076477,
        0.3390713009345172,
        0.304360501305456,
        0.9011281283887052,
        0.6992206322672749,
        0.22259474629425893,
        0.4366973864916854,
        0.3115586361362965,
        0.159212300200663,
        0.43150604160307937,
        0.3803726469558819,
        0.17053760640050725,
        0.7082893907824569,
        0.2915276178718074,
        0.5940826461534721,
        0.4096287197625384,
        0.1975631791544051,
        0.5893008500535636,
        0.5069084242324511,
        0.21954737658487375,
        0.6883522111299419,
        0.3988286240321281,
        0.8982056766921336,
    ];

    /// Model (B) at eps = 0.024, four recorded rows of eight sites.
    const MODEL_B_EPS_0_024: [f64; 32] = [
        0.045543516657787324,
        0.09931733422426631,
        0.6461627624129911,
        0.07847875453892156,
        0.6960697579618991,
        0.052229813057464325,
        0.06500076765785406,
        0.6978735746593577,
        0.6408724414337659,
        0.7460077324872603,
        0.06900309098584798,
        0.6909364030157007,
        0.08962576549406343,
        0.6524047137248185,
        0.6774051888755622,
        0.08938224999619432,
        0.04904349842449303,
        0.11581639085510118,
        0.6697619408605119,
        0.08836469840968096,
        0.7125848221352967,
        0.05410387404492959,
        0.06597537239745296,
        0.7130491347465527,
        0.6487273197466532,
        0.776015471136554,
        0.07996625168471654,
        0.7089614432902812,
        0.10175254174945098,
        0.6559516682712113,
        0.6791851623856258,
        0.10024401299013663,
    ];

    /// Model (C) at eps = 0.2, four recorded rows of eight sites.
    const MODEL_C_EPS_0_2: [f64; 32] = [
        0.4464769551015942,
        0.7029652566346797,
        0.5083118747893265,
        0.5072186424235594,
        0.06006871547968437,
        0.7378642635675215,
        0.3515261394704477,
        0.2569158956473313,
        0.6224616424707463,
        0.22719212551988957,
        0.5062025493374342,
        0.5935088050342571,
        0.8544823091358006,
        0.2146260107908447,
        0.7198526661289533,
        0.850912067706327,
        0.3210412064461022,
        0.814878491891917,
        0.5700947426142545,
        0.33346865557856076,
        -0.09314939028651081,
        0.7167291986212598,
        0.13878362847372133,
        -0.17349960970622455,
        0.7339292400795373,
        -0.0056990049558351685,
        0.40864876709572295,
        0.785678614647826,
        0.8783557387413246,
        0.27510235117213955,
        0.8777296679733537,
        0.9363769489913174,
    ];

    /// Model (B) at eps = 0.024 from x0 = (-0.9, 0.2, 0.5, -0.3), no
    /// transient, two recorded rows. The negative entries exercise the
    /// floored modulo in `model_b_f`.
    const MODEL_B_NEGATIVE_X0: [f64; 8] = [
        0.7534502044034754,
        0.9244393698954568,
        0.05000000000000004,
        0.08966747615956269,
        0.12841478152367405,
        0.3856961671947739,
        0.6553102143656074,
        0.7253733350906701,
    ];

    #[test]
    fn cml_model_a_matches_python_exactly() {
        let out = cml_spacetime_tile(0, 0.07, 10, 4, &X0).unwrap();
        assert_eq!(out, MODEL_A_EPS_0_07);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn cml_model_b_matches_python_exactly() {
        let out = cml_spacetime_tile(1, 0.024, 10, 4, &X0).unwrap();
        assert_eq!(out, MODEL_B_EPS_0_024);
    }

    /// macOS's libm `sin` differs from glibc's by 1 ulp on some inputs, so
    /// model (B) is not bit-exact there (CI: 3.3e-16 at one value). Eight
    /// numpy runs with a 1-ulp error on every sin result stay within
    /// 1.6e-15 of the literals, so this key is not sensitive and the rule
    /// of `scripts/check_wasm_cml_spacetime.py` applies: within 1e-9.
    #[test]
    #[cfg(target_os = "macos")]
    fn cml_model_b_matches_python_within_sin_rounding() {
        let out = cml_spacetime_tile(1, 0.024, 10, 4, &X0).unwrap();
        assert_eq!(out.len(), MODEL_B_EPS_0_024.len());
        for (k, (got, want)) in out.iter().zip(MODEL_B_EPS_0_024).enumerate() {
            assert!((got - want).abs() <= 1e-9, "value {k}: {got} != {want}");
        }
    }

    #[test]
    fn cml_model_c_matches_python_exactly() {
        let out = cml_spacetime_tile(2, 0.2, 10, 4, &X0).unwrap();
        assert_eq!(out, MODEL_C_EPS_0_2);
    }

    #[test]
    fn cml_model_b_wraps_negative_arguments_with_floored_modulo() {
        // Python `x % 1.0` is floored: f(-0.9) = 0.7675570504584948, where
        // Rust's truncating `%` would return the raw -0.23244294954150524.
        // The exact-match literals pin the whole two-step field.
        assert_eq!(model_b_f(-0.9), 0.7675570504584948);
        let out = cml_spacetime_tile(1, 0.024, 0, 2, &[-0.9, 0.2, 0.5, -0.3]).unwrap();
        assert_eq!(out, MODEL_B_NEGATIVE_X0);
    }

    #[test]
    fn cml_record_is_row_major_by_time_then_site() {
        // With n_transient = 0 and n_record = 1 the single row is one step of
        // the lattice: site i is f(x_i) + eps/2 (g(x_{i+1}) + g(x_{i-1}) -
        // 2 g(x_i)) with periodic neighbours.
        let out = cml_spacetime_tile(2, 0.2, 0, 1, &[0.25, 0.5, 0.75]).unwrap();
        let f = |u: f64| 1.0 - 1.752 * u * u;
        let expected = [
            f(0.25) + 0.1 * (f(0.5) + f(0.75) - 2.0 * f(0.25)),
            f(0.5) + 0.1 * (f(0.75) + f(0.25) - 2.0 * f(0.5)),
            f(0.75) + 0.1 * (f(0.25) + f(0.5) - 2.0 * f(0.75)),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn cml_rejects_invalid_input() {
        assert!(cml_spacetime_tile(3, 0.2, 10, 4, &X0).is_err());
        assert!(cml_spacetime_tile(0, 0.07, 10, 4, &[]).is_err());
        assert!(cml_spacetime_tile(0, 0.07, 10, 0, &X0).is_err());
        assert!(cml_spacetime_tile(0, f64::NAN, 10, 4, &X0).is_err());
        assert!(cml_spacetime_tile(0, 0.07, 10, 4, &[0.5, f64::INFINITY]).is_err());
    }
}
