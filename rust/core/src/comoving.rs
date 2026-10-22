//! Co-moving Lyapunov kernels for logistic coupled-map lattices.

use crate::CoreError;

/// Compute the co-moving Lyapunov spectrum for a logistic CML with g=f.
///
/// The caller owns RNG and initial-state construction.
pub fn comoving_lyapunov_logistic(
    x_init: &[f64],
    v_values: &[f64],
    a: f64,
    eps: f64,
    n_iter: usize,
    n_transient: usize,
) -> Result<Vec<f64>, CoreError> {
    if x_init.is_empty() {
        return Err(CoreError::invalid_argument(
            "x_init must contain at least one site",
        ));
    }
    if n_iter == 0 {
        return Err(CoreError::invalid_argument("n_iter must be positive"));
    }

    let x_attractor = logistic_cml_after_transient(x_init, a, eps, n_transient);
    let mut lambda_v = Vec::with_capacity(v_values.len());
    for &v in v_values {
        lambda_v.push(comoving_velocity(&x_attractor, v, a, eps, n_iter));
    }
    Ok(lambda_v)
}

fn logistic_cml_after_transient(x_init: &[f64], a: f64, eps: f64, n_transient: usize) -> Vec<f64> {
    let n = x_init.len();
    let mut x = x_init.to_vec();
    let mut x_next = vec![0.0; n];
    let mut fx = vec![0.0; n];

    for _ in 0..n_transient {
        logistic_cml_step(&x, &mut x_next, &mut fx, a, eps);
        std::mem::swap(&mut x, &mut x_next);
    }

    x
}

fn comoving_velocity(x_attractor: &[f64], v: f64, a: f64, eps: f64, n_iter: usize) -> f64 {
    let n_sites = x_attractor.len();
    let center = n_sites / 2;
    let segment_length = n_iter.min(100);
    let n_segments = (n_iter / segment_length).max(1);

    let mut x = x_attractor.to_vec();
    let mut x_next = vec![0.0; n_sites];
    let mut fx = vec![0.0; n_sites];
    let mut delta = vec![0.0; n_sites];
    let mut delta_next = vec![0.0; n_sites];
    let mut tangent = vec![0.0; n_sites];

    let mut total_log_growth = 0.0;
    let mut valid_segments = 0usize;

    for _ in 0..n_segments {
        delta.fill(0.0);
        delta[center] = 1.0;
        let mut renorm_accum = 0.0;

        for step in 0..segment_length {
            logistic_cml_step_with_tangent(
                &x,
                &delta,
                &mut x_next,
                &mut delta_next,
                &mut fx,
                &mut tangent,
                a,
                eps,
            );
            std::mem::swap(&mut x, &mut x_next);
            std::mem::swap(&mut delta, &mut delta_next);

            if (step + 1) % 10 == 0 {
                let max_delta = max_abs_like_numpy(&delta);
                if max_delta > 0.0 {
                    renorm_accum += max_delta.ln();
                    for value in &mut delta {
                        *value /= max_delta;
                    }
                }
            }
        }

        let site_offset = (v * segment_length as f64).floor() as isize;
        let site_index = (center as isize + site_offset).rem_euclid(n_sites as isize) as usize;
        let amp = delta[site_index].abs();
        if amp > 0.0 {
            total_log_growth += amp.ln() + renorm_accum;
            valid_segments += 1;
        }
    }

    if valid_segments > 0 {
        total_log_growth / (valid_segments as f64 * segment_length as f64)
    } else {
        -10.0
    }
}

fn logistic_cml_step(x: &[f64], x_next: &mut [f64], fx: &mut [f64], a: f64, eps: f64) {
    for (out, &value) in fx.iter_mut().zip(x.iter()) {
        *out = logistic(value, a);
    }
    apply_periodic_cml(fx, x_next, eps);
}

#[allow(clippy::too_many_arguments)]
fn logistic_cml_step_with_tangent(
    x: &[f64],
    delta: &[f64],
    x_next: &mut [f64],
    delta_next: &mut [f64],
    fx: &mut [f64],
    tangent: &mut [f64],
    a: f64,
    eps: f64,
) {
    for i in 0..x.len() {
        fx[i] = logistic(x[i], a);
        tangent[i] = logistic_derivative(x[i], a) * delta[i];
    }
    apply_periodic_cml(fx, x_next, eps);
    apply_periodic_cml(tangent, delta_next, eps);
}

fn apply_periodic_cml(values: &[f64], out: &mut [f64], eps: f64) {
    let n = values.len();
    let half_eps = eps / 2.0;
    for i in 0..n {
        let left = if i == 0 { n - 1 } else { i - 1 };
        let right = (i + 1) % n;
        out[i] = values[i] + half_eps * (values[right] + values[left] - 2.0 * values[i]);
    }
}

#[inline]
fn logistic(x: f64, a: f64) -> f64 {
    1.0 - a * x * x
}

#[inline]
fn logistic_derivative(x: f64, a: f64) -> f64 {
    -2.0 * a * x
}

fn max_abs_like_numpy(values: &[f64]) -> f64 {
    let mut max_value = 0.0;
    for &value in values {
        let abs_value = value.abs();
        if abs_value.is_nan() {
            return f64::NAN;
        }
        if abs_value > max_value {
            max_value = abs_value;
        }
    }
    max_value
}
