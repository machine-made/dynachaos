//! Multifractal partition moments over dyadic box scales.
//!
//! For each box size `r` and moment order `q`, computes:
//! - `log_z(r, q) = ln(sum_i p_i(r)^q)`
//! - `alpha_num(r, q) = sum_i mu_i(r, q) ln p_i(r)`
//! - `f_num(r, q) = sum_i mu_i(r, q) ln mu_i(r, q)`
//!
//! where `p_i(r)` are box probabilities and
//! `mu_i(r, q) = p_i(r)^q / sum_j p_j(r)^q`.
//!
//! These are the canonical moments used to recover `tau(q)`, `D_q`,
//! `alpha(q)`, and `f(alpha)` via log-log regressions.

use ndarray::Array2;

use crate::CoreError;

/// Canonical moments at each box scale and moment order q.
pub struct MultifractalMoments {
    pub log_z: Array2<f64>,
    pub alpha_num: Array2<f64>,
    pub f_num: Array2<f64>,
    pub ln_scales: Vec<f64>,
}

/// Compute multifractal canonical moments for a 2D nonnegative measure field.
///
/// `field` is row-major with shape (ny, nx). Each box scale uses
/// non-overlapping boxes and truncates edge remainders.
pub fn multifractal_moments(
    field: &[f64],
    ny: usize,
    nx: usize,
    box_sizes: &[i64],
    q_values: &[f64],
) -> Result<MultifractalMoments, CoreError> {
    let n_scales = box_sizes.len();
    let n_q = q_values.len();

    let mut ln_scales = vec![f64::NAN; n_scales];
    let mut log_z = vec![f64::NAN; n_scales * n_q];
    let mut alpha_num = vec![f64::NAN; n_scales * n_q];
    let mut f_num = vec![f64::NAN; n_scales * n_q];

    let total_mass: f64 = field.iter().copied().sum();
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return Err(CoreError::invalid_argument(
            "field must have a positive finite total mass",
        ));
    }
    if field.iter().any(|&v| !v.is_finite() || v < 0.0) {
        return Err(CoreError::invalid_argument(
            "field must contain only finite nonnegative values",
        ));
    }

    for (si, &b_i64) in box_sizes.iter().enumerate() {
        if b_i64 <= 0 {
            continue;
        }
        let b = b_i64 as usize;
        let n_by = ny / b;
        let n_bx = nx / b;
        if n_by == 0 || n_bx == 0 {
            continue;
        }
        ln_scales[si] = (b as f64).ln();

        let n_boxes = n_by * n_bx;
        let mut probs: Vec<f64> = Vec::with_capacity(n_boxes);
        let mut used_mass = 0.0f64;

        for by in 0..n_by {
            let y0 = by * b;
            for bx in 0..n_bx {
                let x0 = bx * b;
                let mut mass = 0.0f64;
                for yy in 0..b {
                    let row = y0 + yy;
                    let base = row * nx;
                    for xx in 0..b {
                        mass += field[base + x0 + xx];
                    }
                }
                if mass > 0.0 {
                    probs.push(mass);
                    used_mass += mass;
                }
            }
        }

        if probs.is_empty() || !used_mass.is_finite() || used_mass <= 0.0 {
            continue;
        }
        let inv_used_mass = 1.0 / used_mass;
        for p in &mut probs {
            *p *= inv_used_mass;
        }

        for (qi, &q) in q_values.iter().enumerate() {
            let idx = si * n_q + qi;
            if !q.is_finite() {
                continue;
            }

            if (q - 1.0).abs() < 1e-12 {
                let mut shannon = 0.0f64;
                for &p in &probs {
                    shannon += p * p.ln();
                }
                // Z_1(r) = sum_i p_i = 1 by normalization.
                log_z[idx] = 0.0;
                alpha_num[idx] = shannon;
                f_num[idx] = shannon;
                continue;
            }

            let mut z = 0.0f64;
            for &p in &probs {
                z += p.powf(q);
            }
            if !z.is_finite() || z <= 0.0 {
                continue;
            }

            let ln_z = z.ln();
            let mut a = 0.0f64;
            let mut f = 0.0f64;
            for &p in &probs {
                let p_q = p.powf(q);
                let mu = p_q / z;
                if mu > 0.0 && mu.is_finite() {
                    a += mu * p.ln();
                    f += mu * mu.ln();
                }
            }

            log_z[idx] = ln_z;
            alpha_num[idx] = a;
            f_num[idx] = f;
        }
    }

    let log_z_arr = Array2::from_shape_vec((n_scales, n_q), log_z)
        .map_err(|e| CoreError::runtime(format!("shape error log_z: {e}")))?;
    let alpha_arr = Array2::from_shape_vec((n_scales, n_q), alpha_num)
        .map_err(|e| CoreError::runtime(format!("shape error alpha_num: {e}")))?;
    let f_arr = Array2::from_shape_vec((n_scales, n_q), f_num)
        .map_err(|e| CoreError::runtime(format!("shape error f_num: {e}")))?;

    Ok(MultifractalMoments {
        log_z: log_z_arr,
        alpha_num: alpha_arr,
        f_num: f_arr,
        ln_scales,
    })
}
