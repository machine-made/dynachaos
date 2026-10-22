//! Average Mutual Information via histogram estimation.
//!
//! Computes the delayed mutual information I(τ) for τ = 1..τ_max using
//! fixed-width histograms.
//!
//! Reference: Fraser & Swinney (1986), Phys. Rev. A 33(2), 1134-1140.

use crate::CoreError;

/// Compute delayed mutual information I(τ) for τ = 1..tau_max.
///
/// The estimator uses a uniform histogram with `n_bins` bins on each axis.
pub fn ami_histogram(x: &[f64], tau_max: usize, n_bins: usize) -> Result<Vec<f64>, CoreError> {
    if n_bins == 0 {
        return Err(CoreError::invalid_argument("n_bins must be > 0"));
    }

    let n = x.len();
    if n < 2 {
        return Err(CoreError::invalid_argument(
            "x must contain at least two values",
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::invalid_argument(
            "x must contain only finite values",
        ));
    }

    // Find data range
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    for &v in x {
        if v < x_min {
            x_min = v;
        }
        if v > x_max {
            x_max = v;
        }
    }
    // Slight padding so max value falls inside last bin
    let range = x_max - x_min;
    if range <= 0.0 {
        return Ok(vec![0.0f64; tau_max]);
    }
    let x_max_padded = x_max + range * 1e-10;
    let bin_width = (x_max_padded - x_min) / n_bins as f64;

    let mut mi_values = vec![0.0f64; tau_max];

    // Reusable histogram buffers
    let n_bins2 = n_bins * n_bins;
    let mut joint = vec![0u64; n_bins2];
    let mut marginal_x = vec![0u64; n_bins];
    let mut marginal_y = vec![0u64; n_bins];

    for tau in 1..=tau_max {
        if tau >= n {
            break;
        }
        let n_pairs = n - tau;
        if n_pairs < 2 {
            continue;
        }

        // Clear histograms
        for v in joint.iter_mut() {
            *v = 0;
        }
        for v in marginal_x.iter_mut() {
            *v = 0;
        }
        for v in marginal_y.iter_mut() {
            *v = 0;
        }

        // Accumulate
        for t in 0..n_pairs {
            let bx = ((x[t] - x_min) / bin_width) as usize;
            let by = ((x[t + tau] - x_min) / bin_width) as usize;
            // Clamp to valid range (shouldn't be needed but safety)
            let bx = bx.min(n_bins - 1);
            let by = by.min(n_bins - 1);
            joint[bx * n_bins + by] += 1;
            marginal_x[bx] += 1;
            marginal_y[by] += 1;
        }

        // Compute MI = Σ p(i,j) * log(p(i,j) / (p_x(i) * p_y(j)))
        let n_f = n_pairs as f64;
        let mut mi = 0.0f64;
        for i in 0..n_bins {
            if marginal_x[i] == 0 {
                continue;
            }
            let px = marginal_x[i] as f64 / n_f;
            for j in 0..n_bins {
                let count = joint[i * n_bins + j];
                if count == 0 || marginal_y[j] == 0 {
                    continue;
                }
                let pxy = count as f64 / n_f;
                let py = marginal_y[j] as f64 / n_f;
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
        mi_values[tau - 1] = mi;
    }

    Ok(mi_values)
}
