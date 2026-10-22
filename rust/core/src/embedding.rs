//! Cao embedding-dimension selector helpers.
//!
//! The expensive Cao/FNN nearest-neighbor statistics stay in Python/SciPy,
//! where cKDTree is substantially faster than the old brute-force Rust kernels.

/// Moving-average smoothing helper for 1D diagnostics.
#[inline]
fn smooth_series(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if window <= 1 || n < window {
        return values.to_vec();
    }
    let left = window / 2;
    let right = window - 1 - left;
    let mut padded = Vec::with_capacity(n + left + right);
    padded.extend(std::iter::repeat_n(values[0], left));
    padded.extend(values.iter().copied());
    padded.extend(std::iter::repeat_n(values[n - 1], right));

    let mut out = vec![0.0f64; n];
    let inv = 1.0f64 / window as f64;
    for i in 0..n {
        let mut acc = 0.0f64;
        for v in &padded[i..i + window] {
            acc += *v;
        }
        out[i] = acc * inv;
    }
    out
}

/// Select embedding dimension from a Cao E1(d) curve.
///
/// This mirrors the Python selector logic:
/// 1) choose onset of a stable near-1 plateau (forward window),
/// 2) fallback to first near-one crossing,
/// 3) fallback to closest value to 1.
#[allow(clippy::too_many_arguments)]
pub fn select_dimension_cao(
    e1: &[f64],
    near_one_lower: f64,
    near_one_upper: f64,
    saturation_tol: f64,
    plateau_span: usize,
    smoothing_window: usize,
    min_dim: usize,
    max_dim: Option<usize>,
) -> usize {
    let min_dim = min_dim.max(1);
    if e1.is_empty() {
        return min_dim;
    }

    let mut lo = near_one_lower;
    let mut hi = near_one_upper;
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }

    let smoothed = smooth_series(e1, smoothing_window);
    if !smoothed.iter().any(|v| v.is_finite()) {
        return min_dim;
    }

    let span = plateau_span.max(2);
    let max_allowed = max_dim.unwrap_or(smoothed.len() + 1).max(min_dim);
    let clamp = |d: usize| -> usize { d.max(min_dim).min(max_allowed) };

    // Primary: plateau onset in forward window.
    for idx in 0..smoothed.len() {
        let dim = idx + 1;
        let value = smoothed[idx];
        if dim < min_dim || !value.is_finite() {
            continue;
        }
        if value < lo || value > hi {
            continue;
        }
        if idx + span > smoothed.len() {
            continue;
        }

        let window = &smoothed[idx..idx + span];
        if window.iter().any(|v| !v.is_finite() || *v < lo || *v > hi) {
            continue;
        }

        let mut w_min = f64::INFINITY;
        let mut w_max = f64::NEG_INFINITY;
        let mut max_diff = 0.0f64;
        for (k, v) in window.iter().enumerate() {
            w_min = w_min.min(*v);
            w_max = w_max.max(*v);
            if k > 0 {
                let diff = (window[k] - window[k - 1]).abs();
                max_diff = max_diff.max(diff);
            }
        }
        if (w_max - w_min) <= 1.5 * saturation_tol || max_diff <= saturation_tol {
            return clamp(dim);
        }
    }

    // Fallback 1: first near-one crossing.
    for (idx, value) in smoothed.iter().enumerate() {
        let dim = idx + 1;
        if dim >= min_dim && value.is_finite() && *value >= lo {
            return clamp(dim);
        }
    }

    // Fallback 2: closest to 1 among valid dimensions.
    let mut best_dim: Option<usize> = None;
    let mut best_dist = f64::INFINITY;
    for (idx, value) in smoothed.iter().enumerate() {
        let dim = idx + 1;
        if dim < min_dim || !value.is_finite() {
            continue;
        }
        let dist = (*value - 1.0).abs();
        if dist < best_dist {
            best_dist = dist;
            best_dim = Some(dim);
        }
    }
    if let Some(dim) = best_dim {
        return clamp(dim);
    }

    clamp(e1.len() + 1)
}
