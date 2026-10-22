//! Ordinal pattern distribution for permutation entropy.
//!
//! Encodes each sliding window's ordinal pattern (argsort permutation) as a
//! Lehmer-code integer and accumulates counts in a flat array.

use crate::CoreError;

/// Compute the argsort of a small window, then encode it as a Lehmer code.
///
/// This matches Python's `tuple(np.argsort(window))` convention:
/// the pattern records the *indices* that would sort the window.
#[inline]
fn pattern_index(window: &[f64]) -> usize {
    let d = window.len();

    // Step 1: argsort — indices that sort the window in ascending order.
    // For d ≤ 10, insertion sort is optimal (no allocation needed beyond stack).
    let mut indices = [0usize; 10]; // d ≤ 10 in practice
    for (i, slot) in indices.iter_mut().enumerate().take(d) {
        *slot = i;
    }
    // Insertion sort on indices by window value
    for i in 1..d {
        let key = indices[i];
        let key_val = window[key];
        let mut j = i;
        while j > 0 && window[indices[j - 1]] > key_val {
            indices[j] = indices[j - 1];
            j -= 1;
        }
        indices[j] = key;
    }

    // Step 2: Lehmer code of the argsort permutation.
    let mut index: usize = 0;
    let mut factor: usize = 1;
    for i in (0..d).rev() {
        let mut count = 0usize;
        for j in (i + 1)..d {
            if indices[j] < indices[i] {
                count += 1;
            }
        }
        index += count * factor;
        if i > 0 {
            factor *= d - i;
        }
    }
    index
}

/// Factorial of a small integer (d ≤ 10 in practice).
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Compute the ordinal pattern distribution of a time series.
///
/// Returns raw Lehmer-code counts of length d! and the number of windows.
pub fn ordinal_distribution(x: &[f64], d: usize, tau: usize) -> Result<(Vec<i64>, i64), CoreError> {
    if d < 2 {
        return Err(CoreError::invalid_argument("d must be >= 2"));
    }
    if d > 10 {
        return Err(CoreError::invalid_argument("d must be <= 10"));
    }
    if tau < 1 {
        return Err(CoreError::invalid_argument("tau must be >= 1"));
    }

    let n = x.len();
    let n_windows = n.saturating_sub((d - 1) * tau);
    if n_windows == 0 {
        return Err(CoreError::invalid_argument(
            "time series is too short for the requested d and tau",
        ));
    }
    let n_perm = factorial(d);

    let mut counts = vec![0i64; n_perm];
    let mut window_buf = vec![0.0f64; d];

    for i in 0..n_windows {
        // Gather the delayed window
        for j in 0..d {
            window_buf[j] = x[i + j * tau];
        }
        let idx = pattern_index(&window_buf);
        counts[idx] += 1;
    }

    Ok((counts, n_windows as i64))
}
