//! Exact all-pairs correlation integral (Grassberger-Procaccia).
//!
//! Streams through all valid pairs (i, j) with |i-j| > theiler_window,
//! computes the distance, and increments count bins for each r threshold
//! exceeded.  Uses O(n_r) memory for the count array — no distance matrix.
//!
//! Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), 189-208.

use crate::CoreError;

/// Count pairs within each distance threshold (exact all-pairs).
///
/// `traj` is row-major with shape (n, dim). `r_values` must be sorted
/// ascending. counts[k] is the number of valid pairs with dist < r_values[k].
pub fn correlation_counts(
    traj: &[f64],
    n: usize,
    dim: usize,
    r_values: &[f64],
    theiler_window: usize,
    use_chebyshev: bool,
) -> Result<Vec<i64>, CoreError> {
    let n_r = r_values.len();

    // Runtime validation: r_values must be sorted ascending
    if !r_values.windows(2).all(|w| w[0] <= w[1]) {
        return Err(CoreError::invalid_argument(
            "r_values must be sorted in ascending order",
        ));
    }

    if n_r == 0 {
        return Ok(Vec::new());
    }

    let r_sq: Vec<f64> = r_values.iter().map(|&r| r * r).collect();
    let r_max = r_values[n_r - 1];
    let r_max_sq = r_max * r_max;

    let mut counts = if use_chebyshev {
        accumulate_diff_counts(n, n_r, |i, local| {
            count_chebyshev_from_i(local, i, n, dim, theiler_window, traj, r_values, r_max);
        })
    } else {
        accumulate_diff_counts(n, n_r, |i, local| {
            count_euclidean_from_i(local, i, n, dim, theiler_window, traj, &r_sq, r_max_sq);
        })
    };

    // Forward prefix sum: convert differential counts to cumulative.
    // counts[k] = #{pairs with dist < r_values[k]}.
    for k in 1..n_r {
        counts[k] += counts[k - 1];
    }

    Ok(counts)
}

fn accumulate_diff_counts<F>(n: usize, n_r: usize, count_from_i: F) -> Vec<i64>
where
    F: Fn(usize, &mut [i64]) + Sync,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .fold(
                || vec![0i64; n_r],
                |mut local, i| {
                    count_from_i(i, &mut local);
                    local
                },
            )
            .reduce(
                || vec![0i64; n_r],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut local = vec![0i64; n_r];
        for i in 0..n {
            count_from_i(i, &mut local);
        }
        local
    }
}

#[allow(clippy::too_many_arguments)]
fn count_chebyshev_from_i(
    local: &mut [i64],
    i: usize,
    n: usize,
    dim: usize,
    theiler_window: usize,
    traj: &[f64],
    r_values: &[f64],
    r_max: f64,
) {
    let n_r = r_values.len();
    let j_start = i.saturating_add(theiler_window).saturating_add(1);
    if j_start >= n {
        return;
    }
    let row_i = i * dim;
    for j in j_start..n {
        let row_j = j * dim;
        let mut d_max = 0.0f64;
        let mut skip = false;
        for k in 0..dim {
            let diff = (traj[row_i + k] - traj[row_j + k]).abs();
            if diff > r_max {
                skip = true;
                break;
            }
            if diff > d_max {
                d_max = diff;
            }
        }
        if skip {
            continue;
        }
        let pos = r_values.partition_point(|&r| r <= d_max);
        if pos < n_r {
            local[pos] += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn count_euclidean_from_i(
    local: &mut [i64],
    i: usize,
    n: usize,
    dim: usize,
    theiler_window: usize,
    traj: &[f64],
    r_sq: &[f64],
    r_max_sq: f64,
) {
    let n_r = r_sq.len();
    let j_start = i.saturating_add(theiler_window).saturating_add(1);
    if j_start >= n {
        return;
    }
    let row_i = i * dim;
    for j in j_start..n {
        let row_j = j * dim;
        let mut sq = 0.0f64;
        let mut skip = false;
        for k in 0..dim {
            let diff = traj[row_i + k] - traj[row_j + k];
            sq += diff * diff;
            if sq > r_max_sq {
                skip = true;
                break;
            }
        }
        if skip {
            continue;
        }
        let pos = r_sq.partition_point(|&rsq| rsq <= sq);
        if pos < n_r {
            local[pos] += 1;
        }
    }
}
