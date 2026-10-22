//! Fuzzy entropy pairwise-membership accumulator (Chen et al., 2007).
//!
//! Computes the upper-triangle sum of fuzzy memberships
//! `exp(-(d / r)^n)` using Chebyshev distance for all valid pairs with
//! `j > i + theiler_window`.
//!
//! Reference: Chen, W. et al. (2007), Medical Engineering & Physics 29(2), 164-169.

use crate::CoreError;

/// Count approximate-entropy template matches for every template row.
///
/// `traj` is row-major with shape (n_pts, dim). counts[i] is the number of
/// rows j with max(abs(traj[i] - traj[j])) <= r, including self-matches.
pub fn apen_counts(traj: &[f64], n_pts: usize, dim: usize, r: f64) -> Result<Vec<i64>, CoreError> {
    if r <= 0.0 {
        return Err(CoreError::invalid_argument("r must be positive"));
    }

    let all_rows_count =
        i64::try_from(n_pts).map_err(|_| CoreError::invalid_argument("too many template rows"))?;

    if dim == 0 {
        return Ok(vec![all_rows_count; n_pts]);
    }

    let rows: Vec<&[f64]> = traj.chunks_exact(dim).collect();
    let counts_usize = map_row_counts(&rows, r);
    counts_usize
        .into_iter()
        .map(|count| {
            i64::try_from(count)
                .map_err(|_| CoreError::invalid_argument("too many template matches"))
        })
        .collect()
}

fn map_row_counts(rows: &[&[f64]], r: f64) -> Vec<usize> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        rows.par_iter()
            .map(|row_i| {
                rows.iter()
                    .filter(|row_j| {
                        row_i
                            .iter()
                            .zip(row_j.iter())
                            .all(|(a, b)| (*a - *b).abs() <= r)
                    })
                    .count()
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        rows.iter()
            .map(|row_i| {
                rows.iter()
                    .filter(|row_j| {
                        row_i
                            .iter()
                            .zip(row_j.iter())
                            .all(|(a, b)| (*a - *b).abs() <= r)
                    })
                    .count()
            })
            .collect()
    }
}

/// Sum fuzzy memberships over all valid upper-triangle template pairs.
///
/// `traj` is row-major with shape (n_pts, dim). Pairs use
/// `j > i + theiler_window`.
pub fn fuzzy_entropy_sum(
    traj: &[f64],
    n_pts: usize,
    dim: usize,
    r: f64,
    n: u32,
    theiler_window: usize,
) -> Result<f64, CoreError> {
    if r <= 0.0 {
        return Err(CoreError::invalid_argument("r must be positive"));
    }

    let n_f = n as f64;
    Ok(sum_fuzzy_pairs(traj, n_pts, dim, r, n_f, theiler_window))
}

fn sum_fuzzy_pairs(
    traj: &[f64],
    n_pts: usize,
    dim: usize,
    r: f64,
    n_f: f64,
    theiler_window: usize,
) -> f64 {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n_pts)
            .into_par_iter()
            .fold(
                || 0.0_f64,
                |local_sum, i| {
                    local_sum + fuzzy_sum_from_i(traj, i, n_pts, dim, r, n_f, theiler_window)
                },
            )
            .reduce(|| 0.0_f64, |a, b| a + b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut total = 0.0_f64;
        for i in 0..n_pts {
            total += fuzzy_sum_from_i(traj, i, n_pts, dim, r, n_f, theiler_window);
        }
        total
    }
}

fn fuzzy_sum_from_i(
    traj: &[f64],
    i: usize,
    n_pts: usize,
    dim: usize,
    r: f64,
    n_f: f64,
    theiler_window: usize,
) -> f64 {
    let j_start = i.saturating_add(theiler_window).saturating_add(1);
    if j_start >= n_pts {
        return 0.0;
    }

    let row_i = i * dim;
    let mut local_sum = 0.0_f64;
    for j in j_start..n_pts {
        let row_j = j * dim;
        let mut d_max = 0.0_f64;
        for k in 0..dim {
            let diff = (traj[row_i + k] - traj[row_j + k]).abs();
            if diff > d_max {
                d_max = diff;
            }
        }
        let scaled = d_max / r;
        local_sum += (-(scaled.powf(n_f))).exp();
    }
    local_sum
}
