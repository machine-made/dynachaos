//! Recurrence Quantification Analysis — diagonal and vertical line extraction.
//!
//! These are the hot inner loops of RQA: scanning an N×N boolean recurrence
//! matrix for consecutive runs of `true` along diagonals (determinism) and
//! columns (laminarity).
//! They validate non-empty square shape only; public RQA semantics such as
//! recurrence-matrix symmetry are enforced by `dynachaos.diagnostics.recurrence.rqa`.

use ndarray::ArrayView2;

use crate::CoreError;

/// Count consecutive `true` run lengths in a one-dimensional boolean mask.
pub fn count_line_lengths(mask: &[bool], min_length: usize) -> Result<Vec<i64>, CoreError> {
    if min_length == 0 {
        return Err(CoreError::invalid_argument("min_length must be > 0"));
    }

    let mut lengths: Vec<i64> = Vec::new();
    let mut current: usize = 0;

    for &value in mask {
        if value {
            current += 1;
        } else {
            if current >= min_length {
                lengths.push(current as i64);
            }
            current = 0;
        }
    }
    if current >= min_length {
        lengths.push(current as i64);
    }

    Ok(lengths)
}

/// Extract diagonal line lengths from the upper triangle of a recurrence matrix.
///
/// Scans every super-diagonal k = 1, 2, ..., N-1 and records the length of
/// each consecutive run of `true` values that meets the minimum threshold.
pub fn diagonal_lines(r: ArrayView2<bool>, l_min: usize) -> Result<Vec<i64>, CoreError> {
    if l_min == 0 {
        return Err(CoreError::invalid_argument("l_min must be > 0"));
    }
    let n = r.shape()[0];
    if n == 0 || r.shape()[1] != n {
        return Err(CoreError::invalid_argument(
            "R must be a non-empty square matrix",
        ));
    }
    let mut lengths: Vec<i64> = Vec::new();

    for k in 1..n {
        let diag_len = n - k;
        let mut current: usize = 0;

        for i in 0..diag_len {
            if r[[i, i + k]] {
                current += 1;
            } else {
                if current >= l_min {
                    lengths.push(current as i64);
                }
                current = 0;
            }
        }
        if current >= l_min {
            lengths.push(current as i64);
        }
    }

    Ok(lengths)
}

/// Extract vertical line lengths from a recurrence matrix.
///
/// Scans each column j and records consecutive runs of `true` values.
pub fn vertical_lines(r: ArrayView2<bool>, v_min: usize) -> Result<Vec<i64>, CoreError> {
    if v_min == 0 {
        return Err(CoreError::invalid_argument("v_min must be > 0"));
    }
    let n = r.shape()[0];
    if n == 0 || r.shape()[1] != n {
        return Err(CoreError::invalid_argument(
            "R must be a non-empty square matrix",
        ));
    }
    let mut lengths: Vec<i64> = Vec::new();

    // Transpose so column scans become row scans (cache-friendly).
    let rt = r.t();

    for j in 0..n {
        let mut current: usize = 0;

        for i in 0..n {
            if rt[[j, i]] {
                current += 1;
            } else {
                if current >= v_min {
                    lengths.push(current as i64);
                }
                current = 0;
            }
        }
        if current >= v_min {
            lengths.push(current as i64);
        }
    }

    Ok(lengths)
}
