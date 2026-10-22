//! Synthetic intermittency oracle kernels.

use ndarray::Array2;

use crate::CoreError;

/// Integrate the Pomeau–Manneville type-I map for `n` steps.
pub fn pm_type_i_oracle(
    n: usize,
    x0: f64,
    eps: f64,
    a: f64,
    modulo: bool,
) -> Result<Vec<f64>, CoreError> {
    validate_n(n)?;
    validate_finite(&[x0, eps, a])?;

    let mut x = x0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = x + eps + a * x * x;
        if modulo {
            x = x.rem_euclid(1.0);
        }
        out.push(x);
    }
    Ok(out)
}

/// Integrate the Pomeau–Manneville type-II map for `n` steps.
pub fn pm_type_ii_oracle(
    n: usize,
    x0: f64,
    y0: f64,
    eps: f64,
    a: f64,
    theta: f64,
) -> Result<Array2<f64>, CoreError> {
    validate_n(n)?;
    validate_finite(&[x0, y0, eps, a, theta])?;

    let mut x = x0;
    let mut y = y0;
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let r2 = x * x + y * y;
        let growth = 1.0 + eps + a * r2;
        let xr = cos_theta * x - sin_theta * y;
        let yr = sin_theta * x + cos_theta * y;
        x = growth * xr;
        y = growth * yr;
        out.push(x);
        out.push(y);
    }

    Array2::from_shape_vec((n, 2), out)
        .map_err(|err| CoreError::invalid_argument(format!("shape error: {err}")))
}

/// Integrate the Pomeau–Manneville type-III map for `n` steps.
pub fn pm_type_iii_oracle(n: usize, x0: f64, eps: f64, a: f64) -> Result<Vec<f64>, CoreError> {
    validate_n(n)?;
    validate_finite(&[x0, eps, a])?;

    let mut x = x0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = -(1.0 + eps) * x - a * x * x * x;
        out.push(x);
    }
    Ok(out)
}

/// Integrate the on-off intermittency map along a driver series.
pub fn on_off_oracle(
    driver: &[f64],
    x0: f64,
    transverse_lyapunov: f64,
    noise_scale: f64,
) -> Result<Vec<f64>, CoreError> {
    if driver.is_empty() {
        return Err(CoreError::invalid_argument("driver must be non-empty"));
    }
    validate_finite(&[x0, transverse_lyapunov, noise_scale])?;

    let mut x = x0;
    let mut out = Vec::with_capacity(driver.len());
    for &eta in driver {
        if !eta.is_finite() {
            return Err(CoreError::invalid_argument(
                "driver must contain only finite values",
            ));
        }
        let multiplier = (transverse_lyapunov + noise_scale * eta).exp();
        x = multiplier * x / (1.0 + x * x);
        out.push(x);
    }
    Ok(out)
}

/// Integrate the on-off skew-logistic map for `n` steps.
pub fn on_off_skew_logistic_oracle(
    n: usize,
    x0: f64,
    y0: f64,
    eps: f64,
) -> Result<Array2<f64>, CoreError> {
    validate_n(n)?;
    validate_finite(&[x0, y0, eps])?;

    let mut x = x0;
    let mut y = y0;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let driver = 4.0 * x * (1.0 - x);
        let multiplier = 4.0 * eps * (1.0 - 2.0 * x);
        y = multiplier * y / (1.0 + y * y);
        x = driver;
        out.push(x);
        out.push(y);
    }

    Array2::from_shape_vec((n, 2), out)
        .map_err(|err| CoreError::invalid_argument(format!("shape error: {err}")))
}

/// Integrate the logistic map for `n` steps.
pub fn logistic_type_i_oracle(n: usize, x0: f64, r: f64) -> Result<Vec<f64>, CoreError> {
    validate_n(n)?;
    validate_finite(&[x0, r])?;

    let mut x = x0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = r * x * (1.0 - x);
        out.push(x);
    }
    Ok(out)
}

fn validate_n(n: usize) -> Result<(), CoreError> {
    if n == 0 {
        Err(CoreError::invalid_argument("n must be positive"))
    } else {
        Ok(())
    }
}

fn validate_finite(values: &[f64]) -> Result<(), CoreError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(CoreError::invalid_argument("parameters must be finite"))
    }
}
