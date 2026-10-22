"""Synthetic intermittency oracle generators.

These helpers provide deterministic signals for testing intermittency
diagnostics. They are building-block generators, not classifiers.
"""

import os

import numpy as np

from dynachaos.maps.flows import lorenz_trajectory

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import logistic_type_i_oracle as _logistic_type_i_oracle_rs
    from dynachaos._rust import on_off_oracle as _on_off_oracle_rs
    from dynachaos._rust import on_off_skew_logistic_oracle as _on_off_skew_logistic_oracle_rs
    from dynachaos._rust import pm_type_i_oracle as _pm_type_i_oracle_rs
    from dynachaos._rust import pm_type_ii_oracle as _pm_type_ii_oracle_rs
    from dynachaos._rust import pm_type_iii_oracle as _pm_type_iii_oracle_rs

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _logistic_type_i_oracle_rs = None
    _on_off_oracle_rs = None
    _on_off_skew_logistic_oracle_rs = None
    _pm_type_i_oracle_rs = None
    _pm_type_ii_oracle_rs = None
    _pm_type_iii_oracle_rs = None


LOGISTIC_TYPE_I_ONSET = 1.0 + np.sqrt(8.0)
ON_OFF_SKEW_LOGISTIC_ONSET = 0.5
LORENZ_INTERMITTENCY_RHO = 166.2


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value_int != value or value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _finite_float(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def pm_type_i_oracle(n, x0=0.0, eps=1e-4, a=1.0, modulo=True):
    """Pomeau-Manneville Type-I normal-form orbit ``x -> x + eps + a*x**2``."""
    n = _positive_int(n, "n")
    x0 = _finite_float(x0, "x0")
    eps = _finite_float(eps, "eps")
    a = _finite_float(a, "a")
    modulo = bool(modulo)
    if _RUST_AVAILABLE and _pm_type_i_oracle_rs is not None:
        return np.asarray(_pm_type_i_oracle_rs(n, x0, eps, a, modulo), dtype=np.float64)
    return _pm_type_i_oracle_python(n, x0, eps, a, modulo)


def _pm_type_i_oracle_python(n, x0, eps, a, modulo):
    x = x0
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = x + eps + a * x * x
        if modulo:
            x = np.mod(x, 1.0)
        out[i] = x
    return out


def pm_type_ii_oracle(n, x0=1e-3, y0=0.0, eps=1e-3, a=-1.0, theta=None):
    """Pomeau-Manneville Type-II subcritical-Hopf normal-form orbit."""
    n = _positive_int(n, "n")
    x0 = _finite_float(x0, "x0")
    y0 = _finite_float(y0, "y0")
    eps = _finite_float(eps, "eps")
    a = _finite_float(a, "a")
    theta = np.sqrt(5.0) if theta is None else _finite_float(theta, "theta")
    if _RUST_AVAILABLE and _pm_type_ii_oracle_rs is not None:
        return np.asarray(_pm_type_ii_oracle_rs(n, x0, y0, eps, a, theta), dtype=np.float64)
    return _pm_type_ii_oracle_python(n, x0, y0, eps, a, theta)


def _pm_type_ii_oracle_python(n, x0, y0, eps, a, theta):
    x = x0
    y = y0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    out = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        r2 = x * x + y * y
        growth = 1.0 + eps + a * r2
        xr = cos_theta * x - sin_theta * y
        yr = sin_theta * x + cos_theta * y
        x = growth * xr
        y = growth * yr
        out[i] = (x, y)
    return out


def pm_type_iii_oracle(n, x0=1e-3, eps=1e-3, a=1.0):
    """Pomeau-Manneville Type-III flip normal-form orbit."""
    n = _positive_int(n, "n")
    x0 = _finite_float(x0, "x0")
    eps = _finite_float(eps, "eps")
    a = _finite_float(a, "a")
    if _RUST_AVAILABLE and _pm_type_iii_oracle_rs is not None:
        return np.asarray(_pm_type_iii_oracle_rs(n, x0, eps, a), dtype=np.float64)
    return _pm_type_iii_oracle_python(n, x0, eps, a)


def _pm_type_iii_oracle_python(n, x0, eps, a):
    x = x0
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = -(1.0 + eps) * x - a * x * x * x
        out[i] = x
    return out


def on_off_oracle(n, x0=1e-6, transverse_lyapunov=0.0, noise_scale=0.25, seed=0):
    """On-off intermittency skew-product amplitude driven by seeded noise."""
    n = _positive_int(n, "n")
    x0 = _finite_float(x0, "x0")
    transverse_lyapunov = _finite_float(transverse_lyapunov, "transverse_lyapunov")
    noise_scale = _finite_float(noise_scale, "noise_scale")
    rng = np.random.default_rng(seed)
    driver = rng.normal(size=n)
    if _RUST_AVAILABLE and _on_off_oracle_rs is not None:
        return np.asarray(
            _on_off_oracle_rs(driver, x0, transverse_lyapunov, noise_scale), dtype=np.float64
        )
    return _on_off_oracle_python(driver, x0, transverse_lyapunov, noise_scale)


def _on_off_oracle_python(driver, x0, transverse_lyapunov, noise_scale):
    driver = np.asarray(driver, dtype=np.float64)
    if driver.ndim != 1 or driver.size == 0 or not np.all(np.isfinite(driver)):
        raise ValueError("driver must be a non-empty finite 1D array")
    x = x0
    out = np.empty(driver.size, dtype=np.float64)
    for i, eta in enumerate(driver):
        multiplier = np.exp(transverse_lyapunov + noise_scale * eta)
        x = multiplier * x / (1.0 + x * x)
        out[i] = x
    return out


def on_off_skew_logistic_oracle(n, x0=0.217, y0=1e-2, eps=0.499):
    """Canonical logistic-driver on-off intermittency skew-product.

    The driver is the fully chaotic logistic map ``x -> 4*x*(1-x)``. The
    transverse variable follows ``y -> eps * f'(x) * y`` with bounded nonlinear
    saturation, so the transverse Lyapunov exponent is
    ``lambda_perp = log(eps) + log(2)`` and the blowout onset is
    ``ON_OFF_SKEW_LOGISTIC_ONSET = 1/2``. Values just below onset, roughly
    ``eps in [0.45, 0.499]``, produce on-off bursting from small nonzero ``y0``.
    """
    n = _positive_int(n, "n")
    x0 = _finite_float(x0, "x0")
    y0 = _finite_float(y0, "y0")
    eps = _finite_float(eps, "eps")
    if _RUST_AVAILABLE and _on_off_skew_logistic_oracle_rs is not None:
        return np.asarray(_on_off_skew_logistic_oracle_rs(n, x0, y0, eps), dtype=np.float64)
    return _on_off_skew_logistic_oracle_python(n, x0, y0, eps)


def _on_off_skew_logistic_oracle_python(n, x0, y0, eps):
    x = x0
    y = y0
    out = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        driver = 4.0 * x * (1.0 - x)
        multiplier = 4.0 * eps * (1.0 - 2.0 * x)
        y = multiplier * y / (1.0 + y * y)
        x = driver
        out[i] = (x, y)
    return out


def logistic_type_i_oracle(n, x0=0.2, r=None):
    """Logistic-map Type-I intermittency oracle near ``r_c = 1 + sqrt(8)``."""
    n = _positive_int(n, "n")
    x0 = _finite_float(x0, "x0")
    r = LOGISTIC_TYPE_I_ONSET - 1e-4 if r is None else _finite_float(r, "r")
    if _RUST_AVAILABLE and _logistic_type_i_oracle_rs is not None:
        return np.asarray(_logistic_type_i_oracle_rs(n, x0, r), dtype=np.float64)
    return _logistic_type_i_oracle_python(n, x0, r)


def _logistic_type_i_oracle_python(n, x0, r):
    x = x0
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = r * x * (1.0 - x)
        out[i] = x
    return out


def lorenz_1662_oracle(x0=(0.0, 1.0, 1.05), t_span=(0.0, 80.0), dt=0.01, t_transient=20.0):
    """Lorenz trajectory at the classical intermittent ``rho=166.2`` setting."""
    return lorenz_trajectory(
        x0=x0,
        t_span=t_span,
        dt=dt,
        t_transient=t_transient,
        rho=LORENZ_INTERMITTENCY_RHO,
    )


__all__ = [
    "LOGISTIC_TYPE_I_ONSET",
    "LORENZ_INTERMITTENCY_RHO",
    "ON_OFF_SKEW_LOGISTIC_ONSET",
    "logistic_type_i_oracle",
    "lorenz_1662_oracle",
    "on_off_oracle",
    "on_off_skew_logistic_oracle",
    "pm_type_i_oracle",
    "pm_type_ii_oracle",
    "pm_type_iii_oracle",
]
