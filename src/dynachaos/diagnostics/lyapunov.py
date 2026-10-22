"""Lyapunov exponent and spectrum computation via QR decomposition.

Implements the standard algorithm of Benettin et al. (1980) for computing the
full Lyapunov spectrum of discrete maps and continuous-time flows.  The approach
evolves a set of orthonormal tangent vectors alongside the trajectory,
periodically re-orthonormalising via QR decomposition to prevent collapse onto
the most unstable direction.

Usage
-----
For a 1D map with known derivative::

    lam = lyapunov_exponent_1d(f, df, x0, n_iter=100_000)

For an N-dimensional map with Jacobian::

    spectrum = lyapunov_spectrum(f, jac, x0, n_iter=100_000)

For a continuous-time flow with known Jacobian::

    spectrum = flow_lyapunov_spectrum(rhs, jac, x0, t_total=200)

For sweeping a parameter and computing the maximal exponent::

    lams = lyapunov_sweep_1d(f, df, x0_func, params, n_iter=50_000,
                             n_transient=10_000)
"""

import numpy as np
from scipy.integrate import solve_ivp

from dynachaos.diagnostics._validation import nonnegative_int, positive_int


def lyapunov_exponent_1d(f, df, x0, n_iter=100_000, n_transient=10_000):
    """Compute the Lyapunov exponent of a 1D map.

    Parameters
    ----------
    f : callable
        The map x_{n+1} = f(x_n).
    df : callable
        The derivative f'(x).
    x0 : float
        Initial condition.
    n_iter : int
        Number of iterations for accumulation (after transient).
    n_transient : int
        Number of transient iterations to discard.

    Returns
    -------
    float
        The Lyapunov exponent.
    """
    n_iter = positive_int(n_iter, "n_iter")
    n_transient = nonnegative_int(n_transient, "n_transient")
    x = x0
    # Transient
    for _ in range(n_transient):
        x = f(x)

    # Accumulate
    log_sum = 0.0
    for _ in range(n_iter):
        deriv = abs(df(x))
        if deriv > 0:
            log_sum += np.log(deriv)
        else:
            log_sum += np.log(1e-300)  # ≈ -690.8; consistent with 1e-300 floor in spectrum funcs
        x = f(x)

    return log_sum / n_iter


def lyapunov_spectrum(
    f, jac, x0, n_iter=100_000, n_transient=10_000, reorth_interval=1, return_convergence=False
):
    """Compute the full Lyapunov spectrum of an N-dimensional map.

    Uses QR decomposition (Benettin et al. 1980) to track all N Lyapunov
    exponents simultaneously.

    Parameters
    ----------
    f : callable
        The map x_{n+1} = f(x_n), where x is a 1D array of length N.
    jac : callable
        The Jacobian matrix J(x), returns (N, N) array.
    x0 : array_like
        Initial condition, shape (N,).
    n_iter : int
        Iterations for accumulation.
    n_transient : int
        Transient iterations to discard.
    reorth_interval : int
        QR re-orthonormalisation interval.  1 means every step (most
        accurate); larger values trade accuracy for speed.
    return_convergence : bool
        If True, also return a convergence-error estimate computed as the
        population standard deviation of running exponent estimates sampled at
        approximately ten checkpoints from halfway through the accumulation to
        the end.

    Returns
    -------
    spectrum : ndarray, shape (N,), or tuple of ndarray
        Lyapunov exponents in descending order.
        If return_convergence is True, returns ``(spectrum, conv_err)``, where
        ``conv_err`` is reordered to match the descending spectrum.
    """
    n_iter = positive_int(n_iter, "n_iter")
    n_transient = nonnegative_int(n_transient, "n_transient")
    x = np.asarray(x0, dtype=np.float64)
    dim = len(x)

    # Transient
    for _ in range(n_transient):
        x = f(x)

    # Initialise orthonormal frame
    Q = np.eye(dim, dtype=np.float64)
    log_sums = np.zeros(dim, dtype=np.float64)
    count = 0
    checkpoint_iters = np.unique(np.linspace(n_iter // 2, n_iter, 10, dtype=int))
    checkpoint_iters = checkpoint_iters[checkpoint_iters > 0]
    checkpoint_values = []

    for i in range(n_iter):
        J = jac(x)
        Q = J @ Q
        x = f(x)

        if (i + 1) % reorth_interval == 0:
            Q, R = np.linalg.qr(Q)
            # Accumulate log of diagonal (absolute value)
            diag = np.abs(np.diag(R))
            # Guard against zero
            diag = np.where(diag > 0, diag, 1e-300)
            log_sums += np.log(diag)
            count += 1
            if return_convergence and (i + 1) in checkpoint_iters:
                checkpoint_values.append(log_sums / (count * reorth_interval))

    spectrum = log_sums / (count * reorth_interval)
    # Sort descending
    order = np.argsort(spectrum)[::-1]
    spectrum_sorted = spectrum[order]
    if not return_convergence:
        return spectrum_sorted
    if checkpoint_values:
        conv_err = np.std(np.asarray(checkpoint_values), axis=0, ddof=0)
    else:
        conv_err = np.zeros(dim, dtype=np.float64)
    return spectrum_sorted, conv_err[order]


def lyapunov_max(f, jac, x0, n_iter=100_000, n_transient=10_000, rng=None):
    """Compute only the maximal Lyapunov exponent of an N-dimensional map.

    More efficient than computing the full spectrum when only the largest
    exponent is needed — tracks a single tangent vector.

    Parameters
    ----------
    f : callable
        The map x_{n+1} = f(x_n).
    jac : callable
        The Jacobian J(x), returns (N, N) array.
    x0 : array_like
        Initial condition.
    n_iter : int
        Iterations for accumulation.
    n_transient : int
        Transient iterations to discard.
    rng : numpy.random.Generator or None
        For reproducibility.  Defaults to ``default_rng(42)``.

    Returns
    -------
    float
        The maximal Lyapunov exponent.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    x = np.asarray(x0, dtype=np.float64)
    dim = len(x)

    for _ in range(n_transient):
        x = f(x)

    # Random unit tangent vector
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)

    log_sum = 0.0
    for _ in range(n_iter):
        J = jac(x)
        v = J @ v
        norm_v = np.linalg.norm(v)
        if norm_v > 0:
            log_sum += np.log(norm_v)
            v /= norm_v
        else:
            log_sum += -100.0
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v)
        x = f(x)

    return log_sum / n_iter


def lyapunov_sweep_1d(f, df, x0_func, params, n_iter=50_000, n_transient=10_000):
    """Sweep a parameter and compute the Lyapunov exponent at each value.

    Parameters
    ----------
    f : callable
        f(x, p) — the map, parameterised by p.
    df : callable
        df(x, p) — the derivative with respect to x.
    x0_func : callable
        x0_func(p) — initial condition as a function of the parameter.
    params : array_like
        1D array of parameter values to sweep.
    n_iter : int
        Iterations for accumulation at each parameter.
    n_transient : int
        Transient iterations at each parameter.

    Returns
    -------
    lams : ndarray
        Lyapunov exponent at each parameter value.
    """
    params = np.asarray(params)
    lams = np.empty(len(params))

    for i, p in enumerate(params):

        def fp(x, _p=p):
            return f(x, _p)

        def dfp(x, _p=p):
            return df(x, _p)

        x0 = x0_func(p)
        lams[i] = lyapunov_exponent_1d(fp, dfp, x0, n_iter, n_transient)

    return lams


def lyapunov_sweep_nd(
    f, jac, x0_func, params, n_iter=50_000, n_transient=10_000, full_spectrum=False
):
    """Sweep a parameter and compute Lyapunov exponent(s) for an ND map.

    Parameters
    ----------
    f : callable
        f(x, p) — the map, parameterised by p.
    jac : callable
        jac(x, p) — the Jacobian with respect to x.
    x0_func : callable
        x0_func(p) — initial condition as a function of the parameter.
    params : array_like
        1D array of parameter values.
    n_iter : int
        Iterations for accumulation.
    n_transient : int
        Transient iterations.
    full_spectrum : bool
        If True, return the full spectrum at each parameter value.

    Returns
    -------
    result : ndarray
        If full_spectrum is False: shape (len(params),) — maximal exponent.
        If full_spectrum is True: shape (len(params), N) — full spectrum.
    """
    params = np.asarray(params)

    results = []
    for p in params:

        def fp(x, _p=p):
            return f(x, _p)

        def jacp(x, _p=p):
            return jac(x, _p)

        x0 = x0_func(p)

        if full_spectrum:
            spec = lyapunov_spectrum(fp, jacp, x0, n_iter, n_transient)
            results.append(spec)
        else:
            lam = lyapunov_max(fp, jacp, x0, n_iter, n_transient)
            results.append(lam)

    return np.array(results)


def flow_lyapunov_spectrum(rhs, jac, x0, t_total=200.0, dt=0.01, t_transient=50.0, reorth_dt=1.0):
    """Lyapunov spectrum of a continuous-time flow.

    Integrates the variational equations dPhi/dt = J(x(t)) Phi alongside
    the flow dx/dt = rhs(t, x), with periodic QR reorthonormalization
    (Benettin et al. 1980, adapted for flows).

    Parameters
    ----------
    rhs : callable
        Right-hand side f(t, state) -> array of shape (dim,).
    jac : callable
        Jacobian J(t, state) -> array of shape (dim, dim).
    x0 : array_like, shape (dim,)
        Initial condition on the attractor (or near it).
    t_total : float
        Integration time after transient for accumulation.
    dt : float
        Maximum integration step size.
    t_transient : float
        Transient time to discard before accumulation.
    reorth_dt : float
        QR reorthonormalization interval (seconds).

    Returns
    -------
    spectrum : ndarray, shape (dim,)
        Lyapunov exponents in descending order (units: 1/time).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    dim = len(x0)

    # ── Build augmented ODE: state = [x (dim), Phi_flat (dim^2)] ──
    def augmented_rhs(t, y):
        x = y[:dim]
        Phi = y[dim:].reshape(dim, dim)
        dx = rhs(t, x)
        J = jac(t, x)
        dPhi = (J @ Phi).ravel()
        return np.concatenate([dx, dPhi])

    # ── Transient: integrate flow only, discard ──
    if t_transient > 0:
        sol_trans = solve_ivp(
            lambda t, x: rhs(t, x),
            (0, t_transient),
            x0,
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
            max_step=dt,
        )
        x0 = sol_trans.y[:, -1]

    # ── Accumulation phase ──
    Q = np.eye(dim, dtype=np.float64)
    log_sums = np.zeros(dim, dtype=np.float64)
    n_reorth = int(t_total / reorth_dt)

    if n_reorth == 0:
        raise ValueError(
            f"t_total={t_total} < reorth_dt={reorth_dt}: "
            "at least one reorthogonalization interval is required"
        )

    x = x0.copy()
    t_current = t_transient if t_transient > 0 else 0.0

    for _ in range(n_reorth):
        # Augmented initial condition
        y0 = np.concatenate([x, Q.ravel()])
        t_end = t_current + reorth_dt

        sol = solve_ivp(
            augmented_rhs,
            (t_current, t_end),
            y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
            max_step=dt,
        )

        # Extract final state
        y_final = sol.y[:, -1]
        x = y_final[:dim]
        Phi = y_final[dim:].reshape(dim, dim)

        # QR decomposition
        Q, R = np.linalg.qr(Phi)
        diag = np.abs(np.diag(R))
        diag = np.where(diag > 0, diag, 1e-300)
        log_sums += np.log(diag)

        t_current = t_end

    actual_time = n_reorth * reorth_dt
    spectrum = log_sums / actual_time
    return np.sort(spectrum)[::-1]
