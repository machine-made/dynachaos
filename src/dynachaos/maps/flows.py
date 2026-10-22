"""Continuous-time chaotic flows and a delay-differential equation.

Contains the Lorenz (1963), Rossler (1976), and Mackey-Glass (1977) systems
with RHS, Jacobian (where applicable), and trajectory helper functions.

Each ``*_trajectory`` function returns a post-transient array of shape (N, d)
suitable for embedding analysis and Lyapunov computation.

References
----------
Lorenz, E. N. (1963) "Deterministic nonperiodic flow",
  J. Atmos. Sci. 20(2), 130-141.
Rossler, O. E. (1976) "An equation for continuous chaos",
  Phys. Lett. A 57(5), 397-398.
Mackey, M. C. & Glass, L. (1977) "Oscillation and chaos in physiological
  control systems", Science 197(4300), 287-289.
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── Lorenz system ──────────────────────────────────────────────────────────


def lorenz_rhs(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Right-hand side of the Lorenz equations.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    x, y, z = state
    return np.array(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]
    )


def lorenz_jac(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Jacobian of the Lorenz equations.

    [[-sigma,  sigma,    0 ],
     [rho-z,    -1,     -x ],
     [  y,       x,   -beta]]
    """
    x, y, z = state
    return np.array(
        [
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta],
        ]
    )


def lorenz_trajectory(x0=(1.0, 1.0, 1.0), t_span=(0, 100), dt=0.01, t_transient=20.0, **params):
    """Integrate the Lorenz system and return post-transient trajectory.

    Parameters
    ----------
    x0 : array_like, shape (3,)
        Initial condition.
    t_span : tuple
        (t_start, t_end) integration interval.
    dt : float
        Output sampling interval.
    t_transient : float
        Time to discard as transient (must be < t_span[1] - t_span[0]).
    **params
        Passed to ``lorenz_rhs`` (sigma, rho, beta).

    Returns
    -------
    ndarray, shape (N, 3)
        Post-transient trajectory sampled at interval dt.
    """
    t_end = t_span[1]
    t_eval = np.arange(t_span[0], t_end, dt)

    def rhs(t, y):
        return lorenz_rhs(t, y, **params)

    sol = solve_ivp(
        rhs, t_span, x0, method="RK45", t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=dt
    )
    traj = sol.y.T  # (N, 3)

    # Discard transient
    mask = sol.t >= t_transient
    return traj[mask]


# ── Rossler system ─────────────────────────────────────────────────────────


def rossler_rhs(t, state, a=0.2, b=0.2, c=5.7):
    """Right-hand side of the Rossler equations.

    dx/dt = -y - z
    dy/dt = x + a * y
    dz/dt = b + z * (x - c)
    """
    x, y, z = state
    return np.array(
        [
            -y - z,
            x + a * y,
            b + z * (x - c),
        ]
    )


def rossler_jac(t, state, a=0.2, b=0.2, c=5.7):
    """Jacobian of the Rossler equations.

    [[ 0,  -1,  -1 ],
     [ 1,   a,   0 ],
     [ z,   0, x-c ]]
    """
    x, _y, z = state
    return np.array(
        [
            [0.0, -1.0, -1.0],
            [1.0, a, 0.0],
            [z, 0.0, x - c],
        ]
    )


def rossler_trajectory(x0=(1.0, 1.0, 0.0), t_span=(0, 500), dt=0.05, t_transient=100.0, **params):
    """Integrate the Rossler system and return post-transient trajectory.

    Parameters
    ----------
    x0 : array_like, shape (3,)
    t_span : tuple
    dt : float
    t_transient : float
    **params
        Passed to ``rossler_rhs`` (a, b, c).

    Returns
    -------
    ndarray, shape (N, 3)
    """
    t_end = t_span[1]
    t_eval = np.arange(t_span[0], t_end, dt)

    def rhs(t, y):
        return rossler_rhs(t, y, **params)

    sol = solve_ivp(
        rhs, t_span, x0, method="RK45", t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=dt
    )
    traj = sol.y.T

    mask = sol.t >= t_transient
    return traj[mask]


# ── Mackey-Glass DDE ───────────────────────────────────────────────────────


def mackey_glass_series(
    n_points=10_000, dt=1.0, tau=17, beta_mg=0.2, gamma=0.1, n=10, t_transient=500
):
    """Generate a Mackey-Glass time series via Euler integration.

    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    Uses internal step dt_internal=0.1, subsampled to output dt.
    Matches the approach of Farmer (1982) and the nolitsa package.

    Parameters
    ----------
    n_points : int
        Number of output points after transient.
    dt : float
        Output sampling interval.
    tau : int
        Delay in units of dt.
    beta_mg : float
        Production rate (called beta_mg to avoid shadowing).
    gamma : float
        Decay rate.
    n : int
        Nonlinearity exponent.
    t_transient : float
        Transient time to discard (in time units).

    Returns
    -------
    ndarray, shape (n_points,)
    """
    dt_internal = 0.1
    subsample = max(1, int(dt / dt_internal))
    dt_internal = dt / subsample  # exact subdivision

    tau_steps = int(tau * dt / dt_internal)
    n_transient_steps = int(t_transient / dt_internal)
    n_total_steps = n_transient_steps + n_points * subsample

    # History buffer: need tau_steps of past values
    history_len = tau_steps + 1
    x_hist = np.ones(history_len) * 1.2  # constant initial history

    # Euler integration
    output = np.empty(n_points)
    out_idx = 0
    x = x_hist[-1]

    for step in range(n_total_steps):
        # In a ring buffer of size tau_steps + 1, the oldest entry
        # (= tau_steps steps ago) sits at the same index we will overwrite.
        idx = step % history_len
        x_delayed = x_hist[idx]

        # Mackey-Glass RHS
        dxdt = beta_mg * x_delayed / (1.0 + x_delayed**n) - gamma * x

        x_new = x + dt_internal * dxdt
        x_hist[idx] = x_new  # overwrite oldest with newest
        x = x_new

        # Record post-transient, subsampled
        if step >= n_transient_steps and (step - n_transient_steps) % subsample == 0:
            if out_idx < n_points:
                output[out_idx] = x
                out_idx += 1

    return output[:out_idx]
