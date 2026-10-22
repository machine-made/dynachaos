"""Co-moving Lyapunov exponent for coupled map lattices.

Implements the co-moving Lyapunov exponent lambda(v) that measures the
growth rate of perturbations in a reference frame moving at velocity v
through the lattice.  This diagnostic distinguishes convective from
absolute instability in spatiotemporal chaos (Deissler & Kaneko, 1987).

For each velocity v, a localized perturbation delta is initialized at the
center of the lattice and evolved under the linearized CML dynamics.  At
time n the perturbation amplitude is sampled at site i = i_0 + floor(v*n),
giving the co-moving growth rate:

    lambda(v) = lim_{n->inf} (1/n) ln |delta_n(floor(v*n))|

Positive lambda(v) for all v indicates absolute instability (fully
developed turbulence), while lambda(v) < 0 for large |v| but > 0 near
v = 0 indicates convective instability.

Usage
-----
::

    from dynachaos.diagnostics.comoving_lyapunov import comoving_lyapunov_spectrum
    lambda_v = comoving_lyapunov_spectrum(f, df, g, dg, eps, N, v_values,
                                          n_iter, n_transient)
"""

import os

import numpy as np

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import comoving_lyapunov_logistic as _comoving_lyapunov_logistic_rs

    _RUST_AVAILABLE = True
except ImportError:
    _comoving_lyapunov_logistic_rs = None
    _RUST_AVAILABLE = False


def _initial_state(N, x_init=None):
    if x_init is None:
        rng = np.random.default_rng(42)
        return rng.uniform(-1.0, 1.0, N)

    x_init = np.asarray(x_init, dtype=np.float64)
    if x_init.shape != (N,):
        raise ValueError(f"x_init must have shape ({N},)")
    return np.ascontiguousarray(x_init)


def comoving_lyapunov_spectrum(f, df, g, dg, eps, N, v_values, n_iter, n_transient):
    """Compute co-moving Lyapunov exponent lambda(v) for a CML.

    The co-moving Lyapunov exponent measures the growth rate of perturbations
    in a reference frame moving at velocity v through the lattice.

    For each velocity v, track a perturbation delta initialized at site i=N//2,
    evolved under the linearized CML dynamics::

        delta_{n+1}(i) = df(x_n(i)) * delta_n(i)
            + (eps/2) * [dg(x_n(i+1)) * delta_n(i+1)
                       + dg(x_n(i-1)) * delta_n(i-1)
                       - 2 * dg(x_n(i)) * delta_n(i)]

    At time n, evaluate the perturbation at site i = i_0 + floor(v*n)
    (co-moving frame).

        lambda(v) = lim (1/n) ln |delta_n(floor(v*n))|

    Parameters
    ----------
    f : callable
        Local map f(x) applied element-wise to array of site values.
    df : callable
        Derivative of f, applied element-wise.
    g : callable
        Coupling function g(x).
    dg : callable
        Derivative of g.
    eps : float
        Coupling strength.
    N : int
        Number of lattice sites.
    v_values : array_like
        Velocities at which to evaluate lambda(v).
    n_iter : int
        Number of iterations for averaging.
    n_transient : int
        Transient iterations before measuring.

    Returns
    -------
    lambda_v : ndarray, shape (len(v_values),)
        Co-moving Lyapunov exponent at each velocity.
    """
    x_init = _initial_state(N)
    return _comoving_lyapunov_spectrum_python(
        f,
        df,
        g,
        dg,
        eps,
        x_init,
        v_values,
        n_iter,
        n_transient,
    )


def comoving_lyapunov_spectrum_logistic(
    *,
    a,
    eps,
    N,
    v_values,
    n_iter,
    n_transient,
    x_init=None,
):
    """Compute co-moving Lyapunov exponents for logistic CML with g=f."""
    from dynachaos.maps.primitives import logistic, logistic_derivative

    x_init = _initial_state(N, x_init)
    v_values = np.asarray(v_values, dtype=np.float64)

    if _RUST_AVAILABLE and _comoving_lyapunov_logistic_rs is not None:
        lambda_v = np.asarray(
            _comoving_lyapunov_logistic_rs(
                x_init,
                np.ascontiguousarray(v_values, dtype=np.float64),
                float(a),
                float(eps),
                int(n_iter),
                int(n_transient),
            )
        )
        _print_progress(v_values, lambda_v)
        return lambda_v

    return _comoving_lyapunov_spectrum_python(
        lambda x: logistic(x, a),
        lambda x: logistic_derivative(x, a),
        lambda x: logistic(x, a),
        lambda x: logistic_derivative(x, a),
        eps,
        x_init,
        v_values,
        n_iter,
        n_transient,
    )


def _comoving_lyapunov_spectrum_python(f, df, g, dg, eps, x_init, v_values, n_iter, n_transient):
    v_values = np.asarray(v_values, dtype=np.float64)
    n_vel = len(v_values)
    N = len(x_init)
    center = N // 2

    x = np.asarray(x_init, dtype=np.float64).copy()
    for _ in range(n_transient):
        fx = f(x)
        gx = g(x)
        gx_left = np.roll(gx, -1)
        gx_right = np.roll(gx, 1)
        x = fx + (eps / 2.0) * (gx_left + gx_right - 2.0 * gx)

    # Save the attractor state so we can reset for each velocity
    x_attractor = x.copy()

    # Use a segment approach: divide n_iter into independent segments.
    # Each segment starts with a fresh localized perturbation and
    # evaluates the co-moving amplitude at the segment end.  This avoids
    # the pitfall of accumulating renormalization factors across a single
    # very long trajectory (which conflates global norm growth with
    # site-specific co-moving growth).
    segment_length = min(100, n_iter)
    n_segments = max(n_iter // segment_length, 1)
    renorm_interval = 10

    lambda_v = np.zeros(n_vel, dtype=np.float64)

    for iv, v in enumerate(v_values):
        # Reset CML state to attractor
        x = x_attractor.copy()

        total_log_growth = 0.0
        valid_segments = 0

        for _seg in range(n_segments):
            # Fresh localized perturbation at center for each segment
            delta = np.zeros(N, dtype=np.float64)
            delta[center] = 1.0
            renorm_accum = 0.0

            for n in range(segment_length):
                # Store old state for linearized step
                x_old = x.copy()

                # Evolve x one nonlinear CML step
                fx = f(x_old)
                gx = g(x_old)
                gx_left = np.roll(gx, -1)
                gx_right = np.roll(gx, 1)
                x = fx + (eps / 2.0) * (gx_left + gx_right - 2.0 * gx)

                # Evolve delta one linearized step
                df_x = df(x_old)
                dg_x = dg(x_old)
                dg_left = np.roll(dg_x * delta, -1)
                dg_right = np.roll(dg_x * delta, 1)
                delta = df_x * delta + (eps / 2.0) * (dg_left + dg_right - 2.0 * dg_x * delta)

                # Renormalize periodically to prevent overflow
                if (n + 1) % renorm_interval == 0:
                    max_delta = np.max(np.abs(delta))
                    if max_delta > 0:
                        renorm_accum += np.log(max_delta)
                        delta /= max_delta

            # Evaluate at co-moving site at end of segment
            site_offset = int(np.floor(v * segment_length))
            site_index = (center + site_offset) % N
            amp = np.abs(delta[site_index])
            if amp > 0:
                total_log_growth += np.log(amp) + renorm_accum
                valid_segments += 1

        if valid_segments > 0:
            lambda_v[iv] = total_log_growth / (valid_segments * segment_length)
        else:
            lambda_v[iv] = -10.0

        if (iv + 1) % 50 == 0 or iv == 0:
            _print_one_progress(v, lambda_v[iv], iv, n_vel)

    return lambda_v


def _print_progress(v_values, lambda_v):
    n_vel = len(v_values)
    for iv, v in enumerate(v_values):
        if (iv + 1) % 50 == 0 or iv == 0:
            _print_one_progress(v, lambda_v[iv], iv, n_vel)


def _print_one_progress(v, value, iv, n_vel):
    print(f"  v={v:+.2f}: lambda={value:.4f}  [{iv + 1}/{n_vel}]")
