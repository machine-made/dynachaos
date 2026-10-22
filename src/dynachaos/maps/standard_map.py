"""Chirikov standard map and analytic Jacobian.

The standard map is an area-preserving map on the two-torus:

    p' = p + K sin(theta)
    theta' = theta + p'

Both coordinates are returned modulo ``2*pi``.
"""

import numpy as np


def standard_map(state, K=1.0):
    """One iteration of the Chirikov standard map."""
    theta, p = state
    p_next = p + K * np.sin(theta)
    theta_next = theta + p_next
    return np.mod(np.array([theta_next, p_next], dtype=np.float64), 2.0 * np.pi)


def standard_map_jac(state, K=1.0):
    """Jacobian of the standard map at ``state``."""
    theta, _p = state
    kcos = K * np.cos(theta)
    return np.array([[1.0 + kcos, 1.0], [kcos, 1.0]], dtype=np.float64)
