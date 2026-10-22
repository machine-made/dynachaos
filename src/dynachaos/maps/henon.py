"""Henon map and Jacobian.

The classical two-dimensional quadratic map:

    x' = 1 - a x^2 + y
    y' = b x

At the canonical parameters (a=1.4, b=0.3) the map exhibits a strange
attractor with correlation dimension D_2 ~ 1.22 and Lyapunov exponents
lambda_1 ~ 0.42, lambda_2 ~ -1.62.

Reference
---------
Henon, M. (1976) "A two-dimensional mapping with a strange attractor",
  Commun. Math. Phys. 50, 69-77.
"""

import numpy as np


def henon(state, a=1.4, b=0.3):
    """One iteration of the Henon map.

    Parameters
    ----------
    state : array_like, shape (2,)
        Current state [x, y].
    a, b : float
        Map parameters.

    Returns
    -------
    ndarray, shape (2,)
    """
    x, y = state
    return np.array([1.0 - a * x * x + y, b * x])


def henon_jac(state, a=1.4, b=0.3):
    """Jacobian of the Henon map.

    J = [[-2ax, 1],
         [  b,  0]]

    Parameters
    ----------
    state : array_like, shape (2,)
    a, b : float

    Returns
    -------
    ndarray, shape (2, 2)
    """
    x, _y = state
    return np.array([[-2.0 * a * x, 1.0], [b, 0.0]])
