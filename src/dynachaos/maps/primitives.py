"""Primitive map functions shared across dynachaos modules."""

from dynachaos.maps.delayed_logistic import delayed_logistic, delayed_logistic_jac


def logistic(x, a):
    """Logistic map f(x) = 1 - ax^2. Vectorized."""
    return 1.0 - a * x * x


def logistic_derivative(x, a):
    """Derivative f'(x) = -2ax."""
    return -2.0 * a * x


__all__ = [
    "logistic",
    "logistic_derivative",
    "delayed_logistic",
    "delayed_logistic_jac",
]
