"""Base class for discrete dynamical maps.

Wraps a map function with convenience methods for trajectory generation,
Lyapunov exponent computation, and bifurcation analysis.

Usage
-----
    from dynachaos.maps.base import LogisticMap

    lm = LogisticMap(a=1.99)
    traj = lm.trajectory(x0=0.1, n_iter=1000)
    lam = lm.lyapunov(x0=0.1, n_iter=100_000)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BifurcationData:
    """Container for bifurcation diagram data."""

    param_values: npt.NDArray[np.float64]
    attractors: list[npt.NDArray[np.float64]]


class DiscreteMap:
    """A discrete-time dynamical map with optional Jacobian.

    Parameters
    ----------
    f : callable
        The map function.  For 1D: ``f(x) -> x``.
        For ND: ``f(x) -> x`` where x is a 1D array.
    df : callable or None
        Derivative (1D) or Jacobian (ND).
    name : str
        Human-readable name.
    dim : int
        Phase-space dimension.
    """

    def __init__(
        self,
        f: Callable,
        df: Callable | None = None,
        name: str = "",
        dim: int = 1,
    ):
        self.f = f
        self.df = df
        self.name = name
        self.dim = dim

    def __repr__(self) -> str:
        return f"DiscreteMap({self.name!r}, dim={self.dim})"

    def trajectory(
        self,
        x0: float | npt.NDArray,
        n_iter: int,
        n_transient: int = 0,
    ) -> npt.NDArray[np.float64]:
        """Iterate the map and return the trajectory.

        Parameters
        ----------
        x0 : float or ndarray
            Initial condition.
        n_iter : int
            Number of iterates to record.
        n_transient : int
            Transient iterates to discard.

        Returns
        -------
        ndarray, shape (n_iter,) for 1D or (n_iter, dim) for ND.
        """
        x = np.asarray(x0, dtype=np.float64)

        for _ in range(n_transient):
            x = self.f(x)

        shape = n_iter if self.dim == 1 else (n_iter, self.dim)
        out = np.empty(shape, dtype=np.float64)
        for i in range(n_iter):
            out[i] = x
            x = self.f(x)

        return out

    def lyapunov(
        self,
        x0: float | npt.NDArray,
        n_iter: int = 100_000,
        n_transient: int = 10_000,
    ) -> float | npt.NDArray[np.float64]:
        """Compute the Lyapunov exponent(s).

        Requires ``df`` (derivative/Jacobian) to be set.

        Returns
        -------
        float for 1D maps, ndarray for ND maps (full spectrum).
        """
        if self.df is None:
            raise ValueError(f"No derivative/Jacobian provided for {self.name!r}")

        if self.dim == 1:
            from dynachaos.diagnostics.lyapunov import lyapunov_exponent_1d

            return lyapunov_exponent_1d(self.f, self.df, x0, n_iter, n_transient)
        else:
            from dynachaos.diagnostics.lyapunov import lyapunov_spectrum

            return lyapunov_spectrum(self.f, self.df, x0, n_iter, n_transient)

    def bifurcation(
        self,
        param_values: npt.NDArray[np.float64],
        make_map: Callable,
        x0: float | npt.NDArray = 0.5,
        n_transient: int = 500,
        n_record: int = 200,
    ) -> BifurcationData:
        """Generate bifurcation diagram data.

        Parameters
        ----------
        param_values : ndarray
            1D array of parameter values to sweep.
        make_map : callable
            ``make_map(p)`` returns a callable ``f(x)`` for parameter ``p``.
        x0 : float or ndarray
            Initial condition (re-used at each parameter).
        n_transient : int
            Transient iterates at each parameter.
        n_record : int
            Iterates to record at each parameter.

        Returns
        -------
        BifurcationData
        """
        extract = float if self.dim == 1 else (lambda v: v[0])
        attractors = []
        for p in param_values:
            f_p = make_map(p)
            x = np.asarray(x0, dtype=np.float64).copy()
            for _ in range(n_transient):
                x = f_p(x)
            pts = np.empty(n_record, dtype=np.float64)
            for i in range(n_record):
                pts[i] = extract(x)
                x = f_p(x)
            attractors.append(pts)

        return BifurcationData(
            param_values=np.asarray(param_values),
            attractors=attractors,
        )


# ── Pre-built map instances ──────────────────────────────────────────────────


def _make_logistic_map(a: float = 1.99) -> DiscreteMap:
    """Create a logistic map x → 1 - ax² with given parameter."""
    from dynachaos.maps.primitives import logistic, logistic_derivative

    return DiscreteMap(
        f=lambda x: logistic(x, a),
        df=lambda x: logistic_derivative(x, a),
        name=f"Logistic(a={a})",
        dim=1,
    )


def _make_circle_map(omega: float = 0.0, K: float = 1.0) -> DiscreteMap:
    """Create a circle map θ → θ + ω - (K/2π)sin(2πθ)."""
    from dynachaos.maps.circle_map import circle_map, circle_map_derivative

    return DiscreteMap(
        f=lambda x: circle_map(x, omega, K),
        df=lambda x: circle_map_derivative(x, K),
        name=f"Circle(ω={omega}, K={K})",
        dim=1,
    )


def _make_henon_map(a: float = 1.4, b: float = 0.3) -> DiscreteMap:
    """Create a Henon map x' = 1 - ax^2 + y, y' = bx."""
    from dynachaos.maps.henon import henon, henon_jac

    return DiscreteMap(
        f=lambda state: henon(state, a, b),
        df=lambda state: henon_jac(state, a, b),
        name=f"Henon(a={a}, b={b})",
        dim=2,
    )


def _make_standard_map(K: float = 1.0) -> DiscreteMap:
    """Create the area-preserving Chirikov standard map."""
    from dynachaos.maps.standard_map import standard_map, standard_map_jac

    return DiscreteMap(
        f=lambda state: standard_map(state, K),
        df=lambda state: standard_map_jac(state, K),
        name=f"StandardMap(K={K})",
        dim=2,
    )


# Convenience constructors
LogisticMap = _make_logistic_map
CircleMap = _make_circle_map
HenonMap = _make_henon_map
StandardMap = _make_standard_map
