"""Multifractal diagnostics (global spectrum + local tile maps).

Implements a practical multifractal workflow based on partition moments:

1. Build box probabilities p_i(r) over box scales r.
2. Compute Z_q(r) = sum_i p_i(r)^q.
3. Regress ln Z_q vs ln r to obtain tau(q), then D_q = tau(q)/(q-1).
4. Build alpha and f(alpha) using a Legendre transform of tau(q).
5. For local analysis, apply the same pipeline independently per tile and
   summarize local multifractality by Phi = std(alpha).

References
----------
Mukherjee, S. et al. (2024), Phys. Rev. Lett. 132, 184002.
Grassberger, P. & Procaccia, I. (1983), Phys. Rev. Lett. 50, 346.
"""

from __future__ import annotations

import os

import numpy as np

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import multifractal_moments as _multifractal_moments_rs

    _RUST_AVAILABLE = True
except ImportError:
    _multifractal_moments_rs = None
    _RUST_AVAILABLE = False


def _coerce_field(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError("field must be 1D or 2D")
    if not np.all(np.isfinite(arr)):
        raise ValueError("field must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError("field must be nonnegative")
    if float(arr.sum()) <= 0.0:
        raise ValueError("field must have positive total mass")
    return np.ascontiguousarray(arr)


def _default_box_sizes(shape: tuple[int, int]) -> np.ndarray:
    min_side = min(shape)
    max_box = min_side // 2
    if max_box < 2:
        raise ValueError("field too small for multifractal box scales")
    boxes: list[int] = []
    b = 2
    while b <= max_box:
        boxes.append(b)
        b *= 2
    if len(boxes) < 2:
        raise ValueError("need at least two box sizes for log-log regression")
    return np.asarray(boxes, dtype=np.int64)


def _clean_box_sizes(box_sizes: np.ndarray) -> np.ndarray:
    b = np.asarray(box_sizes, dtype=np.int64).ravel()
    b = b[np.isfinite(b)]
    b = np.unique(b)
    b = b[b > 0]
    if b.size < 2:
        raise ValueError("box_sizes must contain at least two positive scales")
    return b


def _clean_q_values(q_values: np.ndarray | None) -> np.ndarray:
    if q_values is None:
        q = np.linspace(-8.0, 8.0, 65, dtype=np.float64)
    else:
        q = np.asarray(q_values, dtype=np.float64).ravel()
    q = q[np.isfinite(q)]
    q = np.unique(q)
    if q.size < 3:
        raise ValueError("q_values must contain at least three finite values")
    return q


def _linear_fit_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan
    xx = x[mask]
    yy = y[mask]
    slope, intercept = np.polyfit(xx, yy, 1)
    yhat = slope * xx + intercept
    ss_res = float(np.sum((yy - yhat) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    if ss_tot <= 0.0:
        r2 = 1.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(r2)


def _multifractal_moments_python(
    field: np.ndarray,
    box_sizes: np.ndarray,
    q_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = field.shape
    n_scales = len(box_sizes)
    n_q = len(q_values)

    ln_scales = np.full(n_scales, np.nan, dtype=np.float64)
    log_z = np.full((n_scales, n_q), np.nan, dtype=np.float64)
    alpha_num = np.full((n_scales, n_q), np.nan, dtype=np.float64)
    f_num = np.full((n_scales, n_q), np.nan, dtype=np.float64)

    for si, b in enumerate(box_sizes):
        b = int(b)
        n_by = ny // b
        n_bx = nx // b
        if n_by == 0 or n_bx == 0:
            continue
        ln_scales[si] = np.log(float(b))

        # Non-overlapping boxes; truncate edge remainders.
        masses = []
        for by in range(n_by):
            y0 = by * b
            for bx in range(n_bx):
                x0 = bx * b
                mass = float(np.sum(field[y0 : y0 + b, x0 : x0 + b]))
                if mass > 0.0:
                    masses.append(mass)
        if not masses:
            continue
        p = np.asarray(masses, dtype=np.float64)
        used_mass = float(np.sum(p))
        if not np.isfinite(used_mass) or used_mass <= 0.0:
            continue
        p /= used_mass

        for qi, q in enumerate(q_values):
            if np.abs(q - 1.0) < 1e-12:
                shannon = float(np.sum(p * np.log(p)))
                # Z_1(r) = sum_i p_i = 1 by normalization.
                log_z[si, qi] = 0.0
                alpha_num[si, qi] = shannon
                f_num[si, qi] = shannon
                continue

            p_q = p**q
            z = float(np.sum(p_q))
            if not np.isfinite(z) or z <= 0.0:
                continue
            mu = p_q / z
            log_z[si, qi] = np.log(z)
            alpha_num[si, qi] = float(np.sum(mu * np.log(p)))
            # mu can contain tiny values; mask strictly positive to avoid log(0).
            mask = mu > 0.0
            f_num[si, qi] = float(np.sum(mu[mask] * np.log(mu[mask])))

    return log_z, alpha_num, f_num, ln_scales


def multifractal_spectrum(
    field: np.ndarray,
    *,
    box_sizes: np.ndarray | None = None,
    q_values: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | str]:
    """Compute multifractal spectra from a 1D/2D nonnegative field.

    Parameters
    ----------
    field : ndarray
        Input nonnegative measure field. If 1D, it is treated as shape (N, 1).
    box_sizes : array_like of int or None
        Box side lengths. If None, uses dyadic sizes 2, 4, ... up to min(shape)/2.
    q_values : array_like of float or None
        Moment orders q. If None, uses 65 values in [-8, 8].

    Returns
    -------
    dict
        Keys: q, box_sizes, ln_scales, tau, Dq, alpha, f_alpha,
        alpha_legendre, f_legendre, r2_tau, r2_alpha, r2_f, phi, backend.
    """
    arr = _coerce_field(field)
    q = _clean_q_values(q_values)
    if box_sizes is None:
        b = _default_box_sizes(arr.shape)
    else:
        b = _clean_box_sizes(box_sizes)

    if _RUST_AVAILABLE:
        log_z, alpha_num, f_num, ln_scales = _multifractal_moments_rs(arr, b, q)
        log_z = np.asarray(log_z, dtype=np.float64)
        alpha_num = np.asarray(alpha_num, dtype=np.float64)
        f_num = np.asarray(f_num, dtype=np.float64)
        ln_scales = np.asarray(ln_scales, dtype=np.float64)
        backend = "Rust"
    else:
        log_z, alpha_num, f_num, ln_scales = _multifractal_moments_python(arr, b, q)
        backend = "Python"

    tau = np.full_like(q, np.nan, dtype=np.float64)
    alpha_cj = np.full_like(q, np.nan, dtype=np.float64)
    f_cj = np.full_like(q, np.nan, dtype=np.float64)
    r2_tau = np.full_like(q, np.nan, dtype=np.float64)
    r2_alpha = np.full_like(q, np.nan, dtype=np.float64)
    r2_f = np.full_like(q, np.nan, dtype=np.float64)

    for qi in range(len(q)):
        tau[qi], r2_tau[qi] = _linear_fit_slope_r2(ln_scales, log_z[:, qi])
        alpha_cj[qi], r2_alpha[qi] = _linear_fit_slope_r2(ln_scales, alpha_num[:, qi])
        f_cj[qi], r2_f[qi] = _linear_fit_slope_r2(ln_scales, f_num[:, qi])

    dq = np.full_like(q, np.nan, dtype=np.float64)
    mask_q1 = np.abs(q - 1.0) < 1e-12
    mask_qn = ~mask_q1
    dq[mask_qn] = tau[mask_qn] / (q[mask_qn] - 1.0)
    # Information dimension limit at q=1 from canonical alpha slope.
    dq[mask_q1] = alpha_cj[mask_q1]

    order = np.argsort(q)
    q_sorted = q[order]
    tau_sorted = tau[order]
    alpha_leg = np.full_like(q_sorted, np.nan, dtype=np.float64)
    f_leg = np.full_like(q_sorted, np.nan, dtype=np.float64)
    valid_tau = np.isfinite(q_sorted) & np.isfinite(tau_sorted)
    if valid_tau.sum() >= 3:
        alpha_leg[valid_tau] = np.gradient(tau_sorted[valid_tau], q_sorted[valid_tau])
        f_leg[valid_tau] = q_sorted[valid_tau] * alpha_leg[valid_tau] - tau_sorted[valid_tau]

    # Map Legendre outputs back to original q ordering.
    alpha_leg_unsorted = np.full_like(q, np.nan, dtype=np.float64)
    f_leg_unsorted = np.full_like(q, np.nan, dtype=np.float64)
    alpha_leg_unsorted[order] = alpha_leg
    f_leg_unsorted[order] = f_leg

    finite_alpha = np.isfinite(alpha_leg_unsorted)
    phi = float(np.std(alpha_leg_unsorted[finite_alpha])) if finite_alpha.any() else np.nan

    return {
        "q": q,
        "box_sizes": b,
        "ln_scales": ln_scales,
        "tau": tau,
        "Dq": dq,
        "alpha": alpha_leg_unsorted,
        "f_alpha": f_leg_unsorted,
        "alpha_legendre": alpha_leg_unsorted,
        "f_legendre": f_leg_unsorted,
        "alpha_canonical": alpha_cj,
        "f_canonical": f_cj,
        "r2_tau": r2_tau,
        "r2_alpha": r2_alpha,
        "r2_f": r2_f,
        "phi": phi,
        "backend": backend,
    }


def local_multifractality(
    field: np.ndarray,
    *,
    tile_size: int | tuple[int, int],
    box_sizes: np.ndarray | None = None,
    q_values: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute local multifractality maps over non-overlapping tiles.

    Parameters
    ----------
    field : ndarray, shape (ny, nx)
        Nonnegative 2D field.
    tile_size : int or (int, int)
        Tile side length(s) in pixels.
    box_sizes : array_like of int or None
        Box side lengths within each tile.
    q_values : array_like of float or None
        Moment orders q.

    Returns
    -------
    dict
        Keys: phi, delta, mean, r2_tau_median, q, box_sizes, slope_phi_log_delta,
        intercept_phi_log_delta, r2_phi_log_delta.
    """
    arr = _coerce_field(field)
    if arr.shape[1] == 1:
        raise ValueError("local_multifractality expects a 2D field, not 1D input")

    if isinstance(tile_size, int):
        ty = tx = int(tile_size)
    else:
        ty, tx = int(tile_size[0]), int(tile_size[1])
    if ty <= 0 or tx <= 0:
        raise ValueError("tile_size must contain positive integers")

    ny, nx = arr.shape
    n_ty = ny // ty
    n_tx = nx // tx
    if n_ty == 0 or n_tx == 0:
        raise ValueError("tile_size is larger than the field extent")

    # Derive defaults from one tile to keep scale policy consistent.
    sample_tile = arr[:ty, :tx]
    q = _clean_q_values(q_values)
    if box_sizes is None:
        b = _default_box_sizes(sample_tile.shape)
    else:
        b = _clean_box_sizes(box_sizes)

    phi = np.full((n_ty, n_tx), np.nan, dtype=np.float64)
    delta = np.full((n_ty, n_tx), np.nan, dtype=np.float64)
    mean = np.full((n_ty, n_tx), np.nan, dtype=np.float64)
    r2_tau_median = np.full((n_ty, n_tx), np.nan, dtype=np.float64)

    for iy in range(n_ty):
        y0 = iy * ty
        for ix in range(n_tx):
            x0 = ix * tx
            tile = arr[y0 : y0 + ty, x0 : x0 + tx]
            delta[iy, ix] = float(np.max(tile) - np.min(tile))
            mean[iy, ix] = float(np.mean(tile))
            tile_mass = float(np.sum(tile))
            if not np.isfinite(tile_mass) or tile_mass <= 0.0:
                continue
            spec = multifractal_spectrum(tile, box_sizes=b, q_values=q)
            phi[iy, ix] = float(spec["phi"])
            r2_tau = np.asarray(spec["r2_tau"], dtype=np.float64)
            finite = np.isfinite(r2_tau)
            if finite.any():
                r2_tau_median[iy, ix] = float(np.median(r2_tau[finite]))

    valid = np.isfinite(phi) & np.isfinite(delta) & (delta > 0.0)
    if valid.sum() >= 3:
        x = np.log(delta[valid])
        y = phi[valid]
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    else:
        slope = np.nan
        intercept = np.nan
        r2 = np.nan

    return {
        "phi": phi,
        "delta": delta,
        "mean": mean,
        "r2_tau_median": r2_tau_median,
        "q": q,
        "box_sizes": b,
        "slope_phi_log_delta": float(slope),
        "intercept_phi_log_delta": float(intercept),
        "r2_phi_log_delta": float(r2),
    }


__all__ = ["multifractal_spectrum", "local_multifractality"]
