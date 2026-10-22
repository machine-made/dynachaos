"""Poincare section diagnostics for scalar time series.

This module provides reusable, implementation-focused Poincare section
machinery for dynamical-systems pipelines:

1. Detect level crossings (default: mean level) with optional linear
   interpolation.
2. Construct common section planes from sampled signal/derivative values.
3. Estimate quality metrics based on crossing regularity and spectral dominance.

The API is intentionally general and side-effect free so downstream projects
can map these results into their own storage/plot schemas.
"""

from __future__ import annotations

import numpy as np


def _coerce_signal(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def _auto_delay_from_autocorr(signal: np.ndarray) -> int:
    centered = signal - np.mean(signal)
    n = centered.size
    if n < 4:
        return 1
    spectrum = np.fft.rfft(centered, n=2 * n)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    if autocorr[0] == 0.0:
        return max(1, n // 10)
    autocorr = autocorr / autocorr[0]

    scan_limit = max(3, n // 4)
    for i in range(1, scan_limit - 1):
        if autocorr[i] < autocorr[i - 1] and autocorr[i] < autocorr[i + 1]:
            return int(i)
    return max(1, n // 10)


def _crossing_indices(
    shifted: np.ndarray,
    direction: str,
    eps: float,
) -> np.ndarray:
    if direction == "up":
        mask = (shifted[:-1] <= eps) & (shifted[1:] > eps)
    elif direction == "down":
        mask = (shifted[:-1] >= -eps) & (shifted[1:] < -eps)
    elif direction == "both":
        mask_up = (shifted[:-1] <= eps) & (shifted[1:] > eps)
        mask_down = (shifted[:-1] >= -eps) & (shifted[1:] < -eps)
        mask = mask_up | mask_down
    else:
        raise ValueError("direction must be one of: 'up', 'down', 'both'")
    return np.nonzero(mask)[0].astype(np.int64)


def _sample_linear(array: np.ndarray, index: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    left = array[index]
    right = array[index + 1]
    return left + alpha * (right - left)


def _spectral_peak_ratio(signal: np.ndarray) -> float:
    if signal.size < 4:
        return np.nan
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    magnitudes = np.abs(spectrum)
    if magnitudes.size <= 1:
        return np.nan

    magnitudes[0] = 0.0
    primary_idx = int(np.argmax(magnitudes))
    primary_amp = float(magnitudes[primary_idx])
    if primary_amp == 0.0:
        return np.nan

    magnitudes[primary_idx] = 0.0
    secondary_amp = float(np.max(magnitudes))
    if secondary_amp == 0.0:
        return np.inf
    return primary_amp / secondary_amp


def _quality_metrics(
    crossing_times: np.ndarray,
    signal: np.ndarray,
) -> dict[str, float | str | int]:
    if crossing_times.size < 3:
        return {
            "coefficient_of_variation": np.nan,
            "mean_period": np.nan,
            "spectral_peak_ratio": np.nan,
            "quality": "insufficient_data",
            "num_crossings": int(crossing_times.size),
        }

    intervals = np.diff(crossing_times)
    mean_period = float(np.mean(intervals))
    std_period = float(np.std(intervals))
    cv = std_period / mean_period if mean_period > 0.0 else np.nan
    spectral_ratio = float(_spectral_peak_ratio(signal))

    if np.isnan(cv) or np.isnan(spectral_ratio):
        quality = "indeterminate"
    elif cv < 0.05 and spectral_ratio >= 5.0:
        quality = "highly_periodic"
    elif cv < 0.2 and spectral_ratio >= 3.0:
        quality = "periodic"
    elif cv < 0.6 and spectral_ratio >= 1.5:
        quality = "quasi_periodic"
    else:
        quality = "chaotic"

    return {
        "coefficient_of_variation": float(cv),
        "mean_period": float(mean_period),
        "spectral_peak_ratio": float(spectral_ratio),
        "quality": quality,
        "num_crossings": int(crossing_times.size),
    }


def _empty_result(arr, fs, dimension, interpolation_used, mean_signal, delay_samples):
    """Canonical empty Poincare section result."""
    empty = np.empty(0, dtype=np.float64)
    return {
        "crossing_times": empty,
        "crossing_values": empty,
        "planes": {},
        "metrics": _quality_metrics(empty, arr),
        "delay": int(delay_samples),
        "dimension": int(max(1, dimension)),
        "interpolation_used": bool(interpolation_used),
        "mean_signal": float(mean_signal),
        "sampling_frequency": float(fs),
        "section_plane_type": "unavailable",
        "section_points": np.empty((0, 2), dtype=np.float64),
    }


def poincare_section(
    signal: np.ndarray,
    fs: float,
    *,
    delay: int | None = None,
    level: float | None = None,
    direction: str = "up",
    interpolation: bool = True,
    dimension: int = 3,
) -> dict[str, object]:
    """Compute a Poincare section summary for a scalar time series.

    Parameters
    ----------
    signal : ndarray
        Scalar input signal.
    fs : float
        Sampling frequency in Hz.
    delay : int or None
        Delay in samples for the ``signal_delay_pair`` plane. If None or <= 0,
        uses the first local minimum of the autocorrelation.
    level : float or None
        Crossing level. Defaults to the signal mean.
    direction : {"up", "down", "both"}
        Crossing direction through ``level``.
    interpolation : bool
        If True, use linear interpolation for crossing time/value and plane
        sampling; otherwise use discrete sample indices.
    dimension : int
        Metadata field (kept for downstream compatibility).

    Returns
    -------
    dict
        Keys:
        - crossing_times, crossing_values
        - planes (dict[str, ndarray])
        - metrics (dict with quality fields)
        - delay, dimension, interpolation_used
        - mean_signal, sampling_frequency
        - section_plane_type, section_points
    """
    if fs <= 0.0:
        raise ValueError("fs must be positive")

    arr = _coerce_signal(signal)
    if arr.size < 3:
        return _empty_result(
            arr,
            fs,
            dimension,
            interpolation,
            float(np.mean(arr)) if arr.size else 0.0,
            1,
        )

    mean_signal = float(np.mean(arr)) if level is None else float(level)
    shifted = arr - mean_signal
    eps = 1e-12

    crossing_idx = _crossing_indices(shifted, direction=direction, eps=eps)
    if crossing_idx.size == 0:
        return _empty_result(
            arr,
            fs,
            dimension,
            interpolation,
            mean_signal,
            int(delay) if delay and delay > 0 else _auto_delay_from_autocorr(arr),
        )

    if interpolation:
        den = shifted[crossing_idx + 1] - shifted[crossing_idx]
        good = np.abs(den) > eps
        crossing_idx = crossing_idx[good]
        den = den[good]
        if crossing_idx.size == 0:
            return _empty_result(
                arr,
                fs,
                dimension,
                True,
                mean_signal,
                int(delay) if delay and delay > 0 else _auto_delay_from_autocorr(arr),
            )
        alpha = -shifted[crossing_idx] / den
    else:
        alpha = np.zeros(crossing_idx.size, dtype=np.float64)

    crossing_times = (crossing_idx.astype(np.float64) + alpha) / float(fs)
    if interpolation:
        crossing_values = _sample_linear(arr, crossing_idx, alpha)
    else:
        crossing_values = arr[crossing_idx].astype(np.float64)

    delay_samples = int(delay) if delay is not None else 0
    if delay_samples <= 0:
        delay_samples = _auto_delay_from_autocorr(arr)

    dt = 1.0 / float(fs)
    d1 = np.gradient(arr, dt)
    d2 = np.gradient(d1, dt)

    planes: dict[str, np.ndarray] = {}

    if interpolation:
        signal_at_crossings = crossing_values
        d1_at_crossings = _sample_linear(d1, crossing_idx, alpha)
        d2_at_crossings = _sample_linear(d2, crossing_idx, alpha)
    else:
        signal_at_crossings = arr[crossing_idx]
        d1_at_crossings = d1[crossing_idx]
        d2_at_crossings = d2[crossing_idx]

    if signal_at_crossings.size and d1_at_crossings.size:
        planes["signal_derivative"] = np.column_stack((signal_at_crossings, d1_at_crossings))

    if delay_samples > 0 and delay_samples < arr.size:
        valid = (crossing_idx - delay_samples >= 0) & (crossing_idx + delay_samples < arr.size)
        if np.any(valid):
            base = crossing_idx[valid]
            past_values = arr[base - delay_samples].astype(np.float64)
            future_values = arr[base + delay_samples].astype(np.float64)
            planes["signal_delay_pair"] = np.column_stack((past_values, future_values))

    if d1_at_crossings.size and d2_at_crossings.size:
        planes["derivative_second"] = np.column_stack((d1_at_crossings, d2_at_crossings))

    metrics = _quality_metrics(crossing_times, arr)

    if "signal_delay_pair" in planes:
        section_label = "signal_delay_pair"
        section_points = planes["signal_delay_pair"]
    elif "signal_derivative" in planes:
        section_label = "signal_derivative"
        section_points = planes["signal_derivative"]
    else:
        section_label = "unavailable"
        section_points = np.empty((0, 2), dtype=np.float64)

    return {
        "crossing_times": np.asarray(crossing_times, dtype=np.float64),
        "crossing_values": np.asarray(crossing_values, dtype=np.float64),
        "planes": {name: np.asarray(values, dtype=np.float64) for name, values in planes.items()},
        "metrics": metrics,
        "delay": int(delay_samples),
        "dimension": int(max(1, dimension)),
        "interpolation_used": bool(interpolation),
        "mean_signal": float(mean_signal),
        "sampling_frequency": float(fs),
        "section_plane_type": section_label,
        "section_points": np.asarray(section_points, dtype=np.float64),
    }


__all__ = ["poincare_section"]
