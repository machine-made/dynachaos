"""Robust Grassberger-Procaccia protocol: a banded regime classifier with a CI.

A single GP dimension number is fragile (delay/embedding/scaling-region choices
shift it, and it cannot separate adjacent integers or resolve D=3 at finite N).
This module wraps the Takens-Theiler estimator in the protocol that makes it
trustworthy as a *classifier*:

  1. delay tau          -- AMI first minimum (Fraser & Swinney 1986)
  2. embedding m        -- Cao (1997); plus a D_c(m) sweep to test saturation
  3. Theiler window     -- max((m-1)*tau, one dominant period in samples), to
                           exclude temporally-correlated pairs (Theiler 1986)
  4. point estimate     -- Takens-Theiler ML (correlation.takens_theiler_dimension)
  5. confidence interval-- ensemble over K disjoint trajectory SEGMENTS (never a
                           naive pair-bootstrap, which is falsely narrow)
  6. plateau gate       -- D_c(m) must flatten for m >= ceil(D_c); a monotonically
                           rising curve (T3 at finite N, or noise riding D_c=m)
                           sets plateau_ok=False -> "unresolved", deferring to the
                           complementary diagnostics (spectral peak count, 0-1).
  7. banded class       -- T1 / T2 / T3 / chaotic-or-unresolved by D_c band + gate.

Returns D_c +/- sigma with an explicit band label and reliability flags rather
than a single over-precise scalar. See docs/correlation_dimension_methodology.md
(protocol v2) for the literature basis.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

from dynachaos.diagnostics.correlation import takens_theiler_dimension
from dynachaos.diagnostics.embedding import _embed, optimal_delay, optimal_dimension


def _dominant_period_samples(x: np.ndarray) -> int:
    """Dominant period in samples from the periodogram peak (fs = 1/sample)."""
    f, p = periodogram(x - np.mean(x))
    if f.size < 2:
        return 1
    i = int(np.argmax(p[1:])) + 1
    f_peak = f[i]
    if f_peak <= 0:
        return 1
    return max(1, int(round(1.0 / f_peak)))


def _is_map_like(x: np.ndarray) -> bool:
    """Maps decorrelate in one step; AMI mis-picks tau for them. Heuristic: the
    lag-1 autocorrelation is small (|rho_1| < 0.2) for a chaotic map sampled at
    its natural rate, but ~1 for an oversampled flow."""
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return False
    rho1 = float(np.dot(x[:-1], x[1:]) / denom)
    return abs(rho1) < 0.2


def _dc_of_m(signal, tau, theiler, m_values, max_pairs, norm):
    """D_TT(m) curve for the saturation/plateau test (full signal)."""
    out = []
    for m in m_values:
        emb = _embed(signal, m, tau)
        if len(emb) < 500:
            out.append(np.nan)
            continue
        d, *_ = takens_theiler_dimension(
            emb, max_pairs=max_pairs, theiler_window=theiler, norm=norm
        )
        out.append(d)
    return np.asarray(out, dtype=float)


# GP is validated trustworthy only for D_c <~ 2.5 (recovers Lorenz/Rossler/T2);
# above that it is at/over the finite-N data wall (Eckmann-Ruelle / Nerenberg-Essex)
# and apparent plateaus are unreliable -> defer to complementary diagnostics.
_RELIABLE_DC_CEILING = 2.5


def _saturation(m_values, dcm, slope_tol: float = 0.04):
    """Has D_TT(m) saturated? Fit a line to D_TT(m) over the upper-m half and test
    |slope| < slope_tol (per unit m). Returns (saturated, slope, D_c) where D_c is
    the mean over the upper-m half (the plateau value). A still-rising curve (T3 at
    finite N, noise riding D_c=m) is NOT saturated."""
    dcm = np.asarray(dcm, float)
    mv = np.asarray(m_values, float)
    fin = np.isfinite(dcm)
    if fin.sum() < 4:
        return False, np.nan, (float(np.nanmean(dcm)) if fin.any() else np.nan)
    mvf, dcf = mv[fin], dcm[fin]
    upper = mvf >= np.median(mvf)
    if upper.sum() < 3:
        upper = np.ones(mvf.size, bool)
    slope = float(np.polyfit(mvf[upper], dcf[upper], 1)[0])
    D_c = float(np.mean(dcf[upper]))
    return abs(slope) < slope_tol, slope, D_c


def _classify(D_c: float, saturated: bool):
    """(band_class, gp_certifiable). GP can certify only a saturated, sub-ceiling
    dimension; otherwise the label is 'unresolved' and the regime must come from
    the complementary diagnostics (spectral peak count, 0-1 test, surrogates)."""
    if not np.isfinite(D_c) or not saturated:
        return "unresolved (no plateau)", False
    if D_c > _RELIABLE_DC_CEILING:
        return "high-D: GP cannot certify (>2.5, data wall)", False
    if D_c < 1.25:
        return "T1", True
    if 1.75 <= D_c <= 2.35:
        return "T2", True
    return "ambiguous (band gap)", True  # 1.25-1.75 or 2.35-2.5


def gp_dimension_robust(
    signal,
    tau=None,
    m_cao=None,
    theiler=None,
    n_segments=8,
    m_max=10,
    max_pairs=2_000_000,
    norm="chebyshev",
    is_map=None,
):
    """Robust GP correlation dimension with a banded classification and CI.

    Parameters
    ----------
    signal : 1-D array
        Scalar observable (e.g. a lift/drag time series).
    tau, m_cao, theiler : int, optional
        Override the auto-selected delay / embedding dimension / Theiler window.
    n_segments : int
        Number of disjoint trajectory segments for the ensemble CI.
    m_max : int
        Largest embedding dimension in the D_c(m) saturation sweep.
    is_map : bool, optional
        Force map handling (tau=1). If None, auto-detected from lag-1 autocorr.

    Returns
    -------
    dict with keys:
        D_c        -- median Takens-Theiler dimension over the segment ensemble
        sigma      -- std of the segment estimates (the honest uncertainty)
        ci         -- (2.5, 97.5) percentile interval over segments
        tau_used, m_used, theiler_used
        plateau_ok -- whether D_c(m) saturated (else defer to other diagnostics)
        band_class -- "T1"/"T2"/"T3"/"T3?-unresolved"/"chaotic/high-D"/...
        dc_of_m    -- the D_c(m) sweep (for plotting / auditing)
        n, n_segments_used
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = (x - x.mean()) / (x.std() + 1e-300)
    n = x.size

    if is_map is None:
        is_map = _is_map_like(x)

    if tau is None:
        tau = 1 if is_map else max(1, int(optimal_delay(x, tau_max=min(200, n // 4))))
    if m_cao is None:
        try:
            m_cao = int(optimal_dimension(x, tau, d_max=min(12, m_max)))
        except Exception:
            m_cao = 5
    m_cao = int(np.clip(m_cao, 2, m_max))

    if theiler is None:
        period = _dominant_period_samples(x)
        theiler = int(max((m_cao - 1) * tau, period))

    # Central value: D_TT(m) sweep on the FULL signal -> plateau mean + saturation
    # test (short segments inflate D for D>~2.5, so the central value must NOT come
    # from them; segments are used only for the uncertainty below).
    m_values = list(range(2, m_max + 1))
    dcm = _dc_of_m(x, tau, theiler, m_values, max_pairs, norm)
    saturated, dcm_slope, D_c = _saturation(m_values, dcm)
    band_class, gp_certifiable = _classify(D_c, saturated)

    # Uncertainty: ensemble over disjoint segments (each long enough to embed).
    # NOT a naive pair-bootstrap (which is falsely narrow). Large sigma or a big
    # segment-vs-full gap is itself a reliability flag.
    seg_len = n // n_segments
    seg_vals = []
    min_seg = max(1000, (m_cao - 1) * tau + 50)
    if seg_len >= min_seg:
        for k in range(n_segments):
            seg = x[k * seg_len : (k + 1) * seg_len]
            emb = _embed(seg, m_cao, tau)
            if len(emb) < 500:
                continue
            d, *_ = takens_theiler_dimension(
                emb, max_pairs=max_pairs, theiler_window=theiler, norm=norm
            )
            if np.isfinite(d):
                seg_vals.append(d)
    if len(seg_vals) >= 3:
        seg_vals = np.asarray(seg_vals)
        sigma = float(np.std(seg_vals))
        ci = (float(np.percentile(seg_vals, 2.5)), float(np.percentile(seg_vals, 97.5)))
        seg_median = float(np.median(seg_vals))
        n_used = len(seg_vals)
    else:
        sigma = float("nan")
        ci = (float("nan"), float("nan"))
        seg_median = float("nan")
        n_used = 0

    return {
        "D_c": D_c,  # full-signal D_TT(m) plateau mean
        "sigma": sigma,  # std over independent segments
        "ci": ci,
        "tau_used": int(tau),
        "m_used": int(m_cao),
        "theiler_used": int(theiler),
        "plateau_ok": bool(saturated),
        "gp_certifiable": bool(gp_certifiable),
        "band_class": band_class,
        "dcm_slope": float(dcm_slope),
        "seg_median": seg_median,
        "dc_of_m": dcm,
        "m_values": np.asarray(m_values),
        "n": int(n),
        "n_segments_used": int(n_used),
        "is_map": bool(is_map),
    }
