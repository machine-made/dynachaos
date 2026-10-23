"""Agreement of chaos diagnostics with the sign of the largest Lyapunov exponent.

The reference label of a parameter value comes from the Lyapunov spectrum and
its standard errors:

    chaotic    lambda_1 > 2 SE
    regular    lambda_1 < -2 SE, or a quasi-periodic orbit
    marginal   otherwise

A quasi-periodic orbit has lambda_1 within 2 SE of zero and every other
exponent below -2 SE, or within 2 SE of zero.  It needs at least two
dimensions, so a one-dimensional map with lambda_1 near zero is marginal.
A fixed point or periodic orbit near a bifurcation also has lambda_1 near
zero.  Its other exponents are negative, so it is labelled regular as well.

Marginal points are counted and not scored.  Scoring uses the agreement
fraction at a fixed threshold (K_01 > 0.5) or the area under the ROC curve
(threshold free).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

REGULAR, CHAOTIC, MARGINAL = 0, 1, 2
LABEL_NAMES = {REGULAR: "regular", CHAOTIC: "chaotic", MARGINAL: "marginal"}


def label_from_spectrum(spectrum: ArrayLike, se: ArrayLike, n_sigma: float = 2.0) -> int:
    """Label one parameter value from its Lyapunov spectrum and standard errors.

    Parameters
    ----------
    spectrum, se : array_like, shape (N,)
        Exponents in descending order and their standard errors.
    n_sigma : float
        Width of the zero band in standard errors.

    Returns
    -------
    int
        ``REGULAR``, ``CHAOTIC`` or ``MARGINAL``.
    """
    lam = np.asarray(spectrum, dtype=np.float64)
    err = np.asarray(se, dtype=np.float64)
    if lam.ndim != 1 or lam.shape != err.shape or lam.size == 0:
        raise ValueError("spectrum and se must be 1D arrays of equal length")
    band = n_sigma * err
    if lam[0] > band[0]:
        return CHAOTIC
    if lam[0] < -band[0]:
        return REGULAR
    # lambda_1 is within the zero band.  Regular only if no other exponent is positive
    # and the system has a direction left to contract: quasi-periodic or marginal-periodic.
    if lam.size > 1 and np.all(lam[1:] < band[1:]):
        return REGULAR
    return MARGINAL


def label_sweep(
    spectra: ArrayLike, spectra_se: ArrayLike, n_sigma: float = 2.0
) -> NDArray[np.int8]:
    """Labels for a sweep: ``spectra`` and ``spectra_se`` have shape (n_params, N)."""
    return np.array(
        [label_from_spectrum(s, e, n_sigma) for s, e in zip(spectra, spectra_se, strict=True)],
        dtype=np.int8,
    )


def roc_auc(scores: ArrayLike, is_positive: ArrayLike) -> float:
    """Area under the ROC curve; ties count one half.

    The value is the probability that a random positive point scores above a
    random negative point.  It equals 1 for a perfect separator and 0.5 for
    scores that carry no information.

    Parameters
    ----------
    scores : array_like, shape (n,)
        Larger score means more likely positive.
    is_positive : array_like of bool, shape (n,)
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    pos = np.asarray(is_positive, dtype=bool).ravel()
    if scores.shape != pos.shape:
        raise ValueError("scores and is_positive must have the same length")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    s_pos, s_neg = scores[pos], np.sort(scores[~pos])
    if s_pos.size == 0 or s_neg.size == 0:
        raise ValueError("need at least one positive and one negative point")
    below = np.searchsorted(s_neg, s_pos, side="left")
    below_or_tied = np.searchsorted(s_neg, s_pos, side="right")
    wins = below + 0.5 * (below_or_tied - below)
    return float(wins.sum() / (s_pos.size * s_neg.size))


def roc_curve(
    scores: ArrayLike, is_positive: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """False- and true-positive rates for thresholds at every distinct score.

    Returns ``fpr, tpr`` starting at (0, 0) and ending at (1, 1).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    pos = np.asarray(is_positive, dtype=bool).ravel()
    order = np.argsort(-scores, kind="stable")
    s, p = scores[order], pos[order]
    last_of_run = np.r_[np.diff(s) != 0, True]
    tps = np.cumsum(p)[last_of_run]
    fps = np.cumsum(~p)[last_of_run]
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]]


def scored_agreement(
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    threshold: float | None = None,
    chaotic_high: bool = True,
) -> dict[str, object]:
    """Compare one diagnostic with the reference labels on the scored points.

    Parameters
    ----------
    scores : array_like
        Diagnostic value per parameter value.
    labels : array_like of int
        ``REGULAR``, ``CHAOTIC`` or ``MARGINAL`` per parameter value.
    threshold : float or None
        If given, a point is called chaotic when ``score > threshold``
        (``chaotic_high``) or ``score < threshold`` (otherwise), and the
        agreement fraction is returned.
    chaotic_high : bool
        True when a larger score means more chaotic.

    Returns
    -------
    dict
        ``n_scored``, ``n_marginal``, ``n_chaotic``, ``n_regular``, ``auc``
        (oriented so that chaotic is positive), and, with a threshold,
        ``agreement`` and ``disagree`` (indices of the disagreeing points).
        ``misranked_regular`` holds regular points whose score is above the
        median chaotic score, and ``misranked_chaotic`` holds chaotic points
        below the median regular score, both as indices into ``scores``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    scored = labels != MARGINAL
    oriented = scores if chaotic_high else -scores
    chaotic = labels == CHAOTIC
    regular = labels == REGULAR
    out = {
        "n_scored": int(scored.sum()),
        "n_marginal": int((~scored).sum()),
        "n_chaotic": int(chaotic.sum()),
        "n_regular": int(regular.sum()),
        "auc": roc_auc(oriented[scored], chaotic[scored]),
    }
    med_chaotic = np.median(oriented[chaotic])
    med_regular = np.median(oriented[regular])
    out["misranked_regular"] = np.flatnonzero(regular & (oriented > med_chaotic))
    out["misranked_chaotic"] = np.flatnonzero(chaotic & (oriented < med_regular))
    if threshold is not None:
        called_chaotic = oriented > (threshold if chaotic_high else -threshold)
        wrong = scored & (called_chaotic != chaotic)
        out["agreement"] = float(1.0 - wrong.sum() / scored.sum())
        out["disagree"] = np.flatnonzero(wrong)
    return out
