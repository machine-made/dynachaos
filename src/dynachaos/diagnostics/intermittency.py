"""Building-block diagnostics for intermittency in scalar signals.

The routines here expose laminar masks, laminar-run lengths, empirical
distributions, reinjection statistics, return-map geometry, and recurrence
laminarity. They deliberately do not return a Pomeau-Manneville or on-off type
label; downstream analysis should interpret the statistics with the
assumption-dependent caveats from the literature.

Signature guide
---------------
- Type I: laminar-length tail near ``ell^-3/2`` and mean laminar length often
  scales like ``eps^-1/2`` under standard uniform-reinjection assumptions.
- Type II/III: laminar lengths are usually better described by exponential
  decay; the mean-scaling law is reinjection-distribution dependent, so both
  ``eps^-1`` and ``log(1/eps)`` candidates should be inspected.
- On-off: laminar durations have the same ``-3/2`` onset exponent as Type I,
  so use burst-amplitude ``|x|^-1`` behavior and laminar/burst duration
  symmetry as additional discriminators.

Caveats
-------
Reinjection probability density, finite data windows, colored driving, and the
choice of observable can change apparent exponents and cutoffs. Treat the
returned objects as diagnostic evidence, not as a theorem or classifier.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.special import zeta
from scipy.stats import expon, ks_2samp, linregress, norm

from dynachaos.diagnostics.correlation import fit_power_law_loglog as _shared_fit_power_law_loglog
from dynachaos.diagnostics.poincare import _auto_delay_from_autocorr, poincare_section
from dynachaos.diagnostics.recurrence import LaminarLengthsResult, recurrence_matrix
from dynachaos.diagnostics.recurrence import laminar_lengths as recurrence_laminar_lengths


@dataclass(frozen=True)
class LaminarLengthDistribution:
    """Binned and discrete empirical laminar-length distributions."""

    bin_edges: np.ndarray
    density: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class PowerLawMLE:
    """Primary CSN power-law tail fit.

    ``csn_alpha`` is the positive Clauset-Shalizi-Newman exponent in
    ``p(x) ~ x**(-csn_alpha)``. ``alpha`` is the signed log-log tail exponent
    used in the intermittency literature, e.g. ``alpha ~= -1.5``.
    """

    alpha: float
    csn_alpha: float
    x_min: float
    ks_distance: float
    n_tail: int
    standard_error: float
    discrete: bool


@dataclass(frozen=True)
class PowerLawGoF:
    """Semiparametric bootstrap goodness-of-fit for a CSN power-law tail."""

    statistic: float
    p_value: float
    n_bootstrap: int
    fit: PowerLawMLE


@dataclass(frozen=True)
class ExponentialFit:
    """Shifted exponential MLE on the tail ``x >= x_min``."""

    rate: float
    scale: float
    x_min: float
    n_tail: int
    log_likelihood: float


@dataclass(frozen=True)
class VuongComparison:
    """Normalized Vuong likelihood-ratio comparison."""

    log_likelihood_ratio: float
    z: float
    p_value: float
    n_tail: int
    power_law: PowerLawMLE
    exponential: ExponentialFit


@dataclass(frozen=True)
class DiagnosticLogLogFit:
    """Diagnostic-only log-log power-law regression result."""

    slope: float
    intercept: float
    rvalue: float
    local_slopes: np.ndarray
    scaling_mask: np.ndarray
    diagnostic_only: bool = True


@dataclass(frozen=True)
class CandidateScalingLaw:
    """Least-squares candidate law for contested mean-laminar scaling."""

    name: str
    coefficients: np.ndarray
    residual_sum_squares: float


@dataclass(frozen=True)
class MeanLaminarScaling:
    """Mean laminar length scaling summary.

    ``beta`` is estimated from ``<l> ~ eps**(-beta)``. The Type-II/III mean
    scaling is assumption dependent, so this result also reports the
    ``eps^-1`` and ``log(1/eps)`` candidate-law residuals without choosing a
    mechanism label.
    """

    beta: float
    loglog: DiagnosticLogLogFit
    inverse_epsilon: CandidateScalingLaw
    logarithmic: CandidateScalingLaw
    rpd_alpha: float | None


@dataclass(frozen=True)
class ReinjectionMx:
    """Linear RPD characteristic relation from reinjection points."""

    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float
    alpha: float
    x_hat: float
    reinjection_points: np.ndarray
    thresholds: np.ndarray
    conditional_means: np.ndarray


@dataclass(frozen=True)
class BurstAmplitudeDistribution:
    """Burst-amplitude empirical distribution and CSN power-law fit."""

    amplitudes: np.ndarray
    bin_edges: np.ndarray
    density: np.ndarray
    power_law: PowerLawMLE


@dataclass(frozen=True)
class LaminarBurstSymmetry:
    """Two-sample KS comparison of laminar and burst duration distributions."""

    statistic: float
    p_value: float
    laminar_lengths: np.ndarray
    burst_lengths: np.ndarray


@dataclass(frozen=True)
class ExtremaReturnMap:
    """First-return map built from successive extrema."""

    indices: np.ndarray
    values: np.ndarray
    points: np.ndarray
    kind: str
    prominence: float | None
    distance: int | None


@dataclass(frozen=True)
class TangentChannel:
    """Near-diagonal return-map channel and its linear trend."""

    points: np.ndarray
    mask: np.ndarray
    distance_threshold: float
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float


@dataclass(frozen=True)
class ReturnMapReconstruction:
    """Poincare section plus extrema return-map reconstruction."""

    poincare: dict[str, object]
    extrema: ExtremaReturnMap
    tangent_channel: TangentChannel


@dataclass(frozen=True)
class IntermittencySummary:
    """Composed intermittency diagnostics without a mechanism label."""

    laminar_mask: np.ndarray
    laminar_lengths: np.ndarray
    laminar_distribution: LaminarLengthDistribution
    laminar_power_law: PowerLawMLE
    powerlaw_gof: PowerLawGoF
    family_comparison: VuongComparison
    recurrence_laminarity: LaminarLengthsResult
    reinjection: ReinjectionMx
    burst_amplitude: BurstAmplitudeDistribution
    laminar_burst_symmetry: LaminarBurstSymmetry
    return_map: ReturnMapReconstruction


def detect_laminar_phases(
    x,
    *,
    method="recurrence",
    eps=None,
    period=None,
    window=None,
    percentile=5.0,
    v_min=2,
    drop_final_censored=False,
):
    """Detect laminar samples and laminar-run lengths in a scalar signal.

    Parameters are estimated from the signal when omitted: recurrence uses the
    existing recurrence-rate percentile threshold, period uses the first
    autocorrelation local minimum, and variance uses that same delay as its
    rolling window. The returned lengths are measured in samples.
    """
    series = _finite_series(x)
    if method == "recurrence":
        mask, lengths = _detect_laminar_recurrence(series, eps, percentile, v_min)
    elif method == "period":
        mask, lengths = _detect_laminar_period(series, eps, period, percentile)
    elif method == "variance":
        mask, lengths = _detect_laminar_variance(series, eps, window, percentile)
    else:
        raise ValueError("method must be one of: 'recurrence', 'period', 'variance'")
    if drop_final_censored and mask[-1] and lengths.size:
        lengths = lengths[:-1]
    return mask, lengths


def laminar_length_distribution(lengths, *, drop_final_censored=False):
    """Return Freedman-Diaconis-binned and exact-count distributions.

    ``drop_final_censored`` is an opt-in finite-window correction for ordered
    laminar-run lengths: when the source series ends inside a laminar phase, the
    final run is right-censored, so this option drops only that terminal run
    before estimating the empirical distribution.
    """
    lengths = _positive_lengths(lengths)
    if drop_final_censored and lengths.size:
        lengths = lengths[:-1]
    if lengths.size == 0:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return LaminarLengthDistribution(
            bin_edges=empty_float,
            density=empty_float,
            values=empty_int,
            counts=empty_int,
            probabilities=empty_float,
        )

    bin_edges = _bounded_histogram_bin_edges(lengths)
    density, bin_edges = _finite_histogram_mass(lengths, bin_edges)
    values, counts = np.unique(lengths, return_counts=True)
    probabilities = counts / np.sum(counts)
    return LaminarLengthDistribution(
        bin_edges=bin_edges.astype(np.float64),
        density=density.astype(np.float64),
        values=values.astype(np.int64),
        counts=counts.astype(np.int64),
        probabilities=probabilities.astype(np.float64),
    )


def pooled_laminar_lengths(
    oracle_factory,
    *,
    n_runs,
    seed,
    laminar_method="recurrence",
    **laminar_kwargs,
):
    """Pool laminar-run lengths across independent seeded oracle runs.

    ``oracle_factory`` is called once per derived seed and must return one
    scalar oracle series. Laminar detection itself is delegated unchanged to
    :func:`detect_laminar_phases`.
    """
    n_runs = _positive_int(n_runs, "n_runs")
    rng = np.random.default_rng(seed)
    lengths = []
    for run_seed in rng.integers(0, np.iinfo(np.uint32).max, size=n_runs, dtype=np.uint32):
        series = oracle_factory(int(run_seed))
        _, run_lengths = detect_laminar_phases(
            series,
            method=laminar_method,
            **laminar_kwargs,
        )
        lengths.append(run_lengths)
    return np.concatenate(lengths).astype(np.int64, copy=False)


def fit_power_law_mle(lengths, *, discrete=None, min_tail=None, drop_final_censored=False):
    """Fit a power-law tail by the Clauset-Shalizi-Newman MLE procedure.

    ``x_min`` is chosen by the minimum Kolmogorov-Smirnov distance over the
    observed support. Log-log fitting is intentionally not used here.
    """
    observations = _positive_observations(lengths)
    if drop_final_censored:
        observations = observations[:-1]
        if observations.size < 2:
            raise ValueError("at least two uncensored observations are required")
    data = observations
    if discrete is None:
        discrete = bool(np.allclose(data, np.rint(data)))
    discrete = bool(discrete)
    if min_tail is None:
        min_tail = max(3, int(np.ceil(np.sqrt(data.size))))
    min_tail = _positive_int(min_tail, "min_tail")
    if min_tail > data.size:
        raise ValueError("min_tail must not exceed the number of observations")

    candidates = np.unique(data)
    best_fit = None
    for x_min in candidates:
        tail = data[data >= x_min]
        if tail.size < min_tail:
            continue
        if discrete:
            csn_alpha = _discrete_power_law_alpha(tail, x_min)
        else:
            csn_alpha = _continuous_power_law_alpha(tail, x_min)
        ks_distance = _power_law_ks(tail, x_min, csn_alpha, discrete)
        if best_fit is None or ks_distance < best_fit.ks_distance:
            best_fit = PowerLawMLE(
                alpha=-float(csn_alpha),
                csn_alpha=float(csn_alpha),
                x_min=float(x_min),
                ks_distance=float(ks_distance),
                n_tail=int(tail.size),
                standard_error=float((csn_alpha - 1.0) / np.sqrt(tail.size)),
                discrete=discrete,
            )

    if best_fit is None:
        raise ValueError("not enough tail observations to fit a power law")
    return best_fit


def powerlaw_gof(lengths, *, fit=None, n_bootstrap=100, rng=None):
    """CSN semiparametric bootstrap goodness-of-fit.

    The returned ``p_value`` is the fraction of bootstrap KS distances at least
    as large as the observed KS distance; values above 0.1 mean the fitted
    power law is not rejected by this diagnostic.
    """
    data = _positive_observations(lengths)
    fit = fit_power_law_mle(data) if fit is None else fit
    n_bootstrap = _positive_int(n_bootstrap, "n_bootstrap")
    rng = np.random.default_rng(rng)

    below_tail = data[data < fit.x_min]
    bootstrap_ks = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        synthetic_tail = _sample_power_law_tail(fit, rng, fit.n_tail)
        if below_tail.size:
            synthetic_below = rng.choice(below_tail, size=below_tail.size, replace=True)
            synthetic = np.concatenate([synthetic_below, synthetic_tail])
        else:
            synthetic = synthetic_tail
        bootstrap_fit = fit_power_law_mle(synthetic, discrete=fit.discrete, min_tail=fit.n_tail)
        bootstrap_ks[i] = bootstrap_fit.ks_distance

    p_value = np.mean(bootstrap_ks >= fit.ks_distance)
    return PowerLawGoF(
        statistic=float(fit.ks_distance),
        p_value=float(p_value),
        n_bootstrap=int(n_bootstrap),
        fit=fit,
    )


def powerlaw_alpha_ci(lengths, *, fit=None, n_bootstrap=200, confidence=0.95, rng=None):
    """Bootstrap confidence interval for the signed CSN tail exponent alpha."""
    data = _positive_observations(lengths)
    fit = fit_power_law_mle(data) if fit is None else fit
    n_bootstrap = _positive_int(n_bootstrap, "n_bootstrap")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(rng)

    alphas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = _sample_power_law_tail(fit, rng, fit.n_tail)
        if fit.discrete:
            csn_alpha = _discrete_power_law_alpha(sample, fit.x_min)
        else:
            csn_alpha = _continuous_power_law_alpha(sample, fit.x_min)
        alphas[i] = -float(csn_alpha)

    tail_probability = (1.0 - confidence) / 2.0
    return np.quantile(alphas, [tail_probability, 1.0 - tail_probability]).astype(np.float64)


def fit_exponential(lengths, *, x_min=None):
    """Fit a shifted exponential tail with scipy's MLE primitives."""
    data = _positive_observations(lengths)
    if x_min is None:
        x_min = fit_power_law_mle(data).x_min
    x_min = _finite_nonnegative_float(x_min, "x_min")
    tail = data[data >= x_min]
    if tail.size < 2:
        raise ValueError("at least two tail observations are required")
    shifted = tail - x_min
    _, scale = expon.fit(shifted, floc=0.0)
    rate = 1.0 / scale if scale > 0.0 else np.inf
    log_likelihood = np.sum(expon.logpdf(shifted, loc=0.0, scale=scale))
    return ExponentialFit(
        rate=float(rate),
        scale=float(scale),
        x_min=float(x_min),
        n_tail=int(tail.size),
        log_likelihood=float(log_likelihood),
    )


def compare_powerlaw_exponential(lengths, *, discrete=None):
    """Compare CSN power law against exponential by normalized Vuong LR."""
    data = _positive_observations(lengths)
    power_law = fit_power_law_mle(data, discrete=discrete)
    exponential = fit_exponential(data, x_min=power_law.x_min)
    tail = data[data >= power_law.x_min]

    ll_power = _power_law_logpdf(tail, power_law.x_min, power_law.csn_alpha, power_law.discrete)
    ll_exp = expon.logpdf(tail - power_law.x_min, loc=0.0, scale=exponential.scale)
    pointwise = ll_power - ll_exp
    ratio = float(np.sum(pointwise))
    sigma = float(np.std(pointwise, ddof=1))
    z_score = ratio / (np.sqrt(tail.size) * sigma) if sigma > 0.0 else np.nan
    p_value = 2.0 * norm.sf(abs(z_score)) if np.isfinite(z_score) else np.nan
    return VuongComparison(
        log_likelihood_ratio=ratio,
        z=float(z_score),
        p_value=float(p_value),
        n_tail=int(tail.size),
        power_law=power_law,
        exponential=exponential,
    )


def fit_power_law_loglog(x, y, *, min_points=5):
    """Diagnostic-only log-log power-law fit.

    This reuses the shared scaling-region detector from
    :mod:`dynachaos.diagnostics.correlation`. CSN MLE remains the primary tail
    estimator because log-log regression is biased for power-law inference.
    """
    slope, intercept, rvalue, local_slopes, scaling_mask = _shared_fit_power_law_loglog(
        x, y, min_points=min_points
    )
    return DiagnosticLogLogFit(
        slope=slope,
        intercept=intercept,
        rvalue=rvalue,
        local_slopes=local_slopes,
        scaling_mask=scaling_mask,
    )


def mean_laminar_scaling(eps, mean_lengths, *, rpd_alpha=None, min_points=3):
    """Estimate ``<l> ~ eps**(-beta)`` and report contested alternatives.

    Type-I intermittency has robust ``beta ~= 1/2`` under the standard
    assumptions. Type-II/III mean scaling is RPD-dependent, so both
    ``eps^-1`` and ``log(1/eps)`` candidate fits are returned along with the
    optional RPD exponent from :func:`reinjection_Mx` once available.
    """
    eps = _positive_observations(eps)
    mean_lengths = _positive_observations(mean_lengths)
    if eps.shape != mean_lengths.shape:
        raise ValueError("eps and mean_lengths must have matching shape")
    if rpd_alpha is not None:
        rpd_alpha = float(rpd_alpha)
        if not np.isfinite(rpd_alpha):
            raise ValueError("rpd_alpha must be finite when provided")

    loglog = fit_power_law_loglog(eps, mean_lengths, min_points=min_points)
    inverse = _fit_inverse_epsilon_candidate(eps, mean_lengths)
    logarithmic = _fit_logarithmic_candidate(eps, mean_lengths)
    return MeanLaminarScaling(
        beta=float(-loglog.slope),
        loglog=loglog,
        inverse_epsilon=inverse,
        logarithmic=logarithmic,
        rpd_alpha=rpd_alpha,
    )


def reinjection_Mx(x, mask):
    """Estimate the reinjection probability-density exponent from ``M(x)``.

    Reinjection points are samples where a burst switches into a laminar run.
    The cumulative conditional mean relation is fitted with
    :func:`scipy.stats.linregress`; the RPD exponent is
    ``alpha = (2m - 1) / (1 - m)`` from the fitted slope ``m``.
    """
    series = _finite_series(x)
    mask = _finite_mask(mask, series.size)
    starts = np.flatnonzero(mask & np.r_[False, ~mask[:-1]])
    reinjections = np.sort(series[starts])
    if reinjections.size < 3:
        raise ValueError("at least three reinjection points are required")

    thresholds = np.unique(reinjections)
    if thresholds.size < 3:
        raise ValueError("at least three unique reinjection points are required")
    cumulative_sum = np.cumsum(reinjections)
    right_edges = np.searchsorted(reinjections, thresholds, side="right")
    conditional_means = cumulative_sum[right_edges - 1] / right_edges

    result = linregress(thresholds, conditional_means)
    if result.slope >= 1.0:
        alpha = np.inf
        x_hat = np.nan
    else:
        alpha = (2.0 * result.slope - 1.0) / (1.0 - result.slope)
        x_hat = result.intercept / (1.0 - result.slope)

    return ReinjectionMx(
        slope=float(result.slope),
        intercept=float(result.intercept),
        rvalue=float(result.rvalue),
        pvalue=float(result.pvalue),
        stderr=float(result.stderr),
        alpha=float(alpha),
        x_hat=float(x_hat),
        reinjection_points=reinjections.astype(np.float64),
        thresholds=thresholds.astype(np.float64),
        conditional_means=conditional_means.astype(np.float64),
    )


def burst_amplitude_distribution(x, mask, *, min_tail=None):
    """Fit the burst-amplitude law on non-laminar samples."""
    series = np.abs(_finite_series(x))
    mask = _finite_mask(mask, series.size)
    amplitudes = series[~mask]
    amplitudes = amplitudes[amplitudes > 0.0]
    if amplitudes.size < 2:
        raise ValueError("at least two positive burst amplitudes are required")

    bin_edges = _bounded_histogram_bin_edges(amplitudes)
    density, bin_edges = _finite_histogram_mass(amplitudes, bin_edges)
    return BurstAmplitudeDistribution(
        amplitudes=amplitudes.astype(np.float64),
        bin_edges=bin_edges.astype(np.float64),
        density=density.astype(np.float64),
        power_law=fit_power_law_mle(amplitudes, discrete=False, min_tail=min_tail),
    )


def laminar_burst_symmetry(x, mask):
    """Compare laminar and burst duration distributions with a KS test."""
    series = _finite_series(x)
    mask = _finite_mask(mask, series.size)
    laminar = _mask_run_lengths(mask)
    burst = _mask_run_lengths(~mask)
    if laminar.size < 2 or burst.size < 2:
        raise ValueError("at least two laminar and burst runs are required")
    result = ks_2samp(laminar, burst)
    return LaminarBurstSymmetry(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        laminar_lengths=laminar,
        burst_lengths=burst,
    )


def extrema_return_map(x, *, kind="max", prominence=None, distance=None):
    """Build a first-return map from successive extrema in a scalar signal."""
    series = _finite_series(x)
    if kind == "max":
        peak_source = series
    elif kind == "min":
        peak_source = -series
    else:
        raise ValueError("kind must be one of: 'max', 'min'")

    if prominence is not None:
        prominence = _finite_nonnegative_float(prominence, "prominence")
    if distance is not None:
        distance = _positive_int(distance, "distance")
    indices, _ = find_peaks(peak_source, prominence=prominence, distance=distance)
    values = series[indices].astype(np.float64)
    if values.size < 2:
        points = np.empty((0, 2), dtype=np.float64)
    else:
        points = np.column_stack((values[:-1], values[1:])).astype(np.float64)
    return ExtremaReturnMap(
        indices=indices.astype(np.int64),
        values=values,
        points=points,
        kind=kind,
        prominence=prominence,
        distance=distance,
    )


def near_diagonal_tangent_channel(points, *, percentile=10.0, min_points=3):
    """Extract a near-diagonal return-map channel using a distance percentile."""
    points = np.asarray(points, dtype=np.float64)
    min_points = _positive_int(min_points, "min_points")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if points.shape[0] < min_points:
        raise ValueError("at least min_points return-map points are required")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")

    distances = np.abs(points[:, 1] - points[:, 0]) / np.sqrt(2.0)
    threshold = float(np.percentile(distances, percentile))
    mask = distances <= threshold
    if np.count_nonzero(mask) < min_points:
        order = np.argsort(distances)
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[order[:min_points]] = True
        threshold = float(distances[order[min_points - 1]])
    channel = points[mask]
    result = linregress(channel[:, 0], channel[:, 1])
    return TangentChannel(
        points=channel.astype(np.float64),
        mask=mask,
        distance_threshold=threshold,
        slope=float(result.slope),
        intercept=float(result.intercept),
        rvalue=float(result.rvalue),
        pvalue=float(result.pvalue),
        stderr=float(result.stderr),
    )


def return_map_reconstruction(
    x,
    fs=1.0,
    *,
    kind="max",
    prominence=None,
    distance=None,
    channel_percentile=10.0,
    min_channel_points=3,
    **poincare_kwargs,
):
    """Compose existing Poincare reconstruction with extrema return-map geometry."""
    section = poincare_section(_finite_series(x), fs, **poincare_kwargs)
    extrema = extrema_return_map(x, kind=kind, prominence=prominence, distance=distance)
    channel = near_diagonal_tangent_channel(
        extrema.points,
        percentile=channel_percentile,
        min_points=min_channel_points,
    )
    return ReturnMapReconstruction(
        poincare=section,
        extrema=extrema,
        tangent_channel=channel,
    )


def intermittency_summary(
    x,
    *,
    fs=1.0,
    laminar_method="recurrence",
    eps=None,
    period=None,
    window=None,
    percentile=5.0,
    v_min=2,
    recurrence_metric="euclidean",
    powerlaw_gof_bootstrap=100,
    rng=None,
    burst_min_tail=None,
    return_map_kwargs=None,
):
    """Compose the intermittency building blocks without assigning a type label."""
    series = _finite_series(x)
    laminar_mask, lengths = detect_laminar_phases(
        series,
        method=laminar_method,
        eps=eps,
        period=period,
        window=window,
        percentile=percentile,
        v_min=v_min,
    )
    distribution = laminar_length_distribution(lengths)
    power_law = fit_power_law_mle(lengths)
    gof = powerlaw_gof(lengths, fit=power_law, n_bootstrap=powerlaw_gof_bootstrap, rng=rng)
    comparison = compare_powerlaw_exponential(lengths)
    recurrence = recurrence_laminar_lengths(
        series[:, np.newaxis],
        eps=eps,
        metric=recurrence_metric,
        percentile=percentile,
        v_min=v_min,
    )
    return_kwargs = {} if return_map_kwargs is None else dict(return_map_kwargs)
    return IntermittencySummary(
        laminar_mask=laminar_mask,
        laminar_lengths=lengths,
        laminar_distribution=distribution,
        laminar_power_law=power_law,
        powerlaw_gof=gof,
        family_comparison=comparison,
        recurrence_laminarity=recurrence,
        reinjection=reinjection_Mx(series, laminar_mask),
        burst_amplitude=burst_amplitude_distribution(series, laminar_mask, min_tail=burst_min_tail),
        laminar_burst_symmetry=laminar_burst_symmetry(series, laminar_mask),
        return_map=return_map_reconstruction(series, fs=fs, **return_kwargs),
    )


def _detect_laminar_recurrence(series, eps, percentile, v_min):
    result = recurrence_laminar_lengths(
        series[:, np.newaxis],
        eps=eps,
        percentile=percentile,
        v_min=v_min,
    )
    rmat, _ = recurrence_matrix(series[:, np.newaxis], eps=result.eps)
    mask = _vertical_structure_mask(rmat, v_min)
    return mask, result.lengths


def _detect_laminar_period(series, eps, period, percentile):
    if period is None:
        period = _auto_delay_from_autocorr(series)
    period = _positive_int(period, "period")
    if period >= series.size:
        raise ValueError("period must be shorter than x")

    diffs = np.abs(series[period:] - series[:-period])
    threshold = _threshold_from_percentile(diffs, eps, percentile)
    mask = np.zeros(series.size, dtype=bool)
    mask[period:] = diffs <= threshold
    lengths = _mask_run_lengths(mask)
    return mask, lengths


def _detect_laminar_variance(series, eps, window, percentile):
    if window is None:
        window = _auto_delay_from_autocorr(series)
    window = _positive_int(window, "window")
    if window > series.size:
        raise ValueError("window must be no longer than x")

    windows = np.lib.stride_tricks.sliding_window_view(series, window)
    local_std = np.std(windows, axis=1)
    threshold = _threshold_from_percentile(local_std, eps, percentile)
    window_mask = local_std <= threshold

    mask = np.zeros(series.size, dtype=bool)
    starts = np.nonzero(window_mask)[0]
    if starts.size:
        offsets = np.arange(window)
        mask[(starts[:, np.newaxis] + offsets).ravel()] = True
    lengths = _mask_run_lengths(mask)
    return mask, lengths


def _vertical_structure_mask(rmat, v_min):
    mask = np.zeros(rmat.shape[0], dtype=bool)
    for column in rmat.T:
        labels, n_labels = ndimage.label(column)
        if n_labels == 0:
            continue
        lengths = ndimage.sum(column, labels, index=np.arange(1, n_labels + 1))
        for label_id in np.nonzero(lengths >= v_min)[0] + 1:
            mask[labels == label_id] = True
    return mask


def _mask_run_lengths(mask):
    labels, n_labels = ndimage.label(np.asarray(mask, dtype=bool))
    if n_labels == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(ndimage.sum(mask, labels, index=np.arange(1, n_labels + 1)), dtype=np.int64)


def _threshold_from_percentile(values, eps, percentile):
    if eps is not None:
        threshold = float(eps)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("eps must be a finite non-negative number")
        return threshold

    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    return float(np.percentile(positive, percentile))


def _finite_series(x):
    series = np.asarray(x, dtype=np.float64).ravel()
    if series.size < 2:
        raise ValueError("x must contain at least two values")
    if not np.all(np.isfinite(series)):
        raise ValueError("x must contain only finite values")
    return series


def _positive_lengths(lengths):
    arr = np.asarray(lengths, dtype=np.int64).ravel()
    if arr.size == 0:
        return arr
    if np.any(arr < 1):
        raise ValueError("lengths must contain positive integers")
    return arr


def _positive_observations(values):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < 2:
        raise ValueError("at least two observations are required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("observations must contain only finite values")
    if np.any(arr <= 0.0):
        raise ValueError("observations must be positive")
    return arr


def _finite_mask(mask, size):
    arr = np.asarray(mask, dtype=bool).ravel()
    if arr.size != size:
        raise ValueError("mask must have the same length as x")
    return arr


def _finite_nonnegative_float(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value_int != value or value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _continuous_power_law_alpha(tail, x_min):
    denominator = np.sum(np.log(tail) - np.log(x_min))
    if denominator <= 0.0:
        raise ValueError("continuous power-law tail has zero logarithmic spread")
    return 1.0 + tail.size / denominator


def _discrete_power_law_alpha(tail, x_min):
    sum_log = float(np.sum(np.log(tail)))
    n = tail.size

    def score(alpha):
        step = _zeta_derivative_step(alpha)
        z_left = zeta(alpha - step, x_min)
        z_right = zeta(alpha + step, x_min)
        derivative = (np.log(z_right) - np.log(z_left)) / (2.0 * step)
        return -sum_log - n * derivative

    lower = 1.0 + 1e-6
    upper = 10.0
    try:
        while score(upper) > 0.0 and upper < 100.0:
            upper *= 2.0
        return float(brentq(score, lower, upper, maxiter=100))
    except ValueError:
        denominator = np.sum(np.log(tail / (x_min - 0.5)))
        if denominator <= 0.0:
            raise
        return float(1.0 + n / denominator)


def _bounded_histogram_bin_edges(values, *, max_bins=512):
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        return np.empty(0, dtype=np.float64)
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    if data_min == data_max:
        width = abs(data_min) * 0.5 if data_min != 0.0 else 0.5
        return np.array([data_min - width, data_max + width], dtype=np.float64)

    max_bins = _positive_int(max_bins, "max_bins")
    q25, q75 = np.percentile(data, [25.0, 75.0])
    iqr = q75 - q25
    data_range = data_max - data_min
    if iqr > 0.0:
        log_fd_width = np.log(2.0) + np.log(iqr) - np.log(np.cbrt(data.size))
        log_bin_ratio = np.log(data_range) - log_fd_width
        if log_bin_ratio > np.log(max_bins):
            n_bins = max_bins + 1
        else:
            n_bins = int(np.ceil(np.exp(log_bin_ratio)))
    else:
        n_bins = int(np.ceil(np.sqrt(data.size)))

    if n_bins > max_bins and data_min > 0.0 and data_max / data_min > 1e6:
        with np.errstate(under="ignore"):
            return np.geomspace(data_min, data_max, max_bins + 1, dtype=np.float64)

    n_bins = min(max(n_bins, 1), max_bins)
    return np.linspace(data_min, data_max, n_bins + 1, dtype=np.float64)


def _finite_histogram_mass(values, bin_edges):
    counts, bin_edges = np.histogram(values, bins=bin_edges, density=False)
    total = np.sum(counts)
    if total == 0:
        mass = np.zeros_like(counts, dtype=np.float64)
    else:
        mass = counts.astype(np.float64) / total
    return mass, bin_edges


def _zeta_derivative_step(alpha):
    distance_to_pole = alpha - 1.0
    if distance_to_pole <= 0.0:
        raise ValueError("discrete power-law alpha must be greater than one")
    return min(max(1e-6, alpha * 1e-5), 0.5 * distance_to_pole)


def _power_law_ks(tail, x_min, csn_alpha, discrete):
    tail = np.sort(np.asarray(tail, dtype=np.float64))
    empirical = np.arange(1, tail.size + 1, dtype=np.float64) / tail.size
    model = _power_law_cdf(tail, x_min, csn_alpha, discrete)
    return float(np.max(np.abs(empirical - model)))


def _power_law_cdf(values, x_min, csn_alpha, discrete):
    values = np.asarray(values, dtype=np.float64)
    if discrete:
        floors = np.floor(values)
        return 1.0 - zeta(csn_alpha, floors + 1.0) / zeta(csn_alpha, x_min)
    log_ratio = np.log(values) - np.log(x_min)
    return 1.0 - np.exp((1.0 - csn_alpha) * log_ratio)


def _power_law_logpdf(values, x_min, csn_alpha, discrete):
    values = np.asarray(values, dtype=np.float64)
    if discrete:
        return -csn_alpha * np.log(values) - np.log(zeta(csn_alpha, x_min))
    log_ratio = np.log(values) - np.log(x_min)
    return np.log(csn_alpha - 1.0) - np.log(x_min) - csn_alpha * log_ratio


def _sample_power_law_tail(fit, rng, size):
    if fit.discrete:
        samples = rng.zipf(fit.csn_alpha, size=size)
        while True:
            bad = samples < fit.x_min
            if not np.any(bad):
                return samples.astype(np.float64)
            samples[bad] = rng.zipf(fit.csn_alpha, size=np.count_nonzero(bad))

    u = rng.random(size)
    samples = fit.x_min * (1.0 - u) ** (-1.0 / (fit.csn_alpha - 1.0))
    return samples.astype(np.float64)


def _fit_inverse_epsilon_candidate(eps, mean_lengths):
    design = (1.0 / eps)[:, np.newaxis]
    coefficients, *_ = np.linalg.lstsq(design, mean_lengths, rcond=None)
    residual = mean_lengths - design @ coefficients
    return CandidateScalingLaw(
        name="eps^-1",
        coefficients=coefficients.astype(np.float64),
        residual_sum_squares=float(np.sum(residual * residual)),
    )


def _fit_logarithmic_candidate(eps, mean_lengths):
    x = np.log(1.0 / eps)
    result = linregress(x, mean_lengths)
    residual = mean_lengths - (result.intercept + result.slope * x)
    return CandidateScalingLaw(
        name="log(1/eps)",
        coefficients=np.array([result.intercept, result.slope], dtype=np.float64),
        residual_sum_squares=float(np.sum(residual * residual)),
    )


__all__ = [
    "CandidateScalingLaw",
    "BurstAmplitudeDistribution",
    "DiagnosticLogLogFit",
    "ExponentialFit",
    "LaminarLengthDistribution",
    "LaminarBurstSymmetry",
    "MeanLaminarScaling",
    "PowerLawGoF",
    "PowerLawMLE",
    "ReinjectionMx",
    "ExtremaReturnMap",
    "IntermittencySummary",
    "ReturnMapReconstruction",
    "TangentChannel",
    "VuongComparison",
    "compare_powerlaw_exponential",
    "burst_amplitude_distribution",
    "detect_laminar_phases",
    "extrema_return_map",
    "fit_exponential",
    "fit_power_law_loglog",
    "fit_power_law_mle",
    "intermittency_summary",
    "laminar_length_distribution",
    "laminar_burst_symmetry",
    "mean_laminar_scaling",
    "near_diagonal_tangent_channel",
    "pooled_laminar_lengths",
    "powerlaw_alpha_ci",
    "powerlaw_gof",
    "reinjection_Mx",
    "return_map_reconstruction",
]
