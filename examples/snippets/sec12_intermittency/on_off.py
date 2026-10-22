# Figure: on-off intermittency -- a logistic-driven skew-product whose
# transverse Lyapunov exponent lambda_perp = log(2*eps) approaches zero as
# eps -> 0.5 (blowout onset); the transverse amplitude alternates between
# quiescent ("off", below the 90th percentile) and bursting phases. We
# print lambda_perp and the mean off-phase (laminar) run length observed
# in each short realization.
import numpy as np

from dynachaos.maps.intermittency import ON_OFF_SKEW_LOGISTIC_ONSET, on_off_skew_logistic_oracle


def mean_run_length(mask):
    lengths, run = [], 0
    for flag in mask:
        if flag:
            run += 1
        elif run > 0:
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return np.mean(lengths) if lengths else 0.0


for eps in (0.45, 0.48, ON_OFF_SKEW_LOGISTIC_ONSET - 0.01):
    skew = on_off_skew_logistic_oracle(20_000, x0=0.217, y0=1e-2, eps=eps)
    y = np.abs(skew[1000:, 1])
    off_mask = y <= np.percentile(y, 90)
    lambda_perp = np.log(2.0 * eps)
    print(
        f"eps={eps:.3f}  lambda_perp={lambda_perp:+.4f}  "
        f"mean_off_run_length={mean_run_length(off_mask):.1f}"
    )

# Full figure: dynachaos run sec12_intermittency
