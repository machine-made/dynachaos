# Reduced preview (N = 1000, 3 perturbations). Full figure: dynachaos run sec10_gcm
# Finite-size Lyapunov exponent (FSLE) of the mean field of the heterogeneous GCM
# a(i) = 1.9 + 0.05 (2i - N) / (2N), eps = 0.098 (Shibata & Kaneko 1998, Figs. 3-5):
# lambda(Delta) = log(Delta / delta0) / <tau(Delta)>, tau = first time |h' - h| reaches Delta.
# Full figure: N = 1e4, 1e5, 1e6, 100 perturbations; Eq. (6) fits give lambda_C, Delta_m.
# Larger sizes: python -m dynachaos.cml.gcm_collective_fsle --add 10000000
import numpy as np

from dynachaos.cml.gcm_collective_fsle import (
    CHAOS,
    DELTAS,
    a_profile,
    first_passage,
    lambda_m,
    step,
)

N = 1000
delta0 = 1e-7
a = a_profile(CHAOS["a0"], CHAOS["delta_a"], N)
rng = np.random.default_rng(N)
x = rng.uniform(-1.0, 1.0, N)
for _ in range(2000):
    x, _ = step(x, a, CHAOS["eps"])

taus = []
for _ in range(3):
    tau, x = first_passage(x, a, CHAOS["eps"], delta0, rng, 3000)
    taus.append(tau)
mean_tau = np.mean(taus, axis=0)

lam_m = lambda_m(a, CHAOS["eps"], x, n_transient=500, n_measure=2000)
print(f"lambda_m = {lam_m:.3f} (tangent map)")
for D in (1e-5, 1e-4, 1e-3, 1e-2):
    k = int(np.argmin(np.abs(np.log(DELTAS / D))))
    if mean_tau[k] <= 3000:
        print(
            f"Delta={DELTAS[k]:.1e}  lambda(Delta)={np.log(DELTAS[k] / delta0) / mean_tau[k]:.3f}"
        )
