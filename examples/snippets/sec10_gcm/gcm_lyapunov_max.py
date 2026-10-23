# Reduced preview (N = 200, 3 a values, 1 seed). Full figure: dynachaos run sec10_gcm
# Largest Lyapunov exponent lambda_max of the identical-element GCM (eps = 0.1) by the tangent map:
# v -> (1 - eps) f'(x) v + eps <f'(x) v>, renormalised every step. This is the microscopic
# (site) exponent, not the collective exponent of the mean field.
# Full figure: 8 seeds for N = 500, 1000, 2000 over a in [1.40, 2.00].
from dynachaos.cml.gcm_lyapunov_max import lyapunov_max

for a in (1.50, 1.80, 1.95):
    lam = lyapunov_max(a, 0.1, 200, n_transient=500, n_measure=2000, seed=42)
    print(f"a={a:.2f}  lambda_max={lam:+.4f}")
