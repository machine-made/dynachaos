# Reduced preview (N = 60, 4 a values). Full figure: dynachaos run sec08_sti
# Spatial autocorrelation C(r)/C(0) of a logistic coupled map lattice (eps = 0.3).
# 4 a values (1.50, 1.70, 1.85, 1.95); static figure adds panel (b), the KS entropy density
# h_KS(L)/L, the sum of the positive Lyapunov exponents of an L-site subsystem divided by L.
import numpy as np

from dynachaos.cml.correlation_figure import compute_ks_entropy_density
from dynachaos.cml.primitives import cml_step_logistic as cml_step

N = 60
eps = 0.3
rng = np.random.default_rng(42)
for a in (1.50, 1.70, 1.85, 1.95):
    x = rng.uniform(-0.5, 0.5, N)
    for _ in range(500):
        x = cml_step(x, a, eps)
    snapshots = np.empty((200, N))
    for s in range(200):
        x = cml_step(x, a, eps)
        snapshots[s] = x
    fluct = snapshots - snapshots.mean()
    fft_f = np.fft.rfft(fluct, axis=1)
    power = (fft_f * np.conj(fft_f)).real
    full_corr = np.fft.irfft(power, n=N, axis=1).mean(axis=0) / N
    c_norm = full_corr / full_corr[0]
    print(f"a={a:.2f}  C(r=0..3)/C(0)={np.round(c_norm[:4], 3)}")

# h_KS(L)/L on a small lattice (N = 40, L = 10 and 20, 400 steps): the chaotic states carry
# entropy per site, the pattern-selection state (a = 1.70) very little.
a_values, L_values, h_density = compute_ks_entropy_density(
    N=40, a_values=(1.70, 1.95), L_values=(10, 20), n_transient=300, n_iter=400
)
for a, row in zip(a_values, h_density, strict=True):
    print(f"a={a:.2f}  h_KS(L)/L for L={L_values.tolist()}: {np.round(row, 4)}")

# Full figure: dynachaos run sec08_sti
