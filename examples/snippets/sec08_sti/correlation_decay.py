# Figure: spatial autocorrelation C(r)/C(0) of a logistic coupled map
# lattice, decaying from 1 as separation r grows -- faster decay signals a
# shorter correlation length as the nonlinearity a increases toward chaos.
import numpy as np

from dynachaos.cml.primitives import cml_step_logistic as cml_step

N = 60
eps = 0.3
rng = np.random.default_rng(42)
for a in (1.5, 1.85, 1.95):
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

# Full figure: dynachaos run sec08_sti
