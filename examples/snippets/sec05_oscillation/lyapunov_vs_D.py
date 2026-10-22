# Reduced preview (3 D values, 2 power spectra). Full figure: dynachaos run sec05_oscillation
# Panel (a): Lyapunov spectrum of the delayed logistic map vs D at alpha=0.3.
# Panel (b): Power spectrum S(f) harmonics/sidebands (f0 approx 0.225) at D=1.70 and D=1.86.
import numpy as np

from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
from dynachaos.maps.delayed_logistic import compute_psd, delayed_logistic, delayed_logistic_jac

alpha = 0.3
for D in (1.6, 1.95, 2.2):
    fp = (np.sqrt(1.0 + 4.0 * D) - 1.0) / (2.0 * D)
    x0 = np.array([fp + 0.01, fp - 0.01])

    def f(state, D=D):
        return delayed_logistic(state, alpha, D)

    def jac(state, D=D):
        return delayed_logistic_jac(state, alpha, D)

    spectrum = lyapunov_spectrum(f, jac, x0, n_iter=4000, n_transient=2000)
    print(f"D={D:.2f}  lambda_1={spectrum[0]:+.4f}  lambda_2={spectrum[1]:+.4f}")

# Power spectrum peak / harmonic check at D = 1.70 and D = 1.86 (rotation f0 approx 0.225)
for d_val in (1.70, 1.86):
    freqs, psd = compute_psd(alpha, d_val, n_transient=2000, n_record=2048)
    print(f"D={d_val:.2f}  dominant harmonic={freqs[np.argmax(psd)]:.3f}")
