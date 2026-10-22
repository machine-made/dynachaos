# Reduced preview (N = 256, 100 steps). Full figure: dynachaos run sec12_intermittency
# Quantitative STI spine for Kaneko Model-A CML: turbulent fraction vs coupling eps, the critical
# coupling eps_c, and maximum-likelihood fits of the laminar-length tail (power law, exponential,
# truncated power law).
# Full figure: N = 512, phi_turb at 19 couplings in [0.03, 0.12], fits at eps_c, 0.060 and 0.080.
import numpy as np

from dynachaos.cml.spatiotemporal import model_A_f, simulate_cml
from dynachaos.cml.sti_spine_figure import (
    _burst_mask,
    _periodic_run_lengths,
    fit_tail_models,
    locate_critical_coupling,
)

N = 256
sweep_eps = np.linspace(0.03, 0.12, 10)
fractions = []
for idx, eps in enumerate(sweep_eps):
    rng = np.random.default_rng(42 + idx)
    spacetime = simulate_cml(
        model_A_f,
        model_A_f,
        float(eps),
        N=N,
        n_transient=500,
        n_record=100,
        x0=rng.uniform(0, 1, N),
    )
    fractions.append(_burst_mask(spacetime).mean())
    print(f"eps={eps:.3f}  turbulent_fraction={fractions[-1]:.3f}")
print(f"eps_c = {locate_critical_coupling(sweep_eps, fractions):.3f}")

lengths = _periodic_run_lengths(~_burst_mask(spacetime))
fit = fit_tail_models(lengths, x_min=2)
print(f"laminar tail at eps=0.12: n={fit['n_tail']}  preferred={fit['preferred']}")
