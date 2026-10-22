# Figure: spacetime diagrams of a coupled map lattice (Kaneko 1985 Model C,
# logistic local map) at three couplings spanning frozen, pattern, and
# turbulent regimes -- shown here via each snapshot's spatial variance.
from dynachaos.cml.spatiotemporal import model_C_f, simulate_cml

N = 60
for eps in (0.16, 0.20, 0.30):
    spacetime = simulate_cml(model_C_f, model_C_f, eps, N=N, n_transient=200, n_record=50)
    print(f"eps={eps:.2f}  final-row var={spacetime[-1].var():.4f}")

# Full figure: dynachaos run sec08_sti
