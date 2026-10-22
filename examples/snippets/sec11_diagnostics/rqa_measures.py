# Figure: RQA measures (RR, DET, LAM, ENTR) along the delayed logistic map.
import numpy as np

from dynachaos.diagnostics.recurrence import recurrence_matrix, rqa
from dynachaos.maps.primitives import delayed_logistic

A = 0.3
for D in (1.6, 1.9, 2.1):
    state = np.array([0.51, 0.49])
    for _ in range(2000):
        state = delayed_logistic(state, A, D)
    traj = np.empty((500, 2))
    for i in range(500):
        state = delayed_logistic(state, A, D)
        traj[i] = state
    R, _ = recurrence_matrix(traj, percentile=5)
    stats = rqa(R, l_min=2, v_min=2)
    print(f"D={D:.2f}  RR={stats['RR']:.3f}  DET={stats['DET']:.3f}  LAM={stats['LAM']:.3f}")

# Full figure: dynachaos run sec11_diagnostics
