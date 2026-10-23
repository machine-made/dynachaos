# Reduced preview (4 a values of the logistic map). Full figure: dynachaos run sec11_diagnostics
# Label each parameter by the sign of lambda_1, then compare K_01 with the label.
import numpy as np

from dynachaos.diagnostics.agreement import LABEL_NAMES, label_from_spectrum, roc_auc
from dynachaos.diagnostics.lyapunov import lyapunov_spectrum_blocks
from dynachaos.diagnostics.zero_one_test import zero_one_statistic
from dynachaos.maps.primitives import logistic

labels, k01 = [], []
for a in (1.1, 1.4, 1.75, 1.9):
    f = lambda s, a=a: np.array([logistic(s[0], a)])  # noqa: E731
    jac = lambda s, a=a: np.array([[-2.0 * a * s[0]]])  # noqa: E731
    spectrum, se = lyapunov_spectrum_blocks(f, jac, np.array([0.1]), 20_000, 2000)
    x, series = 0.1, np.empty(5000)
    for _ in range(2000):
        x = logistic(x, a)
    for i in range(5000):
        x = logistic(x, a)
        series[i] = x
    labels.append(label_from_spectrum(spectrum, se))
    k01.append(zero_one_statistic(series, n_c=50))
    print(f"a={a:.2f}  lambda_1={spectrum[0]:+.4f}  {LABEL_NAMES[labels[-1]]}  K_01={k01[-1]:.3f}")

print(f"AUC of K_01 on these points: {roc_auc(k01, np.array(labels) == 1):.2f}")
