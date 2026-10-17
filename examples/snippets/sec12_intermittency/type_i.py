# Type-I intermittency: tangent bifurcation and laminar-phase statistics.
# Full figure: dynachaos run sec12_intermittency
from dynachaos.diagnostics.intermittency import detect_laminar_phases
from dynachaos.maps.intermittency import LOGISTIC_TYPE_I_ONSET, logistic_type_i_oracle

series = logistic_type_i_oracle(20_000, x0=0.2, r=LOGISTIC_TYPE_I_ONSET - 1e-4)
_, lengths = detect_laminar_phases(series, method="period", period=3, percentile=70.0)
print(
    f"r_c={LOGISTIC_TYPE_I_ONSET:.4f}  n_laminar_phases={len(lengths)}  "
    f"mean_length={lengths.mean():.2f}"
)
