"""Map definitions for dynamical systems."""

from dynachaos.maps.base import (
    BifurcationData,
    CircleMap,
    DiscreteMap,
    HenonMap,
    LogisticMap,
    StandardMap,
)
from dynachaos.maps.circle_map import circle_map, circle_map_derivative, rotation_number
from dynachaos.maps.coupled_delayed import coupled_delayed, coupled_delayed_jac
from dynachaos.maps.coupled_logistic import coupled_logistic, coupled_logistic_jac
from dynachaos.maps.flows import (
    lorenz_jac,
    lorenz_rhs,
    lorenz_trajectory,
    mackey_glass_series,
    rossler_jac,
    rossler_rhs,
    rossler_trajectory,
)
from dynachaos.maps.henon import henon, henon_jac
from dynachaos.maps.intermittency import (
    LOGISTIC_TYPE_I_ONSET,
    LORENZ_INTERMITTENCY_RHO,
    ON_OFF_SKEW_LOGISTIC_ONSET,
    logistic_type_i_oracle,
    lorenz_1662_oracle,
    on_off_oracle,
    on_off_skew_logistic_oracle,
    pm_type_i_oracle,
    pm_type_ii_oracle,
    pm_type_iii_oracle,
)
from dynachaos.maps.modulated_circle import modulated_circle, rotation_numbers
from dynachaos.maps.primitives import (
    delayed_logistic,
    delayed_logistic_jac,
    logistic,
    logistic_derivative,
)
from dynachaos.maps.standard_map import standard_map, standard_map_jac
from dynachaos.maps.torus_doubling import map_I, map_I_jac, map_IV, map_IV_jac

__all__ = [
    "BifurcationData",
    "CircleMap",
    "DiscreteMap",
    "HenonMap",
    "LOGISTIC_TYPE_I_ONSET",
    "LORENZ_INTERMITTENCY_RHO",
    "ON_OFF_SKEW_LOGISTIC_ONSET",
    "LogisticMap",
    "StandardMap",
    "circle_map",
    "circle_map_derivative",
    "coupled_delayed",
    "coupled_delayed_jac",
    "coupled_logistic",
    "coupled_logistic_jac",
    "delayed_logistic",
    "delayed_logistic_jac",
    "henon",
    "henon_jac",
    "logistic",
    "logistic_derivative",
    "logistic_type_i_oracle",
    "lorenz_1662_oracle",
    "lorenz_jac",
    "lorenz_rhs",
    "lorenz_trajectory",
    "mackey_glass_series",
    "map_I",
    "map_I_jac",
    "map_IV",
    "map_IV_jac",
    "modulated_circle",
    "on_off_oracle",
    "on_off_skew_logistic_oracle",
    "pm_type_i_oracle",
    "pm_type_ii_oracle",
    "pm_type_iii_oracle",
    "rossler_jac",
    "rossler_rhs",
    "rossler_trajectory",
    "rotation_number",
    "rotation_numbers",
    "standard_map",
    "standard_map_jac",
]
