"""Type stubs for the optional Rust extension module."""

import numpy as np
import numpy.typing as npt

def diagonal_lines(
    r: npt.NDArray[np.bool_],
    l_min: int = 2,
) -> npt.NDArray[np.int64]: ...
def vertical_lines(
    r: npt.NDArray[np.bool_],
    v_min: int = 2,
) -> npt.NDArray[np.int64]: ...
def count_line_lengths(
    mask: npt.NDArray[np.bool_],
    min_length: int,
) -> npt.NDArray[np.int64]: ...
def ordinal_distribution(
    x: npt.NDArray[np.float64],
    d: int = 5,
    tau: int = 1,
) -> tuple[npt.NDArray[np.int64], int]: ...
def ami_histogram(
    x: npt.NDArray[np.float64],
    tau_max: int,
    n_bins: int = 64,
) -> npt.NDArray[np.float64]: ...
def select_dimension_cao(
    e1: npt.NDArray[np.float64],
    near_one_lower: float = 0.95,
    near_one_upper: float = 1.05,
    saturation_tol: float = 0.02,
    plateau_span: int = 3,
    smoothing_window: int = 1,
    min_dim: int = 2,
    max_dim: int | None = None,
) -> int: ...
def correlation_counts(
    traj: npt.NDArray[np.float64],
    r_values: npt.NDArray[np.float64],
    theiler_window: int = 0,
    use_chebyshev: bool = True,
) -> npt.NDArray[np.int64]: ...
def cml_jacobian_logistic(
    x: npt.NDArray[np.float64],
    a: float,
    eps: float,
    L: int,
) -> npt.NDArray[np.float64]: ...
def comoving_lyapunov_logistic(
    x_init: npt.NDArray[np.float64],
    v_values: npt.NDArray[np.float64],
    a: float,
    eps: float,
    n_iter: int,
    n_transient: int,
) -> npt.NDArray[np.float64]: ...
def coupled_logistic_basin_grid(
    x_values: npt.NDArray[np.float64],
    y_values: npt.NDArray[np.float64],
    A: float,
    D: float,
    n_transient: int,
    ref_a: npt.NDArray[np.float64],
) -> npt.NDArray[np.int8]: ...
def coupled_logistic_phase_tile(
    a_values: npt.NDArray[np.float64],
    d_values: npt.NDArray[np.float64],
    n_transient: int,
    n_sample: int,
    x0: float,
    y0: float,
) -> npt.NDArray[np.float64]: ...
def coupled_logistic_attractor_tile(
    a_values: npt.NDArray[np.float64],
    D: float,
    n_transient: int,
    n_plot: int,
    state0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def apen_counts(
    traj: npt.NDArray[np.float64],
    r: float,
) -> npt.NDArray[np.int64]: ...
def fuzzy_entropy_sum(
    traj: npt.NDArray[np.float64],
    r: float,
    n: int,
    theiler_window: int = 0,
) -> float: ...
def multifractal_moments(
    field: npt.NDArray[np.float64],
    box_sizes: npt.NDArray[np.int64],
    q_values: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...
def pm_type_i_oracle(
    n: int,
    x0: float,
    eps: float,
    a: float,
    modulo: bool,
) -> npt.NDArray[np.float64]: ...
def pm_type_ii_oracle(
    n: int,
    x0: float,
    y0: float,
    eps: float,
    a: float,
    theta: float,
) -> npt.NDArray[np.float64]: ...
def pm_type_iii_oracle(
    n: int,
    x0: float,
    eps: float,
    a: float,
) -> npt.NDArray[np.float64]: ...
def on_off_oracle(
    driver: npt.NDArray[np.float64],
    x0: float,
    transverse_lyapunov: float,
    noise_scale: float,
) -> npt.NDArray[np.float64]: ...
def on_off_skew_logistic_oracle(
    n: int,
    x0: float,
    y0: float,
    eps: float,
) -> npt.NDArray[np.float64]: ...
def logistic_type_i_oracle(
    n: int,
    x0: float,
    r: float,
) -> npt.NDArray[np.float64]: ...
def delayed_logistic_attractor_tile(
    A: float,
    d_values: npt.NDArray[np.float64],
    n_transient: int,
    n_plot: int,
    state0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def torus_doubling_attractor_tile(
    map_kind: int,
    A: float,
    D: float,
    n_transient: int,
    n_plot: int,
    state0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def modulated_circle_rotation_tile(
    A: float,
    C: float,
    d_values: npt.NDArray[np.float64],
    eps: float,
    n_transient: int,
    n_iter: int,
    state0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def cml_spacetime_tile(
    model: int,
    eps: float,
    n_transient: int,
    n_record: int,
    x0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def circle_map_lyapunov_sum(
    A: float,
    D: float,
    n_transient: int,
    n_iter: int,
    theta0: float,
) -> float: ...
def delayed_logistic_lyapunov_tile(
    A: float,
    d_values: npt.NDArray[np.float64],
    n_transient: int,
    n_iter: int,
    state0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def torus_doubling_lyapunov_tile(
    map_kind: int,
    A: float,
    d_values: npt.NDArray[np.float64],
    n_transient: int,
    n_iter: int,
    state0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def coupled_delayed_lyapunov_tile(
    A: float,
    db_values: npt.NDArray[np.float64],
    eps: float,
    n_transient: int,
    n_iter: int,
    state0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def coupled_delayed_projection_tile(
    A: float,
    db_values: npt.NDArray[np.float64],
    eps: float,
    n_transient: int,
    n_plot: int,
    state0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
