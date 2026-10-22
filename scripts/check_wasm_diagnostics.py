"""Check that the diagnostics and map WebAssembly exports match the native build.

The map and lattice exports compared are ``delayed_logistic_attractor_tile``,
``torus_doubling_attractor_tile`` (maps I and IV),
``modulated_circle_rotation_tile``, ``cml_spacetime_tile`` (models A, B and
C), the four Lyapunov kernels ``circle_map_lyapunov_sum``,
``delayed_logistic_lyapunov_tile``, ``torus_doubling_lyapunov_tile`` (maps I
and IV) and ``coupled_delayed_lyapunov_tile``, and the five trajectory and
grid exports ``coupled_logistic_phase_tile``,
``coupled_logistic_attractor_tile``, ``coupled_logistic_basin_grid``,
``coupled_delayed_projection_tile`` and ``fractalization_attractor_tile``.
The native side is the pyo3 extension, which runs the same ``rust/core``
code; the browser side is the ``site/wasm`` bundle driven through node.
Both sides receive the same inputs, built once below.

The tolerance is set per export by what the kernel computes:

* Counts — correlation pairs, ApEn template matches, ordinal patterns,
  recurrence line lengths — are integers produced by comparisons and
  additions. They must match exactly. The Cao dimension is an integer and
  must match exactly too.
* ``fuzzy_entropy_sum`` accumulates ``exp(-(d / r)^n)`` over the pairs. The
  wasm module carries its own ``exp`` and sums sequentially, while the
  native build folds per-thread partial sums, so the last bits can differ.
  Measured 1.1e-15 relative on the fixed input below; the limit leaves room
  for another platform ``exp`` without letting a defect through.
* ``ami_histogram`` and ``multifractal_moments`` evaluate ``ln`` (and
  ``powf`` for the moments) inside the kernel, so the same last-bit argument
  applies. They are held to the same 1e-12 relative tolerance.
* ``zero_one_k`` forms the mean-square displacement from an FFT
  autocovariance and correlates it with the lag. The wasm module's own
  ``sin``/``cos`` and FFT differ in the last bits from the native ones, so
  the per-c K values are held to 1e-9 absolute — loose enough for a
  transcendental last-bit difference, tight enough that an all-zero kernel
  (which returns exact zeros) fails the median check below.
* ``modulated_circle_rotation_tile`` and model (B) of
  ``cml_spacetime_tile`` evaluate ``sin`` inside the kernel. The wasm module
  carries its own sine, which can differ from the platform one in the last
  bit; on this build the measured difference is 0 for the rotation numbers
  and 4.6e-13 for the model-(B) field over the horizon below. They are held
  to 1e-12 and 1e-9 absolute respectively — loose enough for a last-bit sine
  difference amplified over the recorded steps, tight enough that a wrong
  formula fails. Models (A) and (C) carry no transcendental call and must
  match exactly, as must both torus maps and the delayed logistic.
* The five Lyapunov comparisons accumulate one ``ln`` per step and a QR
  renormalisation. The wasm module's own ``ln`` can differ in the last
  bits; the frame renormalises every step, so the difference does not
  grow. They are held to 1e-9 absolute, loose enough for the last-bit
  logarithm difference, tight enough that a wrong Jacobian fails.
* ``coupled_logistic_phase_tile`` accumulates one ``ln`` and one ``sqrt``
  per sample for the Lyapunov column. That column is held to 1e-9
  absolute. The asymmetry column, and the attractor, projection,
  fractalization and basin exports, use only ``+``, ``-`` and ``*`` (basin
  labels are integers) and must match exactly.

Each export also gets a self-check: one value of the native result is
perturbed and the comparison must fail, so a script that compares nothing
cannot pass. ``zero_one_k`` gets a stronger guard instead: the median K on
the logistic map at r = 4 must exceed 0.9, so an all-zero wasm kernel fails
even though every element compares equal.

Run it with::

    uv run python scripts/check_wasm_diagnostics.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from math import isfinite
from pathlib import Path

import numpy as np

from dynachaos._rust import (
    ami_histogram,
    apen_counts,
    circle_map_lyapunov_sum,
    cml_spacetime_tile,
    correlation_counts,
    coupled_delayed_lyapunov_tile,
    coupled_delayed_projection_tile,
    coupled_logistic_attractor_tile,
    coupled_logistic_basin_grid,
    coupled_logistic_phase_tile,
    delayed_logistic_attractor_tile,
    delayed_logistic_lyapunov_tile,
    diagonal_lines,
    fuzzy_entropy_sum,
    modulated_circle_rotation_tile,
    multifractal_moments,
    ordinal_distribution,
    select_dimension_cao,
    torus_doubling_attractor_tile,
    torus_doubling_lyapunov_tile,
    vertical_lines,
    zero_one_k,  # type: ignore[attr-defined]
)
from dynachaos.diagnostics.recurrence import embed_time_delay, recurrence_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_WASM = PROJECT_ROOT / "site" / "wasm"

# The fixed input every comparison below runs on: a deterministic chaotic
# series (logistic map at r = 4) embedded once and shared by both sides.
N_SAMPLES = 2000
EMBED_DIM = 3
EMBED_TAU = 2
THEILER = 0
R_VALUES = [0.02, 0.05, 0.1, 0.2, 0.4]
APEN_R = 0.2
FUZZY_R = 0.2
FUZZY_N = 2
ORDINAL_D = 3
ORDINAL_TAU = 2

# Recurrence lines: a small matrix keeps the node message small. The mask is
# sent as 0/1 bytes, row-major, with its side length.
RQA_SIDE = 60
RQA_PERCENTILE = 10
L_MIN = 2
V_MIN = 2

# Multifractal: a small nonnegative field, box sizes that divide it, and a
# few q values. Sizes that divide the field keep every scale valid so no
# NaN crosses the JSON boundary.
MF_NY = 16
MF_NX = 16
MF_BOXES = [2, 4, 8]
MF_Q = [0.0, 1.0, 2.0]

# AMI: a modest lag and bin count.
AMI_TAU_MAX = 20
AMI_BINS = 16

# Cao: an E1 curve that plateaus at 1 from dimension 3 on.
CAO_E1 = [0.35, 0.62, 0.97, 1.0, 1.0, 1.0, 1.0, 1.0]

# zero_one_k: a handful of frequencies over the same series.

# Map-attractor trajectories: bounded, evenly spaced D values keep JSON finite
# and exercise the flat headers without making this diagnostic heavyweight.
MAP_TRANSIENT = 100
MAP_PLOT = 1024
DELAYED_A = 0.3
DELAYED_D_MIN = 1.55
DELAYED_D_MAX = 2.16
DELAYED_N_D = 3
DELAYED_STATE = [0.4, 0.35]
TORUS_I_KIND = 1
TORUS_I_A = 0.4
TORUS_I_D = 2.19
TORUS_I_STATE = [0.5, 0.5, 0.5]
TORUS_KIND = 4
TORUS_A = 0.3
TORUS_D = 1.5212
TORUS_STATE = [0.5, 0.45, 0.52, 0.48]

# Modulated circle: a modest sweep keeps the node message small; the rotation
# numbers are well conditioned at the subcritical A the paper uses.
CIRCLE_A = 0.1
CIRCLE_C = (np.sqrt(5) - 1) / 2
CIRCLE_D_MIN = 0.0
CIRCLE_D_MAX = 1.0
CIRCLE_N_D = 16
CIRCLE_EPS = 0.05
CIRCLE_TRANSIENT = 200
CIRCLE_ITER = 2000
# The four Lyapunov kernels. The spectra accumulate a `log` per step, so the
# wasm module's own `ln` can differ in the last bits from the native one;
# the QR frame is renormalised every step so the difference does not grow.
# A small sweep and a short iter keep the node message small while covering
# every dimension: 1 (circle), 2 (delayed), 3 and 4 (torus I/IV), 4
# (coupled). All five exports are held to 1e-9 absolute.
LYAP_ABS_TOLERANCE = 1e-9
LYAP_TRANSIENT = 200
LYAP_ITER = 2000
CIRCLE_LYAP_A = 0.1
CIRCLE_LYAP_D = 0.25
CIRCLE_LYAP_THETA0 = 0.1
DELAYED_LYAP_A = 0.3
DELAYED_LYAP_D_MIN = 1.3
DELAYED_LYAP_D_MAX = 1.7
DELAYED_LYAP_N_D = 4
DELAYED_LYAP_STATE = [0.4, 0.35]
TORUS_I_LYAP_A = 0.4
TORUS_I_LYAP_D = 2.19
TORUS_I_LYAP_STATE = [0.5, 0.5, 0.5]
TORUS_LYAP_A = 0.3
TORUS_LYAP_D_MIN = 1.48
TORUS_LYAP_D_MAX = 1.53
TORUS_LYAP_N_D = 4
TORUS_LYAP_STATE = [0.5, 0.45, 0.52, 0.48]
COUPLED_LYAP_A = 0.4
COUPLED_LYAP_DB_MIN = 2.1
COUPLED_LYAP_DB_MAX = 2.4
COUPLED_LYAP_N_DB = 4
COUPLED_LYAP_EPS = 5e-3
COUPLED_LYAP_STATE = [0.5, 0.5, 0.3, 0.3]

# Trajectory and grid exports. Phase Lyapunov uses sqrt and ln, so that
# column allows a last-bit wasm/native difference. The other four are
# polynomial or integer labels and must match exactly.
PHASE_A_MIN = 0.8
PHASE_A_MAX = 1.2
PHASE_N_A = 4
PHASE_D_MIN = 0.0
PHASE_D_MAX = 0.2
PHASE_N_D = 3
PHASE_TRANSIENT = 20
PHASE_SAMPLE = 40
PHASE_X0 = 0.1
PHASE_Y0 = 0.2
PHASE_LYAP_ABS = 1e-9
ATTRACTOR_A_MIN = 1.0
ATTRACTOR_A_MAX = 1.3
ATTRACTOR_N_A = 3
ATTRACTOR_D = 0.1
ATTRACTOR_X0 = 0.1
ATTRACTOR_Y0 = 0.2
BASIN_A = 1.35344
BASIN_D = 0.1
BASIN_N = 8
BASIN_TRANSIENT = 12
BASIN_REFERENCE = 20
BASIN_PERIOD = 4
BASIN_X_REF = 0.1
BASIN_Y_REF = 0.6
PROJECTION_A = 0.4
PROJECTION_DB_MIN = 2.2
PROJECTION_DB_MAX = 2.5
PROJECTION_N_DB = 3
PROJECTION_EPS = 5e-3
PROJECTION_STATE = [0.5, 0.5, 0.3, 0.3]
FRACTAL_A = 0.3
FRACTAL_D_MIN = 1.8
FRACTAL_D_MAX = 1.9
FRACTAL_N_D = 3
FRACTAL_STATE = [0.4, 0.35]

CIRCLE_STATE = [0.1, 0.1]

# CML space-time: a 64-site lattice over a short record. Model (B) carries a
# sine, so its field is compared at 1e-9 absolute — measured 4.6e-13 on this
# build, with the growth rate (~x70 per 250 rows) leaving the horizon far
# inside the bound. Models (A) and (C) are compared exactly.
CML_N_SITES = 64
CML_TRANSIENT = 50
CML_RECORD = 64
CML_EPS = {0: 0.07, 1: 0.024, 2: 0.2}
CIRCLE_ABS_TOLERANCE = 1e-12
CML_B_ABS_TOLERANCE = 1e-9
ZERO_ONE_C = [0.7, 1.1, 1.9, 2.6, 3.3]
ZERO_ONE_N_CUT = 100

# Justified above: measured 1.1e-15 relative on this input.
FUZZY_REL_TOLERANCE = 1e-12
# ln / powf differ in the last bits between the browser and native math
# libraries; the same 1e-12 relative limit covers them.
LOG_REL_TOLERANCE = 1e-12
# Per-c K carries an FFT and sin/cos; held to a loose absolute bound, with
# the median guard below doing the real work.
ZERO_ONE_ABS_TOLERANCE = 1e-9
ZERO_ONE_MEDIAN_MIN = 0.9

# One node process evaluates all the exports on the spec it reads from
# stdin and prints the flat results as JSON. JSON round-trips f64 exactly,
# so an integer count cannot drift through the encoding.
NODE_DRIVER = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
const dir = process.argv[1];
const spec = JSON.parse(readFileSync(0, "utf8"));
const mod = await import(pathToFileURL(join(dir, "dynachaos_wasm.js")).href);
await mod.default({ module_or_path: readFileSync(join(dir, "dynachaos_wasm_bg.wasm")) });
const traj = new Float64Array(spec.traj);
const x = new Float64Array(spec.x);
const mask = new Uint8Array(spec.mask);
const field = new Float64Array(spec.field);
const e1 = new Float64Array(spec.e1);
process.stdout.write(JSON.stringify({
  correlation_counts: Array.from(mod.correlation_counts(
    traj, spec.dim, new Float64Array(spec.r_values), spec.theiler, true)),
  torus_i_attractor_tile: Array.from(mod.torus_doubling_attractor_tile(
    spec.torus_i_kind, spec.torus_i_a, spec.torus_i_d, spec.map_transient,
    spec.map_plot, new Float64Array(spec.torus_i_state))),
  modulated_circle_rotation_tile: Array.from(mod.modulated_circle_rotation_tile(
    spec.circle_a, spec.circle_c, spec.circle_d_min, spec.circle_d_max,
    spec.circle_n_d, spec.circle_eps, spec.circle_transient, spec.circle_iter,
    spec.circle_state[0], spec.circle_state[1])),
  cml_spacetime_a: Array.from(mod.cml_spacetime_tile(
    0, spec.cml_eps[0], spec.cml_n_sites, spec.cml_transient, spec.cml_record,
    new Float64Array(spec.cml_x0))),
  cml_spacetime_b: Array.from(mod.cml_spacetime_tile(
    1, spec.cml_eps[1], spec.cml_n_sites, spec.cml_transient, spec.cml_record,
    new Float64Array(spec.cml_x0))),
  cml_spacetime_c: Array.from(mod.cml_spacetime_tile(
    2, spec.cml_eps[2], spec.cml_n_sites, spec.cml_transient, spec.cml_record,
    new Float64Array(spec.cml_x0))),
  correlation_counts_euclidean: Array.from(mod.correlation_counts(
    traj, spec.dim, new Float64Array(spec.r_values), spec.theiler_nz, false)),
  apen_counts: Array.from(mod.apen_counts(traj, spec.dim, spec.apen_r)),
  fuzzy_entropy_sum: Array.from(mod.fuzzy_entropy_sum(
    traj, spec.dim, spec.fuzzy_r, spec.fuzzy_n, spec.theiler)),
  ordinal_distribution: Array.from(mod.ordinal_distribution(
    x, spec.ordinal_d, spec.ordinal_tau)),
  diagonal_lines: Array.from(mod.diagonal_lines(mask, spec.side, spec.l_min)),
  vertical_lines: Array.from(mod.vertical_lines(mask, spec.side, spec.v_min)),
  multifractal_moments: Array.from(mod.multifractal_moments(
    field, spec.mf_ny, spec.mf_nx,
    new Float64Array(spec.mf_boxes), new Float64Array(spec.mf_q))),
  ami_histogram: Array.from(mod.ami_histogram(x, spec.ami_tau, spec.ami_bins)),
  select_dimension_cao: Array.from(mod.select_dimension_cao(
    e1, spec.cao_lo, spec.cao_hi, spec.cao_tol,
    spec.cao_span, spec.cao_smooth, spec.cao_min, spec.cao_max)),
  zero_one_k: Array.from(mod.zero_one_k(
    x, new Float64Array(spec.zero_one_c), spec.zero_one_n_cut)),
  delayed_logistic_attractor_tile: Array.from(mod.delayed_logistic_attractor_tile(
    spec.delayed_a, spec.delayed_d_min, spec.delayed_d_max, spec.delayed_n_d,
    spec.map_transient, spec.map_plot, new Float64Array(spec.delayed_state))),
  torus_doubling_attractor_tile: Array.from(mod.torus_doubling_attractor_tile(
    spec.torus_kind, spec.torus_a, spec.torus_d, spec.map_transient, spec.map_plot,
    new Float64Array(spec.torus_state))),
  circle_map_lyapunov_sum: Array.from(mod.circle_map_lyapunov_sum(
    spec.circle_lyap_a, spec.circle_lyap_d, spec.lyap_transient, spec.lyap_iter,
    spec.circle_lyap_theta0)),
  delayed_logistic_lyapunov_tile: Array.from(mod.delayed_logistic_lyapunov_tile(
    spec.delayed_lyap_a, spec.delayed_lyap_d_min, spec.delayed_lyap_d_max,
    spec.delayed_lyap_n_d, spec.lyap_transient, spec.lyap_iter,
    new Float64Array(spec.delayed_lyap_state))),
  torus_i_lyapunov_tile: Array.from(mod.torus_doubling_lyapunov_tile(
    spec.torus_i_kind, spec.torus_i_lyap_a, spec.torus_i_lyap_d, spec.torus_i_lyap_d,
    1, spec.lyap_transient, spec.lyap_iter, new Float64Array(spec.torus_i_lyap_state))),
  torus_doubling_lyapunov_tile: Array.from(mod.torus_doubling_lyapunov_tile(
    spec.torus_kind, spec.torus_lyap_a, spec.torus_lyap_d_min, spec.torus_lyap_d_max,
    spec.torus_lyap_n_d, spec.lyap_transient, spec.lyap_iter,
    new Float64Array(spec.torus_lyap_state))),
  coupled_delayed_lyapunov_tile: Array.from(mod.coupled_delayed_lyapunov_tile(
    spec.coupled_lyap_a, spec.coupled_lyap_db_min, spec.coupled_lyap_db_max,
    spec.coupled_lyap_n_db, spec.coupled_lyap_eps, spec.lyap_transient,
    spec.lyap_iter, new Float64Array(spec.coupled_lyap_state))),
  coupled_logistic_phase_tile: Array.from(mod.coupled_logistic_phase_tile(
    spec.phase_a_min, spec.phase_a_max, spec.phase_n_a,
    spec.phase_d_min, spec.phase_d_max, spec.phase_n_d,
    spec.phase_transient, spec.phase_sample, spec.phase_x0, spec.phase_y0)),
  coupled_logistic_attractor_tile: Array.from(mod.coupled_logistic_attractor_tile(
    spec.attractor_a_min, spec.attractor_a_max, spec.attractor_n_a, spec.attractor_d,
    spec.map_transient, spec.map_plot, spec.attractor_x0, spec.attractor_y0)),
  coupled_logistic_basin_grid: Array.from(mod.coupled_logistic_basin_grid(
    spec.basin_a, spec.basin_d, spec.basin_x_min, spec.basin_x_max, spec.basin_n,
    spec.basin_y_min, spec.basin_y_max, spec.basin_n, spec.basin_transient,
    spec.basin_reference, spec.basin_period, spec.basin_x_ref, spec.basin_y_ref)),
  coupled_delayed_projection_tile: Array.from(mod.coupled_delayed_projection_tile(
    spec.projection_a, spec.projection_db_min, spec.projection_db_max,
    spec.projection_n_db, spec.projection_eps, spec.map_transient, spec.map_plot,
    new Float64Array(spec.projection_state))),
  fractalization_attractor_tile: Array.from(mod.fractalization_attractor_tile(
    spec.fractal_a, spec.fractal_d_min, spec.fractal_d_max, spec.fractal_n_d,
    spec.map_transient, spec.map_plot, new Float64Array(spec.fractal_state))),
}));
"""


def logistic_series(n: int) -> np.ndarray:
    """Return n samples of the logistic map at r = 4."""
    x = np.empty(n)
    value = 0.123456789
    for i in range(n):
        value = 4.0 * value * (1.0 - value)
        x[i] = value
    return x


def wasm_results(spec: dict[str, object]) -> dict[str, list[float]]:
    """Return the flat results from the WebAssembly build."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", NODE_DRIVER, str(SITE_WASM)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def compare(
    name: str,
    native_values: list[float],
    wasm_values: list[float],
    rel_tolerance: float,
) -> tuple[bool, float]:
    """Compare element by element. Return (failed, worst difference)."""
    if len(native_values) != len(wasm_values):
        return True, float("inf")
    worst = 0.0
    for index, (native, wasm) in enumerate(zip(native_values, wasm_values, strict=True)):
        if not (isfinite(native) and isfinite(wasm)):
            print(f"FAIL: {name} value {index} is not finite ({native!r} vs {wasm!r})")
            return True, float("inf")
        difference = abs(native - wasm)
        worst = max(worst, difference)
        scale = max(abs(native), abs(wasm), 1.0)
        if difference > rel_tolerance * scale:
            return True, worst
    return False, worst


def check_header(name: str, header: list[float], expected: list[float]) -> bool:
    """Return True when the wasm header reports the values the script sent."""
    if header != expected:
        print(f"FAIL: {name} header {header} does not match the request {expected}")
        return False
    return True


def self_check(
    name: str,
    native_values: list[float],
    wasm_values: list[float],
    rel_tolerance: float,
    perturbation: float,
) -> bool:
    """Return True when perturbing one native value trips the comparison."""
    perturbed = list(native_values)
    perturbed[0] += perturbation
    failed, _ = compare(name, perturbed, wasm_values, rel_tolerance)
    return failed


def main() -> int:
    """Compare the exports and report the worst difference of each."""
    if not (SITE_WASM / "dynachaos_wasm.js").exists():
        print(
            f"no bundle in {SITE_WASM}. Run: uv run python scripts/build_wasm.py",
            file=sys.stderr,
        )
        return 1

    x = logistic_series(N_SAMPLES)
    traj = embed_time_delay(x, EMBED_DIM, EMBED_TAU)
    n_templates = len(traj)

    # A small recurrence matrix for the line kernels: threshold the embedded
    # trajectory's pairwise distances at a fixed percentile, as rqa does.
    r_traj = embed_time_delay(x[:RQA_SIDE], EMBED_DIM, EMBED_TAU)
    R, _eps = recurrence_matrix(r_traj, percentile=RQA_PERCENTILE)
    side = int(R.shape[0])
    mask_u8 = R.astype(np.uint8).ravel().tolist()

    # A small nonnegative multifractal field from the same series.
    field = (x[: MF_NY * MF_NX] + 0.5).tolist()

    spec: dict[str, object] = {
        "traj": traj.ravel().tolist(),
        "x": x.tolist(),
        "dim": EMBED_DIM,
        "r_values": R_VALUES,
        "theiler": THEILER,
        "theiler_nz": 5,
        "apen_r": APEN_R,
        "fuzzy_r": FUZZY_R,
        "fuzzy_n": FUZZY_N,
        "ordinal_d": ORDINAL_D,
        "ordinal_tau": ORDINAL_TAU,
        "mask": mask_u8,
        "side": side,
        "l_min": L_MIN,
        "v_min": V_MIN,
        "field": field,
        "mf_ny": MF_NY,
        "mf_nx": MF_NX,
        "mf_boxes": MF_BOXES,
        "mf_q": MF_Q,
        "ami_tau": AMI_TAU_MAX,
        "ami_bins": AMI_BINS,
        "e1": CAO_E1,
        "cao_lo": 0.95,
        "cao_hi": 1.05,
        "cao_tol": 0.02,
        "cao_span": 3,
        "cao_smooth": 1,
        "cao_min": 2,
        "cao_max": 0.0,
        "zero_one_c": ZERO_ONE_C,
        "torus_i_kind": TORUS_I_KIND,
        "torus_i_a": TORUS_I_A,
        "torus_i_d": TORUS_I_D,
        "torus_i_state": TORUS_I_STATE,
        "circle_a": CIRCLE_A,
        "circle_c": CIRCLE_C,
        "circle_d_min": CIRCLE_D_MIN,
        "circle_d_max": CIRCLE_D_MAX,
        "circle_n_d": CIRCLE_N_D,
        "circle_eps": CIRCLE_EPS,
        "circle_transient": CIRCLE_TRANSIENT,
        "lyap_transient": LYAP_TRANSIENT,
        "lyap_iter": LYAP_ITER,
        "circle_lyap_a": CIRCLE_LYAP_A,
        "circle_lyap_d": CIRCLE_LYAP_D,
        "circle_lyap_theta0": CIRCLE_LYAP_THETA0,
        "delayed_lyap_a": DELAYED_LYAP_A,
        "delayed_lyap_d_min": DELAYED_LYAP_D_MIN,
        "delayed_lyap_d_max": DELAYED_LYAP_D_MAX,
        "delayed_lyap_n_d": DELAYED_LYAP_N_D,
        "delayed_lyap_state": DELAYED_LYAP_STATE,
        "torus_i_lyap_a": TORUS_I_LYAP_A,
        "torus_i_lyap_d": TORUS_I_LYAP_D,
        "torus_i_lyap_state": TORUS_I_LYAP_STATE,
        "torus_lyap_a": TORUS_LYAP_A,
        "torus_lyap_d_min": TORUS_LYAP_D_MIN,
        "torus_lyap_d_max": TORUS_LYAP_D_MAX,
        "torus_lyap_n_d": TORUS_LYAP_N_D,
        "torus_lyap_state": TORUS_LYAP_STATE,
        "coupled_lyap_a": COUPLED_LYAP_A,
        "coupled_lyap_db_min": COUPLED_LYAP_DB_MIN,
        "coupled_lyap_db_max": COUPLED_LYAP_DB_MAX,
        "coupled_lyap_n_db": COUPLED_LYAP_N_DB,
        "coupled_lyap_eps": COUPLED_LYAP_EPS,
        "coupled_lyap_state": COUPLED_LYAP_STATE,
        "circle_iter": CIRCLE_ITER,
        "circle_state": CIRCLE_STATE,
        "cml_n_sites": CML_N_SITES,
        "cml_transient": CML_TRANSIENT,
        "cml_record": CML_RECORD,
        "cml_eps": CML_EPS,
        "cml_x0": x[:CML_N_SITES].tolist(),
        "zero_one_n_cut": ZERO_ONE_N_CUT,
        "map_transient": MAP_TRANSIENT,
        "map_plot": MAP_PLOT,
        "delayed_a": DELAYED_A,
        "delayed_d_min": DELAYED_D_MIN,
        "delayed_d_max": DELAYED_D_MAX,
        "delayed_n_d": DELAYED_N_D,
        "delayed_state": DELAYED_STATE,
        "torus_kind": TORUS_KIND,
        "torus_a": TORUS_A,
        "torus_d": TORUS_D,
        "torus_state": TORUS_STATE,
        "phase_a_min": PHASE_A_MIN,
        "phase_a_max": PHASE_A_MAX,
        "phase_n_a": PHASE_N_A,
        "phase_d_min": PHASE_D_MIN,
        "phase_d_max": PHASE_D_MAX,
        "phase_n_d": PHASE_N_D,
        "phase_transient": PHASE_TRANSIENT,
        "phase_sample": PHASE_SAMPLE,
        "phase_x0": PHASE_X0,
        "phase_y0": PHASE_Y0,
        "attractor_a_min": ATTRACTOR_A_MIN,
        "attractor_a_max": ATTRACTOR_A_MAX,
        "attractor_n_a": ATTRACTOR_N_A,
        "attractor_d": ATTRACTOR_D,
        "attractor_x0": ATTRACTOR_X0,
        "attractor_y0": ATTRACTOR_Y0,
        "basin_a": BASIN_A,
        "basin_d": BASIN_D,
        "basin_x_min": -1.0,
        "basin_x_max": 1.0,
        "basin_y_min": -1.0,
        "basin_y_max": 1.0,
        "basin_n": BASIN_N,
        "basin_transient": BASIN_TRANSIENT,
        "basin_reference": BASIN_REFERENCE,
        "basin_period": BASIN_PERIOD,
        "basin_x_ref": BASIN_X_REF,
        "basin_y_ref": BASIN_Y_REF,
        "projection_a": PROJECTION_A,
        "projection_db_min": PROJECTION_DB_MIN,
        "projection_db_max": PROJECTION_DB_MAX,
        "projection_n_db": PROJECTION_N_DB,
        "projection_eps": PROJECTION_EPS,
        "projection_state": PROJECTION_STATE,
        "fractal_a": FRACTAL_A,
        "fractal_d_min": FRACTAL_D_MIN,
        "fractal_d_max": FRACTAL_D_MAX,
        "fractal_n_d": FRACTAL_N_D,
        "fractal_state": FRACTAL_STATE,
    }
    wasm = wasm_results(spec)

    failed = False

    # correlation_counts: [n_pts, dim, n_r, theiler, chebyshev, radii..., counts...]
    out = wasm["correlation_counts"]
    n_r = len(R_VALUES)
    if not check_header(
        "correlation_counts",
        out[:5],
        [float(n_templates), float(EMBED_DIM), float(n_r), float(THEILER), 1.0],
    ):
        failed = True
    elif out[5 : 5 + n_r] != sorted(R_VALUES):
        print(f"FAIL: correlation_counts radii {out[5 : 5 + n_r]} != {sorted(R_VALUES)}")
        failed = True
    else:
        native = [float(v) for v in correlation_counts(traj, np.array(R_VALUES), THEILER, True)]
        wasm_counts = out[5 + n_r :]
        bad, worst = compare("correlation_counts", native, wasm_counts, 0.0)
        print(f"correlation_counts: {n_r} radii over {n_templates} rows, worst {worst:.3e}")
        if bad:
            print("FAIL: correlation_counts counts differ")
            failed = True
        elif not self_check("correlation_counts", native, wasm_counts, 0.0, 1.0):
            print("FAIL: correlation_counts self-check did not trip")
            failed = True

    # correlation_counts, Euclidean norm and a nonzero Theiler window.
    out = wasm["correlation_counts_euclidean"]
    theiler_nz = 5
    if not check_header(
        "correlation_counts_euclidean",
        out[:5],
        [float(n_templates), float(EMBED_DIM), float(n_r), float(theiler_nz), 0.0],
    ):
        failed = True
    else:
        native = [float(v) for v in correlation_counts(traj, np.array(R_VALUES), theiler_nz, False)]
        wasm_counts = out[5 + n_r :]
        bad, worst = compare("correlation_counts_euclidean", native, wasm_counts, 0.0)
        print(
            f"correlation_counts_euclidean: {n_r} radii over {n_templates} rows "
            f"(theiler {theiler_nz}), worst {worst:.3e}"
        )
        if bad:
            print("FAIL: correlation_counts_euclidean counts differ")
            failed = True
        elif not self_check("correlation_counts_euclidean", native, wasm_counts, 0.0, 1.0):
            print("FAIL: correlation_counts_euclidean self-check did not trip")
            failed = True

    # apen_counts: [n_pts, dim, r, counts...]
    out = wasm["apen_counts"]
    if not check_header("apen_counts", out[:3], [float(n_templates), float(EMBED_DIM), APEN_R]):
        failed = True
    else:
        native = [float(v) for v in apen_counts(traj, APEN_R)]
        wasm_counts = out[3:]
        bad, worst = compare("apen_counts", native, wasm_counts, 0.0)
        print(f"apen_counts: {n_templates} rows, worst {worst:.3e}")
        if bad:
            print("FAIL: apen_counts counts differ")
            failed = True
        elif not self_check("apen_counts", native, wasm_counts, 0.0, 1.0):
            print("FAIL: apen_counts self-check did not trip")
            failed = True

    # fuzzy_entropy_sum: [n_pts, dim, r, n, theiler, sum]
    out = wasm["fuzzy_entropy_sum"]
    if not check_header(
        "fuzzy_entropy_sum",
        out[:5],
        [float(n_templates), float(EMBED_DIM), FUZZY_R, float(FUZZY_N), float(THEILER)],
    ):
        failed = True
    else:
        native = [float(fuzzy_entropy_sum(traj, FUZZY_R, FUZZY_N, THEILER))]
        wasm_sum = out[5:]
        bad, worst = compare("fuzzy_entropy_sum", native, wasm_sum, FUZZY_REL_TOLERANCE)
        rel = worst / max(abs(native[0]), 1.0)
        print(
            f"fuzzy_entropy_sum: worst {worst:.3e} (rel {rel:.3e}, limit {FUZZY_REL_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: fuzzy_entropy_sum differs beyond the stated tolerance")
            failed = True
        elif not self_check(
            "fuzzy_entropy_sum", native, wasm_sum, FUZZY_REL_TOLERANCE, abs(native[0]) * 1e-6
        ):
            print("FAIL: fuzzy_entropy_sum self-check did not trip")
            failed = True

    # ordinal_distribution: [n, d, tau, n_windows, counts...]
    out = wasm["ordinal_distribution"]
    native_counts, native_windows = ordinal_distribution(x, ORDINAL_D, ORDINAL_TAU)
    if not check_header(
        "ordinal_distribution",
        out[:4],
        [float(N_SAMPLES), float(ORDINAL_D), float(ORDINAL_TAU), float(native_windows)],
    ):
        failed = True
    else:
        native = [float(v) for v in native_counts]
        wasm_counts = out[4:]
        bad, worst = compare("ordinal_distribution", native, wasm_counts, 0.0)
        print(f"ordinal_distribution: {len(native)} patterns, worst {worst:.3e}")
        if bad:
            print("FAIL: ordinal_distribution counts differ")
            failed = True
        elif not self_check("ordinal_distribution", native, wasm_counts, 0.0, 1.0):
            print("FAIL: ordinal_distribution self-check did not trip")
            failed = True

    # diagonal_lines: [side, l_min, n_lines, lengths...]
    out = wasm["diagonal_lines"]
    native = [float(v) for v in diagonal_lines(R, L_MIN)]
    if not check_header("diagonal_lines", out[:3], [float(side), float(L_MIN), float(len(native))]):
        failed = True
    else:
        wasm_lengths = out[3:]
        bad, worst = compare("diagonal_lines", native, wasm_lengths, 0.0)
        print(f"diagonal_lines: {len(native)} lines over side {side}, worst {worst:.3e}")
        if bad:
            print("FAIL: diagonal_lines lengths differ")
            failed = True
        elif not self_check("diagonal_lines", native, wasm_lengths, 0.0, 1.0):
            print("FAIL: diagonal_lines self-check did not trip")
            failed = True

    # vertical_lines: [side, v_min, n_lines, lengths...]
    out = wasm["vertical_lines"]
    native = [float(v) for v in vertical_lines(R, V_MIN)]
    if not check_header("vertical_lines", out[:3], [float(side), float(V_MIN), float(len(native))]):
        failed = True
    else:
        wasm_lengths = out[3:]
        bad, worst = compare("vertical_lines", native, wasm_lengths, 0.0)
        print(f"vertical_lines: {len(native)} lines over side {side}, worst {worst:.3e}")
        if bad:
            print("FAIL: vertical_lines lengths differ")
            failed = True
        elif not self_check("vertical_lines", native, wasm_lengths, 0.0, 1.0):
            print("FAIL: vertical_lines self-check did not trip")
            failed = True

    # multifractal_moments: [ny, nx, n_scales, n_q, log_z, alpha_num, f_num, ln_scales]
    out = wasm["multifractal_moments"]
    n_scales = len(MF_BOXES)
    n_q = len(MF_Q)
    if not check_header(
        "multifractal_moments",
        out[:4],
        [float(MF_NY), float(MF_NX), float(n_scales), float(n_q)],
    ):
        failed = True
    else:
        lz, an, fn, ln = multifractal_moments(
            np.array(field).reshape(MF_NY, MF_NX),
            np.array(MF_BOXES, dtype=np.int64),
            np.array(MF_Q),
        )
        native = (
            [float(v) for v in np.asarray(lz).ravel()]
            + [float(v) for v in np.asarray(an).ravel()]
            + [float(v) for v in np.asarray(fn).ravel()]
            + [float(v) for v in np.asarray(ln).ravel()]
        )
        wasm_values = out[4:]
        bad, worst = compare("multifractal_moments", native, wasm_values, LOG_REL_TOLERANCE)
        rel = worst / max(max(abs(v) for v in native), 1.0)
        print(
            f"multifractal_moments: {n_scales} scales x {n_q} q, worst {worst:.3e} "
            f"(rel {rel:.3e}, limit {LOG_REL_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: multifractal_moments differs beyond the stated tolerance")
            failed = True
        elif not self_check("multifractal_moments", native, wasm_values, LOG_REL_TOLERANCE, 1e-6):
            print("FAIL: multifractal_moments self-check did not trip")
            failed = True

    # ami_histogram: [n, tau_max, n_bins, mi...]
    out = wasm["ami_histogram"]
    native = [float(v) for v in ami_histogram(x, AMI_TAU_MAX, AMI_BINS)]
    if not check_header(
        "ami_histogram",
        out[:3],
        [float(N_SAMPLES), float(AMI_TAU_MAX), float(AMI_BINS)],
    ):
        failed = True
    else:
        wasm_mi = out[3:]
        bad, worst = compare("ami_histogram", native, wasm_mi, LOG_REL_TOLERANCE)
        rel = worst / max(max(abs(v) for v in native), 1.0)
        print(
            f"ami_histogram: {AMI_TAU_MAX} lags x {AMI_BINS} bins, worst {worst:.3e} "
            f"(rel {rel:.3e}, limit {LOG_REL_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: ami_histogram differs beyond the stated tolerance")
            failed = True
        elif not self_check("ami_histogram", native, wasm_mi, LOG_REL_TOLERANCE, 1e-6):
            print("FAIL: ami_histogram self-check did not trip")
            failed = True

    # select_dimension_cao: [n, min_dim, max_dim, dim]
    out = wasm["select_dimension_cao"]
    native_dim = float(select_dimension_cao(np.array(CAO_E1), 0.95, 1.05, 0.02, 3, 1, 2, None))
    if not check_header(
        "select_dimension_cao", out[:4], [float(len(CAO_E1)), 2.0, 0.0, native_dim]
    ):
        failed = True
    else:
        print(f"select_dimension_cao: dim {int(out[3])} over {len(CAO_E1)} E1 values")
        if not self_check("select_dimension_cao", [native_dim], [out[3]], 0.0, 1.0):
            print("FAIL: select_dimension_cao self-check did not trip")
            failed = True

    # zero_one_k: [n, n_c, n_cut, k...] — per-c K against pyo3, plus a median
    # guard so an all-zero wasm kernel fails even though zeros compare equal.
    out = wasm["zero_one_k"]
    native = [float(v) for v in zero_one_k(x, np.array(ZERO_ONE_C), ZERO_ONE_N_CUT)]
    if not check_header(
        "zero_one_k",
        out[:3],
        [float(N_SAMPLES), float(len(ZERO_ONE_C)), float(ZERO_ONE_N_CUT)],
    ):
        failed = True
    else:
        wasm_k = out[3:]
        bad, worst = compare("zero_one_k", native, wasm_k, ZERO_ONE_ABS_TOLERANCE)
        median_k = float(np.median(wasm_k))
        print(
            f"zero_one_k: {len(ZERO_ONE_C)} c values, worst {worst:.3e}, "
            f"median K {median_k:.3f} (min {ZERO_ONE_MEDIAN_MIN})"
        )
        if bad:
            print("FAIL: zero_one_k differs beyond the stated tolerance")
            failed = True
        elif median_k <= ZERO_ONE_MEDIAN_MIN:
            print(f"FAIL: zero_one_k median {median_k:.3f} <= {ZERO_ONE_MEDIAN_MIN}")
            failed = True
        elif not self_check("zero_one_k", native, wasm_k, ZERO_ONE_ABS_TOLERANCE, 0.5):
            print("FAIL: zero_one_k self-check did not trip")
            failed = True

    # delayed_logistic_attractor_tile: [n_D, n_plot, n_transient, dim, samples...].
    delayed_d_values = [
        DELAYED_D_MIN + (DELAYED_D_MAX - DELAYED_D_MIN) * k / (DELAYED_N_D - 1)
        for k in range(DELAYED_N_D)
    ]
    delayed_native = delayed_logistic_attractor_tile(
        DELAYED_A,
        np.array(delayed_d_values, dtype=np.float64),
        MAP_TRANSIENT,
        MAP_PLOT,
        np.array(DELAYED_STATE, dtype=np.float64),
    )
    out = wasm["delayed_logistic_attractor_tile"]
    delayed_expected_header = [
        float(DELAYED_N_D),
        float(MAP_PLOT),
        float(MAP_TRANSIENT),
        2.0,
    ]
    if not check_header("delayed_logistic_attractor_tile", out[:4], delayed_expected_header):
        failed = True
    else:
        bad, worst = compare(
            "delayed_logistic_attractor_tile",
            [float(v) for v in np.asarray(delayed_native).ravel()],
            out[4:],
            0.0,
        )
        print(
            f"delayed_logistic_attractor_tile: {DELAYED_N_D} D values x "
            f"{MAP_PLOT} samples, worst {worst:.3e}"
        )
        if bad:
            print("FAIL: delayed_logistic_attractor_tile differs")
            failed = True
        elif not self_check(
            "delayed_logistic_attractor_tile",
            [float(v) for v in np.asarray(delayed_native).ravel()],
            out[4:],
            0.0,
            1.0,
        ):
            print("FAIL: delayed_logistic_attractor_tile self-check did not trip")
            failed = True

    # torus_doubling_attractor_tile: [kind, dim, n_transient, n_produced, samples...].
    torus_native = torus_doubling_attractor_tile(
        TORUS_KIND,
        TORUS_A,
        TORUS_D,
        MAP_TRANSIENT,
        MAP_PLOT,
        np.array(TORUS_STATE, dtype=np.float64),
    )
    out = wasm["torus_doubling_attractor_tile"]
    torus_expected_header = [float(TORUS_KIND), 4.0, float(MAP_TRANSIENT), float(MAP_PLOT)]
    if not check_header("torus_doubling_attractor_tile", out[:4], torus_expected_header):
        failed = True
    else:
        bad, worst = compare(
            "torus_doubling_attractor_tile",
            [float(v) for v in np.asarray(torus_native).ravel()],
            out[4:],
            0.0,
        )
        print(
            f"torus_doubling_attractor_tile: map {TORUS_KIND}, "
            f"{MAP_PLOT} samples, worst {worst:.3e}"
        )
        if bad:
            print("FAIL: torus_doubling_attractor_tile differs")
            failed = True
        elif not self_check(
            "torus_doubling_attractor_tile",
            [float(v) for v in np.asarray(torus_native).ravel()],
            out[4:],
            0.0,
            1.0,
        ):
            print("FAIL: torus_doubling_attractor_tile self-check did not trip")
            failed = True

    # torus_doubling_attractor_tile, map I: same layout as map IV above.
    torus_i_native = torus_doubling_attractor_tile(
        TORUS_I_KIND,
        TORUS_I_A,
        TORUS_I_D,
        MAP_TRANSIENT,
        MAP_PLOT,
        np.array(TORUS_I_STATE, dtype=np.float64),
    )
    out = wasm["torus_i_attractor_tile"]
    torus_i_expected_header = [
        float(TORUS_I_KIND),
        3.0,
        float(MAP_TRANSIENT),
        float(MAP_PLOT),
    ]
    if not check_header("torus_i_attractor_tile", out[:4], torus_i_expected_header):
        failed = True
    else:
        bad, worst = compare(
            "torus_i_attractor_tile",
            [float(v) for v in np.asarray(torus_i_native).ravel()],
            out[4:],
            0.0,
        )
        print(f"torus_i_attractor_tile: map {TORUS_I_KIND}, {MAP_PLOT} samples, worst {worst:.3e}")
        if bad:
            print("FAIL: torus_i_attractor_tile differs")
            failed = True
        elif not self_check(
            "torus_i_attractor_tile",
            [float(v) for v in np.asarray(torus_i_native).ravel()],
            out[4:],
            0.0,
            1.0,
        ):
            print("FAIL: torus_i_attractor_tile self-check did not trip")
            failed = True

    # modulated_circle_rotation_tile: [n_D, n_transient, n_iter, 2, pairs...].
    circle_d_values = np.linspace(CIRCLE_D_MIN, CIRCLE_D_MAX, CIRCLE_N_D)
    circle_native = modulated_circle_rotation_tile(
        CIRCLE_A,
        CIRCLE_C,
        circle_d_values,
        CIRCLE_EPS,
        CIRCLE_TRANSIENT,
        CIRCLE_ITER,
        np.array(CIRCLE_STATE, dtype=np.float64),
    )
    out = wasm["modulated_circle_rotation_tile"]
    circle_expected_header = [
        float(CIRCLE_N_D),
        float(CIRCLE_TRANSIENT),
        float(CIRCLE_ITER),
        2.0,
    ]
    if not check_header("modulated_circle_rotation_tile", out[:4], circle_expected_header):
        failed = True
    else:
        circle_native_flat = [float(v) for v in np.asarray(circle_native).ravel()]
        bad, worst = compare(
            "modulated_circle_rotation_tile",
            circle_native_flat,
            out[4:],
            CIRCLE_ABS_TOLERANCE,
        )
        print(
            f"modulated_circle_rotation_tile: {CIRCLE_N_D} D values x "
            f"{CIRCLE_ITER} steps, worst {worst:.3e} "
            f"(limit {CIRCLE_ABS_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: modulated_circle_rotation_tile differs")
            failed = True
        elif not self_check(
            "modulated_circle_rotation_tile",
            circle_native_flat,
            out[4:],
            CIRCLE_ABS_TOLERANCE,
            1e-6,
        ):
            print("FAIL: modulated_circle_rotation_tile self-check did not trip")
            failed = True

    # cml_spacetime_tile: [model, n_sites, n_transient, n_record, field...].
    cml_x0 = np.array(x[:CML_N_SITES], dtype=np.float64)
    for model, key in [(0, "cml_spacetime_a"), (1, "cml_spacetime_b"), (2, "cml_spacetime_c")]:
        cml_native = cml_spacetime_tile(model, CML_EPS[model], CML_TRANSIENT, CML_RECORD, cml_x0)
        out = wasm[key]
        cml_expected_header = [
            float(model),
            float(CML_N_SITES),
            float(CML_TRANSIENT),
            float(CML_RECORD),
        ]
        tolerance = CML_B_ABS_TOLERANCE if model == 1 else 0.0
        if not check_header(key, out[:4], cml_expected_header):
            failed = True
        else:
            cml_native_flat = [float(v) for v in np.asarray(cml_native).ravel()]
            bad, worst = compare(key, cml_native_flat, out[4:], tolerance)
            print(
                f"{key}: {CML_RECORD} rows x {CML_N_SITES} sites, "
                f"worst {worst:.3e} (limit {tolerance:.1e})"
            )
            if bad:
                print(f"FAIL: {key} differs")
                failed = True
            elif not self_check(key, cml_native_flat, out[4:], tolerance, 1e-6):
                print(f"FAIL: {key} self-check did not trip")
                failed = True

    # circle_map_lyapunov_sum: [n_transient, n_iter, lambda].
    circle_lyap_native = circle_map_lyapunov_sum(
        CIRCLE_LYAP_A, CIRCLE_LYAP_D, LYAP_TRANSIENT, LYAP_ITER, CIRCLE_LYAP_THETA0
    )
    out = wasm["circle_map_lyapunov_sum"]
    circle_lyap_header = [float(LYAP_TRANSIENT), float(LYAP_ITER)]
    if not check_header("circle_map_lyapunov_sum", out[:2], circle_lyap_header):
        failed = True
    else:
        bad, worst = compare(
            "circle_map_lyapunov_sum", [circle_lyap_native], [out[2]], LYAP_ABS_TOLERANCE
        )
        print(
            f"circle_map_lyapunov_sum: {LYAP_ITER} steps, "
            f"worst {worst:.3e} (limit {LYAP_ABS_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: circle_map_lyapunov_sum differs")
            failed = True
        elif not self_check(
            "circle_map_lyapunov_sum", [circle_lyap_native], [out[2]], LYAP_ABS_TOLERANCE, 1e-6
        ):
            print("FAIL: circle_map_lyapunov_sum self-check did not trip")
            failed = True

    # delayed_logistic_lyapunov_tile: [n_D, n_transient, n_iter, 2, spectra...].
    delayed_lyap_d = np.linspace(DELAYED_LYAP_D_MIN, DELAYED_LYAP_D_MAX, DELAYED_LYAP_N_D)
    delayed_lyap_native = delayed_logistic_lyapunov_tile(
        DELAYED_LYAP_A,
        delayed_lyap_d,
        LYAP_TRANSIENT,
        LYAP_ITER,
        np.array(DELAYED_LYAP_STATE, dtype=np.float64),
    )
    out = wasm["delayed_logistic_lyapunov_tile"]
    delayed_lyap_header = [
        float(DELAYED_LYAP_N_D),
        float(LYAP_TRANSIENT),
        float(LYAP_ITER),
        2.0,
    ]
    if not check_header("delayed_logistic_lyapunov_tile", out[:4], delayed_lyap_header):
        failed = True
    else:
        delayed_lyap_flat = [float(v) for v in np.asarray(delayed_lyap_native).ravel()]
        bad, worst = compare(
            "delayed_logistic_lyapunov_tile", delayed_lyap_flat, out[4:], LYAP_ABS_TOLERANCE
        )
        print(
            f"delayed_logistic_lyapunov_tile: {DELAYED_LYAP_N_D} D values, "
            f"worst {worst:.3e} (limit {LYAP_ABS_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: delayed_logistic_lyapunov_tile differs")
            failed = True
        elif not self_check(
            "delayed_logistic_lyapunov_tile",
            delayed_lyap_flat,
            out[4:],
            LYAP_ABS_TOLERANCE,
            1e-6,
        ):
            print("FAIL: delayed_logistic_lyapunov_tile self-check did not trip")
            failed = True

    # torus_doubling_lyapunov_tile, map I: [n_D, n_transient, n_iter, 3, spectra...].
    torus_i_lyap_native = torus_doubling_lyapunov_tile(
        TORUS_I_KIND,
        TORUS_I_LYAP_A,
        np.array([TORUS_I_LYAP_D], dtype=np.float64),
        LYAP_TRANSIENT,
        LYAP_ITER,
        np.array(TORUS_I_LYAP_STATE, dtype=np.float64),
    )
    out = wasm["torus_i_lyapunov_tile"]
    torus_i_lyap_header = [1.0, float(LYAP_TRANSIENT), float(LYAP_ITER), 3.0]
    if not check_header("torus_i_lyapunov_tile", out[:4], torus_i_lyap_header):
        failed = True
    else:
        torus_i_lyap_flat = [float(v) for v in np.asarray(torus_i_lyap_native).ravel()]
        bad, worst = compare(
            "torus_i_lyapunov_tile", torus_i_lyap_flat, out[4:], LYAP_ABS_TOLERANCE
        )
        print(f"torus_i_lyapunov_tile: map I, worst {worst:.3e} (limit {LYAP_ABS_TOLERANCE:.1e})")
        if bad:
            print("FAIL: torus_i_lyapunov_tile differs")
            failed = True
        elif not self_check(
            "torus_i_lyapunov_tile", torus_i_lyap_flat, out[4:], LYAP_ABS_TOLERANCE, 1e-6
        ):
            print("FAIL: torus_i_lyapunov_tile self-check did not trip")
            failed = True

    # torus_doubling_lyapunov_tile, map IV: [n_D, n_transient, n_iter, 4, spectra...].
    torus_lyap_d = np.linspace(TORUS_LYAP_D_MIN, TORUS_LYAP_D_MAX, TORUS_LYAP_N_D)
    torus_lyap_native = torus_doubling_lyapunov_tile(
        TORUS_KIND,
        TORUS_LYAP_A,
        torus_lyap_d,
        LYAP_TRANSIENT,
        LYAP_ITER,
        np.array(TORUS_LYAP_STATE, dtype=np.float64),
    )
    out = wasm["torus_doubling_lyapunov_tile"]
    torus_lyap_header = [
        float(TORUS_LYAP_N_D),
        float(LYAP_TRANSIENT),
        float(LYAP_ITER),
        4.0,
    ]
    if not check_header("torus_doubling_lyapunov_tile", out[:4], torus_lyap_header):
        failed = True
    else:
        torus_lyap_flat = [float(v) for v in np.asarray(torus_lyap_native).ravel()]
        bad, worst = compare(
            "torus_doubling_lyapunov_tile", torus_lyap_flat, out[4:], LYAP_ABS_TOLERANCE
        )
        print(
            f"torus_doubling_lyapunov_tile: {TORUS_LYAP_N_D} D values, "
            f"worst {worst:.3e} (limit {LYAP_ABS_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: torus_doubling_lyapunov_tile differs")
            failed = True
        elif not self_check(
            "torus_doubling_lyapunov_tile",
            torus_lyap_flat,
            out[4:],
            LYAP_ABS_TOLERANCE,
            1e-6,
        ):
            print("FAIL: torus_doubling_lyapunov_tile self-check did not trip")
            failed = True

    # coupled_delayed_lyapunov_tile: [n_DB, n_transient, n_iter, 4, spectra...].
    coupled_lyap_db = np.linspace(COUPLED_LYAP_DB_MIN, COUPLED_LYAP_DB_MAX, COUPLED_LYAP_N_DB)
    coupled_lyap_native = coupled_delayed_lyapunov_tile(
        COUPLED_LYAP_A,
        coupled_lyap_db,
        COUPLED_LYAP_EPS,
        LYAP_TRANSIENT,
        LYAP_ITER,
        np.array(COUPLED_LYAP_STATE, dtype=np.float64),
    )
    out = wasm["coupled_delayed_lyapunov_tile"]
    coupled_lyap_header = [
        float(COUPLED_LYAP_N_DB),
        float(LYAP_TRANSIENT),
        float(LYAP_ITER),
        4.0,
    ]
    if not check_header("coupled_delayed_lyapunov_tile", out[:4], coupled_lyap_header):
        failed = True
    else:
        coupled_lyap_flat = [float(v) for v in np.asarray(coupled_lyap_native).ravel()]
        bad, worst = compare(
            "coupled_delayed_lyapunov_tile", coupled_lyap_flat, out[4:], LYAP_ABS_TOLERANCE
        )
        print(
            f"coupled_delayed_lyapunov_tile: {COUPLED_LYAP_N_DB} DB values, "
            f"worst {worst:.3e} (limit {LYAP_ABS_TOLERANCE:.1e})"
        )
        if bad:
            print("FAIL: coupled_delayed_lyapunov_tile differs")
            failed = True
        elif not self_check(
            "coupled_delayed_lyapunov_tile",
            coupled_lyap_flat,
            out[4:],
            LYAP_ABS_TOLERANCE,
            1e-6,
        ):
            print("FAIL: coupled_delayed_lyapunov_tile self-check did not trip")
            failed = True

    def _swept(lo: float, hi: float, n: int) -> np.ndarray:
        denom = max(n - 1, 1)
        return np.array([lo + (hi - lo) * k / denom for k in range(n)], dtype=np.float64)

    # coupled_logistic_phase_tile: [n_A, n_D, n_transient, n_sample, pairs...].
    phase_native = coupled_logistic_phase_tile(
        _swept(PHASE_A_MIN, PHASE_A_MAX, PHASE_N_A),
        _swept(PHASE_D_MIN, PHASE_D_MAX, PHASE_N_D),
        PHASE_TRANSIENT,
        PHASE_SAMPLE,
        PHASE_X0,
        PHASE_Y0,
    )
    out = wasm["coupled_logistic_phase_tile"]
    phase_header = [
        float(PHASE_N_A),
        float(PHASE_N_D),
        float(PHASE_TRANSIENT),
        float(PHASE_SAMPLE),
    ]
    if not check_header("coupled_logistic_phase_tile", out[:4], phase_header):
        failed = True
    else:
        phase_flat = [float(v) for v in np.asarray(phase_native).ravel()]
        phase_body = out[4:]
        bad_asym, worst_asym = compare(
            "coupled_logistic_phase_tile asym", phase_flat[0::2], phase_body[0::2], 0.0
        )
        bad_lyap, worst_lyap = compare(
            "coupled_logistic_phase_tile lyap",
            phase_flat[1::2],
            phase_body[1::2],
            PHASE_LYAP_ABS,
        )
        print(
            f"coupled_logistic_phase_tile: {PHASE_N_D} x {PHASE_N_A}, "
            f"asym worst {worst_asym:.3e}, lyap worst {worst_lyap:.3e}"
        )
        if bad_asym or bad_lyap:
            print("FAIL: coupled_logistic_phase_tile differs")
            failed = True
        elif not self_check("coupled_logistic_phase_tile", phase_flat, phase_body, 0.0, 1.0):
            print("FAIL: coupled_logistic_phase_tile self-check did not trip")
            failed = True

    # coupled_logistic_attractor_tile: [n_A, n_plot, n_transient, 2, samples...].
    attractor_native = coupled_logistic_attractor_tile(
        _swept(ATTRACTOR_A_MIN, ATTRACTOR_A_MAX, ATTRACTOR_N_A),
        ATTRACTOR_D,
        MAP_TRANSIENT,
        MAP_PLOT,
        np.array([ATTRACTOR_X0, ATTRACTOR_Y0], dtype=np.float64),
    )
    out = wasm["coupled_logistic_attractor_tile"]
    attractor_header = [
        float(ATTRACTOR_N_A),
        float(MAP_PLOT),
        float(MAP_TRANSIENT),
        2.0,
    ]
    if not check_header("coupled_logistic_attractor_tile", out[:4], attractor_header):
        failed = True
    else:
        attractor_flat = [float(v) for v in np.asarray(attractor_native).ravel()]
        bad, worst = compare("coupled_logistic_attractor_tile", attractor_flat, out[4:], 0.0)
        print(
            f"coupled_logistic_attractor_tile: {ATTRACTOR_N_A} A values x "
            f"{MAP_PLOT} samples, worst {worst:.3e}"
        )
        if bad:
            print("FAIL: coupled_logistic_attractor_tile differs")
            failed = True
        elif not self_check("coupled_logistic_attractor_tile", attractor_flat, out[4:], 0.0, 1.0):
            print("FAIL: coupled_logistic_attractor_tile self-check did not trip")
            failed = True

    # coupled_logistic_basin_grid: [n_x, n_y, n_transient, period, labels...].
    from dynachaos.maps.coupled_logistic import _find_reference_orbit

    basin_x = _swept(-1.0, 1.0, BASIN_N)
    basin_y = _swept(-1.0, 1.0, BASIN_N)
    basin_ref = _find_reference_orbit(
        BASIN_A,
        BASIN_D,
        BASIN_X_REF,
        BASIN_Y_REF,
        n_transient=BASIN_REFERENCE,
        period=BASIN_PERIOD,
    )
    assert basin_ref is not None
    basin_native = coupled_logistic_basin_grid(
        basin_x, basin_y, BASIN_A, BASIN_D, BASIN_TRANSIENT, basin_ref
    )
    out = wasm["coupled_logistic_basin_grid"]
    basin_header = [
        float(BASIN_N),
        float(BASIN_N),
        float(BASIN_TRANSIENT),
        float(BASIN_PERIOD),
    ]
    if not check_header("coupled_logistic_basin_grid", out[:4], basin_header):
        failed = True
    else:
        basin_flat = [float(v) for v in np.asarray(basin_native).ravel()]
        bad, worst = compare("coupled_logistic_basin_grid", basin_flat, out[4:], 0.0)
        print(f"coupled_logistic_basin_grid: {BASIN_N} x {BASIN_N}, worst {worst:.3e}")
        if bad:
            print("FAIL: coupled_logistic_basin_grid differs")
            failed = True
        elif not self_check("coupled_logistic_basin_grid", basin_flat, out[4:], 0.0, 1.0):
            print("FAIL: coupled_logistic_basin_grid self-check did not trip")
            failed = True

    # coupled_delayed_projection_tile: [n_DB, n_plot, n_transient, 2, pairs...].
    projection_native = coupled_delayed_projection_tile(
        PROJECTION_A,
        _swept(PROJECTION_DB_MIN, PROJECTION_DB_MAX, PROJECTION_N_DB),
        PROJECTION_EPS,
        MAP_TRANSIENT,
        MAP_PLOT,
        np.array(PROJECTION_STATE, dtype=np.float64),
    )
    out = wasm["coupled_delayed_projection_tile"]
    projection_header = [
        float(PROJECTION_N_DB),
        float(MAP_PLOT),
        float(MAP_TRANSIENT),
        2.0,
    ]
    if not check_header("coupled_delayed_projection_tile", out[:4], projection_header):
        failed = True
    else:
        projection_flat = [float(v) for v in np.asarray(projection_native).ravel()]
        bad, worst = compare("coupled_delayed_projection_tile", projection_flat, out[4:], 0.0)
        print(
            f"coupled_delayed_projection_tile: {PROJECTION_N_DB} DB values x "
            f"{MAP_PLOT} samples, worst {worst:.3e}"
        )
        if bad:
            print("FAIL: coupled_delayed_projection_tile differs")
            failed = True
        elif not self_check("coupled_delayed_projection_tile", projection_flat, out[4:], 0.0, 1.0):
            print("FAIL: coupled_delayed_projection_tile self-check did not trip")
            failed = True

    # fractalization_attractor_tile calls the delayed-logistic kernel.
    fractal_native = delayed_logistic_attractor_tile(
        FRACTAL_A,
        _swept(FRACTAL_D_MIN, FRACTAL_D_MAX, FRACTAL_N_D),
        MAP_TRANSIENT,
        MAP_PLOT,
        np.array(FRACTAL_STATE, dtype=np.float64),
    )
    out = wasm["fractalization_attractor_tile"]
    fractal_header = [
        float(FRACTAL_N_D),
        float(MAP_PLOT),
        float(MAP_TRANSIENT),
        2.0,
    ]
    if not check_header("fractalization_attractor_tile", out[:4], fractal_header):
        failed = True
    else:
        fractal_flat = [float(v) for v in np.asarray(fractal_native).ravel()]
        bad, worst = compare("fractalization_attractor_tile", fractal_flat, out[4:], 0.0)
        print(
            f"fractalization_attractor_tile: {FRACTAL_N_D} D values x "
            f"{MAP_PLOT} samples, worst {worst:.3e}"
        )
        if bad:
            print("FAIL: fractalization_attractor_tile differs")
            failed = True
        elif not self_check("fractalization_attractor_tile", fractal_flat, out[4:], 0.0, 1.0):
            print("FAIL: fractalization_attractor_tile self-check did not trip")
            failed = True

    if failed:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
