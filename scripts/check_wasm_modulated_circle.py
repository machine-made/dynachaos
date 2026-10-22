"""Check the live double devil's staircase against the published npz.

The paper's double staircase (``figures/sec06_three_torus/double_staircase.npz``)
is (rho_theta, rho_phi) at A = 0.1, C = (sqrt(5) - 1) / 2, eps = 0.05,
state0 = (0.1, 0.1), n_transient 3000, n_iter 20000 over D on
``np.linspace(0, 1, 10000)`` (``maps/modulated_circle.py`` ``compute``). The
live figure computes the same pair with the wasm
``modulated_circle_rotation_tile`` over a reader-chosen D window.

The wasm D grid is ``d_min + (d_max - d_min) k / (n - 1)``, not numpy's
linspace, so no multi-point tile lands exactly on the npz grid. This script
therefore evaluates one tile per D with ``d_min = d_max = D[i]`` — k = 0
gives D[i] exactly — at the committed grid entries D[0], D[2500], D[5000],
D[7500], D[9999] and at every npz D inside both zoom windows, the same
windows ``plot_zoom`` draws: ``longest_plateau_window(D, rho_theta, target,
5e-4)`` padded by 0.008 for target 1/4 and by 0.006 for target C. Exact npz
D values only, never rounded.

The rule is |rho_theta difference| <= 1e-6 and |rho_phi - C| <= 1e-12, fixed
before the run. The reason is the map itself (see
``docs/plans/plan_live-figure_modulated-circle.md`` § Numerics): A = 0.1 <
1/(2 pi), so every fibre map has derivative 1 + 2 pi A cos(2 pi theta) in
[0.37, 1.63] > 0 and is an orientation-preserving diffeomorphism — no
chaotic amplification. The only difference between the wasm sine and the
platform one is about 1e-16 per step. Ceiling: the smallest zoom panel spans
at least 0.02 in rho_theta, so one display step is 0.02/256 = 7.8e-5; 1e-6
is 78 times finer. rho_phi is the mean of a constant increment C, so it must
equal C to 1e-12. If a point fails, this script reports the worst point and
fails; the fix is in the kernel, never a tolerance here.

Run it with::

    uv run python scripts/check_wasm_modulated_circle.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dynachaos.maps.modulated_circle import C_GOLDEN, longest_plateau_window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_WASM = PROJECT_ROOT / "site" / "wasm"
NPZ = PROJECT_ROOT / "figures" / "sec06_three_torus" / "double_staircase.npz"

# The published figure's iteration parameters (modulated_circle.py compute).
N_TRANSIENT = 3000
N_ITER = 20_000
STATE0 = [0.1, 0.1]

# Tolerances justified in the docstring: never raise them to make a run pass.
RHO_THETA_TOL = 1e-6
RHO_PHI_TOL = 1e-12

# The zoom panels' windows (plot_zoom): widest plateau within 5e-4 of the
# target, padded by the panel's own pad.
ZOOM_TARGETS = [(0.25, 0.008), (float(C_GOLDEN), 0.006)]

# One node process evaluates one tile per D and prints the flat results as
# JSON. JSON round-trips f64 exactly, so no value drifts through the
# encoding.
NODE_DRIVER = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
const dir = process.argv[1];
const spec = JSON.parse(readFileSync(0, "utf8"));
const mod = await import(pathToFileURL(join(dir, "dynachaos_wasm.js")).href);
await mod.default({ module_or_path: readFileSync(join(dir, "dynachaos_wasm_bg.wasm")) });
process.stdout.write(JSON.stringify(spec.map(s => Array.from(
  mod.modulated_circle_rotation_tile(
    s.a, s.c, s.d, s.d, 1, s.eps, s.nTransient, s.nIter, s.state0[0], s.state0[1])))));
"""


def wasm_tiles(spec: list[dict[str, float]]) -> list[list[float]]:
    """Return the wasm tile for each spec entry, as flat lists."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", NODE_DRIVER, str(SITE_WASM)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def zoom_points(D: np.ndarray, rho_theta: np.ndarray) -> list[int]:
    """Indices of every npz D inside either zoom window, as plot_zoom draws it."""
    idx: set[int] = set()
    for target, pad in ZOOM_TARGETS:
        window = longest_plateau_window(D, rho_theta, target, 5e-4)
        if window is None:
            continue
        mask = (D >= window[0] - pad) & (D <= window[1] + pad)
        idx.update(np.flatnonzero(mask).tolist())
    return sorted(idx)


def main() -> int:
    """Compare the wasm rotation numbers to the npz and report."""
    if not (SITE_WASM / "dynachaos_wasm.js").exists():
        print(
            f"no bundle in {SITE_WASM}. Run: uv run python scripts/build_wasm.py",
            file=sys.stderr,
        )
        return 1
    if not NPZ.exists():
        print(f"missing {NPZ}", file=sys.stderr)
        return 1

    data = np.load(NPZ)
    D = data["D"]
    rho_theta = data["rho_theta"]
    A = float(data["A"][0])
    C = float(data["C"][0])
    eps = float(data["eps"][0])

    indices = sorted({0, 2500, 5000, 7500, 9999} | set(zoom_points(D, rho_theta)))
    spec = [
        {
            "a": A,
            "c": C,
            "d": float(D[i]),
            "eps": eps,
            "nTransient": N_TRANSIENT,
            "nIter": N_ITER,
            "state0": STATE0,
        }
        for i in indices
    ]
    tiles = wasm_tiles(spec)

    failures: list[str] = []
    worst_theta = (-1.0, None)
    worst_phi = (-1.0, None)
    for i, tile in zip(indices, tiles, strict=True):
        d = float(D[i])
        header = tile[:4]
        want = [1, N_TRANSIENT, N_ITER, 2]
        if header != want:
            failures.append(f"D={d}: tile header {header} does not report {want}")
            continue
        rt, rp = tile[4], tile[5]
        d_theta = abs(rt - float(rho_theta[i]))
        d_phi = abs(rp - C)
        if d_theta > worst_theta[0]:
            worst_theta = (d_theta, d)
        if d_phi > worst_phi[0]:
            worst_phi = (d_phi, d)
        if d_theta > RHO_THETA_TOL:
            failures.append(
                f"D={d}: |rho_theta {rt!r} - npz {float(rho_theta[i])!r}| = "
                f"{d_theta:.3e} > {RHO_THETA_TOL}"
            )
        if d_phi > RHO_PHI_TOL:
            failures.append(f"D={d}: |rho_phi {rp!r} - C {C!r}| = {d_phi:.3e} > {RHO_PHI_TOL}")

    n_zoom = len(zoom_points(D, rho_theta))
    print(f"compared {len(indices)} D values ({n_zoom} inside the zoom windows)")
    print(f"worst |rho_theta difference| = {worst_theta[0]:.3e} at D = {worst_theta[1]}")
    print(f"worst |rho_phi - C| = {worst_phi[0]:.3e} at D = {worst_phi[1]}")
    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
