"""Check the live devil's staircase against the published npz.

The paper's staircase (``figures/sec02_circle_map/devils_staircase.npz``) is
rho(A) at D = 0.25 on ``np.linspace(0, 0.25, 200000)`` with n_transient 5000,
n_iter 50000 and theta0 0.1. The live figure computes the same curve with the
wasm ``rotation_number_tile`` at Omega = D (n_omega = 1) and K = A.

This script evaluates the wasm tile at a few hundred of the npz A values and
compares element by element. The tile's K grid must reproduce npz points bit
for bit — comparing against a nearby point would measure the grid, not the
kernel — so each tile's K range is chosen such that its linspace lands exactly
on npz entries, and the script asserts that before trusting the comparison.

The rule is the one ``scripts/check_wasm_parity.py`` applies to this kernel,
with its thresholds imported rather than copied: below K_c = 1/(2*pi) the map
is well conditioned and the tile's bounded-error stop keeps every value within
one colour step (DISPLAY_TOLERANCE = 1/256) of the full-iteration npz value.
Above K_c the orbit is chaotic and the wasm sine diverges from the platform
one, so divergence is allowed but must stay rare — the same share limit the
parity check uses, applied to cells past the display tolerance.

Run it with::

    uv run python scripts/check_wasm_staircase.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from math import isfinite
from pathlib import Path

import numpy as np
from check_wasm_parity import CHAOTIC_SHARE_LIMIT, DISPLAY_TOLERANCE, K_CRITICAL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_WASM = PROJECT_ROOT / "site" / "wasm"
NPZ = PROJECT_ROOT / "figures" / "sec02_circle_map" / "devils_staircase.npz"

# The published figure's parameters (maps/circle_map.py compute()).
D = 0.25
N_TRANSIENT = 5000
N_ITER = 50_000
THETA0 = 0.1

# Tiles of 64 rows: large enough to exercise the kernel's grid path, small
# enough that bit-exact K ranges are easy to find. Eight tiles spread over
# [0, 0.25] give 512 points, a few hundred as the goal asks.
TILE_ROWS = 64
N_TILES = 8

# One node process evaluates every selected tile and prints the flat results
# as JSON. JSON round-trips f64 exactly, so no value drifts through the
# encoding.
NODE_DRIVER = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
const dir = process.argv[1];
const spec = JSON.parse(readFileSync(0, "utf8"));
const mod = await import(pathToFileURL(join(dir, "dynachaos_wasm.js")).href);
await mod.default({ module_or_path: readFileSync(join(dir, "dynachaos_wasm_bg.wasm")) });
process.stdout.write(JSON.stringify(spec.map(s => Array.from(mod.rotation_number_tile(
  s.omega, s.omega, 1, s.kMin, s.kMax, s.nK, s.nTransient, s.nIter, s.theta0)))));
"""


def tile_k_grid(k_min: float, k_max: float, n_k: int) -> np.ndarray:
    """Return the K values the kernel iterates, as numpy computes them.

    The kernel's linspace is start + i * (stop - start) / (n - 1) with the
    last value forced to stop — the same formula numpy uses here.
    """
    grid = k_min + np.arange(n_k) * ((k_max - k_min) / (n_k - 1))
    grid[-1] = k_max
    return grid


def find_exact_tiles(a_values: np.ndarray) -> list[tuple[int, int]]:
    """Return (i0, stride) pairs whose 64-row tile grid is bit-exact on the npz.

    A tile [A[i0], A[i0 + 63*stride]] with TILE_ROWS rows reproduces the npz
    points A[i0 + j*stride] bit for bit only for some (i0, stride) pairs —
    the linspace step must come out identical. Scan a spread of start indices
    and keep the first stride that works at each.
    """
    last = len(a_values) - 1
    found: list[tuple[int, int]] = []
    for i0 in range(0, last, 997):
        for stride in range(1, 400):
            i1 = i0 + (TILE_ROWS - 1) * stride
            if i1 > last:
                break
            grid = tile_k_grid(a_values[i0], a_values[i1], TILE_ROWS)
            if np.array_equal(grid, a_values[i0 + np.arange(TILE_ROWS) * stride]):
                found.append((i0, stride))
                break
    return found


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


def main() -> int:
    """Compare the wasm staircase to the npz and report."""
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
    a_values = data["A"]
    rho = data["rho"]

    found = find_exact_tiles(a_values)
    if len(found) < N_TILES:
        print(f"FAIL: only {len(found)} bit-exact tile ranges found, need {N_TILES}")
        return 1
    step = max(1, len(found) // N_TILES)
    selected = found[::step][:N_TILES]

    spec = []
    expected: list[tuple[float, float]] = []  # (A, npz rho) per compared point
    for i0, stride in selected:
        indices = i0 + np.arange(TILE_ROWS) * stride
        k_min = float(a_values[i0])
        k_max = float(a_values[indices[-1]])
        # The comparison is only honest at the npz's own A values: assert the
        # tile's K grid reproduces them bit for bit before trusting it.
        grid = tile_k_grid(k_min, k_max, TILE_ROWS)
        if not np.array_equal(grid, a_values[indices]):
            print(
                f"FAIL: tile K range [{k_min}, {k_max}] does not reproduce "
                f"npz A values bit for bit at i0={i0} stride={stride}"
            )
            return 1
        spec.append(
            {
                "omega": D,
                "kMin": k_min,
                "kMax": k_max,
                "nK": TILE_ROWS,
                "nTransient": N_TRANSIENT,
                "nIter": N_ITER,
                "theta0": THETA0,
            }
        )
        expected.extend((float(a_values[i]), float(rho[i])) for i in indices)

    tiles = wasm_tiles(spec)
    if len(tiles) != len(spec):
        print(f"FAIL: wasm returned {len(tiles)} tiles, expected {len(spec)}")
        return 1

    below_worst = 0.0
    above_cells = 0
    above_diverged = 0
    above_worst = 0.0
    compared = 0
    for tile, entry in zip(tiles, spec, strict=True):
        header = tile[:4]
        want = [1, TILE_ROWS, N_TRANSIENT, N_ITER]
        if header != want:
            print(f"FAIL: tile header {header} does not report {want}")
            return 1
        values = tile[4:]
        if len(values) != TILE_ROWS:
            print(f"FAIL: tile has {len(values)} values, expected {TILE_ROWS}")
            return 1
        for j, value in enumerate(values):
            a_value, reference = expected[compared]
            compared += 1
            if not (isfinite(value) and isfinite(reference)):
                print(f"FAIL: non-finite value at A={a_value} (wasm {value!r}, npz {reference!r})")
                return 1
            difference = abs(value - reference)
            if a_value <= K_CRITICAL:
                below_worst = max(below_worst, difference)
                if difference > DISPLAY_TOLERANCE:
                    print(
                        f"FAIL: A={a_value} below K_c differs by {difference:.3e}, "
                        f"above {DISPLAY_TOLERANCE:.3e}"
                    )
                    return 1
            else:
                above_cells += 1
                above_worst = max(above_worst, difference)
                if difference > DISPLAY_TOLERANCE:
                    above_diverged += 1

    share = above_diverged / max(above_cells, 1)
    print(f"compared {compared} points over {len(spec)} tiles at D = {D}")
    print(f"  A <= K_c = {K_CRITICAL:.4f}: worst difference {below_worst:.3e}")
    print(
        f"  A > K_c: {above_diverged} of {above_cells} points diverged past "
        f"{DISPLAY_TOLERANCE:.3e} ({share:.1%}), worst {above_worst:.3e}"
    )
    if share > CHAOTIC_SHARE_LIMIT:
        print(
            f"FAIL: {share:.1%} of the chaotic points diverged past the display "
            f"tolerance, above the {CHAOTIC_SHARE_LIMIT:.0%} guide."
        )
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
