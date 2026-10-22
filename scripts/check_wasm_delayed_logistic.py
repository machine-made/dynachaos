"""Check the live delayed-logistic attractor against the published npz files.

The paper's attractor portraits (``figures/sec05_oscillation/attractors.npz``)
and the locking-to-chaos zoom (``locking_sequence.npz``) were computed by
``maps/delayed_logistic.py`` ``compute_attractor``: A = 0.3, n_transient
20000, n_plot 100000, starting from ``(fp + 0.01, fp - 0.01)`` where
``fp = (sqrt(1 + 4 D) - 1) / (2 D)`` is the analytic fixed point. The live
figure computes the same orbit with the wasm
``delayed_logistic_attractor_tile`` (n_D = 1 at the chosen D) and plots at
most 4096 states, so this script compares the tile's first 4096 post-transient
samples against the first 4096 rows of each committed trajectory.

The rule is bit-exact equality, and the reason is the map itself:
``(x, y) -> (A x + (1 - A)(1 - D y^2), x)`` uses only +, -, *, so there is no
transcendental call for the wasm build to evaluate differently — unlike the
circle map's sine, whose one-ulp platform difference is what forces the
tolerances in ``check_wasm_parity.py`` and ``check_wasm_staircase.py``. The
Rust kernel evaluates the same expression in the same order
(``rust/core/src/maps.rs`` ``delayed_logistic_step``), so every sample must
match bit for bit, including at the chaotic D values. If a D is not
bit-exact, this script fails and reports it; the fix is in the kernel's
operation order, never a tolerance here.

Run it with::

    uv run python scripts/check_wasm_delayed_logistic.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_WASM = PROJECT_ROOT / "site" / "wasm"
NPZ_DIR = PROJECT_ROOT / "figures" / "sec05_oscillation"

# The published figures' parameters (maps/delayed_logistic.py).
A = 0.3
N_TRANSIENT = 20_000
N_PLOT = 4096  # the live figure's cap; the npz trajectories hold 100000

# One node process evaluates one tile per D and prints the flat results as
# JSON. JSON round-trips f64 exactly, so no value drifts through the
# encoding; NaN arrives as null and is mapped back before comparing.
NODE_DRIVER = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
const dir = process.argv[1];
const spec = JSON.parse(readFileSync(0, "utf8"));
const mod = await import(pathToFileURL(join(dir, "dynachaos_wasm.js")).href);
await mod.default({ module_or_path: readFileSync(join(dir, "dynachaos_wasm_bg.wasm")) });
process.stdout.write(JSON.stringify(spec.map(s => Array.from(
  mod.delayed_logistic_attractor_tile(
    s.a, s.d, s.d, 1, s.nTransient, s.nPlot, s.state0)))));
"""


def fixed_point(d: float) -> float:
    """The analytic fixed point the published orbits start beside."""
    return (math.sqrt(1.0 + 4.0 * d) - 1.0) / (2.0 * d)


def wasm_tiles(spec: list[dict]) -> list[list[float]]:
    """Return the wasm tile for each spec entry, as flat lists."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", NODE_DRIVER, str(SITE_WASM)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def compare(name: str, d: float, reference: np.ndarray) -> list[str]:
    """Compare one D's wasm tile against the npz trajectory, bit for bit."""
    fp = fixed_point(d)
    spec = {
        "a": A,
        "d": d,
        "nTransient": N_TRANSIENT,
        "nPlot": N_PLOT,
        "state0": [fp + 0.01, fp - 0.01],
    }
    (tile,) = wasm_tiles([spec])
    header = tile[:4]
    want = [1, N_PLOT, N_TRANSIENT, 2]
    if header != want:
        return [f"{name} D={d}: tile header {header} does not report {want}"]
    values = np.array([np.nan if v is None else v for v in tile[4:]], dtype=np.float64).reshape(
        N_PLOT, 2
    )
    expected = reference[:N_PLOT]
    if np.array_equal(values, expected):
        return []
    bad = np.nonzero(~(values == expected))[0]
    i = int(bad[0])
    return [
        f"{name} D={d}: {len(bad)} of {N_PLOT} states differ; "
        f"first at sample {i}: wasm {values[i]!r} vs npz {expected[i]!r}"
    ]


def main() -> int:
    """Compare the wasm attractor tiles to both npz files and report."""
    if not (SITE_WASM / "dynachaos_wasm.js").exists():
        print(
            f"no bundle in {SITE_WASM}. Run: uv run python scripts/build_wasm.py",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []
    compared = 0
    n_d = 0
    for npz_name, fmt in (("attractors", ".2f"), ("locking_sequence", ".3f")):
        path = NPZ_DIR / f"{npz_name}.npz"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1
        data = np.load(path)
        d_values = data["D_values"]
        if float(data["A"][0]) != A:
            print(f"FAIL: {npz_name}.npz was computed at A={data['A'][0]}, not {A}")
            return 1
        for d in d_values:
            d = float(d)
            n_d += 1
            key = f"D_{d:{fmt}}_x"
            reference = np.column_stack([data[key], data[f"D_{d:{fmt}}_y"]])
            problems = compare(npz_name, d, reference)
            compared += N_PLOT
            if problems:
                failures.extend(problems)
            else:
                print(f"{npz_name} D={d}: {N_PLOT} states bit-exact")

    print(f"compared {compared} plotted states across {n_d} D values")
    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
