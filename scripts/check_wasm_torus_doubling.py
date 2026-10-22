"""Check the live torus-doubling attractor against the published npz files.

The paper's attractor portraits (``figures/sec04_doubling/map_I_attractors.npz``
and ``map_IV_attractors.npz``) were computed by ``maps/torus_doubling.py``
``compute_map_I``/``compute_map_IV``: map (I) at A = 0.4 from
x0 = (0.5, 0.5, 0.5), map (IV) at A = 0.3 from x0 = (0.5, 0.45, 0.52, 0.48),
both with n_transient 20000 and n_plot 100000. The live figure computes the
same orbit with the wasm ``torus_doubling_attractor_tile`` (one call at the
chosen map kind and D) and plots at most 4096 states, so this script compares
the tile's first 4096 post-transient samples — every state component, not
just the plotted (X, Y) pair — against the first 4096 rows of each committed
trajectory.

The rule is bit-exact equality, and the reason is the map itself: with
``L_D(u) = 1 - D u^2``, map (I) steps ``(X, Y, Z)`` to
``(A X + (1 - A) L_D(Y), Z, X)`` and map (IV) steps ``(X, Y, Z, W)`` to
``(A X + (1 - A) L_D(Y), Z, A Z + (1 - A) L_D(W), X)`` — only +, -, *, so
there is no transcendental call for the wasm build to evaluate differently.
The Rust kernel evaluates the same expressions in the same order
(``rust/core/src/maps.rs`` ``torus_map_i_step``/``torus_map_iv_step``), so
every sample must match bit for bit, including at the chaotic D values. If a
D is not bit-exact, this script fails and reports it; the fix is in the
kernel's operation order, never a tolerance here.

Run it with::

    uv run python scripts/check_wasm_torus_doubling.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_WASM = PROJECT_ROOT / "site" / "wasm"
NPZ_DIR = PROJECT_ROOT / "figures" / "sec04_doubling"

# The published figures' parameters (maps/torus_doubling.py).
N_TRANSIENT = 20_000
N_PLOT = 4096  # the live figure's cap; the npz trajectories hold 100000
MAPS = {
    "map_I_attractors": {"kind": 1, "a": 0.4, "dim": 3, "x0": [0.5, 0.5, 0.5]},
    "map_IV_attractors": {
        "kind": 4,
        "a": 0.3,
        "dim": 4,
        "x0": [0.5, 0.45, 0.52, 0.48],
    },
}

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
  mod.torus_doubling_attractor_tile(
    s.kind, s.a, s.d, s.nTransient, s.nPlot, s.state0)))));
"""


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


def compare(name: str, cfg: dict, d: float, reference: np.ndarray) -> list[str]:
    """Compare one D's wasm tile against the npz trajectory, bit for bit."""
    spec = {
        "kind": cfg["kind"],
        "a": cfg["a"],
        "d": d,
        "nTransient": N_TRANSIENT,
        "nPlot": N_PLOT,
        "state0": cfg["x0"],
    }
    (tile,) = wasm_tiles([spec])
    header = tile[:4]
    want = [cfg["kind"], cfg["dim"], N_TRANSIENT, N_PLOT]
    if header != want:
        return [f"{name} D={d}: tile header {header} does not report {want}"]
    values = np.array([np.nan if v is None else v for v in tile[4:]], dtype=np.float64).reshape(
        N_PLOT, cfg["dim"]
    )
    expected = reference[:N_PLOT]
    if np.array_equal(values, expected):
        return []
    bad = np.nonzero(~(values == expected).all(axis=1))[0]
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
    for npz_name, cfg in MAPS.items():
        path = NPZ_DIR / f"{npz_name}.npz"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1
        data = np.load(path)
        for d in data["D_values"]:
            d = float(d)
            n_d += 1
            key = f"D_{d}_traj"
            if key not in data:
                print(f"missing {key} in {path}", file=sys.stderr)
                return 1
            reference = data[key]
            problems = compare(npz_name, cfg, d, reference)
            compared += N_PLOT
            if problems:
                failures.extend(problems)
            else:
                print(
                    f"{npz_name} (map {cfg['kind']}, A={cfg['a']}) D={d}: "
                    f"{N_PLOT} states x {cfg['dim']} components bit-exact"
                )

    print(f"compared {compared} plotted states across {n_d} D values")
    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
