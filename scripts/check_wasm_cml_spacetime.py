"""Check the live CML space-time figure against the published npz.

The paper's space-time panels
(``figures/sec08_sti/spacetime_diagrams.npz``) are the field x_i^n of each
model at N = 200 sites, n_transient 2000, n_record 500, from
``x0 = default_rng(42).uniform(0, 1, 200)``
(``cml/spatiotemporal.py`` ``simulate_cml``). The live figure computes the
same field with the wasm ``cml_spacetime_tile`` and continues it in chunks
from the window's last row.

The rule: models A and C are bit-exact at every value — they use only
``+, -, *`` so bit-exact is required by construction. Model B's ``sin`` is
the wasm module's own, which differs from the platform sine by 1 ulp on
about 3 % of inputs, so bit-exact is not achievable there. Each B key is
classified with numpy alone: eight runs of ``simulate_cml`` in which
every ``sin`` result is moved one ulp up or down (random sign, seed k
for run k, ``np.nextafter``). This models the cause of the wasm
difference, a 1-ulp error in the sine at each step. The ensemble max is
the largest ``max |run - npz|`` over the eight runs.

- Not sensitive (ensemble max < 1e-9): the wasm field must match the npz
  sample by sample within 1e-9. The per-step sin error is about 1e-16
  and nothing amplifies it; the colour map resolves (range)/256 >= 1e-3,
  so 1e-9 is far below anything visible.
- Sensitive: compare value histograms — 64 bins between the npz min and
  max. The wasm-vs-npz L1 distance must be at most the largest L1
  distance of the eight jittered numpy runs vs the npz. Each jittered run
  has a larger sine error than the wasm run (every step, not 3 % of
  them), so the wasm field must not look more different than the worst
  of them.

Two more checks pin the live figure's own conventions:

- ``site/live/cml-x0.json``'s first 200 values must equal the paper's x0
  exactly — the page slices that file to the site count, so a wrong file
  silently changes every orbit.
- A run of 2000 + 250 rows continued by 0 + 250 rows from its last row
  must equal the 500-row run bit for bit — the play button's chunk
  continuation is only honest if the kernel sees the last row unchanged.

On any difference the script prints the first differing
(key, row, site, both values) and stops.

Run it with::

    uv run python scripts/check_wasm_cml_spacetime.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from dynachaos.cml.spatiotemporal import simulate_cml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_WASM = PROJECT_ROOT / "site" / "wasm"
NPZ = PROJECT_ROOT / "figures" / "sec08_sti" / "spacetime_diagrams.npz"
X0_JSON = PROJECT_ROOT / "site" / "live" / "cml-x0.json"

# The published figure's parameters (spatiotemporal.py compute/simulate_cml).
N_SITES = 200
N_TRANSIENT = 2000
N_RECORD = 500
CHUNK_ROWS = 250

MODEL_IDS = {"A": 0, "B": 1, "C": 2}
# Each model's default eps on the page: the middle of its published sweep,
# also the wasm export's non-finite fallback (CML_EPS_FALLBACK).
MODEL_EPS = {"A": 0.07, "B": 0.024, "C": 0.2}

# Model B's rule: a key is sensitive when a 1-ulp error on every sin
# result decorrelates the orbit (ensemble max >= ENSEMBLE_TOL); a
# not-sensitive key must match the npz sample by sample within
# SAMPLE_TOL, a sensitive key must match its value histogram within
# HIST_FACTOR times the ensemble's worst.
ENSEMBLE_TOL = 1e-9
SAMPLE_TOL = 1e-9
HIST_BINS = 64
HIST_FACTOR = 1.0

# One node process evaluates every tile and prints the flat results as
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
  mod.cml_spacetime_tile(
    s.model, s.eps, s.nSites, s.nTransient, s.nRecord,
    Float64Array.from(s.x0))))));
"""


def wasm_tiles(spec: list[dict[str, object]]) -> list[list[float]]:
    """Return the wasm tile for each spec entry, as flat lists."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", NODE_DRIVER, str(SITE_WASM)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def fail(key: str, row: int, site: int, got: float, want: float) -> int:
    """Report the first differing value and stop."""
    print(
        f"FAIL: {key} row {row} site {site}: wasm {got!r} != npz {want!r}",
        file=sys.stderr,
    )
    return 1


def compare_field(key: str, tile: list[float], want: np.ndarray) -> int:
    """Compare one wasm tile against an npz field, bit for bit."""
    n_rows, n_sites = want.shape
    header = tile[:4]
    if len(tile) != 4 + n_rows * n_sites:
        print(
            f"FAIL: {key}: tile length {len(tile)} != 4 + {n_rows} x {n_sites}",
            file=sys.stderr,
        )
        return 1
    if header[1] != n_sites or header[3] != n_rows:
        print(
            f"FAIL: {key}: tile header {header} does not report {n_sites} sites x {n_rows} rows",
            file=sys.stderr,
        )
        return 1
    flat = want.ravel()
    for k, got in enumerate(tile[4:]):
        if got != float(flat[k]):
            return fail(key, k // n_sites, k % n_sites, got, float(flat[k]))
    return 0


def jittered_model_b(seed: int) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
    """Return model B's (f, g) with every sin result moved one ulp.

    The direction of each move is random (up or down, equal odds), drawn
    from ``default_rng(seed)``. The formulas are those of
    ``model_B_f`` and ``model_B_g`` with their default A and C.
    """
    rng = np.random.default_rng(seed)

    def sin_1ulp(x: np.ndarray) -> np.ndarray:
        s = np.sin(2 * np.pi * x)
        up = rng.random(s.shape) < 0.5
        return np.where(up, np.nextafter(s, np.inf), np.nextafter(s, -np.inf))

    def f(x: np.ndarray, A: float = 0.2, C: float = 0.55) -> np.ndarray:
        return (x + A * sin_1ulp(x) + C) % 1.0

    return f, sin_1ulp


def perturbation_ensemble(eps: float, x0: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    """Return (ensemble max |run - npz|, ensemble max histogram L1).

    Eight numpy runs of ``simulate_cml`` at the key's eps from the paper's
    x0: run k uses ``jittered_model_b(k)``. The first value classifies the
    key; the second is the histogram reference for a sensitive key.
    """
    lo, hi = float(want.min()), float(want.max())
    want_hist = np.histogram(want, bins=HIST_BINS, range=(lo, hi))[0]
    ens_max = 0.0
    ens_l1 = 0.0
    for k in range(8):
        f, g = jittered_model_b(k)
        run = simulate_cml(
            f,
            g,
            eps,
            N=N_SITES,
            n_transient=N_TRANSIENT,
            n_record=N_RECORD,
            x0=x0,
        )
        ens_max = max(ens_max, float(np.abs(run - want).max()))
        run_hist = np.histogram(run, bins=HIST_BINS, range=(lo, hi))[0]
        ens_l1 = max(ens_l1, float(np.abs(run_hist - want_hist).sum()))
    return ens_max, ens_l1


def compare_b_key(key: str, tile: list[float], want: np.ndarray, x0: np.ndarray) -> int:
    """Compare one model-B wasm tile against the npz by the plan's rule."""
    n_rows, n_sites = want.shape
    header = tile[:4]
    if len(tile) != 4 + n_rows * n_sites:
        print(
            f"FAIL: {key}: tile length {len(tile)} != 4 + {n_rows} x {n_sites}",
            file=sys.stderr,
        )
        return 1
    if header[1] != n_sites or header[3] != n_rows:
        print(
            f"FAIL: {key}: tile header {header} does not report {n_sites} sites x {n_rows} rows",
            file=sys.stderr,
        )
        return 1
    eps = float(key.rsplit("_", 1)[1])
    ens_max, ens_l1 = perturbation_ensemble(eps, x0, want)
    got = np.asarray(tile[4:], dtype=np.float64).reshape(want.shape)
    if ens_max < ENSEMBLE_TOL:
        stat = float(np.abs(got - want).max())
        bound = SAMPLE_TOL
        print(
            f"{key}: class=not-sensitive ensemble_max={ens_max:.3e} "
            f"wasm max|d|={stat:.3e} bound={bound:.3e}"
        )
        if stat <= bound:
            return 0
        diff = np.abs(got - want)
        k = int(np.flatnonzero(diff.ravel() > bound)[0])
        return fail(key, k // n_sites, k % n_sites, float(got.ravel()[k]), float(want.ravel()[k]))
    lo, hi = float(want.min()), float(want.max())
    want_hist = np.histogram(want, bins=HIST_BINS, range=(lo, hi))[0]
    got_hist = np.histogram(got, bins=HIST_BINS, range=(lo, hi))[0]
    stat = float(np.abs(got_hist - want_hist).sum())
    bound = HIST_FACTOR * ens_l1
    print(
        f"{key}: class=sensitive ensemble_max={ens_max:.3e} "
        f"wasm hist L1={stat:.3e} bound={bound:.3e}"
    )
    if stat <= bound:
        return 0
    print(
        f"FAIL: {key}: histogram L1 {stat:.3e} exceeds {HIST_FACTOR:g} x "
        f"the ensemble's worst {ens_l1:.3e}",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    """Compare the wasm fields to the npz and report."""
    if not (SITE_WASM / "dynachaos_wasm.js").exists():
        print(
            f"no bundle in {SITE_WASM}. Run: uv run python scripts/build_wasm.py",
            file=sys.stderr,
        )
        return 1
    if not NPZ.exists():
        print(f"missing {NPZ}", file=sys.stderr)
        return 1
    if not X0_JSON.exists():
        print(
            f"missing {X0_JSON}. Run: uv run python scripts/build_paper.py",
            file=sys.stderr,
        )
        return 1

    data = np.load(NPZ)
    rng = np.random.default_rng(42)
    x0 = rng.uniform(0, 1, N_SITES)

    # The page's initial field must be the paper's, exactly.
    page_x0 = np.array(json.loads(X0_JSON.read_text(encoding="utf-8")))
    if page_x0.shape[0] < N_SITES or not np.array_equal(page_x0[:N_SITES], x0):
        print(
            f"FAIL: {X0_JSON.name}'s first {N_SITES} values differ from "
            "default_rng(42).uniform(0, 1, 200)",
            file=sys.stderr,
        )
        return 1
    print(f"cml-x0.json: first {N_SITES} values equal the paper's x0 exactly")

    keys = sorted(data.files)
    spec = [
        {
            "model": MODEL_IDS[key.split("_")[0]],
            "eps": float(key.rsplit("_", 1)[1]),
            "nSites": N_SITES,
            "nTransient": N_TRANSIENT,
            "nRecord": N_RECORD,
            "x0": x0.tolist(),
        }
        for key in keys
    ]
    tiles = wasm_tiles(spec)
    for key, tile in zip(keys, tiles, strict=True):
        if key.startswith("B_"):
            rc = compare_b_key(key, tile, data[key], x0)
        else:
            rc = compare_field(key, tile, data[key])
            if not rc:
                print(
                    f"{key}: class=bit-exact ensemble_max=- "
                    f"wasm max|d|=0.000e+00 bound=0.000e+00 "
                    f"({N_RECORD} rows x {N_SITES} sites)"
                )
        if rc:
            return rc
    cont_spec = []
    for letter in ("A", "B", "C"):
        eps = MODEL_EPS[letter]
        model = MODEL_IDS[letter]
        cont_spec.append(
            {
                "model": model,
                "eps": eps,
                "nSites": N_SITES,
                "nTransient": N_TRANSIENT,
                "nRecord": N_RECORD,
                "x0": x0.tolist(),
            }
        )
        cont_spec.append(
            {
                "model": model,
                "eps": eps,
                "nSites": N_SITES,
                "nTransient": N_TRANSIENT,
                "nRecord": CHUNK_ROWS,
                "x0": x0.tolist(),
            }
        )
    cont_tiles = wasm_tiles(cont_spec)
    last_rows = []
    for i, letter in enumerate(("A", "B", "C")):
        first = cont_tiles[2 * i + 1]
        last_rows.append(first[4 + (CHUNK_ROWS - 1) * N_SITES : 4 + CHUNK_ROWS * N_SITES])
    tail_spec = [
        {
            "model": MODEL_IDS[letter],
            "eps": MODEL_EPS[letter],
            "nSites": N_SITES,
            "nTransient": 0,
            "nRecord": CHUNK_ROWS,
            "x0": last_rows[i],
        }
        for i, letter in enumerate(("A", "B", "C"))
    ]
    tail_tiles = wasm_tiles(tail_spec)
    for i, letter in enumerate(("A", "B", "C")):
        full = cont_tiles[2 * i]
        tail = tail_tiles[i]
        key = f"{letter}_eps_{MODEL_EPS[letter]} continuation"
        for r in range(CHUNK_ROWS):
            for s in range(N_SITES):
                got = tail[4 + r * N_SITES + s]
                want = full[4 + (CHUNK_ROWS + r) * N_SITES + s]
                if got != want:
                    return fail(key, CHUNK_ROWS + r, s, got, want)
        print(f"{key}: 250 continued rows bit-exact")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
