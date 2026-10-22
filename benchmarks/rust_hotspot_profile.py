#!/usr/bin/env python3
"""Profile candidate Rust-acceleration hotspots for dynachaos.

Usage is config-file based, matching ``scale_envelope.py``:

    uv run python benchmarks/rust_hotspot_profile.py [benchmarks/rust_hotspot_profile.jsonc]

Rows run in separate worker subprocesses so peak RSS is per case.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scale_envelope import _load_config, _machine, _vmhwm_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_suffix(".jsonc")
WORKER_ENV = "DYNACHAOS_RUST_HOTSPOT_WORKER"


def _rng(case: dict[str, Any]) -> np.random.Generator:
    return np.random.default_rng(int(case["config"].get("seed", 20260611)))


def _trajectory(case: dict[str, Any]) -> np.ndarray:
    rng = _rng(case)
    n = int(case["N"])
    dim = int(case.get("dim", 2))
    return np.cumsum(rng.normal(size=(n, dim)), axis=0).astype(np.float64)


def _radii(traj: np.ndarray, count: int) -> np.ndarray:
    span = float(np.max(np.ptp(traj, axis=0)))
    hi = max(span * 0.25, np.finfo(float).eps)
    lo = max(hi / 1000.0, np.finfo(float).eps)
    return np.geomspace(lo, hi, int(count)).astype(np.float64)


def _streaming_rqa_once(case: dict[str, Any]) -> dict[str, Any]:
    from dynachaos.diagnostics.rqa_streaming import rqa_streaming_from_trajectory

    traj = _trajectory(case)
    t0 = time.perf_counter()
    stats = rqa_streaming_from_trajectory(
        traj,
        eps=float(case["eps"]),
        metric=str(case["metric"]),
        l_min=int(case["l_min"]),
        v_min=int(case["v_min"]),
    )
    wall = time.perf_counter() - t0
    return {"wall_time_s": wall, "RR": float(stats["RR"]), "DET": float(stats["DET"])}


def _exact_pair_count_once(case: dict[str, Any]) -> dict[str, Any]:
    from dynachaos.diagnostics import correlation as corr_mod

    traj = _trajectory(case)
    r_values = _radii(traj, int(case["r_count"]))
    old = bool(corr_mod._RUST_AVAILABLE)
    if case["backend"] == "rust" and not old:
        return {"skipped": True, "skip_reason": "rust extension unavailable"}
    corr_mod._RUST_AVAILABLE = case["backend"] == "rust" and old
    try:
        t0 = time.perf_counter()
        cvals = corr_mod.correlation_integral(
            traj,
            r_values,
            theiler_window=int(case["theiler_window"]),
            norm=str(case["norm"]),
        )
        wall = time.perf_counter() - t0
    finally:
        corr_mod._RUST_AVAILABLE = old
    return {"wall_time_s": wall, "C_last": float(cvals[-1])}


def _comoving_once(case: dict[str, Any]) -> dict[str, Any]:
    from dynachaos.diagnostics import comoving_lyapunov as cm_mod

    rng = _rng(case)
    x_init = rng.uniform(-1.0, 1.0, int(case["N"])).astype(np.float64)
    v_values = np.linspace(-1.0, 1.0, int(case["velocity_count"]), dtype=np.float64)
    old = bool(cm_mod._RUST_AVAILABLE)
    if case["backend"] == "rust" and not old:
        return {"skipped": True, "skip_reason": "rust extension unavailable"}
    cm_mod._RUST_AVAILABLE = case["backend"] == "rust" and old
    try:
        t0 = time.perf_counter()
        lam = cm_mod.comoving_lyapunov_spectrum_logistic(
            a=float(case["a"]),
            eps=float(case["eps"]),
            N=int(case["N"]),
            v_values=v_values,
            n_iter=int(case["n_iter"]),
            n_transient=int(case["n_transient"]),
            x_init=x_init,
        )
        wall = time.perf_counter() - t0
    finally:
        cm_mod._RUST_AVAILABLE = old
    return {"wall_time_s": wall, "lambda_mean": float(np.mean(lam))}


def _basin_once(case: dict[str, Any]) -> dict[str, Any]:
    import importlib

    basin_mod = importlib.import_module("dynachaos.maps.coupled_logistic")
    x_range = np.linspace(-1.0, 1.0, int(case["n_grid"]), dtype=np.float64)
    y_range = np.linspace(-1.0, 1.0, int(case["n_grid"]), dtype=np.float64)
    ref_a = basin_mod._find_reference_orbit(
        float(case["A"]),
        float(case["D"]),
        0.1,
        -0.2,
        n_transient=int(case["reference_transient"]),
        period=int(case["period"]),
    )
    old = bool(basin_mod._RUST_AVAILABLE)
    if case["backend"] == "rust" and not old:
        return {"skipped": True, "skip_reason": "rust extension unavailable"}
    basin_mod._RUST_AVAILABLE = case["backend"] == "rust" and old
    try:
        t0 = time.perf_counter()
        basin = basin_mod._basin_grid(
            float(case["A"]),
            float(case["D"]),
            x_range,
            y_range,
            int(case["n_transient"]),
            ref_a,
        )
        wall = time.perf_counter() - t0
    finally:
        basin_mod._RUST_AVAILABLE = old
    unique, counts = np.unique(basin, return_counts=True)
    label_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    return {"wall_time_s": wall, "label_counts": label_counts}


def _run_once(case: dict[str, Any]) -> dict[str, Any]:
    if case["kind"] == "streaming_rqa":
        return _streaming_rqa_once(case)
    if case["kind"] == "exact_pair_count":
        return _exact_pair_count_once(case)
    if case["kind"] == "comoving_logistic":
        return _comoving_once(case)
    if case["kind"] == "coupled_logistic_basin":
        return _basin_once(case)
    raise ValueError(case["kind"])


def _worker() -> None:
    case = json.loads(os.environ[WORKER_ENV])
    repeats = int(case["config"].get("repeats", 3))
    runs = [_run_once(case) for _ in range(repeats)]
    if runs and runs[0].get("skipped"):
        representative = runs[0]
    else:
        p50 = float(median([run["wall_time_s"] for run in runs]))
        representative = runs[len(runs) // 2]
        representative["wall_time_s_p50"] = p50
        representative["repeat_wall_time_s"] = [float(run["wall_time_s"]) for run in runs]
    representative["peak_rss_bytes"] = _vmhwm_bytes()
    print(json.dumps(representative, allow_nan=False))


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env[WORKER_ENV] = json.dumps(case)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed for {case.get('name')}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Rust hotspot profile",
        "",
        f"Generated: {results['generated_at_utc']}",
        f"Command: `{results['command']}`",
        "",
        "## Hardware and software",
        "",
        f"- CPU: {results['machine']['cpu_model']}",
        f"- Platform: {results['machine']['platform']}",
        f"- Python: {results['machine']['python']}; NumPy: {results['machine']['numpy']}",
        "",
        "## Measurements",
        "",
        "| case | kind | backend | size | p50 wall s | peak RSS MB | notes |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in results["rows"]:
        case = row["case"]
        backend = case.get("backend", "python")
        if row.get("skipped"):
            lines.append(
                f"| {case['name']} | {case['kind']} | {backend} | {row['size']} | skipped "
                f"| {row['peak_rss_bytes'] / 1e6:.1f} | {row['skip_reason']} |"
            )
            continue
        notes = ""
        if "RR" in row:
            notes = f"RR={row['RR']:.6g}; DET={row['DET']:.6g}"
        elif "C_last" in row:
            notes = f"C_last={row['C_last']:.6g}"
        elif "lambda_mean" in row:
            notes = f"lambda_mean={row['lambda_mean']:.6g}"
        elif "label_counts" in row:
            notes = f"labels={row['label_counts']}"
        lines.append(
            f"| {case['name']} | {case['kind']} | {backend} | {row['size']} "
            f"| {row['wall_time_s_p50']:.6g} | {row['peak_rss_bytes'] / 1e6:.1f} | {notes} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_size(case: dict[str, Any]) -> int:
    if "N" in case:
        return int(case["N"])
    return int(case["n_grid"]) * int(case["n_grid"])


def main(argv: list[str] | None = None) -> int:
    if WORKER_ENV in os.environ:
        _worker()
        return 0
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) > 1:
        raise SystemExit("usage: rust_hotspot_profile.py [config.jsonc]")
    cfg_path = Path(argv[0]) if argv else DEFAULT_CONFIG
    cfg = _load_config(cfg_path)
    rows = []
    for case_cfg in cfg["cases"]:
        case = {**case_cfg, "config": cfg}
        res = _run_case(case)
        rows.append({"case": case_cfg, "size": _case_size(case_cfg), **res})
    results = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Relative to the repository root; this artifact is committed publicly.
        "command": f"python benchmarks/{Path(__file__).name} {cfg_path.name}",
        "machine": _machine() | {"platform_uname": platform.platform()},
        "rows": rows,
    }
    json_path = ROOT / cfg["output"]["json"]
    md_path = ROOT / cfg["output"]["markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
