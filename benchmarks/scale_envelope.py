#!/usr/bin/env python3
"""Scale-envelope benchmark for long-signal diagnostics.

Public usage is intentionally config-file based:

    uv run python benchmarks/scale_envelope.py [benchmarks/scale_envelope.jsonc]

The script writes JSON and Markdown summaries.  Worker subprocesses are launched
via environment variables so every measured case has isolated peak-RSS state.
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

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_suffix(".jsonc")
WORKER_ENV = "DYNACHAOS_SCALE_ENVELOPE_WORKER"
RQA_TEMPORARIES_MULTIPLIER = 3


def _strip_jsonc(text: str) -> str:
    """Remove JSONC comments without touching comment markers in strings."""
    out: list[str] = []
    i = 0
    in_string = False
    escaped = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(i + 2, len(text))
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))


def _vmhwm_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    if resource is None:
        # Windows has no resource module and no /proc, so peak RSS is simply
        # unavailable. Report 0 rather than failing at import time, which took
        # the whole scale-envelope test module down on Windows CI.
        return 0
    # Linux ru_maxrss is KiB; macOS is bytes.  This repo is developed on Linux,
    # but keep the fallback sensible.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _logistic_signal(n: int, cfg: dict[str, Any]) -> np.ndarray:
    a = float(cfg["signals"]["logistic_a"])
    burn = int(cfg["signals"]["burn_in"])
    x = 0.123456789
    out = np.empty(n, dtype=np.float64)
    for i in range(n + burn):
        x = 1.0 - a * x * x
        if i >= burn:
            out[i - burn] = x
    return out


def _cml_flat_signal(n: int, cfg: dict[str, Any]) -> np.ndarray:
    from dynachaos.cml.primitives import cml_step_logistic

    rng = np.random.default_rng(int(cfg["seed"]))
    sites = int(cfg["signals"]["cml_sites"])
    steps = int(np.ceil(n / sites)) + int(cfg["signals"]["burn_in"])
    x = rng.uniform(-0.5, 0.5, size=sites)
    rows = []
    for t in range(steps):
        x = cml_step_logistic(
            x,
            a=float(cfg["signals"]["cml_a"]),
            eps=float(cfg["signals"]["cml_eps"]),
            axis=0,
        )
        if t >= int(cfg["signals"]["burn_in"]):
            rows.append(x.copy())
    return np.asarray(rows, dtype=np.float64).ravel()[:n]


def _signal(name: str, n: int, cfg: dict[str, Any]) -> np.ndarray:
    if name == "logistic":
        return _logistic_signal(n, cfg)
    if name == "cml_flat":
        return _cml_flat_signal(n, cfg)
    raise ValueError(name)


def _embed(series: np.ndarray, dim: int, delay: int) -> np.ndarray:
    from dynachaos.diagnostics.recurrence import embed_time_delay

    return embed_time_delay(series, d=dim, tau=delay)


def _gp_once(case: dict[str, Any]) -> dict[str, Any]:
    from dynachaos.diagnostics import correlation as corr_mod

    cfg = case["config"]
    gp = cfg["gp"]
    series = _signal(case["signal"], int(case["N"]), cfg)
    embedded = _embed(series, int(gp["embedding_dim"]), int(gp["delay"]))
    old = bool(corr_mod._RUST_AVAILABLE)
    if case["backend"] == "rust" and not old:
        return {"skipped": True, "skip_reason": "rust extension unavailable"}
    corr_mod._RUST_AVAILABLE = case["backend"] == "rust" and old
    try:
        t0 = time.perf_counter()
        d2, radii, cvals, slopes, scaling = corr_mod.correlation_dimension(
            embedded,
            n_r=int(gp["radius_count"]),
            theiler_window=int(gp["theiler_window"]),
            norm=str(gp["norm"]),
        )
        wall = time.perf_counter() - t0
    finally:
        corr_mod._RUST_AVAILABLE = old
    return {
        "wall_time_s": wall,
        "d2": float(d2),
        "r_values": np.asarray(radii, dtype=float).tolist(),
        "C_values": np.asarray(cvals, dtype=float).tolist(),
        "local_slopes": np.asarray(slopes, dtype=float).tolist(),
        "scaling_mask": np.asarray(scaling, dtype=bool).tolist(),
    }


def _rqa_once(case: dict[str, Any]) -> dict[str, Any]:
    from dynachaos.diagnostics.recurrence import recurrence_matrix, rqa

    cfg = case["config"]
    rq = cfg["rqa"]
    series = _signal(case["signal"], int(case["N"]), cfg)
    t0 = time.perf_counter()
    R, eps = recurrence_matrix(
        series,
        metric=str(rq["metric"]),
        percentile=float(rq["percentile"]),
    )
    measures = rqa(R, l_min=int(rq["l_min"]), v_min=int(rq["v_min"]))
    return {"wall_time_s": time.perf_counter() - t0, "eps": float(eps), "rqa": measures}


def _worker() -> None:
    case = json.loads(os.environ[WORKER_ENV])
    repeats = int(case["config"].get("repeats", 3))
    runs = []
    for _ in range(repeats):
        runs.append(_gp_once(case) if case["kind"] == "gp" else _rqa_once(case))
    p50 = float(median([run["wall_time_s"] for run in runs]))
    representative = runs[len(runs) // 2]
    representative["wall_time_s_p50"] = p50
    representative["repeat_wall_time_s"] = [float(run["wall_time_s"]) for run in runs]
    representative["peak_rss_bytes"] = _vmhwm_bytes()
    print(json.dumps(representative, allow_nan=True))


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env[WORKER_ENV] = json.dumps(case)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _machine() -> dict[str, Any]:
    cpu = "unknown"
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    # Host total memory is deliberately not recorded: it says nothing about the
    # measurements below (every case reports its own peak RSS) and this report is
    # committed to a public repository.
    return {
        "platform": platform.platform(),
        "cpu_model": cpu,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def _parity(rust: dict[str, Any] | None, py: dict[str, Any] | None) -> dict[str, Any] | None:
    if rust is None or py is None:
        return None
    c_rs = np.asarray(rust["C_values"], dtype=float)
    c_py = np.asarray(py["C_values"], dtype=float)
    s_rs = np.asarray(rust["local_slopes"], dtype=float)
    s_py = np.asarray(py["local_slopes"], dtype=float)
    valid_c = (c_rs > 0) & (c_py > 0) & np.isfinite(c_rs) & np.isfinite(c_py)
    valid_s = np.isfinite(s_rs) & np.isfinite(s_py)
    return {
        "max_abs_delta_logC": float(np.max(np.abs(np.log(c_rs[valid_c]) - np.log(c_py[valid_c])))) if np.any(valid_c) else None,
        "max_abs_delta_slope": float(np.max(np.abs(s_rs[valid_s] - s_py[valid_s]))) if np.any(valid_s) else None,
    }


def _correlation_rust_available() -> bool:
    from dynachaos.diagnostics import correlation as corr_mod

    return bool(corr_mod._RUST_AVAILABLE)


def _bench_gp(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    mode = cfg["mode"]
    lengths = cfg["gp"][f"{mode}_lengths"]
    py_cap = int(cfg["gp"][f"python_{mode}_cap"])
    rust_available = _correlation_rust_available()
    rows = []
    for signal_name in ["logistic", "cml_flat"]:
        for n in lengths:
            by_backend: dict[str, dict[str, Any]] = {}
            for backend in ["rust", "python"]:
                if backend == "rust" and not rust_available:
                    rows.append({
                        "kind": "gp", "signal": signal_name, "N": int(n), "backend": backend,
                        "skipped": True, "skip_reason": "rust extension unavailable",
                    })
                    continue
                if backend == "python" and int(n) > py_cap:
                    rows.append({
                        "kind": "gp", "signal": signal_name, "N": int(n), "backend": backend,
                        "skipped": True, "skip_reason": f"Python fallback capped at N={py_cap} in {mode} mode because runtime grows as all-pairs Python loops.",
                    })
                    continue
                res = _run_case({"kind": "gp", "signal": signal_name, "N": int(n), "backend": backend, "config": cfg})
                row = {
                    "kind": "gp", "signal": signal_name, "N": int(n),
                    "embedding_dim": int(cfg["gp"]["embedding_dim"]),
                    "delay": int(cfg["gp"]["delay"]),
                    "radius_count": int(cfg["gp"]["radius_count"]),
                    "backend": backend, "skipped": False, **res,
                }
                rows.append(row)
                by_backend[backend] = row
            parity = _parity(by_backend.get("rust"), by_backend.get("python"))
            if parity is not None:
                for row in rows:
                    if row.get("kind") == "gp" and row.get("signal") == signal_name and row.get("N") == int(n) and not row.get("skipped"):
                        row["parity_vs_other_backend"] = parity
    return rows


def _bench_rqa(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    mode = cfg["mode"]
    stop = int(cfg["rqa"]["stop_predicted_bytes"])
    rows = []
    for signal_name in ["logistic", "cml_flat"]:
        for n in cfg["rqa"][f"{mode}_lengths"]:
            predicted = int(8 * int(n) * int(n))
            predicted_peak = int(RQA_TEMPORARIES_MULTIPLIER * predicted)
            base = {
                "kind": "dense_recurrence_rqa",
                "signal": signal_name,
                "N": int(n),
                "predicted_dense_distance_bytes": predicted,
                "predicted_peak_with_temporaries_bytes": predicted_peak,
                "dense_rqa_temporaries_multiplier": RQA_TEMPORARIES_MULTIPLIER,
            }
            if predicted_peak > stop:
                rows.append({**base, "skipped": True, "skip_reason": f"Predicted dense recurrence peak with temporaries exceeds configured safe bound {stop} bytes."})
                continue
            res = _run_case({"kind": "rqa", "signal": signal_name, "N": int(n), "config": cfg})
            rows.append({**base, "skipped": False, **res})
    threshold = int(np.floor(np.sqrt(stop / 8.0)))
    for row in rows:
        row["configured_impractical_threshold_N"] = threshold
        row["configured_impractical_threshold_bytes"] = stop
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_markdown(path: Path, results: dict[str, Any]) -> None:
    gp_rows = [r for r in results["gp"] if not r.get("skipped")]
    common = [r for r in gp_rows if r["backend"] == "rust" and any(p for p in gp_rows if p["backend"] == "python" and p["signal"] == r["signal"] and p["N"] == r["N"])]
    speed_line = "No common Rust/Python size measured."
    if common:
        r = max(common, key=lambda x: x["N"])
        p = next(x for x in gp_rows if x["backend"] == "python" and x["signal"] == r["signal"] and x["N"] == r["N"])
        speed_line = f"Largest common GP case: {r['signal']} N={r['N']}; Rust {r['wall_time_s_p50']:.6g} s vs Python {p['wall_time_s_p50']:.6g} s ({p['wall_time_s_p50']/r['wall_time_s_p50']:.2f}x)."
    thr = results["rqa"][0]["configured_impractical_threshold_N"] if results["rqa"] else None
    lines = [
        "# Scale-envelope benchmark (CI mode artifacts)", "",
        f"Generated: {results['generated_at_utc']}",
        f"Command: `{results['command']}`", "",
        "## Hardware and software", "",
        f"- CPU: {results['machine']['cpu_model']}",
        f"- Platform: {results['machine']['platform']}",
        f"- Python: {results['machine']['python']}; NumPy: {results['machine']['numpy']}",
        f"- Full-mode command: set `mode` to `full` in the JSONC config, then run `{results['command']}` on the target machine; this report header records the resulting hardware context.", "",
        "## Caveats", "",
        "- CI mode uses small sizes to check that the benchmark runs and that the backends agree. It is not a publication-scale timing claim.",
        "- Correlation-dimension parity is finite-data parity on identical synthetic inputs and radius grids.",
        f"- Python fallback GP cases are capped at N={results['benchmark_limits']['python_fallback_cap_N']} in this mode because the all-pairs fallback runtime grows rapidly; skipped full-mode rows say this explicitly.",
        f"- Dense recurrence estimates report the analytical 8*N^2 byte cost of the distance matrix; safety skips use a {RQA_TEMPORARIES_MULTIPLIER}x temporary-array multiplier for pdist, positive-distance, and boolean recurrence intermediates, while real peak RSS also includes interpreter overhead.",
        "- The CML-flat signal is a coupled-logistic lattice row sequence flattened to mimic larger-DOF simulation output while keeping a scalar diagnostic input.", "",
        "## Headline", "", speed_line,
        f"Dense recurrence/RQA configured impracticality threshold: N≈{thr} at the configured bound of {results['benchmark_limits']['dense_rqa_stop_predicted_bytes']} predicted 8*N^2 bytes.", "",
        "## Grassberger--Procaccia", "",
        "| signal | N | backend | p50 wall s | peak RSS MB | max delta logC | max delta slope |", "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in results["gp"]:
        if r.get("skipped"):
            lines.append(f"| {r['signal']} | {r['N']} | {r['backend']} | skipped | skipped |  |  |")
            continue
        par = r.get("parity_vs_other_backend") or {}
        lines.append(f"| {r['signal']} | {r['N']} | {r['backend']} | {r['wall_time_s_p50']:.6g} | {r['peak_rss_bytes']/1e6:.1f} | {par.get('max_abs_delta_logC', '')} | {par.get('max_abs_delta_slope', '')} |")
    lines += ["", "## Dense recurrence + RQA", "", "| signal | N | p50 wall s | peak RSS MB | predicted dense bytes | predicted peak with temporaries bytes | RR | DET |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results["rqa"]:
        if r.get("skipped"):
            lines.append(f"| {r['signal']} | {r['N']} | skipped | skipped | {r['predicted_dense_distance_bytes']} | {r['predicted_peak_with_temporaries_bytes']} |  |  |")
        else:
            lines.append(f"| {r['signal']} | {r['N']} | {r['wall_time_s_p50']:.6g} | {r['peak_rss_bytes']/1e6:.1f} | {r['predicted_dense_distance_bytes']} | {r['predicted_peak_with_temporaries_bytes']} | {r['rqa']['RR']:.6g} | {r['rqa']['DET']:.6g} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if WORKER_ENV in os.environ:
        _worker()
        return 0
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) > 1:
        raise SystemExit("usage: scale_envelope.py [config.jsonc]")
    cfg_path = Path(argv[0]) if argv else DEFAULT_CONFIG
    cfg = _load_config(cfg_path)
    results = {
        "schema_version": 1,
        "mode": cfg["mode"],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Recorded relative to the repository root: an absolute interpreter path
        # would pin this public artifact to one developer's filesystem.
        "command": f"python benchmarks/{Path(__file__).name} {cfg_path.name}",
        "config_path": str(cfg_path),
        "machine": _machine(),
        "signals": {
            "logistic": "Synthetic scalar logistic map x[n+1]=1-a*x[n]^2.",
            "cml_flat": "Coupled-logistic-map lattice rows flattened; representative larger-DOF simulation output reduced to a scalar stream.",
        },
        "benchmark_limits": {
            "python_fallback_cap_N": int(cfg["gp"][f"python_{cfg['mode']}_cap"]),
            "python_fallback_cap_reason": "all-pairs Python fallback runtime grows rapidly; Rust-only larger-N cases are used in full mode",
            "dense_rqa_stop_predicted_bytes": int(cfg["rqa"]["stop_predicted_bytes"]),
        },
        "gp": _bench_gp(cfg),
        "rqa": _bench_rqa(cfg),
    }
    json_path = ROOT / cfg["output"]["json"]
    md_path = ROOT / cfg["output"]["markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    results = _json_safe(results)
    json_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
