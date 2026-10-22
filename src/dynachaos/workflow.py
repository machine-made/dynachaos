"""Config-driven analysis workflow for scalar/reduced time signals."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from dynachaos.config import load_jsonc
from dynachaos.diagnostics.correlation import correlation_dimension
from dynachaos.diagnostics.permutation import permutation_entropy
from dynachaos.diagnostics.recurrence import (
    embed_time_delay,
    recurrence_matrix,
    rqa,
    rqa_from_trajectory,
)
from dynachaos.diagnostics.reliability import ReliabilityRecord
from dynachaos.maps import logistic
from dynachaos.utils.system import get_rss_mb

DENSE_RQA_DEFAULT_MAX_BYTES = 4 * 1024**3
DENSE_RQA_SCALE_EVIDENCE = "benchmarks/results/scale_envelope.md"


class WorkflowError(RuntimeError):
    """User-facing workflow error with no traceback required."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkflowError(f"{name} must be an object")
    return value


def _load_signal_from_file(path: Path, *, npz_key: str | None) -> np.ndarray:
    if not path.exists():
        raise WorkflowError(f"input file does not exist: {path}")
    try:
        data = np.load(path)
    except OSError as exc:
        raise WorkflowError(f"could not read input file {path}: {exc}") from exc

    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            keys = list(data.files)
            if npz_key is None:
                if len(keys) != 1:
                    raise WorkflowError(
                        f"npz input has multiple arrays {keys}; set input.npz_key in the config"
                    )
                npz_key = keys[0]
            if npz_key not in data:
                raise WorkflowError(f"npz input does not contain array '{npz_key}'")
            return np.asarray(data[npz_key], dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def _generated_signal(cfg: dict[str, Any]) -> np.ndarray:
    name = str(cfg.get("name", ""))
    if name != "logistic":
        raise WorkflowError(f"unsupported generated benchmark signal '{name}'")
    n = int(cfg.get("n", 0))
    if n < 1:
        raise WorkflowError("generated.n must be a positive integer")
    a = float(cfg.get("a", 1.99))
    x = np.empty(n, dtype=np.float64)
    if "x0" in cfg:
        x[0] = float(cfg["x0"])
    else:
        seed = int(cfg.get("seed", 0))
        x[0] = float(np.random.default_rng(seed).uniform(0.05, 0.95))
    for i in range(1, n):
        x[i] = logistic(x[i - 1], a)
    return x


def _load_signal(cfg: dict[str, Any], config_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    input_cfg = _require_mapping(cfg.get("input", {}), "input")
    if "generated" in input_cfg and "path" in input_cfg:
        raise WorkflowError("input must set either path or generated, not both")
    if "generated" in input_cfg:
        generated = _require_mapping(input_cfg["generated"], "input.generated")
        signal = _generated_signal(generated)
        source = {"kind": "generated", **generated}
    elif "path" in input_cfg:
        path = Path(str(input_cfg["path"]))
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        signal = _load_signal_from_file(path, npz_key=input_cfg.get("npz_key"))
        source = {"kind": "file", "path": str(path), "npz_key": input_cfg.get("npz_key")}
    else:
        raise WorkflowError("input must set either path or generated")

    if signal.ndim != 1:
        raise WorkflowError(f"input signal must be 1D; got shape {tuple(signal.shape)}")
    if signal.size == 0:
        raise WorkflowError("input signal must be non-empty")
    if not np.all(np.isfinite(signal)):
        raise WorkflowError("input signal must contain only finite values")
    return signal.astype(np.float64, copy=False), source


def _embedded(signal: np.ndarray, diag_cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    emb_cfg = _require_mapping(diag_cfg.get("embedding", {}), "diagnostic.embedding")
    d = int(emb_cfg.get("d", 1))
    tau = int(emb_cfg.get("tau", 1))
    if d == 1:
        return signal[:, np.newaxis], {"d": d, "tau": tau}
    return embed_time_delay(signal, d=d, tau=tau), {"d": d, "tau": tau}


def _dense_rqa_limit(cfg: dict[str, Any]) -> tuple[int, bool]:
    limits = _require_mapping(cfg.get("scale_limits", {}), "scale_limits")
    max_bytes = int(limits.get("dense_rqa_max_bytes", DENSE_RQA_DEFAULT_MAX_BYTES))
    allow = limits.get("allow_dense_rqa_beyond_envelope", False)
    if not isinstance(allow, bool):
        raise WorkflowError(
            "scale_limits.allow_dense_rqa_beyond_envelope must be the JSON boolean true or false"
        )
    return max_bytes, allow


def _run_diagnostic(
    signal: np.ndarray, diag_cfg: dict[str, Any], workflow_cfg: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    name = str(diag_cfg.get("name", ""))
    if not name:
        raise WorkflowError("diagnostic entry is missing name")

    t0 = time.perf_counter()
    if name == "permutation_entropy":
        d = int(diag_cfg.get("d", 5))
        tau = int(diag_cfg.get("tau", 1))
        normalise = bool(diag_cfg.get("normalise", True))
        value = permutation_entropy(signal, d=d, tau=tau, normalise=normalise)
        record = ReliabilityRecord(
            method_name=name,
            backend="rust/python ordinal distribution",
            parameters={"d": d, "tau": tau, "normalise": normalise},
            data_length=int(signal.size),
            data_shape=tuple(int(v) for v in signal.shape),
            sampling_downsampling_note="no sampling/downsampling; scalar signal used directly",
        )
        result = {"value": value}
    elif name == "correlation_dimension":
        traj, embedding = _embedded(signal, diag_cfg)
        return_stderr = bool(diag_cfg.get("return_stderr", True))
        output = correlation_dimension(
            traj,
            n_r=int(diag_cfg.get("n_r", 50)),
            r_range=diag_cfg.get("r_range"),
            max_pairs=int(diag_cfg.get("max_pairs", 500_000)),
            theiler_window=int(diag_cfg.get("theiler_window", 0)),
            norm=str(diag_cfg.get("norm", "chebyshev")),
            return_stderr=return_stderr,
            return_metadata=True,
        )
        if return_stderr:
            d2, radii, corr, stderr, slopes, mask, record = output
            result = {"D2": d2, "D2_stderr": stderr}
        else:
            d2, radii, corr, slopes, mask, record = output
            result = {"D2": d2}
        result.update(
            {
                "r_values": radii,
                "C_values": corr,
                "local_slopes": slopes,
                "scaling_mask": mask,
                "embedding": embedding,
            }
        )
    elif name in {"rqa_streaming", "rqa_dense"}:
        traj, embedding = _embedded(signal, diag_cfg)
        eps = diag_cfg.get("eps")
        eps = None if eps is None else float(eps)
        metric = str(diag_cfg.get("metric", "euclidean"))
        percentile = float(diag_cfg.get("percentile", 5))
        l_min = int(diag_cfg.get("l_min", 2))
        v_min = int(diag_cfg.get("v_min", 2))
        if name == "rqa_dense":
            predicted_bytes = int(8 * traj.shape[0] * traj.shape[0])
            max_bytes, allow = _dense_rqa_limit(workflow_cfg)
            if predicted_bytes > max_bytes and not allow:
                raise WorkflowError(
                    "dense RQA request exceeds the configured scale envelope: "
                    f"8*N^2={predicted_bytes} bytes > {max_bytes} bytes for N={traj.shape[0]}; "
                    "use rqa_streaming_from_trajectory/rqa_streaming or set "
                    "scale_limits.allow_dense_rqa_beyond_envelope=true explicitly in the config"
                )
            matrix, eps_used = recurrence_matrix(
                traj, eps=eps, metric=metric, percentile=percentile
            )
            stats = rqa(matrix, l_min=l_min, v_min=v_min)
            record = ReliabilityRecord(
                method_name="rqa_dense",
                backend="dense recurrence_matrix + rqa",
                parameters={
                    "eps": eps_used,
                    "metric": metric,
                    "percentile": percentile,
                    "l_min": l_min,
                    "v_min": v_min,
                },
                data_length=int(traj.shape[0]),
                data_shape=tuple(int(v) for v in traj.shape),
                sampling_downsampling_note=(
                    "no sampling/downsampling; dense recurrence matrix materialized"
                ),
                validity_warnings=[],
                unresolved_verdicts=[],
                scale_evidence={
                    "artifact_path": DENSE_RQA_SCALE_EVIDENCE,
                    "dense_bytes_formula": "8*N^2",
                    "predicted_dense_bytes": predicted_bytes,
                },
            )
        else:
            stats, record = rqa_from_trajectory(
                traj,
                eps=eps,
                metric=metric,
                percentile=percentile,
                l_min=l_min,
                v_min=v_min,
                return_metadata=True,
            )
        result = {"stats": stats, "embedding": embedding}
    else:
        raise WorkflowError(f"unsupported diagnostic name '{name}'")

    elapsed = time.perf_counter() - t0
    cost = {"wall_time_seconds": elapsed, "peak_rss_mb": get_rss_mb()}
    return name, _json_safe(result), {"cost": _json_safe(cost), "reliability": record.to_dict()}


def _write_summary(
    path: Path,
    *,
    config_path: Path,
    source: dict[str, Any],
    signal: np.ndarray,
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# dynachaos analysis workflow summary",
        "",
        f"- Config: `{config_path}`",
        f"- Input source: `{source}`",
        f"- Signal length N: {signal.size}",
        "- Results JSON: `results.json`",
        "- Metadata JSON: `metadata.json`",
        "",
        "## Diagnostics",
    ]
    for name, result in diagnostics.items():
        lines.append(f"- `{name}`: {json.dumps(_json_safe(result), sort_keys=True)[:500]}")
    lines.extend(
        [
            "",
            "## Scale/cost metadata",
            "",
            "```json",
            json.dumps(metadata["scale_cost"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_workflow(config_path: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise WorkflowError(f"config file does not exist: {config_path}")
    cfg = _require_mapping(load_jsonc(config_path), "config")
    signal, source = _load_signal(cfg, config_path)

    out_cfg = _require_mapping(cfg.get("output", {}), "output")
    output_dir_value = out_cfg.get("dir", cfg.get("output_dir"))
    if output_dir_value is None:
        raise WorkflowError("output.dir (or output_dir) must be set in the config")
    output_dir = Path(str(output_dir_value))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_cfg = cfg.get("diagnostics")
    if not isinstance(diagnostics_cfg, list) or not diagnostics_cfg:
        raise WorkflowError("diagnostics must be a non-empty list")

    workflow_t0 = time.perf_counter()
    results: dict[str, Any] = {}
    reliability: dict[str, Any] = {}
    costs: dict[str, Any] = {}
    for item in diagnostics_cfg:
        diag_cfg = _require_mapping(item, "diagnostics[]")
        name, result, meta = _run_diagnostic(signal, diag_cfg, cfg)
        if name in results:
            raise WorkflowError(f"duplicate diagnostic name '{name}'")
        results[name] = result
        reliability[name] = meta["reliability"]
        costs[name] = meta["cost"]

    scale_cost = {
        "signal_length_N": int(signal.size),
        "signal_shape": tuple(int(v) for v in signal.shape),
        "workflow_wall_time_seconds": time.perf_counter() - workflow_t0,
        "peak_rss_mb": get_rss_mb(),
        "diagnostics": costs,
    }
    metadata = {
        "config_path": str(config_path),
        "input": source,
        "scale_cost": _json_safe(scale_cost),
        "reliability": reliability,
    }
    results_payload = {"schema_version": "1.0", "diagnostics": results}

    results_path = output_dir / "results.json"
    metadata_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.md"
    results_path.write_text(
        json.dumps(_json_safe(results_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(
        summary_path,
        config_path=config_path,
        source=source,
        signal=signal,
        diagnostics=results,
        metadata=metadata,
    )
    return {
        "output_dir": output_dir,
        "results": results_path,
        "metadata": metadata_path,
        "summary": summary_path,
    }
