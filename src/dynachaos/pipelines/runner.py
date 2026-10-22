"""Pipeline runner for paper section generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dynachaos.pipelines.registry import get_section, list_sections
from dynachaos.utils.system import get_rss_mb

_ALLOWED_OUTPUT_SUFFIXES = {".npz", ".png"}


@dataclass(frozen=True)
class ArtifactInspection:
    """Inspection result for one expected pipeline artifact."""

    role: str
    path: Path
    status: str
    detail: str
    required_keys: tuple[str, ...] = ()


def _repo_src_dir() -> Path | None:
    """Return local repo src/ path when running from source checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "dynachaos"
        if candidate.is_dir():
            return candidate.parent
    return None


def _runner_env(output_root: Path, profile: str) -> dict[str, str]:
    env = os.environ.copy()
    env["DYNACHAOS_OUTPUT_ROOT"] = str(output_root)
    env["DYNACHAOS_PROFILE"] = profile

    src_dir = _repo_src_dir()
    if src_dir is not None:
        current = env.get("PYTHONPATH")
        if current:
            env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{current}"
        else:
            env["PYTHONPATH"] = str(src_dir)

    return env


def _rss_raw_to_mb(rss_raw: int) -> float:
    """Convert ru_maxrss to megabytes using the same platform convention as get_rss_mb."""
    # macOS: ru_maxrss is in bytes; Linux: in KiB
    return rss_raw / (1024 * 1024) if sys.platform == "darwin" else rss_raw / 1024


def _run_module(module_name: str, output_root: Path, profile: str) -> float | None:
    """Run a module subprocess and return the child's peak RSS in MB (POSIX), or None (Windows)."""
    env = _runner_env(output_root, profile)
    cmd = [sys.executable, "-m", module_name]
    if hasattr(os, "wait4"):
        proc = subprocess.Popen(cmd, env=env)
        _pid, exit_status, rusage = os.wait4(proc.pid, 0)
        if hasattr(os, "waitstatus_to_exitcode"):
            returncode = os.waitstatus_to_exitcode(exit_status)
        elif os.WIFEXITED(exit_status):
            returncode = os.WEXITSTATUS(exit_status)
        else:
            returncode = -os.WTERMSIG(exit_status)
        proc.returncode = returncode
        if returncode != 0:
            raise RuntimeError(f"Module run failed: {' '.join(cmd)} (exit {returncode})")
        return _rss_raw_to_mb(rusage.ru_maxrss)
    else:
        # Windows or systems without wait4 — fall back; caller uses get_rss_mb() instead
        proc = subprocess.Popen(cmd, env=env)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Module run failed: {' '.join(cmd)} (exit {proc.returncode})")
        return None


def _timing_ledger_path(timing_ledger: str | Path | None) -> Path | None:
    if timing_ledger is not None:
        return Path(timing_ledger)
    env_path = os.environ.get("DYNACHAOS_TIMING_LEDGER")
    return Path(env_path) if env_path else None


def _append_timing_event(
    ledger_path: Path,
    *,
    section_id: str,
    module_name: str,
    profile: str,
    cache_state: str,
    wall_time_s: float,
    peak_rss_mb: float,
) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "section_id": section_id,
        "module": module_name,
        "profile": profile,
        "cache_state": cache_state,
        "wall_time_s": round(wall_time_s, 9),
        "peak_rss_mb": round(peak_rss_mb, 3),
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _classify_artifact(path: Path, *, required_keys: tuple[str, ...]) -> tuple[str, str]:
    if not path.exists():
        return "missing", "missing expected artifact"

    suffix = path.suffix.lower()
    if suffix not in _ALLOWED_OUTPUT_SUFFIXES:
        return "unsupported", f"unsupported extension: {path.suffix}"

    if suffix != ".npz":
        if path.stat().st_size == 0:
            return "empty", "empty artifact"
        return "ok", "present"

    try:
        with np.load(path, allow_pickle=False) as data:
            missing = tuple(key for key in required_keys if key not in data.files)
    except Exception as exc:
        return "malformed", f"malformed NPZ artifact: {exc}"

    if missing:
        return "missing_keys", "missing required NPZ keys: " + ", ".join(missing)

    return "ok", "present"


def _validate_artifact(path: Path, *, required_keys: tuple[str, ...], section_id: str) -> None:
    status, detail = _classify_artifact(path, required_keys=required_keys)
    if status == "ok":
        return
    if status == "missing":
        raise RuntimeError(f"Section {section_id} is missing expected artifact: {path}")
    if status == "unsupported":
        raise RuntimeError(
            f"Section {section_id} artifact {path} has unsupported extension: {path.suffix}"
        )
    if status == "empty":
        raise RuntimeError(f"Section {section_id} artifact {path} is empty")
    if status == "malformed":
        raise RuntimeError(f"Section {section_id} has malformed NPZ artifact: {path}")
    if status == "missing_keys":
        missing_text = detail[len("missing required NPZ keys: ") :]
        raise RuntimeError(
            f"Section {section_id} artifact {path} is missing required NPZ keys: {missing_text}"
        )
    raise AssertionError(f"unknown artifact status: {status}")


def _inspect_artifact(path: Path, *, required_keys: tuple[str, ...]) -> tuple[str, str]:
    return _classify_artifact(path, required_keys=required_keys)


def _section_root(output_root: str | Path | None) -> Path:
    return (
        Path(output_root).resolve()
        if output_root is not None
        else (Path.cwd() / "figures").resolve()
    )


def validate_section_cache(
    section_id: str,
    *,
    output_root: str | Path | None = None,
) -> list[Path]:
    """Validate precomputed cache artifacts for one section without running modules."""
    spec = get_section(section_id)
    root = _section_root(output_root)
    cache_paths = [root / section_id / name for name in spec.cache_files]
    for path in cache_paths:
        _validate_artifact(
            path,
            required_keys=spec.required_npz_keys(path.name),
            section_id=section_id,
        )
    return cache_paths


def validate_section_outputs(
    section_id: str,
    *,
    output_root: str | Path | None = None,
) -> list[Path]:
    """Validate expected output artifacts for one section without running modules."""
    spec = get_section(section_id)
    root = _section_root(output_root)
    output_paths = [root / section_id / name for name in spec.output_files]
    for path in output_paths:
        _validate_artifact(
            path,
            required_keys=spec.required_npz_keys(path.name),
            section_id=section_id,
        )
    return output_paths


def inspect_section_artifacts(
    section_id: str,
    *,
    output_root: str | Path | None = None,
) -> list[ArtifactInspection]:
    """Inspect expected section artifacts without running modules."""
    spec = get_section(section_id)
    root = _section_root(output_root)
    results: list[ArtifactInspection] = []

    for role, names in (("cache", spec.cache_files), ("output", spec.output_files)):
        for name in names:
            path = root / section_id / name
            required_keys = spec.required_npz_keys(path.name)
            status, detail = _inspect_artifact(path, required_keys=required_keys)
            results.append(
                ArtifactInspection(
                    role=role,
                    path=path,
                    status=status,
                    detail=detail,
                    required_keys=required_keys,
                )
            )

    return results


def run_section(
    section_id: str,
    *,
    output_root: str | Path | None = None,
    profile: str = "paper",
    recompute: bool = False,
    timing_ledger: str | Path | None = None,
) -> list[Path]:
    """Run one section pipeline and return expected output paths."""
    spec = get_section(section_id)
    root = _section_root(output_root)

    if profile not in {"paper", "smoke"}:
        raise ValueError("profile must be one of: paper, smoke")

    if recompute:
        for name in spec.output_files:
            path = root / section_id / name
            if path.exists():
                path.unlink()

    cache_state = "not_checked"
    if profile == "smoke":
        validate_section_cache(section_id, output_root=root)
        cache_state = "validated"

    ledger_path = _timing_ledger_path(timing_ledger)
    for module_name in spec.modules:
        started = time.perf_counter() if ledger_path is not None else 0.0
        child_rss = _run_module(module_name, root, profile)
        if ledger_path is not None:
            if child_rss is None:
                child_rss = get_rss_mb()
            _append_timing_event(
                ledger_path,
                section_id=section_id,
                module_name=module_name,
                profile=profile,
                cache_state=cache_state,
                wall_time_s=time.perf_counter() - started,
                peak_rss_mb=child_rss,
            )

    return validate_section_outputs(section_id, output_root=root)


def run_all(
    *,
    output_root: str | Path | None = None,
    profile: str = "paper",
    recompute: bool = False,
    timing_ledger: str | Path | None = None,
) -> dict[str, list[Path]]:
    """Run all section pipelines in paper order."""
    results: dict[str, list[Path]] = {}
    for section_id in list_sections():
        results[section_id] = run_section(
            section_id,
            output_root=output_root,
            profile=profile,
            recompute=recompute,
            timing_ledger=timing_ledger,
        )
    return results
