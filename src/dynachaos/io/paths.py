"""Path resolution and I/O helpers for generated figure/data artifacts."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np


def output_root() -> Path:
    """Return the base output directory for generated artifacts.

    Precedence:
    1. ``DYNACHAOS_OUTPUT_ROOT`` environment variable
    2. ``<cwd>/figures``
    """
    env_root = os.environ.get("DYNACHAOS_OUTPUT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.cwd() / "figures").resolve()


def section_dir(section_name: str) -> Path:
    """Return the output directory for a paper section."""
    return output_root() / section_name


def safe_load(path):
    """Load .npz without deserializing arbitrary objects."""
    return np.load(path, allow_pickle=False)


def load_or_compute_npz(
    npz_path: str | Path,
    section_name: str,
    compute_fn: Callable[[], Any],
    *,
    required_keys: Iterable[str] = (),
):
    """Load a cache file, recomputing it when missing or schema-stale."""
    path = Path(npz_path)
    keys = tuple(required_keys)
    try:
        data = safe_load(path)
        missing = tuple(key for key in keys if key not in data.files)
        if missing:
            data.close()
            missing_text = ", ".join(missing)
            print(f"Cache {path} missing keys ({missing_text}); recomputing {section_name}...")
            compute_fn()
            data = safe_load(path)
        else:
            print(f"Loaded {path}")
    except FileNotFoundError:
        print(f"Computing {section_name}...")
        compute_fn()
        data = safe_load(path)

    missing = tuple(key for key in keys if key not in data.files)
    if missing:
        data.close()
        missing_text = ", ".join(missing)
        raise KeyError(f"Cache {path} is missing required keys after compute: {missing_text}")
    return data


def write_payload(output_path, payload, *, base_dir: Path) -> dict:
    """Write a compute payload when requested and return it unchanged.

    Parameters
    ----------
    output_path : str, Path, or None
        Destination path.  A bare string is resolved relative to *base_dir*.
        ``None`` skips the write and returns *payload* immediately.
    payload : dict
        Mapping of array name to array (passed as ``**payload`` to
        :func:`numpy.savez_compressed`).
    base_dir : Path
        Module-level figure directory; used only when *output_path* is a str.
    """
    if output_path is None:
        return payload
    output_path = base_dir / output_path if isinstance(output_path, str) else output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"Saved {output_path}")
    return payload


def load_or_compute_payload(
    npz_path,
    section_name: str,
    compute_fn,
    *,
    required_keys=(),
) -> dict:
    """Load a cache .npz, or call compute_fn() and return its payload dict.

    Unlike :func:`load_or_compute_npz`, this returns the raw ``dict`` from
    *compute_fn* on a cache miss (instead of a
    :class:`numpy.lib.npyio.NpzFile`).  On a cache hit it returns the open
    :class:`~numpy.lib.npyio.NpzFile`.

    Parameters
    ----------
    npz_path : str or Path
    section_name : str
        Human-readable label used in progress messages.
    compute_fn : callable
        Called with no arguments; must return a dict whose keys include
        *required_keys*.
    required_keys : iterable of str
        If any key is absent from a loaded cache, the cache is discarded
        and *compute_fn* is called.
    """
    path = npz_path
    keys = tuple(required_keys)
    try:
        data = safe_load(path)
        missing = tuple(key for key in keys if key not in data.files)
        if not missing:
            print(f"Loaded {path}")
            return data
        data.close()
        missing_text = ", ".join(missing)
        print(f"Cache {path} missing keys ({missing_text}); recomputing {section_name}...")
    except FileNotFoundError:
        print(f"Computing {section_name}...")

    result = compute_fn()
    missing = tuple(key for key in keys if key not in result)
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"Cache {path} is missing required keys after compute: {missing_text}")
    return result
