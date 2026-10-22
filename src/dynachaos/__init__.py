"""Dynamical systems, chaos theory, and time series analysis."""

import importlib
import os

__version__ = "0.4.1"


def _ensure_rust_backend(env=os.environ, import_module=importlib.import_module):
    if env.get("DYNACHAOS_NO_RUST"):
        return False

    try:
        import_module("dynachaos._rust")
    except ImportError as exc:
        raise RuntimeError(
            "dynachaos requires the Rust extension by default.\n"
            "Build it with: uv run maturin develop --release\n"
            "To opt in to pure-Python fallbacks, set DYNACHAOS_NO_RUST=1."
        ) from exc

    return True


_RUST_BACKEND_AVAILABLE = _ensure_rust_backend()

__all__ = [
    "maps",
    "cml",
    "diagnostics",
    "utils",
    "io",
    "pipelines",
    "viz",
]
