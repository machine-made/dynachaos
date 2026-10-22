"""Runtime profile helpers (paper vs smoke)."""

from __future__ import annotations

import os


def current_profile() -> str:
    """Return active runtime profile (defaults to 'paper')."""
    profile = os.environ.get("DYNACHAOS_PROFILE", "paper").strip().lower()
    if profile in {"paper", "smoke"}:
        return profile
    return "paper"


def is_smoke() -> bool:
    """Whether smoke profile is active."""
    return current_profile() == "smoke"


def choose[T](paper: T, smoke: T) -> T:
    """Pick a value according to the active runtime profile."""
    return smoke if is_smoke() else paper
