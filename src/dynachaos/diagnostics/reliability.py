"""Serializable reliability metadata for diagnostic outputs.

``ReliabilityRecord`` keeps measured facts (method name, backend, parameters,
data length/shape, sampling/downsampling note, and scale-evidence pointers)
separate from interpretation caveats (validity warnings and unresolved
verdicts).  Publication-facing reports can therefore cite how a result was produced
without mixing measured computation settings with finite-data caveats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe representation with no NaN/Inf values."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        # np.longdouble.item() returns another NumPy scalar; force a builtin.
        if isinstance(item, np.generic):
            item = float(item)
        return _json_safe(item)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    return str(value)


@dataclass(frozen=True)
class ReliabilityRecord:
    """Small JSON-serializable reliability record attached to diagnostics."""

    method_name: str
    backend: str
    parameters: dict[str, Any]
    data_length: int | None
    data_shape: tuple[int, ...]
    sampling_downsampling_note: str
    validity_warnings: list[str] = field(default_factory=list)
    unresolved_verdicts: list[str] = field(default_factory=list)
    scale_evidence: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary (finite numbers, real booleans, no NaN)."""
        return _json_safe(
            {
                "method_name": self.method_name,
                "backend": self.backend,
                "parameters": self.parameters,
                "data_length": self.data_length,
                "data_shape": self.data_shape,
                "sampling_downsampling_note": self.sampling_downsampling_note,
                "validity_warnings": self.validity_warnings,
                "unresolved_verdicts": self.unresolved_verdicts,
                "scale_evidence": self.scale_evidence,
                "schema_version": self.schema_version,
            }
        )

    def to_json(self) -> str:
        """Serialize the record as stable JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
