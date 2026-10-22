import json

import numpy as np
import pytest

from dynachaos.diagnostics import ReliabilityRecord
from dynachaos.diagnostics import correlation as corr_mod
from dynachaos.diagnostics import recurrence as rec_mod
from dynachaos.diagnostics.correlation import correlation_dimension
from dynachaos.diagnostics.recurrence import rqa_from_trajectory

REQUIRED_FIELDS = {
    "method_name",
    "backend",
    "parameters",
    "data_length",
    "data_shape",
    "sampling_downsampling_note",
    "validity_warnings",
    "unresolved_verdicts",
    "scale_evidence",
    "schema_version",
}


def _circle(n=80):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


def test_reliability_record_json_round_trips_without_nan():
    record = ReliabilityRecord(
        method_name="example",
        backend="n.a.",
        parameters={"threshold": np.nan, "enabled": np.bool_(True)},
        data_length=3,
        data_shape=(3, 1),
        sampling_downsampling_note="no sampling/downsampling",
        validity_warnings=["finite-data caveat"],
        unresolved_verdicts=["scaling region not identified"],
        scale_evidence={"artifact_path": "benchmarks/results/scale_envelope.json", "entry": None},
    )

    payload = record.to_dict()
    assert set(payload) == REQUIRED_FIELDS
    assert payload["parameters"]["threshold"] is None
    assert payload["parameters"]["enabled"] is True
    assert "NaN" not in record.to_json()
    assert json.loads(record.to_json()) == payload


def test_correlation_dimension_metadata_emission_and_default_shape():
    traj = _circle()

    default = correlation_dimension(traj, n_r=12)
    with_metadata = correlation_dimension(traj, n_r=12, return_metadata=True)

    assert len(default) == 5
    assert len(with_metadata) == 6
    for old_value, new_value in zip(default, with_metadata[:-1]):
        if isinstance(old_value, np.ndarray):
            np.testing.assert_allclose(old_value, new_value, equal_nan=True)
        else:
            assert new_value == pytest.approx(old_value, nan_ok=True)

    metadata = with_metadata[-1]
    payload = metadata.to_dict()
    assert set(payload) == REQUIRED_FIELDS
    assert payload["method_name"] == "correlation_dimension"
    assert payload["backend"] in {"rust", "python"}
    assert payload["parameters"]["n_r"] == 12
    assert payload["data_length"] == len(traj)
    assert payload["data_shape"] == [len(traj), 2]
    assert isinstance(payload["validity_warnings"], list)
    assert isinstance(payload["unresolved_verdicts"], list)
    assert payload["scale_evidence"]["artifact_path"] == "benchmarks/results/scale_envelope.json"
    assert json.loads(metadata.to_json()) == payload


def test_rqa_from_trajectory_metadata_emission_and_default_shape():
    traj = _circle(36)

    default = rqa_from_trajectory(traj, percentile=10, metric="chebyshev", l_min=2, v_min=2)
    stats, metadata = rqa_from_trajectory(
        traj, percentile=10, metric="chebyshev", l_min=2, v_min=2, return_metadata=True
    )

    assert stats == default
    assert set(default) == {"RR", "DET", "LAM", "L", "TT", "ENTR", "Lmax"}

    payload = metadata.to_dict()
    assert set(payload) == REQUIRED_FIELDS
    assert payload["method_name"] == "rqa_from_trajectory"
    assert payload["backend"] in {"rust", "python"}
    assert payload["parameters"]["metric"] == "chebyshev"
    assert payload["data_length"] == len(traj)
    assert payload["data_shape"] == [len(traj), 2]
    assert isinstance(payload["validity_warnings"], list)
    assert isinstance(payload["unresolved_verdicts"], list)
    assert json.loads(metadata.to_json()) == payload


def test_backend_metadata_reports_python_when_fallback_flag_is_active(monkeypatch):
    traj = _circle(30)

    monkeypatch.setattr(corr_mod, "_RUST_AVAILABLE", False)
    corr_record = corr_mod.correlation_dimension(traj, n_r=10, return_metadata=True)[-1]
    assert corr_record.backend == "python"

    monkeypatch.setattr(rec_mod, "_RUST_AVAILABLE", False)
    stats, rqa_record = rec_mod.rqa_from_trajectory(traj, percentile=10, return_metadata=True)
    assert stats.keys() == {"RR", "DET", "LAM", "L", "TT", "ENTR", "Lmax"}
    assert rqa_record.backend == "python"


def test_backend_metadata_reports_rust_when_rust_scanners_are_used(monkeypatch):
    if corr_mod._correlation_counts_rs is None or rec_mod._count_line_lengths_rs is None:
        pytest.skip("Rust extension not importable in this environment")
    traj = _circle(30)

    monkeypatch.setattr(corr_mod, "_RUST_AVAILABLE", True)
    corr_record = corr_mod.correlation_dimension(traj, n_r=10, return_metadata=True)[-1]
    assert corr_record.backend == "rust"

    monkeypatch.setattr(rec_mod, "_RUST_AVAILABLE", True)
    _, rqa_record = rec_mod.rqa_from_trajectory(traj, percentile=10, return_metadata=True)
    assert rqa_record.backend == "rust"


# ── Coverage: _json_safe edge cases ────────────────────────────────────────────


def test_json_safe_ndarray_conversion():
    """_json_safe converts ndarray to list (line 25)."""
    from dynachaos.diagnostics.reliability import _json_safe

    arr = np.array([1.0, 2.0, 3.0])
    result = _json_safe(arr)
    assert isinstance(result, list)
    assert result == [1.0, 2.0, 3.0]


def test_json_safe_numpy_scalar_to_builtin_float():
    """_json_safe converts np.generic to builtin float (line 30)."""
    from dynachaos.diagnostics.reliability import _json_safe

    scalar = np.float64(3.14)
    result = _json_safe(scalar)
    assert isinstance(result, float)
    assert result == 3.14


def test_json_safe_fallback_str_conversion():
    """_json_safe falls back to str() for unmatched types (line 42)."""
    from dynachaos.diagnostics.reliability import _json_safe

    class CustomType:
        def __str__(self):
            return "custom_value"

    result = _json_safe(CustomType())
    assert result == "custom_value"
