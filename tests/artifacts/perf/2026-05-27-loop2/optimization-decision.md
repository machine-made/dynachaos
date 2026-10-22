# Extreme optimization decision - Loop 2

## Input Profile

Source: `tests/artifacts/perf/2026-05-27-loop2/summary.md`

Measured top diagnostic hotspot:

- `recurrence_matrix()+rqa()` N=6000, p50 `0.251490s`, max `0.254493s`,
  peak RSS delta `457.7 MB`.
- Dense recurrence matrix construction remains the main memory pressure.

## Opportunity Matrix

| Candidate | Impact | Confidence | Effort | Score | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Reuse the full recurrence count inside `rqa()` | 2 | 5 | 1 | 10.0 | Implement |
| Replace dense recurrence matrix with sparse/streaming RQA | 5 | 2 | 5 | 2.0 | Defer: needs public-interface decision and separate prototype |
| Optimize entropy kernels | 1 | 5 | 3 | 1.7 | Reject: not a measured bottleneck |
| Rewrite figure simulation loops | 3 | 1 | 5 | 0.6 | Reject: cold-cache section profile missing |

## Implemented Change

`rqa()` previously scanned the full recurrence matrix three times with
`np.sum(R)`:

1. recurrence rate,
2. upper-triangle recurrent count for `DET`,
3. total recurrent count for `LAM`.

It now computes the total recurrent count once and reuses it. This does not
change the dense public interface, threshold semantics, line extraction, or
floating-point formulas.

## Isomorphism Proof

- Ordering preserved: yes; diagonal and vertical line extraction are unchanged.
- Tie-breaking unchanged: not applicable.
- Floating-point: `RR`, `DET`, and `LAM` use the same integer count value, just
  reused rather than recomputed.
- RNG seeds: not applicable.
- Golden outputs: benchmark script asserted exact dictionary equality between a
  local copy of the old `rqa()` and the new `rqa()` for N=2000, N=4000, and
  N=6000 recurrence matrices.

## Before/After Measurement

The benchmark compared a local copy of the old implementation against the new
implementation on the same dense recurrence matrices.

| Scenario | Old p50 | New p50 | Speedup |
| --- | ---: | ---: | ---: |
| `rqa()` N=2000 | `0.005844s` | `0.004540s` | `1.287x` |
| `rqa()` N=4000 | `0.025161s` | `0.017580s` | `1.431x` |
| `rqa()` N=6000 | `0.055270s` | `0.037607s` | `1.470x` |

The broader `recurrence_matrix()+rqa()` path will still be dominated by dense
distance/recurrence matrix construction for large N. This is a real but bounded
improvement inside the measured hotspot.

## Verification

```bash
uv run --extra viz pytest tests/test_diagnostics.py tests/test_rust_fallback.py -q
DYNACHAOS_NO_RUST=1 uv run --extra viz pytest tests/test_diagnostics.py -q
uv run --extra viz pytest tests/ -q
uv run --extra viz ruff check src/ tests/
uv run --extra viz ruff format src/ tests/ --check
```

Results:

- Recurrence/Rust-focused tests: `64 passed`.
- Python-only diagnostics tests: `36 passed`.
- Full tests: `223 passed`.
- Ruff check and format check passed.

## Remaining Optimization Boundary

Sparse or streaming RQA remains the only high-impact candidate. It should not
be implemented as an incidental optimization because it changes where the
public dense recurrence matrix is produced and would need exact metric parity
plus peak RSS proof in separate processes.
