# Extreme Optimization Decision Loop 1

## Input Profile

Source: `tests/artifacts/perf/2026-05-27-loop1/summary.md`

Measured top candidate:

- `recurrence_matrix()+rqa()` N=4000, p50 `0.108933s`, p95 `0.109216s`.
- Scaling from N=2000 to N=4000 was approximately quadratic.

## Opportunity Matrix

| Candidate | Impact | Confidence | Effort | Score | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Replace dense recurrence matrix with sparse/streaming RQA | 4 | 2 | 5 | 1.6 | Reject for Loop 1 |
| Hand-optimize `rqa()` sums and trace reuse | 1 | 4 | 1 | 4.0 | Reject: not measured as bottleneck |
| Replace SciPy `pdist`/`squareform` in `recurrence_matrix()` | 2 | 2 | 4 | 1.0 | Reject for Loop 1 |

## Isomorphism Assessment

No optimization was applied.

Dense recurrence matrices are currently part of the public interface:
`recurrence_matrix()` returns the full boolean matrix and `rqa()` consumes that
matrix. A sparse or streaming RQA path could reduce memory at large N, but it
would introduce a new behavior contract:

- exact agreement for `RR`, `DET`, `LAM`, `L`, `TT`, `ENTR`, and `Lmax`;
- matching diagonal and vertical line extraction for all `l_min`/`v_min`;
- same threshold semantics for `eps` and percentile-selected thresholds;
- a clear decision about whether callers still receive the dense recurrence
  matrix.

The measured p95 at N=4000 is only about `0.11s`, so a semantic rewrite does not
clear the Loop 1 risk bar.

## Hand-Off

Loop 2 can revisit this if a larger representative workload shows recurrence
matrix memory or runtime dominating. Required evidence before implementation:

1. peak RSS for N >= 10000;
2. profile separating `pdist`, `squareform`, diagonal extraction, and vertical
   extraction;
3. parity tests comparing dense and proposed sparse/streaming RQA outputs;
4. a public-interface decision for whether `recurrence_matrix()` remains dense.
