# RQA scaling design note

## Problem and scale evidence

Dense `recurrence_matrix(X)` constructs an `N x N` distance matrix before the
boolean recurrence matrix. The checked-in benchmark reports an analytical
`8*N^2` byte distance-matrix cost and uses a 3x temporary multiplier for
pdist, positive-distance, and recurrence intermediates. That gives a configured
an impracticality threshold near `N=23170` under the default cap, before Python/interpreter
overhead. CI-mode measurements already show dense recurrence/RQA peak RSS rising
from 107.5 MB at `N=100` to 110.6 MB at `N=350`.

## Options compared

| option | memory | metric parity | decision |
|---|---:|---|---|
| Dense-only with documented limits | `O(N^2)` distances plus `O(N^2)` bools | current reference | reject: documents the cliff but does not improve long signals |
| Sparse recurrence storage | `O(nnz)` after building/searching neighbors | exact if all line scans preserve ordering | defer: needs a neighbor-search API and sparse run scanners |
| Streaming diagonal+vertical scans | `O(N*d)` trajectory plus one `O(N)` mask | exact for scalar RQA metrics | choose for prototype |
| Staged API | can expose dense, streaming, then sparse | exact when each mode states limits | use: additive streaming function now, sparse later |

Chosen path: add an explicit streaming RQA function that scans upper diagonals
for diagonal runs and columns for vertical runs. It never materializes the dense
recurrence matrix. The default result remains the same scalar dict as `rqa`.

## Public-interface decision recorded before implementation

Function: `rqa_streaming_from_trajectory(X, eps=None, metric="euclidean",
percentile=5, l_min=2, v_min=2, *, theiler=0, return_counts=False)`.

Accepted inputs:
- `X`: finite trajectory accepted by the existing trajectory validator.
- `eps`: finite non-negative threshold. If omitted, the exact percentile of
  positive pairwise distances is used, matching `recurrence_matrix`; this exact
  threshold path still allocates SciPy's condensed distance vector.
- `metric`: exact streaming support for `euclidean`, `sqeuclidean`, `cityblock`,
  `manhattan`, and `chebyshev`.
- `l_min`, `v_min`: positive integers with the same semantics as dense `rqa`.
- `theiler`: non-negative integer. `0` preserves current dense behavior; positive
  values mask pairs with `abs(i-j) <= theiler` before all RR/DET/LAM/L/TT/ENTR/
  Lmax counts.

Return fields:
- default: dict with exactly `RR`, `DET`, `LAM`, `L`, `TT`, `ENTR`, `Lmax`, with
  values intended to match `rqa(recurrence_matrix(...))` byte-for-byte in keys
  and numerically for the same threshold/mask.
- `return_counts=True`: `(stats, details)` where `details` includes `eps`,
  `theiler`, `diagonal_lengths`, `vertical_lengths`, `entr_bins`, and
  `entr_counts`. The bins/counts are the exact diagonal-line histogram used for
  `ENTR`.

Approximation flags: none. The function is exact for the listed metrics. The
only limitation is threshold selection: `eps=None` is exact but not memory-bound
because exact percentile selection uses all condensed pairwise distances. Long
signal callers should pass an explicit `eps`; approximate threshold selection is
not hidden behind this API.

Compatibility:
- Existing `recurrence_matrix`, `rqa`, and `rqa_from_trajectory` signatures and
  outputs remain unchanged.
- The new API is additive. With `theiler=0`, explicit `eps`, and a supported
  metric, it matches dense recurrence/RQA for RR, DET, LAM, L, TT, ENTR, Lmax,
  and ENTR bins.

## Prototype memory evidence

Measured in separate Python subprocesses with `np.random.default_rng(20260611)`,
2D cumulative-normal trajectories, `eps=0.2`, `metric="euclidean"`, `l_min=2`,
`v_min=2`. Peak RSS is process high-water mark from `/proc/self/status`.

| mode | N | peak RSS MB | wall s | RR | DET |
|---|---:|---:|---:|---:|---:|
| dense `recurrence_matrix+rqa` | 300 | 107.049 | 0.000718 | 0.003911 | 0.000000 |
| streaming | 300 | 105.431 | 0.007088 | 0.003911 | 0.000000 |
| dense `recurrence_matrix+rqa` | 700 | 111.301 | 0.002753 | 0.001686 | 0.063492 |
| streaming | 700 | 105.824 | 0.039641 | 0.001686 | 0.063492 |

The small CI-friendly sizes are dominated by interpreter baseline RSS, but the
dense path grows with `N^2` while the streaming path stays near baseline here.
