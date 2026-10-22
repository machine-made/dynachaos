# Performance Profiling Loop 1

## Scenario Fingerprint

- Run ID: `2026-05-27-loop1`
- Git SHA at profiling start: `e7ef0ec`
- Host: `nexus-dev`
- Kernel: `Linux 7.0.0-15-generic x86_64`
- Python command: `uv run --extra viz python`
- Python runtime: `3.13.13` inside uv environment
- NumPy: `2.4.2`
- SciPy: `1.17.0`
- Rust extension: available
- Workload isolation: normal interactive shell; no kernel/governor tuning
- Cache state: warm in-process arrays; no disk I/O in measured functions

## Scenarios

1. `correlation_dimension()` on circle trajectories.
2. `recurrence_matrix()` plus `rqa()` on deterministic 2D trajectories.
3. `sample_entropy()` and `fuzzy_entropy()` on fixed random series.

Each benchmark used warmup calls followed by repeated wall-clock timings with
`time.perf_counter()`.

## Baseline Timings

| Rank | Scenario | Runs | p50 | p95 | Max | Category | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `recurrence_matrix()+rqa()` N=4000 | 5 | 0.108933s | 0.109216s | 0.110143s | CPU/memory | command output |
| 2 | `recurrence_matrix()+rqa()` N=2000 | 5 | 0.028673s | 0.029123s | 0.029191s | CPU/memory | command output |
| 3 | `correlation_dimension()` circle N=6000, n_r=30 | 5 | 0.007255s | 0.007361s | 0.007816s | CPU | command output |
| 4 | `correlation_dimension()` circle N=3000, n_r=30 | 5 | 0.002389s | 0.002476s | 0.002576s | CPU | command output |
| 5 | `recurrence_matrix()+rqa()` N=900 | 10 | 0.005534s | 0.005639s | 0.005644s | CPU/memory | command output |
| 6 | `fuzzy_entropy()` N=900 | 10 | 0.001030s | 0.001183s | 0.001442s | CPU | command output |
| 7 | `correlation_dimension()` circle N=1200, n_r=30 | 10 | 0.000714s | 0.000807s | 0.000939s | CPU | command output |
| 8 | `sample_entropy()` N=900 | 10 | 0.000291s | 0.000417s | 0.000494s | CPU | command output |

## Scaling Evidence

- `recurrence_matrix()+rqa()` rose from `0.028673s` at N=2000 to `0.108933s`
  at N=4000, approximately the expected dense `O(N^2)` scaling.
- `correlation_dimension()` with the Rust extension active stayed below
  `0.008s` at N=6000 for this circle workload.
- Entropy kernels at N=900 are not Loop 1 optimization targets with the Rust
  extension active.

## Hypothesis Ledger

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| Correlation dimension is the top current CPU bottleneck | Rejects | Rust-backed N=6000 circle timing p95 was `0.007361s`. |
| Recurrence/RQA dense matrix is the best optimization candidate | Supports | It is the slowest measured diagnostic path and shows quadratic scaling. |
| Entropy pair counts need immediate optimization | Rejects | Rust-backed `sample_entropy()` and `fuzzy_entropy()` were below `0.002s` at N=900. |
| Optimize before profiling whole figure sections | Rejects | Section-level cold/warm cache costs were not measured yet. |

## Hand-Off

For `extreme-software-optimization`, the only measured candidate with enough
signal is recurrence/RQA memory and CPU scaling. A safe optimization would need:

- a representative target size larger than N=4000,
- peak RSS measurement,
- behavior parity for `RR`, `DET`, `LAM`, `L`, `TT`, `ENTR`, and `Lmax`,
- a clear decision on whether sparse/streaming recurrence behavior is allowed.

No optimization should be applied to correlation dimension or entropy in Loop 1
without a heavier workload proving those are bottlenecks.
