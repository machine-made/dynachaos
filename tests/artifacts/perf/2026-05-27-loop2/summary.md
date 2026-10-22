# Performance profiling - Loop 2

## Scenario Fingerprint

- Run ID: `2026-05-27-loop2`
- Git SHA at profiling start: `ee0f98c`
- Host: `nexus-dev`
- Kernel: `Linux 7.0.0-15-generic x86_64`
- CPU: AMD Ryzen 9 9900X 12-Core Processor, 18 online CPUs reported by
  `lscpu`, SMT listed as 1 thread per core in this VM.
- RAM: 24 GiB total, 21 GiB available at fingerprint time.
- Python command: `uv run --extra viz python`
- Python runtime: `3.13.13`
- NumPy: `2.4.2`
- SciPy: `1.17.0`
- Rust extension: available
- Workload isolation: normal interactive shell; no sudo, governor, perf, or
  kernel tuning.
- Peak RSS source: Python `resource.getrusage(RUSAGE_SELF).ru_maxrss`.
- Cache state: warm in-process arrays for diagnostics; warm existing `figures/`
  cache for the section smoke run.

## Scenarios

1. `recurrence_matrix()+rqa()` on deterministic 2D trajectories.
2. `correlation_dimension()` on circle trajectories.
3. `sample_entropy()` and `fuzzy_entropy()` on fixed random series.
4. `multifractal_spectrum()` on a fixed lognormal field.
5. Warm-cache `sec02_circle_map` smoke pipeline.

## Baseline Timings

| Rank | Scenario | Runs | p50 | p95/Max | Peak RSS Delta | Category | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `recurrence_matrix()+rqa()` N=6000 | 5 | `0.251490s` | `0.254493s` | `457.7 MB` | CPU/memory | command output |
| 2 | Warm-cache `sec02_circle_map` smoke pipeline | 3 | `2.291826s` | `2.294081s` | `0.0 MB` | pipeline/render/cache | command output |
| 3 | `recurrence_matrix()+rqa()` N=4000 | 7 | `0.119860s` | `0.122747s` | `275.1 MB` | CPU/memory | command output |
| 4 | `correlation_dimension()` circle N=24000 | 7 | `0.104469s` | `0.107240s` | `0.0 MB` | CPU | command output |
| 5 | `recurrence_matrix()+rqa()` N=2000 | 7 | `0.034117s` | `0.034770s` | `108.2 MB` | CPU/memory | command output |
| 6 | `multifractal_spectrum()` 192x192 | 7 | `0.018565s` | `0.018606s` | `0.0 MB` | CPU | command output |
| 7 | `correlation_dimension()` circle N=12000 | 7 | `0.027711s` | `0.028752s` | `0.0 MB` | CPU | command output |
| 8 | `fuzzy_entropy()` N=3000 | 15 | `0.009475s` | `0.011132s` | `0.0 MB` | CPU | command output |
| 9 | `correlation_dimension()` circle N=6000 | 7 | `0.007429s` | `0.008058s` | `0.0 MB` | CPU | command output |
| 10 | `sample_entropy()` N=3000 | 15 | `0.001393s` | `0.001840s` | `0.0 MB` | CPU | command output |

## Scaling Evidence

- Dense recurrence/RQA peak RSS delta rose from `108.2 MB` at N=2000 to
  `457.7 MB` at N=6000 in one process. The timing rose from `0.034117s` to
  `0.251490s`, matching the expected dense `O(N^2)` pressure.
- Rust-backed `correlation_dimension()` reached N=24000 with p50 `0.104469s`
  and no additional high-water RSS above the recurrence run in the same
  process.
- Rust-backed entropy kernels were not measurable bottlenecks at N=3000.
- Warm-cache `sec02_circle_map` smoke runs took about `2.29s`, dominated by
  module execution and PNG rendering while reusing existing NPZ caches.

## Hypothesis Ledger

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| Dense recurrence/RQA is the top diagnostic optimization target | Supports | Highest diagnostic CPU time and largest peak RSS delta. |
| Correlation dimension is the current top diagnostic bottleneck | Rejects | N=24000 p50 was below recurrence N=6000 and had no new high-water RSS in this run. |
| Entropy pair counts need immediate optimization | Rejects | N=3000 SampEn/FuzzyEn stayed below `0.012s` max with Rust active. |
| Section execution still has nontrivial warm-cache overhead | Supports | `sec02_circle_map` smoke p50 `2.291826s` despite cache reuse. |
| Cold-cache figure sweeps should be optimized from this profile | Rejects | Cold-cache section runs were not measured in this pass to avoid expensive paper-data recomputation. |

## Hand-Off

For `extreme-software-optimization`, the best-supported candidate is still
recurrence/RQA memory scaling. A safe optimization would need:

1. a public-interface decision about whether callers still require the dense
   recurrence matrix;
2. a streaming or sparse RQA prototype measured against this baseline;
3. exact parity tests for `RR`, `DET`, `LAM`, `L`, `TT`, `ENTR`, and `Lmax`;
4. peak RSS comparison in separate processes so high-water marks do not hide
   later allocations.

The second candidate is pipeline warm-cache overhead. It should be addressed
through section-runner orchestration or render profiling, not by changing
scientific simulation loops from this evidence alone.
