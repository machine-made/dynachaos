# Rust-Port Assessment: Comoving Lyapunov and Basins

## Scenario Fingerprint

- Run ID: `2026-06-01-rust-assess`
- Git SHA at profiling start: `280a24b`
- Host: `nexus-dev`
- Kernel: `Linux 7.0.0-15-generic x86_64`
- Python command: `uv run python` inline timing probe
- Python runtime: `3.13.13`
- NumPy: `2.4.2`
- Workload isolation: normal interactive shell; no sudo, governor, or perf tuning
- Cache state: warm Python process; probe NPZ files deleted after each run

## Production Workload Shapes

| Candidate | Production shape | Dominant loop |
| --- | --- | --- |
| `comoving_lyapunov_spectrum` | `N=500`, `301` velocities, `n_iter=100_000`, `3` `a` values | `len(v_values) * n_iter` vector CML steps per `a` |
| `compute_basins` | `n_grid=800`, `n_transient=50_000`, `reference_transient=500_000` | `n_grid * n_grid * n_transient` grid-element updates |

## Baseline Timings

| Rank | Scenario | Runs | p50 | p95 | Normalized cost | Production extrapolation | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `comoving`, `N=500`, `31` velocities, `n_iter=1000` | 3 | `0.634336s` | `0.643468s` | `20.46 us` per inner vector step | about `10.3 min` per `a`, about `30.8 min` for 3 `a` values | inline timing output |
| 2 | `basin`, `n_grid=160`, `n_transient=1000` | 3 | `1.094816s` | `1.097864s` | `42.77 ns` per grid-element update | about `22.8 min` for `800^2 * 50_000` updates | inline timing output |
| 3 | `basin`, `n_grid=160`, `n_transient=500` | 3 | `0.560264s` | `0.560299s` | `43.77 ns` per grid-element update | confirms near-linear scaling in transient length | inline timing output |
| 4 | `basin_reference_transient`, `100_000` steps | 3 | `0.091948s` | `0.092437s` | `0.92 us` per scalar step | about `0.46s` for `500_000` reference steps | inline timing output |
| 5 | `comoving`, `N=500`, `9` velocities, `n_iter=300` | 3 | `0.059531s` | `0.059614s` | `22.05 us` per inner vector step | consistent with the larger comoving probe | inline timing output |

The production extrapolations are scaled estimates, not full production runs.
They are sufficient to decide that both candidates deserve split follow-up
porting Beads.

## Attribution

### Comoving Lyapunov

`cProfile` on `N=500`, `9` velocities, `n_iter=300`, `n_transient=500`:

| Rank | Location | Cumulative time | Call count | Interpretation |
| ---: | --- | ---: | ---: | --- |
| 1 | `comoving_lyapunov_spectrum` | `0.104s` | 1 | orchestrates all work |
| 2 | `numpy.roll` | `0.066s` | 11800 primitive calls | repeated allocation/copy pressure inside the segment loop |
| 3 | logistic closure and `logistic` | `0.015s` | 6400 calls | generic callable API adds overhead that a specialized logistic kernel can avoid |
| 4 | derivative closure and `logistic_derivative` | `0.006s` | 5400 calls | same specialization opportunity |

Decision: go. Add a specialized logistic co-moving CML kernel rather than
trying to port the generic callable API.

## Basin Computation

`cProfile` on `n_grid=80`, `n_transient=300`, `reference_transient=2000`:

| Rank | Location | Cumulative time | Call count | Interpretation |
| ---: | --- | ---: | ---: | --- |
| 1 | `compute_basins` | `0.175s` | 1 | row loop and transient loop dominate |
| 2 | `logistic` | `0.043s` | 52064 calls | called twice per row-step plus reference work |
| 3 | reference orbit helpers | `0.003s` | 2000 transient steps | reference transient is not the production bottleneck |
| 4 | `numpy.where` | `0.002s` | 48240 calls | row-step divergence masking adds allocation pressure |

Decision: go. Add a closed-form Rust basin kernel for this fixed two-site
coupled logistic map. The generic plotting and NPZ payload contract should
stay in Python.

## Hypothesis Ledger

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| Comoving Lyapunov is worth a Rust port | Supports | Production estimate is roughly `30 min` for the current figure shape; profile points at repeated `np.roll` and callable overhead. |
| The generic callable comoving API should be ported directly | Rejects | Rust cannot call arbitrary Python `f`, `df`, `g`, `dg` cheaply; the production caller uses logistic `g=f`, so specialize that path. |
| Coupled-logistic basins are worth a Rust port | Supports | Production grid estimate is roughly `23 min`, dominated by a closed-form two-variable recurrence. |
| Basin reference-orbit search is the bottleneck | Rejects | `500_000` reference steps extrapolate to about `0.46s`, negligible next to the grid transient loop. |
| Implement either port in the assessment bead | Rejects | Both require new parity tests and API decisions; split follow-up Beads keep the risk bounded. |

## Follow-Up Beads

Create separate porting Beads:

1. Specialized logistic co-moving Lyapunov kernel.
2. Coupled-logistic basin grid kernel.

Both should preserve the public Python orchestration and add small-shape
Rust/Python parity tests before any production figure work changes.
