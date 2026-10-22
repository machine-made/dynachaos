# Rust acceleration roadmap

Generated for bead `dynachaos-rust-kernel-hotspot-roadmap-8i3`.
No new kernel was implemented here; this is a measurement-backed roadmap.

## Measurement commands

- `cd dynachaos && uv run python benchmarks/rust_hotspot_profile.py benchmarks/rust_hotspot_profile.jsonc`
- Prior scale evidence: `cd dynachaos && uv run python benchmarks/scale_envelope.py benchmarks/scale_envelope.jsonc`
- Result artifact used here: `benchmarks/results/rust_hotspot_profile.{json,md}`.
- Scale artifact used here: `benchmarks/results/scale_envelope.{json,md}`.

All rows ran in fresh subprocesses with fixed seed `20260611`; wall time is p50 of
3 repeats and RSS is `/proc/self/status` VmHWM. Sizes are deliberately
local-friendly, not publication-scale.

## Ranked candidates

| rank | candidate | measured p50 wall s / RSS | value | risk | roadmap decision |
|---:|---|---:|---|---|---|
| 1 | Streaming RQA run scans (`rqa_streaming_from_trajectory`) | 0.729482 s, 107.3 MB at N=4000 explicit eps; dense RQA has an 8*N^2 byte distance-matrix cost, impractical near N=23170 under the default cap, per `scale_envelope.md` | High: enables long-signal RQA without dense recurrence matrices | Medium: exact diagonal + vertical counting, metrics/theiler edge cases | Next new Rust kernel if porting starts |
| 2 | Coupled-logistic basin classification inner loop | Python 1.32182 s, 81.3 MB at 180^2*1000; existing Rust path 0.100735 s, 82.1 MB at 360^2*3000 | High for Sec. 3 basin figures and large grids | Low: already specialized and parity-tested | Keep pattern; no new work here |
| 3 | Comoving Lyapunov logistic CML loop | Python 0.646648 s, 106.8 MB; existing Rust path 0.0123518 s, 107.1 MB at N=500, 31 velocities, 1000 iterations | High for spatiotemporal diagnostics | Medium: generic callable API cannot port cleanly; logistic specialization only | Keep specialized Rust path; no generic rewrite |
| 4 | Exact all-pairs correlation counts | Python 0.442567 s, 107.2 MB at N=4000; existing Rust path 0.0208908 s, 108.2 MB at N=12000 | High but GP acceleration already landed | Low: existing exact Rust counter with fallback parity | Maintain; use as template |

## Justification

Streaming RQA is ranked first for future work because it is the unported hotspot
with the clearest scaling pressure: the current streaming Python path avoids the
`O(N^2)` dense matrix but still performs Python-level diagonal and column scans.
The dense recurrence evidence gives the scientific motivation, while the N=4000
streaming profile is already in the same subsecond-to-second band as other
previously ported kernels at small local sizes. The basin and comoving kernels
remain important, but they already have Rust implementations and parity tests;
this bead should not duplicate them. Exact pair counting is the package's
successful precedent rather than the next port.

## Parity test plan for the top candidate: streaming RQA

Python remains the reference implementation. Before any Rust dispatch is wired:

1. Fixtures: deterministic 1D logistic map embeddings, 2D cumulative-normal
   trajectories with `np.random.default_rng(20260611)`, constant trajectories,
   two-point/minimal trajectories, and the existing dense-parity cases in
   `tests/test_rqa_streaming.py`.
2. Exact comparisons: dictionary keys exactly `RR`, `DET`, `LAM`, `L`, `TT`,
   `ENTR`, `Lmax`; `diagonal_lengths`, `vertical_lengths`, `entr_bins`, and
   `entr_counts` exactly equal when `return_counts=True`.
3. Tolerances: scalar floating metrics compare to Python with `rtol=0`,
   `atol=1e-15` for identical explicit `eps`; percentile-selected `eps` must
   match the Python threshold first, then use the same tolerance.
4. Coverage: metrics `euclidean`, `sqeuclidean`, `cityblock`/`manhattan`, and
   `chebyshev`; `theiler=0` and positive theiler masks; no-recurrence and
   all-recurrence degeneracies.
5. Backend honesty: default API remains Python fallback when Rust is absent or
   disabled by `DYNACHAOS_NO_RUST`; `ReliabilityRecord.backend` reports
   `"rust"` only when the Rust scanner is actually used and `"python"`
   otherwise. Scale evidence points to this roadmap/profile artifact.

## Package pattern

Keep public analysis APIs Python-facing and stable. Add Rust only when a measured
hotspot justifies the maintenance cost. Every Rust kernel must have: a pure
Python reference path, fallback tests with Rust disabled, parity tests against
that reference, a reproducible before/after benchmark, and honest reliability
metadata naming the backend actually used. The existing GP, comoving-logistic,
and coupled-basin kernels are the model; streaming RQA is the next candidate,
not a commitment made by this bead.
