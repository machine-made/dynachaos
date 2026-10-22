# Claims checklist

| Claim location | Claim | Status |
|---|---|---|
| README lead; JOSS Summary | dynachaos is a reusable tool for simulated or measured dynamical-systems time signals | supported-by: package API surface and test suite |
| README/JOSS Rust wording | Rust kernels accelerate selected heavy diagnostics | supported-by: test suite parity plus scale_envelope artifact |
| README Benchmarks | CI-mode Grassberger-Procaccia logistic N=1000 Rust speedup is 42.95x | **not-reproducible-do-not-quote** — see below |
| README Benchmarks | The Rust Grassberger-Procaccia kernel is roughly an order of magnitude faster than the Python fallback at N=1000 | supported-by: repeated warm measurement, see below |
| README Benchmarks | Dense recurrence/RQA has an 8*N^2 byte analytical matrix cost | supported-by: `benchmarks/results/scale_envelope.md` |
| README fallback policy; JOSS Software design | Pure-Python paths are parity/portability paths, not large-run performance claims | supported-by: test suite; unsupported performance implication removed |
| README reproduction gallery | The reproduction gallery is the flagship application and stress test, not the package boundary | supported-by: reproduction pipeline; scope claim reframed |
| JOSS reproducibility text | Fixed seeds and selected cache checks guard the fast figure pipeline | supported-by: test suite; universal figure-output claim softened |
| JOSS package scope | External trajectories can be analysed by the diagnostics | supported-by: diagnostics accepting array-like trajectory/time-series inputs |
| Removed/softened claims | Hype or unqualified superlatives about performance/scope | unsupported-removed |
| Gallery site; any live-figure text | A live figure computes with the same kernels the published figure used | supported-by: shared `rust/core`, plus the CI wasm-vs-native parity check |
| Gallery site; any live-figure text | A live figure reproduces the published pixel values everywhere | **unsupported-do-not-claim** — true where the dynamics are locked, false in the chaotic region; see below |
| Gallery site; any live-figure text | A live figure keeps computing in a hidden tab, starts itself on a phone, or logs telemetry for every reader | **unsupported-do-not-claim** — a hidden tab pauses new work, a viewport below 480 CSS px keeps the PNG until the reader asks, and telemetry is off unless `#live-debug`, `?debug=1`, or `localStorage.dynachaosDebug` is set |
| sec12 intermittency; any manuscript text | The Lorenz laminar-channel fit recovers the type-I tangency slope (~1) | **unsupported-do-not-claim** — see below |


## `lorenz_channel_slope` is not a converged quantity

Type-I intermittency makes the return map tangent to the diagonal in the
laminar channel, so a channel-slope estimate near 1 is the expected physics.
That physics is not in question. What is not supported is the claim that the
Lorenz channel fit in `sec12_intermittency` *measures* it.

Measured on 2026-07-27 at the shipped configuration
(`lorenz_1662_oracle(t_span=(0, 80), dt=0.01)`, `channel_percentile=30`,
~274 extracted maxima):

- Perturbing the initial condition by `1e-12` moves the fitted slope across
  **[0.617, 1.540]**. The committed value 0.98549932 is one draw from that
  distribution, not a reproducible measurement. CI runners observed 2.048.
- Longer integration does not fix it: at `t_span=(0, 1500)` (2017 channel
  points) the spread across the same perturbations is still 0.48.
- The falsifying check fails. If the fit resolved the tangency, narrowing the
  channel would drive the slope to 1. At `t_span=(0, 1500)` the slope runs
  1.223, 0.519, 0.652, 0.725, 0.850, 0.822, 0.819 for `channel_percentile`
  30, 20, 10, 5, 2, 1, 0.5 — non-monotonic, tending to ~0.82, not 1.

The estimator is not simply wrong: applied to `logistic_f3_channel_slope`, a
deterministic 1-D map, it returns **1.00042829**. The Lorenz case is a chaotic
ODE where the extracted maxima do not determine the channel.

**Do not state a Lorenz tangency-slope value in the paper or the README**
without first reworking the fit and demonstrating convergence under refinement.
The test asserts only that the channel is found and the slope is finite and
positive; `tests/test_intermittency_figure.py` records the reasoning.


## The 42.95x Grassberger-Procaccia speedup is not a reproducible number

The kernel is fast, and that is not in question. What is not supported is the
specific figure **42.95x**, because `benchmarks/scale_envelope.py` times one
un-warmed call per case and lets rayon use every core.

Measured on 2026-09-07 on the development box (24 cores), same harness, same
committed configuration:

- Two consecutive runs of the harness reported **12.03x** and **8.86x** for the
  same case (logistic, N=1000). The committed artifact reports 42.95x.
- The Python column barely moved across all three (0.0370, 0.0344, 0.0356 s).
  The Rust column swung by 40% (0.000861, 0.00286, 0.00402 s). The single-threaded
  path is stable; the parallel one is not, which is the signature of a cold,
  contended measurement rather than a slower machine.
- Timed properly — 3 warm-up calls, then the median of 15 — the same kernel runs
  **0.000674 s at 8 threads**, faster than the 0.000861 s in the committed
  artifact. Nothing regressed; the harness is measuring start-up.
- rayon's default of one thread per core is past the optimum here: 0.67 ms on 8
  threads against 0.88 ms on 24.

So the ratio the harness prints depends on how loaded the machine is and how
many cores it has, not on the software. The README already says these numbers
must be regenerated on the release target hardware, which is the right
instinct, but a single-shot measurement cannot support a two-decimal claim.

**Do not quote a specific speedup multiple** until `scale_envelope.py` warms up
and reports a median over repeats, and the reported thread count is recorded
alongside the number. An order-of-magnitude statement is supportable today; a
figure like 42.95x is not.


## A browser figure cannot reproduce chaotic pixels, and should not claim to

The WebAssembly build runs the same kernels as the published figures, and in
the well conditioned parts of a parameter plane it returns the same numbers.
In the chaotic parts it does not, and no amount of care would make it.

Measured on 2026-09-08 over a 64 x 64 tile of the Arnold-tongue plane
(`scripts/check_wasm_parity.py`):

- NumPy and native Rust agree on all 4096 cells exactly. Both call the platform
  sine.
- The wasm module carries its own sine, which differs by about one unit in the
  last place.
- Below the critical line `K = 1 / (2 * pi)`, about 0.159, that difference
  stays one unit in the last place: 2 cells differed, by at most 7.8e-16.
- Above it the map is not invertible and the orbit is chaotic. The same one bit
  grows over 700 iterations: 15 of 1920 cells differed, 13 of them by more than
  the 1e-12 floor the check counts from, by up to 2.9e-2.

The locked tongues are what the Arnold-tongue figure is about, and they are
reproduced exactly. The chaotic sea between them is not reproducible per pixel
across sine implementations, compilers, or processors. That is a property of
the dynamics.

**Say that a live figure computes the same system with the same code. Do not
say it reproduces the published image pixel for pixel**, and do not invite a
reader to compare a zoomed chaotic region against the PNG and infer that
something is broken when the two differ. The PNG remains the record of what the
paper computed; see `docs/wasm-architecture.md`.

A live figure also does not run unasked. A hidden tab pauses new tiles and
new kernel calls. A viewport below 480 CSS px keeps the PNG until the reader
presses interact. Telemetry is off unless the hash contains `#live-debug`,
the query is `?debug=1`, or `localStorage.dynachaosDebug` is set.
