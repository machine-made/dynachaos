# Changelog

Versions before 0.4.0 were private and unpublished; their entries were
reconstructed from git history. 0.4.0 is the first public release.

## Unreleased

### Added

- The paper page computes the devil's staircase live: the rotation number
  rho(A) is drawn in the browser at a drive frequency D the reader sets with a
  slider or number field (default 0.25). The values go into the page address,
  so a link reproduces the view. At D = 0.25 the live curve matches the
  published data within 1/256 below K_c = 1/(2 pi)
  (`scripts/check_wasm_staircase.py`, run in CI). The Lyapunov panel stays a
  static image.

- The paper page computes the delayed-logistic attractor live: the (x, y)
  point cloud at a map parameter D the reader sets with a slider or number
  field (default 1.90, the kernel's [1.4, 3.5] clamp as the range), plus an
  iteration-count control bounded at 4096 plotted states. The values go
  into the page address, so a link reproduces the view. The map uses only
  +, -, *, so the browser orbit matches the published trajectories bit for
  bit (`scripts/check_wasm_delayed_logistic.py`, run in CI). The published
  twelve-panel figure stays below the live cloud.

- The paper page computes the torus-doubling attractors live: the (X, Y)
  projection of map (I) or map (IV) at a map parameter D the reader sets
  with a slider or number field, plus an iteration-count control bounded
  at 4096 plotted states. A map selector switches between map (I)
  (A = 0.4, D in [1.9, 2.25]) and map (IV) (A = 0.3, D in [1.48, 1.53]),
  each window the published doubling sweep. The values go into the page
  address, so a link reproduces the view. The maps use only +, -, *, so
  the browser orbit matches the published trajectories bit for bit
  (`scripts/check_wasm_torus_doubling.py`, run in CI). The published
  three-panel figure stays below the live cloud.

- The paper page computes the double devil's staircase live: the rotation
  numbers rho_theta and rho_phi of the modulated circle map are drawn in
  the browser over a bare-frequency window the reader sets with min/max
  fields, two preset buttons that reproduce the zoom panels' D windows, or
  a drag zoom, plus a forcing-strength slider (eps in [0, 0.2], default
  0.05) and a point-count control bounded at 512. The values go into the
  page address, so a link reproduces the view. A = 0.1 is subcritical, so
  the browser curve matches the published sweep to 1e-6 in rho_theta and
  1e-12 in rho_phi (`scripts/check_wasm_modulated_circle.py`, run in CI).
  The published figure stays below the live curves.

- The paper page computes the CML space-time diagrams live: the field
  x_i^n of the chosen coupled-map lattice is drawn in the browser as one
  heat map (site i across, time n up, magma at the shown field's 1st/99th
  percentiles). A model selector switches between models A, B and C, an
  eps slider moves the coupling inside the chosen model's published
  window (A [0, 0.2] default 0.07; B [0, 0.1] default 0.024; C [0, 0.5]
  default 0.2), a site-count control sets N in [16, 512] (default 200),
  and a play button appends 250-row chunks computed from the window's
  last row while the oldest rows drop off. The values go into the page
  address, so a link reproduces the view. The published nine-panel figure
  stays below the live field.

- The compiled extension exports `delayed_logistic_attractor_tile` and
  `torus_doubling_attractor_tile`. They iterate the delayed logistic map and
  the torus-doubling maps (I) and (IV) in Rust and return the same
  trajectories as the Python modules, bit for bit. The WebAssembly build
  exports both for the browser, with every input clamped. The paper pipeline
  still uses the Python path.

- The compiled extension exports `modulated_circle_rotation_tile` and
  `cml_spacetime_tile`. They compute the modulated circle map's rotation
  numbers and the coupled-map-lattice space-time field (models A, B and C)
  in Rust and return the same results as the Python modules, bit for bit
  where numpy and Rust call the same `sin` (Linux x86-64, checked).
  The WebAssembly build exports both for the browser, with every input
  clamped. The paper pipeline still uses the Python path.

- The compiled extension exports `circle_map_lyapunov_sum`,
  `delayed_logistic_lyapunov_tile`, `torus_doubling_lyapunov_tile` and
  `coupled_delayed_lyapunov_tile`. They compute the circle-map Lyapunov
  exponent and the delayed-logistic, torus-doubling (maps I and IV) and
  coupled-delayed spectra in Rust, agreeing with the Python reference to
  1e-9 absolute per exponent (the orbit is bit-identical; the Householder
  QR differs from `np.linalg.qr` only in the last bits). The WebAssembly
  build exports all four for the browser, with every input clamped. The
  paper pipeline still uses the Python path.

- The compiled extension exports `coupled_logistic_phase_tile`,
  `coupled_logistic_attractor_tile` and `coupled_delayed_projection_tile`.
  They compute the coupled-logistic phase diagram, attractor portraits and
  coupled-delayed `(x, z)` projections in Rust. The WebAssembly build
  exports those three plus `coupled_logistic_basin_grid` and
  `fractalization_attractor_tile` (the latter calls the existing delayed
  logistic kernel), with every input clamped. They are exported; no page
  calls them yet. The paper pipeline still uses the Python path.

- In debug mode (`?debug=1`, `#live-debug` in the page address, or
  `localStorage.dynachaosDebug`), each live figure shows one line of numbers
  under it with a copy button: the figure id, the browser name and version,
  the worker count, the milliseconds from the interact press to the first full
  paint, the median tile time, and the number of main-thread tasks over 50 ms.
  A phone has no console, so a report from one can now carry these numbers.
  The task count reads `n/a` where the browser has no long-task timing
  (Safari). With debug mode off the page adds no box and no observer.

### Changed

- Long-running WebAssembly kernel calls on the double-staircase and space-time live figures now run
  in a dedicated Web Worker instead of blocking the browser's main thread. Modulated circle rotation
  and coupled-map lattice spacetime evaluations execute asynchronously with results transferred back
  without detaching critical headers, and late-arriving results from superseded parameter states are
  dropped before painting. At a four-fold CPU throttle matching phone-class hardware, main-thread tasks
  during parameter slider moves and playback stay under 100 milliseconds, eliminating page freezes while
  preserving bit-identical numerical output.

- The live Arnold-tongue figure displays fraction labels above the plot along a top axis matching
  the published figure, tracking the visible frequency window during drag zooms in the standard
  axis tick typography. The live space-time diagram uses the published panels' aspect ratio and
  fits inside the window.

- Live canvas tick numbers use the same serif typeface as the axis labels, and padding reserves
  room measured against the drawn face. Rotated vertical axis labels no longer intersect tick
  numbers, and horizontal tick numbers remain inside the canvas width at both desktop and mobile
  sizes. Critical-line marks and their text labels follow the site's vermilion accent in light
  and dark schemes. Arnold-tongue tick numbers show no more decimal places than the published
  figure without creating ambiguous tick labels, and the double devil's staircase note clarifies
  that presets jump to the zoom panels' windows and the full frequency range.

- Live figures use the quantities and symbols from the paper captions across visible labels, plot
  titles, and readouts, naming the devil's staircase sweep by nonlinearity K at drive frequency Ω.
  Parameter sliders display readable names instead of internal keys, the double-staircase note
  reflects the full set of three presets, and canvas marks and colour-bar labels render real
  subscripts and superscripts.

- The paper page links to the figure gallery from its top, the gallery links back, and gallery images use the caption as alt text.

- On a 390 px phone the reading controls wrap inside the screen, a figure opened from a link or a
  jump lands below the fixed bar, 'reset view' stays clear of the plot title, and the page no longer
  scrolls sideways; citations wrap at phone widths and stay on one line on wider screens. On 360 px
  phones, long links wrap, wide equations scroll inside their own box, and the help and
  return-to-default buttons have 44 px tap targets.

- Figures opened from a shared link show a note reporting the applied values with a button to return
  to defaults, figures that compute in the browser carry a live tag beside the interact button, and the
  interaction hint under each live figure is shortened to one line with the full list behind a help toggle.

- `zero_one_statistic` and `zero_one_series` run on the Rust core when the
  compiled extension is available. The results agree with the Python path to
  within 1e-9 for each random frequency. The Python path is still there, and
  `DYNACHAOS_NO_RUST=1` still selects it. The Rust kernel computes the
  autocovariance by FFT, so its cost grows as N log N: at the default
  `n_cut = N/10` and 100 frequencies it takes 0.34 s for N = 100000, where
  the Python path takes 15 s.

### Fixed

- The WebAssembly build no longer embeds the build machine's source paths.
  rustc had written the absolute path of `rust/core` into the module, so a
  module built in a home folder carried that folder name to every reader.
  `scripts/build_wasm.py` now remaps the home folder, the cargo and rustup
  folders and the repository root, and both build scripts fail if the built
  files still hold `/home/`, `/Users/` or `C:\Users`
  (`scripts/check_wasm_paths.py`, also a CI step).

## 0.4.1 — 2026-07-28

Packaging and documentation only. No functional change to the library.

### Fixed

- The published 0.4.0 metadata carried only `Programming Language :: Python ::
  3`, so PyPI — and every badge reading from it — reported the supported version
  as "3". The per-version classifiers for 3.12, 3.13 and 3.14 were added to
  `pyproject.toml` after 0.4.0 shipped, which left the released artifact
  disagreeing with the source at the same version number. This release carries
  them.

### Changed

- README: banner-first layout with no H1, and the opening sentence now leads
  with `dynachaos` and names the actual diagnostics, matching the other
  openfluids repositories. Dropped the Lorenz epigraph, the redundant
  "Available on PyPI" line, and the logistic-map bifurcation figure.

## 0.4.0 — 2026-07-26

First public release, under the openfluids organization and on PyPI.

### Project

- Moved to `github.com/openfluids/dynachaos` and published to PyPI as
  `dynachaos`. Install with `pip install dynachaos`.
- Removed material specific to a separate, unpublished manuscript that used
  this package. Citations to Kunihiko Kaneko's published papers throughout
  `src/` are unchanged — they are scientific attribution.
- `CITATION.cff` now cites the software itself rather than an unpublished
  manuscript.
- Reframed the `figures/` tree as a reproduction gallery: section-indexed
  reproductions of Kaneko's published work that double as golden test data.
- Removed changelog commit links that pointed at a repository and commit
  hashes that no longer exist.

### Dependencies

- Rust edition 2021 → 2024; pyo3 0.28 → 0.29, numpy 0.28 → 0.29,
  rayon 1.10 → 1.12, ndarray → 0.17.2.
- Development tooling moved to current releases (maturin, ruff, pytest).
- Runtime floors for numpy and scipy are unchanged; they are minimums, not
  targets, and raising them would force needless upgrades on users.

### Fixed

- `TestVersion` asserted a hardcoded `"0.2.0"` while the package reported
  `0.3.0`, so the suite was red from the 0.3.0 release prep onward. It now
  compares `__version__` against the installed package metadata, which is the
  drift this test existed to catch and which no longer breaks on a bump.

### Packaging

- Renamed the release workflow to `release.yml`, matching the PyPI trusted
  publisher and the other openfluids packages.
- Excluded `figures/`, `tests/`, and benchmark results from the sdist; the
  tracked figure data would otherwise have pushed it past PyPI's size limit.

## 0.3.0 — 2026-06-11

### Licensing

- Relicensed the project from MIT to Apache-2.0 with a `NOTICE` file; earlier
  unpublished versions were MIT.

### User documentation spine

- Added `docs/real-analysis-guide.md`, a user guide for real analyses: input
  expectations, diagnostic choice, long-signal/RQA scaling guidance,
  reliability-metadata interpretation, and package positioning. README
  quickstart now runs the tested external-signal recipe; all shown commands
  are executed in checks or explicitly marked local/full-run.

### RQA consolidation

- Consolidated the matrix-free RQA scan into one shared core
  (`_trajectory_rqa_scan`); `rqa_streaming_from_trajectory` now delegates to
  it. Fixed `rqa_from_trajectory`'s `eps=None` percentile to match the dense
  `recurrence_matrix` squareform multiset, with a pinning regression test.

### Example recipes

- Added tested example recipes under `examples/recipes/`: external-signal
  analysis with diagnostic selection and reliability metadata, and a
  long-signal/downsampled streaming-RQA recipe that stays inside the dense
  recurrence memory envelope; smoke-tested by new `tests/test_examples.py`.

### Rust acceleration roadmap

- Added a measured Rust-kernel acceleration roadmap
  (`docs/rust-acceleration-roadmap.md`) and the subprocess-isolated hotspot
  profiler `benchmarks/rust_hotspot_profile.py` with checked-in artifacts;
  streaming RQA is ranked as the next port candidate with a recorded parity
  test plan.

### Scalable analysis workflow

- Added `dynachaos analyze <config.jsonc>`: config-driven workflow for external
  `.npy`/`.npz` or generated signals writing a stable output directory
  (`results.json`, `metadata.json` with scale/cost and reliability metadata,
  `summary.md`), with explicit failure modes and a dense-RQA scale-envelope
  guard.

### Long-signal RQA scaling

- Added `rqa_streaming_from_trajectory`: exact matrix-free RQA (RR, DET, LAM,
  L, TT, ENTR, Lmax and ENTR bins) with Theiler-window support and a recorded
  interface decision plus RSS evidence in `docs/rqa-scaling-design.md`.

### Private release posture and documentation

- Renamed the default branch to `main` and hardened project checks around that
  branch convention.
- Sharpened the internal maintenance workflow, including private-only pushes
  and per-change review/commit discipline.
- Clarified that the repository, package publication, benchmark numbers, and
  citation metadata remain private/provisional until a future public release
  phase.
- Refreshed README backend notes so they match the exported Rust surface and
  documented the verified manuscript build sequence.

### Diagnostics correctness and numerical edge cases

- Added validation for correlation norms, radius grids, Theiler windows, and
  undefined correlation-dimension cases.
- Hardened time-delay embedding parameter validation with deterministic
  fuzz-style tests.
- Fixed degenerate recurrence auto-thresholding for constant signals.
- Rejected non-finite entropy and recurrence diagnostic inputs, invalid
  recurrence thresholds, invalid 0-1 test parameters, and zero-MSD 0-1 test
  regressions found during a systematic bug hunt.
- Added entropy and recurrence metamorphic tests for translation and scaling
  invariants.
- Tightened RQA recurrence-matrix and line-threshold validation across Python
  and direct Rust entry points.
- Added metamorphic tests for correlation-integral radius monotonicity and
  Theiler-window valid-pair counts.

### Rust backend hardening

- Fixed CI and parity-test behavior for pure-Python runs without the Rust
  extension.
- Added direct validation for ordinal-pattern Rust inputs and matching Python
  wrapper validation.
- Recorded that the in-tree Rust extension contains no `unsafe` sites.
- Avoided debug-build overflow in direct Rust calls with huge Theiler windows.
- Validated Rust AMI inputs directly and recorded a Rust undefined-behavior audit.

### Pipeline architecture and performance evidence

- Added a reusable NPZ cache contract with required-key validation for figure
  pipeline caches.
- Adopted the cache contract in the circle-map figure pipeline.
- Added architecture, simplification, complexity, profiling, and optimization
  decision artifacts so future maintainers can distinguish measured work from
  deferred ideas.
- Captured further architecture, simplification, complexity, profiling, and
  optimization artifacts, including an RQA count-reuse optimization measured at
  roughly 1.3--1.5x faster in the profiled range.
- Added a `dynachaos --version` CLI path for installed-package introspection.

### Paper and README maintenance

- Relaxed optional dependencies in the vendored journal class so local builds can
  progress farther on smaller TeX installations.
- Removed optional `enumitem`/TikZ manuscript dependencies and regenerated the
  tracked paper PDF after a clean pdflatex/BibTeX build.
- Added Python, Rust unsafe, and LaTeX review artifacts for Loop 1 of the
  improvement campaign.

### Earlier project build-out

- Created the initial dynachaos codebase, maps, diagnostics, visual assets,
  paper materials, and CI/release workflow scaffolding.
- Added paper figure polish, manuscript synchronization, entropy-family
  diagnostics, JSONC example configs, Rust solver timing work, and benchmark
  corrections across the pre-goal history.

## Evidence Sources

- `git log --reverse --oneline --decorate=no --no-merges`
- `git tag --list --sort=-creatordate`
- `gh release list --limit 50`
- `README.md`, `pyproject.toml`, `.github/workflows/ci.yml`,
  `.github/workflows/publish.yml`
