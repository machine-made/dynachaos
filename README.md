![dynachaos banner](https://raw.githubusercontent.com/openfluids/dynachaos/main/assets/readme-banner-v3.jpg)

[![CI](https://github.com/openfluids/dynachaos/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/openfluids/dynachaos/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dynachaos.svg)](https://pypi.org/project/dynachaos/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fopenfluids%2Fdynachaos%2Fmain%2Fpyproject.toml&label=python)](https://pypi.org/project/dynachaos/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`dynachaos` analyses time signals from simulations and experiments: maps and
coupled-map lattices, Lyapunov exponents, recurrence quantification, entropy
diagnostics, correlation dimension and multifractal spectra. Performance-critical
kernels have Rust backends where those have parity tests, with pure-Python
fallbacks everywhere else.

## Why dynachaos?

dynachaos is a reusable Python/Rust package for inspecting simulated or
measured dynamical-systems time signals. It collects maps, coupled-map
lattices, recurrence analysis, entropy diagnostics, Grassberger-Procaccia
correlation dimension, multifractal spectra, and reproducible analysis
pipelines in one codebase. Performance-sensitive kernels use Rust backends
where they have parity tests, while pure-Python fallbacks support portability
and reference checks.

## Features

**Maps** — logistic, circle, coupled logistic, delayed logistic, coupled
delayed, modulated circle, torus doubling (Map I / Map IV)

**Coupled Map Lattices** — CML with nearest-neighbor diffusive coupling,
globally coupled maps (GCM), pattern dynamics, cluster statistics

**Diagnostics** — Lyapunov exponents (1D + QR spectrum + flow systems),
0-1 test for chaos, SALI/GALI alignment indices, permutation entropy +
complexity-entropy planes, sample/approximate/fuzzy/multiscale entropy,
recurrence quantification analysis (RQA), Grassberger-Procaccia
correlation dimension, multifractal spectra ($D_q$, $f(\alpha)$),
AMI + Cao + FNN embedding

**Rust backends** — correlation integral (Grassberger-Procaccia), fuzzy
entropy sum, recurrence line extraction, ordinal distribution counting,
AMI histograms, Cao dimension selection, multifractal moments

**Visualization** — bifurcation diagrams, cobweb plots, return maps,
curated Swiss-inspired style themes

## Quick Start

From a fresh checkout, run the tested external-signal workflow recipe:

```bash
uv run dynachaos analyze examples/recipes/external_signal/external_signal_recipe.jsonc
```

The command writes
`examples/recipes/external_signal/outputs/external_signal_recipe/results.json`,
`examples/recipes/external_signal/outputs/external_signal_recipe/metadata.json`,
and
`examples/recipes/external_signal/outputs/external_signal_recipe/summary.md`.
It is intentionally small so it runs in seconds; use the recipe gallery for the
long-signal streaming example.

## User documentation spine

- [Real-analysis user guide](https://github.com/openfluids/dynachaos/blob/main/docs/real-analysis-guide.md): input expectations,
  diagnostic choice, long-signal scaling, workflow outputs, reliability metadata,
  and finite-data caveats.
- [Example gallery](https://github.com/openfluids/dynachaos/blob/main/examples/README.md): tested recipe commands for external
  signals and long-signal streaming RQA.
- [RQA scaling design note](https://github.com/openfluids/dynachaos/blob/main/docs/rqa-scaling-design.md): how much memory dense
  recurrence needs, and the streaming RQA design.
- [Rust acceleration roadmap](https://github.com/openfluids/dynachaos/blob/main/docs/rust-acceleration-roadmap.md): measured
  acceleration claims and future kernel priorities.
- [Claims checklist](https://github.com/openfluids/dynachaos/blob/main/docs/claims-checklist.md): wording boundaries for public
  documentation and manuscripts.

## Config-driven signal analysis workflow

Run scalar/reduced time-series analyses with a JSONC config; all tuning lives in
the config, not CLI flags.

Config schema summary:
- `input`: either `{ "path": "signal.npy" }` / `{ "path": "signal.npz", "npz_key": "x" }` for a 1D finite scalar/reduced signal, or `{ "generated": { "name": "logistic", "n": 1000, "seed": 0 } }` for self-contained runs.
- `output.dir`: stable output directory, resolved relative to the config file.
- `diagnostics`: list of `{ "name": ... }` entries. Supported workflow names are `permutation_entropy`, `correlation_dimension`, `rqa_streaming`, and `rqa_dense`.
- `scale_limits`: optional dense-RQA guard. Dense RQA builds an `N x N` distance matrix costing `8*N^2` bytes; `dense_rqa_max_bytes` caps that, and `allow_dense_rqa_beyond_envelope: true` is required to exceed the cap.

Output layout is stable and referenceable by path: `results.json`
(machine-readable diagnostic values), `metadata.json` (N, shape, wall time in
seconds, peak RSS in MB, and per-diagnostic `ReliabilityRecord` metadata), and
`summary.md` (human-readable report with relative artifact names). Reliability
metadata records backend, parameters, data shape, sampling/downsampling notes,
warnings, unresolved verdicts, and scale evidence. It helps you decide how much
scientific confidence to place in a number; it is not an automatic pass/fail
certificate.

For long-signal local RQA, avoid dense recurrence matrices and set an explicit
threshold in config:

```jsonc
{
  "input": {"path": "long_signal.npy"},
  "output": {"dir": "results/long_signal_rqa"},
  "diagnostics": [
    {"name": "rqa_streaming", "embedding": {"d": 3, "tau": 2}, "eps": 0.08, "l_min": 2, "v_min": 2}
  ]
}
```

The local/full-run command `uv run dynachaos analyze long_signal_rqa.jsonc`
requires a local `long_signal_rqa.jsonc` file and the `long_signal.npy` input it
names. The tested commands live in the quickstart and in `examples/README.md`.

## Rust-Accelerated Backends

Performance-critical algorithms are implemented as Rust kernels. The Rust
extension is required by default: `import dynachaos` fails loudly if it has not
been built.

Build the extension in editable mode:

```bash
uv run maturin develop --release
```

### Pure-Python fallback policy

The Rust kernels are the intended path for production-sized all-pairs and
large-N diagnostics. Pure-Python paths are an explicit opt-in for parity testing
and portability, not an automatic silent fallback. Set `DYNACHAOS_NO_RUST=1`
when you need to exercise them; the test suite checks parity on representative
small workloads. Some fallback implementations remain exact and quadratic by
design, so large pure-Python runs should be treated as diagnostic or development
runs unless a future release explicitly makes large fallback performance a
target.

### Correlation integral (Grassberger-Procaccia)

The all-pairs kernel evaluates all N(N−1)/2 pairs with Theiler-window
exclusion and multi-radius binning in a single pass.
The implementation is designed for correctness and low memory use,
offering several advantages over common baseline scripts (e.g., [notsebastiano/GP_algorithm](https://github.com/notsebastiano/GP_algorithm/blob/master/GP_algorithm.py)):

- **Algorithmic Correctness** — Supports the **Theiler window** ($|i-j| > w$) to
  exclude temporally correlated pairs (Theiler 1986) and uses proper
  normalization ($C(r) \le 1$).
- **Scaling Region Detection** — Employs a **stable plateau search** on local
  slopes instead of simple heuristics, making it robust to noise and saturation.
- **Memory Efficiency** — Uses **O(1) auxiliary memory** per pair (streaming)
  instead of an $O(N^2)$ distance matrix.

The Rust kernel uses:

- **Raw slice indexing** — C-contiguous slice access, avoiding ndarray's
  per-index `Index` overhead while staying in safe Rust
- **Prefix-sum binning** — one write per pair instead of up to 50;
  converted to cumulative counts with a single O(n_r) pass after the loop
- **Squared-distance comparison** — pre-squared thresholds eliminate
  `sqrt()` in Euclidean mode (saves 10–20 cycles per pair)
- **Branch-separated loops** — Chebyshev and Euclidean paths are fully
  separated, enabling independent auto-vectorization
- **Rayon parallelism** — outer loop distributed across all cores via
  Rayon; GIL released with `py.detach()` before the parallel region
- **Native compiler optimization** — release builds can use the local CPU
  target configured under `.cargo/`.

Benchmark numbers should be regenerated on the release target hardware before
being used in public documentation.

### Fuzzy entropy sum

Computes Σ exp(−(d/r)ⁿ) over all upper-triangle pairs on mean-centered
templates, using the same Rayon parallel fold + reduce pattern.

## Algorithm Reference

| Algorithm | Module | Rust kernel | Reference |
|-----------|--------|-------------|-----------|
| Lyapunov exponent (1D) | `diagnostics.lyapunov` | — | Benettin et al. 1980 |
| Lyapunov spectrum (QR) | `diagnostics.lyapunov` | — | Benettin et al. 1980 |
| Flow Lyapunov spectrum | `diagnostics.lyapunov` | — | Benettin et al. 1980 |
| 0-1 test for chaos | `diagnostics.zero_one_test` | — | Gottwald & Melbourne 2004 |
| SALI / GALI | `diagnostics.sali_gali` | — | Skokos et al. 2007 |
| Permutation entropy | `diagnostics.permutation` | ordinal dist. | Bandt & Pompe 2002 |
| Complexity-entropy plane | `diagnostics.permutation` | ordinal dist. | Rosso et al. 2007 |
| Sample entropy | `diagnostics.entropy` | correlation counts | Richman & Moorman 2000 |
| Approximate entropy | `diagnostics.entropy` | — | Pincus 1991 |
| Fuzzy entropy | `diagnostics.entropy` | fuzzy sum | Chen et al. 2007 |
| Multiscale entropy | `diagnostics.entropy` | correlation counts | Costa et al. 2002 |
| RQA (DET, LAM, ENTR, …) | `diagnostics.recurrence` | line extraction | Marwan et al. 2007 |
| Correlation dimension | `diagnostics.correlation` | all-pairs kernel | Grassberger & Procaccia 1983 |
| Multifractal spectrum ($D_q$, $f(\alpha)$) | `diagnostics.multifractal` | multifractal moments | Mukherjee et al. 2024 |
| AMI (embedding) | `diagnostics.embedding` | histogram | Fraser & Swinney 1986 |
| Cao's method | `diagnostics.embedding` | dimension selector only | Cao 1997 |
| False nearest neighbors | `diagnostics.embedding` | — | Kennel et al. 1992 |

`diagnostics.recurrence` keeps `recurrence_matrix()` for callers that need the
binary matrix. For large trajectories where only scalar RQA measures are needed,
use `rqa_from_trajectory()` to avoid materializing the dense recurrence matrix.
When starting from an existing recurrence matrix, compute public RQA metrics
through `rqa()`, which validates that the matrix is non-empty, square, and
symmetric. The direct line extractors, including the Rust-accelerated helpers,
are lower-level square-matrix scanners and do not replace that public RQA
validation boundary.

## Reproduction gallery

The `figures/` tree holds a section-indexed set of reproductions of Kunihiko
Kaneko's published work on circle maps, torus doubling, fractalization,
coupled map lattices, and globally coupled maps. Each section is regenerated
from scratch by the same public entry points users call:

```bash
dynachaos list
dynachaos run sec02_circle_map
dynachaos run all
```

The committed `.npz` caches double as golden data for the reproducibility and
determinism tests, so the gallery is both a worked example and a standing
regression check. It is a demanding stress test of the package rather than its
boundary: for general use on your own signals, see the quickstart and recipe
gallery above.

Eight of the thirty-seven figures are shown below; click any of them for the
full-resolution render. **[Browse the complete gallery](https://openfluids.github.io/dynachaos/gallery.html)**
for all thirty-seven, in section order, with captions.

| | |
|---|---|
| [![Arnold tongues](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/thumbs/sec02_circle_map/arnold_tongues.webp)](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/sec02_circle_map/arnold_tongues.png) | [![Torus doubling in Map IV](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/thumbs/sec04_doubling/map_IV_attractors.webp)](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/sec04_doubling/map_IV_attractors.png) |
| Rotation number over the circle-map parameter plane; the uniform wedges are frequency-locked Arnold tongues. | Torus-doubling cascade of Map (IV): a fourfold torus, an eightfold torus, and chaos. |
| [![Double devil's staircase](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/thumbs/sec06_three_torus/double_staircase.webp)](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/sec06_three_torus/double_staircase.png) | [![Torus fractalization](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/thumbs/sec07_fractalization/fractal_attractors.webp)](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/sec07_fractalization/fractal_attractors.png) |
| Double devil's staircase of the modulated circle map, showing locking plateaus in both rotation numbers. | A smooth torus in the delayed logistic map develops wrinkles at finer and finer scales as it fractalizes. |
| [![Spatiotemporal intermittency](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/thumbs/sec08_sti/spacetime_diagrams.webp)](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/sec08_sti/spacetime_diagrams.png) | [![Globally coupled map clusters](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/thumbs/sec10_gcm/gcm_clusters.webp)](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/sec10_gcm/gcm_clusters.png) |
| Spacetime diagrams of spatiotemporal intermittency in three coupled map lattice models. | Cluster states in a globally coupled map, showing a partially ordered regime. |
| [![Complexity-entropy plane](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/thumbs/sec11_diagnostics/complexity_entropy_plane.webp)](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/sec11_diagnostics/complexity_entropy_plane.png) | [![Type-I intermittency](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/thumbs/sec12_intermittency/type_i_intermittency.webp)](https://raw.githubusercontent.com/openfluids/dynachaos/main/figures/sec12_intermittency/type_i_intermittency.png) |
| Complexity-entropy plane locations for the logistic and delayed logistic maps. | Type-I intermittency: a tangent-bifurcation channel, laminar-length statistics, and a Lorenz reinjection channel. |

## Live figures

Six figures on the paper page can be computed in the browser by the same
WebAssembly kernels the published figure used. The PNG is what the page
shows first, and what a reader without JavaScript gets. Press **interact**
to compute live; press **image** to return to the PNG.

That work does not start on its own in two cases. A hidden tab pauses new
tiles and new kernel calls, and resumes them when the tab is visible again.
A viewport below 480 CSS px — a phone in portrait — keeps the PNG until the
reader presses interact. `prefers-reduced-data` does not download the
WebAssembly module at all.

Telemetry (per-tile milliseconds, worker count, tiles in flight, and a one-line
debug report with a copy button under each live figure) is off unless the URL
has `?debug=1`, the hash contains `#live-debug`, or `localStorage.dynachaosDebug`
is set. The PNG and the committed `.npz`
caches stay the record of what the paper computed; a live view is a
companion, not a replacement. See
[docs/wasm-architecture.md](docs/wasm-architecture.md).


## Benchmarks

The reproducible benchmark for Rust Grassberger-Procaccia parity and dense
recurrence/RQA memory limits lives in `benchmarks/scale_envelope.py`.
The local benchmark command is
`uv run python benchmarks/scale_envelope.py benchmarks/scale_envelope.jsonc`;
inspect `benchmarks/results/scale_envelope.{json,md}` after it runs. At N=1000
the Rust Grassberger-Procaccia kernel runs about an order of magnitude faster
than the Python fallback for the largest common logistic case; the harness
times one un-warmed call per case, so the exact multiple it prints varies with
machine load and core count and should not be quoted as a fixed figure (see
`docs/claims-checklist.md`). The benchmark also predicts a dense-RQA
distance-matrix cost of 8*N^2 bytes, which becomes impractical near N≈23170
under the default cap. The measured Rust acceleration roadmap and local hotspot profiler are
documented in `docs/rust-acceleration-roadmap.md` and
`benchmarks/rust_hotspot_profile.py`.

## Development

These commands are for working on dynachaos itself. To *use* the package, see
Quick Start above.

```bash
git clone https://github.com/openfluids/dynachaos.git
cd dynachaos
uv sync
uv run --extra viz pytest tests/ -q
uv run --extra viz ruff check src/ tests/ scripts/
uv run --extra viz ruff format --check src/ tests/ scripts/
```

To exercise the installed Rust extension locally:

```bash
uv run maturin develop --release
uv run --extra viz pytest tests/ -q
```

To verify the pure-Python fallback path:

```bash
DYNACHAOS_NO_RUST=1 uv run --extra viz pytest tests/ -q
```

To rebuild the gallery thumbnails and the static gallery site:

```bash
uv run --extra viz python scripts/build_gallery.py
```

To build the WebAssembly bundle that the live figures use:

```bash
rustup target add wasm32-unknown-unknown   # once
uv run python scripts/build_wasm.py
```

This compiles the kernels in `rust/core` for the browser and writes
`site/wasm/`. The script installs the `wasm-bindgen` command line tool at the
version `rust/wasm/Cargo.lock` resolves to; pass `--no-install` to make a
mismatch an error instead. A live figure therefore computes with the same code
the published figure used — see
[docs/wasm-architecture.md](docs/wasm-architecture.md) for the contract those
kernels follow and what the browser build does and does not guarantee.

## Contributing

Contributions are welcome, and questions and bug reports count. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup and the checks CI runs, and the
[openfluids Code of Conduct](https://github.com/openfluids/.github/blob/main/CODE_OF_CONDUCT.md)
for how we work together.

## Citation

If you use dynachaos in published work, please cite the software:

```bibtex
@software{dynachaos2026,
  author  = {Frantz, Ricardo},
  title   = {dynachaos: Dynamical-systems analysis for time signals with Rust-accelerated kernels},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/openfluids/dynachaos},
  license = {Apache-2.0}
}
```

## License

This project is licensed under Apache-2.0.

Originally developed by Ricardo A S Frantz. See `LICENSE` and `NOTICE` for
license terms and attribution notices. As of v0.3.0, this project is licensed
under Apache-2.0; earlier (unpublished) versions were MIT.

## Disclaimer

This software is provided "as is", without warranty of any kind. To the extent
permitted by law, the authors and contributors are not liable for any damage, loss
or claim arising from its use or misuse. You are responsible for how you use it and
for following the laws and rules that apply to you. The full terms are in
[LICENSE](LICENSE).

This is research software. It has not been validated for engineering design,
certification or safety-critical use. Check its results independently before you
rely on them.
