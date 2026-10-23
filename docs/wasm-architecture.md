# Running the kernels in a browser: architecture contract

This note records the decisions for computing figures in the reader's browser
with the same Rust kernels the paper used. It is written before most of the
implementation, so that the kernels are built to a settled contract instead of
one reverse-engineered afterwards. Update it when a decision here changes.

## Why do this at all

The gallery ships pre-rendered PNGs plus decimated JSON (`scripts/export_figure_data.py`
writes `site/data/`). A reader can look at a parameter plane but cannot zoom into
it, and cannot move a parameter and watch the attractor change. Both are cheap
to compute and expensive to ship as data: the Arnold-tongue plane is a 2000x1000
grid today, and any zoom deeper than that grid has no data behind it.

Compiling the kernels to WebAssembly removes the shipping step. The reader's
machine recomputes the visible region at the resolution of their screen. The
same code that produced the published figure produces what they explore, so the
two cannot drift.

## Crate layout

The split landed in Phase 0 (2026-09).

| crate | contents | depends on |
|---|---|---|
| `rust/core` | every kernel; slices in, plain Rust out; no pyo3, no numpy | ndarray, rayon (optional) |
| `rust/py` | `#[pyfunction]` wrappers, numpy conversion, `py.detach` around core calls | `dynachaos-core`, pyo3, numpy |
| `rust/wasm` | `wasm-bindgen` exports over `rust/core`; `rotation_number_tile` for the picture, `rotation_number_point` for the quoted readout, `zero_one_k` for the 0-1 test for chaos, `correlation_counts`, `apen_counts`, `fuzzy_entropy_sum`, `ordinal_distribution`, `diagonal_lines`, `vertical_lines`, `multifractal_moments`, `ami_histogram`, `select_dimension_cao`, `delayed_logistic_attractor_tile`, `torus_doubling_attractor_tile`, `modulated_circle_rotation_tile`, `cml_spacetime_tile`, `circle_map_lyapunov_sum`, `delayed_logistic_lyapunov_tile`, `torus_doubling_lyapunov_tile` and `coupled_delayed_lyapunov_tile` for the diagnostics, map and Lyapunov panels; `coupled_logistic_phase_tile`, `coupled_logistic_attractor_tile`, `coupled_logistic_basin_grid`, `coupled_delayed_projection_tile` and `fractalization_attractor_tile` are exported (no page calls them yet) | `dynachaos-core` with `--no-default-features` |

`rust/core` carries 1491 lines of kernel code; `rust/py` carries 629 lines of
binding with no arithmetic in it. Keeping the arithmetic in exactly one place is
the point of the split: a browser and a Python caller cannot disagree about a
number if only one implementation exists.

`rust/wasm` gets its own `[workspace]`, so a browser build never depends on the
state of the native build. This follows the FrankenSim layout
(`crates/fs-wasm` in github.com/Dicklesworthstone/frankensim), which is the
closest published example of shipping real Rust numerics to a browser.

### rayon is optional

`rust/core` declares `rayon` as an optional dependency behind a `parallel`
feature, `default = ["parallel"]`. The Python extension keeps the default and
runs parallel; a wasm build compiles with `--no-default-features` and takes a
sequential path. `cargo build -p dynachaos-core --no-default-features` is
checked in CI so this path cannot rot.

Both paths call the same per-item function. Only the accumulation differs: the
parallel path folds per-thread partial sums and reduces them pairwise, the
sequential path adds left to right. **Floating-point addition is not
associative, so the two can differ in the last digit for kernels that sum
floats** (`fuzzy_entropy_sum`, the correlation sums). This is inherent to
parallel reduction, not a defect, and it is why the parity check below states a
tolerance for those kernels instead of demanding equality.

## The kernel contract

Every function exported to the browser obeys these rules. They come from the
FrankenSim `fs-wasm` contract and exist because a web page is a hostile caller:
its arguments come from a URL, a slider, or a stranger.

1. **Clamp every input.** No argument is trusted. A grid size, an iteration
   count and a parameter value each get a documented valid range and are
   clamped into it, not rejected.
2. **Cap every loop.** An iteration count has a hard ceiling. A browser tab that
   locks up is worse than a coarse picture.
3. **Return one flat `Vec<f64>`** with the layout written in the doc comment,
   including where each block starts and what the first elements mean. The
   JavaScript side reads that layout; there is no shared struct.
4. **No allocation surprises.** Size the output from the clamped inputs so the
   caller can predict the buffer.

`zero_one_k` is the exception to "cap every loop": its cost is `n_c` FFTs of
a power-of-two length below `4N`, so the `N <= 20000` and `n_c <= 100`
truncations bound the work on their own and no pair budget is needed. Its
clamps are `phi` to 20000 samples, `c_values` to 100 frequencies, and
`n_cut` to `[2, N]` — a request below 2 or above `N` comes back clamped,
never refused, and the header reports the value used. A series shorter than
three samples cannot produce a statistic and returns an empty array.

## Threads: worker tiling, not SharedArrayBuffer

WebAssembly threads put the linear memory in a `SharedArrayBuffer`, which
browsers only expose to a cross-origin isolated page. Isolation requires the
`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` response
headers, and **GitHub Pages cannot set response headers**. The gallery is
served from GitHub Pages.

Decision: parallelism comes from a pool of independent web workers, each
holding its own wasm instance and computing a tile of the domain. Tiles need no
shared memory because they share no state. This also keeps the main thread free,
so the page never freezes while a figure computes.

| option | works on GitHub Pages | cost | decision |
|---|---|---|---|
| `SharedArrayBuffer` + `wasm-bindgen-rayon` | no, headers cannot be set | — | reject |
| `coi-serviceworker` (service worker forges the headers) | yes | a reload on the visitor's first arrival | keep as a later upgrade, not now |
| pool of independent workers, one tile each | yes | tiles must be independent | **choose** |

The service-worker route is real and documented, and it is the way in if a
figure ever needs genuinely shared memory. It costs the visitor a page reload
the first time they arrive, which is a poor trade for figures that tile cleanly.

### Live JS runtime

The worker pool is hand-written ESM in `site-src/live/`, copied verbatim by
`scripts/build_paper.py` into `site/live/`. No bundler, no npm. Seven files:

| file | job |
|---|---|
| `scheduler.js` | Pure function: state in, commands out. No DOM, no Worker, no timer. Node unit-tests this. `lockOmega` switches it to a 1-D mode for the staircase: tiles carry `nOmega = 1` at the viewport's fixed Omega and the pyramid refines K only. |
| `pool.js` | `N = min(navigator.hardwareConcurrency, 8)` workers, each with its own wasm instance. A pan or zoom bumps the generation; workers stay alive so the in-flight tile can finish, and a late result is dropped. `terminate()` runs only in `destroy()`. |
| `tile-worker.js` | Loads `site/wasm/dynachaos_wasm.js`, calls `rotation_number_tile`, transfers the `Float64Array` back. Reads the 4-element header rather than trusting the request. |
| `raster.js` | Pure tile→pixel mapping, `liveTileColorKey`, and the tile lookup used as the readout fallback. Row 0 of a tile is `kMin`, the bottom; the canvas y axis points down. Node unit-tests this. |
| `point.js` | Main-thread lazy load of the wasm glue; synchronous `sample()` of one (Omega, K) point with lock detection on and the display stop off. Hover and keyboard readout share this path. |
| `live-figure.js` | The glue mountLive calls: the onPaint store update with its paint-sequence stamp, the readout fallback (point kernel, else the raster lookup), the onView generation reset, and the staircase's 1-D pieces (view mapping, tile→trace fold, y-fit, parameter wiring). No DOM at import time; node unit-tests this. |
| `params.js` | Pure parameter plumbing: spec clamping, debounce, and URL-hash serialise/parse (`#section&fig:<id>.<name>=<value>`). Node unit-tests this. |

The paper page loads `pool.js` from `app.js` with a dynamic `import()` when the
reader presses interact on `figure#fig:arnold_tongues`. `Plot()` keeps axes,
colourbar, the K_c line, readout, drag zoom, keyboard and reset; only the heatmap
raster comes from the pool. Colour scale is fixed at ρ ∈ [0, 1]. Live mode
replaces the JSON chart; it never fetches it. Under `prefers-reduced-data` the
page does not create the pool or download the wasm module, and falls back to
the JSON chart if present.

Tiles cover the viewport at a pyramid of levels. Level 0 is one tile over the
whole view; each finer level splits 2×2. Coarser levels use fewer cells: the
finest level uses `tileCells`, and each coarser level halves that count, so
the first paint is one cheap tile on one worker. The live Arnold-tongues
figure uses five levels. The scheduler issues every coarser tile before any
finer one, and issues each visible tile exactly once per generation. A
viewport change increments the generation; workers are not terminated, and
results that carry an older generation are dropped, not painted. In-flight
bookkeeping is `{ id, generation }`, so a late result cannot evict the
current generation's entry for the same tile. Degenerate viewports
(`omegaMax <= omegaMin` or `kMax <= kMin`) are rejected. An error reply, a
worker-level failure, or a malformed worker message frees its tile and is
counted as a drop, not left in flight. A failed worker is not handed more work.
Live tiles are coloured once; redraw blits the cached ImageData. The cache
key is `liveTileColorKey` in `raster.js` and does not include the page theme,
because live colour is a fixed viridis ramp.

Debug telemetry (`?debug=1`, `#live-debug` in the hash, or
`localStorage.dynachaosDebug`) logs per-tile compute milliseconds, worker
count, queue depth, dropped generations (viewport changes that abandoned
work), and dropped tiles (stale or error results), and shows a copyable one-line
report under each live figure (figure id, short user agent, worker count,
time to first full paint, median tile time, and main-thread tasks over 50 ms).
It is off by default.
A hidden tab pauses new tile issue in `pool.js` and new kernel calls in
the shared runner (`visibility.js`, used by every direct-call figure).
Work already in flight may finish; a result whose generation is no longer
current is dropped, not painted. Below 480 CSS px the page does not mount
live mode from a shared link; the interact button still does. There is no
`SharedArrayBuffer` anywhere in this path.

The devil's staircase (`figure#fig:devils_staircase`) is the second live
figure and the first 1-D one: ρ(A) at a reader-chosen drive frequency D,
drawn as a line with `Plot()`'s axes and readout rather than a heatmap. A
slider (range + number field, keyboard accessible) sets D ∈ [0, 0.5],
default 0.25; a debounced apply pushes `{omegaMin: D, omegaMax: D, kMin,
kMax}` into the pool — the scheduler's `lockOmega` mode emits `nOmega = 1`
tiles, so Omega = D and K = A exactly — and the generation drop clears the
stale curve. Painted tiles fold into one sorted polyline; the first paint
of a generation refits the y domain. The readout quotes `point.sample(D, A)`
at the cursor's A. The Lyapunov panel has no wasm kernel yet, so the
published PNG stays below the live curve with a note saying it is not live.
Parameter state serialises into the URL hash (`#section&fig:<id>.D=<v>`) so
a shared link restores the exact view; the scroll-spy preserves the
`&key=value` tokens when it rewrites the section id. Under
`prefers-reduced-data` no slider is created and the PNG stays.

The delayed-logistic attractors (`figure#fig:delayed_logistic_attractors`)
are the third live figure and the first point cloud: the (x, y) orbit of
the delayed logistic map at a reader-chosen D ∈ [1.4, 3.5] (the kernel's
own clamp), A = 0.3 fixed, with an iteration-count control bounded by the
kernel's 4096 plotted states. It does not use the tile pool: one
`delayed_logistic_attractor_tile` call per parameter set is at most
20000 + 4096 map steps, microseconds of work, so the worker round trip
would buy nothing. The debounce collapses a slider burst into one call and
a sequence number drops a stale result. The start state is the published
convention (fp + 0.01, fp − 0.01) at the analytic fixed point, which the
canvas marks in vermilion over the slate cloud. The map uses only +, −, *,
so the wasm orbit is bit-identical to the npz trajectories;
`scripts/check_wasm_delayed_logistic.py` asserts exactly that at every
committed D in the wasm-parity CI job. Parameter state rides the URL hash
like the staircase's; under `prefers-reduced-data` no slider is created
and the PNG stays.

The torus-doubling attractors (`figure#fig:map_I_attractors`) are the
fourth live figure: the (X, Y) projection of map (I) or map (IV) at a
reader-chosen D, drawn by `torus_doubling_attractor_tile` on the main
thread like the delayed-logistic cloud. A map selector — the page's first
non-slider control, a `<select>` — switches between map (I) (A = 0.4,
x0 = (0.5, 0.5, 0.5), D ∈ [1.9, 2.25]) and map (IV) (A = 0.3,
x0 = (0.5, 0.45, 0.52, 0.48), D ∈ [1.48, 1.53]); each window is the
published sweep inside the kernel's [1.48, 2.25] clamp, and switching maps
resets D to the new map's published default rather than clamping the old
value into the new window. An iteration-count control is bounded by the
kernel's 4096 plotted states. Both maps project the state onto (X, Y) —
components 0 and 1 — on one fixed domain, the union of both committed npz
extents, so the selector compares the maps on the paper's axes. The maps
use only +, −, *, so the wasm orbit is bit-identical to the npz
trajectories; `scripts/check_wasm_torus_doubling.py` asserts exactly that
at every committed D in the wasm-parity CI job. Parameter state — the map
included — rides the URL hash like the staircase's; under
`prefers-reduced-data` no controls are created and the PNG stays.

The double devil's staircase (`figure#fig:double_staircase`) is the fifth
live figure and the second 1-D one: rho_theta and rho_phi of the modulated
circle map against the bare frequency D, drawn by
`modulated_circle_rotation_tile` on the main thread like the attractor
clouds. A = 0.1 and C = (sqrt(5) - 1)/2 stay at the paper's values; the
reader moves eps in [0, 0.2] (default 0.05), a point count bounded by the
kernel's 512, and the D window itself — min/max fields, two preset buttons
that reproduce the zoom panels' windows (the npz's longest plateaus near
rho_theta = 1/4 and rho_theta = C, padded as `plot_zoom` pads them), or a
drag zoom, since the plot's x domain is the compute window. The dashed
rho_theta = D diagonal is the paper's rigid-rotation reference. The kernel
returns (rho_theta, rho_phi) pairs after a 4-value header; the fold splits
them into two curves sharing one D axis. Because A = 0.1 < 1/(2 pi) every
fibre map is an orientation-preserving diffeomorphism, so the wasm sine's
one-ulp difference cannot amplify: `scripts/check_wasm_modulated_circle.py`
holds rho_theta to 1e-6 and rho_phi to 1e-12 at the npz's exact D values —
five grid entries plus every D inside both zoom windows — in the
wasm-parity CI job. Parameter state rides the URL hash like the other live
figures; under `prefers-reduced-data` no controls are created and the PNG
stays.

The CML space-time diagrams (`figure#fig:spacetime_diagrams`) are the
sixth live figure and the first field: the lattice state x_i^n of the
chosen model, site i across and time n up, drawn by
`cml_spacetime_tile` on the main thread like the attractor clouds. A
model selector switches between models A, B and C; each model's eps
slider window is its published sweep's range (A [0, 0.2] default 0.07,
B [0, 0.1] default 0.024, C [0, 0.5] default 0.2) and a site-count
control sets N in [16, 512] (default 200). The colour limits are the
shown field's 1st and 99th percentiles, the same rule `plot()` applies
per row, and the ramp is magma — the paper's `CMAP_SPACETIME`. A play
button appends 250-row chunks computed from the window's last row with
n_transient 0 and drops the oldest rows, so the window is a rolling
500-row view of one continuing orbit. The initial field is not
reproduced in JS — PCG64 is not cheap there — so `build_paper.py` writes
the first 512 values of `default_rng(42).uniform(0, 1, 512)` to
`site/live/cml-x0.json` and the request slices it to N. At N = 200,
`scripts/check_wasm_cml_spacetime.py` compares all nine npz fields —
bit for bit for A and C, and for B against eight numpy runs with every
sin result moved one ulp (sample-by-sample within 1e-9 when those runs
stay on the npz orbit, a value-histogram L1 no larger than the worst
run's when they leave it, as at eps 0.024) — checks the x0 JSON
against the paper's seed-42 field, and proves a play chunk continues
the run bit for bit, in the wasm-parity CI job. Parameter state rides the
URL hash like the other live figures; under `prefers-reduced-data` no
controls are created and the PNG stays.

## WebGPU is a fast path, never the baseline

As of 2026-09: Chromium ships WebGPU (113+, and Android 121+), Safari turned it
on by default in Safari 26 (2025-09), and **Firefox does not ship it in the
release channel**. A figure that only works in Chromium is not acceptable for a
paper.

Decision: CPU kernels in WebAssembly drawing to a canvas are the baseline and
must be sufficient for every live figure. WebGPU may be added later for a
figure that genuinely needs it, always behind a capability check with the CPU
path as the fallback.

## The PNG stays canonical

Every live figure keeps its published PNG, and the PNG is what the document
means.

- The PNG paints first, so the reader sees the figure immediately.
- Live mode is opt-in per figure and replaces the canvas contents afterwards.
- Live mode is off under `prefers-reduced-data` (the page already reads it) and
  absent without JavaScript.
- A figure gets exactly **one** interactive mode. Where a figure already has a
  decimated-JSON chart, live mode replaces that chart; the two never stack.

The archived reproduction pipeline and the committed `.npz` caches stay the
record of what the paper computed. Live views are companions to it.

## Proving the browser agrees with the paper

A live figure is only worth showing if its numbers are the paper's numbers. CI
checks this directly: a small fixed tile is evaluated by the native `rust/core`
test binary and by the wasm module, and the two are compared element by element.

Build the native reference **without** `target-cpu=native`. The repository's
`.cargo/config.toml` enables it for local builds, and it lets the compiler
contract multiplications and additions into fused instructions that `wasm32`
does not have. Comparing against a natively-tuned build measures the compiler,
not the port.

### The two builds cannot agree everywhere, and that is physics

Measured on 2026-09-08 over a 64 x 64 tile of the Arnold-tongue plane
(`scripts/check_wasm_parity.py`):

| comparison | cells differing at all | worst difference |
|---|---|---|
| NumPy vs native Rust | 0 of 4096 | 0 |
| native Rust vs wasm, `K <= 0.159` | 2 of 2176 | 7.8e-16 |
| native Rust vs wasm, `K > 0.159` | 15 of 1920 | 2.9e-2 |

The check itself prints a smaller number above the critical line, **13 of
1920**, because it counts a cell only past a `1e-12` floor. The other two
differ by less than that, which is the same orbit through a slightly different
sine rather than a diverged one.

NumPy and native Rust agree exactly because they call the same platform sine.
The wasm module carries its own sine, compiled into it, and that one differs by
about one unit in the last place. Below `K = 1 / (2 * pi)` the map is
invertible and the rotation number is well conditioned, so a one-bit difference
stays a one-bit difference. Above it the map is not invertible, the orbit is
chaotic, and the same one bit grows by fourteen orders of magnitude over 700
iterations.

So the honest claim is narrower than "the browser returns the paper's
numbers":

- **Where the orbit locks, the rotation number is exact** — the kernel returns
  the rational `p / q` as soon as the unwrapped orbit closes. Those regions
  are the subject of the figure; the tongues are exactly where the reader is
  looking.
- **The picture is accurate to the displayed colour resolution** (one step in
  256). The tile kernel stops once its running estimate is provably within
  half that tolerance. The colour a reader sees never promises more.
- **The quoted readout does not carry that display tolerance.** Hover and
  keyboard sample one (Omega, K) point with lock detection on and the display
  stop off: the exact rational when the orbit locks, the full `n_iter`
  average otherwise. Until that main-thread module is ready, the readout
  falls back to the tile sample.
- **In the chaotic sea, a single pixel is still not reproducible** across sine
  implementations. It is not reproducible across compilers or CPUs either.
  That is a property of the system, not of this port, and a figure that showed
  otherwise would be hiding it.

`scripts/check_wasm_parity.py` asserts each population separately: locked cells
are held to the existing subcritical bound at any `K`; display-stop cells are
held to `1 / 256`; exhausted cells below the critical line stay on the
subcritical bound; exhausted cells above it keep the chaotic share limit.
Never loosen the subcritical bound to make a red run pass — it still binds
every locked cell. A deliberate `1e-7` perturbation of a tile value trips the
assertions; the script prints a self-check line so that trip is known to work
rather than assumed to.

This is also why a live figure must not be presented as a re-derivation of the
published pixel values in the chaotic region. It shows the same system computed
by the same code; it does not promise the same last digits where the system
itself refuses to have any.

## Checking the live figure in CI

Parity of one tile is not the live path. The live path is: the paper page,
the interact button, a pool of workers, the coarse-to-fine pyramid, a view
change, the rotation numbers at the tongues, and `prefers-reduced-data`.
That check lives at `tests/e2e/live_figure.mjs` and CI runs it.

**Driver: node plus the Chrome DevTools protocol.** No npm, no `package.json`,
no bundler, no playwright, no new dependency of any kind. The reasons, checked
rather than assumed:

- `ubuntu-latest` already carries node and Chrome. The `wasm-parity` job
  already depends on node being on the image and says so in a comment.
- A working driver of exactly this shape has caught real defects: it serves
  the built site with `python3 -m http.server` and drives
  `google-chrome --headless=new` over the DevTools protocol using only node
  built-ins (`node:child_process`, `node:net`, `node:fs`) plus the global
  `WebSocket` and `fetch` that node 22 provides.
- playwright-python would add a dev dependency and a browser download to get
  capabilities this repository does not need.

figure's chart JSON, then runs `node --test tests/js/index.js tests/js/raster.js tests/js/point.js tests/js/live-figure.js tests/js/params.js`
and the end-to-end script. A missing Chrome fails the job; the script prints
the command it tried and exits non-zero. The job does not skip the browser
check.

`scripts/check_wasm_staircase.py` is the staircase's parity sibling, run in
the wasm-parity CI job: the wasm tile at D = 0.25 against the committed
`devils_staircase.npz` rho at exactly the npz A values (tile K ranges chosen
so their linspace reproduces npz points bit for bit, asserted before the
comparison). Below K_c every point stays within the display tolerance; above
it divergence is allowed but bounded by the same share limit.

`scripts/check_wasm_delayed_logistic.py` is the attractor figure's parity
sibling in the same job: the wasm tile at each committed D in
`attractors.npz` and `locking_sequence.npz` against the first 4096 plotted
states, same x0 convention and n_transient 20000. The delayed logistic map
has no transcendental call, so the rule is bit-exact equality — a
difference means the kernel's operation order drifted, not a tolerance to
negotiate.

`scripts/check_wasm_torus_doubling.py` is the torus figure's parity
sibling in the same job: the wasm tile at each committed D in
`map_I_attractors.npz` and `map_IV_attractors.npz` against the first 4096
plotted states, every state component, same x0 and n_transient 20000. The
torus-doubling maps have no transcendental call either, so the rule is
again bit-exact equality.

## Measuring performance honestly

Kernel timings in this project are small (a Grassberger-Procaccia pass at
N=1000 takes under a millisecond) and easy to mismeasure. Two effects dominate:

- **Cold start.** A single un-warmed call measures thread-pool spin-up, first
  page touches and a cold cache. Warm up, then take a median of repeats.
- **Oversubscription.** rayon defaults to one thread per core. On the 24-core
  development box the same kernel runs 0.67 ms on 8 threads and 0.88 ms on 24,
  because the threads contend. More threads is not more speed for a workload
  this short.

Measured on the development box (median of 15, after 3 warm-up calls,
Grassberger-Procaccia, logistic, N=1000):

| threads | 1 | 2 | 4 | 8 | 24 |
|---|---|---|---|---|---|
| time (ms) | 2.68 | 1.38 | 0.81 | **0.67** | 0.88 |

A browser has far fewer workers than this box has cores, so the useful figure
for planning a live figure is the 2-to-8-thread range, not the single-thread
number and not the 24-thread number.

## Performance headroom

Measured 2026-09-24 on the development box (AMD Ryzen 9 9900X, 12 cores / 24
threads, 15.8 GB RAM, shared with other projects), node v22.19.0, Chrome for
Testing 142.0.7444.175 headless. The 1-minute load average was recorded before
and after each timing series; a series was repeated, up to three times, when
the load exceeded 2.0 before it started.

### SIMD: bit-identical, not faster

A second wasm build with `RUSTFLAGS="-C target-feature=+simd128"` (the flag
took effect: the module carries 407 `0xFD`-prefixed SIMD opcodes against 19
in the scalar build) ran `rotation_number_tile` on the flagship's default
view — a 64 x 64 tile over Omega in [0, 1], K in [0, 0.3], `nTransient` 200,
`nIter` 2000, `theta0` 0.1 — 3 warm-ups, median of 15, in node. Load
average 1.86 before the series, 2.73 after.

| build | median (ms) | min-max (ms) |
|---|---|---|
| scalar (the shipped `site/wasm` build) | 43.38 | 42.92-45.06 |
| simd128 | 43.01 | 42.81-43.56 |

Every returned f64 of the tile — all 4100 values, header included — is
bit-identical between the two builds. The speed ratio is 1.01, under the
1.3 needed to adopt it, so **SIMD is not adopted**. The kernel is a
sequential recurrence: each circle-map iteration needs the previous state,
so the hot loop has no independent values to put in SIMD lanes.

The SIMD build:

    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_DIR=target-simd \
      cargo build --manifest-path rust/wasm/Cargo.toml \
        --target wasm32-unknown-unknown --release
    wasm-bindgen --target web --out-dir simd-pkg \
      target-simd/wasm32-unknown-unknown/release/dynachaos_wasm.wasm

### Workers: faster, but well under linear

First full paint of the flagship (`fig:arnold_tongues`) in headless Chrome,
from the click that starts live mode until every tile is painted: 341 tiles
(levels 0-4, 1 + 4 + 16 + 64 + 256 tiles; the finest level is 62 x 62
cells). `navigator.hardwareConcurrency` was overridden for each run;
`pool.js` caps the pool at 8. Three series were run; the table gives the
first, with the full-paint times of the other two in parentheses. Load
average before the series: 0.89 / 1.68 / 1.52; after: 1.87 / 2.79 / 3.56.

| workers | first tile (ms) | full paint (ms) | tiles/s | sum of per-tile compute (s) |
|---|---|---|---|---|
| 1 | 90 | 10820 (13673, 13787) | 31.5 | 10.2 |
| 2 | 57 | 10122 (10144, 10170) | 33.7 | 19.4 |
| 4 | 73 | 4536 (4508, 4606) | 75.2 | 17.1 |
| 8 | 61 | 3415 (3428, 3443) | 99.9 | 25.6 |

The first tile arrives in 60-90 ms at every pool size, so one worker is
enough for the first response. Full paint is 3.2 times faster with 8
workers than with 1. The summed per-tile compute time grows with the pool
(10.2 s with 1 worker, 25.6 s with 8): each tile runs slower when more run
at once. These measurements do not separate the causes (other projects'
jobs on the same box, shared cache, clock speed under load). In the first
series two workers gave almost no gain over one (10.1 s against 10.8 s).
The gain from 4 to 8 workers is smaller than from 2 to 4.

The browser check in `tests/e2e/live_figure.mjs` reuses this full-paint
definition (click until the pool queue and in-flight set are empty). The
old 4x CPU-throttle wall-clock budget is removed. Wasm download and worker
start-up dominated that time, so a tile worker that calls
`rotation_number_tile` three times per tile still finished inside both old
limits (13155 ms on this box, 17868 ms on a GitHub runner). On this shared
box the throttle also stalled tile issue (26 tiles in 120 s) while
`computeMs` stayed near 29 ms, so a throttled wall clock does not measure
the kernel. What remains is one hang ceiling, 120000 ms, on every machine.
It only fails a paint that does not finish.

Kernel cost is a ratio from the first full paint, with one numeric limit and no
machine/CI branch. The numerator is the **25th percentile**, with linear
interpolation between sorted ranks, of per-tile `computeMs` for the finest-level
tiles from pool telemetry under `#live-debug`. The e2e also logs the p10 and p50
values and ratios; p10 was pre-registered as the only fallback if p25 failed
the margin criterion. The denominator is the median of five worker-timed
batches of exactly 20 calls each to
`circle_map_lyapunov_sum(0.12, 0.25, 2000, 200000, 0.1)` in a module `Blob`
worker importing `wasm/dynachaos_wasm.js` by absolute URL. One equally-sized
warmup batch is discarded. The worker's `performance.now()` times each complete
batch; there is no duration target. Twenty calls were calibrated to take about
100 ms per batch on the local unloaded box. Since the reference is fixed work
on worker wasm, load or a slower machine lengthens it along with the kernel
timing.

Known blind spot: a slowdown of all wasm at once moves both sides and is not
caught by the ratio. The local measurements support the 0.75 limit on this box;
they do not independently establish transfer to a different machine.

On 2026-10-06, final calibration bands (min / median / max) were:

| set | kernel p25 (ms) | reference (ms) | ratio |
|---|---:|---:|---:|
| unchanged, 10 runs | 40.075 / 41.288 / 41.950 | 96.900 / 99.250 / 100.700 | 0.4026 / 0.4152 / 0.4329 |
| four `nice -n 19` busy loops, 5 runs | 41.000 / 41.575 / 41.875 | 104.900 / 105.100 / 105.200 | 0.3901 / 0.3952 / 0.3984 |
| tile kernel called 3x, 5 runs | 122.125 / 125.200 / 126.275 | 97.900 / 98.400 / 99.700 | 1.2474 / 1.2689 / 1.2756 |

The one limit is **0.75**. It is at least 1.5x the largest unchanged or
loaded p25 ratio (`1.5 × 0.4329 = 0.6494`) and at most the lowest 3x p25
ratio / 1.5 (`1.2474 / 1.5 = 0.8316`). The margin window is
`[0.6494, 0.8316]`; the chosen limit is within it. The 2x ratios
(0.8156 / 0.8258 / 0.8569) all fail this limit. The 120000 ms hang ceiling
and other checks are unchanged.


Worker tiling only helps figures whose cells are independent. These
figures follow one orbit or one field forward in time and give no
independent tiles: `sec03_transition/attractors.png`
(`coupled_logistic_attractor_tile`), `sec04_doubling/map_I_attractors.png`
and `map_IV_attractors.png` (`torus_doubling_attractor_tile`),
`sec05_oscillation/attractors.png` and `locking_sequence.png`
(`delayed_logistic_attractor_tile`), `sec06_three_torus/xz_projections.png`
(`coupled_delayed_projection_tile`),
`sec07_fractalization/fractal_attractors.png`
(`fractalization_attractor_tile`), and `sec08_sti/spacetime_diagrams.png`
(`cml_spacetime_tile`; neighbouring sites couple at every step, so the
lattice cannot be split into independent tiles either). To run these
faster, the kernel itself needs threads, which need `SharedArrayBuffer` and
the COOP/COEP headers GitHub Pages cannot send; `coi-serviceworker` (see the
threads section) provides them at the cost of one reload on the first
visit.

### WebGPU: API present, no adapter

Headless Chrome 142 on this box exposes `navigator.gpu` on the served page,
but `requestAdapter()` returns `null`, with and without `--disable-gpu`.
There is no GPU behind the API here. CPU-wasm stays the baseline, the only
path every visitor can run.

### Main-thread task duration: long kernels off-main-thread

Measured 2026-10-16 on the development box (AMD Ryzen 9 9900X, 12 cores / 24
threads, 15.8 GB RAM), node v22.19.0, headless Chrome (390x844 viewport,
Chrome DevTools CPU throttle rate 4). Main-thread tasks exceeding 50 ms were
recorded using an in-page PerformanceObserver observing the `"longtask"` entry
type. Each of the six live figures was measured for (i) one slider change and
(ii) 5 s of continuous play where the figure provides a play control.

Before moving the kernels into web workers, both the double-staircase and
space-time diagrams executed their compute kernels on the main thread, causing
prolonged UI freezes on each parameter adjustment and repeated stutters during
space-time playback.

Before (kernels on main thread):

| figure | action | count (>50 ms) | longest task (ms) |
|---|---|---|---|
| arnold_tongues | slider change | 0 | 0.0 (no slider) |
| arnold_tongues | 5 s play | 0 | 0.0 (no play) |
| devils_staircase | slider change | 0 | 0.0 |
| devils_staircase | 5 s play | 0 | 0.0 (no play) |
| delayed_logistic_attractors | slider change | 0 | 0.0 |
| delayed_logistic_attractors | 5 s play | 0 | 0.0 (no play) |
| torus_doubling_attractors | slider change | 0 | 0.0 |
| torus_doubling_attractors | 5 s play | 0 | 0.0 (no play) |
| double_staircase | slider change | 1 | 490.0 |
| double_staircase | 5 s play | 0 | 0.0 (no play) |
| spacetime_diagrams | slider change | 1 | 180.0 |
| spacetime_diagrams | 5 s play | 35 | 179.0 |

After moving `modulated_circle_rotation_tile` and `cml_spacetime_tile` into a
web worker, long kernel calls no longer run on the main thread. A stale-result
rule ensures that results from superseded requests arriving out of order are
dropped without painting.

After (long kernels in worker):

| figure | action | count (>50 ms) | longest task (ms) |
|---|---|---|---|
| arnold_tongues | slider change | 0 | 0.0 (no slider) |
| arnold_tongues | 5 s play | 0 | 0.0 (no play) |
| devils_staircase | slider change | 0 | 0.0 |
| devils_staircase | 5 s play | 0 | 0.0 (no play) |
| delayed_logistic_attractors | slider change | 0 | 0.0 |
| delayed_logistic_attractors | 5 s play | 0 | 0.0 (no play) |
| torus_doubling_attractors | slider change | 0 | 0.0 |
| torus_doubling_attractors | 5 s play | 0 | 0.0 (no play) |
| double_staircase | slider change | 0 | 0.0 |
| double_staircase | 5 s play | 0 | 0.0 (no play) |
| spacetime_diagrams | slider change | 1 | 68.0 |
| spacetime_diagrams | 5 s play | 64 | 93.0 |

On the double-staircase figure, moving the kernel off the main thread eliminated
all main-thread tasks over 50 ms on slider adjustments. On space-time diagrams,
the longest main-thread task dropped from 180.0 ms to 68.0 ms on a slider move,
and during 5 seconds of play peak task length dropped from 179.0 ms to 93.0 ms,
ensuring every task remains below 100 ms under a 4x CPU throttle. The remaining
sub-100 ms tasks represent the 100,000-cell canvas raster blit rather than
kernel execution. The other four live figures remain on the main thread as
their kernel steps require only microseconds.

## Figure inventory for live mode

The class names the cheapest browser path that reproduces the figure: A has a
WASM export for the plotted quantity, B needs one bounded map kernel, and C
remains a canonical precomputed figure. `MAX_STEPS` is the per-call budget
defined in `rust/wasm/src/lib.rs:32`.

| section | png | module | class | interaction | kernel | caps |
|---|---|---|---|---|---|---|
| sec02_circle_map | devils_staircase.png | maps.circle_map | A | slider D and zoom A (rotation panel) | rotation_number_tile | A for the rotation panel; n_omega=1 and n_k<=512; Lyapunov panel stays PNG (circle_map_lyapunov_sum is exported but no page calls it yet) |
| sec02_circle_map | arnold_tongues.png | maps.arnold_tongues | A | zoom (Omega,K) | rotation_number_tile | n_omega,n_k<=512; n_cells*(transient+iter)<=MAX_STEPS |
| sec02_circle_map | staircase_zoom.png | maps.circle_map | A | zoom K window | rotation_number_tile | n_omega,n_k<=512; n_cells*(transient+iter)<=MAX_STEPS |
| sec03_transition | phase_diagram.png | maps.coupled_logistic | B | zoom (A,D) | coupled_logistic_phase_tile | viewport cells<=512; cells*(transient+sample)<=MAX_STEPS |
| sec03_transition | attractors.png | maps.coupled_logistic | B | A slider with iteration-count control | coupled_logistic_attractor_tile | n_A<=64,n_plot<=4096; n_A*(transient+plot)<=MAX_STEPS |
| sec03_transition | basins.png | maps.coupled_logistic | B | zoom initial-state plane | coupled_logistic_basin_grid | n_x,n_y<=512; cells*transient+reference_transient<=MAX_STEPS |
| sec04_doubling | map_I_attractors.png | maps.torus_doubling | B | D slider with iteration-count control | torus_doubling_attractor_tile | map I dim=3; n_plot<=4096; transient+plot<=MAX_STEPS |
| sec04_doubling | map_IV_attractors.png | maps.torus_doubling | B | D slider with iteration-count control | torus_doubling_attractor_tile | map IV dim=4; n_plot<=4096; transient+plot<=MAX_STEPS |
| sec04_doubling | map_IV_lyapunov.png | maps.torus_doubling | B | zoom D in the doubling window | torus_doubling_lyapunov_tile | n_D<=512; n_D*(transient+iter)<=MAX_STEPS |
| sec05_oscillation | attractors.png | maps.delayed_logistic | B | D slider with iteration-count control | delayed_logistic_attractor_tile | n_D<=64,n_plot<=4096; n_D*(transient+plot)<=MAX_STEPS |
| sec05_oscillation | lyapunov_vs_D.png | maps.delayed_logistic | B | zoom D and iteration-count control | delayed_logistic_lyapunov_tile | n_D<=512; n_D*(transient+iter)<=MAX_STEPS |
| sec05_oscillation | locking_sequence.png | maps.delayed_logistic | B | D slider through the locking window | delayed_logistic_attractor_tile | n_D<=64,n_plot<=4096; n_D*(transient+plot)<=MAX_STEPS |
| sec06_three_torus | lyapunov_vs_DB.png | maps.coupled_delayed | B | DB and epsilon sliders | coupled_delayed_lyapunov_tile | n_DB<=128; n_DB*(transient+iter)<=MAX_STEPS |
| sec06_three_torus | xz_projections.png | maps.coupled_delayed | B | DB slider with projection zoom | coupled_delayed_projection_tile | n_DB<=32,n_plot<=4096; n_DB*(transient+plot)<=MAX_STEPS |
| sec06_three_torus | double_staircase.png | maps.modulated_circle | B | D slider and zoom | modulated_circle_rotation_tile | n_D<=512; n_D*(transient+iter)<=MAX_STEPS |
| sec06_three_torus | double_staircase_zoom.png | maps.modulated_circle | B | zoom either locking window | modulated_circle_rotation_tile | n_D<=512; n_D*(transient+iter)<=MAX_STEPS |
| sec07_fractalization | fractal_attractors.png | maps.fractalization | B | D slider with iteration-count control | fractalization_attractor_tile | n_D<=32,n_plot<=4096; n_D*(transient+plot)<=MAX_STEPS |
| sec07_fractalization | correlation_dimension.png | maps.fractalization | C | none | precomputed | C: 200 parameters, 100,000 samples, and max_pairs=1,000,000 are paper-scale |
| sec08_sti | spacetime_diagrams.png | cml.spatiotemporal | B | play/pause and epsilon slider | cml_spacetime_tile | n_sites<=512,n_record<=2048; n_sites*(transient+record)<=MAX_STEPS |
| sec08_sti | comoving_lyapunov.png | cml.comoving_figure | C | none | precomputed | C: three 301-velocity scans at 100,000 iterations on N=500 exceed a tile budget |
| sec08_sti | correlation_decay.png | cml.correlation_figure | C | none | precomputed | C: four regimes plus 20,000-step Lyapunov-density diagnostics are a published cache |
| sec09_pattern | phase_diagram.png | cml.pattern_dynamics | C | none | precomputed | C: 160x200 phase sweep with 5,000 transient and 2,000 sample steps on N=100 |
| sec09_pattern | space_amplitude.png | cml.pattern_dynamics | C | none | precomputed | C: five long phase exemplars are static snapshots, not a bounded exploration |
| sec10_gcm | gcm_msd.png | cml.globally_coupled | C | none | precomputed | C: N reaches 20,000 and each series has 100,000 samples |
| sec10_gcm | gcm_distribution.png | cml.globally_coupled | C | none | precomputed | C: N reaches 20,000 and each series has 100,000 samples |
| sec10_gcm | gcm_clusters.png | cml.gcm_clusters | C | none | precomputed | C: cluster labels are a fixed phase exemplar with 20,000 transient and 500 record steps |
| sec11_diagnostics | test01_sweep.png | diagnostics.compare_all | C | none | precomputed | C: 500-parameter sweep, 5,000-series samples, and 50 frequencies per point |
| sec11_diagnostics | alignment_three_torus.png | diagnostics.alignment_three_torus | C | none | precomputed | C: four orbits, 1,000,000 iterates each, three seeds, with block-error Lyapunov spectra |
| sec11_diagnostics | agreement_roc.png | diagnostics.agreement_figure | C | none | precomputed | C: three 300 to 500 point parameter sweeps, each with a block-error Lyapunov spectrum and four diagnostics |
| sec11_diagnostics | permutation_entropy.png | diagnostics.compare_all | C | none | precomputed | C: 500 logistic plus 300 delayed-logistic parameter sweeps over 5,000 samples |
| sec11_diagnostics | complexity_entropy_plane.png | diagnostics.compare_all | C | none | precomputed | C: two 200-parameter sweeps over 5,000 samples feed a diagnostic plane |
| sec11_diagnostics | rqa_measures.png | diagnostics.compare_all | C | none | precomputed | C: 80 parameters each require a 2,000-point recurrence matrix |
| sec12_intermittency | type_i_intermittency.png | diagnostics.intermittency_figure | C | none | precomputed | C: 200,000-point tail, bootstrap fits, normal-form scaling, and a Lorenz return map |
| sec12_intermittency | on_off_intermittency.png | diagnostics.on_off_intermittency_figure | C | none | precomputed | C: benchmark orbit, 100,000-point scaling, and bootstrap fits are proof diagnostics |
| sec12_intermittency | type_ii_intermittency.png | diagnostics.type_ii_intermittency_figure | C | none | precomputed | C: stochastic reinjection proof data and bootstrap fits are published values |
| sec12_intermittency | type_iii_intermittency.png | diagnostics.type_iii_intermittency_figure | C | none | precomputed | C: stochastic reinjection, escape episodes, and bootstrap fits are proof data |
| sec12_intermittency | sti_spine.png | cml.sti_spine_figure | C | none | precomputed | C: 512-site spacetime plus a nine-point coupling sweep and cluster fit are a diagnostic cache |

## Live figure checklist

Six figures on the paper page are live. Each row below checks one figure
against the same six items. The browser checks are in
`tests/e2e/live_figure.mjs`, quoted by their printed name. The parity scripts
run in the CI job "WASM matches native" (`.github/workflows/ci.yml`), and the
browser test runs in the CI job "Live figure" on a site built with every
figure's chart JSON, as the Pages build makes it.

Items common to all six:

- **PNG first, and the fallback.** `scripts/build_paper.py` writes the PNG
  into every figure block whether or not the figure is live, so a reader
  without JavaScript gets the published image. Each figure loads static and
  goes live only on "interact" (e2e: "... is marked live-capable", state
  `static`). Under `prefers-reduced-data` the page downloads no WebAssembly
  and starts no worker (e2e: "under prefers-reduced-data the WebAssembly
  module is never downloaded", "... no worker target is attached").
- **No change to the paper pipeline.** The live code lives in `site-src/live/`,
  `scripts/paper_shell.py` and `rust/wasm/`. The figure modules, the
  committed `.npz` caches and the captions are not touched by any of the
  commits below.
- **House style.** Curves and points use the page's `--slate` and
  `--vermilion` tokens (`scripts/paper_shell.py`). Axis labels use the
  page's serif, TeX Gyre Pagella, not the Computer Modern of the PNGs. This
  is a decision (2026-10-02): the labels match the page around them, and the
  page loads no extra webfont.

| figure (`data-live`) | commit | parity against the paper | browser check of one value | reduced-data check | mutation check at review |
|---|---|---|---|---|---|
| `arnold_tongues` | aa0e334 | `check_wasm_parity.py`: tight bound where the map is well conditioned, rare divergence where the orbit is chaotic | "rho ~ 0.0743 at (Omega 0.08, K 0.03)"; "the readout is the point kernel, not the tile lookup" | "a PNG or JSON chart is shown and no live canvas is created" | yes: dropping the tile reset in `onView` fails a node test |
| `devils_staircase` | 7499ab4 | `check_wasm_staircase.py`: npz at D = 0.25, with the thresholds of `check_wasm_parity.py` | "the staircase readout is the point kernel at (D, A)" | "the staircase keeps its PNG or JSON chart and mounts no slider" | yes: (A, D) swapped in the readout fails a node test |
| `delayed_logistic_attractors` | fe8a64d | `check_wasm_delayed_logistic.py`: bit-exact | "a plotted state equals the kernel's at the same (A, D, n)" | "the attractors figure keeps its PNG and mounts no slider" | yes: a moved initial state fails a node test |
| `torus_doubling_attractors` | 359f99a | `check_wasm_torus_doubling.py`: bit-exact, maps (I) and (IV) | "a plotted torus state equals the kernel's at the same (map, A, D, n)" | "the torus figure keeps its PNG and mounts no controls" | yes: map IV initial state, A/D swap and projection swap each fail a test |
| `double_staircase` | cc25155 | `check_wasm_modulated_circle.py`: rho_theta within 1e-6, rho_phi within 1e-12 of C | "a plotted rho_theta equals the kernel's at the same D" | "the double staircase keeps its PNG and mounts no controls" | yes: theta0 0.1 to 0.2 fails two node tests |
| `spacetime_diagrams` | e6d1c61 | `check_wasm_cml_spacetime.py`: models A and C bit-exact; model B against a numpy ensemble with a 1-ulp error on every `sin` | "a plotted field value equals the kernel's at the same (model, eps, sites)" | "the spacetime figure keeps its PNG and mounts no controls" | yes: unsnapped eps fails "the spacetime figure snaps eps to its decimal literal"; a model switch that keeps the old eps fails "switching the spacetime model resets eps to the new model's default"; play from the initial field fails "play appends a chunk from the last row and pause stops the transport" |
