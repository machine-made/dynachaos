/**
 * Glue between the live figure's DOM wiring and its kernels.
 *
 * The page template (scripts/paper_shell.py, mountLive) imports this module
 * and keeps only DOM work: elements, the Plot() call, the pool options. The
 * decisions live here so node tests can drive them — the handlers used to be
 * inside the Python template string, which node cannot import.
 *
 * No DOM and no globals at import time: every collaborator (store, pool,
 * raster, point, plot) arrives as an argument. Handlers that must exist
 * before their collaborator does take a getter — onView is built before the
 * pool is created, and onPaint can only fire after the pool exists, so both
 * read the current binding when they run.
 */

import { HEADER } from "./raster.js";
import { clampValue, debounce, snapToStep } from "./params.js";
import { pageHidden, watchVisibility } from "./visibility.js";

/**
 * Below this CSS width a live figure does not start by itself.
 *
 * 480px is a phone in portrait: common widths are 360-430 CSS px, and 480
 * still sits under the page's 36rem (576px) tablet breakpoint. A wasm
 * worker pool on that width is the load this guard exists to avoid. The
 * interact button still starts live mode when the reader asks.
 */
export const LIVE_MIN_WIDTH = 480;

/**
 * True when a shared link may mount live mode without a click. A missing
 * or non-positive width stays on the PNG: an unknown viewport is not a
 * reason to start workers.
 *
 * @param {number} width CSS pixels
 * @returns {boolean}
 */
export function liveAutoStarts(width) {
  const w = Number(width);
  if (!Number.isFinite(w) || w <= 0) return false;
  return w >= LIVE_MIN_WIDTH;
}

/**
 * The published figure's iteration parameters. The readout's point.sample
 * call, the tile scheduler options and the raster's tile-size budget all
 * read this one object, so the quoted number and the picture cannot drift
 * apart. Frozen: a mutation would silently change only the readers that
 * already captured a value.
 */
export const LIVE_ARNOLD = Object.freeze({nTransient: 200, nIter: 2000, theta0: 0.1});

/**
 * The devil's staircase figure: rho(A) at a reader-chosen drive frequency D.
 * The paper computes D = 0.25, A in [0, 0.25], n_transient 5000, n_iter
 * 50000, theta0 0.1 (maps/circle_map.py). The tile kernel reads Omega = D
 * and K = A, so one tile column (n_omega = 1) at the current D is the whole
 * curve. D is capped at 0.5: rho stays under D + K <= 0.75, inside the
 * plot's [0, 1] y base.
 */
export const LIVE_STAIRCASE = Object.freeze({
  nTransient: 5000,
  nIter: 50000,
  theta0: 0.1,
  paramSpecs: Object.freeze([
    Object.freeze({ name: "D", label: "drive frequency D", min: 0, max: 0.5, step: 0.005, default: 0.25 }),
  ]),
});

/**
 * The delayed-logistic attractors figure: the (x, y) point cloud of
 * (x, y)' = (A x + (1 - A)(1 - D y^2), x) at a reader-chosen D, A = 0.3
 * fixed. The paper computes A = 0.3, n_transient 20000, n_plot 100000
 * (maps/delayed_logistic.py compute_attractors); the live figure plots at
 * most 4096 states, the wasm kernel's cap. D's range is the kernel's own
 * clamp [1.4, 3.5]; the default 1.90 is one of the published panels. The
 * slider's own max is narrower: measured with this kernel at A = 0.3,
 * n_transient 20000, the orbit is bounded through D = 2.815 and first
 * escapes at D = 2.8155 (2026-10-08, step 0.0005), so 2.765 keeps the
 * bounded side with 0.05 of margin and a hash value above it lands back
 * on the max instead of painting a blank plot. `n`
 * is the plotted-state count: the control that trades cloud density for
 * response time. `domain` is the published figure's shared axis limits —
 * the union of both committed npz files padded by 5% — so the slider
 * compares attractors on the paper's axes instead of a refit frame.
 */
export const LIVE_ATTRACTORS = Object.freeze({
  A: 0.3,
  nTransient: 20000,
  nPlot: 2048,
  domain: Object.freeze({ x0: -0.5642697119632384, x1: 0.9817445276044394, y0: -0.5642697119632384, y1: 0.9817445276044394 }),
  paramSpecs: Object.freeze([
    Object.freeze({ name: "D", label: "map parameter D", min: 1.4, max: 2.765, step: 0.005, default: 1.9 }),
    Object.freeze({ name: "n", label: "plotted states n", min: 256, max: 4096, step: 256, default: 2048 }),
  ]),
});

/**
 * The torus-doubling attractors figure: the (X, Y) projection of map (I) or
 * map (IV) at a reader-chosen D. The paper computes map I at A = 0.4 from
 * x0 = (0.5, 0.5, 0.5) and map IV at A = 0.3 from x0 = (0.5, 0.45, 0.52,
 * 0.48), both with n_transient 20000 (maps/torus_doubling.py
 * compute_map_I/compute_map_IV); the live figure plots at most 4096 states,
 * the wasm kernel's cap. Each map's D window is its published sweep —
 * [1.9, 2.25] for map I, [1.48, 1.53] for map IV — inside the kernel's
 * [1.48, 2.25] clamp; the defaults 2.16 and 1.5206 are the published
 * doubled-torus panels. The D spec's range is the union of both windows so
 * a hash token for either map parses; the figure itself clamps D into the
 * active map's window. `domain` is the union of both committed npz files'
 * (X, Y) extents padded by 5%, so the selector compares the two maps on
 * one fixed frame. `project` is the paper's projection: both figures
 * scatter columns 0 and 1 (X, Y) of the state.
 */
export const LIVE_TORUS = Object.freeze({
  nTransient: 20000,
  domain: Object.freeze({ x0: -0.5148468759369567, x1: 1.0121499654527584, y0: -0.5253812228060238, y1: 0.971379815168809 }),
  maps: Object.freeze({
    1: Object.freeze({
      kind: 1,
      a: 0.4,
      x0: Object.freeze([0.5, 0.5, 0.5]),
      dMin: 1.9,
      dMax: 2.25,
      dStep: 0.005,
      dDefault: 2.16,
      title: "Map (I), \u03b1 = 0.4",
      project: Object.freeze([0, 1]),
    }),
    4: Object.freeze({
      kind: 4,
      a: 0.3,
      x0: Object.freeze([0.5, 0.45, 0.52, 0.48]),
      dMin: 1.48,
      dMax: 1.53,
      dStep: 0.0002,
      dDefault: 1.5206,
      title: "Map (IV), \u03b1 = 0.3",
      project: Object.freeze([0, 1]),
    }),
  }),
  paramSpecs: Object.freeze([
    Object.freeze({
      name: "map",
      label: "map",
      min: 1,
      max: 4,
      step: 3,
      default: 1,
      options: Object.freeze([
        Object.freeze({ value: 1, label: "map (I), \u03b1 = 0.4" }),
        Object.freeze({ value: 4, label: "map (IV), \u03b1 = 0.3" }),
      ]),
    }),
    Object.freeze({ name: "D", label: "map parameter D", min: 1.48, max: 2.25, step: 0.005, default: 2.16 }),
    Object.freeze({ name: "n", label: "plotted states n", min: 256, max: 4096, step: 256, default: 2048 }),
  ]),
});

/**
 * The double devil's staircase figure: rho_theta and rho_phi against D for
 * the modulated circle map (Kaneko Eq. 3.1). The paper computes A = 0.1,
 * C = (sqrt(5) - 1) / 2, eps = 0.05, theta0 = phi0 = 0.1, n_transient 3000,
 * n_iter 20000 over D on np.linspace(0, 1, 10000)
 * (maps/modulated_circle.py compute); the live figure recomputes the same
 * pair over a reader-chosen D window with modulated_circle_rotation_tile.
 * eps is the one free parameter (0 to 0.2, the paper's moderate-forcing
 * neighbourhood); A and C stay at the paper values, so rho_phi is C to
 * numerical precision and the curve is a flat line the reader can check.
 * `dMin`/`dMax` are the compute window and the plot's x domain at once —
 * the number fields, the two presets and a drag zoom all move the same
 * pair. `presets` reproduce the D windows of the zoom panels in
 * modulated_circle.py plot_zoom: longest_plateau_window(D, rho_theta,
 * target, 5e-4) on the committed npz padded by 0.008 (target 1/4) and
 * 0.006 (target C) — computed once from
 * figures/sec06_three_torus/double_staircase.npz and hard-coded here.
 * `n` is the D count, bounded by the kernel's 512 cap.
 */
export const LIVE_MODULATED = Object.freeze({
  A: 0.1,
  C: 0.6180339887498949,
  theta0: 0.1,
  phi0: 0.1,
  nTransient: 3000,
  nIter: 20000,
  presets: Object.freeze([
    Object.freeze({ name: "zoom (a): plateau near 1/4", dMin: 0.25502630263026305, dMax: 0.27372657265726574 }),
    Object.freeze({ name: "zoom (b): plateau near C", dMin: 0.5987604760476047, dMax: 0.6270621062106211 }),
  ]),
  paramSpecs: Object.freeze([
    Object.freeze({ name: "eps", label: "forcing \u03b5", min: 0, max: 0.2, step: 0.005, default: 0.05 }),
    Object.freeze({ name: "dMin", label: "D range min", min: 0, max: 1, step: 0.001, default: 0, numberOnly: true }),
    Object.freeze({ name: "dMax", label: "D range max", min: 0, max: 1, step: 0.001, default: 1, numberOnly: true }),
    Object.freeze({ name: "n", label: "points n", min: 16, max: 512, step: 16, default: 256 }),
  ]),
});


/**
 * The CML space-time figure: the field x_i^n of a coupled-map lattice,
 * site i across, time n up, drawn as one heat map. The paper computes
 * N = 200 sites, n_transient 2000, n_record 500 from
 * x0 = default_rng(42).uniform(0, 1, N) (cml/spatiotemporal.py
 * simulate_cml); the live figure runs the same counts through
 * cml_spacetime_tile and keeps a rolling window of the last `nRecord`
 * rows. `models` maps the selector's kernel id to the model's eps window
 * — the published sweep's range and middle value, which is also the wasm
 * export's non-finite fallback. `chunkRows` is the play step: one press
 * appends that many rows computed from the window's last row with
 * n_transient 0, then drops the oldest rows. `x0` arrives from
 * site/live/cml-x0.json (the first 512 values of the paper's seed-42
 * uniform field); the request slices it to the chosen site count, so
 * sites = 200 reproduces the paper's initial condition exactly. The eps
 * spec's range is the union of the three models' windows so a hash token
 * for any model parses; the figure clamps eps into the active model's
 * window. `snap` marks the specs whose values must equal their decimal
 * literals — a range input's stepped double (0.07000000000000001) would
 * otherwise reach the kernel as a different eps.
 */
export const LIVE_SPACETIME = Object.freeze({
  nTransient: 2000,
  nRecord: 500,
  chunkRows: 250,
  models: Object.freeze({
    0: Object.freeze({ kind: 0, title: "Model (A): piecewise", epsMin: 0, epsMax: 0.2, epsDefault: 0.07 }),
    1: Object.freeze({ kind: 1, title: "Model (B): circle", epsMin: 0, epsMax: 0.1, epsDefault: 0.024 }),
    2: Object.freeze({ kind: 2, title: "Model (C): logistic", epsMin: 0, epsMax: 0.5, epsDefault: 0.2 }),
  }),
  paramSpecs: Object.freeze([
    Object.freeze({
      name: "model",
      label: "model",
      min: 0,
      max: 2,
      step: 1,
      default: 0,
      options: Object.freeze([
        Object.freeze({ value: 0, label: "model (A): piecewise" }),
        Object.freeze({ value: 1, label: "model (B): circle" }),
        Object.freeze({ value: 2, label: "model (C): logistic" }),
      ]),
    }),
    Object.freeze({ name: "eps", label: "coupling \u03b5", min: 0, max: 0.5, step: 0.001, default: 0.07, snap: true }),
    Object.freeze({ name: "sites", label: "sites N", min: 16, max: 512, step: 1, default: 200, snap: true }),
  ]),
});

/**
 * data-live kind -> live figure config. The hash-restore path in the page
 * reads paramSpecs from here so a shared link can name its figure.
 */
export const LIVE_FIGURES = Object.freeze({
  arnold_tongues: LIVE_ARNOLD,
  devils_staircase: LIVE_STAIRCASE,
  delayed_logistic_attractors: LIVE_ATTRACTORS,
  torus_doubling_attractors: LIVE_TORUS,
  double_staircase: LIVE_MODULATED,
  spacetime_diagrams: LIVE_SPACETIME,
});

/**
 * The figure's mutable tile store. `tiles` is the same array the Plot()
 * live adapter reads, so a generation reset empties it in place. `trace`
 * is the 1-D figure's derived polyline ({x, y}); the heatmap never uses it.
 *
 * @param {{ nIter: number }} [params]
 * @returns {{ tiles: object[], trace: {x: number[], y: number[]}, generation: number, painted: number, nIter: number }}
 */
export function createStore(params = LIVE_ARNOLD) {
  return { tiles: [], trace: { x: [], y: [] }, generation: 0, painted: 0, nIter: params.nIter };
}

/**
 * The pool's onPaint: fold one finished tile into the store and repaint.
 * A stale-generation command and a tile id outside the viewport are dropped.
 * Every stored record carries a paint sequence so a re-painted tile keys
 * differently in the colour cache — without it the stale bitmap shows.
 * nIter rides in on the tile header so the readout tolerance tracks the
 * kernel's actual iteration count. `onStore`, when given, runs after the
 * fold and before the redraw: the 1-D figure rebuilds its trace there.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ tileWorld: (id: string, viewport: object) => object | null }} deps.raster
 * @param {() => object} deps.getPool
 * @param {() => object} deps.getPlot
 * @param {(store: object) => void} [deps.onStore]
 * @returns {(cmd: object) => void}
 */
export function createPaintHandler({ store, raster, getPool, getPlot, onStore, onFirstPaintComplete }) {
  let paintSeq = 0;
  let firstPaintDone = false;
  const firstPaintTileTimes = [];
  return function onPaint(cmd) {
    const pool = getPool();
    const state = pool ? pool.getState() : null;
    if (!state || cmd.generation !== state.generation) return;
    const world = raster.tileWorld(cmd.id, state.viewport);
    if (!world) return;
    const rec = {
      ...world,
      generation: cmd.generation,
      header: cmd.header,
      data: cmd.data,
      paintSeq: ++paintSeq,
    };
    const idx = store.tiles.findIndex((t) => t.id === rec.id);
    if (idx >= 0) store.tiles[idx] = rec;
    else store.tiles.push(rec);
    store.generation = cmd.generation;
    store.painted = store.tiles.filter(
      (t) => t.generation === store.generation,
    ).length;
    if (cmd.header && Number.isFinite(cmd.header[3])) store.nIter = cmd.header[3];
    if (onStore) onStore(store);
    getPlot().redraw();
    if (!firstPaintDone) {
      if (Number.isFinite(cmd.computeMs)) {
        firstPaintTileTimes.push(cmd.computeMs);
      }
      if (state.queueDepth === 0 && Array.isArray(state.inFlight) && state.inFlight.length === 0) {
        firstPaintDone = true;
        if (onFirstPaintComplete) {
          onFirstPaintComplete({
            tileTimes: firstPaintTileTimes.slice(),
            generation: cmd.generation,
          });
        }
      }
    }
  };
}

/**
 * The hovered/keyboard readout: the point kernel when it is ready and
 * returns a finite number, else the raster lookup on the current
 * generation. The point path runs the full n_iter without the display
 * stop, so it is the value the paper quotes; the raster is the fallback
 * while the wasm glue is still loading or cannot answer.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ isReady: () => boolean, sample: (omega: number, K: number, nTransient: number, nIter: number, theta0: number) => number | null }} deps.point
 * @param {{ sampleAt: (tiles: object[], generation: number, omega: number, K: number) => number | null }} deps.raster
 * @param {{ nTransient: number, nIter: number, theta0: number }} [deps.params]
 * @returns {(omega: number, K: number) => number | null}
 */
export function createReadout({ store, point, raster, params = LIVE_ARNOLD }) {
  return function sampleReadout(omega, K) {
    if (point.isReady()) {
      try {
        const rho = point.sample(omega, K, params.nTransient, params.nIter, params.theta0);
        if (typeof rho === "number" && Number.isFinite(rho)) return rho;
      } catch (_) {}
    }
    return raster.sampleAt(store.tiles, store.generation, omega, K);
  };
}

/**
 * The staircase readout: rho at (D, A) where A is the cursor's x value and
 * D is the slider's current value. Same point-kernel-first rule as the
 * heatmap readout; the raster fallback samples (D, A) on the stored tiles.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ isReady: () => boolean, sample: (omega: number, K: number, nTransient: number, nIter: number, theta0: number) => number | null }} deps.point
 * @param {{ sampleAt: (tiles: object[], generation: number, omega: number, K: number) => number | null }} deps.raster
 * @param {() => number} deps.getD
 * @param {{ nTransient: number, nIter: number, theta0: number }} [deps.params]
 * @returns {(a: number) => number | null}
 */
export function createStaircaseReadout({ store, point, raster, getD, params = LIVE_STAIRCASE }) {
  return function sampleStaircase(a) {
    if (point.isReady()) {
      try {
        const rho = point.sample(getD(), a, params.nTransient, params.nIter, params.theta0);
        if (typeof rho === "number" && Number.isFinite(rho)) return rho;
      } catch (_) {}
    }
    return raster.sampleAt(store.tiles, store.generation, getD(), a);
  };
}

/**
 * The raster half of the readout, exposed to the page as
 * fig._live.rasterSample so the e2e test can compare the two paths at the
 * same (Omega, K) — not at a neighbouring tile cell.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ sampleAt: (tiles: object[], generation: number, omega: number, K: number) => number | null }} deps.raster
 * @returns {(omega: number, K: number) => number | null}
 */
export function createRasterSample({ store, raster }) {
  return (omega, K) => raster.sampleAt(store.tiles, store.generation, omega, K);
}

/**
 * The scheduler options for the pool: the pyramid shape (levels, tileCells)
 * plus the iteration parameters every tile command carries. The same
 * `params` object feeds the readout, so the picture and the quoted number
 * are computed with identical settings. `lockOmega` switches the scheduler
 * to its 1-D mode (nOmega = 1 at the viewport's fixed Omega).
 *
 * @param {object} deps
 * @param {number} deps.levels
 * @param {number} deps.tileCells
 * @param {{ nTransient: number, nIter: number, theta0: number }} [deps.params]
 * @param {boolean} [deps.lockOmega]
 * @returns {{ levels: number, tileCells: number, nTransient: number, nIter: number, theta0: number, lockOmega?: boolean }}
 */
export function createSchedulerOptions({ levels, tileCells, params = LIVE_ARNOLD, lockOmega = false }) {
  const out = {
    levels,
    tileCells,
    nTransient: params.nTransient,
    nIter: params.nIter,
    theta0: params.theta0,
  };
  if (lockOmega) out.lockOmega = true;
  return out;
}

/**
 * The Plot() live adapter's view hook: push the new domain into the pool
 * and, when the pool answers with a new generation, reset the store. A new
 * generation also resets the scheduler's per-generation retry list, so a
 * tile that spent its one retry before the pan may be retried again after
 * it. Emptying `tiles` in place keeps the array identity the adapter
 * captured.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => object | null} deps.getPool
 * @param {() => object | null} deps.getPlot
 * @returns {(d: { x0: number, x1: number, y0: number, y1: number }) => void}
 */
export function createViewHandler({ store, getPool, getPlot }) {
  return function onView(d) {
    const pool = getPool();
    if (!pool) return;
    pool.setViewport({ omegaMin: d.x0, omegaMax: d.x1, kMin: d.y0, kMax: d.y1 });
    const gen = pool.getState().generation;
    if (gen !== store.generation) {
      store.generation = gen;
      store.painted = 0;
      store.tiles.length = 0;
      const plot = getPlot();
      if (plot) plot.redraw();
    }
  };
}

/**
 * The staircase's view hook and parameter push, sharing one push helper.
 * The plot's x axis is A (the kernel's K); Omega is the slider's D. A
 * y-only domain change (the auto-fit, a y drag) must not recompute, so
 * onView pushes only when the x domain actually moved. applyD pushes the
 * current D at the current x domain — the slider's recompute path. Both
 * funnel through the same generation reset as the heatmap's onView.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => object | null} deps.getPool
 * @param {() => object | null} deps.getPlot
 * @param {() => number} deps.getD
 * @returns {{ onView: (d: { x0: number, x1: number }) => void, applyD: () => void }}
 */
export function createStaircaseView({ store, getPool, getPlot, getD }) {
  let lastK = null;
  function push(kMin, kMax) {
    const pool = getPool();
    if (!pool) return;
    const D = getD();
    pool.setViewport({ omegaMin: D, omegaMax: D, kMin, kMax });
    lastK = { kMin, kMax };
    const gen = pool.getState().generation;
    if (gen !== store.generation) {
      store.generation = gen;
      store.painted = 0;
      store.tiles.length = 0;
      const plot = getPlot();
      if (plot) plot.redraw();
    }
  }
  return {
    onView(d) {
      if (lastK && d.x0 === lastK.kMin && d.x1 === lastK.kMax) return;
      push(d.x0, d.x1);
    },
    applyD() {
      const plot = getPlot();
      if (!plot) return;
      const dom = plot.getDomain();
      push(dom.x0, dom.x1);
    },
  };
}

/**
 * Fold the stored tiles into the staircase's polyline. Every current-
 * generation tile contributes its cells at their K coordinates; where two
 * tiles share a coordinate the finer level wins. The result is sorted by A
 * so the Plot() line path can stroke it directly. A 1-D tile carries
 * nOmega = 1, so cell iy sits at kMin + iy * (kMax - kMin) / (nK - 1) —
 * the same linspace the kernel iterates.
 *
 * @param {object[]} tiles
 * @param {number} generation
 * @returns {{ x: number[], y: number[] }}
 */
export function tilesToTrace(tiles, generation) {
  const byA = new Map();
  for (const t of tiles) {
    if (!t || t.generation !== generation || !t.header || !t.data) continue;
    const nK = Math.max(1, Math.floor(Number(t.header[1])));
    const nOmega = Math.max(1, Math.floor(Number(t.header[0])));
    if (t.data.length < HEADER + nK * nOmega) continue;
    const step = nK > 1 ? (t.kMax - t.kMin) / (nK - 1) : 0;
    for (let iy = 0; iy < nK; iy++) {
      const a = nK > 1 ? t.kMin + iy * step : t.kMin;
      const v = t.data[HEADER + iy * nOmega];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      const prev = byA.get(a);
      if (!prev || t.level >= prev.level) byA.set(a, { level: t.level, v });
    }
  }
  const xs = [...byA.keys()].sort((p, q) => p - q);
  return { x: xs, y: xs.map((a) => byA.get(a).v) };
}

/**
 * Refit the plot's y domain to the trace as it refines. The base domain is
 * [0, 1] so a reset always shows the full rotation-number range. While the
 * reader has not moved the y range, each paint refits: the first tile is
 * coarse and its min/max is not the curve's. A call that finds the y range
 * away from the last fitted value (or the base, before the first fit) means
 * the reader zoomed, panned or reset, so the fit latches off for good and
 * leaves the view alone. The setDomain call is silent: an automatic fit is
 * not a reader change and must not mark the figure modified or touch the
 * URL.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => object | null} deps.getPlot
 * @returns {() => void}
 */
export function createLineYFit({ store, getPlot }) {
  let locked = false;
  let fittedY = null;
  return function fitLineY() {
    if (locked) return;
    const ys = store.trace.y;
    if (!ys.length) return;
    const plot = getPlot();
    if (!plot) return;
    const dom = plot.getDomain();
    const base = plot.getBase ? plot.getBase() : null;
    const ref = fittedY || base;
    if (ref && (dom.y0 !== ref.y0 || dom.y1 !== ref.y1)) {
      // The reader moved the y range since the last fit (or the base view).
      locked = true;
      return;
    }
    const lo = Math.min(...ys);
    const hi = Math.max(...ys);
    const pad = (hi - lo) * 0.08 || 0.05;
    const y0 = lo - pad;
    const y1 = hi + pad;
    if (dom.y0 === y0 && dom.y1 === y1) return;
    fittedY = { y0, y1 };
    plot.setDomain({ x0: dom.x0, x1: dom.x1, y0, y1 }, { silent: true });
  };
}

/**
 * Slider wiring for one live figure's parameter set. `onInput(name, raw)`
 * clamps the value into its spec, updates the state, echoes the clamped
 * value back through `echo` (the page syncs its two inputs there), and
 * schedules `apply` through the debounce — a drag burst collapses into one
 * recompute. `setParams` applies a parsed hash state immediately, with no
 * debounce: a shared link must not wait. `writeHash` runs inside apply so
 * the URL and the recompute cannot drift apart. `onSet`, when given, runs
 * after each spec's state update — in `setParams` between specs, so a
 * selector that changes another parameter's range (the torus figure's map
 * switch moves D's window) sees the new value before the next spec applies
 * — and the echo reads the post-hook state, so a snapped or derived value
 * never disagrees with its control.
 *
 * @param {object} deps
 * @param {{ name: string, min: number, max: number, default: number }[]} deps.specs
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {() => void} deps.apply
 * @param {() => void} deps.writeHash
 * @param {(name: string, state: object) => void} [deps.onSet]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void }}
 */
export function createParamWiring({ specs, debounceMs, echo, apply, writeHash, onSet, timers }) {
  const state = {};
  for (const spec of specs) state[spec.name] = spec.default;
  const fire = debounce(() => {
    apply();
    writeHash();
  }, debounceMs, timers);
  function onInput(name, raw) {
    const spec = specs.find((s) => s.name === name);
    if (!spec) return;
    state[name] = clampValue(raw, spec);
    if (onSet) onSet(name, state);
    echo(name, state[name]);
    fire();
  }
  function setParams(values) {
    for (const spec of specs) {
      if (values[spec.name] === undefined) continue;
      state[spec.name] = clampValue(values[spec.name], spec);
      if (onSet) onSet(spec.name, state);
      echo(spec.name, state[spec.name]);
    }
    apply();
    writeHash();
  }
  return { state, onInput, setParams };
}

const ATTRACTOR_GLUE_HREF = new URL("../wasm/dynachaos_wasm.js", import.meta.url).href;

/**
 * The analytic fixed point of the delayed logistic map at parameter D:
 * solving x = A x + (1 - A)(1 - D x^2) gives x = (sqrt(1 + 4D) - 1) / 2D.
 * The published figure starts its orbits at (fp + 0.01, fp - 0.01), so the
 * live figure does too, and the canvas marks the point itself.
 *
 * @param {number} D
 * @returns {number}
 */
export function attractorFixedPoint(D) {
  return (Math.sqrt(1 + 4 * D) - 1) / (2 * D);
}

/**
 * The kernel request for one attractor state: A first, then the D range —
 * the order delayed_logistic_attractor_tile takes them. n_D is 1, so the
 * tile holds one block of n_plot (x, y) pairs. state0 is the published
 * convention (fp + 0.01, fp - 0.01); the same expression in the same order
 * as maps/delayed_logistic.py, so the orbit is bit-identical.
 *
 * @param {object} state name -> value, from the parameter wiring
 * @param {{ A: number, nTransient: number, nPlot: number }} [params]
 * @returns {{ a: number, dMin: number, dMax: number, nD: number, nTransient: number, nPlot: number, state0: Float64Array }}
 */
export function attractorRequest(state, params = LIVE_ATTRACTORS) {
  const D = state.D;
  const fp = attractorFixedPoint(D);
  return {
    a: params.A,
    dMin: D,
    dMax: D,
    nD: 1,
    nTransient: params.nTransient,
    nPlot: Math.round(state.n),
    state0: new Float64Array([fp + 0.01, fp - 0.01]),
  };
}

/**
 * Fold an attractor tile into the point cloud. The layout is the kernel's:
 * a 4-element header [n_D, n_plot, n_transient, 2], then n_D blocks of
 * n_plot (x, y) pairs — x first, then y. A diverged D is a block of NaN
 * pairs; the plot skips non-finite points, so they fold in harmlessly.
 * Anything that does not match the header yields an empty cloud rather
 * than a misread buffer.
 *
 * @param {ArrayLike<number> | null | undefined} tile
 * @returns {{ x: number[], y: number[] }}
 */
export function attractorTrace(tile) {
  const x = [];
  const y = [];
  if (!tile || tile.length < HEADER) return { x, y };
  const nD = Math.floor(Number(tile[0]));
  const nPlot = Math.floor(Number(tile[1]));
  if (!(nD >= 1) || !(nPlot >= 1) || tile.length < HEADER + nD * nPlot * 2) {
    return { x, y };
  }
  for (let i = 0; i < nD * nPlot; i++) {
    x.push(tile[HEADER + 2 * i]);
    y.push(tile[HEADER + 2 * i + 1]);
  }
  return { x, y };
}

const kernelPromises = new Map();
const kernelCache = new Map();


/**
 * Lazily load the wasm glue and return one named export. The module is the
 * same instance point.js initialises (the import cache keys on the resolved
 * URL), and the glue's init is idempotent. A failed load is not cached
 * forever: the next call retries once.
 *
 * @param {string} name the wasm-bindgen export to return
 * @returns {Promise<Function>}
 */
function loadWasmExport(name) {
  let promise = kernelPromises.get(name);
  if (promise == null) {
    promise = (async () => {
      const glue = await import(ATTRACTOR_GLUE_HREF);
      if (typeof glue[name] !== "function") {
        throw new Error(`wasm glue is missing ${name}`);
      }
      const isNode =
        typeof process !== "undefined" && process.versions && process.versions.node;
      if (!isNode) {
        await glue.default();
      } else {
        const { readFile } = await import("node:fs/promises");
        const bytes = await readFile(new URL("dynachaos_wasm_bg.wasm", ATTRACTOR_GLUE_HREF));
        glue.initSync({ module: bytes });
      }
      kernelCache.set(name, glue[name]);
      return glue[name];
    })();
    promise.catch(() => {
      kernelPromises.delete(name);
    });
    kernelPromises.set(name, promise);
  }
  return promise;
}

/**
 * The already-resolved export, or null while the glue is still loading.
 * The double staircase's readout samples through this synchronously — a
 * per-pointermove await would lag the cursor.
 *
 * @param {string} name the wasm-bindgen export
 * @returns {Function | null}
 */
export function loadedKernel(name) {
  return kernelCache.get(name) || null;
}

/**
 * delayed_logistic_attractor_tile, loaded through the shared loader.
 *
 * @returns {Promise<(a: number, dMin: number, dMax: number, nD: number, nTransient: number, nPlot: number, state0: Float64Array) => Float64Array>}
 */
export function loadAttractorKernel() {
  return loadWasmExport("delayed_logistic_attractor_tile");
}

/**
 * torus_doubling_attractor_tile, loaded through the shared loader.
 *
 * @returns {Promise<(mapKind: number, a: number, d: number, nTransient: number, nPlot: number, state0: Float64Array) => Float64Array>}
 */
export function loadTorusKernel() {
  return loadWasmExport("torus_doubling_attractor_tile");
}

/**
 * modulated_circle_rotation_tile, loaded through the shared loader.
 *
 * @returns {Promise<(a: number, c: number, dMin: number, dMax: number, nD: number, eps: number, nTransient: number, nIter: number, theta0: number, phi0: number) => Float64Array>}
 */
export function loadModulatedKernel() {
  return loadWasmExport("modulated_circle_rotation_tile");
}


/**
 * The default kernel call: load the glue, then invoke the export with the
 * request's fields in the kernel's own order — A, then the D range, the
 * counts, and the start state.
 *
 * @param {ReturnType<typeof attractorRequest>} req
 * @returns {Promise<Float64Array>}
 */
export async function attractorKernelCall(req) {
  const kernel = await loadAttractorKernel();
  return kernel(req.a, req.dMin, req.dMax, req.nD, req.nTransient, req.nPlot, req.state0);
}

/**
 * The recompute loop behind the direct-call attractor figures. `request`
 * turns the parameter state into a kernel request, `call` runs it, `fold`
 * turns the returned tile into the (x, y) cloud, and a sequence number
 * drops a stale result: a request that resolves after a newer one was
 * issued is discarded, never painted. `onTrace` runs after the store's
 * arrays are spliced so the page can update its derived marks and redraw.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => object} deps.getState
 * @param {(state: object) => object} deps.request
 * @param {(tile: ArrayLike<number>, req: object) => {x: number[], y: number[]}} deps.fold
 * @param {(req: object) => Promise<ArrayLike<number>>} deps.call
 * @param {(store: object, trace: {x: number[], y: number[]}, req: object) => void} [deps.onTrace]
 * @param {(err: unknown) => void} [deps.onError]
 * @param {object} [deps.document]
 * @returns {{ run: () => void, ready: () => boolean, destroy: () => void }}
 */
function createKernelRunner({ store, getState, request, fold, call, onTrace, onError, document: doc }) {
  let seq = 0;
  let ready = false;
  let paused = pageHidden(doc);
  let pending = false;
  const unwatch = watchVisibility(doc, (hidden) => {
    paused = hidden;
    if (!hidden && pending) {
      pending = false;
      run();
    }
  });
  async function run() {
    if (paused) {
      pending = true;
      // Bump the sequence so an in-flight call that resolves while the tab
      // is hidden is dropped, the same way a newer request already is. The
      // resume below issues the current parameters.
      seq += 1;
      return;
    }
    const req = request(getState());
    const mySeq = ++seq;
    try {
      const t0 = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
      const tile = await call(req);
      const t1 = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
      const computeMs = Math.max(0, t1 - t0);
      ready = true;
      if (mySeq !== seq) return; // a newer request superseded this one
      const trace = fold(tile, req);
      // Splice into the store's arrays instead of replacing them: the
      // page's panel.traces captured these arrays at mount, and a fresh
      // object would orphan the plot's view of the cloud. Every channel
      // the fold returns is spliced — the double staircase carries a
      // second curve in trace.y2 beside x and y.
      for (const key of Object.keys(trace)) {
        const target = store.trace[key];
        const source = trace[key];
        if (Array.isArray(target)) target.splice(0, target.length, ...source);
        else store.trace[key] = source;
      }
      store.generation = mySeq;
      store.painted = trace.x.length;
      if (onTrace) onTrace(store, trace, req, computeMs);
    } catch (err) {
      if (onError) onError(err);
    }
  }
  return { run, ready: () => ready, destroy: unwatch };
}

/**
 * The attractor figure's parameter wiring and recompute path. The page
 * (mountAttractors in scripts/paper_shell.py) builds the inputs and hands
 * in the DOM callbacks; the decisions live here so node tests can drive
 * them. `call` is injectable: the page passes attractorKernelCall, a test
 * passes a spy.
 *
 * The kernel is called directly on the main thread — one call is at most
 * 20000 + 4096 map steps, microseconds of work, so the tile pool's worker
 * round trip would buy nothing. The debounce collapses a slider burst into
 * one apply, and a sequence number drops a stale result: a request that
 * resolves after a newer one was issued is discarded, never painted.
 *
 * @param {object} deps
 * @param {{ A: number, nTransient: number, nPlot: number, paramSpecs: object[] }} [deps.params]
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {(req: object) => Promise<ArrayLike<number>>} [deps.call]
 * @param {(store: object, trace: {x: number[], y: number[]}, req: object) => void} deps.onTrace
 * @param {() => void} deps.writeHash
 * @param {(err: unknown) => void} [deps.onError]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, store: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void, refresh: () => void, ready: () => boolean }}
 */
export function createAttractorFigure({
  params = LIVE_ATTRACTORS,
  debounceMs,
  echo,
  call = attractorKernelCall,
  onTrace,
  writeHash,
  onError,
  timers,
  document: doc,
}) {
  const store = { trace: { x: [], y: [] }, generation: 0, painted: 0 };
  const runner = createKernelRunner({
    store,
    getState: () => wiring.state,
    request: (state) => attractorRequest(state, params),
    fold: (tile) => attractorTrace(tile),
    call,
    onTrace,
    onError,
    document: doc,
  });
  const wiring = createParamWiring({
    specs: params.paramSpecs,
    debounceMs,
    echo,
    apply: () => {
      runner.run();
    },
    writeHash,
    timers,
  });
  return {
    state: wiring.state,
    store,
    onInput: wiring.onInput,
    setParams: wiring.setParams,
    refresh: () => {
      runner.run();
    },
    ready: runner.ready,
    destroy: runner.destroy,
  };
}

/**
 * Snap a map selector value to a kernel map kind: 2 and below is map I, 3
 * and above is map IV — the same snap the wasm export applies, so the
 * figure and the kernel can never disagree about which map runs.
 *
 * @param {number} value
 * @returns {1 | 4}
 */
export function torusMapKind(value) {
  return Number(value) >= 3 ? 4 : 1;
}

/**
 * The kernel request for one torus-doubling state: map kind first, then A,
 * D, the counts, and the start state — the order
 * torus_doubling_attractor_tile takes them. A and x0 come from the chosen
 * map's config; D is clamped into that map's doubling window so a stale or
 * hand-edited hash value can never reach the kernel outside it.
 *
 * @param {object} state name -> value, from the parameter wiring
 * @param {{ nTransient: number, maps: object }} [params]
 * @returns {{ mapKind: number, a: number, d: number, nTransient: number, nPlot: number, state0: Float64Array }}
 */
export function torusRequest(state, params = LIVE_TORUS) {
  const map = params.maps[torusMapKind(state.map)];
  const d = Math.min(map.dMax, Math.max(map.dMin, state.D));
  return {
    mapKind: map.kind,
    a: map.a,
    d,
    nTransient: params.nTransient,
    nPlot: Math.round(state.n),
    state0: new Float64Array(map.x0),
  };
}

/**
 * Fold a torus-doubling tile into the (X, Y) point cloud. The layout is
 * the kernel's: a 4-element header [map_kind, dim, n_transient,
 * n_produced], then n_produced states of dim components each. `project`
 * names the two state components the paper scatters — [0, 1] (X, Y) for
 * both maps. A tile that does not match its header, or a diverged orbit
 * with zero produced states, yields an empty cloud rather than a misread
 * buffer.
 *
 * @param {ArrayLike<number> | null | undefined} tile
 * @param {{ project: ArrayLike<number> }} map the chosen map's config
 * @returns {{ x: number[], y: number[] }}
 */
export function torusTrace(tile, map) {
  const x = [];
  const y = [];
  if (!tile || tile.length < HEADER) return { x, y };
  const dim = Math.floor(Number(tile[1]));
  const nProduced = Math.floor(Number(tile[3]));
  const px = Math.floor(Number(map.project[0]));
  const py = Math.floor(Number(map.project[1]));
  if (!(dim >= 1) || !(nProduced >= 0) || px >= dim || py >= dim) return { x, y };
  if (tile.length < HEADER + nProduced * dim) return { x, y };
  for (let i = 0; i < nProduced; i++) {
    x.push(tile[HEADER + i * dim + px]);
    y.push(tile[HEADER + i * dim + py]);
  }
  return { x, y };
}

/**
 * The default kernel call: load the glue, then invoke the export with the
 * request's fields in the kernel's own order — map kind, A, D, the counts,
 * and the start state.
 *
 * @param {ReturnType<typeof torusRequest>} req
 * @returns {Promise<Float64Array>}
 */
export async function torusKernelCall(req) {
  const kernel = await loadTorusKernel();
  return kernel(req.mapKind, req.a, req.d, req.nTransient, req.nPlot, req.state0);
}

/**
 * The torus-doubling figure's parameter wiring and recompute path, sharing
 * the delayed-logistic figure's runner. The page (mountTorus in
 * scripts/paper_shell.py) builds the inputs and hands in the DOM
 * callbacks; the decisions live here so node tests can drive them.
 *
 * The map selector is the figure's one non-slider control. Switching maps
 * snaps the value to a kernel kind, resets D to the new map's published
 * default — the window's own D range would clamp a map-I D to map IV's
 * ceiling, which reads as a broken slider — and lets the wiring's echo
 * carry the snapped value back to the control. A D input outside the
 * active window (the number field accepts the union range) clamps into it.
 *
 * @param {object} deps
 * @param {{ nTransient: number, maps: object, paramSpecs: object[] }} [deps.params]
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {(req: object) => Promise<ArrayLike<number>>} [deps.call]
 * @param {(store: object, trace: {x: number[], y: number[]}, req: object) => void} deps.onTrace
 * @param {() => void} deps.writeHash
 * @param {(err: unknown) => void} [deps.onError]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, store: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void, refresh: () => void, ready: () => boolean }}
 */
export function createTorusFigure({
  params = LIVE_TORUS,
  debounceMs,
  echo,
  call = torusKernelCall,
  onTrace,
  writeHash,
  onError,
  timers,
  document: doc,
}) {
  const store = { trace: { x: [], y: [] }, generation: 0, painted: 0 };
  const runner = createKernelRunner({
    store,
    getState: () => wiring.state,
    request: (state) => torusRequest(state, params),
    fold: (tile, req) => torusTrace(tile, params.maps[req.mapKind]),
    call,
    onTrace,
    onError,
    document: doc,
  });
  let activeKind = null;
  function syncState(name) {
    const kind = torusMapKind(wiring.state.map);
    if (wiring.state.map !== kind) {
      wiring.state.map = kind;
      echo("map", kind);
    }
    const map = params.maps[kind];
    if (kind !== activeKind) {
      if (name === "map") {
        // A map switch moves D's window; reset to the new map's published
        // default rather than clamping the old map's D into it.
        wiring.state.D = map.dDefault;
        echo("D", map.dDefault);
      }
      activeKind = kind;
    } else {
      const d = Math.min(map.dMax, Math.max(map.dMin, wiring.state.D));
      if (wiring.state.D !== d) {
        wiring.state.D = d;
        echo("D", d);
      }
    }
  }
  const wiring = createParamWiring({
    specs: params.paramSpecs,
    debounceMs,
    echo,
    apply: () => {
      runner.run();
    },
    writeHash,
    onSet: (name) => syncState(name),
    timers,
  });
  return {
    state: wiring.state,
    store,
    onInput: wiring.onInput,
    setParams: wiring.setParams,
    refresh: () => {
      runner.run();
    },
    ready: runner.ready,
    destroy: runner.destroy,
  };
}

/**
 * The kernel request for one double-staircase sweep: A and C first, then
 * the D window, the point count, eps, the iteration counts and the start
 * phases — the order modulated_circle_rotation_tile takes them. A, C and
 * the phases come from the figure's config; eps, the window and n are the
 * reader's. dMin <= dMax is an invariant of the wiring's syncRange, so the
 * request never sees an inverted window.
 *
 * @param {object} state name -> value, from the parameter wiring
 * @param {{ A: number, C: number, theta0: number, phi0: number, nTransient: number, nIter: number }} [params]
 * @returns {{ a: number, c: number, dMin: number, dMax: number, nD: number, eps: number, nTransient: number, nIter: number, theta0: number, phi0: number }}
 */
export function modulatedRequest(state, params = LIVE_MODULATED) {
  return {
    a: params.A,
    c: params.C,
    dMin: state.dMin,
    dMax: state.dMax,
    nD: Math.round(state.n),
    eps: state.eps,
    nTransient: params.nTransient,
    nIter: params.nIter,
    theta0: params.theta0,
    phi0: params.phi0,
  };
}

/**
 * Fold a modulated-circle tile into the two rotation-number curves. The
 * layout is the kernel's: a 4-element header [n_d, n_transient, n_iter,
 * 2], then n_d (rho_theta, rho_phi) pairs, D-major. The D axis is the
 * request's window on the kernel's own grid — dMin + (dMax - dMin) k /
 * (n_d - 1) — so a plotted point sits at exactly the D the kernel
 * iterated. A tile that does not match its header yields empty curves
 * rather than a misread buffer.
 *
 * @param {ArrayLike<number> | null | undefined} tile
 * @param {ReturnType<typeof modulatedRequest>} req
 * @returns {{ x: number[], y: number[], y2: number[] }}
 */
export function modulatedTrace(tile, req) {
  const x = [];
  const y = [];
  const y2 = [];
  if (!tile || tile.length < HEADER) return { x, y, y2 };
  const nD = Math.floor(Number(tile[0]));
  if (!(nD >= 1) || tile.length < HEADER + nD * 2) return { x, y, y2 };
  const step = nD > 1 ? (req.dMax - req.dMin) / (nD - 1) : 0;
  for (let k = 0; k < nD; k++) {
    const rt = tile[HEADER + 2 * k];
    const rp = tile[HEADER + 2 * k + 1];
    if (!Number.isFinite(rt) || !Number.isFinite(rp)) continue;
    x.push(req.dMin + k * step);
    y.push(rt);
    y2.push(rp);
  }
  return { x, y, y2 };
}

/**
 * The default kernel call: load the glue, then invoke the export with the
 * request's fields in the kernel's own order — A, C, the D window, n_d,
 * eps, the counts, and the start phases. `kernel` is injectable so a node
 * test can check the argument order without the wasm build.
 *
 * @param {ReturnType<typeof modulatedRequest>} req
 * @param {Function} [kernel]
 * @returns {Promise<Float64Array>}
 */
export async function modulatedKernelCall(req, kernel) {
  const fn = kernel || (await loadModulatedKernel());
  return fn(req.a, req.c, req.dMin, req.dMax, req.nD, req.eps, req.nTransient, req.nIter, req.theta0, req.phi0);
}

/**
 * The double-staircase readout: rho_theta at the cursor's D, computed by
 * the loaded kernel at the figure's current eps — the same value the e2e
 * check recomputes. Until the glue resolves, the nearest plotted point of
 * the rho_theta trace answers instead, so the tip never shows a gap while
 * the module loads.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => number} deps.getEps
 * @param {{ nTransient: number, nIter: number, A: number, C: number, theta0: number, phi0: number }} [deps.params]
 * @returns {(d: number) => number | null}
 */
export function createModulatedReadout({ store, getEps, params = LIVE_MODULATED }) {
  return function sampleStaircase(d) {
    const kernel = loadedKernel("modulated_circle_rotation_tile");
    if (kernel) {
      try {
        const tile = kernel(
          params.A, params.C, d, d, 1, getEps(),
          params.nTransient, params.nIter, params.theta0, params.phi0,
        );
        if (tile && tile.length >= HEADER + 2 && Number.isFinite(tile[HEADER])) {
          return tile[HEADER];
        }
      } catch (_) {}
    }
    const xs = store.trace.x;
    const ys = store.trace.y;
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const dist = Math.abs(xs[i] - d);
      if (dist < bestDist) {
        bestDist = dist;
        best = ys[i];
      }
    }
    return best;
  };
}

/**
 * The double staircase's view hook and range sync. The plot's x domain IS
 * the compute window: a drag zoom or a reset pushes the new [x0, x1] into
 * the dMin/dMax inputs through `onRange` (the debounced input path), and
 * `syncView` — run once per painted trace — moves the domain onto the
 * current range when the change came from the inputs, a preset or a hash
 * restore instead. The two directions cannot loop: onView only fires when
 * the domain's x range differs from the state, and syncView only calls
 * setDomain when it does.
 *
 * @param {object} deps
 * @param {() => { dMin: number, dMax: number }} deps.getRange
 * @param {(dMin: number, dMax: number) => void} deps.onRange
 * @param {() => object | null} deps.getPlot
 * @returns {{ onView: (d: { x0: number, x1: number }) => void, syncView: () => void }}
 */
export function createRangeView({ getRange, onRange, getPlot }) {
  return {
    onView(d) {
      const r = getRange();
      if (d.x0 === r.dMin && d.x1 === r.dMax) return;
      onRange(d.x0, d.x1);
    },
    syncView() {
      const plot = getPlot();
      if (!plot) return;
      const r = getRange();
      const dom = plot.getDomain();
      if (dom.x0 === r.dMin && dom.x1 === r.dMax) return;
      plot.setDomain({ x0: r.dMin, x1: r.dMax, y0: dom.y0, y1: dom.y1 });
    },
  };
}

/**
 * The double-staircase figure's parameter wiring and recompute path,
 * sharing the attractor figures' runner. The page (mountDoubleStaircase
 * in scripts/paper_shell.py) builds the inputs and hands in the DOM
 * callbacks; the decisions live here so node tests can drive them.
 *
 * The dMin/dMax pair is one logical window: editing one end past the
 * other moves the far end to meet it (the edited value always wins), so
 * the kernel never sees an inverted window and a two-step edit lands
 * where the reader meant it. The view sync runs inside the runner's
 * onTrace — after the new curves are spliced, before the page redraws —
 * so the domain and the data move together.
 *
 * @param {object} deps
 * @param {object} [deps.params]
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {(req: object) => Promise<ArrayLike<number>>} [deps.call]
 * @param {(store: object, trace: object, req: object) => void} deps.onTrace
 * @param {() => void} deps.writeHash
 * @param {() => object | null} deps.getPlot
 * @param {(err: unknown) => void} [deps.onError]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, store: object, view: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void, refresh: () => void, ready: () => boolean }}
 */
export function createModulatedFigure({
  params = LIVE_MODULATED,
  debounceMs,
  echo,
  call = modulatedKernelCall,
  onTrace,
  writeHash,
  getPlot,
  onError,
  timers,
  document: doc,
}) {
  const store = { trace: { x: [], y: [], y2: [] }, generation: 0, painted: 0 };
  const fitY = createLineYFit({ store, getPlot });
  const view = createRangeView({
    getRange: () => ({ dMin: wiring.state.dMin, dMax: wiring.state.dMax }),
    onRange: (dMin, dMax) => {
      wiring.onInput("dMin", dMin);
      wiring.onInput("dMax", dMax);
    },
    getPlot,
  });
  const runner = createKernelRunner({
    store,
    getState: () => wiring.state,
    request: (state) => modulatedRequest(state, params),
    fold: (tile, req) => modulatedTrace(tile, req),
    call,
    onTrace: (s, trace, req) => {
      view.syncView();
      fitY();
      if (onTrace) onTrace(s, trace, req);
    },
    onError,
    document: doc,
  });
  function syncRange(name) {
    // The edited end wins: pushing dMin past dMax moves dMax up to meet
    // it, and vice versa, so the window never inverts.
    if (name === "dMin" && wiring.state.dMin > wiring.state.dMax) {
      wiring.state.dMax = wiring.state.dMin;
      echo("dMax", wiring.state.dMax);
    } else if (name === "dMax" && wiring.state.dMax < wiring.state.dMin) {
      wiring.state.dMin = wiring.state.dMax;
      echo("dMin", wiring.state.dMin);
    }
  }
  const wiring = createParamWiring({
    specs: params.paramSpecs,
    debounceMs,
    echo,
    apply: () => {
      runner.run();
    },
    writeHash,
    onSet: (name) => syncRange(name),
    timers,
  });
  return {
    state: wiring.state,
    store,
    view,
    onInput: wiring.onInput,
    setParams: wiring.setParams,
    refresh: () => {
      runner.run();
    },
    ready: runner.ready,
    destroy: runner.destroy,
  };
}
/**
 * Snap a model selector value to a kernel model id: 0, 1 or 2 — the same
 * clamp the wasm export applies (model.min(2)), so the figure and the
 * kernel never disagree about which lattice runs. A non-finite value
 * falls back to model A.
 *
 * @param {number} value
 * @returns {0 | 1 | 2}
 */
export function cmlModelKind(value) {
  const v = Math.round(Number(value));
  if (!Number.isFinite(v)) return 0;
  return v <= 0 ? 0 : v >= 2 ? 2 : v;
}

/**
 * The kernel request for one fresh space-time field: model id first, then
 * eps, the site count, the transient, the record length and the initial
 * field — the order cml_spacetime_tile takes them. eps is clamped into
 * the chosen model's window so a stale or hand-edited hash value can
 * never reach the kernel outside it. x0 is the published field sliced to
 * the site count: uniform(0, 1, N) draws the first N values of the
 * seed-42 stream, so slicing — not reseeding — reproduces the paper's
 * initial condition at N = 200.
 *
 * @param {object} state name -> value, from the parameter wiring
 * @param {{ nTransient: number, nRecord: number, models: object }} [params]
 * @param {ArrayLike<number>} [x0] the seed-42 uniform field
 * @returns {{ model: number, eps: number, nSites: number, nTransient: number, nRecord: number, x0: Float64Array }}
 */
export function cmlRequest(state, params = LIVE_SPACETIME, x0 = []) {
  const model = params.models[cmlModelKind(state.model)];
  const nSites = Math.round(state.sites);
  return {
    model: model.kind,
    eps: Math.min(model.epsMax, Math.max(model.epsMin, state.eps)),
    nSites,
    nTransient: params.nTransient,
    nRecord: params.nRecord,
    x0: new Float64Array(Array.from(x0).slice(0, nSites)),
  };
}

/**
 * The kernel request for one play chunk: the same model and eps the
 * window was computed with, no transient, and the window's last row as
 * the initial field — continuing the orbit instead of restarting it. The
 * kernel's own continuation is what the parity script checks bit for bit.
 *
 * @param {object} state name -> value, from the parameter wiring
 * @param {{ values: ArrayLike<number>, nSites: number, nRows: number, model: number, eps: number }} field
 * @param {{ chunkRows: number }} [params]
 * @returns {{ model: number, eps: number, nSites: number, nTransient: number, nRecord: number, x0: Float64Array } | null}
 */
export function cmlChunkRequest(state, field, params = LIVE_SPACETIME) {
  if (!field || !field.values || field.nRows < 1 || field.nSites < 1) return null;
  const nSites = field.nSites;
  const last = Array.prototype.slice.call(field.values, (field.nRows - 1) * nSites, field.nRows * nSites);
  return {
    model: field.model,
    eps: field.eps,
    nSites,
    nTransient: 0,
    nRecord: params.chunkRows,
    x0: new Float64Array(last),
  };
}

/**
 * Unpack a cml_spacetime_tile into a field record. The layout is the
 * kernel's: a 4-element header [model, n_sites, n_transient, n_record],
 * then n_record rows of n_sites values, row-major by time then site —
 * values[t * n_sites + i] is site i of the t-th recorded field, exactly
 * as spacetime[t] = x fills the (n_record, N) array in Python. A tile
 * that does not match its header yields null rather than a misread
 * buffer. `req` supplies model and eps so a later chunk can continue the
 * same run.
 *
 * @param {ArrayLike<number> | null | undefined} tile
 * @param {ReturnType<typeof cmlRequest>} req
 * @returns {{ values: Float64Array, nSites: number, nRows: number, model: number, eps: number } | null}
 */
export function cmlField(tile, req) {
  if (!tile || tile.length < HEADER) return null;
  const nSites = Math.floor(Number(tile[1]));
  const nRows = Math.floor(Number(tile[3]));
  if (!(nSites >= 1) || !(nRows >= 1)) return null;
  if (tile.length < HEADER + nSites * nRows) return null;
  return {
    values: Float64Array.prototype.slice.call(tile, HEADER, HEADER + nSites * nRows),
    nSites,
    nRows,
    model: req.model,
    eps: req.eps,
  };
}

/**
 * Append a chunk's rows to the window and drop the oldest rows so the
 * window keeps `nRecord` rows. The chunk must continue the same run —
 * same model, eps and site count — or it is rejected: a parameter change
 * restarts the field through the runner instead.
 *
 * @param {{ values: ArrayLike<number>, nSites: number, nRows: number, model: number, eps: number }} field
 * @param {ArrayLike<number> | null | undefined} tile the chunk's kernel tile
 * @param {number} nRecord the window's row cap
 * @returns {{ values: Float64Array, nSites: number, nRows: number, model: number, eps: number } | null}
 */
export function cmlAppend(field, tile, nRecord) {
  if (!field || !tile || tile.length < HEADER) return null;
  const nSites = Math.floor(Number(tile[1]));
  const nRows = Math.floor(Number(tile[3]));
  if (nSites !== field.nSites || !(nRows >= 1)) return null;
  if (Math.floor(Number(tile[0])) !== field.model) return null;
  if (tile.length < HEADER + nSites * nRows) return null;
  const keep = Math.max(0, Math.min(field.nRows, nRecord - nRows));
  const values = new Float64Array((keep + nRows) * nSites);
  values.set(Array.prototype.slice.call(field.values, (field.nRows - keep) * nSites), 0);
  values.set(Array.prototype.slice.call(tile, HEADER, HEADER + nSites * nRows), keep * nSites);
  return { values, nSites, nRows: keep + nRows, model: field.model, eps: field.eps };
}

/**
 * The colour limits of the shown field: the 1st and 99th percentiles,
 * linear interpolation on the sorted values — the same rule plot() uses
 * on each row's stacked panels, including the ±0.5 pad when the field is
 * constant. Non-finite values are skipped; an empty field falls back to
 * [0, 1].
 *
 * @param {ArrayLike<number> | null | undefined} values
 * @returns {[number, number]}
 */
export function cmlColorLimits(values) {
  if (!values || !values.length) return [0, 1];
  const sorted = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return [0, 1];
  const at = (p) => {
    const h = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(h);
    const hi = Math.min(sorted.length - 1, lo + 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo);
  };
  let vmin = at(1);
  let vmax = at(99);
  if (vmax - vmin < 1e-12) {
    vmin -= 0.5;
    vmax += 0.5;
  }
  return [vmin, vmax];
}

/**
 * cml_spacetime_tile, loaded through the shared loader.
 *
 * @returns {Promise<(model: number, eps: number, nSites: number, nTransient: number, nRecord: number, x0: Float64Array) => Float64Array>}
 */
export function loadCmlKernel() {
  return loadWasmExport("cml_spacetime_tile");
}

/**
 * The default kernel call: load the glue, then invoke the export with the
 * request's fields in the kernel's own order — model, eps, the site
 * count, the transient, the record length, and the initial field.
 * `kernel` is injectable so a node test can check the argument order
 * without the wasm build.
 *
 * @param {ReturnType<typeof cmlRequest>} req
 * @param {Function} [kernel]
 * @returns {Promise<Float64Array>}
 */
export async function cmlKernelCall(req, kernel) {
  const fn = kernel || (await loadCmlKernel());
  return fn(req.model, req.eps, req.nSites, req.nTransient, req.nRecord, req.x0);
}

/**
 * The space-time figure's parameter wiring, recompute path and play
 * transport, sharing the attractor figures' runner. The page
 * (mountSpacetime in scripts/paper_shell.py) builds the inputs and hands
 * in the DOM callbacks; the decisions live here so node tests can drive
 * them.
 *
 * The model selector is the figure's one non-slider control. Switching
 * models snaps the value to a kernel id and resets eps to the new model's
 * published default — the union range would clamp a model-A eps to model
 * B's ceiling, which reads as a broken slider. An eps input snaps onto
 * the spec's 0.001 grid so the kernel sees the decimal literal, then
 * clamps into the active model's window.
 *
 * The store carries `field` (the rolling window) beside the trace arrays
 * the runner splices. `store.tiles` holds one record — the window as a
 * single heatmap tile — so the page's live renderer can reuse the tile
 * path unchanged. `store.generation` bumps on every published field, so
 * the colour cache and the readout never serve a stale window. `play()`
 * appends chunkRows at a time from the window's last row until `pause()`
 * or a parameter change; a chunk that resolves after the field it
 * continues was replaced is dropped, never merged.
 *
 * @param {object} deps
 * @param {object} [deps.params]
 * @param {ArrayLike<number>} [deps.x0] the seed-42 uniform field
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {(req: object) => Promise<ArrayLike<number>>} [deps.call]
 * @param {(store: object, field: object, req: object) => void} [deps.onField]
 * @param {() => void} deps.writeHash
 * @param {(err: unknown) => void} [deps.onError]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, store: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void, refresh: () => void, play: () => void, pause: () => void, playing: () => boolean, ready: () => boolean }}
 */
export function createSpacetimeFigure({
  params = LIVE_SPACETIME,
  x0 = [],
  debounceMs,
  echo,
  call = cmlKernelCall,
  onField,
  writeHash,
  onError,
  timers,
  document: doc,
}) {
  const store = {
    trace: { x: [], y: [] },
    tiles: [],
    field: null,
    zlim: [0, 1],
    generation: 0,
    painted: 0,
    playing: false,
  };
  // One counter for every published field: the runner's sequence and the
  // play transport's appends both draw from it, so a chunk can never stamp
  // a generation the colour cache already saw.
  let gen = 0;
  const publish = (field) => {
    store.field = field;
    store.zlim = cmlColorLimits(field.values);
    store.generation = ++gen;
    store.painted = field.nRows * field.nSites;
    // The window as one tile record: header [n_sites, n_rows] is the
    // raster path's shape, and data[4 + k] is field value k — the same
    // offset the kernel's own tile uses, so the renderer and the readout
    // read the record exactly like a pool tile.
    const data = new Float64Array(HEADER + field.values.length);
    data[0] = field.model;
    data[1] = field.nSites;
    data[2] = 0;
    data[3] = field.nRows;
    data.set(field.values, HEADER);
    store.tiles.splice(0, store.tiles.length, {
      id: "cml",
      level: 0,
      generation: store.generation,
      omegaMin: 0,
      omegaMax: field.nSites,
      kMin: 0,
      kMax: field.nRows,
      header: [field.nSites, field.nRows],
      data,
    });
  };
  const runner = createKernelRunner({
    store,
    getState: () => wiring.state,
    request: (state) => cmlRequest(state, params, x0),
    fold: (tile, req) => {
      const field = cmlField(tile, req);
      return { x: [], y: [], field };
    },
    call,
    onTrace: (s, trace, req, computeMs) => {
      if (!trace.field) return;
      publish(trace.field);
      if (onField) onField(s, trace.field, req, computeMs);
    },
    onError,
    document: doc,
  });
  const schedule = (fn) => {
    if (timers && timers.set) return timers.set(fn, 0);
    return setTimeout(fn, 0);
  };
  let playHeld = false;
  const unwatchPlay = watchVisibility(doc, (hidden) => {
    if (!hidden && playHeld && store.playing) {
      playHeld = false;
      schedule(playChunk);
    }
  });
  async function playChunk() {
    if (!store.playing) return;
    if (pageHidden(doc)) {
      playHeld = true;
      return;
    }
    const field = store.field;
    const req = cmlChunkRequest(wiring.state, field, params);
    if (!req) {
      store.playing = false;
      return;
    }
    try {
      const tile = await call(req);
      // The field this chunk continues may have been replaced while the
      // kernel ran (a parameter change restarts through the runner); a
      // stale chunk is dropped, never merged into the new window.
      if (!store.playing || store.field !== field) return;
      const next = cmlAppend(field, tile, params.nRecord);
      if (!next) return;
      publish(next);
      if (onField) onField(store, next, req);
      if (store.playing) schedule(playChunk);
    } catch (err) {
      store.playing = false;
      if (onError) onError(err);
    }
  }
  let activeKind = null;
  function syncState(name) {
    const kind = cmlModelKind(wiring.state.model);
    if (wiring.state.model !== kind) {
      wiring.state.model = kind;
      echo("model", kind);
    }
    const model = params.models[kind];
    if (kind !== activeKind) {
      if (name === "model") {
        // A model switch moves eps's window; reset to the new model's
        // published default rather than clamping the old model's eps.
        wiring.state.eps = model.epsDefault;
        echo("eps", model.epsDefault);
      }
      activeKind = kind;
    } else {
      const eps = Math.min(model.epsMax, Math.max(model.epsMin, wiring.state.eps));
      if (wiring.state.eps !== eps) {
        wiring.state.eps = eps;
        echo("eps", eps);
      }
    }
    // The snapped value is what the kernel sees: a stepped range input
    // emits 0.07000000000000001 where the reader picked 0.07.
    for (const spec of params.paramSpecs) {
      if (!spec.snap) continue;
      const snapped = snapToStep(wiring.state[spec.name], spec);
      if (wiring.state[spec.name] !== snapped) {
        wiring.state[spec.name] = snapped;
        echo(spec.name, snapped);
      }
    }
  }
  const wiring = createParamWiring({
    specs: params.paramSpecs,
    debounceMs,
    echo,
    apply: () => {
      runner.run();
    },
    writeHash,
    onSet: (name) => syncState(name),
    timers,
  });
  // Settle the initial state: the default model's window is active from
  // the start, so a first eps input clamps into it rather than skipping
  // the clamp on a null activeKind.
  syncState(null);
  return {
    state: wiring.state,
    store,
    onInput: wiring.onInput,
    setParams: wiring.setParams,
    refresh: () => {
      runner.run();
    },
    play: () => {
      if (store.playing) return;
      store.playing = true;
      schedule(playChunk);
    },
    pause: () => {
      store.playing = false;
      playHeld = false;
    },
    playing: () => store.playing,
    ready: runner.ready,
    destroy() {
      playHeld = false;
      unwatchPlay();
      runner.destroy();
    },
  };
}

/**
 * Short user agent for debug reporting: browser and version, without whitespace.
 *
 * @param {string} [ua]
 * @returns {string}
 */
export function shortUserAgent(ua = (typeof navigator !== "undefined" ? navigator.userAgent : "")) {
  if (typeof ua !== "string" || !ua.trim()) return "unknown";
  const edge = ua.match(/Edg(?:e|A|iOS)?\/(\d+)/);
  if (edge) return `Edge/${edge[1]}`;
  const version = ua.match(/Version\/(\d+(?:\.\d+)?)/);
  const chrome = ua.match(/(?:HeadlessChrome|Chrome|CriOS)\/(\d+)/);
  if (version && /Safari/.test(ua) && !chrome) {
    const isMobile = /Mobile/.test(ua) || /iPhone|iPad/.test(ua);
    return isMobile ? `MobileSafari/${version[1]}` : `Safari/${version[1]}`;
  }
  if (chrome) return `Chrome/${chrome[1]}`;
  const ff = ua.match(/(?:Firefox|FxiOS)\/(\d+)/);
  if (ff) return `Firefox/${ff[1]}`;
  const first = ua.trim().split(/\s+/)[0] || "unknown";
  return first.slice(0, 30);
}

/**
 * Standard median of an array of numeric values.
 *
 * @param {number[]} values
 * @returns {number}
 */
export function median(values) {
  if (!Array.isArray(values) || values.length === 0) return 0;
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * True when the PerformanceObserver interface supports the "longtask" entry type.
 *
 * @param {typeof PerformanceObserver} [PerformanceObserverImpl]
 * @returns {boolean}
 */
export function isLongTaskSupported(PerformanceObserverImpl = globalThis.PerformanceObserver) {
  try {
    if (!PerformanceObserverImpl) return false;
    const types = PerformanceObserverImpl.supportedEntryTypes;
    if (Array.isArray(types)) {
      return types.includes("longtask");
    }
  } catch {}
  return false;
}

/**
 * Start a longtask PerformanceObserver if supported. Where unsupported,
 * count() returns "n/a" (Safari). Never 0 where unsupported.
 *
 * @param {typeof PerformanceObserver} [PerformanceObserverImpl]
 * @param {number} [interactT0]
 * @returns {{ supported: boolean, count: () => number | "n/a", disconnect: () => void }}
 */
export function createLongTaskObserver(
  PerformanceObserverImpl = globalThis.PerformanceObserver,
  interactT0 = 0,
) {
  if (!isLongTaskSupported(PerformanceObserverImpl)) {
    return {
      supported: false,
      count: () => "n/a",
      disconnect: () => {},
    };
  }
  const t0 = Number.isFinite(interactT0) ? interactT0 : 0;
  let count = 0;
  let observer = null;
  function drain(records) {
    if (records) {
      for (const entry of records) {
        const start =
          entry && Number.isFinite(entry.startTime) ? entry.startTime : 0;
        if (
          entry &&
          Number.isFinite(entry.duration) &&
          entry.duration > 50 &&
          start >= t0
        ) {
          count++;
        }
      }
    }
  }
  try {
    observer = new PerformanceObserverImpl((list) => {
      drain(list.getEntries());
    });
    observer.observe({ type: "longtask", buffered: true });
  } catch {
    return {
      supported: false,
      count: () => "n/a",
      disconnect: () => {},
    };
  }
  return {
    supported: true,
    count: () => {
      if (observer && typeof observer.takeRecords === "function") {
        drain(observer.takeRecords());
      }
      return count;
    },
    disconnect: () => {
      if (observer) {
        try {
          if (typeof observer.takeRecords === "function") {
            drain(observer.takeRecords());
          }
          observer.disconnect();
        } catch {}
        observer = null;
      }
    },
  };
}

/**
 * One plain-text line holding the six fields in order: figure id, short user agent,
 * worker count, ms from interact to first full paint, median tile ms, and number
 * of main-thread tasks over 50 ms since interact.
 *
 * @param {object} fields
 * @param {string} fields.figId
 * @param {string} [fields.ua]
 * @param {number} fields.workerCount
 * @param {number} fields.elapsedMs
 * @param {number} fields.medianTileMs
 * @param {number | string} fields.longTasks
 * @returns {string}
 */
export function formatDebugReportLine({
  figId,
  ua,
  workerCount,
  elapsedMs,
  medianTileMs,
  longTasks,
}) {
  const id = typeof figId === "string" ? figId : "";
  const shortUa = shortUserAgent(ua);
  const workers = Number.isFinite(workerCount) ? Math.max(0, Math.floor(workerCount)) : 0;
  const elapsed = Number.isFinite(elapsedMs) ? Math.max(0, Math.round(elapsedMs)) : 0;
  const med = Number.isFinite(medianTileMs) ? Math.round(medianTileMs * 10) / 10 : 0;
  const tasks =
    typeof longTasks === "number" && Number.isFinite(longTasks)
      ? Math.max(0, Math.floor(longTasks))
      : longTasks === "n/a"
      ? "n/a"
      : longTasks != null
      ? String(longTasks)
      : "n/a";
  return `${id} ${shortUa} ${workers} ${elapsed} ${med} ${tasks}`;
}

/**
 * Attach the debug report box and copy button under a figure.
 *
 * @param {Element} fig
 * @param {string} lineText
 * @param {Document} [doc]
 * @returns {Element|null}
 */
export function attachDebugReportBox(fig, lineText, doc = globalThis.document) {
  if (!fig || !doc) return null;
  const body = (fig.querySelector && fig.querySelector(".fig-body")) || fig;
  const oldBox = body.querySelector ? body.querySelector(".live-debug-report") : null;
  if (oldBox) oldBox.remove();

  const box = doc.createElement("div");
  box.className = "live-debug-report";

  const lineEl = doc.createElement("span");
  lineEl.className = "live-debug-line";
  lineEl.textContent = lineText;
  box.appendChild(lineEl);

  const copyBtn = doc.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "live-debug-copy";
  copyBtn.textContent = "copy";
  box.appendChild(copyBtn);

  copyBtn.addEventListener("click", async () => {
    let copied = false;
    const nav = (doc.defaultView && doc.defaultView.navigator) || globalThis.navigator;
    try {
      if (nav && nav.clipboard && typeof nav.clipboard.writeText === "function") {
        await nav.clipboard.writeText(lineText);
        copied = true;
      }
    } catch {
      copied = false;
    }
    if (!copied) {
      try {
        const win = doc.defaultView || globalThis;
        const sel = win.getSelection ? win.getSelection() : null;
        if (sel && doc.createRange) {
          const range = doc.createRange();
          range.selectNodeContents(lineEl);
          sel.removeAllRanges();
          sel.addRange(range);
        }
      } catch {}
    }
  });

  body.appendChild(box);
  return box;
}

/**
 * Coordinate debug telemetry and DOM reporting for one figure session.
 *
 * @param {object} deps
 * @param {Element} deps.fig
 * @param {number} deps.interactT0
 * @param {boolean} [deps.debugEnabled]
 * @param {typeof PerformanceObserver} [deps.PerformanceObserverImpl]
 * @param {Document} [deps.doc]
 * @returns {{ onFirstPaintComplete: (info: { workerCount: number, tileTimes: number[] }) => void, destroy: () => void }}
 */
export function createDebugTracker({
  fig,
  interactT0,
  debugEnabled = false,
  PerformanceObserverImpl = globalThis.PerformanceObserver,
  doc = globalThis.document,
}) {
  if (!debugEnabled) {
    return {
      onFirstPaintComplete: () => {},
      destroy: () => {},
    };
  }

  const observer = createLongTaskObserver(PerformanceObserverImpl, interactT0);
  let done = false;

  function onFirstPaintComplete({ workerCount = 0, tileTimes = [] } = {}) {
    if (done) return;
    done = true;
    const now = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
    const elapsedMs = Math.max(0, now - interactT0);
    const medianTileMs = median(tileTimes);
    const longTasks = observer.count();
    observer.disconnect();

    const lineText = formatDebugReportLine({
      figId: fig ? fig.id : "",
      ua:
        (doc && doc.defaultView && doc.defaultView.navigator && doc.defaultView.navigator.userAgent) ||
        (typeof navigator !== "undefined" ? navigator.userAgent : ""),
      workerCount,
      elapsedMs,
      medianTileMs,
      longTasks,
    });

    if (fig && doc) {
      attachDebugReportBox(fig, lineText, doc);
      fig._debugReport = lineText;
    }
  }

  function destroy() {
    observer.disconnect();
  }

  return {
    onFirstPaintComplete,
    destroy,
  };
}

export const LIVE_HINT_SHORT = "drag to zoom · scroll or pinch · tap to read · reset view";
export const LIVE_HINT_SHORT_PLAY = "drag to zoom · play to step · tap to read · reset view";
export const LIVE_HINT_FULL =
  "tap to read values · drag to zoom · scroll or pinch to zoom · one finger to pan · reset view button to restore · focus the plot and use +/- to zoom, 0 or Esc to reset · computed live in this browser";
export const LIVE_HINT_FULL_PLAY =
  "tap to read values · drag to zoom · scroll or pinch to zoom · one finger to pan · reset view button to restore · focus the plot and use +/- to zoom, 0 or Esc to reset · play appends new rows and drops the oldest · computed live in this browser";

/**
 * Build the short interaction hint and the full list for a live figure.
 *
 * @param {{ play?: boolean }} [options]
 * @returns {{ shortText: string, fullText: string }}
 */
export function buildHint(options = {}) {
  const isPlay = Boolean(options && options.play);
  return {
    shortText: isPlay ? LIVE_HINT_SHORT_PLAY : LIVE_HINT_SHORT,
    fullText: isPlay ? LIVE_HINT_FULL_PLAY : LIVE_HINT_FULL,
  };
}
