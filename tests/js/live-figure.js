import { test } from "node:test";
import assert from "node:assert/strict";
import {
  LIVE_ARNOLD,
  LIVE_ATTRACTORS,
  LIVE_FIGURES,
  LIVE_MODULATED,
  LIVE_STAIRCASE,
  LIVE_SPACETIME,
  LIVE_TORUS,
  LIVE_HINT_FULL,
  LIVE_HINT_FULL_PLAY,
  LIVE_HINT_SHORT,
  LIVE_HINT_SHORT_PLAY,
  buildHint,
  attractorFixedPoint,
  attractorRequest,
  attractorTrace,
  createAttractorFigure,
  createLineYFit,
  createModulatedFigure,
  createModulatedReadout,
  createPaintHandler,
  createRangeView,
  createRasterSample,
  createReadout,
  createSchedulerOptions,
  createStaircaseReadout,
  createStaircaseView,
  createStore,
  createTorusFigure,
  createViewHandler,
  modulatedKernelCall,
  modulatedRequest,
  modulatedTrace,
  cmlAppend,
  cmlChunkRequest,
  cmlColorLimits,
  cmlField,
  cmlKernelCall,
  cmlModelKind,
  cmlRequest,
  createSpacetimeFigure,
  tilesToTrace,
  torusMapKind,
  torusRequest,
  torusTrace,
  attachDebugReportBox,
  createDebugTracker,
  createLongTaskObserver,
  formatDebugReportLine,
  isLongTaskSupported,
  median,
  shortUserAgent,
} from "../../site-src/live/live-figure.js";
import {
  HEADER,
  liveTileColorKey,
  sampleAt,
  tileWorld,
} from "../../site-src/live/raster.js";
import { visibleTiles } from "../../site-src/live/scheduler.js";
import { createPool, createKernelClient } from "../../site-src/live/pool.js";

const VIEWPORT = { omegaMin: 0, omegaMax: 1, kMin: 0, kMax: 0.3 };

function makeFakeWorkerClass() {
  const instances = [];
  class FakeWorker {
    constructor(url, options) {
      this.url = url;
      this.options = options;
      this.terminated = false;
      this.onmessage = null;
      this.onerror = null;
      this.posted = [];
      instances.push(this);
    }
    postMessage(msg) {
      this.posted.push(msg);
    }
    terminate() {
      this.terminated = true;
    }
  }
  FakeWorker.instances = instances;
  return FakeWorker;
}

function failWorker(worker) {
  if (typeof worker.onerror === "function") worker.onerror();
}

// A 2x2 tile over the whole viewport whose four cells hold distinct values,
// so a readout that samples a neighbouring cell returns a different number.
function makeTile(values = [0.1, 0.2, 0.3, 0.4]) {
  const data = new Float64Array(HEADER + 4);
  values.forEach((v, i) => {
    data[HEADER + i] = v;
  });
  return {
    ...tileWorld("0:0:0", VIEWPORT),
    generation: 1,
    header: [2, 2, 200, 2000],
    data,
  };
}

test("the hovered readout is the point kernel when it answers a finite number", () => {
  const store = createStore();
  let rasterCalls = 0;
  const point = { isReady: () => true, sample: () => 0.0743 };
  const raster = {
    sampleAt: () => {
      rasterCalls += 1;
      return 0.5;
    },
  };
  const readout = createReadout({ store, point, raster });
  assert.equal(readout(0.08, 0.03), 0.0743);
  assert.equal(rasterCalls, 0);
});

test("the readout falls back to the raster while the point kernel is not ready", () => {
  const store = createStore();
  let pointCalls = 0;
  const point = {
    isReady: () => false,
    sample: () => {
      pointCalls += 1;
      return 0.9;
    },
  };
  const raster = { sampleAt: () => 0.5 };
  const readout = createReadout({ store, point, raster });
  assert.equal(readout(0.08, 0.03), 0.5);
  assert.equal(pointCalls, 0);
});

test("the readout falls back to the raster when the point kernel throws", () => {
  const store = createStore();
  const point = {
    isReady: () => true,
    sample: () => {
      throw new Error("wasm glue exploded");
    },
  };
  const raster = { sampleAt: () => 0.5 };
  const readout = createReadout({ store, point, raster });
  assert.equal(readout(0.08, 0.03), 0.5);
});

test("the readout falls back to the raster when the point kernel returns a non-number", () => {
  const store = createStore();
  for (const bad of [null, undefined, "0.5", { value: 0.5 }]) {
    const point = { isReady: () => true, sample: () => bad };
    const raster = { sampleAt: () => 0.5 };
    const readout = createReadout({ store, point, raster });
    assert.equal(readout(0.08, 0.03), 0.5, `fallback for ${String(bad)}`);
  }
});

test("the readout falls back to the raster when the point kernel returns NaN", () => {
  const store = createStore();
  const point = { isReady: () => true, sample: () => NaN };
  const raster = { sampleAt: () => 0.5 };
  const readout = createReadout({ store, point, raster });
  // Without the Number.isFinite check this returns NaN, not the raster value.
  assert.equal(readout(0.08, 0.03), 0.5);
});

test("the readout and the raster comparison read the same (Omega, K)", () => {
  const store = createStore();
  store.generation = 1;
  store.tiles.push(makeTile());
  const calls = [];
  const raster = {
    sampleAt: (tiles, generation, omega, K) => {
      calls.push({ tiles, generation, omega, K });
      return sampleAt(tiles, generation, omega, K);
    },
  };
  const point = { isReady: () => false, sample: () => 0.9 };
  const readout = createReadout({ store, point, raster });
  const rasterSample = createRasterSample({ store, raster });
  // (0.08, 0.03) sits in cell (0, 0) of the 2x2 tile. A readout that samples
  // a shifted point lands in a neighbouring cell and returns 0.2, not 0.1.
  const readoutValue = readout(0.08, 0.03);
  const rasterValue = rasterSample(0.08, 0.03);
  assert.equal(readoutValue, 0.1);
  assert.equal(rasterValue, 0.1);
  assert.equal(readoutValue, rasterValue);
  assert.equal(calls.length, 2);
  for (const call of calls) {
    assert.equal(call.tiles, store.tiles);
    assert.equal(call.generation, 1);
    assert.equal(call.omega, 0.08);
    assert.equal(call.K, 0.03);
  }
});

test("the figure's onPaint stamps each stored record with a paint sequence", () => {
  // Without the stamp every record keys with an empty paint sequence and the
  // stale-bitmap collision liveTileColorKey was built for comes back.
  const store = createStore();
  const pool = { getState: () => ({ generation: 1, viewport: VIEWPORT }) };
  const raster = { tileWorld };
  let redraws = 0;
  const plot = { redraw: () => { redraws += 1; } };
  const onPaint = createPaintHandler({
    store,
    raster,
    getPool: () => pool,
    getPlot: () => plot,
  });
  const data = new Float64Array(HEADER + 4);
  data[0] = 2;
  data[1] = 2;
  data[HEADER] = 0.1;
  data[HEADER + 3] = 0.9;
  const cmd = { id: "0:0:0", generation: 1, header: [2, 2, 200, 1500], data };
  // The same tile painted twice in one generation replaces its record; the
  // replacement must key differently or the cached bitmap goes stale.
  onPaint(cmd);
  const firstKey = liveTileColorKey(store.tiles[0]);
  onPaint(cmd);
  assert.equal(store.tiles.length, 1);
  assert.equal(store.tiles[0].id, "0:0:0");
  assert.equal(store.tiles[0].paintSeq, 2);
  assert.equal(store.generation, 1);
  assert.equal(store.painted, 1);
  assert.equal(redraws, 2);
  // nIter rides in on header[3]; the store started at 2000.
  assert.equal(store.nIter, 1500);
  const stamped = liveTileColorKey(store.tiles[0]);
  assert.notEqual(stamped, firstKey);
  const { paintSeq, ...unstamped } = store.tiles[0];
  assert.notEqual(stamped, liveTileColorKey(unstamped));
});

test("onPaint ignores a stale generation and a tile outside the viewport", () => {
  const store = createStore();
  const pool = { getState: () => ({ generation: 2, viewport: VIEWPORT }) };
  const raster = { tileWorld };
  let redraws = 0;
  const onPaint = createPaintHandler({
    store,
    raster,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => { redraws += 1; } }),
  });
  onPaint({ id: "0:0:0", generation: 1, header: [2, 2, 200, 2000], data: [] });
  onPaint({ id: "not-a-tile", generation: 2, header: [2, 2, 200, 2000], data: [] });
  assert.equal(store.tiles.length, 0);
  assert.equal(store.painted, 0);
  assert.equal(redraws, 0);
});

test("onView pushes the domain into the pool and resets the store on a new generation", () => {
  const store = createStore();
  store.generation = 1;
  store.painted = 3;
  store.tiles.push(makeTile());
  const seen = [];
  const pool = {
    setViewport: (v) => seen.push(v),
    getState: () => ({ generation: 2 }),
  };
  let redraws = 0;
  const onView = createViewHandler({
    store,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => { redraws += 1; } }),
  });
  onView({ x0: 0, x1: 0.5, y0: 0, y1: 0.15 });
  assert.deepEqual(seen, [{ omegaMin: 0, omegaMax: 0.5, kMin: 0, kMax: 0.15 }]);
  assert.equal(store.generation, 2);
  assert.equal(store.painted, 0);
  assert.equal(store.tiles.length, 0);
  assert.equal(redraws, 1);
});

test("onView keeps the store when the generation did not change, and tolerates no pool", () => {
  const store = createStore();
  store.generation = 2;
  store.painted = 1;
  store.tiles.push(makeTile());
  let redraws = 0;
  const plot = { redraw: () => { redraws += 1; } };
  const pool = {
    setViewport: () => {},
    getState: () => ({ generation: 2 }),
  };
  const onView = createViewHandler({
    store,
    getPool: () => pool,
    getPlot: () => plot,
  });
  onView({ x0: 0, x1: 0.5, y0: 0, y1: 0.15 });
  assert.equal(store.tiles.length, 1);
  assert.equal(store.painted, 1);
  assert.equal(redraws, 0);
  const noPool = createViewHandler({
    store,
    getPool: () => null,
    getPlot: () => plot,
  });
  noPool({ x0: 0, x1: 0.5, y0: 0, y1: 0.15 });
  assert.equal(store.tiles.length, 1);
});

test("a tile that spent its retry is retried again after onView pans", () => {
  // The retry list lives in the scheduler state and resets on a new
  // generation; onView is the glue that turns a pan into that generation.
  // Drive a real pool so the reset is observed through the same path the
  // page uses, not through a second copy of the wiring.
  const FakeWorker = makeFakeWorkerClass();
  const store = createStore();
  let redraws = 0;
  let pool = null;
  const onView = createViewHandler({
    store,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => { redraws += 1; } }),
  });
  pool = createPool({
    Worker: FakeWorker,
    hardwareConcurrency: 4,
    workerUrl: "fake",
    scheduler: { levels: 1, tileCells: 8 },
  });
  onView({ x0: 0, x1: 1, y0: 0, y1: 0.3 });
  assert.equal(store.generation, 1);
  const [w0, w1, w2, w3] = FakeWorker.instances;
  const tile = w0.posted[0];
  // Lose the tile once in generation 1: it is requeued to a live worker and
  // spends its one retry.
  failWorker(w0);
  assert.equal(w1.posted.length, 1);
  assert.equal(w1.posted[0].id, tile.id);
  // Lose it again: the per-generation cap drops it for good.
  failWorker(w1);
  assert.equal(pool.getState().droppedTiles, 1);
  assert.equal(w2.posted.length, 0);
  // Pan through the glue: generation 2 resets the store and the retry list.
  onView({ x0: 0, x1: 0.5, y0: 0, y1: 0.3 });
  assert.equal(store.generation, 2);
  assert.equal(store.tiles.length, 0);
  assert.equal(store.painted, 0);
  assert.equal(redraws, 2);
  assert.equal(w2.posted.length, 1);
  assert.equal(w2.posted[0].id, tile.id);
  assert.equal(w2.posted[0].generation, 2);
  // Without the per-generation reset of the retry list this third loss drops
  // the tile instead of requeueing it to the last live worker.
  failWorker(w2);
  assert.equal(w3.posted.length, 1);
  assert.equal(w3.posted[0].id, tile.id);
  assert.equal(w3.posted[0].generation, 2);
  assert.equal(pool.getState().droppedTiles, 1);
  pool.destroy();
});

test("LIVE_ARNOLD is frozen and holds the published figure's parameters", () => {
  assert.ok(Object.isFrozen(LIVE_ARNOLD));
  assert.deepEqual({ ...LIVE_ARNOLD }, { nTransient: 200, nIter: 2000, theta0: 0.1 });
  // The store starts at the same nIter the tiles will be computed with.
  assert.equal(createStore().nIter, LIVE_ARNOLD.nIter);
});

test("one params object reaches both the readout and the scheduler options", () => {
  const params = { nTransient: 11, nIter: 222, theta0: 0.33 };
  const store = createStore();
  const calls = [];
  const point = {
    isReady: () => true,
    sample: (...args) => {
      calls.push(args);
      return 0.0743;
    },
  };
  const raster = { sampleAt: () => 0.5 };
  const readout = createReadout({ store, point, raster, params });
  readout(0.08, 0.03);
  assert.deepEqual(calls, [[0.08, 0.03, 11, 222, 0.33]]);
  const scheduler = createSchedulerOptions({ levels: 5, tileCells: 32, params });
  assert.deepEqual(scheduler, {
    levels: 5,
    tileCells: 32,
    nTransient: 11,
    nIter: 222,
    theta0: 0.33,
  });
  // Without params both fall back to LIVE_ARNOLD, so a hard-coded value in
  // either path is caught here too.
  const defaultCalls = [];
  const defaultReadout = createReadout({
    store,
    point: {
      isReady: () => true,
      sample: (...args) => {
        defaultCalls.push(args);
        return 0.1;
      },
    },
    raster,
  });
  defaultReadout(0.08, 0.03);
  assert.deepEqual(defaultCalls, [[0.08, 0.03, 200, 2000, 0.1]]);
  assert.deepEqual(createSchedulerOptions({ levels: 5, tileCells: 32 }), {
    levels: 5,
    tileCells: 32,
    nTransient: 200,
    nIter: 2000,
    theta0: 0.1,
  });
});

test("lockOmega tiles fix Omega at the viewport value and sweep only K", () => {
  // The staircase's tile request: Omega = D with nOmega = 1, K = A over the
  // visible range. Swapping Omega and K in the request is caught here: the
  // tile would carry the A range on the Omega axis instead.
  const viewport = { omegaMin: 0.25, omegaMax: 0.25, kMin: 0, kMax: 0.25 };
  const tiles = visibleTiles(viewport, { levels: 3, tileCells: 8, lockOmega: true });
  // 1 + 2 + 4 tiles: the pyramid refines K only.
  assert.equal(tiles.length, 7);
  for (const t of tiles) {
    assert.equal(t.omegaMin, 0.25);
    assert.equal(t.omegaMax, 0.25);
    assert.equal(t.nOmega, 1);
    assert.ok(t.kMin >= 0 && t.kMax <= 0.25 && t.kMax > t.kMin);
  }
  const finest = tiles.filter((t) => t.level === 2);
  assert.equal(finest.length, 4);
  assert.equal(finest[0].kMin, 0);
  assert.equal(finest[3].kMax, 0.25);
  // A degenerate Omega range is still rejected without lockOmega.
  assert.equal(visibleTiles(viewport, { levels: 3, tileCells: 8 }).length, 0);
});

test("the staircase view pushes (D, A range) and resets the store once", () => {
  const store = createStore(LIVE_STAIRCASE);
  store.generation = 1;
  store.tiles.push({ id: "0:0:0" });
  const seen = [];
  let gen = 1;
  const pool = {
    setViewport: (v) => {
      seen.push(v);
      gen += 1;
    },
    getState: () => ({ generation: gen }),
  };
  let D = 0.25;
  const dom = { x0: 0, x1: 0.25, y0: 0, y1: 1 };
  const view = createStaircaseView({
    store,
    getPool: () => pool,
    getPlot: () => ({ getDomain: () => dom, redraw: () => {} }),
    getD: () => D,
  });
  view.onView({ x0: 0, x1: 0.25 });
  assert.deepEqual(seen, [{ omegaMin: 0.25, omegaMax: 0.25, kMin: 0, kMax: 0.25 }]);
  assert.equal(store.generation, 2);
  assert.equal(store.tiles.length, 0);
  // A y-only domain change must not recompute.
  view.onView({ x0: 0, x1: 0.25 });
  assert.equal(seen.length, 1);
  // Moving D pushes the same A range at the new Omega.
  D = 0.4;
  view.applyD();
  assert.deepEqual(seen[1], { omegaMin: 0.4, omegaMax: 0.4, kMin: 0, kMax: 0.25 });
  assert.equal(store.generation, 3);
});

test("the staircase y-fit follows a wider tile inside one generation", () => {
  const store = { generation: 1, trace: { x: [0, 0.1], y: [0.4, 0.6] } };
  let dom = { x0: 0, x1: 0.25, y0: 0, y1: 1 };
  const base = { ...dom };
  const calls = [];
  const fit = createLineYFit({
    store,
    getPlot: () => ({
      getDomain: () => ({ ...dom }),
      getBase: () => ({ ...base }),
      setDomain: (d, opts) => {
        calls.push({ d, opts });
        dom = { ...dom, ...d };
      },
    }),
  });
  fit();
  assert.equal(calls.length, 1);
  assert.ok(dom.y0 > 0 && dom.y0 <= 0.4);
  assert.ok(dom.y1 < 1 && dom.y1 >= 0.6);
  // A finer tile widens the trace: the fit must move again, not freeze on
  // the first coarse tile.
  store.trace.x.push(0.05);
  store.trace.y.push(0.1);
  fit();
  assert.equal(calls.length, 2);
  assert.ok(dom.y0 <= 0.1);
});

test("the staircase y-fit leaves a reader-set domain alone", () => {
  const store = { generation: 1, trace: { x: [0, 0.1], y: [0.4, 0.6] } };
  let dom = { x0: 0, x1: 0.25, y0: 0, y1: 1 };
  const base = { ...dom };
  const calls = [];
  const fit = createLineYFit({
    store,
    getPlot: () => ({
      getDomain: () => ({ ...dom }),
      getBase: () => ({ ...base }),
      setDomain: (d, opts) => {
        calls.push({ d, opts });
        dom = { ...dom, ...d };
      },
    }),
  });
  fit();
  assert.equal(calls.length, 1);
  // The reader zooms the y range, then a wider tile lands: the view is the
  // reader's now, so no more fitting — not even on a new generation.
  dom = { ...dom, y0: 0.45, y1: 0.55 };
  store.trace.y.push(0.1);
  fit();
  assert.equal(calls.length, 1);
  store.generation = 2;
  store.trace.y.push(0.9);
  fit();
  assert.equal(calls.length, 1);
  // A reader zoom that lands before the first paint stops the first fit too.
  const store2 = { generation: 1, trace: { x: [0, 0.1], y: [0.4, 0.6] } };
  let dom2 = { x0: 0, x1: 0.25, y0: 0.2, y1: 0.7 };
  const base2 = { x0: 0, x1: 0.25, y0: 0, y1: 1 };
  const calls2 = [];
  const fit2 = createLineYFit({
    store: store2,
    getPlot: () => ({
      getDomain: () => ({ ...dom2 }),
      getBase: () => ({ ...base2 }),
      setDomain: (d, opts) => {
        calls2.push({ d, opts });
        dom2 = { ...dom2, ...d };
      },
    }),
  });
  fit2();
  assert.equal(calls2.length, 0);
});

test("the staircase y-fit sets the domain silently, off the domain-change hook", () => {
  const store = { generation: 1, trace: { x: [0, 0.1], y: [0.4, 0.6] } };
  let dom = { x0: 0, x1: 0.25, y0: 0, y1: 1 };
  const base = { ...dom };
  const calls = [];
  const fit = createLineYFit({
    store,
    getPlot: () => ({
      getDomain: () => ({ ...dom }),
      getBase: () => ({ ...base }),
      setDomain: (d, opts) => {
        calls.push({ d, opts });
        dom = { ...dom, ...d };
      },
    }),
  });
  fit();
  assert.equal(calls.length, 1);
  // silent:true keeps the set off afterDomain, so figState.notify never
  // fires and no ?fig= reaches the URL.
  assert.equal(calls[0].opts && calls[0].opts.silent, true);
  assert.equal(calls[0].d.x0, 0);
  assert.equal(calls[0].d.x1, 0.25);
  assert.ok(calls[0].d.y0 <= 0.4 && calls[0].d.y1 >= 0.6);
});

test("tilesToTrace folds 1-D tiles into a sorted polyline, finest level winning", () => {
  const viewport = { omegaMin: 0.25, omegaMax: 0.25, kMin: 0, kMax: 0.25 };
  const issued = visibleTiles(viewport, { levels: 2, tileCells: 4, lockOmega: true });
  const tiles = issued.map((t) => {
    const nK = t.nK;
    const data = new Float64Array(HEADER + nK);
    for (let iy = 0; iy < nK; iy++) {
      const a = nK > 1 ? t.kMin + (iy * (t.kMax - t.kMin)) / (nK - 1) : t.kMin;
      data[HEADER + iy] = a * 10 + t.level; // level-tagged value
    }
    return { ...t, generation: 1, header: [1, nK, 5000, 50000], data };
  });
  const trace = tilesToTrace(tiles, 1);
  assert.equal(trace.x.length, trace.y.length);
  assert.ok(trace.x.length > 4);
  for (let i = 1; i < trace.x.length; i++) assert.ok(trace.x[i] > trace.x[i - 1]);
  // A shared boundary coordinate keeps the finer tile's value (level 1).
  const mid = trace.x.indexOf(0.125);
  assert.ok(mid >= 0);
  assert.equal(trace.y[mid], 0.125 * 10 + 1);
  // Stale-generation tiles are ignored.
  assert.equal(tilesToTrace(tiles, 2).x.length, 0);
});

test("the staircase readout samples (D, A) through the point kernel", () => {
  const store = createStore(LIVE_STAIRCASE);
  const calls = [];
  const point = {
    isReady: () => true,
    sample: (...args) => {
      calls.push(args);
      return 0.2;
    },
  };
  const raster = { sampleAt: () => 0.9 };
  const readout = createStaircaseReadout({ store, point, raster, getD: () => 0.4 });
  assert.equal(readout(0.12), 0.2);
  assert.deepEqual(calls, [[0.4, 0.12, 5000, 50000, 0.1]]);
});

test("the staircase scheduler options carry lockOmega and the paper's counts", () => {
  const opts = createSchedulerOptions({ levels: 5, tileCells: 64, params: LIVE_STAIRCASE, lockOmega: true });
  assert.deepEqual(opts, {
    levels: 5,
    tileCells: 64,
    nTransient: 5000,
    nIter: 50000,
    theta0: 0.1,
    lockOmega: true,
  });
});

// A manual timer pair, as in tests/js/params.js: scheduled callbacks queue
// up and run only when the test flushes them, so the debounce is checked
// without real time.
function manualTimers() {
  const queue = [];
  return {
    queue,
    set: (fn) => {
      queue.push(fn);
      return queue.length - 1;
    },
    clear: (id) => {
      queue[id] = null;
    },
    flush() {
      const fns = queue.splice(0);
      for (const fn of fns) if (fn) fn();
    },
  };
}

const tick = () => new Promise((r) => setImmediate(r));

function fakeDocument() {
  const listeners = [];
  return {
    hidden: false,
    visibilityState: "visible",
    addEventListener(type, fn) {
      if (type === "visibilitychange") listeners.push(fn);
    },
    removeEventListener(type, fn) {
      const i = listeners.indexOf(fn);
      if (i >= 0) listeners.splice(i, 1);
    },
    hide() {
      this.hidden = true;
      this.visibilityState = "hidden";
      for (const fn of listeners.slice()) fn();
    },
    show() {
      this.hidden = false;
      this.visibilityState = "visible";
      for (const fn of listeners.slice()) fn();
    },
  };
}

test("a hidden document pauses the kernel runner and drops the in-flight call", async () => {
  const doc = fakeDocument();
  const releases = [];
  const fig = createAttractorFigure({
    debounceMs: 0,
    echo: () => {},
    onTrace: () => {},
    writeHash: () => {},
    document: doc,
    call: () =>
      new Promise((resolve) => {
        releases.push(resolve);
      }),
  });
  fig.refresh();
  assert.equal(releases.length, 1);
  doc.hide();
  fig.refresh();
  assert.equal(releases.length, 1);
  releases[0](new Float64Array([1, 2, 20000, 2, 0.1, 0.2, 0.3, 0.4]));
  await tick();
  assert.equal(fig.store.painted, 0);
  doc.show();
  assert.equal(releases.length, 2);
  releases[1](new Float64Array([1, 2, 20000, 2, 0.1, 0.2, 0.3, 0.4]));
  await tick();
  assert.equal(fig.store.painted, 2);
  fig.destroy();
  doc.show();
  assert.equal(releases.length, 2);
});

test("LIVE_ATTRACTORS is frozen, registered, and holds the paper's parameters", () => {
  assert.equal(LIVE_FIGURES.delayed_logistic_attractors, LIVE_ATTRACTORS);
  assert.equal(LIVE_ATTRACTORS.A, 0.3);
  assert.equal(LIVE_ATTRACTORS.nTransient, 20000);
  const [d, n] = LIVE_ATTRACTORS.paramSpecs;
  // The kernel clamps D to [1.4, 3.5], but the orbit escapes to infinity
  // past D = 2.8155 and a NaN block paints a blank plot. The slider's max
  // is the measured boundary minus a 0.05 margin, snapped to the 0.005
  // grid, so every value the control can ask for stays bounded.
  assert.deepEqual(
    { name: d.name, min: d.min, max: d.max, default: d.default },
    { name: "D", min: 1.4, max: 2.765, default: 1.9 },
  );
  // The iteration-count control is bounded by the kernel's 4096 cap.
  assert.equal(n.name, "n");
  assert.equal(n.max, 4096);
  assert.ok(Object.isFrozen(LIVE_ATTRACTORS));
});

test("the attractor request passes A then D in the kernel's order", () => {
  // delayed_logistic_attractor_tile(a, d_min, d_max, n_d, n_transient,
  // n_plot, state0): swapping A and D here sends the slider's D into the
  // map's A slot, which this test catches.
  const req = attractorRequest({ D: 1.9, n: 2048 });
  assert.equal(req.a, 0.3);
  assert.equal(req.dMin, 1.9);
  assert.equal(req.dMax, 1.9);
  assert.equal(req.nD, 1);
  assert.equal(req.nTransient, 20000);
  assert.equal(req.nPlot, 2048);
  // The published x0 convention: (fp + 0.01, fp - 0.01) at the analytic
  // fixed point fp = (sqrt(1 + 4D) - 1) / 2D = 0.508572542032378 at D=1.9.
  const fp = attractorFixedPoint(1.9);
  assert.equal(fp, 0.508572542032378);
  assert.equal(req.state0[0], fp + 0.01);
  assert.equal(req.state0[1], fp - 0.01);
});

test("attractorTrace folds the tile into x then y pairs", () => {
  // Header [n_D, n_plot, n_transient, 2], then n_plot (x, y) pairs. An x/y
  // swap in the fold lands y on the x axis, which this test catches.
  const tile = new Float64Array([1, 2, 20000, 2, 0.1, 0.2, 0.3, 0.4]);
  const t = attractorTrace(tile);
  assert.deepEqual(t.x, [0.1, 0.3]);
  assert.deepEqual(t.y, [0.2, 0.4]);
});

test("attractorTrace folds every D block and rejects a short tile", () => {
  const two = new Float64Array([2, 1, 20000, 2, 0.5, 0.6, 0.7, 0.8]);
  const t = attractorTrace(two);
  assert.deepEqual(t.x, [0.5, 0.7]);
  assert.deepEqual(t.y, [0.6, 0.8]);
  assert.deepEqual(attractorTrace(new Float64Array([1, 9, 0, 2, 0.1])).x, []);
  assert.deepEqual(attractorTrace(null).x, []);
});

test("a slider burst recomputes once and a stale result is dropped", async () => {
  const timers = manualTimers();
  const calls = [];
  const deferreds = [];
  const fig = createAttractorFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return new Promise((r) => deferreds.push(r));
    },
    onTrace: () => {},
    writeHash: () => {},
    timers,
  });
  for (const v of [1.9, 1.95, 2.0]) fig.onInput("D", v);
  timers.flush();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].dMin, 2.0);

  // Two more requests in flight; the older one resolving last must not
  // overwrite the newer result.
  fig.setParams({ D: 1.86 });
  fig.setParams({ D: 2.16 });
  assert.equal(calls.length, 3);
  const tileFor = (d) => new Float64Array([1, 1, 20000, 2, d, -d]);
  deferreds[2](tileFor(2.16));
  await tick();
  deferreds[1](tileFor(1.86));
  await tick();
  assert.deepEqual(fig.store.trace.x, [2.16]);
  assert.deepEqual(fig.store.trace.y, [-2.16]);
  assert.equal(fig.store.generation, 3);
});

test("onTrace runs once per painted result and never for a stale one", async () => {
  // The page repaints and draws the fixed-point marker in onTrace; in
  // fe8a64d it was accepted but never called, and no test noticed.
  const deferreds = [];
  const traced = [];
  const fig = createAttractorFigure({
    debounceMs: 150,
    echo: () => {},
    call: () => new Promise((r) => deferreds.push(r)),
    onTrace: (store, trace, req) => traced.push([trace.x[0], req.dMin]),
    writeHash: () => {},
    timers: manualTimers(),
  });
  fig.setParams({ D: 1.86 });
  fig.setParams({ D: 2.16 });
  const tileFor = (d) => new Float64Array([1, 1, 20000, 2, d, -d]);
  deferreds[1](tileFor(2.16));
  await tick();
  deferreds[0](tileFor(1.86));
  await tick();
  assert.deepEqual(traced, [[2.16, 2.16]]);
});

test("the attractor store's trace arrays keep their identity across paints", async () => {
  // The page's panel.traces captures these arrays at mount; replacing them
  // would leave the plot drawing the empty arrays forever.
  const fig = createAttractorFigure({
    debounceMs: 150,
    echo: () => {},
    call: async () => new Float64Array([1, 2, 20000, 2, 0.1, 0.2, 0.3, 0.4]),
    onTrace: () => {},
    writeHash: () => {},
    timers: manualTimers(),
  });
  const xs = fig.store.trace.x;
  const ys = fig.store.trace.y;
  fig.refresh();
  await tick();
  fig.refresh();
  await tick();
  assert.equal(fig.store.trace.x, xs);
  assert.equal(fig.store.trace.y, ys);
  assert.deepEqual([...xs], [0.1, 0.3]);
});

test("every delayed-logistic slider value keeps a finite orbit", async () => {
  // The kernel itself decides: for each D on the slider grid, the request
  // the page would send must return 4096 fully finite states. A max pushed
  // back past the escape boundary — where the kernel writes a NaN block —
  // fails here before a reader ever sees the blank plot. The glue file is
  // the same one the site serves, site/wasm/dynachaos_wasm.js, loaded the
  // way live-figure.js's loader does under node.
  const [spec] = LIVE_ATTRACTORS.paramSpecs;
  let kernel = null;
  try {
    const glue = await import("../../site/wasm/dynachaos_wasm.js");
    const { readFile } = await import("node:fs/promises");
    const bytes = await readFile(new URL("../../site/wasm/dynachaos_wasm_bg.wasm", import.meta.url));
    glue.initSync({ module: bytes });
    kernel = glue.delayed_logistic_attractor_tile;
  } catch {
    kernel = null;
  }
  if (kernel) {
    for (let k = 0, n = Math.round((spec.max - spec.min) / spec.step); k <= n; k++) {
      const D = spec.min + k * spec.step;
      const req = attractorRequest({ D, n: 4096 });
      const tile = kernel(req.a, req.dMin, req.dMax, req.nD, req.nTransient, req.nPlot, req.state0);
      const nPlot = Math.floor(tile[1]);
      assert.ok(nPlot >= 1, `empty tile at D=${D}`);
      for (let i = 0; i < nPlot; i++) {
        assert.ok(
          Number.isFinite(tile[4 + 2 * i]) && Number.isFinite(tile[4 + 2 * i + 1]),
          `non-finite state at D=${D}`,
        );
      }
    }
    return;
  }
  // No build artifact under site/wasm: pin the measurement instead. The
  // kernel's orbit first escapes at D = 2.8155 (2026-10-08, A = 0.3,
  // n_transient 20000, step 0.0005), so the slider max must stay at least
  // 0.05 below it; the e2e check at the max exercises the real kernel.
  assert.ok(spec.max <= 2.8155 - 0.05, `slider max ${spec.max} reaches the escape boundary`);
});

test("LIVE_TORUS is frozen, registered, and holds the paper's parameters", () => {
  assert.equal(LIVE_FIGURES.torus_doubling_attractors, LIVE_TORUS);
  assert.equal(LIVE_TORUS.nTransient, 20000);
  // The paper's per-map constants (maps/torus_doubling.py): map I at
  // A = 0.4 from (0.5, 0.5, 0.5), map IV at A = 0.3 from (0.5, 0.45, 0.52,
  // 0.48); each D window is the published sweep inside the kernel's
  // [1.48, 2.25] clamp, and each default is a published panel.
  const m1 = LIVE_TORUS.maps[1];
  const m4 = LIVE_TORUS.maps[4];
  assert.equal(m1.a, 0.4);
  assert.deepEqual([...m1.x0], [0.5, 0.5, 0.5]);
  assert.deepEqual({ dMin: m1.dMin, dMax: m1.dMax, dDefault: m1.dDefault }, { dMin: 1.9, dMax: 2.25, dDefault: 2.16 });
  assert.equal(m4.a, 0.3);
  assert.deepEqual([...m4.x0], [0.5, 0.45, 0.52, 0.48]);
  assert.deepEqual({ dMin: m4.dMin, dMax: m4.dMax, dDefault: m4.dDefault }, { dMin: 1.48, dMax: 1.53, dDefault: 1.5206 });
  const [map, d, n] = LIVE_TORUS.paramSpecs;
  assert.equal(map.name, "map");
  assert.deepEqual(map.options.map((o) => o.value), [1, 4]);
  assert.equal(d.name, "D");
  assert.equal(n.name, "n");
  assert.equal(n.max, 4096);
  assert.ok(Object.isFrozen(LIVE_TORUS));
});

test("the torus request passes map kind, A, D in the kernel's order", () => {
  // torus_doubling_attractor_tile(map_kind, a, d, n_transient, n_plot,
  // state0): swapping A and D here sends the slider's D into the map's A
  // slot, and dropping the map kind runs the wrong map — this test catches
  // both.
  const req1 = torusRequest({ map: 1, D: 2.16, n: 2048 });
  assert.equal(req1.mapKind, 1);
  assert.equal(req1.a, 0.4);
  assert.equal(req1.d, 2.16);
  assert.equal(req1.nTransient, 20000);
  assert.equal(req1.nPlot, 2048);
  assert.deepEqual([...req1.state0], [0.5, 0.5, 0.5]);

  const req4 = torusRequest({ map: 4, D: 1.5206, n: 512 });
  assert.equal(req4.mapKind, 4);
  assert.equal(req4.a, 0.3);
  assert.equal(req4.d, 1.5206);
  assert.equal(req4.nPlot, 512);
  assert.deepEqual([...req4.state0], [0.5, 0.45, 0.52, 0.48]);

  // A D outside the chosen map's window clamps into it before the kernel
  // ever sees it.
  assert.equal(torusRequest({ map: 4, D: 2.16, n: 512 }).d, 1.53);
  assert.equal(torusRequest({ map: 1, D: 1.5, n: 512 }).d, 1.9);
});

test("torusMapKind snaps the selector the way the kernel does", () => {
  assert.equal(torusMapKind(1), 1);
  assert.equal(torusMapKind(2), 1);
  assert.equal(torusMapKind(3), 4);
  assert.equal(torusMapKind(4), 4);
  assert.equal(torusMapKind(0), 1);
});

test("torusTrace folds the tile by the map's projection", () => {
  // Header [map_kind, dim, n_transient, n_produced], then n_produced states
  // of dim components. The paper scatters components 0 and 1 (X, Y); a
  // swapped projection lands Y on the x axis, which this test catches.
  const tile3 = new Float64Array([1, 3, 20000, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  const t3 = torusTrace(tile3, LIVE_TORUS.maps[1]);
  assert.deepEqual(t3.x, [0.1, 0.4]);
  assert.deepEqual(t3.y, [0.2, 0.5]);

  const tile4 = new Float64Array([4, 4, 20000, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
  const t4 = torusTrace(tile4, LIVE_TORUS.maps[4]);
  assert.deepEqual(t4.x, [0.1, 0.5]);
  assert.deepEqual(t4.y, [0.2, 0.6]);

  // A short tile, a null tile and a diverged orbit all yield an empty
  // cloud rather than a misread buffer.
  assert.deepEqual(torusTrace(new Float64Array([1, 3, 0, 9, 0.1]), LIVE_TORUS.maps[1]).x, []);
  assert.deepEqual(torusTrace(null, LIVE_TORUS.maps[1]).x, []);
  assert.deepEqual(torusTrace(new Float64Array([4, 4, 20000, 0]), LIVE_TORUS.maps[4]).x, []);
});

test("switching the torus map resets D to the new map's window and repaints", async () => {
  const timers = manualTimers();
  const calls = [];
  const fig = createTorusFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return Promise.resolve(new Float64Array([req.mapKind, req.mapKind === 1 ? 3 : 4, 20000, 1, 0.1, 0.2, 0.3, 0.4]));
    },
    onTrace: () => {},
    writeHash: () => {},
    timers,
  });
  fig.onInput("map", 4);
  timers.flush();
  await tick();
  // The selector snapped to map IV and D moved to that map's published
  // default — a clamp would have pinned it to the window's edge instead.
  assert.equal(fig.state.map, 4);
  assert.equal(fig.state.D, 1.5206);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].mapKind, 4);
  assert.equal(calls[0].a, 0.3);
  assert.equal(calls[0].d, 1.5206);
  assert.deepEqual([...calls[0].state0], [0.5, 0.45, 0.52, 0.48]);
  assert.equal(fig.store.painted, 1);
});

test("a hash state with map and D restores both, in spec order", async () => {
  // parseHash returns name -> value; setParams applies the specs in order,
  // so map=4 resets D to map IV's default and then D=1.515 lands inside
  // the new window. A wiring that applied D before the map switch would
  // clamp it into map I's window first.
  const timers = manualTimers();
  const calls = [];
  const fig = createTorusFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return Promise.resolve(new Float64Array([4, 4, 20000, 0]));
    },
    onTrace: () => {},
    writeHash: () => {},
    timers,
  });
  fig.setParams({ map: 4, D: 1.515 });
  await tick();
  assert.equal(fig.state.map, 4);
  assert.equal(fig.state.D, 1.515);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].d, 1.515);
});

test("a D outside the active map's window clamps into it", () => {
  const fig = createTorusFigure({
    debounceMs: 150,
    echo: () => {},
    call: () => Promise.resolve(new Float64Array([4, 4, 20000, 0])),
    onTrace: () => {},
    writeHash: () => {},
    timers: manualTimers(),
  });
  fig.setParams({ map: 4 });
  // The number field accepts the union range; the figure clamps into the
  // active window rather than asking the kernel for a D it would move.
  fig.onInput("D", 2.0);
  assert.equal(fig.state.D, 1.53);
});


test("LIVE_MODULATED is frozen, registered, and holds the paper's parameters", () => {
  assert.equal(LIVE_FIGURES.double_staircase, LIVE_MODULATED);
  assert.ok(Object.isFrozen(LIVE_MODULATED));
  // The paper's constants (maps/modulated_circle.py compute): A = 0.1,
  // C = (sqrt(5) - 1) / 2, eps = 0.05, theta0 = phi0 = 0.1,
  // n_transient 3000, n_iter 20000.
  assert.equal(LIVE_MODULATED.A, 0.1);
  assert.equal(LIVE_MODULATED.C, 0.6180339887498949);
  assert.equal(LIVE_MODULATED.nTransient, 3000);
  assert.equal(LIVE_MODULATED.nIter, 20000);
  const [eps, dMin, dMax, n] = LIVE_MODULATED.paramSpecs;
  assert.deepEqual(
    { name: eps.name, min: eps.min, max: eps.max, default: eps.default },
    { name: "eps", min: 0, max: 0.2, default: 0.05 },
  );
  assert.deepEqual(
    { name: dMin.name, min: dMin.min, max: dMin.max, default: dMin.default, numberOnly: dMin.numberOnly },
    { name: "dMin", min: 0, max: 1, default: 0, numberOnly: true },
  );
  assert.deepEqual(
    { name: dMax.name, min: dMax.min, max: dMax.max, default: dMax.default, numberOnly: dMax.numberOnly },
    { name: "dMax", min: 0, max: 1, default: 1, numberOnly: true },
  );
  // The point count is bounded by the kernel's 512 cap.
  assert.equal(n.name, "n");
  assert.equal(n.max, 512);
  assert.equal(n.default, 256);
});

test("the modulated presets reproduce the zoom panels' D windows", () => {
  // The windows of modulated_circle.py plot_zoom: longest_plateau_window
  // on the committed npz padded by 0.008 (target 1/4) and 0.006 (target
  // C). A window off by one grid step (1/9999 ~ 1.0001e-4) fails the exact
  // equality below.
  assert.equal(LIVE_MODULATED.presets.length, 2);
  const [a, b] = LIVE_MODULATED.presets;
  assert.equal(a.dMin, 0.25502630263026305);
  assert.equal(a.dMax, 0.27372657265726574);
  assert.equal(b.dMin, 0.5987604760476047);
  assert.equal(b.dMax, 0.6270621062106211);
  // Each preset sits inside [0, 1] and contains its unpadded plateau
  // window — the pad widens the window in both directions.
  assert.ok(a.dMin < 0.26302630263026305 && a.dMax > 0.26572657265726574);
  assert.ok(b.dMin < 0.6047604760476047 && b.dMax > 0.6210621062106211);
});

test("the modulated request passes A, C, the window, n and eps in the kernel's order", () => {
  // modulated_circle_rotation_tile(a, c, d_min, d_max, n_d, eps,
  // n_transient, n_iter, theta0, phi0): swapping A and D here sends the
  // window into the map's A slot, and dropping eps leaves the forcing at
  // the kernel's 0.05 fallback — this test catches both.
  const req = modulatedRequest({ eps: 0.12, dMin: 0.2, dMax: 0.6, n: 128 });
  assert.equal(req.a, 0.1);
  assert.equal(req.c, 0.6180339887498949);
  assert.equal(req.dMin, 0.2);
  assert.equal(req.dMax, 0.6);
  assert.equal(req.nD, 128);
  assert.equal(req.eps, 0.12);
  assert.equal(req.nTransient, 3000);
  assert.equal(req.nIter, 20000);
  assert.equal(req.theta0, 0.1);
  assert.equal(req.phi0, 0.1);
});

test("modulatedKernelCall invokes the export in the kernel's argument order", async () => {
  const seen = [];
  const kernel = (...args) => {
    seen.push(args);
    return new Float64Array([1, 3000, 20000, 2, 0.25, 0.6]);
  };
  const req = modulatedRequest({ eps: 0.12, dMin: 0.2, dMax: 0.6, n: 128 });
  const out = await modulatedKernelCall(req, kernel);
  assert.deepEqual(seen, [[0.1, 0.6180339887498949, 0.2, 0.6, 128, 0.12, 3000, 20000, 0.1, 0.1]]);
  assert.equal(out[4], 0.25);
});

test("modulatedTrace unpacks (rho_theta, rho_phi) pairs after the 4-value header", () => {
  // Header [n_d, n_transient, n_iter, 2], then n_d pairs D-major. Swapping
  // the pair order lands rho_phi on the rho_theta curve, which this test
  // catches.
  const req = modulatedRequest({ eps: 0.05, dMin: 0.2, dMax: 0.6, n: 3 });
  const tile = new Float64Array([3, 3000, 20000, 2, 0.21, 0.61, 0.25, 0.62, 0.29, 0.63]);
  const t = modulatedTrace(tile, req);
  // The x axis is the kernel's own grid: dMin + (dMax - dMin) k / (n - 1).
  assert.deepEqual(t.x, [0.2, 0.4, 0.6]);
  assert.deepEqual(t.y, [0.21, 0.25, 0.29]);
  assert.deepEqual(t.y2, [0.61, 0.62, 0.63]);
});
test("modulatedTrace rejects a short or malformed tile", () => {
  const req = modulatedRequest({ eps: 0.05, dMin: 0, dMax: 1, n: 4 });
  assert.deepEqual(modulatedTrace(null, req), { x: [], y: [], y2: [] });
  assert.deepEqual(modulatedTrace(new Float64Array([4, 3000, 20000, 2, 0.1, 0.6]), req).x, []);
  assert.deepEqual(modulatedTrace(new Float64Array([0, 3000, 20000, 2]), req).x, []);
  // A non-finite pair is skipped, not plotted.
  const t = modulatedTrace(new Float64Array([2, 3000, 20000, 2, NaN, 0.6, 0.3, 0.62]), req);
  assert.deepEqual(t.x, [1]);
  assert.deepEqual(t.y, [0.3]);
  assert.deepEqual(t.y2, [0.62]);
});

test("the modulated figure recomputes once per debounced burst and drops stale results", async () => {
  const timers = manualTimers();
  const calls = [];
  const deferreds = [];
  const fig = createModulatedFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return new Promise((r) => deferreds.push(r));
    },
    onTrace: () => {},
    writeHash: () => {},
    getPlot: () => null,
    timers,
  });
  for (const v of [0.05, 0.08, 0.12]) fig.onInput("eps", v);
  timers.flush();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].eps, 0.12);

  fig.setParams({ eps: 0.06 });
  fig.setParams({ eps: 0.09 });
  assert.equal(calls.length, 3);
  const tileFor = (e) => new Float64Array([1, 3000, 20000, 2, e, 0.6]);
  deferreds[2](tileFor(0.09));
  await tick();
  deferreds[1](tileFor(0.06));
  await tick();
  assert.deepEqual(fig.store.trace.y, [0.09]);
  assert.deepEqual(fig.store.trace.y2, [0.6]);
  assert.equal(fig.store.generation, 3);
});

test("the modulated figure's window endpoints never invert", () => {
  const echoed = [];
  const fig = createModulatedFigure({
    debounceMs: 150,
    echo: (name, v) => echoed.push([name, v]),
    call: () => Promise.resolve(new Float64Array([1, 3000, 20000, 2, 0.25, 0.6])),
    onTrace: () => {},
    writeHash: () => {},
    getPlot: () => null,
    timers: manualTimers(),
  });
  // Pushing dMin past dMax moves dMax up to meet it — the edited value
  // wins, so a two-step edit lands where the reader meant it.
  fig.onInput("dMax", 0.5);
  assert.deepEqual([fig.state.dMin, fig.state.dMax], [0, 0.5]);
  fig.onInput("dMin", 0.6);
  assert.deepEqual([fig.state.dMin, fig.state.dMax], [0.6, 0.6]);
  fig.onInput("dMax", 0.8);
  assert.deepEqual([fig.state.dMin, fig.state.dMax], [0.6, 0.8]);
  fig.onInput("dMax", 0.3);
  assert.deepEqual([fig.state.dMin, fig.state.dMax], [0.3, 0.3]);
  assert.ok(echoed.some(([n, v]) => n === "dMax" && v === 0.6));
  assert.ok(echoed.some(([n, v]) => n === "dMin" && v === 0.3));
});

test("a preset applies through setParams and reaches the kernel request", async () => {
  const timers = manualTimers();
  const calls = [];
  const fig = createModulatedFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return Promise.resolve(new Float64Array([2, 3000, 20000, 2, 0.25, 0.6, 0.26, 0.6]));
    },
    onTrace: () => {},
    writeHash: () => {},
    getPlot: () => null,
    timers,
  });
  const p = LIVE_MODULATED.presets[0];
  fig.setParams({ dMin: p.dMin, dMax: p.dMax });
  await tick();
  assert.equal(fig.state.dMin, p.dMin);
  assert.equal(fig.state.dMax, p.dMax);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].dMin, p.dMin);
  assert.equal(calls[0].dMax, p.dMax);
});

test("the range view pushes a drag into the inputs and syncs the domain after a recompute", () => {
  const ranges = [];
  const domains = [];
  let dom = { x0: 0, x1: 1, y0: 0, y1: 1 };
  let range = { dMin: 0, dMax: 1 };
  const view = createRangeView({
    getRange: () => range,
    onRange: (dMin, dMax) => {
      ranges.push([dMin, dMax]);
      range = { dMin, dMax };
    },
    getPlot: () => ({
      getDomain: () => dom,
      setDomain: (d) => {
        dom = { ...dom, ...d };
        domains.push(dom);
      },
    }),
  });
  // A drag zoom pushes the new window into the inputs. Plot sets the
  // domain before onView fires, so dom already carries the dragged range.
  dom = { x0: 0.26, x1: 0.27, y0: 0, y1: 1 };
  view.onView({ x0: 0.26, x1: 0.27 });
  assert.deepEqual(ranges, [[0.26, 0.27]]);
  // The same domain again (the setDomain echo) does not re-push.
  view.onView({ x0: 0.26, x1: 0.27 });
  assert.equal(ranges.length, 1);
  // After the recompute the domain already matches — syncView is a no-op.
  view.syncView();
  assert.equal(domains.length, 0);
  // A range change from the inputs moves the domain onto it.
  range = { dMin: 0.6, dMax: 0.62 };
  view.syncView();
  assert.deepEqual(domains, [{ x0: 0.6, x1: 0.62, y0: 0, y1: 1 }]);
});

test("the modulated readout falls back to the nearest plotted rho_theta", () => {
  const store = { trace: { x: [0.2, 0.4, 0.6], y: [0.21, 0.25, 0.29], y2: [0.6, 0.6, 0.6] } };
  const readout = createModulatedReadout({ store, getEps: () => 0.05 });
  // No kernel is loaded in the test process, so the trace answers.
  assert.equal(readout(0.41), 0.25);
  assert.equal(readout(0.55), 0.29);
});

// ---- CML space-time figure ----

// A small deterministic field: values[r * nSites + i] = r + i / nSites,
// so a transposed or shifted read lands on a different number.
function cmlTile(model, nSites, nRows, fill) {
  const tile = new Float64Array(HEADER + nSites * nRows);
  tile[0] = model;
  tile[1] = nSites;
  tile[2] = 0;
  tile[3] = nRows;
  for (let k = 0; k < nSites * nRows; k++) tile[HEADER + k] = fill(k);
  return tile;
}
const cmlFill = (nSites) => (k) => Math.floor(k / nSites) + (k % nSites) / nSites;

test("LIVE_SPACETIME is frozen, registered, and holds the paper's parameters", () => {
  assert.equal(LIVE_FIGURES.spacetime_diagrams, LIVE_SPACETIME);
  assert.ok(Object.isFrozen(LIVE_SPACETIME));
  // The paper's counts (cml/spatiotemporal.py simulate_cml): n_transient
  // 2000, n_record 500, N = 200 default inside the kernel's [2, 512] cap.
  assert.equal(LIVE_SPACETIME.nTransient, 2000);
  assert.equal(LIVE_SPACETIME.nRecord, 500);
  const [model, eps, sites] = LIVE_SPACETIME.paramSpecs;
  assert.equal(model.name, "model");
  assert.equal(model.options.length, 3);
  assert.deepEqual(
    { min: eps.min, max: eps.max, step: eps.step, default: eps.default },
    { min: 0, max: 0.5, step: 0.001, default: 0.07 },
  );
  assert.deepEqual(
    { min: sites.min, max: sites.max, default: sites.default },
    { min: 16, max: 512, default: 200 },
  );
  // Each model's eps window is its published sweep's range and middle
  // value (spatiotemporal.py compute).
  assert.deepEqual(
    [LIVE_SPACETIME.models[0].epsMin, LIVE_SPACETIME.models[0].epsMax, LIVE_SPACETIME.models[0].epsDefault],
    [0, 0.2, 0.07],
  );
  assert.deepEqual(
    [LIVE_SPACETIME.models[1].epsMin, LIVE_SPACETIME.models[1].epsMax, LIVE_SPACETIME.models[1].epsDefault],
    [0, 0.1, 0.024],
  );
  assert.deepEqual(
    [LIVE_SPACETIME.models[2].epsMin, LIVE_SPACETIME.models[2].epsMax, LIVE_SPACETIME.models[2].epsDefault],
    [0, 0.5, 0.2],
  );
});

test("cmlModelKind snaps the selector the way the kernel does", () => {
  assert.equal(cmlModelKind(0), 0);
  assert.equal(cmlModelKind(1), 1);
  assert.equal(cmlModelKind(2), 2);
  assert.equal(cmlModelKind(9), 2);
  assert.equal(cmlModelKind(-3), 0);
  assert.equal(cmlModelKind(NaN), 0);
});

test("the spacetime request passes model, eps, sites and counts in the kernel's order", () => {
  // cml_spacetime_tile(model, eps, n_sites, n_transient, n_record, x0):
  // dropping eps leaves the coupling at the kernel's fallback, and a model
  // read from the wrong field runs the wrong lattice — this test catches
  // both.
  const x0 = Array.from({ length: 512 }, (_, i) => i / 512);
  const req = cmlRequest({ model: 2, eps: 0.3, sites: 64 }, LIVE_SPACETIME, x0);
  assert.equal(req.model, 2);
  assert.equal(req.eps, 0.3);
  assert.equal(req.nSites, 64);
  assert.equal(req.nTransient, 2000);
  assert.equal(req.nRecord, 500);
  assert.equal(req.x0.length, 64);
  assert.equal(req.x0[63], 63 / 512);
});

test("the spacetime request slices x0 to the site count", () => {
  // uniform(0, 1, N) draws the first N values of the seed-42 stream, so
  // the paper's x0 at N = 200 is a prefix of the file's 512 — slicing, not
  // reseeding. A request that passed the whole array would hand the
  // kernel 512 values for a 200-site lattice.
  const x0 = Array.from({ length: 512 }, (_, i) => 0.5 + i / 1000);
  const req = cmlRequest({ model: 0, eps: 0.07, sites: 200 }, LIVE_SPACETIME, x0);
  assert.equal(req.x0.length, 200);
  assert.equal(req.x0[199], 0.5 + 199 / 1000);
});

test("the spacetime request clamps eps into the active model's window", () => {
  const req = cmlRequest({ model: 1, eps: 0.4, sites: 32 }, LIVE_SPACETIME, [0.5]);
  assert.equal(req.eps, 0.1);
});

test("cmlKernelCall invokes the export in the kernel's argument order", async () => {
  const seen = [];
  const kernel = (...args) => {
    seen.push(args);
    return new Float64Array([2, 4, 0, 1, 0.1, 0.2, 0.3, 0.4]);
  };
  const req = cmlRequest({ model: 2, eps: 0.2, sites: 4 }, LIVE_SPACETIME, [0.1, 0.2, 0.3, 0.4]);
  const out = await cmlKernelCall(req, kernel);
  assert.equal(seen[0][0], 2);
  assert.equal(seen[0][1], 0.2);
  assert.equal(seen[0][2], 4);
  assert.equal(seen[0][3], 2000);
  assert.equal(seen[0][4], 500);
  assert.deepEqual(Array.from(seen[0][5]), [0.1, 0.2, 0.3, 0.4]);
  assert.equal(out[4], 0.1);
});

test("cmlField unpacks rows x sites after the 4-value header", () => {
  // values[t * n_sites + i] is site i of the t-th recorded field — a
  // transposed read lands site and row on each other's axis, which the
  // distinct fill catches.
  const req = cmlRequest({ model: 1, eps: 0.024, sites: 3 }, LIVE_SPACETIME, []);
  const tile = cmlTile(1, 3, 4, cmlFill(3));
  const f = cmlField(tile, req);
  assert.equal(f.nSites, 3);
  assert.equal(f.nRows, 4);
  assert.equal(f.model, 1);
  assert.equal(f.eps, 0.024);
  assert.equal(f.values[2 * 3 + 1], 2 + 1 / 3);
  assert.equal(f.values[0], 0);
});

test("cmlField rejects a short or malformed tile", () => {
  const req = cmlRequest({ model: 0, eps: 0.07, sites: 4 }, LIVE_SPACETIME, []);
  assert.equal(cmlField(null, req), null);
  assert.equal(cmlField(new Float64Array([0, 4, 0, 2, 0.1]), req), null);
  assert.equal(cmlField(new Float64Array([0, 4, 0, 0]), req), null);
  assert.equal(cmlField(cmlTile(0, 4, 3, () => 0.5).slice(0, HEADER + 5), req), null);
});

test("the chunk request continues from the window's last row", () => {
  // Play appends n_transient 0 rows starting at the last recorded field —
  // continuing from the first row instead would restart the window.
  const field = {
    values: Float64Array.from({ length: 12 }, (_, k) => 100 + k),
    nSites: 4,
    nRows: 3,
    model: 2,
    eps: 0.2,
  };
  const req = cmlChunkRequest({ model: 0, eps: 0.07, sites: 4 }, field);
  assert.equal(req.model, 2);
  assert.equal(req.eps, 0.2);
  assert.equal(req.nTransient, 0);
  assert.equal(req.nRecord, 250);
  assert.deepEqual(Array.from(req.x0), [108, 109, 110, 111]);
});

test("cmlAppend drops the oldest rows and keeps the window at nRecord", () => {
  const field = {
    values: Float64Array.from({ length: 20 }, (_, k) => k),
    nSites: 4,
    nRows: 5,
    model: 0,
    eps: 0.07,
  };
  const chunk = cmlTile(0, 4, 2, (k) => 100 + k);
  const next = cmlAppend(field, chunk, 5);
  assert.equal(next.nRows, 5);
  assert.equal(next.nSites, 4);
  // Rows 2-4 of the old window survive (keep = min(5, 5-2) = 3); rows
  // 0-1 are dropped.
  assert.deepEqual(Array.from(next.values.slice(0, 12)), [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
  assert.deepEqual(Array.from(next.values.slice(12)), [100, 101, 102, 103, 104, 105, 106, 107]);
});

test("cmlAppend rejects a chunk from a different run", () => {
  const field = { values: new Float64Array(8), nSites: 4, nRows: 2, model: 0, eps: 0.07 };
  assert.equal(cmlAppend(field, cmlTile(1, 4, 2, () => 0), 5), null);
  assert.equal(cmlAppend(field, cmlTile(0, 8, 2, () => 0), 5), null);
  assert.equal(cmlAppend(field, null, 5), null);
});

test("cmlColorLimits are the shown field's 1st and 99th percentiles", () => {
  // The same rule plot() applies per row: linear interpolation on the
  // sorted values, with the +-0.5 pad when the field is constant.
  const values = Float64Array.from({ length: 100 }, (_, i) => i);
  const [lo, hi] = cmlColorLimits(values);
  assert.equal(lo, 0.99);
  assert.equal(hi, 98.01);
  const [clo, chi] = cmlColorLimits(new Float64Array([0.4, 0.4, 0.4]));
  assert.ok(Math.abs(clo - -0.1) < 1e-12);
  assert.ok(Math.abs(chi - 0.9) < 1e-12);
  assert.deepEqual(cmlColorLimits([]), [0, 1]);
});

test("the spacetime figure snaps eps to its decimal literal", () => {
  // A range input's stepped value is a double like 0.07000000000000001;
  // the kernel must see 0.07 exactly — a one-ulp eps change moves a
  // chaotic orbit.
  const echoed = [];
  const fig = createSpacetimeFigure({
    x0: [0.5],
    debounceMs: 150,
    echo: (name, v) => echoed.push([name, v]),
    call: () => Promise.resolve(cmlTile(0, 16, 2, () => 0.5)),
    onField: () => {},
    writeHash: () => {},
    timers: manualTimers(),
  });
  fig.onInput("eps", 0.07000000000000001);
  assert.equal(fig.state.eps, 0.07);
  fig.onInput("eps", 0.1234567);
  assert.equal(fig.state.eps, 0.123);
  fig.onInput("sites", 200.7);
  assert.equal(fig.state.sites, 201);
  assert.ok(echoed.some(([n, v]) => n === "eps" && v === 0.07));
});

test("switching the spacetime model resets eps to the new model's default", async () => {
  const timers = manualTimers();
  const calls = [];
  const fig = createSpacetimeFigure({
    x0: [0.5],
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return Promise.resolve(cmlTile(req.model, req.nSites, 2, () => 0.5));
    },
    onField: () => {},
    writeHash: () => {},
    timers,
  });
  fig.onInput("model", 2);
  timers.flush();
  await tick();
  assert.equal(fig.state.model, 2);
  assert.equal(fig.state.eps, 0.2);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].model, 2);
  assert.equal(calls[0].eps, 0.2);
});

test("a hash state with model, eps and sites restores all three", async () => {
  const fig = createSpacetimeFigure({
    x0: [0.5],
    debounceMs: 150,
    echo: () => {},
    call: (req) => Promise.resolve(cmlTile(req.model, req.nSites, 2, () => 0.5)),
    onField: () => {},
    writeHash: () => {},
    timers: manualTimers(),
  });
  fig.setParams({ model: 1, eps: 0.03, sites: 64 });
  await tick();
  assert.equal(fig.state.model, 1);
  assert.equal(fig.state.eps, 0.03);
  assert.equal(fig.state.sites, 64);
  assert.equal(fig.store.field.nSites, 64);
});

test("play appends a chunk from the last row and pause stops the transport", async () => {
  const timers = manualTimers();
  const calls = [];
  const fig = createSpacetimeFigure({
    x0: [0.5],
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      // A fresh run answers with a small field; a play chunk answers with
      // its own row count, so the window arithmetic is exercised.
      const rows = req.nTransient === 0 ? req.nRecord : 2;
      return Promise.resolve(
        cmlTile(req.model, req.nSites, rows, (k) => 1000 + k),
      );
    },
    onField: () => {},
    writeHash: () => {},
    timers,
  });
  fig.setParams({ sites: 16 });
  await tick();
  const first = fig.store.field;
  assert.equal(first.nRows, 2);
  fig.play();
  assert.equal(fig.playing(), true);
  timers.flush();
  await tick();
  // One chunk of 250 rows appended to the 2-row fake field; the window
  // cap (500) does not bind yet.
  assert.equal(calls.length, 2);
  assert.equal(calls[1].nTransient, 0);
  assert.equal(calls[1].nRecord, 250);
  assert.deepEqual(Array.from(calls[1].x0), Array.from(first.values.slice(16, 32)));
  assert.equal(fig.store.field.nRows, 252);
  fig.pause();
  assert.equal(fig.playing(), false);
  const gen = fig.store.generation;
  timers.flush();
  await tick();
  assert.equal(fig.store.generation, gen);
  assert.equal(calls.length, 2);
});

test("a chunk that resolves after a parameter change is dropped", async () => {
  const timers = manualTimers();
  const deferreds = [];
  const fig = createSpacetimeFigure({
    x0: [0.5],
    debounceMs: 150,
    echo: () => {},
    call: (req) => new Promise((r) => deferreds.push([req, r])),
    onField: () => {},
    writeHash: () => {},
    timers,
  });
  fig.setParams({ sites: 16 });
  deferreds[0][1](cmlTile(0, 16, 2, () => 0.5));
  await tick();
  const first = fig.store.field;
  fig.play();
  timers.flush();
  assert.equal(deferreds.length, 2);
  // A model switch starts a fresh run before the chunk resolves.
  fig.setParams({ model: 2 });
  deferreds[2][1](cmlTile(2, 16, 2, () => 0.7));
  await tick();
  assert.equal(fig.store.field.model, 2);
  // The stale chunk resolves now: it must not merge into the new field.
  deferreds[1][1](cmlTile(0, 16, 250, () => 0.9));
  await tick();
  assert.equal(fig.store.field.nRows, 2);
  assert.equal(fig.store.field.values[0], 0.7);
});

test("shortUserAgent extracts browser and major version without spaces", () => {
  const chrome = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/142.0.7444.175 Safari/537.36";
  assert.equal(shortUserAgent(chrome), "Chrome/142");

  const desktopSafari = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15";
  assert.equal(shortUserAgent(desktopSafari), "Safari/17.5");

  const mobileSafari = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1";
  assert.equal(shortUserAgent(mobileSafari), "MobileSafari/17.5");

  const firefox = "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0";
  assert.equal(shortUserAgent(firefox), "Firefox/130");

  const edge = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0";
  assert.equal(shortUserAgent(edge), "Edge/126");

  assert.equal(shortUserAgent(""), "unknown");
  assert.equal(shortUserAgent("   "), "unknown");
  assert.equal(shortUserAgent(null), "unknown");
  assert.ok(!/\s/.test(shortUserAgent(chrome)));
});

test("median computes standard midpoint of odd, even, and empty arrays", () => {
  assert.equal(median([10, 30, 20]), 20);
  assert.equal(median([10, 20, 30, 40]), 25);
  assert.equal(median([5]), 5);
  assert.equal(median([]), 0);
  assert.equal(median(null), 0);
  assert.equal(median([89.7, 91.5, 93.4, 92.0, 93.8]), 92.0);
});

test("isLongTaskSupported checks PerformanceObserver.supportedEntryTypes", () => {
  assert.ok(isLongTaskSupported({ supportedEntryTypes: ["longtask", "paint"] }));
  assert.ok(!isLongTaskSupported({ supportedEntryTypes: ["paint", "resource"] }));
  assert.ok(!isLongTaskSupported({ supportedEntryTypes: null }));
  assert.ok(!isLongTaskSupported(undefined));
});

test("createLongTaskObserver returns n/a where unsupported and never 0", () => {
  const unsupported = { supportedEntryTypes: ["paint", "resource"] };
  const obs = createLongTaskObserver(unsupported);
  assert.equal(obs.supported, false);
  assert.equal(obs.count(), "n/a");
  assert.notEqual(obs.count(), 0);
});

test("createLongTaskObserver counts entries over 50 ms when supported", () => {
  let registeredCallback = null;
  class FakeObs {
    static supportedEntryTypes = ["longtask"];
    constructor(cb) {
      registeredCallback = cb;
      this.observed = null;
    }
    observe(opts) {
      this.observed = opts;
    }
    disconnect() {}
    takeRecords() {
      return [{ duration: 75 }];
    }
  }
  const obs = createLongTaskObserver(FakeObs);
  assert.equal(obs.supported, true);
  registeredCallback({
    getEntries: () => [
      { duration: 40 }, // under 50 ms, ignored
      { duration: 60 }, // over 50 ms, counted
      { duration: 50 }, // not over 50 ms, ignored
      { duration: 120 }, // over 50 ms, counted
    ],
  });
  // 2 entries from getEntries + 1 entry from takeRecords (75ms)
  assert.equal(obs.count(), 3);
});

test("createLongTaskObserver counts only entries after interactT0 and uses buffered type", () => {
  let registeredCallback = null;
  let observeOpts = null;
  class FakeObs {
    static supportedEntryTypes = ["longtask"];
    constructor(cb) {
      registeredCallback = cb;
    }
    observe(opts) {
      observeOpts = opts;
    }
    disconnect() {}
  }
  const interactT0 = 1000;
  const obs = createLongTaskObserver(FakeObs, interactT0);
  assert.equal(obs.supported, true);
  assert.deepEqual(observeOpts, { type: "longtask", buffered: true });
  registeredCallback({
    getEntries: () => [
      { startTime: 500, duration: 80 },
      { startTime: 1200, duration: 80 },
    ],
  });
  assert.equal(obs.count(), 1);

  const unsupported = { supportedEntryTypes: ["paint"] };
  const unsuppObs = createLongTaskObserver(unsupported, interactT0);
  assert.equal(unsuppObs.supported, false);
  assert.equal(unsuppObs.count(), "n/a");
});

test("formatDebugReportLine produces one line with six fields in order", () => {
  const line = formatDebugReportLine({
    figId: "fig:arnold_tongues",
    ua: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/142.0.7444.175 Safari/537.36",
    workerCount: 8,
    elapsedMs: 6160.4,
    medianTileMs: 89.702,
    longTasks: 0,
  });
  const fields = line.split(" ");
  assert.equal(fields.length, 6);
  assert.equal(fields[0], "fig:arnold_tongues");
  assert.equal(fields[1], "Chrome/142");
  assert.equal(fields[2], "8");
  assert.equal(fields[3], "6160");
  assert.equal(fields[4], "89.7");
  assert.equal(fields[5], "0");

  const safariLine = formatDebugReportLine({
    figId: "fig:devils_staircase",
    ua: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    workerCount: 4,
    elapsedMs: 1200,
    medianTileMs: 45,
    longTasks: "n/a",
  });
  const sFields = safariLine.split(" ");
  assert.equal(sFields.length, 6);
  assert.equal(sFields[5], "n/a");
});

test("attachDebugReportBox creates DOM elements and copy button writes to clipboard", async () => {
  const elements = [];
  class MockNode {
    constructor(tag) {
      this.tagName = tag;
      this.className = "";
      this.textContent = "";
      this.type = "";
      this.children = [];
      this.listeners = {};
      elements.push(this);
    }
    appendChild(child) {
      this.children.push(child);
      return child;
    }
    remove() {
      this.removed = true;
    }
    addEventListener(type, fn) {
      this.listeners[type] = fn;
    }
    querySelector(sel) {
      if (sel === ".fig-body") return this;
      return this.children.find((c) => c.className === sel.replace(".", "")) || null;
    }
  }
  let copiedText = null;
  const mockDoc = {
    createElement: (tag) => new MockNode(tag),
    defaultView: {
      navigator: {
        clipboard: {
          writeText: async (t) => {
            copiedText = t;
          },
        },
      },
    },
  };
  const fig = new MockNode("figure");
  const box = attachDebugReportBox(fig, "fig:test Chrome/142 8 100 20 0", mockDoc);
  assert.ok(box);
  assert.equal(box.className, "live-debug-report");
  const lineEl = box.querySelector(".live-debug-line");
  const copyBtn = box.querySelector(".live-debug-copy");
  assert.ok(lineEl);
  assert.ok(copyBtn);
  assert.equal(lineEl.textContent, "fig:test Chrome/142 8 100 20 0");
  assert.equal(copyBtn.textContent, "copy");

  // Click copy button: calls clipboard.writeText
  await copyBtn.listeners.click();
  assert.equal(copiedText, "fig:test Chrome/142 8 100 20 0");
});

test("attachDebugReportBox copy button selects text when clipboard is refused or absent", async () => {
  class MockNode {
    constructor(tag) {
      this.tagName = tag;
      this.className = "";
      this.textContent = "";
      this.type = "";
      this.children = [];
      this.listeners = {};
    }
    appendChild(child) {
      this.children.push(child);
      return child;
    }
    remove() {}
    addEventListener(type, fn) {
      this.listeners[type] = fn;
    }
    querySelector(sel) {
      if (sel === ".fig-body") return this;
      return this.children.find((c) => c.className === sel.replace(".", "")) || null;
    }
  }
  let selectedNode = null;
  const mockDocRefused = {
    createElement: (tag) => new MockNode(tag),
    createRange: () => ({
      selectNodeContents: (n) => {
        selectedNode = n;
      },
    }),
    defaultView: {
      getSelection: () => ({
        removeAllRanges: () => {},
        addRange: () => {},
      }),
      navigator: {
        clipboard: {
          writeText: async () => {
            throw new Error("clipboard permission denied");
          },
        },
      },
    },
  };
  const fig = new MockNode("figure");
  const box = attachDebugReportBox(fig, "fig:test Chrome/142 8 100 20 0", mockDocRefused);
  const copyBtn = box.querySelector(".live-debug-copy");
  await copyBtn.listeners.click();
  assert.ok(selectedNode);
  assert.equal(selectedNode.textContent, "fig:test Chrome/142 8 100 20 0");

  // Also test absent clipboard
  selectedNode = null;
  const mockDocAbsent = {
    createElement: (tag) => new MockNode(tag),
    createRange: () => ({
      selectNodeContents: (n) => {
        selectedNode = n;
      },
    }),
    defaultView: {
      getSelection: () => ({
        removeAllRanges: () => {},
        addRange: () => {},
      }),
      navigator: {},
    },
  };
  const box2 = attachDebugReportBox(fig, "fig:test2 Chrome/142 8 100 20 0", mockDocAbsent);
  const copyBtn2 = box2.querySelector(".live-debug-copy");
  await copyBtn2.listeners.click();
  assert.ok(selectedNode);
  assert.equal(selectedNode.textContent, "fig:test2 Chrome/142 8 100 20 0");
});

test("createDebugTracker: debugEnabled false creates no observer and no DOM box", () => {
  let obsCreated = false;
  class MockObs {
    static supportedEntryTypes = ["longtask"];
    constructor() {
      obsCreated = true;
    }
    observe() {}
    disconnect() {}
  }
  const tracker = createDebugTracker({
    fig: null,
    interactT0: 100,
    debugEnabled: false,
    PerformanceObserverImpl: MockObs,
  });
  tracker.onFirstPaintComplete({ workerCount: 8, tileTimes: [10, 20] });
  assert.equal(obsCreated, false);
});

test("createDebugTracker passes interactT0 to long-task observer", () => {
  class MockNode {
    constructor(tag) {
      this.tagName = tag;
      this.className = "";
      this.textContent = "";
      this.type = "";
      this.children = [];
      this.listeners = {};
    }
    appendChild(child) {
      this.children.push(child);
      return child;
    }
    remove() {
      this.removed = true;
    }
    addEventListener(type, fn) {
      this.listeners[type] = fn;
    }
    querySelector(sel) {
      if (sel === ".fig-body") return this;
      return this.children.find((c) => c.className === sel.replace(".", "")) || null;
    }
  }
  const mockDoc = {
    createElement: (tag) => new MockNode(tag),
    defaultView: {
      navigator: {
        userAgent:
          "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/142.0.7444.175 Safari/537.36",
      },
    },
  };
  class FakeLongTaskObs {
    static supportedEntryTypes = ["longtask"];
    constructor(cb) {
      this.cb = cb;
    }
    observe() {
      if (typeof this.cb === "function") {
        this.cb({
          getEntries: () => [
            { startTime: 500, duration: 80 },
            { startTime: 1500, duration: 80 },
          ],
        });
      }
    }
    disconnect() {}
  }
  const fig = new MockNode("figure");
  fig.id = "fig:test";
  const interactT0 = 1000;
  const tracker = createDebugTracker({
    fig,
    interactT0,
    debugEnabled: true,
    PerformanceObserverImpl: FakeLongTaskObs,
    doc: mockDoc,
  });
  tracker.onFirstPaintComplete({ workerCount: 4, tileTimes: [20, 30] });

  assert.ok(fig._debugReport);
  const fields = fig._debugReport.split(" ");
  assert.equal(fields.length, 6);
  assert.equal(fields[5], "1");
});

test("createPaintHandler fires onFirstPaintComplete on the last tile of generation 1", () => {
  const store = createStore();
  let inFlight = [{ id: "0:0:0", generation: 1 }];
  let queueDepth = 0;
  const pool = {
    getState: () => ({
      generation: 1,
      viewport: VIEWPORT,
      queueDepth,
      inFlight,
    }),
  };
  const raster = { tileWorld };
  let completedInfo = null;
  const onPaint = createPaintHandler({
    store,
    raster,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => {} }),
    onFirstPaintComplete: (info) => {
      completedInfo = info;
    },
  });

  // Tile 1 arrives, but inFlight still has work
  inFlight = [{ id: "1:0:0", generation: 1 }];
  queueDepth = 1;
  onPaint({ id: "0:0:0", generation: 1, header: [2, 2, 200, 2000], data: [], computeMs: 42 });
  assert.equal(completedInfo, null);

  // Last tile arrives: queueDepth is 0 and inFlight is empty
  queueDepth = 0;
  inFlight = [];
  onPaint({ id: "1:0:0", generation: 1, header: [2, 2, 200, 2000], data: [], computeMs: 58 });
  assert.ok(completedInfo);
  assert.deepEqual(completedInfo.tileTimes, [42, 58]);
  assert.equal(completedInfo.generation, 1);
});

test("createPaintHandler does not fire onFirstPaintComplete when tiles queued but none in flight", () => {
  const store = createStore();
  let queueDepth = 2;
  let inFlight = [];
  const pool = {
    getState: () => ({
      generation: 1,
      viewport: VIEWPORT,
      queueDepth,
      inFlight,
    }),
  };
  const raster = { tileWorld };
  let completedInfo = null;
  const onPaint = createPaintHandler({
    store,
    raster,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => {} }),
    onFirstPaintComplete: (info) => {
      completedInfo = info;
    },
  });

  // Tile arrives, queueDepth is 2 but inFlight is empty: onFirstPaintComplete is NOT called
  onPaint({ id: "0:0:0", generation: 1, header: [2, 2, 200, 2000], data: [], computeMs: 30 });
  assert.equal(completedInfo, null);

  // Once both are empty: it IS called
  queueDepth = 0;
  inFlight = [];
  onPaint({ id: "1:0:0", generation: 1, header: [2, 2, 200, 2000], data: [], computeMs: 40 });
  assert.ok(completedInfo);
  assert.deepEqual(completedInfo.tileTimes, [30, 40]);
  assert.equal(completedInfo.generation, 1);
});

test("createPaintHandler does not fire onFirstPaintComplete when tiles in flight but none queued", () => {
  const store = createStore();
  let queueDepth = 0;
  let inFlight = [{ id: "1:0:0", generation: 1 }];
  const pool = {
    getState: () => ({
      generation: 1,
      viewport: VIEWPORT,
      queueDepth,
      inFlight,
    }),
  };
  const raster = { tileWorld };
  let completedInfo = null;
  const onPaint = createPaintHandler({
    store,
    raster,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => {} }),
    onFirstPaintComplete: (info) => {
      completedInfo = info;
    },
  });

  // Tile arrives, queueDepth is 0 but inFlight has tiles: onFirstPaintComplete is NOT called
  onPaint({ id: "0:0:0", generation: 1, header: [2, 2, 200, 2000], data: [], computeMs: 25 });
  assert.equal(completedInfo, null);

  // Once both are empty: it IS called
  queueDepth = 0;
  inFlight = [];
  onPaint({ id: "1:0:0", generation: 1, header: [2, 2, 200, 2000], data: [], computeMs: 35 });
  assert.ok(completedInfo);
  assert.deepEqual(completedInfo.tileTimes, [25, 35]);
  assert.equal(completedInfo.generation, 1);
});

test("buildHint returns a short hint under 80 characters and unchanged full text", () => {
  const hint = buildHint();
  assert.ok(hint.shortText.length <= 80, `short hint length ${hint.shortText.length} must be <= 80`);
  assert.equal(hint.shortText, LIVE_HINT_SHORT);
  assert.equal(hint.fullText, LIVE_HINT_FULL);
  assert.ok(hint.fullText.includes("tap to read values"));
  assert.ok(hint.fullText.includes("drag to zoom"));
  assert.ok(hint.fullText.includes("computed live in this browser"));
});

test("buildHint with play returns a play-specific short hint under 80 characters and full text", () => {
  const hint = buildHint({ play: true });
  assert.ok(hint.shortText.length <= 80, `play short hint length ${hint.shortText.length} must be <= 80`);
  assert.equal(hint.shortText, LIVE_HINT_SHORT_PLAY);
  assert.equal(hint.fullText, LIVE_HINT_FULL_PLAY);
  assert.ok(hint.fullText.includes("play appends new rows and drops the oldest"));
});

test("worker stale-result rule: a result older than the newest request is dropped and never painted when replies arrive in reverse order", async () => {
  const timers = manualTimers();
  const deferreds = [];
  const paints = [];
  const fig = createModulatedFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => new Promise((resolve) => deferreds.push({ req, resolve })),
    onTrace: (store, trace, req) => paints.push({ eps: req.eps, y: [...trace.y] }),
    writeHash: () => {},
    getPlot: () => null,
    timers,
  });

  fig.onInput("eps", 0.05);
  timers.flush();
  assert.equal(deferreds.length, 1);
  assert.equal(deferreds[0].req.eps, 0.05);

  fig.onInput("eps", 0.12);
  timers.flush();
  assert.equal(deferreds.length, 2);
  assert.equal(deferreds[1].req.eps, 0.12);

  // Newer reply arrives first: painted
  const tileFor = (eps) => new Float64Array([1, 3000, 20000, 2, eps, 0.6]);
  deferreds[1].resolve(tileFor(0.12));
  await tick();

  assert.equal(paints.length, 1);
  assert.equal(paints[0].eps, 0.12);
  assert.deepEqual(fig.store.trace.y, [0.12]);
  assert.equal(fig.store.generation, 2);

  // Older reply arrives in reverse order: dropped, never painted
  deferreds[0].resolve(tileFor(0.05));
  await tick();

  assert.equal(paints.length, 1);
  assert.deepEqual(fig.store.trace.y, [0.12]);
  assert.equal(fig.store.generation, 2);
});

test("worker computeTile returns same Float64Array element by element as direct modulated_circle_rotation_tile call", async () => {
  const { computeTile } = await import("../../site/live/tile-worker.js");
  const glue = await import("../../site/wasm/dynachaos_wasm.js");
  const { readFile } = await import("node:fs/promises");
  const bytes = await readFile(new URL("../../site/wasm/dynachaos_wasm_bg.wasm", import.meta.url));
  glue.initSync({ module: bytes });

  const req = {
    kernel: "modulated_circle_rotation_tile",
    a: 0.1,
    c: 0.6180339887498949,
    dMin: 0.2,
    dMax: 0.6,
    nD: 16,
    eps: 0.05,
    nTransient: 300,
    nIter: 2000,
    theta0: 0.1,
    phi0: 0.1,
  };

  const direct = glue.modulated_circle_rotation_tile(
    req.a, req.c, req.dMin, req.dMax, req.nD, req.eps,
    req.nTransient, req.nIter, req.theta0, req.phi0,
  );
  const workerReply = await computeTile(req);
  assert.equal(workerReply.type, "result");
  const workerData = workerReply.data;

  assert.ok(direct instanceof Float64Array);
  assert.ok(workerData instanceof Float64Array);
  assert.equal(direct.length, workerData.length);
  for (let i = 0; i < direct.length; i++) {
    assert.ok(
      Object.is(direct[i], workerData[i]),
      `element ${i} mismatch: direct ${direct[i]} vs worker ${workerData[i]}`,
    );
  }
});

test("worker computeTile returns same Float64Array element by element as direct cml_spacetime_tile call", async () => {
  const { computeTile } = await import("../../site/live/tile-worker.js");
  const glue = await import("../../site/wasm/dynachaos_wasm.js");
  const { readFile } = await import("node:fs/promises");
  const bytes = await readFile(new URL("../../site/wasm/dynachaos_wasm_bg.wasm", import.meta.url));
  glue.initSync({ module: bytes });

  const x0 = new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
  const req = {
    kernel: "cml_spacetime_tile",
    model: 0,
    eps: 0.1,
    nSites: 8,
    nTransient: 200,
    nRecord: 50,
    x0,
  };

  const direct = glue.cml_spacetime_tile(
    req.model, req.eps, req.nSites, req.nTransient, req.nRecord, req.x0,
  );
  const workerReply = await computeTile(req);
  assert.equal(workerReply.type, "result");
  const workerData = workerReply.data;

  assert.ok(direct instanceof Float64Array);
  assert.ok(workerData instanceof Float64Array);
  assert.equal(direct.length, workerData.length);
  for (let i = 0; i < direct.length; i++) {
    assert.ok(
      Object.is(direct[i], workerData[i]),
      `element ${i} mismatch: direct ${direct[i]} vs worker ${workerData[i]}`,
    );
  }
});

test("createKernelClient handles request, resolves result and terminates cleanly", async () => {
  const FakeWorker = makeFakeWorkerClass();
  const client = createKernelClient({ Worker: FakeWorker });
  assert.equal(FakeWorker.instances.length, 1);
  const worker = FakeWorker.instances[0];

  const callPromise = client.call("modulated_circle_rotation_tile", { a: 0.1 });
  assert.equal(worker.posted.length, 1);
  const msg = worker.posted[0];
  assert.equal(msg.type, "tile");
  assert.equal(msg.kernel, "modulated_circle_rotation_tile");
  assert.equal(msg.a, 0.1);

  const mockData = new Float64Array([1, 2, 3, 4]);
  worker.onmessage({ data: { type: "result", id: msg.id, data: mockData } });
  const result = await callPromise;
  assert.equal(result, mockData);

  client.destroy();
  assert.equal(worker.terminated, true);
});