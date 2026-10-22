/**
 * Pure tile scheduler for live parameter-plane figures.
 *
 * State in, commands out. No DOM, no Worker, no timer, no fetch — a node
 * test file can import this module and drive it with events. The worker pool
 * in pool.js is the only thing that talks to the browser.
 *
 * A viewport is a rectangle in (Omega, K). Tiles cover it at a pyramid of
 * levels: level 0 is one tile over the whole view, and each finer level
 * splits every parent 2x2. Every coarser tile is issued before any finer
 * one. A generation counter rides on each request; after the viewport
 * changes, results carrying an older generation are dropped, not painted.
 * A tile lost to a worker failure is re-issued once per generation, then
 * dropped for good; a tile-level error reply is never retried.
 */

const DEFAULTS = {
  levels: 3,
  tileCells: 64,
  nTransient: 200,
  nIter: 500,
  theta0: 0.1,
};

/**
 * @param {object} [options]
 * @returns {object}
 */
export function initialState(options = {}) {
  return {
    options: {
      levels: optionInt(options.levels, DEFAULTS.levels, 1),
      tileCells: optionInt(options.tileCells, DEFAULTS.tileCells, 1),
      nTransient: optionInt(options.nTransient, DEFAULTS.nTransient, 0),
      nIter: optionInt(options.nIter, DEFAULTS.nIter, 1),
      theta0: Number.isFinite(options.theta0) ? options.theta0 : DEFAULTS.theta0,
      // A 1-D figure (the devil's staircase) fixes Omega and sweeps only K:
      // tiles then carry nOmega = 1 and the viewport's omegaMin is the fixed
      // value, so a degenerate Omega range is the signal, not a rejection.
      lockOmega: options.lockOmega === true,
    },
    generation: 0,
    viewport: null,
    pending: [],
    inFlight: [],
    issued: [],
    droppedGenerations: 0,
    droppedTiles: 0,
    queueDepth: 0,
    retried: [],
  };
}

/**
 * Cell count on one side of a tile at `level`.
 *
 * The finest level uses `tileCells`. Each coarser level halves that count.
 * A level-0 tile occupies one worker, so the coarse side must stay small.
 *
 * @param {number} level
 * @param {{ levels?: number, tileCells?: number }} [options]
 * @returns {number}
 */
export function tileCellsAtLevel(level, options = {}) {
  const levels = optionInt(options.levels, DEFAULTS.levels, 1);
  const tileCells = optionInt(options.tileCells, DEFAULTS.tileCells, 1);
  const raw = Number(level);
  const lv = Number.isFinite(raw) ? Math.floor(raw) : 0;
  const clamped = Math.max(0, Math.min(levels - 1, lv));
  const shifts = levels - 1 - clamped;
  return Math.max(1, Math.floor(tileCells / 2 ** shifts));
}

/**
 * Tiles that cover `viewport`, coarse level first.
 *
 * @param {{ omegaMin: number, omegaMax: number, kMin: number, kMax: number }} viewport
 * @param {{ levels?: number, tileCells?: number, nTransient?: number, nIter?: number, theta0?: number }} [options]
 * @returns {object[]}
 */
export function visibleTiles(viewport, options = {}) {
  const lockOmega = options.lockOmega === true;
  if (!isViewport(viewport, lockOmega)) return [];
  const levels = optionInt(options.levels, DEFAULTS.levels, 1);
  const tileCells = optionInt(options.tileCells, DEFAULTS.tileCells, 1);
  const nTransient = optionInt(options.nTransient, DEFAULTS.nTransient, 0);
  const nIter = optionInt(options.nIter, DEFAULTS.nIter, 1);
  const theta0 = Number.isFinite(options.theta0) ? options.theta0 : DEFAULTS.theta0;
  const { omegaMin, omegaMax, kMin, kMax } = viewport;
  const tiles = [];
  for (let level = 0; level < levels; level++) {
    const n = 2 ** level;
    const dOmega = (omegaMax - omegaMin) / n;
    const dK = (kMax - kMin) / n;
    const cells = tileCellsAtLevel(level, { levels, tileCells });
    // lockOmega: one column at the fixed Omega; the pyramid refines K only.
    const nx = lockOmega ? 1 : n;
    for (let iy = 0; iy < n; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        tiles.push({
          id: tileId(level, ix, iy),
          level,
          ix,
          iy,
          nx,
          ny: n,
          omegaMin: omegaMin + ix * dOmega,
          omegaMax: lockOmega ? omegaMin : omegaMin + (ix + 1) * dOmega,
          kMin: kMin + iy * dK,
          kMax: kMin + (iy + 1) * dK,
          nOmega: lockOmega ? 1 : cells,
          nK: cells,
          nTransient,
          nIter,
          theta0,
        });
      }
    }
  }
  return tiles;
}

export function tileId(level, ix, iy) {
  return `${level}:${ix}:${iy}`;
}

/**
 * @param {object} state
 * @param {{ type: string }} event
 * @returns {{ state: object, commands: object[] }}
 */
export function reduce(state, event) {
  if (state == null || event == null || typeof event.type !== "string") {
    return { state, commands: [] };
  }
  switch (event.type) {
    case "viewport":
      return applyViewport(state, event.viewport);
    case "idle":
      return applyIdle(state);
    case "result":
      return applyResult(state, event);
    case "error":
      return applyError(state, event);
    case "workerError":
      return applyWorkerError(state, event);
    default:
      return { state, commands: [] };
  }
}

function applyViewport(state, viewport) {
  const lockOmega = state.options.lockOmega === true;
  if (!isViewport(viewport, lockOmega)) {
    return { state, commands: [] };
  }
  const hadWork = state.pending.length + state.inFlight.length > 0;
  const generation = state.generation + 1;
  const pending = visibleTiles(viewport, state.options).map((tile) => ({
    ...tile,
    generation,
  }));
  const next = {
    ...state,
    generation,
    viewport,
    pending,
    inFlight: [],
    issued: [],
    retried: [],
    droppedGenerations: state.droppedGenerations + (hadWork ? 1 : 0),
    queueDepth: pending.length,
  };
  const commands = [];
  if (hadWork) {
    commands.push({ type: "cancel", generation: state.generation });
  }
  return { state: next, commands };
}

function applyIdle(state) {
  if (state.pending.length === 0) {
    return { state, commands: [] };
  }
  const tile = state.pending[0];
  const pending = state.pending.slice(1);
  return {
    state: {
      ...state,
      pending,
      issued: state.issued.concat(tile.id),
      inFlight: state.inFlight.concat({
        id: tile.id,
        generation: tile.generation,
      }),
      queueDepth: pending.length,
    },
    commands: [{ type: "issue", tile }],
  };
}

function sameFlight(item, id, generation) {
  return item != null && item.id === id && item.generation === generation;
}

function applyResult(state, event) {
  const id = event.id;
  const generation = event.generation;
  const inFlight = state.inFlight.filter((item) => !sameFlight(item, id, generation));
  const next = { ...state, inFlight };
  if (generation !== state.generation) {
    return {
      state: {
        ...next,
        droppedTiles: state.droppedTiles + 1,
      },
      commands: [
        {
          type: "drop",
          id,
          generation,
          reason: "stale",
          computeMs: event.computeMs,
        },
      ],
    };
  }
  return {
    state: next,
    commands: [
      {
        type: "paint",
        id,
        generation,
        data: event.data,
        header: event.header,
        computeMs: event.computeMs,
      },
    ],
  };
}

function applyError(state, event) {
  const id = event.id;
  const generation = event.generation;
  const inFlight = state.inFlight.filter((item) => !sameFlight(item, id, generation));
  return {
    state: {
      ...state,
      inFlight,
      droppedTiles: state.droppedTiles + 1,
    },
    commands: [
      {
        type: "drop",
        id,
        generation,
        reason: "error",
        message: event.message,
        computeMs: event.computeMs,
      },
    ],
  };
}

/**
 * A worker died or returned a reply that could not be trusted. The tile it
 * held is freed and re-issued at most once within its generation; a second
 * loss drops it for good. `event.tile` is the full issued tile the pool
 * remembered, `event.liveWorkers` the workers still alive after the failure.
 * With no live worker nothing is re-issued: the capacity signal is what the
 * reader gets, not a queue that never drains.
 */
function applyWorkerError(state, event) {
  const id = event.id;
  const generation = event.generation;
  const wasInFlight = state.inFlight.some((item) => sameFlight(item, id, generation));
  const inFlight = state.inFlight.filter((item) => !sameFlight(item, id, generation));
  const canRetry =
    wasInFlight &&
    generation === state.generation &&
    event.liveWorkers > 0 &&
    event.tile != null &&
    !state.retried.includes(id);
  if (canRetry) {
    const pending = [{ ...event.tile, generation }].concat(state.pending);
    return {
      state: {
        ...state,
        inFlight,
        pending,
        retried: state.retried.concat(id),
        queueDepth: pending.length,
      },
      commands: [],
    };
  }
  return {
    state: {
      ...state,
      inFlight,
      droppedTiles: state.droppedTiles + 1,
    },
    commands: [
      {
        type: "drop",
        id,
        generation,
        reason: "worker",
        message: event.message,
        computeMs: event.computeMs,
      },
    ],
  };
}

function isViewport(viewport, lockOmega = false) {
  return (
    viewport != null &&
    Number.isFinite(viewport.omegaMin) &&
    Number.isFinite(viewport.omegaMax) &&
    Number.isFinite(viewport.kMin) &&
    Number.isFinite(viewport.kMax) &&
    (lockOmega ? viewport.omegaMax === viewport.omegaMin : viewport.omegaMax > viewport.omegaMin) &&
    viewport.kMax > viewport.kMin
  );
}

function optionInt(value, fallback, min) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.floor(n));
}
