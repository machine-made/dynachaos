/**
 * Main-thread sample of one (Omega, K) point.
 *
 * Loads the wasm glue lazily and answers sample() synchronously once
 * ready. The readout path is called per pointermove; an async round trip
 * through the worker pool would buy nothing for ~30 microseconds.
 *
 * Lock detection stays on; the display-tolerance stop does not. Unlocked
 * cells run the full n_iter average.
 *
 * This file is copied to site/live/, so ../wasm/ is site/wasm/.
 *
 * The glue is imported dynamically so a module without
 * rotation_number_point cannot take the live figure down with it. A failed
 * load is not cached forever: the next ensureLoaded() retries once.
 */

let ready = false;
let loading = null;
let loadFailed = false;
let pointFn = null;
let glueHref = new URL("../wasm/dynachaos_wasm.js", import.meta.url).href;

export function isReady() {
  return ready;
}

export function ensureLoaded(glueUrl) {
  // A URL is honoured only before a load has started: while one is in flight
  // the argument is ignored, so a late caller cannot retarget the URL the
  // one retry after a failure would use.
  if (loading == null) {
    if (glueUrl) glueHref = glueUrl;
    loading = loadWasm();
  }
  return loading;
}

async function loadWasm() {
  try {
    const glue = await import(glueHref);
    if (typeof glue.rotation_number_point !== "function") {
      throw new Error("wasm glue is missing rotation_number_point");
    }
    const isNode =
      typeof process !== "undefined" && process.versions && process.versions.node;
    if (!isNode) {
      await glue.default();
    } else {
      const { readFile } = await import("node:fs/promises");
      const bytes = await readFile(new URL("dynachaos_wasm_bg.wasm", glueHref));
      glue.initSync({ module: bytes });
    }
    pointFn = glue.rotation_number_point;
    ready = true;
  } catch (err) {
    if (!loadFailed) {
      loadFailed = true;
      loading = null;
    }
    throw err;
  }
}

/**
 * Rotation number of one point. `null` until the module is ready, or if
 * the arguments are not finite.
 *
 * @param {number} omega
 * @param {number} K
 * @param {number} [nTransient]
 * @param {number} [nIter]
 * @param {number} [theta0]
 * @returns {number | null}
 */
export function sample(omega, K, nTransient, nIter, theta0) {
  if (!ready || typeof pointFn !== "function") {
    ensureLoaded().catch(() => {});
    return null;
  }
  if (!Number.isFinite(omega) || !Number.isFinite(K)) return null;
  const transient = Number.isFinite(nTransient) ? nTransient : 200;
  const iter = Number.isFinite(nIter) ? nIter : 2000;
  const theta = Number.isFinite(theta0) ? theta0 : 0.1;
  const out = pointFn(omega, K, transient, iter, theta);
  if (out == null || out.length < 1) return null;
  const value = out[0];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
