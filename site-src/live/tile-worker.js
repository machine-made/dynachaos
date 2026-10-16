/**
 * Worker entry for one tile of the (Omega, K) plane.
 *
 * Loads the wasm module from site/wasm/ (this file is copied to site/live/,
 * so ../wasm/ is that directory), calls rotation_number_tile, and transfers
 * the Float64Array back. The four-element header is copied out first because
 * transferring the buffer detaches it.
 *
 * A DedicatedWorkerGlobalScope is the only place this file attaches
 * onmessage; importing it from node for a smoke run does not start a listener.
 */

import init, {
  initSync,
  rotation_number_tile,
  modulated_circle_rotation_tile,
  cml_spacetime_tile,
} from "../wasm/dynachaos_wasm.js";

const HEADER = 4;
let ready = null;

function ensureReady() {
  if (ready == null) ready = loadWasm();
  return ready;
}

async function loadWasm() {
  // The browser glue fetches the .wasm next to itself. Node cannot fetch a
  // file:// URL, so a smoke run reads the bytes and instantiates them.
  const isNode = typeof process !== "undefined" && process.versions && process.versions.node;
  if (!isNode) return init();
  const { readFile } = await import("node:fs/promises");
  const bytes = await readFile(new URL("../wasm/dynachaos_wasm_bg.wasm", import.meta.url));
  return initSync({ module: bytes });
}

function now() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/**
 * @param {object} msg
 * @returns {Promise<object>}
 */
export async function computeTile(msg) {
  const t0 = now();
  await ensureReady();
  let out;
  if (msg.kernel === "modulated_circle_rotation_tile") {
    out = modulated_circle_rotation_tile(
      msg.a,
      msg.c,
      msg.dMin,
      msg.dMax,
      msg.nD,
      msg.eps,
      msg.nTransient,
      msg.nIter,
      msg.theta0,
      msg.phi0,
    );
  } else if (msg.kernel === "cml_spacetime_tile") {
    out = cml_spacetime_tile(
      msg.model,
      msg.eps,
      msg.nSites,
      msg.nTransient,
      msg.nRecord,
      msg.x0,
    );
  } else {
    out = rotation_number_tile(
      msg.omegaMin,
      msg.omegaMax,
      msg.nOmega,
      msg.kMin,
      msg.kMax,
      msg.nK,
      msg.nTransient,
      msg.nIter,
      msg.theta0,
    );
  }
  const computeMs = now() - t0;
  if (out == null || out.length < HEADER) {
    return {
      type: "error",
      id: msg.id,
      generation: msg.generation,
      message: "empty tile",
      computeMs,
    };
  }
  return {
    type: "result",
    id: msg.id,
    generation: msg.generation,
    header: [out[0], out[1], out[2], out[3]],
    data: out,
    computeMs,
  };
}

const workerScope = typeof self !== "undefined" ? self : undefined;
if (workerScope && typeof workerScope.postMessage === "function") {
  workerScope.onmessage = async (event) => {
    const msg = event && event.data;
    if (!msg || (msg.type !== "tile" && msg.type !== "kernel")) return;
    try {
      const reply = await computeTile(msg);
      if (reply.type === "result" && reply.data && reply.data.buffer) {
        workerScope.postMessage(reply, [reply.data.buffer]);
      } else {
        workerScope.postMessage(reply);
      }
    } catch (err) {
      workerScope.postMessage({
        type: "error",
        id: msg.id,
        generation: msg.generation,
        message: String(err && err.message ? err.message : err),
      });
    }
  };
}
