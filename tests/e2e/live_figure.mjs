// Live Arnold-tongues figure, driven over the Chrome DevTools protocol.
//
//   node tests/e2e/live_figure.mjs <site dir>
//
// Serves the built site, launches headless Chrome, and checks the live path.
// The script uses only node built-ins and the globals that node 22 provides.
// It does not use --virtual-time-budget: that hang waits forever with workers.
//
// Anchor values, from the Python kernel, all below K_c = 1/(2π):
//   rho = 0 at (Omega 0.05, K 0.12) and at (Omega 0.08, K 0.12)
//   rho = 1 at (Omega 0.95, K 0.12)
//   rho ~ 0.0743 at (Omega 0.08, K 0.03)  — this pair catches a flipped K axis

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const SITE = resolve(process.argv[2] || "site");
const F = "document.getElementById('fig:arnold_tongues')";
const S = "document.getElementById('fig:devils_staircase')";
// Displayed non-live canvas.plot. Every reduced-data JSON-chart check uses this.
function shownStaticChart(root) {
  return (
    `Array.from(${root}.querySelectorAll("canvas.plot")).some(` +
    `(el) => !el.classList.contains("live") && getComputedStyle(el).display !== "none")`
  );
}
const CHROME_CMDS = ["google-chrome", "google-chrome-stable"];
const PAGE_READY_MS = 20_000;
const FIGURE_MS = 20_000;
const FIRST_PAINT_MS = 30_000;
const REFINE_MS = 45_000;
const SAMPLE_MS = 20_000;
const VIEW_MS = 20_000;
const REDUCED_WAIT_MS = 3_000;
const NARROW_WAIT_MS = 8_000;
const KERNEL_PAINT_MS = 120_000;
// Hang ceiling for the flagship's first full paint, from the interact click
// until the pool queue and in-flight set are empty. One value on every
// machine. This is not a kernel budget. The old 4x CPU-throttle wall-clock
// check is removed: wasm download and worker start-up dominated it, so a
// worker that calls the kernel three times still finished (2026-10-02: 3x
// passed both 13155 ms here and 17868 ms on CI). On this shared box the
// throttle also stalled tile issue (26 tiles in 120 s) while computeMs stayed
// ~29 ms, so a throttled wall clock does not measure the kernel. 120000 ms
// only fails a paint that does not finish.
// One numeric limit, with no machine/CI branch; cross-machine transfer is not
// established by the local runs below.
// Numerator: the 25th percentile (linear interpolation between sorted ranks)
// of per-tile computeMs for the finest-level tiles in the first full paint.
// Also report p10 and p50 so measured runs retain the pre-registered p10
// fallback and the former median for comparison. Telemetry is under
// #live-debug; tile ids are "<level>:<ix>:<iy>". Denominator: median of five
// timed batches of exactly KERNEL_REF_CALLS calls to
// circle_map_lyapunov_sum(0.12, 0.25, 2000, 200000, 0.1) inside a module Blob
// worker importing the site's wasm/dynachaos_wasm.js by absolute URL. Discard
// one equally sized warmup batch. Each call runs 2000 transient plus 200000
// measured wrapped circle-map steps (theta' = theta + D + A sin(2 pi theta),
// lambda = mean log|d theta' / d theta|). Worker performance.now() times each
// batch. N=20 was calibrated to make an unloaded batch about 100 ms here; there
// is no duration target. A slowdown/load lengthens the fixed-work reference.
// Known blind spot: a slowdown of all wasm at once moves both sides.
// Margin: limit must be >= 1.5x the highest unchanged/loaded ratio and <= the
// lowest 3x ratio / 1.5.
// Final p25 ratio bands (min/median/max): unchanged 0.4026/0.4152/0.4329,
// loaded 0.3901/0.3952/0.3984, and 3x 1.2474/1.2689/1.2756.
// KERNEL_RATIO_LIMIT is 0.75; the 1.5x margin window is [0.6494, 0.8316].
// Calibration tables and the selected limit are in the goal answer file.
const KERNEL_REF_CALLS = 20;
const KERNEL_REF_BATCHES = 5;
const KERNEL_RATIO_LIMIT = 0.75;
const DEADLINE_SUM_MS =
  2 * (PAGE_READY_MS + FIGURE_MS) +
  4 * FIRST_PAINT_MS +
  REFINE_MS +
  8 * SAMPLE_MS +
  VIEW_MS +
  2 * REDUCED_WAIT_MS +
  NARROW_WAIT_MS +
  FIRST_PAINT_MS +
  2 * (PAGE_READY_MS + FIGURE_MS + FIRST_PAINT_MS) +
  2 * (PAGE_READY_MS + FIRST_PAINT_MS) +
  PAGE_READY_MS +
  FIGURE_MS +
  FIRST_PAINT_MS +
  REFINE_MS +
  KERNEL_PAINT_MS;
const WATCHDOG_MS = DEADLINE_SUM_MS + 600_000;
const WASM_RE = /dynachaos_wasm_bg\.wasm/;
const TELEMETRY_EVENTS = ["paint", "drop", "error", "cancel"];
const failures = [];
const pageLogs = [];
const debugLogs = [];
const chromeTried = [];

const check = (name, ok, detail) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail === undefined ? "" : ` -- ${detail}`}`);
  if (!ok) failures.push(name);
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const freePort = () =>
  new Promise((res) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const p = s.address().port;
      s.close(() => res(p));
    });
  });

if (!existsSync(SITE)) {
  console.log(`FAIL the site directory does not exist: ${SITE}`);
  process.exit(1);
}

function formatConsole(event) {
  const args = (event.params && event.params.args) || [];
  return args
    .map((a) => {
      if (a.value !== undefined) return typeof a.value === "string" ? a.value : JSON.stringify(a.value);
      if (a.preview && a.preview.properties) {
        const fields = a.preview.properties.map((p) => `${p.name}:${p.value}`).join(",");
        return `{${fields}}`;
      }
      return a.description || a.type || "";
    })
    .join(" ");
}

function isTelemetry(line) {
  return TELEMETRY_EVENTS.some(
    (event) => line.includes(`"event":"${event}"`) || line.includes(`event:${event}`),
  );
}

function median(values) {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function percentile(values, p) {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (sorted.length - 1) * p;
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  return sorted[low] + (sorted[high] - sorted[low]) * (rank - low);
}

function paintTiles(lines) {
  const tiles = [];
  for (const line of lines) {
    const at = line.indexOf("{");
    if (at < 0) continue;
    let obj;
    try {
      obj = JSON.parse(line.slice(at));
    } catch {
      continue;
    }
    if (!obj || obj.event !== "paint" || !Number.isFinite(obj.computeMs)) continue;
    tiles.push(obj);
  }
  return tiles;
}

function finestComputeMs(tiles) {
  let level = -1;
  for (const tile of tiles) {
    const parsed = Number(String(tile.tile).split(":")[0]);
    if (Number.isFinite(parsed) && parsed > level) level = parsed;
  }
  const ms = [];
  for (const tile of tiles) {
    if (Number(String(tile.tile).split(":")[0]) === level) ms.push(tile.computeMs);
  }
  return { level, ms };
}

function dumpConsole() {
  const lines = debugLogs.length ? debugLogs.concat(pageLogs) : pageLogs;
  console.log("--- page console ---");
  if (lines.length === 0) {
    console.log("(empty)");
    return;
  }
  const interesting = lines.filter(isTelemetry);
  console.log(`${lines.length} debug lines; ${interesting.length} paint/drop/error/cancel`);
  const shown =
    interesting.length <= 12 ? interesting : interesting.slice(0, 6).concat(["..."], interesting.slice(-6));
  for (const line of shown) console.log(line);
  console.log(lines[lines.length - 1]);
}

let chrome = null;
let server = null;
let profile = null;
let chromeArgs = [];
let cleaned = false;

function chromeCommandLine(cmd) {
  return [cmd, ...chromeArgs].join(" ");
}

function failChromeStart(extra) {
  const tried = chromeTried.length ? chromeTried.join(" ; ") : chromeCommandLine(CHROME_CMDS[0]);
  console.log(`FAIL Chrome did not start. Command tried: ${tried}${extra ? ` -- ${extra}` : ""}`);
}

async function stopChild(child, { group = false } = {}) {
  if (!child || child.pid == null) return;
  if (child.exitCode != null || child.signalCode != null) return;
  const pid = child.pid;
  const exited = new Promise((resolve) => child.once("exit", resolve));
  try {
    if (group) process.kill(-pid, "SIGTERM");
    else child.kill("SIGTERM");
  } catch {
    try {
      child.kill("SIGTERM");
    } catch {}
  }
  const outcome = await Promise.race([exited.then(() => "exited"), sleep(2_000).then(() => "timeout")]);
  if (outcome === "timeout") {
    try {
      if (group) process.kill(-pid, "SIGKILL");
    } catch {}
    try {
      child.kill("SIGKILL");
    } catch {}
    await Promise.race([exited, sleep(1_000)]);
  }
}

async function removeProfile() {
  if (!profile) return;
  for (let i = 0; i < 8; i++) {
    try {
      rmSync(profile, { recursive: true, force: true });
      if (!existsSync(profile)) return;
    } catch {}
    await sleep(250);
  }
  try {
    rmSync(profile, { recursive: true, force: true });
  } catch {}
}

async function cleanup() {
  if (cleaned) return;
  cleaned = true;
  await stopChild(chrome, { group: true });
  await stopChild(server);
  await removeProfile();
}

const watchdog = setTimeout(() => {
  console.log(`FAIL the browser check ran past ${WATCHDOG_MS / 1000} s`);
  try {
    dumpConsole();
  } catch {}
  cleanup().finally(() => process.exit(1));
}, WATCHDOG_MS);

const httpPort = await freePort();
const cdpPort = await freePort();
profile = mkdtempSync(join(tmpdir(), "dc-115-"));
chromeArgs = [
  // Chrome ignores an emulated prefers-reduced-data value unless this Blink
  // feature is on. Without the flag the media query stays false.
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--no-first-run",
  "--enable-blink-features=PrefersReducedData",
  `--user-data-dir=${profile}`,
  `--remote-debugging-port=${cdpPort}`,
  "--window-size=1400,1000",
  "about:blank",
];

async function startChrome() {
  for (const cmd of CHROME_CMDS) {
    chromeTried.push(chromeCommandLine(cmd));
    const child = spawn(cmd, chromeArgs, { stdio: "ignore", detached: true });
    const err = await new Promise((resolve) => {
      const onError = (e) => resolve(e);
      child.once("error", onError);
      child.once("spawn", () => {
        child.off("error", onError);
        resolve(null);
      });
    });
    if (!err) return child;
  }
  return null;
}

server = spawn(
  "python3",
  ["-m", "http.server", String(httpPort), "--bind", "127.0.0.1", "--directory", SITE],
  { stdio: "ignore" },
);
chrome = await startChrome();

async function up(url) {
  for (let i = 0; i < 150; i++) {
    try {
      await fetch(url);
      return;
    } catch {
      await sleep(100);
    }
  }
  throw new Error(`not reachable: ${url}`);
}

let exitCode = 1;
try {
  if (!chrome) {
    failChromeStart();
    throw new Error("chrome-missing");
  }
  try {
    await up(`http://127.0.0.1:${cdpPort}/json/version`);
  } catch (err) {
    failChromeStart(err.message);
    throw err;
  }
  await up(`http://127.0.0.1:${httpPort}/index.html`);
  const targets = await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json();
  const page = targets.find((t) => t.type === "page");
  if (!page) throw new Error("Chrome has no page target");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r, j) => {
    ws.onopen = r;
    ws.onerror = j;
  });
  let nextId = 0;
  const waiters = new Map();
  const events = [];
  const attached = new Map();
  let attachChain = Promise.resolve();
  const send = (method, params = {}, sessionId) =>
    new Promise((resolve, reject) => {
      const id = ++nextId;
      const timer = setTimeout(() => {
        if (!waiters.has(id)) return;
        waiters.delete(id);
        reject(new Error(`cdp ${method} timed out`));
      }, 15_000);
      waiters.set(id, (msg) => {
        clearTimeout(timer);
        resolve(msg);
      });
      const payload = { id, method, params };
      if (sessionId) payload.sessionId = sessionId;
      ws.send(JSON.stringify(payload));
    });
  const onAttached = (params) => {
    attached.set(params.sessionId, params.targetInfo);
    attachChain = attachChain
      .then(async () => {
        await send("Network.enable", {}, params.sessionId);
        await send("Runtime.enable", {}, params.sessionId);
        if (params.waitingForDebugger) {
          await send("Runtime.runIfWaitingForDebugger", {}, params.sessionId);
        }
      })
      .catch(() => {});
  };
  ws.onmessage = (m) => {
    let d;
    try {
      d = JSON.parse(m.data);
    } catch {
      return;
    }
    if (d.id != null && waiters.has(d.id)) {
      waiters.get(d.id)(d);
      waiters.delete(d.id);
    } else if (d.method) {
      events.push(d);
      if (d.method === "Runtime.consoleAPICalled") pageLogs.push(formatConsole(d));
      if (d.method === "Target.attachedToTarget") onAttached(d.params);
      if (d.method === "Target.detachedFromTarget") attached.delete(d.params.sessionId);
    }
  };
  const ev = async (expression) => {
    const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (r.result && r.result.exceptionDetails) {
      const x = r.result.exceptionDetails;
      throw new Error((x.exception && x.exception.description) || x.text);
    }
    return r.result && r.result.result ? r.result.result.value : undefined;
  };
  const waitFor = async (expression, ms, step = 100) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      await Promise.race([attachChain, sleep(5_000)]);
      try {
        if (await ev(expression)) return true;
      } catch {}
      await sleep(step);
    }
    return false;
  };
  const pageUrl = `http://127.0.0.1:${httpPort}/index.html?debug=1`;
  const installLogHook = () =>
    ev(`(() => {
      if (window.__dcHooked) return true;
      window.__dcHooked = true;
      window.__dcLog = [];
      const orig = console.info.bind(console);
      console.info = function () {
        try {
          window.__dcLog.push(
            Array.from(arguments)
              .map((a) => (a && typeof a === "object" ? JSON.stringify(a) : String(a)))
              .join(" ")
          );
        } catch (e) {}
        return orig.apply(console, arguments);
      };
      return true;
    })()`);
  const pullDebug = async () => {
    try {
      const hooked = await ev("window.__dcLog || []");
      if (Array.isArray(hooked)) debugLogs.push(...hooked);
    } catch {}
  };
  const load = async (url = pageUrl) => {
    await send("Page.navigate", { url });
    await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
    await waitFor(`!!${F}`, FIGURE_MS);
    await installLogHook();
  };
  const stats = () => ev(`${F} && ${F}._live ? ${F}._live.stats() : null`);
  const wasmHits = (from) =>
    events.slice(from).filter(
      (e) =>
        e.method === "Network.requestWillBeSent" &&
        e.params &&
        e.params.request &&
        WASM_RE.test(e.params.request.url),
    );
  const attachedWorkers = () => [...attached.values()].filter((t) => t && t.type === "worker");
  const imgShown = () =>
    ev(`(() => {
      const img = ${F} && ${F}.querySelector(".fig-body img");
      if (!img) return false;
      const s = getComputedStyle(img);
      return s.display !== "none" && s.visibility !== "hidden";
    })()`);

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Network.enable");
  await send("Network.setCacheDisabled", { cacheDisabled: true });
  await send("Target.setAutoAttach", {
    autoAttach: true,
    waitForDebuggerOnStart: true,
    flatten: true,
  });

  await load(`${pageUrl}#live-debug`);
  check(
    "the figure is marked live-capable",
    await ev(`!!${F} && !!${F}.dataset.live`),
    await ev(`${F} ? JSON.stringify(${F}.dataset) : "no figure"`),
  );
  check("it has an interact button", await ev(`!!(${F} && ${F}.querySelector(".act-interact"))`));

  const kernelT0 = Date.now();
  await ev(`(${F}.querySelector(".act-interact").click(), true)`);
  const firstPaint = await waitFor(`!!${F}._live && ${F}._live.stats().painted >= 1`, FIRST_PAINT_MS, 50);
  await attachChain;
  const sFirst = await stats();
  check("the first coarse tile is painted", firstPaint && sFirst && sFirst.painted >= 1, JSON.stringify(sFirst));
  check("pressing interact spawns workers", !!(sFirst && sFirst.workerCount >= 1), JSON.stringify(sFirst));

  const captured = firstPaint
    ? await ev(`(() => {
        const c = ${F}.querySelector("canvas.plot");
        if (!c) return false;
        window.__dcCoarse = c.getContext("2d").getImageData(0, 0, c.width, c.height);
        return true;
      })()`)
    : false;
  const coarsePainted = sFirst && sFirst.painted ? sFirst.painted : 0;

  let lastPainted = coarsePainted;
  let stableSince = Date.now();
  let refined = false;
  if (firstPaint) {
    const refineT0 = Date.now();
    while (Date.now() - refineT0 < REFINE_MS) {
      await attachChain;
      const p = ((await stats()) || {}).painted || 0;
      if (p !== lastPainted) {
        lastPainted = p;
        stableSince = Date.now();
      } else if (lastPainted > coarsePainted && Date.now() - stableSince >= 2_000) {
        refined = true;
        break;
      }
      await sleep(150);
    }
  }
  const compare = captured
    ? await ev(`(() => {
        const c = ${F}.querySelector("canvas.plot");
        const coarse = window.__dcCoarse;
        if (!c || !coarse) return { comparable: false, reason: "missing canvas" };
        if (c.width !== coarse.width || c.height !== coarse.height) {
          return {
            comparable: false,
            reason: "canvas resized",
            coarseWidth: coarse.width,
            coarseHeight: coarse.height,
            width: c.width,
            height: c.height,
          };
        }
        const a = coarse.data;
        const b = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
        let differ = false;
        for (let i = 0; i < a.length; i++) {
          if (a[i] !== b[i]) {
            differ = true;
            break;
          }
        }
        return { comparable: true, differ };
      })()`)
    : { comparable: false, reason: "no coarse capture" };

  check(
    "the picture changes between the coarse pass and the refined pass",
    Boolean(captured && refined && lastPainted > coarsePainted && compare && compare.comparable && compare.differ),
    JSON.stringify({ captured, coarsePainted, refinedPainted: lastPainted, refined, compare }),
  );
  const kernelPaintDone = await waitFor(
    `(() => {
      const fig = ${F};
      const pool = fig && fig._pool;
      if (!pool || !fig._live) return false;
      const st = pool.getState();
      return fig._live.stats().painted > 1 && st.queueDepth === 0 && st.inFlight.length === 0;
    })()`,
    Math.max(1_000, KERNEL_PAINT_MS - (Date.now() - kernelT0)),
    100,
  );
  const kernelElapsed = Date.now() - kernelT0;
  const lineText = await ev(`(() => {
    const el = ${F}.querySelector(".live-debug-report .live-debug-line");
    return el ? el.textContent : null;
  })()`);
  const reportFields = lineText ? lineText.trim().split(/\s+/) : [];
  const reportSixFields = reportFields.length === 6;
  const reportFigId = reportFields[0] === "fig:arnold_tongues";
  const reportUa = typeof reportFields[1] === "string" && reportFields[1].length > 0;
  const reportWorkers = Number.isFinite(Number(reportFields[2])) && Number(reportFields[2]) > 0;
  const reportElapsed = Number.isFinite(Number(reportFields[3])) && Number(reportFields[3]) > 0;
  const reportMedian = Number.isFinite(Number(reportFields[4])) && Number(reportFields[4]) > 0;
  const reportTasks = Number.isFinite(Number(reportFields[5])) && Number(reportFields[5]) >= 0;
  const reportAllNumbers = reportWorkers && reportElapsed && reportMedian && reportTasks;
  check(
    "the live debug report box shows six filled fields (all numbers in Chrome)",
    Boolean(kernelPaintDone && reportSixFields && reportFigId && reportUa && reportAllNumbers),
    JSON.stringify({ lineText, reportFields }),
  );
  const copyClicked = await ev(`(async () => {
    const btn = ${F}.querySelector(".live-debug-report .live-debug-copy");
    if (!btn) return false;
    btn.click();
    return true;
  })()`);
  check("the live debug report copy button is clickable", copyClicked);
  await pullDebug();
  const kernelLogs = debugLogs.filter((line) => line.includes("[dynachaos-live]"));
  const paintedTiles = paintTiles(kernelLogs);
  const finest = finestComputeMs(paintedTiles);
  const kernelP10Ms = percentile(finest.ms, 0.1);
  const kernelP25Ms = percentile(finest.ms, 0.25);
  const kernelP50Ms = percentile(finest.ms, 0.5);
  const reference = await ev(`(async () => {
    const wasmUrl = new URL("/wasm/dynachaos_wasm.js", location.origin).href;
    const src =
      "import init, { circle_map_lyapunov_sum } from " + JSON.stringify(wasmUrl) + ";\\n" +
      "onmessage = async (e) => {\\n" +
      "  const n = (e.data && e.data.batches) || 5;\\n" +
      "  const calls = (e.data && e.data.calls) || 5;\\n" +
      "  const samples = [];\\n" +
      "  try {\\n" +
      "    await init();\\n" +
      "    let sum = 0;\\n" +
      "    for (let i = 0; i < calls; i++) {\\n" +
      "      sum += circle_map_lyapunov_sum(0.12, 0.25, 2000, 200000, 0.1)[2];\\n" +
      "    }\\n" +
      "    for (let i = 0; i < n; i++) {\\n" +
      "      const t0 = performance.now();\\n" +
      "      for (let j = 0; j < calls; j++) {\\n" +
      "        sum += circle_map_lyapunov_sum(0.12, 0.25, 2000, 200000, 0.1)[2];\\n" +
      "      }\\n" +
      "      samples.push(performance.now() - t0);\\n" +
      "    }\\n" +
      "    postMessage({ samples, sum });\\n" +
      "  } catch (err) {\\n" +
      "    postMessage({ error: String(err && err.message ? err.message : err) });\\n" +
      "  }\\n" +
      "};";
    const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    const w = new Worker(url, { type: "module" });
    try {
      return await new Promise((resolve) => {
        const t = setTimeout(
          () => resolve({ error: "kernel-reference worker timed out" }),
          30000,
        );
        w.onmessage = (m) => {
          clearTimeout(t);
          resolve(m.data);
        };
        w.onerror = (e) => {
          clearTimeout(t);
          resolve({ error: "kernel-reference worker failed: " + e.message });
        };
        w.postMessage({ batches: ${KERNEL_REF_BATCHES}, calls: ${KERNEL_REF_CALLS} });
      });
    } finally {
      w.terminate();
      URL.revokeObjectURL(url);
    }
  })()`);
  const referenceSamples = reference && Array.isArray(reference.samples)
    ? reference.samples
    : [];
  const referenceMs = median(referenceSamples);
  console.log(
    `flagship full paint: ${kernelElapsed} ms, hang ceiling ${KERNEL_PAINT_MS} ms`,
  );
  check(
    "the flagship's first full paint finishes within the hang ceiling",
    Boolean(kernelPaintDone && kernelElapsed <= KERNEL_PAINT_MS),
    `elapsed ${kernelElapsed} ms, hang ceiling ${KERNEL_PAINT_MS} ms, done ${kernelPaintDone}`,
  );
  const kernelP10Ratio = kernelP10Ms / referenceMs;
  const kernelP25Ratio = kernelP25Ms / referenceMs;
  const kernelP50Ratio = kernelP50Ms / referenceMs;
  console.log(
    `kernel cost ratio percentiles: p10 ${kernelP10Ratio} (${kernelP10Ms} ms), p25 ${kernelP25Ratio} (${kernelP25Ms} ms), p50 ${kernelP50Ratio} (${kernelP50Ms} ms), reference ${referenceMs} ms (level ${finest.level}, tiles ${finest.ms.length}/${paintedTiles.length}, limit ${KERNEL_RATIO_LIMIT}, ref batches ${JSON.stringify(referenceSamples)}, calls/batch ${KERNEL_REF_CALLS}, warmup calls ${KERNEL_REF_CALLS}, ref error ${reference && reference.error})`,
  );
  check(
    "the flagship kernel cost stays within the same-run wasm-worker reference budget",
    Boolean(
      kernelPaintDone &&
        finest.ms.length >= 16 &&
        Number.isFinite(kernelP25Ratio) &&
        kernelP25Ratio <= KERNEL_RATIO_LIMIT,
    ),
    `p25 ratio ${kernelP25Ratio} = interpolated p25 computeMs ${kernelP25Ms} / wasm-worker reference ${referenceMs} ms, limit ${KERNEL_RATIO_LIMIT}, p10 ${kernelP10Ratio}, p50 ${kernelP50Ratio}, level ${finest.level}, finest tiles ${finest.ms.length}, paint events ${paintedTiles.length}, done ${kernelPaintDone}, ref error ${reference && reference.error}`,
  );


  const sLive = await stats();
  const nIter = sLive && sLive.nIter ? sLive.nIter : 2000;
  const tol = 2 / nIter;
  const sampleMs = firstPaint ? SAMPLE_MS : 2_000;
  const sampleWhenReady = async (omega, K, ms = sampleMs) => {
    const hasLive = await ev(`${F} && ${F}._live`);
    if (!hasLive) return null;
    await waitFor(`${F}._live.sample(${omega}, ${K}) !== null`, ms);
    return ev(`${F}._live ? ${F}._live.sample(${omega}, ${K}) : null`);
  };
  const a = await sampleWhenReady(0.05, 0.12);
  check("rho = 0 at (Omega 0.05, K 0.12)", a !== null && Math.abs(a) <= tol, a);
  const b = await sampleWhenReady(0.95, 0.12);
  check("rho = 1 at (Omega 0.95, K 0.12)", b !== null && Math.abs(b - 1) <= tol, b);
  const c = await sampleWhenReady(0.08, 0.12);
  check("rho = 0 at (Omega 0.08, K 0.12)", c !== null && Math.abs(c) <= tol, c);
  const d = await sampleWhenReady(0.08, 0.03);
  check("rho ~ 0.0743 at (Omega 0.08, K 0.03)", d !== null && Math.abs(d - 0.0743) < 0.03, d);
  check(
    "readout at locked (Omega 0.05, K 0.12) is 0 to 1e-12",
    a !== null && Math.abs(a) <= 1e-12,
    a,
  );
  const displayTol = 1 / 256;
  const readoutPairs = [
    [0.05, 0.12, a],
    [0.95, 0.12, b],
    [0.08, 0.12, c],
    [0.08, 0.03, d],
  ];
  const vsRaster = [];
  for (const [omega, K, value] of readoutPairs) {
    if (value == null) {
      vsRaster.push({ omega, K, value, raster: null, ok: false });
      continue;
    }
    await waitFor(
      `${F}._live && typeof ${F}._live.rasterSample === "function" && ${F}._live.rasterSample(${omega}, ${K}) !== null`,
      sampleMs,
    );
    const rasterVal = await ev(
      `${F}._live && typeof ${F}._live.rasterSample === "function" ? ${F}._live.rasterSample(${omega}, ${K}) : null`,
    );
    const ok =
      rasterVal !== null &&
      Number.isFinite(rasterVal) &&
      Math.abs(value - rasterVal) <= displayTol;
    vsRaster.push({ omega, K, value, raster: rasterVal, ok });
  }
  check(
    "readout never differs from the raster sample by more than the display tolerance",
    vsRaster.length > 0 && vsRaster.every((row) => row.ok),
    JSON.stringify(vsRaster),
  );

  // Every check above passes on the raster value alone, because a locked cell
  // reads the same either way. This one does not: it imports the same module
  // instance the page imported and demands the readout return that value, at
  // an unlocked point where the two paths differ.
  await waitFor(`${F}._live && ${F}._live.pointReady && ${F}._live.pointReady()`, sampleMs);
  const wired = await ev(`(async () => {
    const m = await import(new URL("live/point.js", document.baseURI).href);
    const direct = m.sample(0.08, 0.03, 200, 2000, 0.1);
    return {
      ready: m.isReady(),
      direct,
      readout: ${F}._live.sample(0.08, 0.03),
      raster: ${F}._live.rasterSample(0.08, 0.03),
    };
  })()`);
  check(
    "the readout is the point kernel, not the tile lookup",
    wired != null && wired.ready === true && wired.direct !== null && wired.readout === wired.direct,
    JSON.stringify(wired),
  );

  // The raster check above cannot see the display stop: rasterSample() reads
  // the tile cell at its grid coordinate, so it differs from the readout
  // whatever the point kernel does. This one computes a 1 x 1 tile at the
  // SAME exact point through the page's own glue module -- the instance
  // point.js already initialised -- so the only difference left is the
  // kernel's early stop. At an unlocked point the tile's display stop returns
  // a truncated average; the readout must run the full n_iter and differ.
  // A locked cell legitimately matches (exact rational either way), so the
  // demand is that at least one point differs.
  const tilePairs = [
    [0.08, 0.03],
    [0.13, 0.04],
    [0.55, 0.07],
    [0.62, 0.1],
    [0.91, 0.04],
  ];
  const vsTile = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    return ${JSON.stringify(tilePairs)}.map(([o, K]) => {
      const readout = ${F}._live.sample(o, K);
      const tile = glue.rotation_number_tile(o, o, 1, K, K, 1, 200, 2000, 0.1)[4];
      return { omega: o, K, readout, tile, differ: Number.isFinite(readout) && readout !== tile };
    });
  })()`);
  check(
    "the readout runs every step the tile's display stop would skip",
    Array.isArray(vsTile) && vsTile.length === tilePairs.length && vsTile.some((row) => row.differ),
    JSON.stringify(vsTile),
  );

  // Keyboard cursor: focus the live canvas and press ArrowRight. The keydown
  // handler steps a stepped x-index cursor through xValues() and shows the
  // same readout the pointer uses; with no x values it returns early and the
  // readout never appears.
  const liveCanvasFocused = await ev(`(() => {
    const c = ${F} && ${F}.querySelector("canvas.plot.live");
    if (!c) return false;
    c.focus();
    return document.activeElement === c;
  })()`);
  if (liveCanvasFocused) {
    await send("Input.dispatchKeyEvent", { type: "keyDown", key: "ArrowRight", code: "ArrowRight" });
    await send("Input.dispatchKeyEvent", { type: "keyUp", key: "ArrowRight", code: "ArrowRight" });
  }
  const keyReadout = liveCanvasFocused
    ? await waitFor(
        `(() => {
          const tip = ${F} && ${F}.querySelector(".readout.on");
          if (!tip) return false;
          const lines = tip.textContent.split("\\n");
          const last = lines[lines.length - 1] || "";
          const m = last.match(/=\\s*(\\S+)\\s*$/);
          return !!m && Number.isFinite(parseFloat(m[1]));
        })()`,
        sampleMs,
      )
    : false;
  check(
    "ArrowRight on the focused live canvas shows a finite rotation number",
    keyReadout,
    JSON.stringify({
      liveCanvasFocused,
      readout: await ev(`(() => {
        const tip = ${F} && ${F}.querySelector(".readout");
        return tip ? { on: tip.classList.contains("on"), text: tip.textContent } : null;
      })()`),
    }),
  );

  const sBeforeView = await stats();
  if (sBeforeView && (await ev(`!!${F}._live && typeof ${F}._live.setView === "function"`))) {
    await ev(`(${F}._live.setView({ omegaMin: 0, omegaMax: 0.3, kMin: 0, kMax: 0.15 }), true)`);
    const grew = await waitFor(
      `${F}._live.stats().generation > ${sBeforeView.generation} && ${F}._live.stats().painted >= 1`,
      firstPaint ? VIEW_MS : 3_000,
    );
    check(
      "at least one refinement generation completes after a view change",
      grew,
      JSON.stringify({ before: sBeforeView, after: await stats() }),
    );
  } else {
    check("at least one refinement generation completes after a view change", false, "no _live.setView");
  }

  // Keyboard seams: the first arrow after a view change reads the snapped
  // position instead of stepping off it, a keyboard zoom refreshes a shown
  // tip, and a degenerate setView domain still reads a finite Omega.
  const pressKey = async (key, code = key) => {
    await send("Input.dispatchKeyEvent", { type: "keyDown", key, code });
    await send("Input.dispatchKeyEvent", { type: "keyUp", key, code });
  };
  const tipField = (line) =>
    `(() => {
      const tip = ${F} && ${F}.querySelector(".readout.on");
      if (!tip) return null;
      const lines = tip.textContent.split("\\n");
      const m = (lines[${line}] || "").match(/=\\s*(\\S+)\\s*$/);
      return m ? parseFloat(m[1]) : null;
    })()`;
  const liveCanvasSel = `${F} && ${F}.querySelector("canvas.plot.live")`;

  // The cursor from the ArrowRight above sits at ~0.548, outside the new
  // [0, 0.3] domain. The first ArrowLeft must read the snapped position
  // (0.3, the last of the 32 grid points); stepping off it would read 0.2903.
  await ev(`(() => { const c = ${liveCanvasSel}; if (c) c.focus(); return document.activeElement === c; })()`);
  await pressKey("ArrowLeft");
  const snapOmega = await ev(tipField(0));
  check(
    "the first arrow after a view change reads the snapped position",
    snapOmega !== null && Math.abs(snapOmega - 0.3) < 0.005,
    JSON.stringify({ snapOmega, tip: await ev(`(${liveCanvasSel}) ? ${F}.querySelector(".readout").textContent : null`) }),
  );
  await pressKey("ArrowLeft");
  const stepOmega = await ev(tipField(0));
  check(
    "the next arrow steps one grid position",
    stepOmega !== null && Math.abs(stepOmega - 0.2903) < 0.005,
    JSON.stringify({ stepOmega }),
  );

  // Zooming out about the cursor clamps the y domain at its lower edge, so
  // the K midline moves (0.075 -> 0.1) while Omega stays put. A stale tip
  // would keep the old K line.
  await ev(`(() => { const tip = ${F}.querySelector(".readout"); window.__dcTip = tip ? tip.textContent : null; return true; })()`);
  await pressKey("ArrowDown");
  const tipAfterZoom = await waitFor(
    `(() => {
      const tip = ${F} && ${F}.querySelector(".readout.on");
      return !!tip && tip.textContent !== window.__dcTip;
    })()`,
    3_000,
  );
  const zoomOmega = await ev(tipField(0));
  const zoomK = await ev(tipField(1));
  check(
    "a keyboard zoom re-renders the shown tip at the same Omega",
    Boolean(
      tipAfterZoom &&
        zoomOmega !== null &&
        Math.abs(zoomOmega - 0.2903) < 0.005 &&
        zoomK !== null &&
        Math.abs(zoomK - 0.1) < 0.005,
    ),
    JSON.stringify({ tipAfterZoom, zoomOmega, zoomK, tip: await ev(`${F}.querySelector(".readout").textContent`) }),
  );

  // A hidden tip must stay hidden through a keyboard zoom.
  await pressKey("0", "Digit0");
  await pressKey("ArrowUp");
  const tipHidden = await ev(`!(${F} && ${F}.querySelector(".readout.on"))`);
  check("a keyboard zoom leaves a hidden tip hidden", tipHidden, JSON.stringify({ tipHidden }));

  // A degenerate domain must be clamped like zoomAt's, so the arrow readout
  // still shows a finite Omega instead of dividing by zero.
  await ev(`(${F}._live.setView({ omegaMin: 0.5, omegaMax: 0.5, kMin: 0.1, kMax: 0.1 }), true)`);
  await ev(`(() => { const c = ${liveCanvasSel}; if (c) c.focus(); return document.activeElement === c; })()`);
  await pressKey("ArrowLeft");
  const degOmega = await ev(tipField(0));
  check(
    "a degenerate setView domain still reads a finite Omega",
    degOmega !== null && Number.isFinite(degOmega),
    JSON.stringify({ degOmega, tip: await ev(`(${liveCanvasSel}) ? ${F}.querySelector(".readout").textContent : null`) }),
  );

  // ---- devil's staircase: rho(A) live at a slider-chosen D ----
  // The second live figure on the page: a 1-D curve (not a heatmap) whose
  // tiles carry Omega = D with n_omega = 1 and K = A. The slider moves D,
  // the hash carries it, and the readout must equal the point kernel at the
  // same (D, A).
  check(
    "the staircase figure is marked live-capable",
    await ev(`!!${S} && ${S}.dataset.live === "devils_staircase"`),
    await ev(`${S} ? JSON.stringify(${S}.dataset) : "no figure"`),
  );
  await ev(`(${S}.querySelector(".act-interact").click(), true)`);
  const stFirstPaint = await waitFor(
    `!!${S}._live && ${S}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    50,
  );
  const stStats = await ev(`${S} && ${S}._live ? ${S}._live.stats() : null`);
  check(
    "the staircase's first tile is painted",
    stFirstPaint && stStats && stStats.painted >= 1,
    JSON.stringify(stStats),
  );
  const stRefined = await waitFor(
    `!!${S}._live && ${S}._live.stats().points > 64`,
    REFINE_MS,
    150,
  );
  check(
    "the staircase curve refines past the coarse tile",
    Boolean(stFirstPaint && stRefined),
    JSON.stringify(await ev(`${S} && ${S}._live ? ${S}._live.stats() : null`)),
  );
  check(
    "the staircase has a D slider and a number field",
    await ev(
      `!!(${S} && ${S}.querySelector(".live-params input[type=range]") && ${S}.querySelector(".live-params input[type=number]"))`,
    ),
  );
  check(
    "the staircase note says the Lyapunov panel is not live",
    await ev(
      `!!(${S} && Array.from(${S}.querySelectorAll(".hint")).some(n => /not live/i.test(n.textContent)))`,
    ),
  );

  // The readout at (D, A) must be the point kernel's value at that exact
  // pair — the same demand the flagship's wired check makes.
  await waitFor(`${S}._live && ${S}._live.pointReady && ${S}._live.pointReady()`, SAMPLE_MS);
  const stWired = await ev(`(async () => {
    const m = await import(new URL("live/point.js", document.baseURI).href);
    const direct = m.sample(0.25, 0.12, 5000, 50000, 0.1);
    return { ready: m.isReady(), direct, readout: ${S}._live.sample(0.12) };
  })()`);
  check(
    "the staircase readout is the point kernel at (D, A)",
    stWired != null && stWired.ready === true && stWired.direct !== null && stWired.readout === stWired.direct,
    JSON.stringify(stWired),
  );

  // Focus the live canvas and step the keyboard cursor: the tip must show a
  // finite A and a rho that matches the point kernel at that A.
  const stFocused = await ev(`(() => {
    const c = ${S} && ${S}.querySelector("canvas.plot.live");
    if (!c) return false;
    c.focus();
    return document.activeElement === c;
  })()`);
  if (stFocused) {
    await send("Input.dispatchKeyEvent", { type: "keyDown", key: "ArrowRight", code: "ArrowRight" });
    await send("Input.dispatchKeyEvent", { type: "keyUp", key: "ArrowRight", code: "ArrowRight" });
  }
  const stTip = stFocused
    ? await waitFor(
        `(() => {
          const tip = ${S} && ${S}.querySelector(".readout.on");
          if (!tip) return false;
          const lines = tip.textContent.split("\\n");
          return lines.length >= 2 && lines.every(l => /=\\s*\\S+\\s*$/.test(l));
        })()`,
        SAMPLE_MS,
      )
    : false;
  const stTipVals = stTip
    ? await ev(`(() => {
        const tip = ${S}.querySelector(".readout.on");
        const lines = tip.textContent.split("\\n");
        const num = l => { const m = l.match(/=\\s*(\\S+)\\s*$/); return m ? parseFloat(m[1]) : null; };
        return { a: num(lines[0]), rho: num(lines[1]) };
      })()`)
    : null;
  const stExpected =
    stTipVals && Number.isFinite(stTipVals.a)
      ? await ev(`(() => {
          const m = ${S}._live;
          return m ? m.sample(${stTipVals.a}) : null;
        })()`)
      : null;
  check(
    "the staircase tip reads the point value at the cursor's A",
    Boolean(
      stTipVals &&
        Number.isFinite(stTipVals.a) &&
        Number.isFinite(stTipVals.rho) &&
        stExpected !== null &&
        Math.abs(stTipVals.rho - stExpected) < 5e-3,
    ),
    JSON.stringify({ stTipVals, stExpected }),
  );

  // Move D: the debounced apply must bump the generation, repaint, and write
  // the new value into the hash.
  const stBefore = await ev(`${S} && ${S}._live ? ${S}._live.stats() : null`);
  const hashBefore = await ev("location.hash");
  await ev(`(() => {
    const r = ${S}.querySelector(".live-params input[type=range]");
    r.value = "0.4";
    r.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`);
  const stMoved = await waitFor(
    `!!${S}._live && ${S}._live.stats().generation > ${stBefore ? stBefore.generation : -1} && ${S}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  await waitFor(`location.hash.includes("D=0.4")`, 5_000);
  const hashAfter = await ev("location.hash");
  check(
    "moving D repaints a new generation",
    Boolean(stBefore && stMoved),
    JSON.stringify({ before: stBefore, after: await ev(`${S}._live ? ${S}._live.stats() : null`) }),
  );
  check(
    "moving D writes the parameter into the hash",
    hashAfter !== hashBefore && hashAfter.includes("D=0.4"),
    JSON.stringify({ hashBefore, hashAfter }),
  );
  const stNewD = await ev(`(async () => {
    const m = await import(new URL("live/point.js", document.baseURI).href);
    const direct = m.sample(0.4, 0.12, 5000, 50000, 0.1);
    return { direct, readout: ${S}._live.sample(0.12) };
  })()`);
  check(
    "after the move the readout samples at the new D",
    stNewD != null && stNewD.direct !== null && stNewD.readout === stNewD.direct,
    JSON.stringify(stNewD),
  );

  // ---- delayed-logistic attractors: the (x, y) cloud live at a slider-chosen D ----
  // The third live figure: a point cloud (not a curve or a heatmap) drawn by
  // one direct kernel call per parameter set. The slider moves D, the n
  // field sets the plotted-state count, the hash carries both, and one
  // plotted state must equal the kernel's own output at the same request.
  const A = "document.getElementById('fig:delayed_logistic_attractors')";
  const atStats = () => ev(`${A} && ${A}._live ? ${A}._live.stats() : null`);
  check(
    "the attractors figure is marked live-capable",
    await ev(`!!${A} && ${A}.dataset.live === "delayed_logistic_attractors"`),
    await ev(`${A} ? JSON.stringify(${A}.dataset) : "no figure"`),
  );
  await ev(`(${A}.querySelector(".act-interact").click(), true)`);
  const atFirstPaint = await waitFor(
    `!!${A}._live && ${A}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    50,
  );
  const atStats0 = await atStats();
  check(
    "the attractor cloud is painted at the default D",
    atFirstPaint && atStats0 && atStats0.points === 2048,
    JSON.stringify(atStats0),
  );
  check(
    "the attractors figure has a D slider and an n field",
    await ev(
      `!!(${A} && ${A}.querySelector(".live-params input[type=range]") && ${A}.querySelector(".live-params input[type=number]"))`,
    ),
  );

  // One plotted state against the kernel computed the same way: the page's
  // own glue module answers delayed_logistic_attractor_tile at the figure's
  // current (D, n), and the trace's first point must be that tile's first
  // (x, y) pair — x first, then y.
  const atCheck = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const p = ${A}._live.params();
    const fp = (Math.sqrt(1 + 4 * p.D) - 1) / (2 * p.D);
    const tile = glue.delayed_logistic_attractor_tile(0.3, p.D, p.D, 1, 20000, p.n, [fp + 0.01, fp - 0.01]);
    const tr = ${A}._live.trace();
    return { x: tr.x[0], y: tr.y[0], kx: tile[4], ky: tile[5], n: tr.x.length, nPlot: tile[1] };
  })()`);
  check(
    "a plotted state equals the kernel's at the same (A, D, n)",
    atCheck != null &&
      atCheck.x === atCheck.kx &&
      atCheck.y === atCheck.ky &&
      atCheck.n === atCheck.nPlot,
    JSON.stringify(atCheck),
  );

  // Move D: the debounced apply must recompute, repaint, and write the new
  // value into the hash. The range input snaps to its step, so the check
  // reads back the snapped value rather than assuming it.
  const atBefore = await atStats();
  const atSet = await ev(`(() => {
    const r = ${A}.querySelector(".live-params input[type=range]");
    r.value = "2.09";
    r.dispatchEvent(new Event("input", { bubbles: true }));
    return r.value;
  })()`);
  const atMoved = await waitFor(
    `!!${A}._live && ${A}._live.stats().generation > ${atBefore ? atBefore.generation : -1} && ${A}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  await waitFor(`location.hash.includes("D=" + ${JSON.stringify(atSet)})`, 5_000);
  const atHash = await ev("location.hash");
  check(
    "moving the attractor D repaints a new generation",
    Boolean(atBefore && atMoved),
    JSON.stringify({ before: atBefore, after: await atStats() }),
  );
  check(
    "moving the attractor D writes the parameter into the hash",
    atHash.includes(`fig:delayed_logistic_attractors.D=${atSet}`),
    JSON.stringify({ atSet, atHash }),
  );
  const atMovedCheck = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const p = ${A}._live.params();
    const fp = (Math.sqrt(1 + 4 * p.D) - 1) / (2 * p.D);
    const tile = glue.delayed_logistic_attractor_tile(0.3, p.D, p.D, 1, 20000, p.n, [fp + 0.01, fp - 0.01]);
    const tr = ${A}._live.trace();
    return { D: p.D, x: tr.x[0], kx: tile[4] };
  })()`);
  check(
    "after the move the cloud is the new D's orbit",
    atMovedCheck != null &&
      atMovedCheck.D === Number(atSet) &&
      atMovedCheck.x === atMovedCheck.kx,
    JSON.stringify(atMovedCheck),
  );

  // The iteration-count control: setting n to 512 must repaint a cloud of
  // exactly 512 states — the density-for-response-time trade the control
  // exists for.
  await ev(`(() => {
    const nums = ${A}.querySelectorAll(".live-params input[type=number]");
    const n = nums[1];
    n.value = "512";
    n.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  const atN = await waitFor(
    `!!${A}._live && ${A}._live.stats().points === 512`,
    FIRST_PAINT_MS,
    100,
  );

  // The slider's top end is capped below the measured escape boundary, so
  // even its maximum paints a bounded, finite cloud. A range input snaps
  // out-of-range sets to the max, so value = max + 1 exercises the top
  // without depending on the number.
  const atTopBefore = await atStats();
  const atTopSet = await ev(`(() => {
    const r = ${A}.querySelector(".live-params input[type=range]");
    r.value = String(Number(r.max) + 1);
    r.dispatchEvent(new Event("input", { bubbles: true }));
    return r.value;
  })()`);
  const atTopPaint = await waitFor(
    `!!${A}._live && ${A}._live.stats().generation > ${atTopBefore ? atTopBefore.generation : -1} && ${A}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  const atTop = await ev(`(() => {
    const r = ${A}.querySelector(".live-params input[type=range]");
    const tr = ${A}._live.trace();
    return { D: ${A}._live.params().D, max: Number(r.max), painted: ${A}._live.stats().painted, finite: tr.x.length > 0 && tr.x.every(Number.isFinite) };
  })()`);
  check(
    "at the slider's top the orbit is still bounded and finite",
    Boolean(
      atTopSet === String(atTop ? atTop.max : "") &&
        atTopPaint &&
        atTop &&
        atTop.D === atTop.max &&
        atTop.painted >= 1 &&
        atTop.finite,
    ),
    JSON.stringify({ atTopSet, atTop }),
  );

  // ---- torus-doubling attractors: the (X, Y) cloud with a map selector ----
  // The fourth live figure: one canvas serving map (I) and map (IV). The
  // select switches the map — and with it A, x0, the D slider's window and
  // the title — the D slider moves inside the chosen map's doubling window,
  // the n field sets the plotted-state count, the hash carries all three,
  // and one plotted state must equal the kernel's own output at the same
  // request.
  const T = "document.getElementById('fig:map_I_attractors')";
  const tStats = () => ev(`${T} && ${T}._live ? ${T}._live.stats() : null`);
  check(
    "the torus figure is marked live-capable",
    await ev(`!!${T} && ${T}.dataset.live === "torus_doubling_attractors"`),
    await ev(`${T} ? JSON.stringify(${T}.dataset) : "no figure"`),
  );
  await ev(`(${T}.querySelector(".act-interact").click(), true)`);
  const tFirstPaint = await waitFor(
    `!!${T}._live && ${T}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    50,
  );
  const tStats0 = await tStats();
  check(
    "the torus cloud is painted at the default map and D",
    tFirstPaint && tStats0 && tStats0.points === 2048,
    JSON.stringify(tStats0),
  );
  check(
    "the torus figure has a map selector, a D slider and an n field",
    await ev(
      `!!(${T} && ${T}.querySelector(".live-params select") && ${T}.querySelector(".live-params input[type=range]") && ${T}.querySelector(".live-params input[type=number]"))`,
    ),
  );

  // One plotted state against the kernel computed the same way: the page's
  // own glue module answers torus_doubling_attractor_tile at the figure's
  // current (map, D, n), and the trace's first point must be that tile's
  // first (X, Y) pair — components 0 and 1 of the first state.
  const tCheck = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const p = ${T}._live.params();
    const x0 = p.map === 4 ? [0.5, 0.45, 0.52, 0.48] : [0.5, 0.5, 0.5];
    const a = p.map === 4 ? 0.3 : 0.4;
    const tile = glue.torus_doubling_attractor_tile(p.map, a, p.D, 20000, p.n, x0);
    const tr = ${T}._live.trace();
    return { x: tr.x[0], y: tr.y[0], kx: tile[4], ky: tile[5], n: tr.x.length, nProduced: tile[3] };
  })()`);
  check(
    "a plotted torus state equals the kernel's at the same (map, A, D, n)",
    tCheck != null &&
      tCheck.x === tCheck.kx &&
      tCheck.y === tCheck.ky &&
      tCheck.n === tCheck.nProduced,
    JSON.stringify(tCheck),
  );

  // Switch to map IV: the selector must move D into the new map's window
  // (its published default 1.5206), repaint a new generation, and write the
  // map into the hash.
  const tBefore = await tStats();
  await ev(`(() => {
    const s = ${T}.querySelector(".live-params select");
    s.value = "4";
    s.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  const tSwitched = await waitFor(
    `!!${T}._live && ${T}._live.stats().generation > ${tBefore ? tBefore.generation : -1} && ${T}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  await waitFor(`location.hash.includes("map=4")`, 5_000);
  const tParams4 = await ev(`${T} && ${T}._live ? ${T}._live.params() : null`);
  check(
    "switching to map IV repaints a new generation",
    Boolean(tBefore && tSwitched),
    JSON.stringify({ before: tBefore, after: await tStats() }),
  );
  check(
    "switching to map IV moves D into the new window and writes the hash",
    tParams4 != null &&
      tParams4.map === 4 &&
      tParams4.D === 1.5206 &&
      (await ev("location.hash")).includes("fig:map_I_attractors.map=4"),
    JSON.stringify({ tParams4, hash: await ev("location.hash") }),
  );

  // Move D inside map IV's window: the debounced apply must recompute,
  // repaint, and write the new value into the hash. The range input snaps
  // to its step, so the check reads back the snapped value rather than
  // assuming it.
  const tBeforeD = await tStats();
  const tSet = await ev(`(() => {
    const r = ${T}.querySelector(".live-params input[type=range]");
    r.value = "1.515";
    r.dispatchEvent(new Event("input", { bubbles: true }));
    return r.value;
  })()`);
  const tMoved = await waitFor(
    `!!${T}._live && ${T}._live.stats().generation > ${tBeforeD ? tBeforeD.generation : -1} && ${T}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  await waitFor(`location.hash.includes("D=" + ${JSON.stringify(tSet)})`, 5_000);
  const tMovedCheck = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const p = ${T}._live.params();
    const tile = glue.torus_doubling_attractor_tile(4, 0.3, p.D, 20000, p.n, [0.5, 0.45, 0.52, 0.48]);
    const tr = ${T}._live.trace();
    return { D: p.D, x: tr.x[0], kx: tile[4] };
  })()`);
  check(
    "moving the torus D repaints a new generation",
    Boolean(tBeforeD && tMoved),
    JSON.stringify({ before: tBeforeD, after: await tStats() }),
  );
  check(
    "after the move the torus cloud is the new D's orbit",
    tMovedCheck != null &&
      tMovedCheck.D === Number(tSet) &&
      tMovedCheck.x === tMovedCheck.kx,
    JSON.stringify(tMovedCheck),
  );
  check(
    "the n control sets the plotted-state count",
    Boolean(atN),
    JSON.stringify(await atStats()),
  );

  // ---- double devil's staircase: rho_theta and rho_phi over a D window ----
  // The fifth live figure: two curves (rho_theta slate, rho_phi vermilion)
  // plus the dashed rho_theta = D reference, drawn by one
  // modulated_circle_rotation_tile call per parameter set. The eps slider
  // moves the forcing, the dMin/dMax fields and the two preset buttons move
  // the compute window, the hash carries all four, and one plotted
  // rho_theta must equal the kernel's own output at the same D.
  const M = "document.getElementById('fig:double_staircase')";
  const mStats = () => ev(`${M} && ${M}._live ? ${M}._live.stats() : null`);
  check(
    "the double staircase is marked live-capable",
    await ev(`!!${M} && ${M}.dataset.live === "double_staircase"`),
    await ev(`${M} ? JSON.stringify(${M}.dataset) : "no figure"`),
  );
  await ev(`(${M}.querySelector(".act-interact").click(), true)`);
  const mFirstPaint = await waitFor(
    `!!${M}._live && ${M}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    50,
  );
  const mStats0 = await mStats();
  check(
    "the double staircase is painted at the defaults",
    mFirstPaint && mStats0 && mStats0.points === 256,
    JSON.stringify(mStats0),
  );
  check(
    "the double staircase has an eps slider, window fields and presets",
    await ev(
      `!!(${M} && ${M}.querySelector(".live-params input[type=range]") && ${M}.querySelectorAll(".live-params input[type=number]").length >= 3 && ${M}.querySelectorAll(".live-presets button").length === 3)`,
    ),
  );

  // One plotted rho_theta against the kernel computed the same way: the
  // page's own glue module answers modulated_circle_rotation_tile at the
  // figure's current (eps, window, n), and the trace's first point must be
  // that tile's first pair — rho_theta first, then rho_phi.
  const mCheck = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const p = ${M}._live.params();
    const tile = glue.modulated_circle_rotation_tile(0.1, 0.6180339887498949, p.dMin, p.dMax, p.n, p.eps, 3000, 20000, 0.1, 0.1);
    const tr = ${M}._live.trace();
    return { x: tr.x[0], y: tr.y[0], y2: tr.y2[0], kx: tile[4], ky2: tile[5], n: tr.x.length, nD: tile[0] };
  })()`);
  check(
    "a plotted rho_theta equals the kernel's at the same D",
    mCheck != null &&
      mCheck.y === mCheck.kx &&
      mCheck.y2 === mCheck.ky2 &&
      mCheck.n === mCheck.nD,
    JSON.stringify(mCheck),
  );

  // Move eps: the debounced apply must recompute, repaint, and write the
  // new value into the hash. The range input snaps to its step, so the
  // check reads back the snapped value rather than assuming it.
  const mBefore = await mStats();
  const mSet = await ev(`(() => {
    const r = ${M}.querySelector(".live-params input[type=range]");
    r.value = "0.12";
    r.dispatchEvent(new Event("input", { bubbles: true }));
    return r.value;
  })()`);
  const mMoved = await waitFor(
    `!!${M}._live && ${M}._live.stats().generation > ${mBefore ? mBefore.generation : -1} && ${M}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  await waitFor(`location.hash.includes("eps=" + ${JSON.stringify(mSet)})`, 5_000);
  const mHash = await ev("location.hash");
  check(
    "moving eps repaints a new generation",
    Boolean(mBefore && mMoved),
    JSON.stringify({ before: mBefore, after: await mStats() }),
  );
  check(
    "moving eps writes the parameter into the hash",
    mHash.includes(`fig:double_staircase.eps=${mSet}`),
    JSON.stringify({ mSet, mHash }),
  );

  // Apply the first preset: the window must move into the 1/4 plateau's D
  // range, repaint a new generation, and write the window into the hash.
  const mBeforeP = await mStats();
  await ev(`(() => {
    const b = ${M}.querySelector(".live-presets button");
    b.click();
    return true;
  })()`);
  const mPreset = await waitFor(
    `!!${M}._live && ${M}._live.stats().generation > ${mBeforeP ? mBeforeP.generation : -1} && ${M}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  await waitFor(`location.hash.includes("dMin=0.25502630263026305")`, 5_000);
  const mParamsP = await ev(`${M} && ${M}._live ? ${M}._live.params() : null`);
  check(
    "applying a preset repaints a new generation",
    Boolean(mBeforeP && mPreset),
    JSON.stringify({ before: mBeforeP, after: await mStats() }),
  );
  check(
    "applying a preset moves the D window and writes the hash",
    mParamsP != null &&
      mParamsP.dMin === 0.25502630263026305 &&
      mParamsP.dMax === 0.27372657265726574 &&
      (await ev("location.hash")).includes("fig:double_staircase.dMin=0.25502630263026305"),
    JSON.stringify({ mParamsP, hash: await ev("location.hash") }),
  );

  const mPresetCheck = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const p = ${M}._live.params();
    const tile = glue.modulated_circle_rotation_tile(0.1, 0.6180339887498949, p.dMin, p.dMax, p.n, p.eps, 3000, 20000, 0.1, 0.1);
    const tr = ${M}._live.trace();
    return { dMin: p.dMin, x: tr.x[0], y: tr.y[0], kx: tile[4] };
  })()`);
  check(
    "after the preset the curve is the new window's sweep",
    mPresetCheck != null &&
      mPresetCheck.dMin === 0.25502630263026305 &&
      mPresetCheck.x === mPresetCheck.dMin &&
      mPresetCheck.y === mPresetCheck.kx,
    JSON.stringify(mPresetCheck),
  );

  // ---- CML space-time diagrams: the field x_i^n with a model selector ----
  // The sixth live figure: one heat map (site i across, time n up, magma)
  // drawn by one cml_spacetime_tile call per parameter set. The select
  // switches the model — and with it the eps slider's window and default —
  // the sites field sets the lattice width, play appends chunks computed
  // from the window's last row, and one plotted value must equal the
  // kernel's own output at the same request.
  const C = "document.getElementById('fig:spacetime_diagrams')";
  const cStats = () => ev(`${C} && ${C}._live ? ${C}._live.stats() : null`);
  check(
    "the spacetime figure is marked live-capable",
    await ev(`!!${C} && ${C}.dataset.live === "spacetime_diagrams"`),
    await ev(`${C} ? JSON.stringify(${C}.dataset) : "no figure"`),
  );
  await ev(`(${C}.querySelector(".act-interact").click(), true)`);
  const cFirstPaint = await waitFor(
    `!!${C}._live && ${C}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    50,
  );
  const cStats0 = await cStats();
  check(
    "the spacetime field is painted at the defaults",
    cFirstPaint && cStats0 && cStats0.rows === 500 && cStats0.sites === 200,
    JSON.stringify(cStats0),
  );
  check(
    "the spacetime figure has a model selector, an eps slider and a sites field",
    await ev(
      `!!(${C} && ${C}.querySelector(".live-params select") && ${C}.querySelector(".live-params input[type=range]") && ${C}.querySelector(".live-params input[type=number]"))`,
    ),
  );
  check(
    "the spacetime figure has a play button",
    await ev(`!!(${C} && ${C}.querySelector(".live-play"))`),
  );

  // One plotted value against the kernel computed the same way: the page's
  // own glue module answers cml_spacetime_tile at the figure's current
  // (model, eps, sites) from the same x0 JSON, and the field's first value
  // must be that tile's first field entry.
  const cCheck = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const x0 = await (await fetch(new URL("live/cml-x0.json", document.baseURI).href)).json();
    const p = ${C}._live.params();
    const tile = glue.cml_spacetime_tile(p.model, p.eps, p.sites, 2000, 500, Float64Array.from(x0.slice(0, p.sites)));
    const f = ${C}._live.field();
    return { v: f.values[0], kv: tile[4], rows: f.nRows, nRecord: tile[3] };
  })()`);
  check(
    "a plotted field value equals the kernel's at the same (model, eps, sites)",
    cCheck != null &&
      cCheck.v === cCheck.kv &&
      cCheck.rows === cCheck.nRecord,
    JSON.stringify(cCheck),
  );

  // Switch to model C: the selector must move eps into the new model's
  // window (its published default 0.2), repaint a new generation, and
  // write the model into the hash.
  const cBefore = await cStats();
  await ev(`(() => {
    const s = ${C}.querySelector(".live-params select");
    s.value = "2";
    s.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  const cSwitched = await waitFor(
    `!!${C}._live && ${C}._live.stats().generation > ${cBefore ? cBefore.generation : -1} && ${C}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  await waitFor(`location.hash.includes("model=2")`, 5_000);
  const cParams2 = await ev(`${C} && ${C}._live ? ${C}._live.params() : null`);
  check(
    "switching to model C repaints a new generation",
    Boolean(cBefore && cSwitched),
    JSON.stringify({ before: cBefore, after: await cStats() }),
  );
  check(
    "switching to model C moves eps into the new window and writes the hash",
    cParams2 != null &&
      cParams2.model === 2 &&
      cParams2.eps === 0.2 &&
      (await ev("location.hash")).includes("fig:spacetime_diagrams.model=2"),
    JSON.stringify({ cParams2, hash: await ev("location.hash") }),
  );

  // Move eps inside model C's window: the debounced apply must recompute,
  // repaint, and write the new value into the hash. The range input snaps
  // to its step, so the check reads back the snapped value rather than
  // assuming it.
  const cBeforeE = await cStats();
  const cSet = await ev(`(() => {
    const r = ${C}.querySelector(".live-params input[type=range]");
    r.value = "0.3";
    r.dispatchEvent(new Event("input", { bubbles: true }));
    return r.value;
  })()`);
  const cMoved = await waitFor(
    `!!${C}._live && ${C}._live.stats().generation > ${cBeforeE ? cBeforeE.generation : -1} && ${C}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  await waitFor(`location.hash.includes("eps=" + ${JSON.stringify(cSet)})`, 5_000);
  const cMovedCheck = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const x0 = await (await fetch(new URL("live/cml-x0.json", document.baseURI).href)).json();
    const p = ${C}._live.params();
    const tile = glue.cml_spacetime_tile(p.model, p.eps, p.sites, 2000, 500, Float64Array.from(x0.slice(0, p.sites)));
    const f = ${C}._live.field();
    return { eps: p.eps, v: f.values[0], kv: tile[4] };
  })()`);
  check(
    "moving the spacetime eps repaints a new generation",
    Boolean(cBeforeE && cMoved),
    JSON.stringify({ before: cBeforeE, after: await cStats() }),
  );
  check(
    "after the move the field is the new eps's run",
    cMovedCheck != null &&
      cMovedCheck.eps === Number(cSet) &&
      cMovedCheck.v === cMovedCheck.kv,
    JSON.stringify(cMovedCheck),
  );

  // Play: chunks append rows continued from the window's last row and the
  // oldest rows drop off. Several chunks may land before the check reads
  // the field, so the comparison anchors on the field itself: its last
  // 250 rows must equal the kernel's continuation of the row just before
  // them — the same call the figure's transport made.
  const cGenBefore = (await cStats()).generation;
  await ev(`(${C}._live.play(), true)`);
  const cPlayed = await waitFor(
    `!!${C}._live && ${C}._live.stats().generation > ${cGenBefore}`,
    SAMPLE_MS,
    100,
  );
  await ev(`(${C}._live.pause(), true)`);
  const cCont = await ev(`(async () => {
    const glue = await import(new URL("wasm/dynachaos_wasm.js", document.baseURI).href);
    const f = ${C}._live.field();
    const n = f.nSites;
    const prior = f.values.slice((f.nRows - 251) * n, (f.nRows - 250) * n);
    const p = ${C}._live.params();
    const tile = glue.cml_spacetime_tile(p.model, p.eps, p.sites, 0, 250, Float64Array.from(prior));
    const tail = f.values.slice((f.nRows - 250) * n);
    let same = tail.length === 250 * n;
    for (let k = 0; k < tail.length && same; k++) same = tail[k] === tile[4 + k];
    return { same, rows: f.nRows, sites: f.nSites };
  })()`);
  check(
    "play appends chunks continued from the window's last row",
    Boolean(cPlayed && cCont != null && cCont.same && cCont.rows === 500),
    JSON.stringify(cCont),
  );
  // Pause: the transport stops, so the generation freezes.
  const cGen = (await cStats()).generation;
  await sleep(600);
  const cGen2 = (await cStats()).generation;
  check(
    "pause stops the transport",
    cGen === cGen2 && !(await ev(`${C}._live.playing()`)),
    JSON.stringify({ cGen, cGen2 }),
  );

  const sEnd = await stats();
  check(
    "every worker is still alive",
    Boolean(sEnd && sEnd.liveWorkers >= 1 && sEnd.liveWorkers === sEnd.workerCount),
    JSON.stringify(sEnd),
  );
  await pullDebug();

  // A shared link that still carries the old slider's D range: the hash
  // parse clamps 3 to the max, the figure mounts live on the link, and the
  // reader sees the bounded orbit there, its value in the controls and the
  // title (read through textContent), not a blank plot.
  await send("Page.navigate", {
    url: `http://127.0.0.1:${httpPort}/index.html#fig:delayed_logistic_attractors.D=3`,
  });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  const atClamp = await waitFor(
    `!!${A} && !!${A}._live && ${A}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    100,
  );
  const atClampState = await ev(`(() => {
    const fig = ${A};
    if (!fig || !fig._live) return null;
    const r = fig.querySelector(".live-params input[type=range]");
    const num = fig.querySelector(".live-params input[type=number]");
    const h = fig.querySelector(".plot-title");
    const tr = fig._live.trace();
    return {
      D: fig._live.params().D,
      max: Number(r.max),
      rangeValue: Number(r.value),
      numValue: Number(num.value),
      title: h ? h.textContent : "",
      painted: fig._live.stats().painted,
      finite: tr.x.length > 0 && tr.x.every(Number.isFinite),
    };
  })()`);
  check(
    "a shared link with D=3 clamps to the slider max and paints its orbit",
    Boolean(
      atClamp &&
        atClampState &&
        atClampState.D === atClampState.max &&
        atClampState.rangeValue === atClampState.max &&
        atClampState.numValue === atClampState.max &&
        atClampState.title.includes("D = " + atClampState.max) &&
        atClampState.painted >= 1 &&
        atClampState.finite,
    ),
    JSON.stringify(atClampState),
  );
  // Page.navigate with only a fragment change stays in the document
  // (Chrome's same-document rule), so after this check the page URL still
  // carries no query — the guardrails' own hash navigation below would not
  // reload either. Land back on ?debug=1 so the next navigation differs
  // in the query and reloads for real.
  await send("Page.navigate", { url: pageUrl });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);

  // Guardrails. Each check navigates on its own so a hidden tab, a narrow
  // viewport, or CPU throttling cannot leak into the figure checks above
  // or the reduced-data checks below.
  await send("Emulation.setDeviceMetricsOverride", {
    width: 400,
    height: 800,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await send("Page.navigate", {
    url: `http://127.0.0.1:${httpPort}/index.html#fig:devils_staircase.D=0.4`,
  });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await waitFor(`!!${S}`, FIGURE_MS);
  const held = await waitFor(
    `${S} && ${S}.dataset.liveHold === "narrow" && ${S}.dataset.state !== "live"`,
    NARROW_WAIT_MS,
    50,
  );
  const narrowShown = await ev(`(() => {
    const img = ${S} && ${S}.querySelector(".fig-body img");
    if (!img) return false;
    const s = getComputedStyle(img);
    return s.display !== "none" && s.visibility !== "hidden";
  })()`);
  const narrowState = await ev(`(() => {
    const fig = ${S};
    if (!fig) return null;
    return { state: fig.dataset.state, hold: fig.dataset.liveHold || "", width: innerWidth, live: !!fig._live };
  })()`);
  check(
    "a narrow viewport keeps the PNG and does not go live by itself",
    Boolean(held && narrowShown && narrowState && !narrowState.live && narrowState.state !== "live"),
    JSON.stringify({ ...narrowState, held: Boolean(held), imgShown: narrowShown }),
  );
  await ev(`(${S}.querySelector(".act-interact").click(), true)`);
  const narrowAsked = await waitFor(
    `!!${S}._live && ${S}.dataset.state === "live"`,
    FIRST_PAINT_MS,
    100,
  );
  check(
    "a narrow viewport still goes live when the reader asks",
    Boolean(narrowAsked),
    await ev(`${S} ? JSON.stringify(${S}.dataset) : "missing"`),
  );
  const narrowD = await ev(
    `${S} && ${S}.querySelector(".live-params input[type=number]") ? Number(${S}.querySelector(".live-params input[type=number]").value) : null`,
  );
  check(
    "a narrow viewport applies the shared parameter when the reader asks",
    narrowAsked && narrowD === 0.4,
    JSON.stringify({ narrowD }),
  );
  await send("Emulation.clearDeviceMetricsOverride");

  const debugLines = async () => {
    await pullDebug();
    return debugLogs.filter((line) => line.includes("[dynachaos-live]"));
  };
  debugLogs.length = 0;
  await send("Page.navigate", {
    url: `http://127.0.0.1:${httpPort}/index.html#live-debug`,
  });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await waitFor(`!!${F}`, FIGURE_MS);
  await installLogHook();
  await ev(`(${F}.querySelector(".act-interact").click(), true)`);
  const hashPaint = await waitFor(
    `!!${F}._live && ${F}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    50,
  );
  const hashLogs = await debugLines();
  check(
    "the #live-debug hash turns telemetry on",
    Boolean(hashPaint && hashLogs.some((line) => line.includes('"event":"paint"') || line.includes('"event":"issue"'))),
    JSON.stringify({ hash: await ev("location.hash"), sample: hashLogs.slice(0, 2) }),
  );

  debugLogs.length = 0;
  await send("Page.navigate", { url: `http://127.0.0.1:${httpPort}/index.html` });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await waitFor(`!!${F}`, FIGURE_MS);
  await installLogHook();
  await ev(`(${F}.querySelector(".act-interact").click(), true)`);
  const quietPaint = await waitFor(
    `!!${F}._live && ${F}._live.stats().painted >= 1`,
    FIRST_PAINT_MS,
    50,
  );
  const quietLogs = await debugLines();
  check(
    "telemetry stays off without #live-debug, ?debug=1, or localStorage",
    Boolean(quietPaint && quietLogs.length === 0),
    JSON.stringify({ hash: await ev("location.hash"), search: await ev("location.search"), sample: quietLogs.slice(0, 2) }),
  );
  const quietBox = await ev(`!!${F}.querySelector(".live-debug-report")`);
  check(
    "without debug mode no debug report box exists in the DOM",
    !quietBox,
    JSON.stringify({ quietBox }),
  );

  // ---- staircase y-fit and the Greek plot titles ----
  // A fresh navigation, since the flagship zooms above put ?fig= in
  // location.search. On this load the only figure that mounts is the
  // staircase, so the automatic y-fit is the only writer a clean URL can
  // come from.
  await send("Page.navigate", { url: pageUrl });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await waitFor(`!!${S}`, FIGURE_MS);
  await ev(`(${S}.querySelector(".act-interact").click(), true)`);
  const gkPaint = await waitFor(`!!${S}._live && ${S}._live.stats().painted >= 1`, FIRST_PAINT_MS, 50);
  check(
    "the staircase title keeps its Greek letter and is not CSS-uppercased",
    await ev(`(() => {
      const h = ${S} && ${S}.querySelector(".plot-title");
      if (!h) return false;
      return h.textContent.includes("\\u03c1") && getComputedStyle(h).textTransform !== "uppercase";
    })()`),
    await ev(`(() => {
      const h = ${S} && ${S}.querySelector(".plot-title");
      return h ? JSON.stringify({ text: h.textContent, transform: getComputedStyle(h).textTransform }) : "no title";
    })()`),
  );
  const gkRefined = await waitFor(`!!${S}._live && ${S}._live.stats().points > 64`, REFINE_MS, 150);
  const gkDom = await ev(`(() => {
    const p = ${S} && ${S}._plots && ${S}._plots[0];
    const tr = ${S} && ${S}._live && ${S}._live.trace();
    if (!p || !tr || !tr.y.length) return null;
    const d = p.getDomain();
    let lo = Infinity, hi = -Infinity;
    for (const y of tr.y) { if (y < lo) lo = y; if (y > hi) hi = y; }
    return { y0: d.y0, y1: d.y1, lo, hi, modified: p.isModified(), search: location.search, badge: !!(${S}.querySelector(".fig-badge:not([hidden])")) };
  })()`);
  check(
    "after refinement the staircase domain covers the whole trace",
    Boolean(gkPaint && gkRefined && gkDom && gkDom.y0 <= gkDom.lo && gkDom.y1 >= gkDom.hi),
    JSON.stringify({ gkDom, stats: await ev(`${S} && ${S}._live ? ${S}._live.stats() : null`) }),
  );
  check(
    "the automatic y-fit leaves the staircase unmodified and the URL clean",
    Boolean(
      gkDom &&
        gkDom.modified === false &&
        gkDom.badge === false &&
        !new URLSearchParams(gkDom.search).has("fig"),
    ),
    JSON.stringify(gkDom),
  );

  // ---- phone layout (390x844) and responsive controls ----
  const measureHeaderButtons = async () =>
    ev(`(() => {
      const fig = ${F};
      if (!fig) return null;
      fig.scrollIntoView({ behavior: "instant" });
      const buttons = Array.from(fig.querySelectorAll(".fig-head button, .fig-head a"))
        .filter((b) => !b.hidden && getComputedStyle(b).display !== "none");
      const hits = buttons.map((b) => {
        const r = b.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const el = document.elementFromPoint(cx, cy);
        const underControls = Boolean(el && el.closest(".controls"));
        return { text: b.textContent.trim(), hitIsButton: el === b, underControls, elTag: el ? el.tagName : null };
      });
      return {
        allButtons: hits.length > 0 && hits.every((h) => h.hitIsButton && !h.underControls),
        hits
      };
    })()`);

  const measureResetVsTitle = async () =>
    ev(`(() => {
      const fig = ${F};
      const reset = fig && fig.querySelector(".plot-reset");
      const title = fig && fig.querySelector(".plot-title");
      if (!reset || !title || reset.hidden) return { error: "not visible" };
      const rReset = reset.getBoundingClientRect();
      const rTitle = title.getBoundingClientRect();
      const intersects = !(
        rReset.right <= rTitle.left ||
        rReset.left >= rTitle.right ||
        rReset.bottom <= rTitle.top ||
        rReset.top >= rTitle.bottom
      );
      return {
        resetVisible: !reset.hidden,
        intersects,
        resetRect: { left: rReset.left, right: rReset.right, top: rReset.top, bottom: rReset.bottom },
        titleRect: { left: rTitle.left, right: rTitle.right, top: rTitle.top, bottom: rTitle.bottom }
      };
    })()`);

  await send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await send("Page.navigate", { url: `http://127.0.0.1:${httpPort}/index.html?debug=1&layout=phone` });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await waitFor(`!!${F}`, FIGURE_MS);

  const phoneControls = await ev(`(() => {
    const de = document.documentElement;
    const scrollOk = de.scrollWidth <= de.clientWidth && window.innerWidth === 390;
    const c = document.querySelector(".controls");
    if (!c) return { missing: true };
    const buttons = Array.from(c.querySelectorAll("button")).map((b) => {
      const r = b.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      const el = document.elementFromPoint(cx, cy);
      return {
        text: b.textContent.trim(),
        inside: r.left >= 0 && r.right <= window.innerWidth && r.top >= 0 && r.bottom <= window.innerHeight,
        hit: el === b,
        h44: r.height >= 44,
        rect: { left: r.left, right: r.right, top: r.top, bottom: r.bottom, height: r.height }
      };
    });
    return {
      scrollOk,
      allInside: buttons.every((b) => b.inside),
      allHit: buttons.every((b) => b.hit),
      allH44: buttons.every((b) => b.h44),
      buttons
    };
  })()`);
  check(
    "at 390x844 the page does not scroll sideways and every reading control is reachable",
    Boolean(phoneControls && phoneControls.scrollOk && phoneControls.allInside && phoneControls.allHit),
    JSON.stringify(phoneControls),
  );
  check(
    "at 390x844 touch targets in .controls stay at least 44 px high",
    Boolean(phoneControls && phoneControls.allH44),
    JSON.stringify(phoneControls ? phoneControls.buttons.map((b) => ({ text: b.text, h: b.rect.height, ok: b.h44 })) : "missing"),
  );

  const phoneCitationWs = await ev(`getComputedStyle(document.querySelector(".citation")).whiteSpace`);
  check(
    "at 390x844 citations wrap (whiteSpace is normal)",
    phoneCitationWs === "normal",
    `whiteSpace=${phoneCitationWs}`,
  );

  // Figure scrollIntoView at 390x844
  const figScroll390 = await measureHeaderButtons();
  check(
    "after scrollIntoView at 390x844 every figure action button is clickable and not under .controls",
    Boolean(figScroll390 && figScroll390.allButtons),
    JSON.stringify(figScroll390),
  );

  // Live figure at 390x844: mount, check "image" action button, and zoom for reset view
  await ev(`(${F}.querySelector(".act-interact").click(), true)`);
  await waitFor(`!!${F}._live`, FIRST_PAINT_MS, 100);

  const figScroll390Live = await measureHeaderButtons();
  check(
    "after scrollIntoView at 390x844 live figure action buttons (including image) are not under .controls",
    Boolean(figScroll390Live && figScroll390Live.allButtons),
    JSON.stringify(figScroll390Live),
  );

  await ev(`(${F}._live.setView({ omegaMin: 0, omegaMax: 0.3, kMin: 0, kMax: 0.15 }), true)`);
  await waitFor(`!${F}.querySelector(".plot-reset").hidden`, 5000);

  const plotReset390 = await measureResetVsTitle();
  check(
    "with the view zoomed at 390x844 reset view and the plot title do not intersect",
    Boolean(plotReset390 && plotReset390.resetVisible && !plotReset390.intersects),
    JSON.stringify(plotReset390),
  );

  // ---- desktop layout (1440x900) ----
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await send("Page.navigate", { url: `http://127.0.0.1:${httpPort}/index.html?debug=1&layout=desktop` });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await waitFor(`!!${F}`, FIGURE_MS);

  const desktopCitationWs = await ev(`getComputedStyle(document.querySelector(".citation")).whiteSpace`);
  check(
    "at 1440x900 citations do not wrap (whiteSpace is nowrap)",
    desktopCitationWs === "nowrap",
    `whiteSpace=${desktopCitationWs}`,
  );

  const figScroll1440 = await measureHeaderButtons();
  check(
    "after scrollIntoView at 1440x900 every figure action button is clickable and not under .controls",
    Boolean(figScroll1440 && figScroll1440.allButtons),
    JSON.stringify(figScroll1440),
  );

  await ev(`(${F}.querySelector(".act-interact").click(), true)`);
  await waitFor(`!!${F}._live`, FIRST_PAINT_MS, 100);

  const figScroll1440Live = await measureHeaderButtons();
  check(
    "after scrollIntoView at 1440x900 live figure action buttons (including image) are not under .controls",
    Boolean(figScroll1440Live && figScroll1440Live.allButtons),
    JSON.stringify(figScroll1440Live),
  );

  await ev(`(${F}._live.setView({ omegaMin: 0, omegaMax: 0.3, kMin: 0, kMax: 0.15 }), true)`);
  await waitFor(`!${F}.querySelector(".plot-reset").hidden`, 5000);

  const plotReset1440 = await measureResetVsTitle();
  check(
    "with the view zoomed at 1440x900 reset view and the plot title do not intersect",
    Boolean(plotReset1440 && plotReset1440.resetVisible && !plotReset1440.intersects),
    JSON.stringify(plotReset1440),
  );

  await send("Emulation.clearDeviceMetricsOverride");

  // ---- discoverability: live tags, shared links, link notes, hints ----
  // 1. Exactly six elements with class live-tag exist, one per figure[data-live]
  const tagReport = await ev(`(() => {
    const figs = Array.from(document.querySelectorAll("figure[data-live]"));
    const tags = Array.from(document.querySelectorAll(".live-tag"));
    const countOk = tags.length === 6 && figs.length === 6;
    const eachHasOne = figs.every((f) => f.querySelectorAll(".live-tag").length === 1);
    const besideInteract = tags.every((t) => {
      const p = t.parentElement;
      const btn = p ? p.querySelector(".act-interact") : null;
      return Boolean(btn && (t.nextElementSibling === btn || t.previousElementSibling === btn));
    });
    const textsOk = tags.every((t) => t.textContent.trim() === "live");
    const titlesOk = tags.every((t) => /computes.*in the browser/i.test(t.title));
    return { countOk, eachHasOne, besideInteract, textsOk, titlesOk, count: tags.length };
  })()`);
  check(
    "exactly six elements with class live-tag exist beside interact with browser title and live text",
    Boolean(tagReport && tagReport.countOk && tagReport.eachHasOne && tagReport.besideInteract && tagReport.textsOk && tagReport.titlesOk),
    JSON.stringify(tagReport),
  );

  // 2. Interaction hint is at most 80 characters, full list behind help toggle
  const hintReport = await ev(`(() => {
    const fig = ${F};
    if (!fig) return null;
    const hintShort = fig.querySelector(".hint-short");
    const hintToggle = fig.querySelector(".hint-toggle");
    const hintFull = fig.querySelector(".hint-full");
    if (!hintShort || !hintToggle || !hintFull) return { missing: true };
    const shortText = hintShort.textContent.trim();
    const shortLen = shortText.length;
    const initialHidden = hintFull.hidden;
    const fullText = hintFull.textContent.trim();
    hintToggle.click();
    const toggledVisible = !hintFull.hidden;
    hintToggle.click();
    const closedAgain = hintFull.hidden;
    return {
      shortLen,
      shortText,
      initialHidden,
      toggledVisible,
      closedAgain,
      hasTap: fullText.includes("tap to read values"),
      hasLive: fullText.includes("computed live in this browser"),
      fullLen: fullText.length,
    };
  })()`);
  check(
    "the visible hint under every live figure is at most 80 characters with the full list behind help toggle",
    Boolean(
      hintReport &&
        hintReport.shortLen > 0 &&
        hintReport.shortLen <= 80 &&
        hintReport.initialHidden &&
        hintReport.toggledVisible &&
        hintReport.closedAgain &&
        hintReport.hasTap &&
        hintReport.hasLive,
    ),
    JSON.stringify(hintReport),
  );

  // 3. Shared link: 1440x900 viewport
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await send("Page.navigate", { url: "about:blank" });
  await send("Page.navigate", {
    url: `http://127.0.0.1:${httpPort}/index.html#fig:devils_staircase.D=0.4`,
  });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await waitFor(`!!${S}`, FIGURE_MS);
  await waitFor(`!!(${S} && ${S}.querySelector(".live-link-note"))`, FIGURE_MS);
  await waitFor(`!!(${S} && ${S}._live)`, FIRST_PAINT_MS);

  const measureShared = (figExpr) =>
    ev(`(() => {
      const fig = ${figExpr};
      if (!fig) return null;
      const ctrl = document.querySelector(".controls");
      const cr = ctrl ? ctrl.getBoundingClientRect() : null;
      const r = fig.getBoundingClientRect();
      const note = fig.querySelector(".live-link-note");
      const noteText = note ? note.textContent.trim() : "";
      const inView = Boolean(cr && r.top >= cr.bottom && r.top < window.innerHeight);
      return {
        inView,
        top: r.top,
        controlsBottom: cr ? cr.bottom : null,
        noteText,
        hasButton: Boolean(note && note.querySelector("button")),
      };
    })()`);

  await sleep(300);
  const shared1440_03 = await measureShared(S);
  await sleep(700);
  const shared1440_10 = await measureShared(S);
  await sleep(2000);
  const shared1440_30 = await measureShared(S);
  const settled1440 = Boolean(
    shared1440_03 &&
      shared1440_03.inView &&
      shared1440_10 &&
      shared1440_10.inView &&
      shared1440_30 &&
      shared1440_30.inView,
  );
  check(
    "at 1440x900 a shared link scrolls the figure into view and shows the opened note",
    Boolean(
      settled1440 &&
        shared1440_30.hasButton &&
        shared1440_30.noteText.includes("Opened with D = 0.4 from the link"),
    ),
    JSON.stringify({ t03: shared1440_03, t10: shared1440_10, t30: shared1440_30 }),
  );

  // Reset at 1440x900 returns to default
  await ev(`(${S}.querySelector(".live-link-note button").click(), true)`);
  const reset1440 = await ev(`(() => {
    const fig = ${S};
    const num = fig && fig.querySelector(".live-params input[type=number]");
    const hash = location.hash;
    const noteGone = Boolean(fig && !fig.querySelector(".live-link-note"));
    return {
      val: num ? Number(num.value) : null,
      hash,
      noteGone,
    };
  })()`);
  check(
    "at 1440x900 returning to default restores the control and hash and removes the note",
    Boolean(reset1440 && reset1440.val === 0.25 && !reset1440.hash.includes("0.4") && reset1440.noteGone),
    JSON.stringify(reset1440),
  );

  // 3b. Shared link: reader scroll during layout settle is not fought
  const targetScroll = 300;
  const scrollRuns = [];
  for (let r = 0; r < 3; r++) {
    await send("Page.navigate", { url: "about:blank" });
    await send("Page.navigate", {
      url: `http://127.0.0.1:${httpPort}/index.html#fig:devils_staircase.D=0.4`,
    });
    await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
    await sleep(150);
    await ev(`window.scrollTo({top: ${targetScroll}, behavior: "instant"})`);
    await sleep(350);
    const y05 = await ev("window.scrollY");
    await sleep(1500);
    const y20 = await ev("window.scrollY");
    scrollRuns.push({
      run: r + 1,
      y05,
      y20,
      ok05: Math.abs(y05 - targetScroll) <= 2,
      ok20: Math.abs(y20 - targetScroll) <= 2,
    });
  }
  const scrollHeld = scrollRuns.every((s) => s.ok05 && s.ok20);
  check(
    "a reader scroll during layout settle is not fought and scrollY stays within 2 px of 300 at 0.5 s and 2 s",
    scrollHeld,
    JSON.stringify(scrollRuns),
  );

  // 4. Shared link: 390x844 phone viewport
  await send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await send("Page.navigate", { url: "about:blank" });
  await send("Page.navigate", {
    url: `http://127.0.0.1:${httpPort}/index.html#fig:devils_staircase.D=0.4`,
  });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await waitFor(`!!${S}`, FIGURE_MS);
  await waitFor(`!!(${S} && ${S}.querySelector(".live-link-note"))`, FIGURE_MS);

  await sleep(300);
  const shared390_03 = await measureShared(S);
  await sleep(700);
  const shared390_10 = await measureShared(S);
  await sleep(2000);
  const shared390_30 = await measureShared(S);
  const settled390 = Boolean(
    shared390_03 &&
      shared390_03.inView &&
      shared390_10 &&
      shared390_10.inView &&
      shared390_30 &&
      shared390_30.inView,
  );
  check(
    "at 390x844 a shared link scrolls the figure into view and shows the opened note",
    Boolean(
      settled390 &&
        shared390_30.hasButton &&
        shared390_30.noteText.includes("Opened with D = 0.4 from the link"),
    ),
    JSON.stringify({ t03: shared390_03, t10: shared390_10, t30: shared390_30 }),
  );

  // Reset at 390x844 returns to default
  await ev(`(${S}.querySelector(".live-link-note button").click(), true)`);
  const reset390 = await ev(`(() => {
    const fig = ${S};
    const hash = location.hash;
    const noteGone = Boolean(fig && !fig.querySelector(".live-link-note"));
    return {
      hash,
      noteGone,
    };
  })()`);
  check(
    "at 390x844 returning to default restores the hash and removes the note",
    Boolean(reset390 && !reset390.hash.includes("0.4") && reset390.noteGone),
    JSON.stringify(reset390),
  );

  // 5. A link without parameters shows no note and does not scroll on load
  await ev("sessionStorage.clear()");
  await send("Page.navigate", { url: "about:blank" });
  await send("Page.navigate", {
    url: `http://127.0.0.1:${httpPort}/index.html`,
  });
  await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
  await sleep(300);
  const noParamsReport = await ev(`(() => ({
    scrollY: window.scrollY,
    notes: document.querySelectorAll(".live-link-note").length,
  }))()`);
  check(
    "a link without parameters shows no note and does not scroll the page on load",
    Boolean(noParamsReport && noParamsReport.scrollY === 0 && noParamsReport.notes === 0),
    JSON.stringify(noParamsReport),
  );

  await send("Emulation.clearDeviceMetricsOverride");


  await send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-data", value: "reduce" }],
  });
  await load();
  const reducedMatches = await ev(`matchMedia("(prefers-reduced-data: reduce)").matches`);
  if (!reducedMatches) {
    check(
      "Chrome honours prefers-reduced-data with --enable-blink-features=PrefersReducedData",
      false,
      "matchMedia stayed false, so the reduced-data checks prove nothing",
    );
  } else {
    const from = events.length;
    await ev(`(${F}.querySelector(".act-interact") && ${F}.querySelector(".act-interact").click(), true)`);
    await sleep(REDUCED_WAIT_MS);
    await attachChain;
    const st = await stats();
    const hasLive = await ev(`!!(${F} && ${F}._live)`);
    const liveCanvas = await ev(`!!(${F} && ${F}.querySelector("canvas.plot.live"))`);
    const jsonChart = await ev(`!!(${F} && ${shownStaticChart(F)})`);
    const pngAfter = await imgShown();
    const shown = Boolean(pngAfter || jsonChart);
    const workers = attachedWorkers();
    const hits = wasmHits(from);
    const reducedState = shown && !liveCanvas;
    check(
      "under prefers-reduced-data no tile is computed",
      workers.length === 0 && reducedState && (!hasLive || (st != null && st.painted === 0)),
      JSON.stringify({
        st,
        hasLive,
        workers: workers.map((t) => t.url || t.type),
        shown,
        liveCanvas,
        pngAfter,
        jsonChart,
        state: await ev(`${F} && ${F}.dataset.state`),
      }),
    );
    check(
      "under prefers-reduced-data the WebAssembly module is never downloaded",
      hits.length === 0,
      hits
        .map((e) => e.params.request.url)
        .slice(0, 4)
        .join(" | "),
    );
    check(
      "under prefers-reduced-data no worker target is attached",
      workers.length === 0,
      JSON.stringify(workers.map((t) => ({ type: t.type, url: t.url }))),
    );
    check(
      "under prefers-reduced-data a PNG or JSON chart is shown and no live canvas is created",
      reducedState,
      JSON.stringify({
        liveCanvas,
        pngAfter,
        jsonChart,
        state: await ev(`${F} && ${F}.dataset.state`),
      }),
    );
    // The staircase under reduced data: no slider, no live canvas, no wasm —
    // the published PNG (both panels), or its JSON chart when the site
    // exports one, is all the reader gets. The chart must be displayed.
    await ev(`(${S}.querySelector(".act-interact") && ${S}.querySelector(".act-interact").click(), true)`);
    await sleep(REDUCED_WAIT_MS);
    const stReduced = await ev(`(() => {
      const fig = ${S};
      if (!fig) return { missing: true };
      const img = fig.querySelector(".fig-body img");
      const imgShown = img && getComputedStyle(img).display !== "none";
      const jsonChart = ${shownStaticChart("fig")};
      return {
        shown: Boolean(imgShown || jsonChart),
        liveCanvas: !!fig.querySelector("canvas.plot.live"),
        slider: !!fig.querySelector(".live-params input"),
        hasLive: !!fig._live,
        state: fig.dataset.state,
      };
    })()`);
    check(
      "under prefers-reduced-data the staircase keeps its PNG or JSON chart and mounts no slider",
      Boolean(stReduced && stReduced.shown && !stReduced.liveCanvas && !stReduced.slider && !stReduced.hasLive),
      JSON.stringify(stReduced),
    );
    // The attractors figure under reduced data: same rule — the published
    // twelve-panel PNG stays, no slider, no live canvas, no wasm.
    await ev(`(${A}.querySelector(".act-interact") && ${A}.querySelector(".act-interact").click(), true)`);
    await sleep(REDUCED_WAIT_MS);
    const atReduced = await ev(`(() => {
      const fig = ${A};
      if (!fig) return { missing: true };
      const img = fig.querySelector(".fig-body img");
      const shown = img && getComputedStyle(img).display !== "none";
      return {
        shown,
        liveCanvas: !!fig.querySelector("canvas.plot.live"),
        slider: !!fig.querySelector(".live-params input"),
        hasLive: !!fig._live,
        state: fig.dataset.state,
      };
    })()`);
    check(
      "under prefers-reduced-data the attractors figure keeps its PNG and mounts no slider",
      Boolean(atReduced && atReduced.shown && !atReduced.liveCanvas && !atReduced.slider && !atReduced.hasLive),
      JSON.stringify(atReduced),
    );
    // The torus figure under reduced data: same rule — the published
    // three-panel PNG (or its JSON chart) stays, no selector, no live
    // canvas, no wasm.
    await ev(`(${T}.querySelector(".act-interact") && ${T}.querySelector(".act-interact").click(), true)`);
    await sleep(REDUCED_WAIT_MS);
    const tReduced = await ev(`(() => {
      const fig = ${T};
      if (!fig) return { missing: true };
      const img = fig.querySelector(".fig-body img");
      const imgShown = img && getComputedStyle(img).display !== "none";
      const jsonChart = ${shownStaticChart("fig")};
      return {
        shown: Boolean(imgShown || jsonChart),
        liveCanvas: !!fig.querySelector("canvas.plot.live"),
        slider: !!fig.querySelector(".live-params input"),
        selector: !!fig.querySelector(".live-params select"),
        hasLive: !!fig._live,
        state: fig.dataset.state,
      };
    })()`);
    check(
      "under prefers-reduced-data the torus figure keeps its PNG and mounts no controls",
      Boolean(tReduced && tReduced.shown && !tReduced.liveCanvas && !tReduced.slider && !tReduced.selector && !tReduced.hasLive),
      JSON.stringify(tReduced),
    );
    // The double staircase under reduced data: same rule — the published
    // PNG (or its JSON chart) stays, no controls, no live canvas, no wasm.
    await ev(`(${M}.querySelector(".act-interact") && ${M}.querySelector(".act-interact").click(), true)`);
    await sleep(REDUCED_WAIT_MS);
    const mReduced = await ev(`(() => {
      const fig = ${M};
      if (!fig) return { missing: true };
      const img = fig.querySelector(".fig-body img");
      const imgShown = img && getComputedStyle(img).display !== "none";
      const jsonChart = ${shownStaticChart("fig")};
      return {
        shown: Boolean(imgShown || jsonChart),
        liveCanvas: !!fig.querySelector("canvas.plot.live"),
        slider: !!fig.querySelector(".live-params input"),
        presets: !!fig.querySelector(".live-presets button"),
        hasLive: !!fig._live,
        state: fig.dataset.state,
      };
    })()`);
    check(
      "under prefers-reduced-data the double staircase keeps its PNG and mounts no controls",
      Boolean(mReduced && mReduced.shown && !mReduced.liveCanvas && !mReduced.slider && !mReduced.presets && !mReduced.hasLive),
      JSON.stringify(mReduced),
    );
    // The space-time figure under reduced data: same rule — the published
    // nine-panel PNG stays, no selector, no play button, no live canvas,
    // no wasm.
    await ev(`(${C}.querySelector(".act-interact") && ${C}.querySelector(".act-interact").click(), true)`);
    await sleep(REDUCED_WAIT_MS);
    const cReduced = await ev(`(() => {
      const fig = ${C};
      if (!fig) return { missing: true };
      const img = fig.querySelector(".fig-body img");
      const imgShown = img && getComputedStyle(img).display !== "none";
      const jsonChart = ${shownStaticChart("fig")};
      return {
        shown: Boolean(imgShown || jsonChart),
        liveCanvas: !!fig.querySelector("canvas.plot.live"),
        slider: !!fig.querySelector(".live-params input"),
        selector: !!fig.querySelector(".live-params select"),
        play: !!fig.querySelector(".live-play"),
        hasLive: !!fig._live,
        state: fig.dataset.state,
      };
    })()`);
    check(
      "under prefers-reduced-data the spacetime figure keeps its PNG and mounts no controls",
      Boolean(cReduced && cReduced.shown && !cReduced.liveCanvas && !cReduced.slider && !cReduced.selector && !cReduced.play && !cReduced.hasLive),
      JSON.stringify(cReduced),
    );
  }
  await pullDebug();

  const thrown = events.filter((x) => x.method === "Runtime.exceptionThrown");
  check(
    "the page raises no uncaught exception",
    thrown.length === 0,
    thrown
      .map((x) => (x.params.exceptionDetails.exception || {}).description || x.params.exceptionDetails.text)
      .join(" | ")
      .slice(0, 400),
  );

  dumpConsole();
  ws.close();
  exitCode = failures.length === 0 ? 0 : 1;
  console.log(exitCode === 0 ? "BROWSER PASS" : `BROWSER FAIL (${failures.length})`);
} catch (err) {
  if (!(err && err.message === "chrome-missing")) {
    console.log(`FAIL the browser check crashed: ${err && err.message}`);
  }
  dumpConsole();
} finally {
  clearTimeout(watchdog);
  await cleanup();
}
process.exit(exitCode);
