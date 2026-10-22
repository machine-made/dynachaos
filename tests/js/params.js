import { test } from "node:test";
import assert from "node:assert/strict";
import {
  clampValue,
  debounce,
  parseHash,
  serializeHash,
  snapToStep,
  writeParamsIntoHash,
} from "../../site-src/live/params.js";
import { createParamWiring, LIVE_ATTRACTORS, LIVE_MODULATED, LIVE_SPACETIME, LIVE_STAIRCASE } from "../../site-src/live/live-figure.js";

const SPECS = [
  { name: "D", min: 0, max: 0.5, step: 0.005, default: 0.25 },
  { name: "n", min: 1, max: 9, step: 1, default: 4 },
];

// A manual timer pair: scheduled callbacks queue up and run only when the
// test flushes them, so debounce behaviour is checked without real time.
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

test("clampValue clamps to the spec and falls back to the default", () => {
  const spec = SPECS[0];
  assert.equal(clampValue(0.3, spec), 0.3);
  assert.equal(clampValue(-1, spec), 0);
  assert.equal(clampValue(2, spec), 0.5);
  assert.equal(clampValue(NaN, spec), 0.25);
  assert.equal(clampValue("not a number", spec), 0.25);
  assert.equal(clampValue(Infinity, spec), 0.25);
});

test("hash serialise and parse round-trip the parameter state", () => {
  const state = { D: 0.4, n: 7 };
  const frag = serializeHash("fig:devils_staircase", state);
  assert.equal(frag, "fig:devils_staircase.D=0.4&fig:devils_staircase.n=7");
  const parsed = parseHash(`#sec:circle_map&${frag}`, SPECS, "fig:devils_staircase");
  assert.deepEqual(parsed, { D: 0.4, n: 7 });
});

test("parseHash clamps out-of-range values and skips junk", () => {
  const parsed = parseHash(
    "#sec:x&fig:devils_staircase.D=9&fig:devils_staircase.n=oops&other=1",
    SPECS,
    "fig:devils_staircase",
  );
  assert.deepEqual(parsed, { D: 0.5 });
  assert.deepEqual(parseHash("", SPECS, "fig:devils_staircase"), {});
  assert.deepEqual(parseHash("#sec:x", SPECS, "fig:devils_staircase"), {});
});

test("the prefixed key wins over a bare key", () => {
  const parsed = parseHash("#D=0.1&fig:devils_staircase.D=0.4", SPECS, "fig:devils_staircase");
  assert.equal(parsed.D, 0.4);
});

test("an old shared link with D above the slider max clamps to it", () => {
  // Links shared while the delayed-logistic slider ran to the kernel's 3.5
  // still carry fig:delayed_logistic_attractors.D=3: the parse must clamp
  // to the narrowed max so the link shows the bounded orbit at the max
  // rather than an escaping NaN block on a blank plot.
  const d = LIVE_ATTRACTORS.paramSpecs.find((s) => s.name === "D");
  const parsed = parseHash(
    "#fig:delayed_logistic_attractors.D=3&fig:delayed_logistic_attractors.n=512",
    LIVE_ATTRACTORS.paramSpecs,
    "fig:delayed_logistic_attractors",
  );
  assert.equal(parsed.D, d.max);
  assert.equal(parsed.n, 512);
});

test("writeParamsIntoHash keeps the section id and replaces only this figure's tokens", () => {
  const hash = "#sec:circle_map&fig:devils_staircase.D=0.9&fig:other.D=0.1";
  const next = writeParamsIntoHash(hash, "fig:devils_staircase", { D: 0.4 });
  assert.equal(next, "sec:circle_map&fig:other.D=0.1&fig:devils_staircase.D=0.4");
  // Round-trip: parsing the written hash returns the state that was written.
  const parsed = parseHash(`#${next}`, SPECS, "fig:devils_staircase");
  assert.equal(parsed.D, 0.4);
});

test("debounce collapses a burst into one trailing call", () => {
  const timers = manualTimers();
  let calls = 0;
  const fire = debounce(() => {
    calls += 1;
  }, 150, timers);
  fire();
  fire();
  fire();
  assert.equal(timers.queue.length, 3);
  timers.flush();
  assert.equal(calls, 1);
});

test("a slider burst recomputes once through the wiring, not once per input", () => {
  // This drives createParamWiring, the code the page's inputs call — a
  // debounce dropped there fires apply per input event, which this catches.
  const timers = manualTimers();
  let applies = 0;
  let hashes = 0;
  const echoes = [];
  const wiring = createParamWiring({
    specs: LIVE_STAIRCASE.paramSpecs,
    debounceMs: 150,
    echo: (name, v) => echoes.push([name, v]),
    apply: () => {
      applies += 1;
    },
    writeHash: () => {
      hashes += 1;
    },
    timers,
  });
  for (const v of [0.3, 0.35, 0.4, 0.45]) wiring.onInput("D", v);
  timers.flush();
  assert.equal(applies, 1);
  assert.equal(hashes, 1);
  assert.equal(wiring.state.D, 0.45);
  assert.deepEqual(echoes, [
    ["D", 0.3],
    ["D", 0.35],
    ["D", 0.4],
    ["D", 0.45],
  ]);
});

test("setParams applies a hash state immediately and clamps it", () => {
  let applies = 0;
  const echoes = [];
  const wiring = createParamWiring({
    specs: LIVE_STAIRCASE.paramSpecs,
    debounceMs: 150,
    echo: (name, v) => echoes.push([name, v]),
    apply: () => {
      applies += 1;
    },
    writeHash: () => {},
    timers: manualTimers(),
  });
  wiring.setParams({ D: 0.9 });
  assert.equal(wiring.state.D, 0.5);
  assert.equal(applies, 1);
  assert.deepEqual(echoes, [["D", 0.5]]);
});

test("the double staircase's hash state round-trips through the URL fragment", () => {
  // eps, the D window and the point count all ride the hash; the preset
  // window's full precision must survive the String()/Number() round trip.
  const state = { eps: 0.12, dMin: 0.25502630263026305, dMax: 0.6270621062106211, n: 384 };
  const frag = serializeHash("fig:double_staircase", state);
  const parsed = parseHash(
    `#sec:three_torus&${frag}`,
    LIVE_MODULATED.paramSpecs,
    "fig:double_staircase",
  );
  assert.deepEqual(parsed, state);
});

test("snapToStep lands on the decimal literal the slider shows", () => {
  // A range input's stepped value is a double like 0.07000000000000001;
  // the CML figure's eps must reach the kernel as 0.07 exactly — a one-ulp
  // difference moves a chaotic orbit.
  const eps = { name: "eps", min: 0, max: 0.5, step: 0.001, default: 0.07 };
  assert.equal(snapToStep(0.07000000000000001, eps), 0.07);
  assert.equal(snapToStep(0.1234567, eps), 0.123);
  assert.equal(snapToStep(0.4999, eps), 0.5);
  assert.equal(snapToStep(-0.2, eps), 0);
  assert.equal(snapToStep(NaN, eps), 0.07);
  // A spec without a step passes through clamped but unsnapped.
  assert.equal(snapToStep(0.1234567, { name: "x", min: 0, max: 1, default: 0.5 }), 0.1234567);
});

test("the spacetime figure's hash state round-trips through the URL fragment", () => {
  // Model, eps and sites ride the hash; the play position does not — it
  // is not part of the parameter state.
  const state = { model: 2, eps: 0.2, sites: 384 };
  const frag = serializeHash("fig:spacetime_diagrams", state);
  const parsed = parseHash(
    `#sec:sti&${frag}`,
    LIVE_SPACETIME.paramSpecs,
    "fig:spacetime_diagrams",
  );
  assert.deepEqual(parsed, state);
});
