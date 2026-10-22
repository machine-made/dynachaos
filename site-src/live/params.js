/**
 * Parameter plumbing for live figures: specs, clamping, debounce, and
 * URL-hash state.
 *
 * Pure ESM: no DOM and no globals at import time, so node tests can drive
 * every function. The page (scripts/paper_shell.py) builds the inputs and
 * hands in the callbacks; the decisions live here.
 *
 * A spec is `{ name, min, max, step, default }`. The hash fragment carries
 * `&`-joined `key=value` tokens after the section id, so a reader can share
 * an exact view: `#sec:circle_map&fig:devils_staircase.D=0.4`. Keys are
 * prefixed with the figure id so two live figures on one page cannot
 * collide.
 */

/**
 * Clamp a value into a spec's [min, max]. A non-finite value falls back to
 * the spec's default: a slider never emits one, but a hash or a number
 * field can carry anything.
 *
 * @param {number} value
 * @param {{ min: number, max: number, default: number }} spec
 * @returns {number}
 */
export function clampValue(value, spec) {
  const v = Number(value);
  if (!Number.isFinite(v)) return spec.default;
  return Math.min(spec.max, Math.max(spec.min, v));
}


/**
 * Snap a value onto a spec's `min + k * step` grid, then round to the
 * step's own decimal places. A range input's stepped value is a double
 * like 0.07000000000000001; the kernel must see the decimal literal the
 * reader picked (0.07), because a one-ulp eps change moves a chaotic
 * orbit. Specs without a `step` (or with a non-positive one) pass through
 * clamped but unsnapped.
 *
 * @param {number} value
 * @param {{ min: number, max: number, step?: number, default: number }} spec
 * @returns {number}
 */
export function snapToStep(value, spec) {
  const v = clampValue(value, spec);
  const step = Number(spec.step);
  if (!Number.isFinite(step) || step <= 0) return v;
  const k = Math.round((v - spec.min) / step);
  let snapped = spec.min + k * step;
  const text = String(step);
  const dot = text.indexOf(".");
  if (dot >= 0 && text.indexOf("e") < 0 && text.indexOf("E") < 0) {
    snapped = Number(snapped.toFixed(text.length - dot - 1));
  }
  return Math.min(spec.max, Math.max(spec.min, snapped));
}

/**
 * Collapse a burst of calls into one trailing call after `ms` quiet.
 * `set`/`clear` are injectable so a node test can drive the timer by hand.
 *
 * @param {(...args: unknown[]) => void} fn
 * @param {number} ms
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [timers]
 * @returns {(...args: unknown[]) => void}
 */
export function debounce(fn, ms, timers = {}) {
  const set = timers.set || ((f, t) => setTimeout(f, t));
  const clear = timers.clear || ((id) => clearTimeout(id));
  let pending = null;
  return function debounced(...args) {
    if (pending !== null) clear(pending);
    pending = set(() => {
      pending = null;
      fn(...args);
    }, ms);
  };
}

/**
 * Serialise parameter state to hash tokens: `fig:<id>.<name>=<value>`,
 * `&`-joined, no leading `&`. `String(v)` round-trips an f64 exactly.
 *
 * @param {string} prefix figure id, e.g. "fig:devils_staircase"
 * @param {object} state name -> value
 * @returns {string}
 */
export function serializeHash(prefix, state) {
  const parts = [];
  for (const name of Object.keys(state)) {
    const v = state[name];
    if (!Number.isFinite(v)) continue;
    parts.push(`${prefix}.${name}=${String(v)}`);
  }
  return parts.join("&");
}

/**
 * Parse hash tokens back into clamped parameter state. The fragment may
 * carry a leading `#` and a bare section id before the `&` tokens; tokens
 * without `=` are skipped. A key is honoured bare (`D`) or prefixed
 * (`fig:devils_staircase.D`); the prefixed form wins when both appear.
 * Unknown keys and non-numeric values are ignored.
 *
 * @param {string} hash
 * @param {{ name: string, min: number, max: number, default: number }[]} specs
 * @param {string} [prefix]
 * @returns {object} name -> clamped value (only names the hash set)
 */
export function parseHash(hash, specs, prefix = "") {
  const out = {};
  const prefixedKeys = new Set();
  const frag = String(hash || "").replace(/^#/, "");
  for (const token of frag.split("&")) {
    const eq = token.indexOf("=");
    if (eq < 0) continue;
    const key = token.slice(0, eq);
    const value = Number(token.slice(eq + 1));
    for (const spec of specs) {
      const prefixed = prefix ? `${prefix}.${spec.name}` : spec.name;
      if (key !== spec.name && key !== prefixed) continue;
      if (!Number.isFinite(value)) continue;
      // A bare key fills in only where the prefixed form did not speak.
      if (key === spec.name && prefixedKeys.has(spec.name)) continue;
      if (key === prefixed) prefixedKeys.add(spec.name);
      out[spec.name] = clampValue(value, spec);
    }
  }
  return out;
}
export function writeParamsIntoHash(hash, prefix, state) {
  const frag = String(hash || "").replace(/^#/, "");
  const kept = [];
  for (const token of frag.split("&")) {
    if (!token) continue;
    const eq = token.indexOf("=");
    if (eq < 0) {
      kept.push(token); // the section id
      continue;
    }
    if (token.slice(0, eq).startsWith(`${prefix}.`)) continue; // stale token of ours
    kept.push(token);
  }
  const params = serializeHash(prefix, state);
  return kept.concat(params ? [params] : []).join("&");
}
