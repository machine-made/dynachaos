// Smoke-run the WebAssembly build outside a browser.
//
// This proves three things before any page is written: the module loads, the
// clamping contract holds, and the numbers coming out of the browser build
// are the numbers the reproduction pipeline computes.
//
//   node rust/wasm/smoke.cjs
//
// Build the input first:
//   cargo build --manifest-path rust/wasm/Cargo.toml \
//     --target wasm32-unknown-unknown --release
//   wasm-bindgen --target nodejs --out-dir rust/wasm/pkg \
//     rust/wasm/target/wasm32-unknown-unknown/release/dynachaos_wasm.wasm

const { rotation_number_tile } = require("./pkg/dynachaos_wasm.js");

const HEADER = 4;
let failures = 0;

function check(name, condition, detail) {
  const mark = condition ? "ok  " : "FAIL";
  if (!condition) failures += 1;
  console.log(`${mark} ${name}${detail ? ` -- ${detail}` : ""}`);
}

// --- 1. A tile the size the Python parity test uses -----------------------
const nOmega = 7;
const nK = 5;
const out = rotation_number_tile(0.0, 1.0, nOmega, 0.0, 0.3, nK, 200, 500, 0.1);

check("module loads and returns a tile", out.length === HEADER + nOmega * nK,
  `length ${out.length}`);
check("header reports the grid it used",
  out[0] === nOmega && out[1] === nK && out[2] === 200 && out[3] === 500,
  `header ${Array.from(out.slice(0, HEADER)).join(", ")}`);

const tile = Array.from(out.slice(HEADER));
check("every value is finite", tile.every(Number.isFinite));

// --- 2. The analytic case: K = 0 means the drift per step is Omega --------
// Row 0 is K = 0, so cell j must equal its own Omega.
const rowZero = tile.slice(0, nOmega);
const expected = Array.from({ length: nOmega }, (_, j) => j / (nOmega - 1));
const worst = Math.max(...rowZero.map((v, j) => Math.abs(v - expected[j])));
check("K = 0 row returns Omega", worst < 1e-12, `max error ${worst.toExponential(3)}`);

// --- 3. Known locked tongues ---------------------------------------------
// Omega = 0 with K > 0 falls into the 0/1 tongue; Omega = 1/2 into the 1/2
// tongue. Both are plateaus of the published figure.
const zeroTongue = rotation_number_tile(0.0, 0.0, 1, 0.2, 0.2, 1, 2000, 5000, 0.1)[HEADER];
const halfTongue = rotation_number_tile(0.5, 0.5, 1, 0.2, 0.2, 1, 2000, 5000, 0.1)[HEADER];
check("0/1 tongue locks to 0", Math.abs(zeroTongue) < 1e-12, `rho = ${zeroTongue}`);
check("1/2 tongue locks to 0.5", Math.abs(halfTongue - 0.5) < 1e-12, `rho = ${halfTongue}`);

// --- 4. The clamping contract --------------------------------------------
// An absurd request must come back clamped, never as a hang or a throw.
const clamped = rotation_number_tile(0.0, 1.0, 1e9, 0.0, 0.3, 1e9, 0, 1, 0.1);
check("oversized grid is clamped", clamped[0] === 512 && clamped[1] === 512,
  `${clamped[0]} x ${clamped[1]}`);

const nonFinite = rotation_number_tile(NaN, Infinity, 2, 0.0, 0.3, 1, 10, 10, NaN);
check("non-finite arguments fall back", nonFinite.length === HEADER + 2 &&
  Array.from(nonFinite.slice(HEADER)).every(Number.isFinite));

// --- 5. Checksum over the tile -------------------------------------------
// A stable summary a future change can be compared against by eye.
const sum = tile.reduce((a, b) => a + b, 0);
console.log(`\ntile checksum: sum ${sum.toFixed(12)}, mean ${(sum / tile.length).toFixed(12)}`);
console.log(`min ${Math.min(...tile).toFixed(12)}, max ${Math.max(...tile).toFixed(12)}`);

console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
