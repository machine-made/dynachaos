// Smoke test for the vendored wasm package: pkg/ must compute what the
// native build computes.
//
// Run with: node --test tests/js/pkg.js
//
// The inputs are the ones the parity scripts use. zero_one_k runs on the
// same logistic-map series, frequencies and n_cut as
// scripts/check_wasm_diagnostics.py. rotation_number_point runs with the
// parity tile's transient, iteration count and theta0 (200, 500, 0.1) from
// rust/wasm/examples/tile_reference.rs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";

const PKG_JS = new URL("../../pkg/dynachaos_wasm.js", import.meta.url);
const PKG_WASM = new URL("../../pkg/dynachaos_wasm_bg.wasm", import.meta.url);

test("the vendored package matches the native kernels", async () => {
  if (!existsSync(PKG_JS) || !existsSync(PKG_WASM)) {
    assert.fail(
      "pkg/ is missing. Build it with: uv run python scripts/build_wasm_pkg.py",
    );
  }
  const mod = await import(PKG_JS.href);
  mod.initSync({ module: readFileSync(PKG_WASM) });

  assert.equal(
    typeof mod.rotation_number_point,
    "function",
    "pkg/ does not export rotation_number_point",
  );
  assert.equal(
    typeof mod.zero_one_k,
    "function",
    "pkg/ does not export zero_one_k",
  );

  // Locked cells return the exact rational. These native values are the
  // ones rust/core/src/circle_map.rs asserts in its own tests
  // (a_locked_point_returns_the_exact_rational and the locked-cells case
  // at (0.5, 0.2)).
  assert.equal(mod.rotation_number_point(0.05, 0.12, 200, 500, 0.1)[0], 0.0);
  assert.equal(mod.rotation_number_point(0.5, 0.2, 200, 500, 0.1)[0], 0.5);

  // Unlocked cell. The native value is a run of the Python kernel:
  //   uv run python -c "from dynachaos.maps.circle_map import rotation_number; \
  //     print(rotation_number(0.03, 0.08, n_transient=200, n_iter=500, theta0=0.1))"
  // printed 0.07427455154930666. The 1e-9 bound matches the loosest
  // tolerance scripts/check_wasm_diagnostics.py grants a sin/cos kernel.
  const rho = mod.rotation_number_point(0.08, 0.03, 200, 500, 0.1)[0];
  assert.ok(
    Math.abs(rho - 0.07427455154930666) <= 1e-9,
    `rotation_number_point(0.08, 0.03) = ${rho}, native 0.07427455154930666`,
  );

  // zero_one_k on the same series, frequencies and n_cut as
  // scripts/check_wasm_diagnostics.py. The native values are a run of
  //   uv run python -c "import numpy as np; from dynachaos._rust import zero_one_k; \
  //     x = ...same logistic series...; \
  //     print(zero_one_k(x, np.array([0.7, 1.1, 1.9, 2.6, 3.3]), 100))"
  // which printed the literals below. The 1e-9 absolute bound is the one
  // the diagnostics script states for this export.
  const x = new Float64Array(2000);
  let v = 0.123456789;
  for (let i = 0; i < 2000; i++) {
    v = 4.0 * v * (1.0 - v);
    x[i] = v;
  }
  const out = mod.zero_one_k(x, new Float64Array([0.7, 1.1, 1.9, 2.6, 3.3]), 100);
  assert.deepEqual(
    Array.from(out.slice(0, 3)),
    [2000, 5, 100],
    "zero_one_k header does not echo the request",
  );
  const native = [
    0.9930764572784933, 0.9948699805232564, 0.9952543995051647,
    0.9996150253763081, 0.992580735157965,
  ];
  for (let i = 0; i < native.length; i++) {
    assert.ok(
      Math.abs(out[3 + i] - native[i]) <= 1e-9,
      `zero_one_k[${i}] = ${out[3 + i]}, native ${native[i]}`,
    );
  }
});
