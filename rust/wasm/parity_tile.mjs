// Print the same fixed tile as examples/tile_reference.rs, from the
// WebAssembly build, as one hexadecimal f64 bit pattern per line.
//
//   node rust/wasm/parity_tile.mjs <path to site/wasm>
//
// Keep the parameters in step with examples/tile_reference.rs.
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { join } from "node:path";

const dir = process.argv[2];
const glue = pathToFileURL(join(dir, "dynachaos_wasm.js")).href;
const { default: init, rotation_number_tile } = await import(glue);
await init({ module_or_path: await readFile(join(dir, "dynachaos_wasm_bg.wasm")) });

const tile = rotation_number_tile(0.0, 1.0, 64, 0.0, 0.3, 64, 200, 500, 0.1);

const view = new DataView(new ArrayBuffer(8));
const lines = [];
for (const value of tile) {
  view.setFloat64(0, value);
  lines.push(view.getBigUint64(0).toString(16).padStart(16, "0"));
}
process.stdout.write(lines.join("\n") + "\n");
