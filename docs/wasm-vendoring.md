# Vendoring the dynachaos wasm package into chaos-atlas

chaos-atlas runs the same kernels the paper ran by vendoring a pre-built
wasm package. A git dependency would force chaos-atlas's static Next.js
build (`output: "export"`) to install Rust and wasm-bindgen; a copied
artifact does not. The package is `pkg/` at the root of this repository,
built by `scripts/build_wasm_pkg.py` and uploaded by CI as the
`dynachaos-wasm-pkg` artifact of the `wasm-parity` job in
`.github/workflows/ci.yml`.

## What the package is

`pkg/` contains:

- `dynachaos_wasm.js` — ES-module glue (`wasm-bindgen --target web` with
  TypeScript output). Initialise once with `initSync` and the `.wasm`
  bytes, or with the default async `init`.
- `dynachaos_wasm_bg.wasm` — the compiled kernels.
- `dynachaos_wasm.d.ts`, `dynachaos_wasm_bg.wasm.d.ts` — declarations.
- `package.json` — name `@openfluids/dynachaos-wasm`, version equal to the
  `rust/wasm` crate version, `"type": "module"`, no dependencies, and a
  `dynachaos` field recording the source commit and wasm-bindgen version.
- `README.md` — the same provenance in prose, plus the build date (UTC).
- `LICENSE` — Apache-2.0, a byte copy of this repository's LICENSE.

Every export clamps its inputs and returns one flat `Float64Array`; the
layouts are documented on each export in `rust/wasm/src/lib.rs`.

## Fetching the artifact

The `wasm-parity` job uploads `pkg/` on every CI run of `main`. From a
checkout of chaos-atlas with `gh` authenticated:

```sh
# Find the newest successful CI run on main.
run_id=$(gh run list --repo openfluids/dynachaos --workflow ci.yml \
  --branch main --status success --limit 1 --json databaseId --jq '.[0].databaseId')

# Download just the wasm package artifact.
gh run download "$run_id" --repo openfluids/dynachaos \
  --name dynachaos-wasm-pkg --dir /tmp/dynachaos-wasm-pkg
```

The artifact is also listed under the run's "Artifacts" section on
github.com for a manual download.

## Building it instead

When the artifact is not reachable, or chaos-atlas needs a dynachaos commit
CI has not built, build the package from a dynachaos checkout:

```sh
git clone https://github.com/openfluids/dynachaos
cd dynachaos
uv run python scripts/build_wasm_pkg.py   # writes ./pkg/
```

This needs the Rust toolchain with the `wasm32-unknown-unknown` target and
a matching `wasm-bindgen` CLI; the script installs the CLI itself when the
installed version does not match `rust/wasm/Cargo.lock`.

## Vendoring

Copy the package into chaos-atlas and record where it came from:

```sh
rsync -a --delete /tmp/dynachaos-wasm-pkg/ \
  /path/to/chaos-atlas/vendor/dynachaos-wasm/
```

The vendored `package.json` already records the source commit in
`dynachaos.commit`; quote it in the chaos-atlas commit message (for example
`vendor: dynachaos-wasm @ <commit>`) so the vendoring commit names the
upstream commit it carries. `vendor/dynachaos-wasm` is committed to
chaos-atlas — it is the dependency, not a build output.

In chaos-atlas code the package is a plain ES module:

```js
import init, { initSync, rotation_number_point, zero_one_k }
  from "../vendor/dynachaos-wasm/dynachaos_wasm.js";
```

## Updating

Updating is the same recipe over the existing directory: fetch (or build)
the new `pkg/`, `rsync -a --delete` it over `vendor/dynachaos-wasm`, and
commit with the new `dynachaos.commit` in the message. The `--delete` flag
matters: a stale file left behind (for example a renamed glue file) would
be silently served next to the new one.

Before committing an update, run the kernels the panel uses once against
the previous vendored copy — or against the native values in
`tests/js/pkg.js` in this repository — so a kernel change upstream is a
deliberate bump, not a surprise.
