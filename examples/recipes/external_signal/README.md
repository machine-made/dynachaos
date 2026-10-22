# External signal workflow recipe

This tested recipe starts from an externally supplied scalar signal and selects a
small diagnostic set. It is the path to copy when your data already live in a
`.npy` or single-array `.npz` file.

## Command

From this directory:

```sh
dynachaos analyze external_signal_recipe.jsonc
```

The config keeps every output relative to this recipe directory, so reruns are
safe and deterministic.

## Inputs and selected diagnostics

- Input signal: `external_signal_fixture.npy`
- Config: `external_signal_recipe.jsonc`
- Diagnostics:
  - `permutation_entropy`
  - `correlation_dimension`
  - `rqa_streaming`

## Output artifacts

Running the command writes these named artifacts under
`outputs/external_signal_recipe/`:

- `results.json` — numerical diagnostic results.
- `metadata.json` — input provenance, scale/cost records, and per-diagnostic
  reliability metadata.
- `summary.md` — compact human-readable summary that names the results and
  metadata files.

The reliability records in `metadata.json` include `method_name`, `backend`,
`parameters`, `data_length`, `data_shape`, sampling/downsampling notes,
validity warnings, unresolved verdicts, scale evidence, and schema version for
each diagnostic.
