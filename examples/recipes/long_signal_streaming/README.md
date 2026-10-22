# Long-signal streaming recipe

This tested recipe demonstrates the memory-safe path for long simulations or
large-degree-of-freedom outputs: reduce the simulation to a scalar observable,
downsample if the sampling rate is higher than the diagnostic needs, and use the
streaming RQA diagnostic instead of materializing a dense recurrence matrix.

## Command

From this directory:

```sh
dynachaos analyze long_signal_streaming_recipe.jsonc
```

The CI fixture `downsampled_observable_fixture.npy` is intentionally small, but
it has the same file layout as a local long run after preprocessing.

## Strategy for local large runs

1. Convert the high-dimensional run to a scalar or low-dimensional observable
   outside this recipe, for example a spatial mean, order parameter, probe point,
   or principal component.
2. Save only the reduced/downsampled signal beside the config as
   `downsampled_observable_fixture.npy` or update `input.path` to your local
   filename.
3. Keep `rqa_streaming` in the diagnostic list. It computes recurrence
   quantifiers from the trajectory without storing the full dense `N x N`
   recurrence matrix.
4. Keep `scale_limits.dense_rqa_max_bytes` in the config. If you deliberately
   switch to `rqa_dense`, the workflow will stop before exceeding the configured
   dense-recurrence memory cap unless you explicitly opt out.

## Output artifacts

Running the command writes deterministic artifacts under
`outputs/long_signal_streaming_recipe/`:

- `results.json` — selected diagnostics (`permutation_entropy`,
  `rqa_streaming`).
- `metadata.json` — reliability metadata and scale/cost records, including the
  signal length and per-diagnostic backend/parameter records.
- `summary.md` — compact summary with artifact names.
