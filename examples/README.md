# dynachaos examples

For input expectations, diagnostic choice, workflow outputs, reliability
metadata, and long-signal caveats, start with the
[real-analysis user guide](../docs/real-analysis-guide.md).

The example gallery has two tiers:

1. **Tested recipes** in `examples/recipes/` are copyable command-line workflows
   with JSONC configs and deterministic output directories.
2. **Legacy/internal benchmark scripts** (`benchmark_*.py`, `_pipeline.py`, and
   their checked-in PNG/JSONC companions) are retained for benchmark and gallery
   continuity. They still use the project helper pipeline, but the supported
   user-facing recipes are the tested directories below.

## Tested recipes

The commands below are local recipe commands: run them from a checkout where the
`dynachaos` command is installed or from an activated project environment.

### External signal workflow

Directory: `examples/recipes/external_signal/`

```sh
cd examples/recipes/external_signal
dynachaos analyze external_signal_recipe.jsonc
```

This recipe demonstrates an external `.npy` signal, explicit diagnostic
selection (`permutation_entropy`, `correlation_dimension`, `rqa_streaming`), and
named outputs under `outputs/external_signal_recipe/`: `results.json`,
`metadata.json`, and `summary.md`. The metadata file contains per-diagnostic
reliability records.

### Long-signal streaming workflow

Directory: `examples/recipes/long_signal_streaming/`

```sh
cd examples/recipes/long_signal_streaming
dynachaos analyze long_signal_streaming_recipe.jsonc
```

This recipe demonstrates the long-signal strategy used to stay inside the dense
recurrence memory cost: reduce/downsample large simulation output to a scalar
observable, then run `rqa_streaming` instead of `rqa_dense`. The CI fixture is
small, while the recipe README documents how to replace it with a local large
run. Outputs are written under `outputs/long_signal_streaming_recipe/`.
