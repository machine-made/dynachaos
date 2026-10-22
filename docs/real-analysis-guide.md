# Real-analysis user guide

This page is the practical spine for applying dynachaos to real simulated or measured time signals. The reproduction gallery in `figures/` is the flagship stress test for the package, but the workflow is general: start from a scalar or reduced observable, choose diagnostics that match the scientific question, and keep the finite-data limits visible in the output metadata.

## 1. Prepare the input signal

The config-driven workflow expects one finite 1D signal:

- `.npy` containing a single NumPy array, or `.npz` with `input.npz_key` when the archive has more than one array.
- Shape `(N,)`; reduce fields, lattices, or particle data to a scalar observable first. Typical choices are a spatial mean, order parameter, probe value, kinetic energy, dissipation, or leading principal-component amplitude.
- Finite values only. Remove NaN/Inf samples before analysis and record the preprocessing in your project notes.
- Use SI units or nondimensional variables consistently. dynachaos does not infer units from arrays.
- Downsample only when the sampling rate is higher than the time scales your diagnostic needs. Keep the downsampling factor outside the array filename or in your run log; the workflow records that the file was used, not the full upstream preprocessing chain.

For self-contained checks and examples, the workflow can also generate a logistic-map signal from the config. Use generated signals for quick checks, not as a substitute for the real input provenance of a study.

## 2. Choose diagnostics

The scalable workflow currently accepts these diagnostic names in `diagnostics`:

| diagnostic | Use when | Main caveat |
|---|---|---|
| `permutation_entropy` | You need a fast ordinal-complexity summary of a scalar signal. | Sensitive to embedding dimension `d`, delay `tau`, ties, and short records. |
| `correlation_dimension` | You want a Grassberger-Procaccia estimate from an embedded trajectory. | Finite data can hide or fake scaling plateaus; inspect radii, local slopes, and reliability warnings. |
| `rqa_streaming` | You need recurrence quantifiers for long signals without storing a dense recurrence matrix. | Pass an explicit `eps` for long runs; automatic percentile thresholding can still require all pairwise distances. |
| `rqa_dense` | You need the dense recurrence matrix semantics for small or deliberately bounded cases. | Memory scales as `8*N^2` bytes for the distance matrix before extra temporaries. The workflow guards this path. |

Other library diagnostics are available through the Python API, but the reproducible user workflow is intentionally narrower. Add a diagnostic to the workflow only after it has a config schema, output contract, and reliability metadata.

## 3. Run the scalable workflow

A minimal external-signal config looks like this:

```jsonc
{
  "input": {"path": "signal.npy"},
  "output": {"dir": "results/run01"},
  "scale_limits": {"dense_rqa_max_bytes": 4294967296},
  "diagnostics": [
    {"name": "permutation_entropy", "d": 5, "tau": 1},
    {"name": "correlation_dimension", "embedding": {"d": 3, "tau": 2}, "theiler_window": 2},
    {"name": "rqa_streaming", "embedding": {"d": 3, "tau": 2}, "eps": 0.08, "l_min": 2, "v_min": 2}
  ]
}
```

Run it from the package checkout or from an environment where `dynachaos` is installed:

```bash
uv run dynachaos analyze path/to/config.jsonc
```

This command is a local/full-run template rather than a runnable repository example: it requires your own `path/to/config.jsonc` and any input files named by that config. For tested commands, use the README quickstart or the recipe gallery.

## 4. Read the outputs

Each workflow run writes three stable artifacts under `output.dir`:

- `results.json` — numerical values, arrays such as radii or local slopes, and per-diagnostic result dictionaries.
- `metadata.json` — input provenance, signal length/shape, workflow wall time in seconds, peak RSS in MB, per-diagnostic cost, and per-diagnostic `ReliabilityRecord` objects.
- `summary.md` — compact human-readable report that names the results and metadata files.

`ReliabilityRecord` is not a pass/fail certificate. It is a compact record of what the method actually did: method name, backend, parameters, data length and shape, sampling/downsampling note, validity warnings, unresolved verdicts, scale evidence, and schema version. Treat warnings and unresolved verdicts as part of the result. They mean that the diagnostic ran, but some scientific judgment remains open.

Plain-language examples:

- A correlation-dimension estimate with a weak scaling mask is evidence to inspect, not a dimension measurement to quote uncritically.
- An RQA result with explicit `eps` is reproducible for that threshold, but the threshold choice still needs scientific justification.
- A Rust backend label means that a tested kernel was used for that diagnostic path. It is not a blanket acceleration claim for every algorithm in the package.

## 5. Long-signal and RQA scaling guidance

For long signals, avoid dense recurrence matrices unless you have calculated the memory cost. Dense recurrence/RQA first builds an `N x N` distance matrix with an analytical `8*N^2` byte cost, before Python and temporary-array overhead. Under the default cap the checked benchmark artifact places that threshold near `N = 23170`; see [RQA scaling design note](rqa-scaling-design.md).

Preferred long-signal pattern:

1. Reduce the raw simulation or experiment to a scalar/low-dimensional observable.
2. Downsample only as much as your science permits.
3. Use `rqa_streaming` with an explicit `eps`.
4. Keep `scale_limits.dense_rqa_max_bytes` in the config so accidental `rqa_dense` requests stop early.
5. Read `metadata.json` before quoting results.

Rust acceleration is targeted, not universal. Current measured and planned kernels are summarized in [Rust acceleration roadmap](rust-acceleration-roadmap.md). Streaming RQA is the leading future Rust candidate because it avoids dense memory growth but still spends time in Python scans.

## 6. Reproduce tested examples

Use only the tested recipe gallery for copyable public commands:

- [Example gallery](../examples/README.md)
- [External signal workflow recipe](../examples/recipes/external_signal/README.md)
- [Long-signal streaming recipe](../examples/recipes/long_signal_streaming/README.md)

Those recipes exercise the same workflow output contract described above and are covered by `tests/test_examples.py`.

## 7. Scope of the reproduction gallery

The section-indexed reproductions of Kaneko's published work are the flagship application: they stress maps, coupled-map lattices, recurrence diagnostics, entropy diagnostics, and scaling behavior at high resolution. The package boundary is broader. dynachaos is meant for reusable dynamical-systems analysis of external simulated or measured time signals, with the reproduction gallery serving as a demanding example rather than the only supported use case.

For claim boundaries and public wording, see [claims-checklist.md](claims-checklist.md).
