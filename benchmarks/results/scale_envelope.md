# Scale-envelope benchmark (CI mode artifacts)

Generated: 2026-06-10T19:31:38Z
Command: `python benchmarks/scale_envelope.py scale_envelope.jsonc`

## Hardware and software

- CPU: AMD Ryzen 9 9900X 12-Core Processor
- Platform: Linux-7.0.0-22-generic-x86_64-with-glibc2.43
- Python: 3.13.13; NumPy: 2.4.2
- Full-mode command: set `mode` to `full` in the JSONC config, then run `python benchmarks/scale_envelope.py scale_envelope.jsonc` on the target machine; this report header records the resulting hardware context.

## Caveats

- CI mode uses small sizes to check that the benchmark runs and that the backends agree. It is not a publication-scale timing claim.
- Correlation-dimension parity is finite-data parity on identical synthetic inputs and radius grids.
- Python fallback GP cases are capped at N=1000 in this mode because the all-pairs fallback runtime grows rapidly; skipped full-mode rows say this explicitly.
- Dense recurrence estimates report the analytical 8*N^2 byte cost of the distance matrix; safety skips use a 3x temporary-array multiplier for pdist, positive-distance, and boolean recurrence intermediates, while real peak RSS also includes interpreter overhead.
- The CML-flat signal is a coupled-logistic lattice row sequence flattened to mimic larger-DOF simulation output while keeping a scalar diagnostic input.

## Headline

Largest common GP case: logistic N=1000; Rust 0.000861385 s vs Python 0.0370001 s (42.95x).
Dense recurrence/RQA configured impracticality threshold: N≈23170 at the configured bound of 4294967296 predicted 8*N^2 bytes.

## Grassberger--Procaccia

| signal | N | backend | p50 wall s | peak RSS MB | max delta logC | max delta slope |
|---|---:|---|---:|---:|---:|---:|
| logistic | 300 | rust | 0.000817536 | 107.9 | 0.0 | 0.0 |
| logistic | 300 | python | 0.00795993 | 107.1 | 0.0 | 0.0 |
| logistic | 600 | rust | 0.000597957 | 108.0 | 0.0 | 0.0 |
| logistic | 600 | python | 0.0184668 | 107.3 | 0.0 | 0.0 |
| logistic | 1000 | rust | 0.000861385 | 107.9 | 0.0 | 0.0 |
| logistic | 1000 | python | 0.0370001 | 107.5 | 0.0 | 0.0 |
| cml_flat | 300 | rust | 0.000904615 | 108.1 | 0.0 | 0.0 |
| cml_flat | 300 | python | 0.00932315 | 107.5 | 0.0 | 0.0 |
| cml_flat | 600 | rust | 0.000750257 | 108.3 | 0.0 | 0.0 |
| cml_flat | 600 | python | 0.0187087 | 107.5 | 0.0 | 0.0 |
| cml_flat | 1000 | rust | 0.000836385 | 108.3 | 0.0 | 0.0 |
| cml_flat | 1000 | python | 0.0382562 | 107.8 | 0.0 | 0.0 |

## Dense recurrence + RQA

| signal | N | p50 wall s | peak RSS MB | predicted dense bytes | predicted peak with temporaries bytes | RR | DET |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 100 | 0.000303068 | 107.5 | 80000 | 240000 | 0.0596 | 0.649194 |
| logistic | 200 | 0.000511627 | 108.3 | 320000 | 960000 | 0.05475 | 0.669347 |
| logistic | 350 | 0.0016175 | 110.4 | 980000 | 2940000 | 0.0527184 | 0.646038 |
| cml_flat | 100 | 0.000325699 | 107.9 | 80000 | 240000 | 0.0596 | 0.153226 |
| cml_flat | 200 | 0.000523268 | 108.8 | 320000 | 960000 | 0.05475 | 0.156784 |
| cml_flat | 350 | 0.00151629 | 110.6 | 980000 | 2940000 | 0.0527184 | 0.176817 |
