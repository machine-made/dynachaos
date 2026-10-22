# Rust hotspot profile

Generated: 2026-06-11T06:07:08Z
Command: `python benchmarks/rust_hotspot_profile.py rust_hotspot_profile.jsonc`

## Hardware and software

- CPU: AMD Ryzen 9 9900X 12-Core Processor
- Platform: Linux-7.0.0-22-generic-x86_64-with-glibc2.43
- Python: 3.13.13; NumPy: 2.4.2

## Measurements

| case | kind | backend | size | p50 wall s | peak RSS MB | notes |
|---|---|---|---:|---:|---:|---|
| streaming_rqa | streaming_rqa | python | 4000 | 0.728803 | 107.5 | RR=0.00032025; DET=0.0355872 |
| exact_pair_count_python | exact_pair_count | python | 4000 | 0.432006 | 107.2 | C_last=0.353946 |
| exact_pair_count_rust | exact_pair_count | rust | 12000 | 0.0206643 | 108.3 | C_last=0.411333 |
| comoving_logistic_python | comoving_logistic | python | 500 | 0.655868 | 107.3 | lambda_mean=-0.136657 |
| comoving_logistic_rust | comoving_logistic | rust | 500 | 0.0112032 | 106.9 | lambda_mean=-0.136657 |
| coupled_logistic_basin_python | coupled_logistic_basin | python | 32400 | 1.3147 | 81.2 | labels={'0': 180, '1': 16110, '2': 16110} |
| coupled_logistic_basin_rust | coupled_logistic_basin | rust | 129600 | 0.138839 | 82.4 | labels={'0': 360, '1': 64620, '2': 64620} |
