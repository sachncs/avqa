# EXP-0001 summary

## Configuration

- python: ``3.12.7``
- torch: ``2.10.0``
- platform: ``macOS-26.6-arm64-arm-64bit``
- cpu_count: ``6``
- warmup: ``5``
- reps: ``10``

| seq_len | sdpa median ms | avqa median ms | avqa/sdpa |
|--------:|---------------:|---------------:|-----------|
| 128 | 0.153 | 4.147 | 0.04 |
| 256 | 0.363 | 8.128 | 0.04 |
| 512 | 1.266 | 11.592 | 0.11 |
| 1024 | 3.983 | 22.215 | 0.18 |
