# EXP-0002 summary

## Configuration

- python: ``3.12.7``
- torch: ``2.10.0``
- platform: ``macOS-26.6-arm64-arm-64bit``
- cpu_count: ``6``
- warmup: ``5``
- reps: ``10``

| seq_len | sdpa median ms | avqa median ms | avqa/sdpa |
|--------:|---------------:|---------------:|-----------|
| 128 | 0.128 | 3.363 | 0.04 |
| 256 | 0.286 | 6.883 | 0.04 |
| 512 | 1.205 | 10.140 | 0.12 |
| 1024 | 3.140 | 19.618 | 0.16 |
