# EXP-0005 summary

## Configuration

- batch: ``2``
- heads: ``4``
- head_dim: ``16``
- num_codewords: ``16``
- children_per_codeword: ``4``
- refinement_budget: ``4``
- seq_len: ``64``
- warmup: ``3``
- reps: ``10``

| method | median ms | mean ms | stdev ms |
|--------|----------:|--------:|---------:|
| sdpa | 0.033 | 0.034 | 0.001 |
| paper single-pass | 1.014 | 1.077 | 0.240 |
| acmpr passes=4 decay=0.5 | 1.171 | 1.275 | 0.326 |

acmpr vs paper max abs diff (attention output): 0.0000
