# EXP-0006 summary

## Configuration

- batch: `2`
- heads: `4`
- head_dim: `16`
- num_codewords: `16`
- children_per_codeword: `4`
- refinement_budget: `4`
- seq_len: `64`
- warmup: `3`
- reps: `10`

| method | median ms | mean ms | stdev ms |
|--------|----------:|--------:|---------:|
| sdpa | 0.049 | 0.052 | 0.005 |
| paper single-pass | 1.174 | 1.227 | 0.169 |
| hvaq entropy | 1.310 | 1.379 | 0.214 |
| hvaq linear | 1.208 | 1.320 | 0.345 |

## Attention output (vs paper)
- HVAQ-ENT max abs diff: 134217728.0000
- HVAQ-LIN max abs diff: 0.0000
