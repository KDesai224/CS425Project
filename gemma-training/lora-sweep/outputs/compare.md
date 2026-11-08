## Validation scores at each checkpoint

| run | epoch | SARI | FKGL | compression | number recall | NLI coverage | val loss |
|---|---|---|---|---|---|---|---|
| zero-shot | - | 46.95 | 7.96 | 0.75 | 0.74 | 0.73 | n/a |
| Run 6 (lr 2e-4) | 0.5 | 41.07 | 8.66 | 0.87 | 0.75 | 0.75 | 1.99 |
| Run 6 (lr 2e-4) | 1 (selected) | 41.49 | 8.74 | 0.85 | 0.75 | 0.79 | 1.99 |
| Run 6 (lr 2e-4) | 1.5 | 40.61 | 8.51 | 0.83 | 0.75 | 0.79 | 2.31 |
| Run 6 (lr 2e-4) | 2 | 40.78 | 8.50 | 0.85 | 0.75 | 0.81 | 2.35 |
| lr 1e-4 | 0.25 | 40.17 | 18.08 | 0.87 | 0.75 | 0.80 | 1.59 |
| lr 1e-4 | 0.5 (selected) | 41.77 | 8.46 | 0.80 | 0.75 | 0.73 | 1.87 |
| lr 1e-4 | 0.75 | 41.06 | 8.04 | 0.82 | 0.75 | 0.75 | 1.97 |
| lr 1e-4 | 1 | 41.27 | 8.53 | 0.86 | 0.75 | 0.79 | 2.02 |
| lr 1e-4 | 1.25 | 40.57 | 8.38 | 0.83 | 0.75 | 0.78 | 2.29 |
| lr 1e-4 | 1.5 | 40.45 | 8.23 | 0.83 | 0.75 | 0.81 | 2.20 |
| lr 1e-4 | 1.75 | 40.64 | 8.75 | 0.83 | 0.75 | 0.80 | 2.24 |
| lr 1e-4 | 2 | 40.57 | 8.79 | 0.82 | 0.75 | 0.81 | 2.25 |

Selection rule: best SARI among checkpoints with number recall >= 0.686 and NLI coverage >= 0.684 (zero-shot minus 0.05).

## Selected checkpoints, full scoring (means over the 20 validation clauses)

| system | sari | fkgl | compression | number_recall | number_precision | negation_kept | nli_supported | nli_contradicted | nli_coverage |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot | 46.95 | 7.96 | 0.75 | 0.74 | 0.88 | 0.80 | 0.92 | 0.03 | 0.73 |
| Run 6 (lr 2e-4) | 41.49 | 8.74 | 0.85 | 0.75 | 1.00 | 1.00 | 0.95 | 0.01 | 0.79 |
| lr 1e-4 | 41.77 | 8.46 | 0.80 | 0.75 | 0.91 | 0.87 | 0.94 | 0.01 | 0.73 |

Paired differences (95% bootstrap interval; * = excludes zero)

| comparison | sari | fkgl | compression | number_recall | number_precision | negation_kept | nli_supported | nli_contradicted | nli_coverage |
|---|---|---|---|---|---|---|---|---|---|
| lr 1e-4 minus Run 6 (lr 2e-4) | +0.27 [-0.39, +1.01] | -0.29 [-0.76, +0.20] | -0.05* [-0.08, -0.02] | +0.00 [+0.00, +0.00] | -0.09 [-0.22, +0.00] | -0.13 [-0.33, +0.00] | -0.01 [-0.06, +0.04] | +0.00 [-0.03, +0.03] | -0.06 [-0.16, +0.02] |
| lr 1e-4 minus zero-shot | -5.18* [-6.85, -3.56] | +0.50 [-0.34, +1.29] | +0.05 [-0.01, +0.10] | +0.02 [+0.00, +0.05] | +0.03 [-0.14, +0.19] | +0.07 [-0.20, +0.33] | +0.03 [-0.03, +0.08] | -0.02 [-0.05, +0.02] | -0.01 [-0.11, +0.08] |
| Run 6 (lr 2e-4) minus zero-shot | -5.45* [-7.26, -3.67] | +0.79* [+0.05, +1.53] | +0.10* [+0.04, +0.15] | +0.02 [+0.00, +0.05] | +0.12 [+0.00, +0.25] | +0.20 [+0.00, +0.40] | +0.03 [-0.01, +0.08] | -0.02 [-0.05, +0.02] | +0.06* [+0.00, +0.11] |
