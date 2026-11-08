# LoRA learning-rate experiment (Model 2, Gemma 4 E4B)

> In this repo the experiment sits inside the Gemma training project, so
> "Grpproj" in these files means the parent folder, `gemma-training/`. The
> adapter weights are not committed; their `history.json` and
> `best/adapter_config.json` are.

Does a gentler learning rate keep more of zero-shot Gemma's readability while
keeping Run 6's faithfulness gain?

Run 6 (`e4b-r16-lr0.0002-e2-s42-syn`) jumped to the style of its synthetic
training pairs within its first ~55 steps. By the first checkpoint, its reading
grade had gone from 7.96 to 8.66 and its SARI from 46.9 to 41.1, and later
checkpoints barely moved. Halving the learning rate slows that jump, so the
checkpoint rule can pick a model in between.

## What changes and what stays

| Setting | Run 6 | This run |
|---|---|---|
| Learning rate (peak, cosine, 10% warmup) | 2e-4 | **1e-4** |
| Validation points | every half epoch | **every quarter epoch** |
| Everything else | Gemma 4 E4B-it in 4-bit; LoRA rank 16, alpha 32, dropout 0.05 on q/k/v/o/gate/up/down; 886 synthetic pairs; 2 epochs; batch 2 x 4; seed 42; loss on the rewrite only | same |
| Checkpoint rule | best SARI among checkpoints within 0.05 of zero-shot on number recall and NLI coverage | same |

The code in `jargonaut/` and `scripts/03_train.py`, `04_generate.py` and
`05_evaluate.py` is copied unchanged from `..` (the Gemma training project), so only the settings
differ from Run 6. The new files are `scripts/run_lr_experiment.sh`,
`scripts/compare.py`, `docker-compose.yml` and this README.

## Run

Needs Docker Desktop with GPU support and the `jargonaut:latest` image built by
the Gemma training project (`..`). The data and the model weights are mounted
from that project's `data/processed/` and `.cache/`, so nothing is downloaded.
Neither is in git: the mounts read `..` by default, so set `GRPPROJ` to the
folder that holds them if it is elsewhere. From this folder:

```powershell
docker compose run --rm lora bash scripts/run_lr_experiment.sh
```

It takes about 40 minutes on the RTX 3090 (the 2026-11-06 run took 33):

- model loading: 1 min
- 222 training steps: about 16 min
- 9 validation scorings (zero-shot plus 8 checkpoints): about 18 min
- final scoring: about 5 min

## Outputs

| Path | Contents |
|---|---|
| `outputs/adapters/e4b-r16-lr0.0001-e2-s42-syn/` | adapters `epoch-0.25` ... `epoch-2`, `best/`, and `history.json` |
| `outputs/predictions/simplegal_val/` | validation rewrites. The zero-shot, raw and Run 6 files are copied from Grpproj. |
| `outputs/metrics/simplegal_val/` | `summary.md`, `summary.json` and `per_item.csv` from `05_evaluate.py` |
| `outputs/compare.md` | checkpoint trajectories of both runs, and paired differences between the selected models |
| `outputs/inspect_epoch-0.25.md`, `outputs/inspect_epoch-0.5.md` | grade, length and word overlap for each clause at two checkpoints (`scripts/inspect_checkpoint.py`) |
| `RESULTS.md` | the full write-up of the results |
| `outputs/baselines/run6_history.json` | Run 6's checkpoint scores (copied) |
| `outputs/logs/` | logs of every step |

Only the 20 validation clauses are used. The 40 test clauses are left untouched,
so the run is not chosen on test. With 20 clauses, gaps under about 0.05 in NLI
coverage or half a reading grade are within noise.

## Results (run on 2026-11-06)

The halved learning rate does not beat Run 6. The checkpoint rule picked epoch
0.5. The full write-up is in [RESULTS.md](RESULTS.md): every checkpoint, all
paired differences, the per-clause evidence and the caveats. The raw tables are
in `outputs/compare.md`.

| Selected checkpoint | Reading grade | SARI | NLI coverage | Negation kept | Number precision | Compression |
|---|---|---|---|---|---|---|
| Zero-shot | 7.96 | 46.9 | 0.73 | 0.80 | 0.88 | 0.75 |
| Run 6, lr 2e-4 (epoch 1) | 8.74 | 41.5 | 0.79 | 1.00 | 1.00 | 0.85 |
| lr 1e-4 (epoch 0.5) | 8.46 | 41.8 | 0.73 | 0.87 | 0.91 | 0.80 |

- **Against Run 6:** only the shorter output is a significant difference
  (compression -0.05). The lower grade (-0.29) and the lower NLI coverage (-0.06)
  are within noise. The model gives back Run 6's faithfulness gain without a
  clear gain in readability.
- **The switch to the synthetic style still happens in the first quarter
  epoch.** At every checkpoint SARI is 40.2-41.8, against 46.9 for zero-shot.
- **Mid-switch, the model copies.** At epoch 0.25 several validation rewrites
  are near-copies of their clause, with word overlaps of 0.60-0.97 (see
  `outputs/inspect_epoch-0.25.md`). This is why that checkpoint's mean grade is
  18.1. By epoch 0.5 the highest grade is 10.8.
- **Same destination.** By epoch 2 both runs read at about grade 8.5-8.8, with
  NLI coverage 0.81. The learning rate changed the path, not the end point. The
  training targets (grade 8.73, compression 0.85) set the style.
