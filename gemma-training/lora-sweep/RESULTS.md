# LoRA learning-rate experiment: results

Model 2, the Gemma 4 E4B rewriter. Run on 2026-11-06 on the RTX 3090. The
design, settings and run command are in [README.md](README.md), and every
number below comes from the files listed under [Files](#files).

## Outcome

Halving the learning rate from 2e-4 to 1e-4 did not give a better rewriter, so
**Run 6 stays the chosen model**.

- The checkpoint rule picked epoch 0.5. It reads about as easily as Run 6: grade
  8.46 against 8.74, a gap within noise.
- It loses Run 6's faithfulness edge over zero-shot:
  - Its NLI coverage is zero-shot's 0.73, against 0.79 for Run 6.
  - In 2 of the 15 clauses with a negation, its rewrite has none. Run 6's
    rewrites kept all 15.
- Against Run 6 directly, only its shorter rewrites are a significant
  difference.
- The 40 test clauses were not used.

## Question

Run 6 took on the style of its synthetic training rewrites before its first
checkpoint (half an epoch, about 55 steps). After that it barely moved. The hope
was that a gentler learning rate would slow the switch. The checkpoint rule could
then pick a model in between: easier to read than Run 6, like zero-shot, and
still keeping Run 6's faithfulness gain.

## Setup

| | Run 6 | This run |
|---|---|---|
| Peak learning rate (cosine, 10% warmup) | 2e-4 | **1e-4** |
| Validation checkpoints | every half epoch (4) | **every quarter epoch (8)** |
| Everything else | Gemma 4 E4B-it in 4-bit NF4; LoRA rank 16, alpha 32, dropout 0.05 on q/k/v/o/gate/up/down; 886 synthetic pairs (624 CUAD, 262 ContractNLI); 2 epochs = 222 steps; batch 2 x 4; seed 42; loss on the rewrite only | same |

- **Checkpoint rule**, the same as Run 6's: the highest SARI among checkpoints
  within 0.05 of zero-shot on number recall and NLI coverage. Here that means
  number recall of at least 0.686 and coverage of at least 0.684.
- **Data:** only the 20 SimpLegal-Pref validation clauses are scored.
- **Code:** copied unchanged from `..` (the Gemma training project), so only the two settings above
  differ from Run 6.
- **Time:** 33 minutes end to end (16:22 to 16:55), against an estimate of 40.
  - loading the model and scoring zero-shot: 3.5 min
  - 222 steps with 8 validation checks: 26.8 min
  - rewriting and scoring the selected checkpoint: 3 min

## Metrics

| Metric | Better | Meaning |
|---|---|---|
| Reading grade | lower | Flesch-Kincaid grade level of the rewrite |
| SARI | higher | 0-100: how well the words the rewrite adds, keeps and deletes match the SimpLegal-Pref reference rewrites |
| NLI coverage | higher | mean probability that the rewrite entails each clause sentence; low means dropped content |
| NLI supported | higher | mean probability that the clause entails each rewrite sentence; low means added content |
| NLI contradicted | lower | share of rewrite sentences that the clause contradicts |
| Negations kept | higher | share of clauses with a negation whose rewrite also has one; 15 of the 20 clauses have one |
| Number precision | higher | share of the rewrite's numbers that come from the clause |
| Number recall | higher | share of the clause's numbers kept in the rewrite |
| Compression | neither | rewrite length over clause length, in words; well below 1 can mean dropped content |

## Selected checkpoints

Full scoring with `05_evaluate.py`. Values are means over the 20 validation
clauses.

| Model | Grade | SARI | NLI coverage | NLI supported | NLI contradicted | Negations kept | Number precision | Number recall | Compression |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 7.96 | 46.95 | 0.73 | 0.92 | 0.03 | 0.80 | 0.88 | 0.74 | 0.75 |
| Run 6, lr 2e-4 (epoch 1) | 8.74 | 41.49 | 0.79 | 0.95 | 0.01 | 1.00 | 1.00 | 0.75 | 0.85 |
| lr 1e-4 (epoch 0.5) | 8.46 | 41.77 | 0.73 | 0.94 | 0.01 | 0.87 | 0.91 | 0.75 | 0.80 |

The table below gives paired differences, item by item over the same 20 clauses.

- The brackets are 95% paired bootstrap intervals (2,000 resamples, seed 0).
- \* marks an interval that excludes zero.
- The intervals are rounded, so one printed as starting at +0.00 can still
  exclude zero.

| Comparison | Grade | SARI | NLI coverage | Negations kept | Number precision | Compression |
|---|---|---|---|---|---|---|
| lr 1e-4 minus Run 6 | -0.29 [-0.76, +0.20] | +0.27 [-0.39, +1.01] | -0.06 [-0.16, +0.02] | -0.13 [-0.33, +0.00] | -0.09 [-0.22, +0.00] | **-0.05\*** [-0.08, -0.02] |
| lr 1e-4 minus zero-shot | +0.50 [-0.34, +1.29] | **-5.18\*** [-6.85, -3.56] | -0.01 [-0.11, +0.08] | +0.07 [-0.20, +0.33] | +0.03 [-0.14, +0.19] | +0.05 [-0.01, +0.10] |
| Run 6 minus zero-shot | **+0.79\*** [+0.05, +1.53] | **-5.45\*** [-7.26, -3.67] | **+0.06\*** [+0.00, +0.11] | +0.20 [+0.00, +0.40] | +0.12 [+0.00, +0.25] | **+0.10\*** [+0.04, +0.15] |

Number recall, NLI supported and NLI contradicted differ by at most 0.03 in every
comparison, and none of those gaps is significant. All nine columns are in
`outputs/compare.md`.

## Every checkpoint

These are the scores logged during training, on the same 20 clauses.

- One epoch is 111 steps, and warmup covers about the first 22.
- The selected checkpoint's scores here match its full scoring above.

| Run | Epoch | SARI | Grade | Compression | Number recall | NLI coverage | Val loss |
|---|---|---|---|---|---|---|---|
| Zero-shot | - | 46.95 | 7.96 | 0.75 | 0.74 | 0.73 | - |
| Run 6, lr 2e-4 | 0.5 | 41.07 | 8.66 | 0.87 | 0.75 | 0.75 | 1.99 |
| Run 6, lr 2e-4 | **1, selected** | 41.49 | 8.74 | 0.85 | 0.75 | 0.79 | 1.99 |
| Run 6, lr 2e-4 | 1.5 | 40.61 | 8.51 | 0.83 | 0.75 | 0.79 | 2.31 |
| Run 6, lr 2e-4 | 2 | 40.78 | 8.50 | 0.85 | 0.75 | 0.81 | 2.35 |
| lr 1e-4 | 0.25 | 40.17 | 18.08 | 0.87 | 0.75 | 0.80 | 1.59 |
| lr 1e-4 | **0.5, selected** | 41.77 | 8.46 | 0.80 | 0.75 | 0.73 | 1.87 |
| lr 1e-4 | 0.75 | 41.06 | 8.04 | 0.82 | 0.75 | 0.75 | 1.97 |
| lr 1e-4 | 1 | 41.27 | 8.53 | 0.86 | 0.75 | 0.79 | 2.02 |
| lr 1e-4 | 1.25 | 40.57 | 8.38 | 0.83 | 0.75 | 0.78 | 2.29 |
| lr 1e-4 | 1.5 | 40.45 | 8.23 | 0.83 | 0.75 | 0.81 | 2.20 |
| lr 1e-4 | 1.75 | 40.64 | 8.75 | 0.83 | 0.75 | 0.80 | 2.24 |
| lr 1e-4 | 2 | 40.57 | 8.79 | 0.82 | 0.75 | 0.81 | 2.25 |

## Findings

### 1. The switch to the synthetic style still happens in the first quarter epoch

The first checkpoint comes at about step 28, just after warmup ends. By then,
SARI had already fallen from 46.95 to 40.17. Every later checkpoint in either
run scores between 40.45 and 41.77. Halving the learning rate left no checkpoint
between zero-shot and Run 6.

### 2. Mid-switch, the model copies the clause

At epoch 0.25 the mean grade is 18.08, because many rewrites are close copies
that keep the clause's long sentences (`outputs/inspect_epoch-0.25.md`).
"Overlap" below is the difflib ratio of the two word sequences; 1.0 means
identical.

- **Overlap:** the median is 0.61, against 0.45 at epoch 0.5. Five of the 20
  rewrites have an overlap of 0.82 to 0.97, and 12 have 0.60 or more.
- **The three highest grades are 68.3, 56.3 and 39.0.** Each of those rewrites
  kept its clause's very long sentences. The clauses themselves grade 79.3, 60.2
  and 40.4, and the overlaps are 0.62, 0.75 and 0.92. simplegal-71, for example,
  came back as 261 words from 271.
- **Without those three clauses,** the mean grade is 11.6.
- **Coverage and length:** copying also explains this checkpoint's high NLI
  coverage (0.80) and compression (0.87), since a copy keeps everything.
- **By epoch 0.5** (`outputs/inspect_epoch-0.5.md`):
  - the overlap is 0.29 to 0.70
  - the highest grade is 10.8
  - the same three clauses read at grades 9.2, 7.7 and 7.8

### 3. Both runs end in the same place

From epoch 1 on, every checkpoint of either run falls in a narrow band:

| Metric | Range |
|---|---|
| Reading grade | 8.23 to 8.79 |
| NLI coverage | 0.78 to 0.81 |
| Compression | 0.82 to 0.86 |

The 886 training rewrites read at grade 8.73, with compression 0.85. The model
converges to its targets' style, so the learning rate changed the path but not
the end point.

### 4. Validation loss is no guide to rewrite quality here

In both runs, validation loss is lowest at the first checkpoint and rises after
it:

- lr 1e-4: from 1.59 to 2.25
- Run 6: from 1.99 to 2.35

Meanwhile, the rewrite scores barely move after epoch 0.5. The loss is measured
on the SimpLegal-Pref reference rewrites. Its rise suggests that the model
drifts from their style as it fits the synthetic one. SARI compares against the
same references, and it stays below zero-shot's throughout.

### 5. The SARI tie-breaker picked the least faithful checkpoint

All eight checkpoints passed the floors. The rule picked epoch 0.5, which has
the lowest NLI coverage of the eight (0.73).

- It leads the next best checkpoint (epoch 1) by only 0.5 SARI. The 95% interval
  for its SARI gap to Run 6 spans about ±0.7.
- Epoch 1.5 ties for the highest coverage in either run (0.81). Its grade (8.23)
  is lower than that of any other checkpoint with that coverage.
- But its gap to Run 6 (-0.51 grade, +0.02 coverage) is about the size of the
  noise.
- Its rewrites weren't kept, so it has no paired test and no negation or
  number-precision scores. Regenerating them would take about 5 min.
- Choosing it after seeing these scores would be choosing on noise.

The rule was left as set before the run.

## Conclusion

- Run 6 stays the chosen rewriter.
- The learning rate changes how fast the model reaches its end point, not where
  that end point is. The training rewrites decide where it ends: the model
  matches their grade and length.
- So the next lever is the training targets, not the LoRA settings.
  - Rank and alpha were not tested.
  - Alpha scales the update much as the learning rate does.
  - Rank is unlikely to change a style that the targets set.

## Caveats

- There are only 20 validation clauses, so gaps under about 0.05 in NLI coverage
  or half a reading grade are within noise.
- Each learning rate ran with one seed (42), so seed-to-seed variation is not
  measured.
- Only validation was used. Run 6's result on the 40 test clauses is still the
  only test result.
- The epoch-0.25 rewrites were regenerated after the run from the saved adapter.
  They reproduce the grade logged during training (18.08).

## Possible next steps (not run)

Train on targets with plainer words. In both Run 5 and Run 6, the model copied
its targets' word length.

- **Run 5's 147 strict pairs, for more steps:**
  - Their targets read at grade 7.79.
  - Run 5 trained on them for only about 38 steps.
  - It picked up its targets' word length but not their sentence length (17.1
    words per sentence, against 12.5).
  - Retraining would take about 15 to 20 min.
- **An untried filter of the synthetic pairs:** NLI coverage of at least 0.7 and
  at most 1.55 syllables per word. It leaves 150 pairs with a mean grade of 7.27.

## Files

| Path | Contents |
|---|---|
| `README.md` | design, settings and run command |
| `outputs/compare.md` | both runs' checkpoints, full scoring and paired differences, from `scripts/compare.py` |
| `outputs/inspect_epoch-0.25.md`, `outputs/inspect_epoch-0.5.md` | grade, length and overlap for each clause, from `scripts/inspect_checkpoint.py` |
| `outputs/metrics/simplegal_val/` | `summary.md`, `summary.json` and `per_item.csv`, from `05_evaluate.py` |
| `outputs/predictions/simplegal_val/` | rewrites from zero-shot, Run 6, and this run at epochs 0.5 (selected) and 0.25 |
| `outputs/adapters/e4b-r16-lr0.0001-e2-s42-syn/` | adapters for all 8 checkpoints, `best/` (epoch 0.5) and `history.json` |
| `outputs/logs/` | train, generate and evaluate logs |

To rerun, from this folder:

```powershell
# the whole experiment, about 35 min
docker compose run --rm lora bash scripts/run_lr_experiment.sh
# the per-clause table for one checkpoint
docker compose run --rm lora python scripts/04_generate.py --system finetuned --config configs/e4b.yaml --adapter outputs/adapters/e4b-r16-lr0.0001-e2-s42-syn/epoch-0.25 --testset simplegal --split val
docker compose run --rm lora bash -c "python scripts/inspect_checkpoint.py outputs/predictions/simplegal_val/ft-e4b-r16-lr0.0001-e2-s42-syn-epoch-0.25.jsonl > outputs/inspect_epoch-0.25.md"
```
