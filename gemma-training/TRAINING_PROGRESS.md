# Training progress

Status on 7 November 2026. This covers every Gemma 4 E4B fine-tuning run so far
(README steps 4, 5 and 10, plus scripts 09 and 10) and its evaluation: Runs 1 to
6 (21 to 24 October), Run 6 on test (1 November), and a learning-rate check and a
review of other training data (6 November). Every number here comes from
`outputs/adapters/*/history.json`, `outputs/logs/` and `outputs/metrics/`, from
`lora-sweep/outputs/` for the learning-rate check, or was computed from those
files as described. The exception is the review of other training data, whose
one-off scripts were not saved; its write-up is `OTHER_TRAINING_DATA.md`.

## Summary

- Six LoRA runs on Gemma 4 E4B in three rounds (21 to 24 Oct), all with rank 16,
  lr 2e-4 and seed 42, and a seventh at lr 1e-4 (Round 4, 6 Nov). **No run
  beats zero-shot on both readability and faithfulness.**
- **Round 1 (21 to 22 Oct):** human pairs only, and human + synthetic pairs.
  - On test, the human-only model has the best SARI (60.1 against 47.8 for
    zero-shot) and reads 0.8 grades easier. But it keeps much less of each
    clause: NLI coverage 0.40 against 0.77.
  - The augmented model cannot be told apart from zero-shot on any metric.
  - Both runs overfit after epoch 2.
- **Round 2 (24 Oct):** removed the human pairs that drop numbers or
  negations, rebalanced the mix, cut training to 2 epochs, and required the
  chosen checkpoint to keep about as much as zero-shot. Neither run did.
- **The cause is the human references.** Even after cleaning, their NLI
  coverage is 0.42; the synthetic references score 0.80.
- **Round 3 (24 Oct):** synthetic pairs only. The all-synthetic model (Run 6)
  is the only one significantly more faithful than zero-shot on validation
  (NLI coverage +0.06). But it reads 0.8 grades harder and is 10% longer.
- Fine-tuning so far moves the model along one trade-off. Human-style data buys
  brevity by dropping content; synthetic data buys completeness at some cost in
  readability.
- **Run 6 on test (1 Nov):** the same picture as validation. It adds less
  unsupported content than zero-shot (NLI supported +0.06, significant) and keeps
  a little more (NLI coverage 0.84 against 0.77, not significant), but reads 1.34
  grades harder (FKGL 9.73 against 8.39).
- **Why it reads harder (6 Nov): word length.** Each run copies the syllables per
  word of its training targets. The synthetic targets average 1.63, against 1.51
  for zero-shot's rewrites. On test, Run 6 and zero-shot have the same sentence
  length.
- **Round 4 (6 Nov): lr 1e-4.** Halving the learning rate reached the same end
  point as Run 6, and its selected checkpoint lost Run 6's faithfulness gain.
  Run 6 stays.
- **Other training data (6 Nov):** no public dataset pairs contract clauses with
  complete, plain rewrites, and SimpLegal-Synth's test split contains our test
  set.
- The Round 1 systems and Run 6 have been run on test; Runs 3 to 5 and the
  Round 4 run have validation scores only. The 31B model has not been trained.

## All runs at a glance

Validation set (20 clauses), selected checkpoint of each run. Runs 1 and 2 were
selected with the Round 1 rule, and Runs 3 to 6 and the Round 4 run with the
Round 2 rule (see [Setup](#setup)). NLI scores for Runs 1 and 2 come from
re-scoring their checkpoints with `05_evaluate.py`. The Round 4 run was trained
in `lora-sweep/` (see [Round 4](#round-4-a-lower-learning-rate-6-nov)).

| Run | Training data | Pairs | Epochs | Selected | SARI | FKGL | Compression | Number recall | NLI coverage | Passes the zero-shot floors |
|---|---|---|---|---|---|---|---|---|---|---|
| zero-shot | (no training) | | | | 46.95 | 7.96 | 0.75 | 0.74 | 0.73 | sets them |
| 1 | human | 139 | 4 | epoch 4 | 61.34 | 7.87 | 0.50 | 0.39 | 0.32 | no |
| 2 | human + synthetic | 1,025 | 4 | epoch 2 | 45.32 | 8.18 | 0.78 | 0.71 | 0.78 | yes |
| 3 | clean human | 87 | 2 | epoch 1 | 53.54 | 10.31 | 0.61 | 0.32 | 0.38 | no |
| 4 | clean human x4 + synthetic | 1,234 | 2 | epoch 1.5 | 49.07 | 8.69 | 0.70 | 0.62 | 0.62 | no |
| 5 | synthetic, strict subset | 147 | 2 | epoch 2 | 41.42 | 9.72 | 0.82 | 0.75 | 0.73 | yes |
| 6 | synthetic, all | 886 | 2 | epoch 1 | 41.49 | 8.74 | 0.85 | 0.75 | 0.79 | yes |
| 6 at lr 1e-4 (Round 4) | synthetic, all | 886 | 2 | epoch 0.5 | 41.77 | 8.46 | 0.80 | 0.75 | 0.73 | yes |

The floors are 0.686 for number recall and 0.684 for NLI coverage (zero-shot
minus 0.05). Runs 2, 5 and 6 and the Round 4 run pass them, but none is simpler
than zero-shot: all four have a higher FKGL and a lower SARI.

## Setup

| Setting | Round 1 (Runs 1 and 2) | Rounds 2 and 3 (Runs 3 to 6) |
|---|---|---|
| Base model | `google/gemma-4-E4B-it` in 4-bit (NF4, double quantisation, bf16 compute) | same |
| Adapter | LoRA rank 16, alpha 32, dropout 0.05, on the q/k/v/o and gate/up/down projections of the text decoder; 34.9M of 7.98B parameters trainable (0.44%) | same |
| Optimisation | lr 2e-4, cosine schedule, 10% warmup, batch 2 x gradient accumulation 4 = 8, gradient checkpointing | same |
| Epochs | 4 | 2 |
| Validation points | end of each epoch | every half epoch |
| Selection | best SARI among epochs whose number recall is within 0.05 of the best epoch's | best SARI among checkpoints whose number recall and NLI coverage are no more than 0.05 below zero-shot's; if none qualifies, the highest NLI coverage, with a warning |
| Loss | rewrite tokens only; prompt tokens are masked | same |
| Maximum length | 2,048 tokens; no examples were dropped | same |
| Hardware | RTX 3090 (24 GB) in Docker | same |

## Data

| Split | Pairs | Contents |
|---|---|---|
| Validation | 20 | SimpLegal-Pref, human only |
| Test | 40 | SimpLegal-Pref, human only |

| Training file (`data/processed/`) | Pairs | Contents | Used in |
|---|---|---|---|
| `simplegal_train.jsonl` | 139 | SimpLegal-Pref `Original` -> `Basic` | Run 1 |
| `simplegal_train_aug.jsonl` | 1,025 | the 139 human pairs + 624 CUAD + 262 ContractNLI synthetic pairs (Claude-written, passed the step 8 filter) | Run 2 |
| `simplegal_train_clean.jsonl` | 87 | human pairs whose reference keeps every number and negation (step 9) | Run 3 |
| `simplegal_train_aug_clean.jsonl` | 1,234 | those 87 repeated four times + the 886 synthetic pairs (28% human) | Run 4 |
| `simplegal_train_syn_strict.jsonl` | 147 | synthetic pairs with NLI coverage of at least 0.7, compression of at most 0.85 and FKGL of at most 9 (step 10) | Run 5 |
| `simplegal_train_syn.jsonl` | 886 | every synthetic pair, no human ones (step 10) | Run 6 |

The human and synthetic targets are written very differently. Synthetic pairs
are 86% of the Round 1 augmented set.

| Origin | Pairs | Median source words | Median target words | Mean compression |
|---|---|---|---|---|
| Human (SimpLegal Basic) | 139 | 99 | 57 | 0.55 |
| Synthetic, CUAD | 624 | 115 | 96 | 0.85 |
| Synthetic, ContractNLI | 262 | 96 | 82 | 0.86 |

The human targets also leave content out. Measured with the project's own
number and negation checks (`jargonaut/metrics.py`):

| Split | Pairs containing numbers | Mean number recall of the reference | References passing the step 8 number + negation gate |
|---|---|---|---|
| Train | 74 | 0.56 | 87 of 139 |
| Validation | 11 | 0.48 | 11 of 20 |
| Test | 22 | 0.66 | 28 of 40 |

Every synthetic pair passed that gate, since it was a condition of acceptance.
How much of the source each reference keeps by NLI is in
[Round 2](#why-the-references-themselves).

## Round 1: first runs (21 to 22 Oct)

### Attempt 1: failed (21 Oct, 22:34)

`outputs/logs/train_e4b-r16-lr0.0002-s42.attempt1-failed.log`. It crashed on
the first training step because Triton could not find a C compiler in the slim
image. Adding `build-essential` to the `Dockerfile` fixed it.

### Run 1: `e4b-r16-lr0.0002-s42` (human pairs only)

Finished 21 Oct 2026 at 23:29: 491 s of training, 72 optimiser steps.
Log: `outputs/logs/train_e4b-r16-lr0.0002-s42.log`.

| Epoch | Last logged train loss | Val loss | Val SARI | Val FKGL | Val number recall |
|---|---|---|---|---|---|
| 1 | 0.80 | 0.861 | 57.13 | 7.22 | 0.415 |
| 2 | 0.55 | **0.830** | 60.01 | 8.35 | 0.388 |
| 3 | 0.25 | 0.960 | 60.61 | 7.81 | 0.388 |
| 4 | 0.15 | 1.013 | **61.34** | 7.87 | 0.388 |

**Selected epoch 4.** All four epochs were inside the number-recall floor, so
SARI alone decided.

### Run 2: `e4b-r16-lr0.0002-s42-aug` (human + synthetic pairs)

22 Oct 2026, 07:21 to 08:06: 2,649 s of training.
Log: `outputs/logs/train_e4b-r16-lr0.0002-s42-aug.log`.

| Epoch | Last logged train loss | Val loss | Val SARI | Val FKGL | Val number recall |
|---|---|---|---|---|---|
| 1 | 0.55 | 0.944 | 44.73 | 9.01 | 0.618 |
| 2 | 0.39 | **0.922** | 45.32 | 8.18 | **0.709** |
| 3 | 0.22 | 1.077 | **54.08** | 7.72 | 0.618 |
| 4 | 0.09 | 1.376 | 53.20 | 7.13 | 0.618 |

**Selected epoch 2,** the only epoch inside the floor (0.709 - 0.05 = 0.659).
The 0.09 gap in number recall is a single clause.

### Validation with full scoring

These epochs were re-scored on validation with the NLI checks
(`outputs/metrics/simplegal_val/summary.md`):

| System | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|
| raw | 37.19 | 28.75 | 1.00 | 1.00 | 0.95 |
| zero-shot | 46.95 | 7.96 | 0.75 | 0.74 | 0.73 |
| human-only, epoch 2 | 60.01 | 8.35 | 0.49 | 0.39 | 0.24 |
| human-only, epoch 4 (selected) | 61.34 | 7.87 | 0.50 | 0.39 | 0.32 |
| augmented, epoch 2 (selected) | 45.32 | 8.18 | 0.78 | 0.71 | 0.78 |
| augmented, epoch 3 | 54.08 | 7.72 | 0.66 | 0.62 | 0.54 |
| augmented, epoch 4 | 53.20 | 7.13 | 0.65 | 0.62 | 0.49 |

### Test results (40 SimpLegal clauses, 22 Oct)

Each system was run on test once. Means with 95% bootstrap intervals, from
`outputs/metrics/simplegal/summary.md`:

| System | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|
| raw | 37.27 [36.67, 37.97] | 25.44 [22.20, 29.25] | 1.00 | 1.00 | 0.95 [0.93, 0.97] |
| zero-shot | 47.76 [45.63, 50.10] | 8.39 [7.73, 9.08] | 0.79 [0.76, 0.83] | 0.77 [0.59, 0.91] | 0.77 [0.69, 0.84] |
| few-shot (3 pairs) | 49.74 [47.19, 52.54] | 8.31 [7.68, 8.94] | 0.66 [0.63, 0.70] | 0.77 [0.59, 0.92] | 0.72 [0.66, 0.78] |
| fine-tuned, human only | 60.05 [57.56, 62.66] | 7.62 [6.97, 8.34] | 0.54 [0.50, 0.60] | 0.66 [0.48, 0.83] | 0.40 [0.31, 0.48] |
| fine-tuned, augmented | 46.46 [43.83, 49.21] | 8.95 [8.29, 9.59] | 0.76 [0.71, 0.81] | 0.77 [0.59, 0.91] | 0.74 [0.66, 0.82] |

The other metrics (means):

| System | Number precision | Negation kept | NLI supported | NLI contradicted |
|---|---|---|---|---|
| raw | 1.00 | 1.00 | 0.95 | 0.01 |
| zero-shot | 0.94 | 0.90 | 0.90 | 0.04 |
| few-shot | 0.93 | 0.87 | 0.95 | 0.01 |
| fine-tuned, human only | 0.88 | 0.90 | 0.87 | 0.02 |
| fine-tuned, augmented | 0.97 | 1.00 | 0.91 | 0.03 |

#### Compared with zero-shot

The saved summary compares every system with `raw`, but fine-tuning has to beat
zero-shot. The table below gives paired differences against `zeroshot-e4b`,
computed from `outputs/metrics/simplegal/per_item.csv` with the same paired
bootstrap as `05_evaluate.py`. (The same calculation against `raw` reproduces
the saved summary exactly.) `*` means the 95% interval excludes zero.

| System | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|
| few-shot | +1.98* [+0.73, +3.35] | -0.08 [-0.62, +0.44] | -0.13* [-0.16, -0.10] | +0.00 [-0.05, +0.05] | -0.05 [-0.12, +0.03] |
| fine-tuned, human only | +12.29* [+9.42, +15.12] | -0.77* [-1.37, -0.16] | -0.25* [-0.28, -0.20] | -0.11 [-0.24, +0.02] | -0.37* [-0.45, -0.29] |
| fine-tuned, augmented | -1.30 [-4.09, +1.66] | +0.57 [-0.14, +1.24] | -0.03 [-0.08, +0.01] | +0.00 [-0.14, +0.14] | -0.03 [-0.13, +0.08] |

None of the augmented model's nine metrics differs significantly from
zero-shot's.

### What Round 1 showed

1. **The human-only model copies the references' habit of leaving things out.**
   Its test compression (0.54) matches the human targets (0.55). SARI rewards
   this because it is scored against those same lossy references, so a higher
   SARI here does not mean a better simplifier.
2. **The augmented model copies the synthetic style.** Synthetic pairs are 86%
   of its data and keep far more of each clause (compression 0.85). The result
   is as faithful as zero-shot but no simpler, so it adds nothing over prompting.
3. **Both runs overfit.** In both, validation loss is lowest at epoch 2 and then
   rises while training loss keeps falling. At epoch 4 the human-only run is at
   0.15 train against 1.01 validation; the augmented run is at 0.09 against
   1.38. Round 2 cut training to 2 epochs.
4. **Epoch selection is noisy.** Validation has 20 clauses, 11 of them with
   numbers, so a single clause moves number recall by 0.09. The augmented run
   picked epoch 2 over epoch 3 by that margin.
5. **The number-recall floor was relative to the best epoch, not to
   zero-shot.** Every human-only epoch kept about 0.39 of the numbers (zero-shot
   keeps 0.74), so the floor never ruled anything out. Round 2 anchored the
   floors to zero-shot and added NLI coverage.

## Round 2: cleaner targets and a stricter selection rule (24 Oct)

### What changed

- `scripts/09_clean_human_pairs.py` keeps the 87 human pairs whose reference
  passes step 8's number and negation checks (`simplegal_train_clean.jsonl`).
  It also writes `simplegal_train_aug_clean.jsonl`: those 87 repeated four times
  plus the 886 synthetic pairs, 1,234 pairs in all, 28% of them human (up from 14%).
- `scripts/03_train.py`:
  - scores the base model on validation before training (zero-shot);
  - adds NLI coverage to every validation point;
  - can validate every half epoch (`--evals-per-epoch 2`);
  - takes `--epochs`;
  - selects the best SARI only among checkpoints whose number recall and NLI
    coverage are no more than 0.05 below zero-shot's. If none qualifies, it
    picks the highest NLI coverage and prints a warning.
- Both runs: 2 epochs (down from 4), lr 2e-4, rank 16, seed 42, validated
  every half epoch.

The in-training zero-shot scores match the earlier validation summary exactly
(SARI 46.95, number recall 0.74, NLI coverage 0.73), so the new scoring agrees
with `05_evaluate.py`. The floors were therefore 0.686 for number recall and
0.684 for NLI coverage.

### Run 3: `e4b-r16-lr0.0002-e2-s42-clean` (87 clean human pairs)

315 s including validation. No checkpoint reached the floors.

| Epoch | Val loss | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|---|
| 0.5 | 1.093 | 52.49 | 7.50 | 0.28 | 0.28 | 0.09 |
| 1 (selected: highest coverage) | 0.908 | 53.54 | 10.31 | 0.61 | 0.32 | 0.38 |
| 1.5 | 0.870 | 55.58 | 8.88 | 0.58 | 0.43 | 0.33 |
| 2 | 0.863 | 57.40 | 8.55 | 0.49 | 0.43 | 0.31 |

### Run 4: `e4b-r16-lr0.0002-e2-s42-aug_clean` (clean human x4 + synthetic)

1,555 s including validation. No checkpoint reached the floors.

| Epoch | Val loss | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|---|
| 0.5 | 0.999 | 46.30 | 9.31 | 0.70 | 0.66 | 0.59 |
| 1 | 1.301 | 49.80 | 7.58 | 0.66 | 0.59 | 0.56 |
| 1.5 (selected: highest coverage) | 1.624 | 49.07 | 8.69 | 0.70 | 0.62 | 0.62 |
| 2 | 1.701 | 49.69 | 8.39 | 0.69 | 0.66 | 0.52 |

Runs 1 to 4 all sit on one trade-off (see [All runs at a glance](#all-runs-at-a-glance)):
the more human-style data, the higher the SARI and the less of the clause is kept.

### Why: the references themselves

NLI coverage of each reference against its own source, scored with the same
DeBERTa model:

| Reference set | Mean NLI coverage | Median | At least 0.7 |
|---|---|---|---|
| Human train, kept by step 9 (87) | 0.42 | 0.41 | 17 of 87 |
| Human train, removed by step 9 (52) | 0.35 | 0.30 | 8 of 52 |
| Validation references (20) | 0.30 | 0.35 | 1 of 20 |
| Synthetic (885 scored) | 0.80 | 0.84 | 621 of 885 |
| *Zero-shot rewrites of validation, for comparison* | *0.73* | | |

The number and negation filter removes the most visible losses, but most human
Basic rewrites also drop conditions and whole clauses. A model trained on them
learns to do the same. The synthetic rewrites keep content about as well as
zero-shot does, but they are longer (compression 0.85). Neither source teaches
"simpler and complete".

The test set was not used in this round.

### Issue found

`metrics.NLIScorer` raised a tokenizer truncation error on one synthetic pair
(a source "sentence" of 273 words) and stopped. The other 885 scored normally.
Long CUAD clauses in the pipeline set (step 6) could hit the same error in
`05_evaluate.py`. Not fixed yet.

## Round 3: synthetic pairs only (24 Oct)

### What changed

`scripts/10_build_synthetic_sets.py` scores every accepted synthetic pair with
the NLI model (scores cached in `data/processed/aug_scored.jsonl`) and writes two
training files with no human pairs:

- `simplegal_train_syn.jsonl`: all 886 synthetic pairs.
- `simplegal_train_syn_strict.jsonl`: the 147 pairs with NLI coverage of at
  least 0.7, compression of at most 0.85 and FKGL of at most 9.

The first plan was to keep only pairs at least as short and easy as zero-shot's
rewrites (compression 0.75, FKGL 8.0) and still complete. Only 6 of 886 pass
that, so the thresholds were loosened. Completeness costs length, and the
synthetic rewrites rarely achieve both.

| Targets | Pairs | NLI coverage | Compression | FKGL |
|---|---|---|---|---|
| All synthetic | 886 | 0.80 | 0.85 | 8.73 |
| Synthetic strict | 147 | 0.92 | 0.79 | 7.79 |
| *Zero-shot rewrites of validation, for comparison* | 20 | *0.73* | *0.75* | *7.96* |

Training settings are the same as Round 2: 2 epochs, lr 2e-4, rank 16, seed 42,
validated every half epoch, with the zero-shot floors.

### Run 5: `e4b-r16-lr0.0002-e2-s42-syn_strict` (147 pairs)

525 s including validation. Epoch 0.5 misses the number-recall floor; the rest
pass.

| Epoch | Val loss | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|---|
| 0.5 | 1.465 | 42.37 | 10.65 | 0.72 | 0.62 | 0.73 |
| 1 | 1.734 | 41.04 | 8.76 | 0.91 | 0.75 | 0.80 |
| 1.5 | 1.849 | 40.72 | 9.96 | 0.83 | 0.75 | 0.73 |
| 2 (selected) | 1.859 | 41.42 | 9.72 | 0.82 | 0.75 | 0.73 |

### Run 6: `e4b-r16-lr0.0002-e2-s42-syn` (all 886 synthetic pairs)

1,449 s including validation. Every checkpoint passes the floors.

| Epoch | Val loss | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|---|
| 0.5 | 1.988 | 41.07 | 8.66 | 0.87 | 0.75 | 0.75 |
| 1 (selected) | 1.988 | 41.49 | 8.74 | 0.85 | 0.75 | 0.79 |
| 1.5 | 2.314 | 40.61 | 8.51 | 0.83 | 0.75 | 0.79 |
| 2 | 2.355 | 40.78 | 8.50 | 0.85 | 0.75 | 0.81 |

Validation loss is high because the validation references are the lossy human
rewrites, which these models no longer imitate. Here it does not mean the
training failed.

### Compared with zero-shot on validation

The selected checkpoints were run through `04_generate.py` and `05_evaluate.py`
on validation; the scores match the in-training ones. Paired differences
against `zeroshot-e4b` (`*` = 95% interval excludes zero):

| System | SARI | FKGL | Compression | Number recall | NLI coverage | NLI supported | Negation kept |
|---|---|---|---|---|---|---|---|
| all synthetic | -5.45* [-7.26, -3.67] | +0.79* [+0.05, +1.53] | +0.10* [+0.04, +0.15] | +0.02 [+0.00, +0.05] | +0.06* [+0.00, +0.11] | +0.03 [-0.01, +0.08] | +0.20 [+0.00, +0.40] |
| synthetic strict | -5.52* [-7.80, -3.45] | +1.77* [+0.83, +2.67] | +0.07* [+0.01, +0.13] | +0.02 [+0.00, +0.05] | -0.00 [-0.10, +0.08] | +0.06* [+0.03, +0.09] | +0.20 [+0.00, +0.40] |

### What Round 3 showed

1. **Synthetic-only training makes the model more complete but less simple.**
   The all-synthetic model keeps more of each clause than zero-shot (NLI
   coverage 0.79 against 0.73). On validation it keeps every negation (1.00
   against 0.80) and invents no numbers (number precision 1.00 against 0.88),
   though it still drops some, like zero-shot (number recall 0.75 against
   0.74). But it reads 0.8 grades harder (8.7 against 8.0) and is 10% longer.
   The lower SARI is expected, since SARI is scored against the lossy human
   references. The coverage gain is only just significant on 20 clauses.
2. **The strict subset did not pass on its easier style.** Its targets average
   FKGL 7.8, yet the model's validation rewrites average 9.7. With 147 pairs
   drawn from CUAD and ContractNLI, the style did not carry over to SimpLegal
   paragraphs.
3. **SARI is a poor tie-breaker among faithful checkpoints.** In Run 5, epoch 1
   (coverage 0.80, FKGL 8.8) lost to epoch 2 (coverage 0.73, FKGL 9.7) on a
   SARI gap of 0.4, which is noise on 20 items.
4. **Across all six runs, fine-tuning moves the model along one trade-off.**
   Human-style data buys SARI and brevity by dropping content; synthetic data
   buys completeness at some cost in readability. No run is both simpler and
   more complete than zero-shot. The all-synthetic model is the only one
   significantly more faithful.

The test set was not used in this round.

## Run 6 on test (1 Nov)

Run 6 was chosen on validation and run once on the 40 test clauses on 1
November, with the same prompt and greedy decoding as the Round 1 systems
(`outputs/logs/eval_2026-11-01/`). Seeds 1 and 2 were not trained first.

| System | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|
| zero-shot | 47.76 [45.63, 50.10] | 8.39 [7.73, 9.08] | 0.79 [0.76, 0.83] | 0.77 [0.59, 0.91] | 0.77 [0.69, 0.84] |
| Run 6 (all synthetic) | 41.53 [40.20, 43.01] | 9.73 [9.08, 10.38] | 0.82 [0.79, 0.85] | 0.81 [0.65, 0.94] | 0.84 [0.78, 0.89] |

The other metrics (means): Run 6 has number precision 0.97, negation kept 1.00,
NLI supported 0.96 and NLI contradicted 0.02, against 0.94, 0.90, 0.90 and 0.04
for zero-shot.

Paired differences against `zeroshot-e4b`, computed from
`outputs/metrics/simplegal/per_item.csv` with the same paired bootstrap as
`05_evaluate.py` (`*` = 95% interval excludes zero):

| System | SARI | FKGL | Compression | Number recall | NLI coverage | NLI supported | Negation kept |
|---|---|---|---|---|---|---|---|
| Run 6 | -6.23* [-8.09, -4.66] | +1.34* [+0.61, +2.05] | +0.03 [-0.01, +0.07] | +0.05 [-0.03, +0.15] | +0.07 [-0.00, +0.15] | +0.06* [+0.03, +0.10] | +0.10 [+0.00, +0.20] |

- **The same picture as validation.** Run 6 adds less that the clause does not
  support and keeps every negation, but it reads 1.34 grades harder, is harder
  to read than zero-shot on 28 of the 40 clauses, and scores 6 SARI points lower.
- **The coverage gain is about the size it was on validation** (+0.07 against
  +0.06), but it is not significant on 40 clauses. Rewrites with NLI coverage
  below 0.5: 2 of 40 for Run 6, 7 for zero-shot.

## Why Run 6 reads harder: word length (6 Nov)

FKGL = 0.39 x words per sentence + 11.8 x syllables per word - 15.59, so 0.1 more
syllables per word adds 1.2 grades. Words, sentences and syllables are counted
with textstat, as in `jargonaut/metrics.py`, and averaged over clauses.

| Training targets | Pairs | Targets: syllables per word | Targets: words per sentence | Model on validation: syllables per word | Model on validation: words per sentence | Model on validation: FKGL |
|---|---|---|---|---|---|---|
| Human (Run 1) | 139 | 1.42 | 18.2 | 1.39 | 18.3 | 7.87 |
| Synthetic, strict subset (Run 5) | 147 | 1.57 | 12.5 | 1.58 | 17.1 | 9.72 |
| Synthetic, all (Run 6) | 886 | 1.63 | 12.9 | 1.60 | 13.9 | 8.74 |
| *Zero-shot, no training* | | | | *1.51* | *14.6* | *7.96* |

- **Each model takes on its targets' word length.** The human targets use short
  words and so does Run 1; the synthetic targets use longer words than zero-shot,
  and so do Runs 5 and 6.
- **Sentence length does not carry over the same way.** Run 5's targets average
  12.5 words per sentence, but its rewrites average 17.1.
- **On test, word length is the whole gap.** Run 6 averages 1.64 syllables per
  word against 1.52 for zero-shot, and both average 15.4 words per sentence:
  11.8 x 0.11 is about 1.3 of the 1.34-grade gap.
- **The long words are the clause's own legal terms.** Run 6's test rewrites use
  "provision" 7 times, "revocation" 6 times and "affiliate" 6 times, each taken
  from the source clause; zero-shot uses none of them.
- **No target set so far is both complete and plain.** The human targets use
  short words but drop content (NLI coverage 0.42); the synthetic ones keep
  content (0.80) in longer words than zero-shot's.

## Round 4: a lower learning rate (6 Nov)

Run 6's setup with the peak learning rate halved, trained in `lora-sweep/` with
this project's code copied unchanged and its data mounted read-only. The full
write-up is `lora-sweep/RESULTS.md`. Only the 20 validation clauses were used.

| Setting | Run 6 | Round 4 |
|---|---|---|
| Peak learning rate (cosine, 10% warmup) | 2e-4 | 1e-4 |
| Validation points | every half epoch | every quarter epoch |
| Everything else | as in [Setup](#setup), on the 886 synthetic pairs | same |

Selected checkpoints, scored with `05_evaluate.py`:

| System | SARI | FKGL | Compression | Number precision | Negation kept | NLI coverage |
|---|---|---|---|---|---|---|
| zero-shot | 46.95 | 7.96 | 0.75 | 0.88 | 0.80 | 0.73 |
| Run 6, lr 2e-4 (epoch 1) | 41.49 | 8.74 | 0.85 | 1.00 | 1.00 | 0.79 |
| lr 1e-4 (epoch 0.5) | 41.77 | 8.46 | 0.80 | 0.91 | 0.87 | 0.73 |

Paired differences of lr 1e-4 from Run 6: compression -0.05* [-0.08, -0.02].
FKGL -0.29 [-0.76, +0.20], NLI coverage -0.06 [-0.16, +0.02], negation kept
-0.13 [-0.33, +0.00] and number precision -0.09 [-0.22, +0.00] are not
significant.

Every checkpoint of the Round 4 run:

| Epoch | Val loss | SARI | FKGL | Compression | Number recall | NLI coverage |
|---|---|---|---|---|---|---|
| 0.25 | 1.59 | 40.17 | 18.08 | 0.87 | 0.75 | 0.80 |
| 0.5 (selected) | 1.87 | 41.77 | 8.46 | 0.80 | 0.75 | 0.73 |
| 0.75 | 1.97 | 41.06 | 8.04 | 0.82 | 0.75 | 0.75 |
| 1 | 2.02 | 41.27 | 8.53 | 0.86 | 0.75 | 0.79 |
| 1.25 | 2.29 | 40.57 | 8.38 | 0.83 | 0.75 | 0.78 |
| 1.5 | 2.20 | 40.45 | 8.23 | 0.83 | 0.75 | 0.81 |
| 1.75 | 2.24 | 40.64 | 8.75 | 0.83 | 0.75 | 0.80 |
| 2 | 2.25 | 40.57 | 8.79 | 0.82 | 0.75 | 0.81 |

### What Round 4 showed

1. **The switch to the synthetic style still happens in the first quarter
   epoch.** At the first check (about step 28 of 222) SARI is already 40.17, so
   no checkpoint of either run lies between zero-shot (46.95) and Run 6.
2. **Mid-switch, the model copies the clause.** At epoch 0.25, 12 of the 20
   rewrites have a word overlap of 0.60 or more with their clause, and the
   three clauses with the longest sentences come back at grades 68.3, 56.3 and
   39.0. That is why the mean grade is 18.08. By epoch 0.5 the highest is 10.8.
3. **Both runs end in the same place.** From epoch 1 on, every checkpoint of
   either run has FKGL 8.23 to 8.79, NLI coverage 0.78 to 0.81 and compression
   0.82 to 0.86, the targets' own grade (8.73) and compression (0.85). The
   learning rate changed the path, not the end point.
4. **The SARI tie-breaker picked the least faithful checkpoint** (epoch 0.5,
   coverage 0.73), by 0.5 SARI over epoch 1. Epoch 1.5 (FKGL 8.23, coverage
   0.81) looks better, but choosing it after seeing these scores would be
   choosing on noise, so the rule was left as set.

Run 6 stays the chosen model. The test set was not used in this round.

## Other training data (6 Nov)

Could an existing dataset give better targets than SimpLegal-Pref and the
synthetic pairs? No. The full review, with sources, is `OTHER_TRAINING_DATA.md`.

- **SimpLegal-Synth, test split: never use.** It contains 38 of our 40 test, 18
  of 20 validation and 132 of 139 training paragraphs.
- **SimpLegal-Synth, train split:** 59,373 preference pairs over 4,700
  paragraphs, with 2 near-duplicates of our paragraphs and 1 paragraph from a
  CUAD test contract. Only 20.3% of its preferred rewrites pass step 8's checks
  plus two added ones for chatbot leftovers and repeated sentences (16.2% of the
  rejected ones pass), and a sample of 150 passing rewrites averages NLI
  coverage 0.52. It is useful as clauses to rewrite, not as targets.
- **AllGenerationData (Basic):** the same 201 paragraphs as SimpLegal-Pref, and
  its "Gold" column is not the Basic reference this project uses.
- **SIMPLE-LAW (LREC 2026):** GPT-3.5 rewrites of mostly legislation, at grade
  15 and half the source length.
- **TL;DRLegal, TOS;DR and BillSum:** summaries, which leave content out by
  design.

Better targets have to be built: whole-clause rewrites in plainer words than
zero-shot's. `OTHER_TRAINING_DATA.md` lists ways to test this at Run 5's size
(about 150 pairs).

## Where the outputs are

| Path | Contents |
|---|---|
| `outputs/adapters/<run>/best/` | the selected checkpoint, a byte-identical copy of one `epoch-N/` |
| `outputs/adapters/<run>/epoch-N/` | the adapter at each validation point (140 MB each); Runs 3 to 6 also have `epoch-0.5/` and `epoch-1.5/` |
| `outputs/adapters/<run>/history.json` | validation scores and loss at each point; from Run 3 on, also the zero-shot scores |
| `outputs/logs/train_*.log` | training logs, one per run |
| `outputs/logs/eval_2026-10-22/` | generation and scoring logs, including the Round 3 validation runs (`val_syn*.log`, `val_eval_syn.log`) |
| `outputs/logs/eval_2026-11-01/` | Run 6's test generation and scoring logs (`test_syn.log`, `test_eval.log`, `test_eval_vs_zeroshot.log`) |
| `outputs/predictions/simplegal/` | test-set rewrites (Round 1 systems and Run 6) |
| `outputs/predictions/simplegal_val/` | validation rewrites, including Runs 5 and 6 |
| `outputs/metrics/simplegal/`, `simplegal_val/` | summary tables and per-item scores |
| `data/processed/simplegal_train_*.jsonl` | the training files listed under [Data](#data) |
| `data/processed/aug_scored.jsonl` | NLI coverage, FKGL and compression of every synthetic pair |
| `lora-sweep/` | Round 4: adapters for every checkpoint, validation rewrites and metrics, `outputs/compare.md` and `RESULTS.md` |
| `OTHER_TRAINING_DATA.md` | the review of other training data and of word length (6 Nov) |

## Not done yet

- **Seeds 1 and 2 of Run 6.** Run 6 was chosen and run on test on 1 November
  without seed repeats, so how much its faithfulness gain varies between seeds
  is unknown (about 50 GPU-minutes).
- The rest of the hyperparameter sweep (rank 8 and 32; lr 1e-4 was Round 4) and
  seed repeats for the other runs. Every run so far uses seed 42, so the spread
  across seeds is unknown. Cross-validation was skipped as well, since no run
  beat zero-shot.
- Training targets in plainer words, which the word-length analysis points to.
  `OTHER_TRAINING_DATA.md` ("What to do instead") lists four ways to build and
  compare them.
- An ablation of Run 3's schedule on all 139 human pairs, which would separate
  the effect of cleaning from the effect of training for 2 epochs.
- Fixing the `NLIScorer` truncation error (see Round 2).
- The 31B run (step 9).
- The pipeline test set (step 6) and human evaluation (step 7). The end-to-end
  test with the extractor (step 8) was done in the app instead:
  `../jargonaut-app` ran Run 6 on the 380 clauses DeBERTa accepted from
  the CUAD test PDFs (see its `FINDINGS.md`).

## Rule for later runs

Six systems have been run on test once: raw, zero-shot, few-shot, Run 1, Run 2
and Run 6 (1 Nov). Runs 3 to 5 and the Round 4 run have not. Choose settings and
checkpoints on validation only, and do not reselect existing ones using the test
numbers. For example, Run 2's epoch 3 looks attractive now, but choosing it
after seeing the test results would be tuning on test.
