# Jargonaut: training and evaluating the simplifier (Stage 3)

This folder holds everything needed to fine-tune Gemma 4 to rewrite contract
clauses in plain English, and to measure whether the rewrites are easier to read
without changing the meaning. Everything runs inside one Docker container on a
single 24 GB GPU (RTX 3090).

It covers Stage 3 of the proposal only. BM25 retrieval and the DeBERTa extractor
are separate; step 8 shows how their output plugs in.

> **In this repo.** This folder is the team's Gemma project folder, which other
> files call `Grpproj`. Git holds the code, configs and Docker setup, the
> write-ups (`TRAINING_PROGRESS.md`, `OTHER_TRAINING_DATA.md`) and the run
> records in `outputs/`: training and evaluation logs, metrics, predictions, and
> each run's `history.json` and `best/adapter_config.json`. The datasets
> (`data/`, including the synthetic pairs from step 10), the model cache
> (`.cache/`) and the adapter weights are not in git. The Round 4
> learning-rate check is in `lora-sweep/`.

## What is in here

| Path | Purpose |
|---|---|
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | The container: Python 3.12, PyTorch with CUDA, pinned libraries |
| `configs/e4b.yaml` | Settings for Gemma 4 E4B (start here) |
| `configs/31b.yaml` | Settings for Gemma 4 31B with 4-bit QLoRA (stretch run) |
| `jargonaut/common.py` | The prompt, model loading and generation, shared by every script |
| `jargonaut/metrics.py` | SARI, Flesch-Kincaid, number/negation checks, NLI faithfulness, bootstrap |
| `scripts/01_prepare_simplegal.py` | Download SimpLegal-Pref, write train/val/test splits |
| `scripts/02_build_pipeline_set.py` | Build a second test set from CUAD and ContractNLI evidence |
| `scripts/03_train.py` | LoRA / QLoRA fine-tuning |
| `scripts/04_generate.py` | Produce rewrites from a system (raw, zero-shot, few-shot, fine-tuned) |
| `scripts/05_evaluate.py` | Score all systems, with confidence intervals |
| `scripts/06_human_eval.py` | Blind rating sheets and their scoring |
| `scripts/07_build_augmentation_set.py` | Sample extra clauses from CUAD / ContractNLI train splits for synthetic pairs |
| `scripts/08_filter_augmentation.py` | Check LLM-written rewrites and merge the good ones with the human pairs |
| `scripts/09_clean_human_pairs.py` | Drop human pairs whose reference loses a number or negation; build cleaned training files |
| `scripts/10_build_synthetic_sets.py` | Score synthetic pairs for NLI coverage; build synthetic-only training files |
| `TRAINING_PROGRESS.md` | Runs so far, their scores, and what they show |
| `OTHER_TRAINING_DATA.md` | The review of other training data and of word length (6 Nov) |
| `lora-sweep/` | Round 4: Run 6's setup at learning rate 1e-4, with its own README and `RESULTS.md` |

Folders created as you go: `data/` (datasets) and `.cache/` (model weights),
both git-ignored, and `outputs/` (adapters, predictions, metrics), whose adapter
weights are git-ignored.

## Step 0: set up Docker (once)

1. Start Docker Desktop and wait until it says the engine is running.
2. Check disk space. The image goes onto Docker's WSL disk, which is on `C:` by
   default and needs roughly 10 GB there. If `C:` is short, move it first:
   Docker Desktop > Settings > Resources > Advanced > Disk image location.
   Model weights do **not** go there; they are stored in `.cache/` in this folder
   (16 GB for E4B, 63 GB for 31B, 4 GB for the NLI model).
3. Build the image and confirm the container sees the GPU:

```powershell
docker compose build
docker compose run --rm jargonaut nvidia-smi
docker compose run --rm jargonaut python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Both checks should name the RTX 3090. Every later command has the form
`docker compose run --rm jargonaut python scripts/<script> ...`. Scripts are
mounted into the container, so edits take effect without rebuilding. Add
`--help` to any script to see its documentation.

## Step 1: prepare SimpLegal-Pref

```powershell
docker compose run --rm jargonaut python scripts/01_prepare_simplegal.py
```

Downloads `SimpLegalPref.xlsx` and keeps the `Original` -> `Basic` pairs. Of the
201 paragraphs, 2 have no Basic rewrite, leaving 199. Eight rewrites begin with
a chatbot preamble ("Here is the simplified text:"), which is removed. The pairs
are shuffled with a
fixed seed into **139 train / 20 validation / 40 test** and written to
`data/processed/simplegal_*.jsonl`.

- Validation is used to choose the epoch and hyperparameters.
- Test is used once per system, for the final numbers. Do not tune on it.
- The dataset has no contract identifier, so a split by source document is not
  possible; that is a limitation to state in the report.
- The Basic rewrites are themselves lossy. Measured with this project's number
  check, they keep about 56% of the dates and amounts in the source (the
  Intermediate and Skilled levels keep 61% and 64%). A model trained on them
  learns to drop detail too, so expect the fine-tuned model's `number_recall`
  and `nli_coverage` to fall below the zero-shot model's. Treat the reference
  rewrites' own scores as the ceiling the fine-tuned model can reach.

## Step 2: build the pipeline test set

```powershell
docker compose run --rm jargonaut python scripts/02_build_pipeline_set.py
```

SimpLegal paragraphs are not what the simplifier receives in the product; there
it gets evidence found by the extractor. This step samples 80 gold evidence
spans from the CUAD test contracts and 20 from the ContractNLI test NDAs, spread
across clause categories, and expands each span to its surrounding clause
(40 to about 300 words). Result: `data/processed/pipeline_test.jsonl`.

This set has no reference rewrites, so SARI cannot be computed on it; it is
scored on readability, faithfulness and human ratings. Passages that share text
with any SimpLegal paragraph are excluded (47 CUAD candidates were, mostly
boilerplate), so nothing seen in training appears here.

## Step 3: run the baselines

These need no training and tell you what fine-tuning has to beat.

```powershell
docker compose run --rm jargonaut python scripts/04_generate.py --system raw --testset simplegal
docker compose run --rm jargonaut python scripts/04_generate.py --system zeroshot --config configs/e4b.yaml --testset simplegal
docker compose run --rm jargonaut python scripts/04_generate.py --system fewshot  --config configs/e4b.yaml --testset simplegal
docker compose run --rm jargonaut python scripts/05_evaluate.py --testset simplegal
```

- `raw` returns the clause unchanged: worst readability, perfect faithfulness.
- `zeroshot` is base Gemma with the instruction prompt.
- `fewshot` adds three fixed training pairs to the prompt.

The first Gemma run downloads the model. All systems share one prompt
(`SYSTEM_PROMPT` in `jargonaut/common.py`) and greedy decoding, so differences
between them come from the model, not the prompt.

## Step 4: fine-tune

```powershell
docker compose run --rm jargonaut python scripts/03_train.py --config configs/e4b.yaml
```

What the script does:

1. Loads Gemma in 4-bit (NF4, double quantisation, bf16 compute).
2. Adds LoRA adapters to the attention and MLP projections of the text decoder
   only. Gemma 4's vision and audio encoders are left alone.
3. Builds each example with the same prompt used at inference and computes the
   loss on the rewrite tokens only.
4. Trains with a cosine schedule, warmup and gradient checkpointing.
5. Before training, rewrites the validation clauses with the base model
   (zero-shot). After every epoch, or every half epoch with
   `--evals-per-epoch 2`, saves the adapter as `epoch-N` and rewrites them
   again. Each set of rewrites is scored with SARI, number recall and NLI
   coverage.
6. Picks the checkpoint with the best SARI among those whose number recall and
   NLI coverage are no more than 0.05 below zero-shot's, and copies it to
   `outputs/adapters/<run>/best`. SARI alone would favour checkpoints that drop
   content, because the Basic references do. `history.json` in the same folder
   records the zero-shot scores and every checkpoint's scores and loss.

The run is named `<name>-r<rank>-lr<lr>-s<seed>`, for example `e4b-r16-lr0.0002-s42`.
`--epochs` overrides the config and adds `-e<epochs>`, as in `e4b-r16-lr0.0002-e2-s42`.

**Sweep and seeds.** With 139 examples a run is short, so try a few settings and
compare `history.json` files:

```powershell
docker compose run --rm jargonaut python scripts/03_train.py --config configs/e4b.yaml --rank 8
docker compose run --rm jargonaut python scripts/03_train.py --config configs/e4b.yaml --rank 32
docker compose run --rm jargonaut python scripts/03_train.py --config configs/e4b.yaml --lr 1e-4
```

Then repeat the best setting with `--seed 1` and `--seed 2` and report the
spread across the three seeds, since results on this little data vary by seed.

## Step 5: evaluate the fine-tuned model

```powershell
docker compose run --rm jargonaut python scripts/04_generate.py --system finetuned --config configs/e4b.yaml --adapter outputs/adapters/e4b-r16-lr0.0002-s42/best --testset simplegal
docker compose run --rm jargonaut python scripts/05_evaluate.py --testset simplegal
```

`05_evaluate.py` scores every prediction file in `outputs/predictions/simplegal/`
and writes `outputs/metrics/simplegal/summary.md`: one row per system, each
metric as a mean with a 95% bootstrap interval, then each system's difference
from `raw` with a `*` where the interval excludes zero. Use
`--baseline zeroshot-e4b` to test fine-tuning against zero-shot instead.

| Metric | Reads as |
|---|---|
| `sari` | Simplification quality against the reference; higher is better |
| `fkgl` | Flesch-Kincaid grade; lower is easier, target about 8 |
| `compression` | Output length / source length; far below 1 can mean dropped content |
| `number_recall` | Share of source dates and amounts kept; below 1 means one was lost |
| `number_precision` | Share of output numbers that are in the source; below 1 means one was invented |
| `negation_kept` | Share of negated clauses whose rewrite still has a negation |
| `nli_supported` | Source entails the rewrite; low means added content |
| `nli_contradicted` | Share of rewrite sentences the source contradicts; should be near 0 |
| `nli_coverage` | Rewrite entails the source; low means dropped conditions or exceptions |

The research question is answered by comparing `ft-...` with `raw`: readability
(`fkgl`) should improve significantly while the faithfulness metrics stay close.
Always read `fkgl` together with `compression` and `nli_coverage`, because a
rewrite can look simpler just by leaving things out.

The rule-based and NLI checks are approximations. `per_item.csv` lists every
item's scores; sort it by `number_recall` or `nli_contradicted` to find the
failures worth reading by hand for the error analysis.

## Step 6: repeat on the pipeline set

```powershell
docker compose run --rm jargonaut python scripts/04_generate.py --system raw --testset pipeline
docker compose run --rm jargonaut python scripts/04_generate.py --system zeroshot --config configs/e4b.yaml --testset pipeline
docker compose run --rm jargonaut python scripts/04_generate.py --system fewshot  --config configs/e4b.yaml --testset pipeline
docker compose run --rm jargonaut python scripts/04_generate.py --system finetuned --config configs/e4b.yaml --adapter outputs/adapters/e4b-r16-lr0.0002-s42/best --testset pipeline
docker compose run --rm jargonaut python scripts/05_evaluate.py --testset pipeline
```

## Step 7: human evaluation

```powershell
docker compose run --rm jargonaut python scripts/06_human_eval.py export --testset pipeline --n 50
```

Writes `outputs/human_eval/pipeline/rating_sheet.csv`: 50 clauses, each followed
by every system's rewrite in shuffled order with no system names. Give each
rater a copy. They fill `meaning_preserved` and `understandable` with 1 to 5 and
save it as `rating_<name>.csv` in the same folder. Keep `key.csv` away from
raters. Then:

```powershell
docker compose run --rm jargonaut python scripts/06_human_eval.py score --testset pipeline
```

prints each system's mean rating with an interval, and Krippendorff's alpha for
agreement between raters. Use at least two raters.

## Step 8: end-to-end with the extractor

Once the DeBERTa extractor exists, save the clauses it finds as a JSONL file
with `id` and `source` fields (use the same ids as `pipeline_test.jsonl`), then:

```powershell
docker compose run --rm jargonaut python scripts/04_generate.py --system finetuned --config configs/e4b.yaml --adapter outputs/adapters/<run>/best --input data/processed/extracted.jsonl
docker compose run --rm jargonaut python scripts/04_generate.py --system raw --input data/processed/extracted.jsonl
docker compose run --rm jargonaut python scripts/05_evaluate.py --testset extracted
```

Comparing these scores with the pipeline set (gold evidence) separates
simplification errors from extraction errors.

## Step 10 (optional): synthetic training pairs

139 human pairs is little data, so steps 7 and 8 add rewrites written by an LLM
(Claude Sonnet agents, prompted with the same `SYSTEM_PROMPT` plus four human
pairs as style examples). This is synthetic data: keep it in its own file, train
with and without it, and say so in the report.

```powershell
docker compose run --rm jargonaut python scripts/07_build_augmentation_set.py
```

Draws 1,150 clauses (805 CUAD, 345 ContractNLI) from the **train** splits only,
round-robin over clause category, rotating short / medium / long, at most 3 per
contract, and excludes anything overlapping SimpLegal-Pref or the pipeline test
set. Writes `data/processed/aug_candidates.jsonl` and 23 interleaved shards of
50 in `data/processed/aug_shards/`, so each shard holds every category.

An agent reads one shard and writes `data/processed/aug_rewrites/shard_NN.jsonl`
with one `{"id", "reference"}` line per clause. Then:

```powershell
docker compose run --rm jargonaut python scripts/08_filter_augmentation.py
```

Keeps a rewrite only if number recall and precision are both 1, a negation cue
survives, FKGL is at most 10, compression is between 0.35 and 1.1, and there is
no preamble or markup. The 15% overdraw is for this gate. Output:
`aug_accepted.jsonl`, `aug_rejected.jsonl` (with reasons) and
`simplegal_train_aug.jsonl`, which is the human train set plus the accepted pairs
with an `origin` field. To train with augmentation:

```powershell
docker compose run --rm jargonaut python scripts/03_train.py --config configs/e4b.yaml --train-file data/processed/simplegal_train_aug.jsonl
```

The run is saved as `e4b-r16-lr0.0002-s42-aug`, next to the human-only run.
Validation and test stay human-only.

**Cleaner training targets.** 52 of the 139 human Basic rewrites drop a number
or a negation from their source, and a model trained on them copies that.

```powershell
docker compose run --rm jargonaut python scripts/09_clean_human_pairs.py
```

keeps the 87 human pairs that pass step 8's number and negation checks
(`simplegal_train_clean.jsonl`). It also writes `simplegal_train_aug_clean.jsonl`:
those 87 repeated four times (`--human-repeat`) plus the 886 synthetic pairs, so
human pairs are 28% of the examples instead of 14%. Train on either file with
`--train-file`; the runs get the suffixes `-clean` and `-aug_clean`.

**Synthetic only.** Even the cleaned human rewrites keep little of their source
by NLI coverage (0.42, against 0.80 for the synthetic ones), so this step drops
the human pairs entirely:

```powershell
docker compose run --rm jargonaut python scripts/10_build_synthetic_sets.py
```

scores every accepted synthetic pair with the NLI model (about 3 minutes; the
scores are cached in `aug_scored.jsonl`) and writes `simplegal_train_syn.jsonl`
(all 886) and `simplegal_train_syn_strict.jsonl` (the 147 with NLI coverage of
at least 0.7, compression of at most 0.85 and FKGL of at most 9). The runs get
the suffixes `-syn` and `-syn_strict`.

## Step 9: the 31B run

Run this after the E4B loop works end to end. Use the same commands with
`--config configs/31b.yaml`.

- The first run downloads 63 GB of weights and quantises them while loading.
- Quantised, the model takes about 19 GB of the 24 GB card. Close browsers and
  other GPU applications first; Windows itself uses over 1 GB.
- `max_seq_len` is 1024 to fit; the script prints how many training examples
  that drops. If training still runs out of memory, lower it to 768.
- If loading is killed, WSL has hit its RAM limit. Create
  `%UserProfile%\.wslconfig` containing `[wsl2]` and `memory=48GB`, then run
  `wsl --shutdown` and restart Docker Desktop.

## Verification status

Run during setup, outside Docker on CPU: steps 1 and 2 on the real data, the
metrics (SARI matches the Hugging Face reference implementation; the NLI scorer
was run on real examples), prompt construction and loss masking against the
real Gemma 4 tokenizer, and the training, generation, evaluation and human-eval
scripts on a tiny randomly initialised Gemma 4 model.

Since then, the Docker build, 4-bit loading, and E4B training and evaluation
have been run on the GPU (October 2026); see `TRAINING_PROGRESS.md`. The 31B
run has not been tried yet.
