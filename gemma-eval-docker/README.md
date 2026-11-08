# Gemma evaluation (local Docker)

A copy of `Notebooks/gemma_evaluation.ipynb` from the `gemma-eval` branch, set up
to run on this machine's GPU in Docker. It evaluates the fine-tuned adapter
(Run 6, `e4b-r16-lr0.0002-e2-s42-syn/best`) against zero-shot Gemma 4 E4B.

## Needs

- Docker Desktop running, with the WSL 2 backend and GPU support (as for Grpproj).
- `E:\CS425\Grpproj` next to this repo. The container reads its test and
  validation sets and its Hugging Face cache, which already holds Gemma 4 E4B and
  the NLI model, so nothing large is downloaded.
- The adapter in `grprepo/outputs/e4b-r16-lr0.0002-e2-s42-syn/best/`.

## Run

From this folder:

```powershell
docker compose build
```

**Interactively.** Start Jupyter, open http://localhost:8888, open
`gemma_evaluation.ipynb` and run all cells:

```powershell
docker compose up
```

**Headless.** Run the whole notebook and save the executed copy next to its results:

```powershell
docker compose run --rm gemma-eval jupyter nbconvert --to notebook --execute gemma_evaluation.ipynb --output-dir results/val --output executed.ipynb --ExecutePreprocessor.timeout=-1
```

The notebook evaluates the **validation** set (20 clauses) by default. Run 6 has
not been on test yet, and the test set should be used once, for the final
system. When you get there, add `-e EVAL_SPLIT=test` after `run --rm` and change
`results/val` to `results/test`.

To evaluate a different adapter, set `ADAPTER` to its folder, for example
`$env:ADAPTER="E:/CS425/Grpproj/outputs/adapters/e4b-r16-lr0.0002-s42/best"` in
PowerShell, then delete `results/<split>/predictions.csv` so it regenerates.

## Output

Everything goes to `results/<split>/`:

| File | Contents |
|---|---|
| `predictions.csv` | every system's rewrite of each clause. It is reused on the next run, so delete it to regenerate. |
| `gemma_eval_summary.csv` | each metric with its 95% bootstrap interval |
| `gemma_eval_per_example.csv` | all metrics for every clause |
| `gemma_eval_summary.png` | the three plots |
| `manual_review_sheet.csv` | 20 random clauses to rate by hand |

The validation run's files are committed in `results/val/`, with the executed
notebook as `executed.ipynb`. Because `predictions.csv` is reused, running the
notebook from a fresh clone re-scores those saved rewrites. Delete it first to
generate new ones.

## Changes from the original notebook

| Change | Why |
|---|---|
| `BASE_MODEL_ID` is `google/gemma-4-E4B-it` | the adapter was trained on E4B, not 31B |
| Prompts use Gemma's chat template with the training system prompt (from `Grpproj/jargonaut/common.py`), thinking off | the original used a plain-text prompt the adapter never saw in training |
| Prompts are not truncated, and `<bos>` is not added twice | truncation cuts the end of the prompt, which is where the model starts answering |
| Greedy decoding, up to 512 new tokens, batches of 4 sorted longest first | matches how Grpproj generated its validation rewrites, so the numbers are comparable |
| Paths point to the container mounts; the split comes from `EVAL_SPLIT`; outputs go to `results/<split>/` | so a test run can't reuse validation's cached predictions |
| NLI falls back to trimming both sides when one sentence is over 512 tokens | Grpproj hit this tokenizer error on long legal sentences |
| `!pip install` cell removed | the image has the dependencies |
