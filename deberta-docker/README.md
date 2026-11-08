# DeBERTa evidence extractor (local Docker)

A copy of `Notebooks/deBERTa_training.ipynb` from the `deBERTa` branch, set up to
run on this machine's GPU in Docker instead of Colab. It trains
`deepset/deberta-v3-large-squad2` on CUAD, tunes the "insufficient evidence"
threshold on validation, scores test, and exports predictions for Gemma.

## Needs

- Docker Desktop running, with the WSL 2 backend and GPU support.
- `E:\CS425\EvidenceExtraction` next to this repo, holding the shared data from
  the team Drive. The container mounts it read-only at `/evidence`:

| File | Contents |
|---|---|
| `Data Prep.zip` | contracts, questions, splits and chunks (Theo's data prep) |
| `BM25/bm25_top10_{train,val,test}.jsonl.gz` | BM25's top 10 chunks for each question |
| `synthetic_qa.jsonl` (only if you set `USE_SYNTH = True`) | synthetic plain-language questions |

To read the data from somewhere else, set `EVIDENCE` to that folder first
(`$env:EVIDENCE="D:/path/to/folder"` in PowerShell).

## Run

From this folder:

```powershell
docker compose build
```

**Interactively.** Start Jupyter, open http://localhost:8889, open
`deBERTa_training.ipynb` and run all cells:

```powershell
docker compose up
```

**Headless.** Run the whole notebook and save the executed copy:

```powershell
docker compose run --rm deberta jupyter nbconvert --to notebook --execute deBERTa_training.ipynb --output-dir runs --output executed.ipynb --ExecutePreprocessor.timeout=-1
```

**Long run in the background, with a live log.** This runs the notebook as a
script, so progress and losses stream to `runs/train_log.txt`, and it keeps
going if you close the terminal:

```powershell
docker compose run -d --name deberta-train deberta bash -c "set -o pipefail; mkdir -p runs && jupyter nbconvert --to python deBERTa_training.ipynb --output-dir /tmp && ipython /tmp/deBERTa_training.py 2>&1 | tee runs/train_log.txt"
```

Follow it with `docker logs -f deberta-train`. When it has finished, remove the
container with `docker rm deberta-train`.

The first run downloads DeBERTa-v3-large (1.7 GB) into `.cache/`.

## Output

| Path | Contents |
|---|---|
| `runs/<run>/final/` | the trained model and tokenizer |
| `runs/<run>/test_preds.jsonl` | test predictions in the format handed to Gemma |
| `runs/<run>/threshold.json` | the tuned threshold and run settings |
| `data/data_prep_extracted/` | the unzipped `Data Prep.zip` |

`<run>` is named from the settings, e.g. `deberta-v3-large-squad2_all0_syn0_neg3`.

For that local run, git holds `runs/train_log.txt`, `test_preds.jsonl`,
`threshold.json` and `final/config.json`. The weights and tokenizer files in
`final/` (1.7 GB) stay local.

## Changes from the original notebook

| Change | Why |
|---|---|
| Data is read from the mounted `EvidenceExtraction` folder instead of Google Drive, the Drive mount is removed, and the zip is unzipped into `data/` | runs locally |
| `TRAIN_BS, GRAD_ACCUM, INFER_BS` = 8, 4, 16 (was 1, 32, 1) | uses the 3090's 24 GB. The effective batch is still 32, so training is otherwise the same. |
| `%pip install` cell removed | the image has the dependencies |
| `pd.set_option("display.max_columns", None, "display.width", 200)` added after the imports | the background run's log is plain text, where pandas would otherwise hide the middle columns of the by-category table |

The model, settings, splits and metrics are unchanged.
