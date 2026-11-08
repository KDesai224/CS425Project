# Jargonaut project

CS425 Group G1T5 project: an evidence-grounded contract question answering and simplification prototype.

## Project folders

- `jargonaut-app/` — application, tests, and saved evaluation outputs. See its README for the system workflow and runtime requirements.
- `deberta-docker/` — DeBERTa training notebook and Docker setup, plus the local training run's log, threshold and test predictions in `runs/` (no weights).
- `gemma-training/` — Gemma 4 E4B LoRA fine-tuning: data-preparation, training and evaluation scripts, Docker setup, and the Runs 1–6 logs, metrics and predictions (see `TRAINING_PROGRESS.md`). `gemma-training/lora-sweep/` holds the Round 4 learning-rate check. Datasets and adapter weights are not included.
- `gemma-eval-docker/` — Gemma evaluation notebook and Docker setup, plus the saved validation run in `results/val/`.
- `docs/proposal/`, `docs/guidelines/`, `docs/report/`, `docs/presentation/`, `docs/evaluation/` — project source documents and review.
- `docs/test-data/` — fictional contract test set archive.
- `final-prep.md`, `final-report.md`, `final-presentation.md` — detailed finalization checklist and document edit briefs.

## Reproducibility note

The full application requires model weights, Hugging Face caches, and shared dataset directories that are not included in this checkout. Docker is configured for offline model loading. Review `jargonaut-app/README.md`, `deberta-docker/README.md`, `gemma-training/README.md`, and `gemma-eval-docker/README.md` for expected mounts and commands before attempting a full run.

## Current scope

The deployed pipeline routes questions among ten CUAD topics, retrieves passages with BM25, extracts a candidate span with DeBERTa, and attempts a Gemma rewrite with heuristic checks. See `docs/evaluation/REVIEW_FINDINGS.md` for verified metric definitions, current limitations, and corrections identified during review.
