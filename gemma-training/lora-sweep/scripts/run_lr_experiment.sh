#!/usr/bin/env bash
# The whole experiment, run inside the container:
#   docker compose run --rm lora bash scripts/run_lr_experiment.sh
# Run 6's settings (886 synthetic pairs, rank 16, alpha 32, 2 epochs, batch 8, seed 42)
# with the learning rate halved to 1e-4, and a validation point every quarter epoch
# instead of every half. About 40 minutes on the RTX 3090.
set -euo pipefail
mkdir -p outputs/logs
RUN=e4b-r16-lr0.0001-e2-s42-syn

echo "== 1/4 train ($RUN)"
python scripts/03_train.py --config configs/e4b.yaml --lr 1e-4 --epochs 2 --evals-per-epoch 4 \
  --train-file data/processed/simplegal_train_syn.jsonl 2>&1 | tee "outputs/logs/train_${RUN}.log"

echo "== 2/4 rewrite the 20 validation clauses with the selected checkpoint"
python scripts/04_generate.py --system finetuned --config configs/e4b.yaml \
  --adapter "outputs/adapters/${RUN}/best" --testset simplegal --split val 2>&1 | tee "outputs/logs/generate_val_${RUN}.log"

echo "== 3/4 score it next to zero-shot and Run 6 (their validation rewrites are copied from Grpproj)"
python scripts/05_evaluate.py --testset simplegal_val --baseline zeroshot-e4b 2>&1 | tee outputs/logs/evaluate_val.log

echo "== 4/4 comparison"
python scripts/compare.py | tee outputs/compare.md
