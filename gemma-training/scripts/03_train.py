"""Step 3: fine-tune Gemma on SimpLegal-Pref with LoRA (QLoRA when load_in_4bit is true).

Input : data/processed/simplegal_{train,val}.jsonl and a config from configs/
        (--train-file replaces the train split, e.g. with simplegal_train_aug.jsonl;
        validation always stays the human-only split)
Output: outputs/adapters/<run>/epoch-N/   LoRA adapter saved at every validation point
                                          (epoch-0.5, epoch-1, ... with --evals-per-epoch 2)
        outputs/adapters/<run>/best/      copy of the selected checkpoint
        outputs/adapters/<run>/history.json  validation scores of the base model and every checkpoint

How a checkpoint is selected: before training, the base model rewrites the
validation clauses (this is the zero-shot system), and so does every checkpoint.
Each set of rewrites is scored with SARI, number recall and NLI coverage. A
checkpoint qualifies if its number recall and NLI coverage are no more than 0.05
below zero-shot's, so one that scores well only by leaving things out is not
picked (SARI rewards that, because the Basic references are lossy themselves).
The qualifying checkpoint with the highest SARI is selected. If none qualifies,
the one with the highest NLI coverage is selected and a warning is printed.

The run name is <name>-r<rank>-lr<lr>-s<seed>; --rank, --lr and --seed override
the config, which is how the hyperparameter sweep and the seed repeats are run.
--epochs also overrides the config and adds -e<epochs> after the learning rate.
A --train-file other than simplegal_train.jsonl adds a suffix from its name
(simplegal_train_aug.jsonl -> "-aug"), so it never overwrites a human-only run.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from jargonaut import metrics
from jargonaut.common import (
    LORA_TARGETS, OUTPUTS, PROCESSED, build_messages, generate, load_config, load_model,
    read_jsonl, render_prompt,
)

# How far below zero-shot a checkpoint's number recall and NLI coverage may fall and still be selected.
FAITHFULNESS_MARGIN = 0.05


def encode(tokenizer, row, max_len):
    """Tokenise one pair. The loss is computed on the rewrite only: every prompt
    token gets the label -100, which the loss function ignores."""
    messages = build_messages(row["source"])
    prompt = render_prompt(tokenizer, messages)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["reference"]}], tokenize=False, enable_thinking=False
    )
    if not full.startswith(prompt):
        raise ValueError("Chat template changed: the training text no longer starts with the inference prompt.")
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) > max_len:
        return None
    labels = [-100 if start < len(prompt) else tok for tok, (start, _) in zip(enc["input_ids"], enc["offset_mapping"])]
    return {"input_ids": enc["input_ids"], "labels": labels}


def make_collator(pad_id):
    def collate(batch):
        width = max(len(ex["input_ids"]) for ex in batch)
        pad = lambda seq, value: seq + [value] * (width - len(seq))
        return {
            "input_ids": torch.tensor([pad(ex["input_ids"], pad_id) for ex in batch]),
            "attention_mask": torch.tensor([pad([1] * len(ex["input_ids"]), 0) for ex in batch]),
            "labels": torch.tensor([pad(ex["labels"], -100) for ex in batch]),
        }
    return collate


def mean(values):
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def validate(model, tokenizer, val_rows, cfg, nli):
    """Rewrite the validation clauses and score them. Items a metric does not apply to
    (no numbers in the source, a rewrite too short to split) are left out of its mean."""
    sources = [r["source"] for r in val_rows]
    outputs = generate(model, tokenizer, sources, cfg)
    return {
        "sari": mean([metrics.sari(r["source"], o, [r["reference"]]) for r, o in zip(val_rows, outputs)]),
        "fkgl": mean(map(metrics.fkgl, outputs)),
        "compression": mean([metrics.compression(s, o) for s, o in zip(sources, outputs)]),
        "number_recall": mean([metrics.number_recall(s, o) for s, o in zip(sources, outputs)]),
        "nli_coverage": mean([nli.score(s, o)["nli_coverage"] for s, o in zip(sources, outputs)]),
    }


class Validation(TrainerCallback):
    """At each validation point: save the adapter, rewrite the validation set and score it.

    `points` maps an optimiser step to the epoch it completes, e.g. {9: 0.5, 18: 1.0}."""

    def __init__(self, score, points, run_dir):
        self.score, self.points, self.run_dir = score, points, run_dir
        self.history = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.points:
            control.should_evaluate = True

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        epoch = self.points[state.global_step]
        model.save_pretrained(self.run_dir / f"epoch-{epoch:g}")
        scores = {"epoch": epoch, **self.score(model), "eval_loss": kwargs["metrics"].get("eval_loss")}
        self.history.append(scores)
        print(f"[validation] {scores}")


def select_best(history, zero_shot):
    """Highest SARI among checkpoints whose number recall and NLI coverage stay close to zero-shot's."""
    value = lambda h, key: 1.0 if h[key] is None else h[key]
    floors = {key: value(zero_shot, key) - FAITHFULNESS_MARGIN for key in ("number_recall", "nli_coverage")}
    qualified = [h for h in history if all(value(h, key) >= floor for key, floor in floors.items())]
    if not qualified:
        print(f"Warning: no checkpoint reached the zero-shot floors {floors}; selecting the highest NLI coverage.")
        return max(history, key=lambda h: value(h, "nli_coverage"))
    return max(qualified, key=lambda h: h["sari"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rank", type=int, help="LoRA rank; alpha is set to twice the rank")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evals-per-epoch", type=int, default=1,
                        help="validation points per epoch; 2 also saves and scores a checkpoint halfway through")
    parser.add_argument("--train-file", type=Path, default=PROCESSED / "simplegal_train.jsonl",
                        help="training pairs, e.g. data/processed/simplegal_train_aug.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg, lora_cfg = cfg["train"], cfg["lora"]
    if args.rank:
        lora_cfg["r"], lora_cfg["alpha"] = args.rank, 2 * args.rank
    if args.lr:
        train_cfg["lr"] = args.lr
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    stem = args.train_file.stem
    suffix = "" if stem == "simplegal_train" else "-" + stem.removeprefix("simplegal_train_")
    epochs = f"-e{train_cfg['epochs']}" if args.epochs else ""
    run_dir = OUTPUTS / "adapters" / f"{cfg['name']}-r{lora_cfg['r']}-lr{train_cfg['lr']:g}{epochs}-s{train_cfg['seed']}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_cfg["seed"])

    model, tokenizer = load_model(cfg)
    val_rows = read_jsonl(PROCESSED / "simplegal_val.jsonl")
    datasets = {}
    for split, path in (("train", args.train_file), ("val", PROCESSED / "simplegal_val.jsonl")):
        rows = read_jsonl(path)
        encoded = [e for e in (encode(tokenizer, r, train_cfg["max_seq_len"]) for r in rows) if e]
        print(f"{split}: {len(encoded)} examples ({len(rows) - len(encoded)} dropped as longer than max_seq_len)")
        datasets[split] = encoded

    # The base model is scored before LoRA is attached; its scores set the selection floors.
    nli = metrics.NLIScorer()
    score = lambda m: validate(m, tokenizer, val_rows, cfg, nli)
    zero_shot = score(model)
    print(f"[validation] zero-shot {zero_shot}")

    # Gradient checkpointing trades speed for memory by recomputing activations
    # during the backward pass. The cache is only useful for generation.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(datasets["train"]) / (train_cfg["batch_size"] * train_cfg["grad_accum"]))
    total_steps = steps_per_epoch * train_cfg["epochs"]
    n = args.evals_per_epoch
    points = {round(k * steps_per_epoch / n): k / n for k in range(1, train_cfg["epochs"] * n + 1)}
    validation = Validation(score, points, run_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(run_dir / "trainer"),
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=train_cfg["grad_accum"],
            learning_rate=train_cfg["lr"],
            lr_scheduler_type="cosine",
            warmup_steps=int(train_cfg["warmup_ratio"] * total_steps),
            logging_steps=5,
            eval_strategy="no",  # the Validation callback asks for evaluation at its own points
            save_strategy="no",
            bf16=torch.cuda.is_available(),
            report_to="none",
            seed=train_cfg["seed"],
        ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=make_collator(tokenizer.pad_token_id),
        callbacks=[validation],
    )
    trainer.train()

    best = select_best(validation.history, zero_shot)
    shutil.copytree(run_dir / f"epoch-{best['epoch']:g}", run_dir / "best", dirs_exist_ok=True)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "train_file": args.train_file.name, "zero_shot": zero_shot,
                   "best_epoch": best["epoch"], "epochs": validation.history}, f, indent=2)
    print(f"Selected epoch {best['epoch']:g} -> {run_dir / 'best'}")


if __name__ == "__main__":
    main()
