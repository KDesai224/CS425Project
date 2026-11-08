"""Step 4: produce rewrites from one system on one test set.

Systems
  raw        the clause unchanged (no simplification): the comparison floor
  zeroshot   base Gemma with the instruction prompt only
  fewshot    base Gemma with the prompt plus a few training pairs as examples
  finetuned  Gemma with a trained LoRA adapter (pass --adapter)

Test sets
  --testset simplegal   data/processed/simplegal_test.jsonl (has reference rewrites)
  --testset pipeline    data/processed/pipeline_test.jsonl  (CUAD / ContractNLI evidence)
  --input FILE.jsonl    any file with "id" and "source" fields, for example the
                        clauses found by the DeBERTa extractor (end-to-end test)

Output: outputs/predictions/<testset>/<system>.jsonl, one line per item with a
"prediction" field added. Decoding is greedy for every system.
"""
import argparse
from pathlib import Path

from jargonaut.common import OUTPUTS, PROCESSED, generate, load_config, load_model, pick_shots, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, choices=["raw", "zeroshot", "fewshot", "finetuned"])
    parser.add_argument("--config", help="model config; not needed for --system raw")
    parser.add_argument("--adapter", help="adapter directory, e.g. outputs/adapters/<run>/best")
    parser.add_argument("--testset", choices=["simplegal", "pipeline"], default="simplegal")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="simplegal only")
    parser.add_argument("--input", help="custom JSONL file; overrides --testset")
    args = parser.parse_args()

    if args.input:
        testset, path = Path(args.input).stem, Path(args.input)
    elif args.testset == "simplegal":
        testset = "simplegal" if args.split == "test" else "simplegal_val"
        path = PROCESSED / f"simplegal_{args.split}.jsonl"
    else:
        testset, path = "pipeline", PROCESSED / "pipeline_test.jsonl"
    rows = read_jsonl(path)
    sources = [r["source"] for r in rows]

    if args.system == "raw":
        name, predictions = "raw", sources
    else:
        if not args.config:
            parser.error("--config is required for this system")
        cfg = load_config(args.config)
        shots = ()
        if args.system == "finetuned":
            if not args.adapter:
                parser.error("--adapter is required for --system finetuned")
            adapter = Path(args.adapter)
            # best/ is named after its run; epoch-N/ keeps the epoch so checkpoints can be compared.
            if adapter.name == "best":
                name = f"ft-{adapter.parent.name}"
            elif adapter.name.startswith("epoch-"):
                name = f"ft-{adapter.parent.name}-{adapter.name}"
            else:
                name = f"ft-{adapter.name}"
            model, tokenizer = load_model(cfg, adapter)
        else:
            name = f"{args.system}-{cfg['name']}"
            model, tokenizer = load_model(cfg)
            if args.system == "fewshot":
                shots = pick_shots(read_jsonl(PROCESSED / "simplegal_train.jsonl"), cfg["generate"]["few_shot_k"])
        predictions = generate(model, tokenizer, sources, cfg, shots)

    out = OUTPUTS / "predictions" / testset / f"{name}.jsonl"
    write_jsonl(out, [{**row, "system": name, "prediction": pred} for row, pred in zip(rows, predictions)])
    print(f"Wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
