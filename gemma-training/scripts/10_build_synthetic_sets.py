"""Step 10: build synthetic-only training sets, one limited to pairs that are short, easy and complete.

Round 2 showed that the human Basic references leave content out (mean NLI
coverage 0.42), while the synthetic references keep it (0.80) but are longer than
zero-shot's rewrites. Both sets here drop the human pairs. The strict set keeps
the short, easy end of the synthetic pairs that still keep the content (NLI
coverage 0.7), so fine-tuning is pushed toward rewrites that are both simple and
complete. Thresholds at zero-shot's validation means (compression 0.75, FKGL
8.0) leave only 6 of the 886 pairs, so the defaults are looser (0.85 and 9.0),
which keeps 147.

Input : data/processed/aug_accepted.jsonl (step 8)
Output: data/processed/aug_scored.jsonl                  every accepted pair with its nli_coverage, fkgl and
                                                         compression; reused on later runs (delete it to rescore)
        data/processed/simplegal_train_syn.jsonl         every synthetic pair, no human pairs
        data/processed/simplegal_train_syn_strict.jsonl  the pairs that pass all three thresholds

Validation and test stay the human SimpLegal splits. Train on either file with
03_train.py --train-file; the runs get the suffixes "-syn" and "-syn_strict".
"""
import argparse

from jargonaut import metrics
from jargonaut.common import PROCESSED, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-coverage", type=float, default=0.7)
    parser.add_argument("--max-compression", type=float, default=0.85, help="zero-shot's validation mean is 0.75")
    parser.add_argument("--max-fkgl", type=float, default=9.0, help="zero-shot's validation mean is 8.0")
    args = parser.parse_args()

    scored_path = PROCESSED / "aug_scored.jsonl"
    if scored_path.exists():
        scored = read_jsonl(scored_path)
    else:
        nli, scored = metrics.NLIScorer(), []
        for p in read_jsonl(PROCESSED / "aug_accepted.jsonl"):
            try:
                coverage = nli.score(p["source"], p["reference"])["nli_coverage"]
            except Exception:  # the NLI tokenizer cannot truncate a few very long source sentences
                coverage = None
            scored.append({"id": p["id"], "origin": p["origin"], "source": p["source"], "reference": p["reference"],
                           "nli_coverage": coverage, "fkgl": metrics.fkgl(p["reference"]),
                           "compression": metrics.compression(p["source"], p["reference"])})
        write_jsonl(scored_path, scored)

    checks = {
        f"nli_coverage >= {args.min_coverage:g}": lambda p: p["nli_coverage"] is not None and p["nli_coverage"] >= args.min_coverage,
        f"compression <= {args.max_compression:g}": lambda p: p["compression"] <= args.max_compression,
        f"fkgl <= {args.max_fkgl:g}": lambda p: p["fkgl"] is not None and p["fkgl"] <= args.max_fkgl,
    }
    pair = lambda p: {k: p[k] for k in ("id", "source", "reference", "origin")}
    strict = [pair(p) for p in scored if all(check(p) for check in checks.values())]
    write_jsonl(PROCESSED / "simplegal_train_syn.jsonl", [pair(p) for p in scored])
    write_jsonl(PROCESSED / "simplegal_train_syn_strict.jsonl", strict)

    print(f"{len(scored)} synthetic pairs ({sum(p['nli_coverage'] is None for p in scored)} could not be NLI-scored)")
    for name, check in checks.items():
        print(f"  {name:22s} {sum(map(check, scored))}")
    print(f"strict set (all three): {len(strict)} pairs")


if __name__ == "__main__":
    main()
