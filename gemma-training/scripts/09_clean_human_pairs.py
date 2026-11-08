"""Step 9: drop human training pairs whose reference loses a number or a negation.

The SimpLegal Basic rewrites keep only about 56% of the dates and amounts in their
source, and a model trained on them learns to leave detail out as well. This keeps
the human pairs whose reference passes the number and negation checks that every
synthetic pair passed in step 8. Number precision is not checked: in these
references its failures are list numbering ("1.", "2.") and phrases like "once a year".

Input : data/processed/simplegal_train.jsonl, data/processed/aug_accepted.jsonl
Output: data/processed/simplegal_train_clean.jsonl      the human pairs that pass
        data/processed/simplegal_train_aug_clean.jsonl  those pairs repeated --human-repeat
                                                        times, plus every accepted synthetic pair

Validation and test are not changed, so scores stay comparable with earlier runs.
Train on either file with 03_train.py --train-file; the runs get the suffixes
"-clean" and "-aug_clean".
"""
import argparse
from collections import Counter

from jargonaut import metrics
from jargonaut.common import PROCESSED, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human-repeat", type=int, default=4,
                        help="copies of each clean human pair in the augmented file, so the shorter human "
                             "style is not swamped by the synthetic pairs")
    args = parser.parse_args()

    human = read_jsonl(PROCESSED / "simplegal_train.jsonl")
    clean, reasons = [], Counter()
    for row in human:
        failed = [name for name, ok in (
            ("number_dropped", metrics.number_recall(row["source"], row["reference"]) in (None, 1.0)),
            ("negation_lost", metrics.negation_kept(row["source"], row["reference"]) != 0.0),
        ) if not ok]
        reasons.update(failed)
        if not failed:
            clean.append(row)

    synthetic = [{"id": p["id"], "source": p["source"], "reference": p["reference"], "origin": p["origin"]}
                 for p in read_jsonl(PROCESSED / "aug_accepted.jsonl")]
    repeated = [{**row, "origin": "simplegal"} for row in clean] * args.human_repeat
    write_jsonl(PROCESSED / "simplegal_train_clean.jsonl", clean)
    write_jsonl(PROCESSED / "simplegal_train_aug_clean.jsonl", repeated + synthetic)

    print(f"kept {len(clean)} of {len(human)} human pairs; failures (a pair can fail both): {dict(reasons)}")
    total = len(repeated) + len(synthetic)
    print(f"augmented file: {len(repeated)} human ({len(clean)} x {args.human_repeat}) + {len(synthetic)} synthetic "
          f"= {total} pairs, {100 * len(repeated) / total:.0f}% human")


if __name__ == "__main__":
    main()
