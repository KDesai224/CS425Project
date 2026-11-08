"""Per-clause look at one prediction file: reading grade, length, and how much of the
rewrite is copied from the clause. Used to explain the grade of 18.1 at epoch 0.25.

    python scripts/inspect_checkpoint.py outputs/predictions/simplegal_val/<system>.jsonl
"""
import difflib
import sys

from jargonaut import metrics
from jargonaut.common import read_jsonl


def main():
    rows = read_jsonl(sys.argv[1])
    print("| id | grade (clause) | grade (rewrite) | words (clause / rewrite) | share copied | rewrite starts |")
    print("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(metrics.fkgl(r["prediction"]) or 0)):
        src, out = r["source"], r["prediction"]
        copied = difflib.SequenceMatcher(None, src.split(), out.split()).ratio()
        print(f"| {r['id']} | {metrics.fkgl(src):.1f} | {metrics.fkgl(out):.1f} | {len(src.split())} / {len(out.split())} "
              f"| {copied:.2f} | {' '.join(out.split()[:14])} ... |")


if __name__ == "__main__":
    main()
