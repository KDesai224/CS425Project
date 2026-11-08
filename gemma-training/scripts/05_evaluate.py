"""Step 5: score every system's predictions on one test set.

Input : outputs/predictions/<testset>/*.jsonl (one file per system, from step 4)
Output: outputs/metrics/<testset>/summary.md     table of means with 95% bootstrap intervals
        outputs/metrics/<testset>/summary.json   the same numbers, plus paired differences
        outputs/metrics/<testset>/per_item.csv   every metric for every item and system

Metrics (see jargonaut/metrics.py)
  sari              simplification quality against the reference (simplegal only); higher is better
  fkgl              Flesch-Kincaid grade level; lower is easier to read (target: about 8)
  compression       output words / source words
  number_recall     share of source quantities kept; below 1 means a date or amount was lost
  number_precision  share of output quantities found in the source; below 1 means an invented number
  negation_kept     share of negated sources whose rewrite still contains a negation
  nli_supported     how strongly the source entails the rewrite; low means added content
  nli_contradicted  share of rewrite sentences the source contradicts; should be near 0
  nli_coverage      how strongly the rewrite entails the source; low means dropped content

Each system is also compared with --baseline item by item (paired bootstrap).
A difference is significant at the 5% level when its interval excludes zero.
"""
import argparse
import json

import pandas as pd

from jargonaut import metrics
from jargonaut.common import OUTPUTS, read_jsonl

METRICS = ["sari", "fkgl", "compression", "number_recall", "number_precision", "negation_kept",
           "nli_supported", "nli_contradicted", "nli_coverage"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", default="simplegal", help="folder name under outputs/predictions")
    parser.add_argument("--baseline", default="raw", help="system the others are compared against")
    parser.add_argument("--no-nli", action="store_true", help="skip the NLI metrics (no model download)")
    args = parser.parse_args()

    files = sorted((OUTPUTS / "predictions" / args.testset).glob("*.jsonl"))
    if not files:
        parser.error(f"no prediction files in outputs/predictions/{args.testset}")
    nli = None if args.no_nli else metrics.NLIScorer()

    records = []
    for file in files:
        for row in read_jsonl(file):
            source, output = row["source"], row["prediction"]
            record = {
                "system": file.stem, "id": row["id"],
                "sari": metrics.sari(source, output, [row["reference"]]) if row.get("reference") else None,
                "fkgl": metrics.fkgl(output),
                "compression": metrics.compression(source, output),
                "number_recall": metrics.number_recall(source, output),
                "number_precision": metrics.number_precision(source, output),
                "negation_kept": metrics.negation_kept(source, output),
            }
            if nli:
                record.update(nli.score(source, output))
            records.append(record)
        print(f"scored {file.stem}")

    per_item = pd.DataFrame(records).astype(object)
    per_item = per_item.where(per_item.notna(), None)
    used = [m for m in METRICS if m in per_item and per_item[m].notna().any()]
    # Pivot to one column per system so paired comparisons line up by item id.
    by_metric = {m: per_item.pivot(index="id", columns="system", values=m) for m in used}
    systems = [f.stem for f in files]
    if args.baseline not in systems:
        parser.error(f"baseline '{args.baseline}' has no prediction file; choose from {systems}")

    summary, lines = {}, ["| system | " + " | ".join(used) + " |", "|---|" + "---|" * len(used)]
    for system in systems:
        summary[system] = {"means": {}, f"minus_{args.baseline}": {}}
        cells = []
        for m in used:
            values = by_metric[m][system].tolist()
            mean, lo, hi = metrics.bootstrap_ci(values)
            summary[system]["means"][m] = {"mean": mean, "ci95": [lo, hi]}
            cells.append("n/a" if mean is None else f"{mean:.2f} [{lo:.2f}, {hi:.2f}]")
            if system != args.baseline:
                diff, dlo, dhi = metrics.paired_bootstrap(values, by_metric[m][args.baseline].tolist())
                summary[system][f"minus_{args.baseline}"][m] = {"mean": diff, "ci95": [dlo, dhi]}
        lines.append(f"| {system} | " + " | ".join(cells) + " |")

    lines += ["", f"Difference from `{args.baseline}` (paired bootstrap, * = 95% interval excludes zero)", "",
              "| system | " + " | ".join(used) + " |", "|---|" + "---|" * len(used)]
    for system in systems:
        if system == args.baseline:
            continue
        cells = []
        for m in used:
            d = summary[system][f"minus_{args.baseline}"][m]
            if d["mean"] is None:
                cells.append("n/a")
            else:
                star = "*" if d["ci95"][0] > 0 or d["ci95"][1] < 0 else ""
                cells.append(f"{d['mean']:+.2f}{star}")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")

    out = OUTPUTS / "metrics" / args.testset
    out.mkdir(parents=True, exist_ok=True)
    per_item.to_csv(out / "per_item.csv", index=False)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "\n".join(lines) + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
