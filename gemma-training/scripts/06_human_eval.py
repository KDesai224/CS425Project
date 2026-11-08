"""Step 6: blind human rating of the systems.

export  Sample items from a test set and write a rating sheet in which each
        system's rewrite appears in shuffled order with no system name.
          outputs/human_eval/<testset>/rating_sheet.csv   give a copy to each rater
          outputs/human_eval/<testset>/key.csv            row -> system; keep from raters

        Each rater saves their copy as rating_<name>.csv in the same folder and
        fills two columns with a whole number from 1 (worst) to 5 (best):
          meaning_preserved  does the rewrite say exactly what the clause says?
          understandable     could someone with no legal training follow it?

score   Read every rating_*.csv, average per system with 95% bootstrap intervals,
        and report agreement between raters (Krippendorff's alpha, ordinal).
"""
import argparse
import random

import pandas as pd

from jargonaut import metrics
from jargonaut.common import OUTPUTS, read_jsonl

CRITERIA = ["meaning_preserved", "understandable"]


def export(args):
    files = sorted((OUTPUTS / "predictions" / args.testset).glob("*.jsonl"))
    if args.systems:
        files = [f for f in files if f.stem in args.systems]
    predictions = {f.stem: {r["id"]: r for r in read_jsonl(f)} for f in files}
    rng = random.Random(args.seed)
    ids = sorted(set.intersection(*(set(p) for p in predictions.values())))
    ids = rng.sample(ids, min(args.n, len(ids)))

    sheet, key = [], []
    for item_id in ids:
        systems = list(predictions)
        rng.shuffle(systems)
        for system in systems:
            row_id = f"r{len(sheet):04d}"
            row = predictions[system][item_id]
            sheet.append({"row_id": row_id, "item_id": item_id, "source": row["source"],
                          "rewrite": row["prediction"], **{c: "" for c in CRITERIA}})
            key.append({"row_id": row_id, "system": system})

    out = OUTPUTS / "human_eval" / args.testset
    out.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens the file with the right encoding.
    pd.DataFrame(sheet).to_csv(out / "rating_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    print(f"Wrote {len(sheet)} rows ({len(ids)} items x {len(predictions)} systems) to {out}")


def score(args):
    import krippendorff

    folder = OUTPUTS / "human_eval" / args.testset
    key = pd.read_csv(folder / "key.csv").set_index("row_id")["system"]
    raters = {f.stem.removeprefix("rating_"): pd.read_csv(f, encoding="utf-8-sig").set_index("row_id")
              for f in sorted(folder.glob("rating_*.csv")) if f.name != "rating_sheet.csv"}
    if not raters:
        raise SystemExit(f"No rating_<name>.csv files in {folder}")
    print(f"Raters: {', '.join(raters)}\n")

    for criterion in CRITERIA:
        ratings = pd.DataFrame({name: pd.to_numeric(df[criterion], errors="coerce") for name, df in raters.items()})
        print(f"{criterion}")
        per_row = ratings.mean(axis=1)
        for system, rows in per_row.groupby(key):
            mean, lo, hi = metrics.bootstrap_ci([None if pd.isna(v) else v for v in rows])
            print(f"  {system:28s} " + ("no ratings" if mean is None else f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"))
        if len(raters) > 1:
            alpha = krippendorff.alpha(reliability_data=ratings.T.values, level_of_measurement="ordinal")
            print(f"  Krippendorff's alpha: {alpha:.2f}  (0.67+ is usually taken as acceptable agreement)")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--testset", default="pipeline")
    p.add_argument("--systems", nargs="*", help="system names to include (default: all)")
    p.add_argument("--n", type=int, default=50, help="number of items")
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=export)
    p = sub.add_parser("score")
    p.add_argument("--testset", default="pipeline")
    p.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
