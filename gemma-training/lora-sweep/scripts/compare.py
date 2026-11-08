"""Compare the learning-rate run with Run 6 and zero-shot on the 20 validation clauses.

Reads
  outputs/baselines/run6_history.json            Run 6's scores at each checkpoint (copied from Grpproj)
  outputs/adapters/<RUN>/history.json            this run's scores at each quarter-epoch checkpoint
  outputs/metrics/simplegal_val/per_item.csv     full scoring of the selected checkpoints (05_evaluate.py)
Prints Markdown: the checkpoint trajectories of both runs, then the selected models
compared item by item with the same paired bootstrap as 05_evaluate.py.

    python scripts/compare.py
"""
import json
import sys

import pandas as pd

from jargonaut import metrics
from jargonaut.common import OUTPUTS

RUN = "e4b-r16-lr0.0001-e2-s42-syn"
RUN6 = "e4b-r16-lr0.0002-e2-s42-syn"
TRAJECTORY = ["sari", "fkgl", "compression", "number_recall", "nli_coverage", "eval_loss"]
FULL = ["sari", "fkgl", "compression", "number_recall", "number_precision", "negation_kept",
        "nli_supported", "nli_contradicted", "nli_coverage"]
SYSTEMS = {"zero-shot": "zeroshot-e4b", "Run 6 (lr 2e-4)": f"ft-{RUN6}", "lr 1e-4": f"ft-{RUN}"}


def fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def trajectory_rows(label, history):
    rows = []
    for h in history["epochs"]:
        chosen = " (selected)" if h["epoch"] == history["best_epoch"] else ""
        rows.append(f"| {label} | {h['epoch']:g}{chosen} | " + " | ".join(fmt(h.get(k)) for k in TRAJECTORY) + " |")
    return rows


def column(per_item, system, metric):
    values = per_item[per_item.system == system].set_index("id")[metric]
    return {i: (None if pd.isna(v) else float(v)) for i, v in values.items()}


def main():
    new_path = OUTPUTS / "adapters" / RUN / "history.json"
    if not new_path.is_file():
        sys.exit(f"{new_path} not found: has the training run finished?")
    new = json.loads(new_path.read_text(encoding="utf-8"))
    run6 = json.loads((OUTPUTS / "baselines" / "run6_history.json").read_text(encoding="utf-8"))

    lines = ["## Validation scores at each checkpoint", "",
             "| run | epoch | SARI | FKGL | compression | number recall | NLI coverage | val loss |",
             "|---|---|---|---|---|---|---|---|",
             "| zero-shot | - | " + " | ".join(fmt(new["zero_shot"].get(k)) for k in TRAJECTORY) + " |"]
    lines += trajectory_rows("Run 6 (lr 2e-4)", run6) + trajectory_rows("lr 1e-4", new)
    floors = {k: new["zero_shot"][k] - 0.05 for k in ("number_recall", "nli_coverage")}
    lines += ["", f"Selection rule: best SARI among checkpoints with number recall >= {floors['number_recall']:.3f} "
              f"and NLI coverage >= {floors['nli_coverage']:.3f} (zero-shot minus 0.05).", ""]

    per_item_path = OUTPUTS / "metrics" / "simplegal_val" / "per_item.csv"
    if per_item_path.is_file():
        per_item = pd.read_csv(per_item_path)
        cols = {name: {m: column(per_item, system, m) for m in FULL} for name, system in SYSTEMS.items()
                if system in set(per_item.system)}
        lines += ["## Selected checkpoints, full scoring (means over the 20 validation clauses)", "",
                  "| system | " + " | ".join(FULL) + " |", "|---|" + "---|" * len(FULL)]
        for name, by_metric in cols.items():
            means = [metrics.bootstrap_ci([v for v in by_metric[m].values() if v is not None])[0] for m in FULL]
            lines.append(f"| {name} | " + " | ".join(fmt(v) for v in means) + " |")
        lines += ["", "Paired differences (95% bootstrap interval; * = excludes zero)", "",
                  "| comparison | " + " | ".join(FULL) + " |", "|---|" + "---|" * len(FULL)]
        for a, b in (("lr 1e-4", "Run 6 (lr 2e-4)"), ("lr 1e-4", "zero-shot"), ("Run 6 (lr 2e-4)", "zero-shot")):
            if a not in cols or b not in cols:
                continue
            cells = []
            for m in FULL:
                ids = sorted(cols[a][m])
                diff, lo, hi = metrics.paired_bootstrap([cols[a][m][i] for i in ids], [cols[b][m].get(i) for i in ids])
                star = "*" if diff is not None and (lo > 0 or hi < 0) else ""
                cells.append("n/a" if diff is None else f"{diff:+.2f}{star} [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {a} minus {b} | " + " | ".join(cells) + " |")
    else:
        lines.append(f"({per_item_path} not found: run scripts/05_evaluate.py --testset simplegal_val first.)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
