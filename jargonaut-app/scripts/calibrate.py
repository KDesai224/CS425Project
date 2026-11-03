"""Turn the DeBERTa score into a probability the user can read.

The extractor's confidence is diff = best span score - null score, a logit margin
(the team's threshold is 6.25). This fits Platt scaling, P = 1 / (1 + exp(-(a*diff + b))),
on the validation split: the probability that the best span overlaps a gold
evidence span (unanswerable questions count as misses). It is fitted on the same
split the threshold was tuned on, and checked on test. Being monotone, it changes
no accept/abstain decision.

    python scripts/calibrate.py          writes eval/calibration.json
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jargonaut.chunking import Chunk
from jargonaut.config import CALIBRATION, CUAD_QUESTIONS
from jargonaut.extractor import Extractor

PREP = Path("/data/prep/outputs")
BM25_DIR = Path("/data/evidence/BM25")


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def scored(ex, split, gold):
    """(diff, hit) for every question of the split in the 10 categories."""
    out = []
    for r in read_jsonl_gz(BM25_DIR / f"bm25_top10_{split}.jsonl.gz"):
        if r["category"] not in CUAD_QUESTIONS:
            continue
        cands = [(Chunk(c["chunk_id"], 0, c["char_start"], c["char_end"], c["text"]), c["bm25_score"])
                 for c in sorted(r["candidates"], key=lambda c: c["rank"])[:10]]
        e = ex.extract(r["question"], cands)
        if e.best is None:
            continue
        hit = any(e.best.start < ge and e.best.end > gs for gs, ge in gold.get(r["question_id"], []))
        out.append((e.best.diff, float(hit), e.accepted))
    return np.array(out, dtype=float)


def fit_platt(x, y, iters=100):
    """Logistic regression on one feature by Newton's method, with a tiny ridge for stability."""
    w = np.zeros(2)
    X = np.stack([x, np.ones_like(x)], 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) + 1e-6 * w
        hess = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def reliability(p, y, bins=(0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001)):
    rows = []
    for lo, hi in zip(bins, bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(m.sum()), "predicted": round(float(p[m].mean()), 3),
                         "observed": round(float(y[m].mean()), 3)})
    ece = sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / len(p)
    return rows, round(ece, 4)


def main():
    gold = {}
    for r in read_jsonl_gz(PREP / "processed/qa.jsonl.gz"):
        if r["category"] in CUAD_QUESTIONS:
            gold[r["question_id"]] = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in r["answers"]]
    ex = Extractor()
    ex.calibration = None
    val, test = scored(ex, "val", gold), scored(ex, "test", gold)
    a, b = fit_platt(val[:, 0], val[:, 1])
    prob = lambda d: 1 / (1 + np.exp(-(a * d + b)))
    result = {"a": float(a), "b": float(b), "fitted_on": "val", "n_val": int(len(val)),
              "label": "the best span overlaps a gold evidence span (unanswerable questions count as misses)",
              "p_at_threshold": round(float(prob(ex.threshold)), 3)}
    for name, data in (("val", val), ("test", test)):
        p, y, acc = prob(data[:, 0]), data[:, 1], data[:, 2].astype(bool)
        rows, ece = reliability(p, y)
        result[name] = {"n": int(len(data)), "brier": round(float(((p - y) ** 2).mean()), 4), "ece": ece,
                        "accepted": int(acc.sum()), "hit_rate_when_accepted": round(float(y[acc].mean()), 3),
                        "mean_p_when_accepted": round(float(p[acc].mean()), 3), "reliability": rows}
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
