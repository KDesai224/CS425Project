"""Evaluation on the team's fictional contract test set (the `test` branch).

Ten invented one-page contracts, each written around one of the 10 categories
(test/Fictional_Contract_Test_Set). They come without answers, so two annotators
labelled every contract x category pair before any model was run
(eval/fictional/labeling_guide.md); this scores the system against the adjudicated
labels in eval/fictional/gold.jsonl. The PDFs are read under neutral names
(eval/fictional/pdfs/doc_a.pdf ...; blind_map.json has the originals).

  extraction  every doc x category with the full CUAD question, as in
              deBERTa_training.ipynb section 8: answerability P/R, EM and token F1 over
              answerable pairs (abstentions score 0), span hit; plus whether each
              contract's own clause was found, and whether any accepted answer sits on a
              sentence that only denies the category (the labelled near-misses)
  questions   plain-English questions (eval/fictional/questions.jsonl) through the whole
              app: router, extraction, Gemma rewrite and checks, as a user would get them
  agreement   how often the two annotators agreed, from gold_a.jsonl and gold_b.jsonl

    python scripts/eval_fictional.py [--no-gemma] [--dir eval/fictional]
Nothing here is tuned: every threshold is the one the app already uses.
"""
import argparse
import collections
import json
import math
import re
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_end_to_end import AlnumIndex
from jargonaut.config import CATEGORIES, CUAD_QUESTIONS
from jargonaut.pipeline import Conversation, Jargonaut, open_contract


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Scoring as in deBERTa_training.ipynb, section 8 ---------------------------
def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(s.split())


def token_f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    n = sum((collections.Counter(p) & collections.Counter(g)).values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def overlaps(start, end, ranges):
    return any(start < e and end > s for s, e in ranges)


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def average_precision(labels, scores):
    """Answerability ranking AP (not the official CUAD AUPR), as the notebook reports it."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp, total, ap = 0, sum(labels), 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            tp += 1
            ap += tp / rank
    return round(ap / total, 3) if total else None


def target_category(filename):
    """'06_anti-assignment.pdf' -> 'Anti-Assignment': the category the contract was written around."""
    key = re.sub(r"[^a-z]+", " ", Path(filename).stem.lower()).strip()
    for c in CATEGORIES:
        if re.sub(r"[^a-z]+", " ", c.lower()).strip() == key:
            return c
    raise ValueError(filename)


def summarise(rows):
    pos = [r for r in rows if r["has_answer"]]
    yes = [r for r in rows if r["accepted"]]
    tp = sum(r["has_answer"] for r in yes)
    hits = sum(r["hit"] for r in pos)
    return {"pairs": len(rows), "answerable": len(pos), "accepted": len(yes), "true_accepts": tp,
            "precision": round(tp / len(yes), 3) if yes else None, "precision_ci": wilson(tp, len(yes)),
            "recall": round(tp / len(pos), 3) if pos else None, "recall_ci": wilson(tp, len(pos)),
            "span_hit": round(hits / len(pos), 3) if pos else None, "span_hit_ci": wilson(hits, len(pos)),
            "em": round(sum(r["em"] for r in pos) / len(pos), 3) if pos else None,
            "f1": round(sum(r["f1"] for r in pos) / len(pos), 3) if pos else None}


# --- Annotator agreement --------------------------------------------------------
def agreement(dir_, texts):
    a_path, b_path = dir_ / "gold_a.jsonl", dir_ / "gold_b.jsonl"
    if not (a_path.is_file() and b_path.is_file()):
        return None
    a = {(r["doc"], r["category"]): r for r in read_jsonl(a_path)}
    b = {(r["doc"], r["category"]): r for r in read_jsonl(b_path)}
    keys = sorted(a)
    ya, yb = [bool(a[k]["spans"]) for k in keys], [bool(b[k]["spans"]) for k in keys]
    n = len(keys)
    po = sum(x == y for x, y in zip(ya, yb)) / n
    pa, pb = sum(ya) / n, sum(yb) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    both = [k for k, x, y in zip(keys, ya, yb) if x and y]
    same_place = 0
    for k in both:
        idx = texts[k[0]]
        ra = [idx.locate(s) for s in a[k]["spans"]]
        rb = [idx.locate(s) for s in b[k]["spans"]]
        if any(x and y and x[0] < y[1] and y[0] < x[1] for x in ra for y in rb):
            same_place += 1
    return {"pairs": n, "positives_a": sum(ya), "positives_b": sum(yb), "presence_agreement": round(po, 3),
            "cohen_kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None, "both_positive": len(both),
            "both_positive_overlapping_spans": same_place,
            "disagreements": [{"doc": k[0], "category": k[1], "a": x, "b": y}
                              for k, x, y in zip(keys, ya, yb) if x != y]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="eval/fictional")
    parser.add_argument("--no-gemma", action="store_true", help="extraction only: skip the question pass (router + rewrite)")
    args = parser.parse_args()
    dir_ = Path(args.dir)
    blind = json.loads((dir_ / "blind_map.json").read_text())
    gold = {(r["doc"], r["category"]): r for r in read_jsonl(dir_ / "gold.jsonl")}
    assert len(gold) == len(blind) * len(CATEGORIES), "gold.jsonl must cover every doc x category once"

    system = Jargonaut(load_simplifier=not args.no_gemma)

    contracts, texts, docs, pairs = {}, {}, [], []
    for d in sorted(blind):
        t = time.time()
        contract = open_contract((dir_ / "pdfs" / f"{d}.pdf").read_bytes(), f"{d}.pdf")
        t_ingest = time.time() - t
        doc = contract.doc
        contracts[d], texts[d] = contract, AlnumIndex(doc.text)
        unlocated = 0
        for category in CATEGORIES:
            g = gold[(d, category)]
            spans = [(text, texts[d].locate(text)) for text in g["spans"]]
            unlocated += sum(loc is None for _, loc in spans)
            ranges = [loc for _, loc in spans if loc]
            near = [loc for loc in (texts[d].locate(s) for s in g.get("near_misses", [])) if loc]
            question = CUAD_QUESTIONS[category]
            t = time.time()
            candidates = contract.index.top_k(question)
            extraction = system.extractor.extract(question, candidates)
            t_extract = time.time() - t
            best, yes = extraction.best, extraction.accepted
            has = bool(g["spans"])
            row = {"doc": d, "file": blind[d], "category": category, "target": category == target_category(blind[d]),
                   "has_answer": has, "accepted": yes, "score": round(best.diff, 3) if best else None,
                   "probability": round(extraction.probability, 3) if extraction.probability is not None else None,
                   "span": best.text if best else None, "chunks_retrieved": len(candidates),
                   "hit": False, "em": 0.0, "f1": 0.0, "on_near_miss": False,
                   "alternatives": len(extraction.alternatives), "t_extract": round(t_extract, 3)}
            if yes:
                row["on_near_miss"] = overlaps(best.start, best.end, near)
                row["section"] = [h.text for h in doc.section(best.start)]
                row["pages"] = doc.pages(best.start, best.end)
                if has:
                    row["hit"] = overlaps(best.start, best.end, ranges)
                    row["em"] = float(any(norm(best.text) == norm(text) for text, _ in spans))
                    row["f1"] = round(max(token_f1(best.text, text) for text, _ in spans), 3)
            pairs.append(row)
        docs.append({"doc": d, "file": blind[d], "pages": doc.n_pages, "words": len(doc.text.split()),
                     "chunks": len(contract.chunks), "headings": len(doc.headings), "warnings": doc.warnings,
                     "gold_unlocated": unlocated, "t_ingest": round(t_ingest, 2)})
        print(f"{d} ({blind[d]}): {doc.n_pages}p {len(doc.text.split())}w {len(contract.chunks)} chunks, "
              f"{len(doc.headings)} headings, gold not found {unlocated}", flush=True)

    by_cat = {c: summarise([r for r in pairs if r["category"] == c]) for c in CATEGORIES}
    negatives = [r for r in pairs if not r["has_answer"]]
    targets = [r for r in pairs if r["target"]]
    summary = {
        "contracts": len(docs),
        "ingestion": {"words_mean": round(sum(d["words"] for d in docs) / len(docs), 1),
                      "chunks": collections.Counter(d["chunks"] for d in docs),
                      "gold_spans_not_found": sum(d["gold_unlocated"] for d in docs)},
        "extraction": summarise(pairs),
        "answerability_average_precision": average_precision([r["has_answer"] for r in pairs],
                                                             [r["score"] if r["score"] is not None else -1e9 for r in pairs]),
        "target_clauses": {"n": len(targets), "accepted": sum(r["accepted"] for r in targets),
                           "hit": sum(r["hit"] for r in targets)},
        "incidental_clauses": summarise([r for r in pairs if not r["target"] and r["has_answer"]]),
        "false_accepts": {"negatives": len(negatives), "accepted": sum(r["accepted"] for r in negatives),
                          "on_near_miss_sentence": sum(r["on_near_miss"] for r in negatives),
                          "negatives_with_near_miss": sum(bool(gold[(r["doc"], r["category"])].get("near_misses"))
                                                          for r in negatives)},
        "by_category": by_cat,
        "agreement": agreement(dir_, texts),
    }

    questions = []
    if not args.no_gemma and (dir_ / "questions.jsonl").is_file():
        for q in read_jsonl(dir_ / "questions.jsonl"):
            conv = Conversation(system, contracts[q["doc"]])
            t = time.time()
            ans = conv.reply(q["question"])
            route = ans.route or {}
            routed = ans.category if ans.kind not in ("out_of_scope", "clarify", "help") else ans.kind
            clarified = None
            if ans.kind == "clarify":
                # A user asked to choose picks the topic they meant, if it is offered.
                offered = [o["category"] for o in ans.options]
                clarified = q["expected"] in offered
                if clarified:
                    ans = conv.reply(str(offered.index(q["expected"]) + 1))
            expected = q["expected"]
            route_ok = (routed == "out_of_scope") if expected == "none" else (routed == expected or bool(clarified))
            row = {**q, "routed": routed, "route_ok": route_ok, "clarified": clarified, "kind": ans.kind,
                   "scores": route.get("scores"), "t_total": round(time.time() - t, 2)}
            if expected != "none" and route_ok:
                g = gold[(q["doc"], expected)]
                ranges = [loc for loc in (texts[q["doc"]].locate(s) for s in g["spans"]) if loc]
                ext = conv.last_extraction
                row["gold_present"] = bool(g["spans"])
                row["evidence_ok"] = (bool(ext and ext.accepted and overlaps(ext.best.start, ext.best.end, ranges))
                                      if g["spans"] else ans.kind == "no_evidence")
                if ans.kind in ("answer", "unchecked") and ans.checks:
                    row.update(checks_passed=ans.checks.get("passed"),
                               nli_coverage=ans.checks.get("nli", {}).get("nli_coverage"),
                               clause=ans.clause, rewrite=ans.rewrite)
            questions.append(row)
            print(f"[{q['doc']}] {routed!s:28} {'ok ' if route_ok else 'BAD'} {ans.kind:12} {q['question']}", flush=True)
        ins = [r for r in questions if r["expected"] != "none"]
        oos = [r for r in questions if r["expected"] == "none"]
        shown = [r for r in ins if "checks_passed" in r]
        summary["questions"] = {
            "n": len(questions),
            "in_scope": {"n": len(ins), "routed_right": sum(r["route_ok"] for r in ins),
                         "via_clarification": sum(bool(r["clarified"]) for r in ins),
                         "right_evidence": sum(bool(r.get("evidence_ok")) for r in ins)},
            "out_of_scope": {"n": len(oos), "rejected": sum(r["route_ok"] for r in oos)},
            "rewrites": {"n": len(shown), "checks_passed": sum(bool(r["checks_passed"]) for r in shown)},
            "seconds_mean": round(sum(r["t_total"] for r in questions) / len(questions), 2) if questions else None,
        }

    with open(dir_ / "per_pair.jsonl", "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(dir_ / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if questions:
        with open(dir_ / "per_question.jsonl", "w", encoding="utf-8") as f:
            for r in questions:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (dir_ / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=2))
    print(f"\n{'category':30} pos acc  P      R      hit    F1")
    for c, s in by_cat.items():
        fmt = lambda v: f"{v:.2f}" if v is not None else "  - "
        print(f"{c:30} {s['answerable']:3} {s['accepted']:3}  {fmt(s['precision'])}   {fmt(s['recall'])}   "
              f"{fmt(s['span_hit'])}   {fmt(s['f1'])}")


if __name__ == "__main__":
    main()
