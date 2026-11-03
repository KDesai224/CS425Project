"""End-to-end evaluation on the real CUAD test contracts, read from their PDF files.

For every test contract (CUAD's own PDF, not the pre-extracted text the models were
trained on) and every one of the 10 categories, this runs the app's steps 1 and 3-7
(routing is evaluated separately by eval_router.py, since CUAD has no user questions):

  ingestion   pages, words, headings found, share of gold spans still findable in our text
  extraction  answerability precision/recall and span hit, as in deBERTa_training.ipynb,
              side by side with the same model on CUAD's text (test_preds.jsonl): the
              difference is what reading the PDF costs
  rewrite     for accepted answers: share passing each check, reading grade before and
              after, length ratio, and time per step

Gold spans are offsets into CUAD's text. They are located in our text by matching
letters and digits only, so whitespace, quotes, line-break hyphens and removed page
headers do not matter; spans that cannot be found are reported, not guessed.

    python scripts/eval_end_to_end.py [--contracts N] [--gemma-contracts N] [--out eval/end_to_end]
Nothing here is tuned: every threshold was fixed before this was run.
"""
import argparse
import gzip
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jargonaut.checks import fkgl
from jargonaut.config import CATEGORIES, CUAD_QUESTIONS, DEBERTA_RUN, TOPIC_LABELS
from jargonaut.pipeline import Jargonaut, open_contract

PREP = Path("/data/prep/outputs")
PDF_ROOT = Path("/data/cuad_v1/full_contract_pdf")


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_pdfs(titles):
    """contract title -> PDF path: exact match on letters and digits, else a unique prefix match."""
    pdfs = [p for p in PDF_ROOT.rglob("*") if p.suffix.lower() == ".pdf"]
    by_name = defaultdict(list)
    for p in pdfs:
        by_name[norm_name(p.stem)].append(p)
    out = {}
    for cid, title in titles.items():
        key = norm_name(title)
        hits = by_name.get(key) or [p for k, ps in by_name.items() if k.startswith(key) for p in ps]
        if len(hits) == 1:
            out[cid] = hits[0]
    return out


class AlnumIndex:
    """Letters and digits of a text, lowercased, with a map back to character offsets."""

    def __init__(self, text):
        self.pos = [i for i, ch in enumerate(text) if ch.isalnum()]
        self.s = "".join(text[i].lower() for i in self.pos)

    @staticmethod
    def key(text):
        return "".join(ch.lower() for ch in text if ch.isalnum())

    def locate(self, span_text, k=40):
        """(start, end) of the span in the original text, or None."""
        g = self.key(span_text)
        if len(g) < 12:
            hits = [m.start() for m in re.finditer(re.escape(g), self.s)] if g else []
            return (self.pos[hits[0]], self.pos[hits[0] + len(g) - 1] + 1) if len(hits) == 1 else None
        i = self.s.find(g)
        if i >= 0:
            return self.pos[i], self.pos[i + len(g) - 1] + 1
        # Fall back to the first and last k characters, close together: the middle may
        # differ where CUAD's text kept a page header inside the clause and ours did not.
        head, tail = g[:k], g[-k:]
        i = self.s.find(head)
        while i >= 0:
            j = self.s.find(tail, i + len(head) - k, i + 2 * len(g) + 400)
            if j >= 0:
                return self.pos[i], self.pos[j + len(tail) - 1] + 1
            i = self.s.find(head, i + 1)
        return None


def summarise(rows, prefix):
    pos = [r for r in rows if r["has_answer"]]
    yes = [r for r in rows if r[f"{prefix}accepted"]]
    tp = sum(r["has_answer"] for r in yes)
    out = {"questions": len(rows), "answerable": len(pos), "accepted": len(yes),
           "precision": tp / len(yes) if yes else None, "recall": tp / len(pos) if pos else None,
           "span_hit": sum(r[f"{prefix}hit"] for r in pos) / len(pos) if pos else None}
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contracts", type=int, default=0, help="only the first N test contracts (0 = all)")
    parser.add_argument("--gemma-contracts", type=int, default=0,
                        help="run steps 5-7 (Gemma + checks) on the accepted answers of the first N contracts (-1 = all)")
    parser.add_argument("--out", default="eval/end_to_end")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    contracts = {r["contract_id"]: r for r in read_jsonl_gz(PREP / "processed/contracts.jsonl.gz") if r["split"] == "test"}
    gold = defaultdict(list)    # (contract, category) -> gold spans in CUAD's text
    for r in read_jsonl_gz(PREP / "processed/qa.jsonl.gz"):
        if r["contract_id"] in contracts and r["category"] in CUAD_QUESTIONS:
            gold[(r["contract_id"], r["category"])] = [(a["answer_start"], a["answer_start"] + len(a["text"]), a["text"])
                                                      for a in r["answers"]]
    baseline = {}
    with open(DEBERTA_RUN / "test_preds.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            baseline[(p["contract_id"], p["category"])] = p
    pdfs = find_pdfs({cid: r["title"] for cid, r in contracts.items()})
    ids = sorted(cid for cid in contracts if cid in pdfs)
    if args.contracts:
        ids = ids[:args.contracts]
    print(f"{len(contracts)} test contracts, {len(pdfs)} with a PDF; evaluating {len(ids)}", flush=True)

    system = Jargonaut(load_simplifier=args.gemma_contracts != 0)
    rows, docs, examples = [], [], []
    for n, cid in enumerate(ids):
        t = time.time()
        contract = open_contract(pdfs[cid].read_bytes(), pdfs[cid].name)
        t_ingest = time.time() - t
        doc = contract.doc
        index = AlnumIndex(doc.text)
        cuad_words = len(contracts[cid]["context"].split())
        located_all = []
        explain = args.gemma_contracts < 0 or n < args.gemma_contracts
        for category in CATEGORIES:
            spans = gold[(cid, category)]
            located = [index.locate(text) for _, _, text in spans]
            located_all += located
            found = [l for l in located if l]
            t = time.time()
            question = CUAD_QUESTIONS[category]
            candidates = contract.index.top_k(question)
            extraction = system.extractor.extract(question, candidates)
            t_extract = time.time() - t
            best = extraction.best
            hit = bool(extraction.accepted and any(best.start < e and best.end > s for s, e in found))
            b = baseline[(cid, category)]
            b_hit = bool(b["answerable"] and any(b["char_start"] < e and b["char_end"] > s for s, e, _ in spans))
            row = {"contract_id": cid, "category": category, "has_answer": bool(spans),
                   "gold_spans": len(spans), "gold_located": len(found),
                   "retr_hit": any(c.char_start <= s and e <= c.char_end for c, _ in candidates for s, e in found),
                   "accepted": extraction.accepted, "hit": hit, "score": round(best.diff, 3) if best else None,
                   "probability": extraction.probability, "pdf_accepted": extraction.accepted,
                   "pdf_hit": hit, "text_accepted": b["answerable"], "text_hit": b_hit, "t_extract": round(t_extract, 3)}
            if extraction.accepted:
                row["section"] = [h.text for h in doc.section(best.start)]
                row["pages"] = doc.pages(best.start, best.end)
                if explain:
                    base = dict(category=category, topic=TOPIC_LABELS[category])
                    answer = system.explain(contract, best, extraction, base)
                    c = answer.checks or {}
                    row.update(kind=answer.kind, clause_words=len(answer.clause.split()),
                               rewrite_words=len((answer.rewrite or "").split()),
                               fkgl_clause=fkgl(answer.clause), fkgl_rewrite=fkgl(answer.rewrite or ""),
                               checks_passed=c.get("passed"), numbers_ok=c.get("numbers", {}).get("passed"),
                               negation_ok=c.get("negation", {}).get("passed"), negation_applies=c.get("negation", {}).get("kept") is not None,
                               nli_coverage=c.get("nli", {}).get("nli_coverage"), nli_ok=c.get("nli", {}).get("passed"),
                               t_gemma=round(answer.timings.get("gemma", 0), 2), t_checks=round(answer.timings.get("checks", 0), 2))
                    if len(examples) < 40:
                        examples.append({"contract_id": cid, "category": category, "hit": hit, **answer.to_dict()})
            rows.append(row)
        found_n = sum(1 for l in located_all if l)
        docs.append({"contract_id": cid, "pdf": pdfs[cid].name, "pages": doc.n_pages, "words": len(doc.text.split()),
                     "cuad_words": cuad_words, "headings": len(doc.headings), "gold_spans": len(located_all),
                     "gold_located": found_n, "t_ingest": round(t_ingest, 2), "warnings": doc.warnings})
        print(f"[{n + 1}/{len(ids)}] {cid} {doc.n_pages}p {len(doc.text.split())}w ({cuad_words} in CUAD) "
              f"{len(doc.headings)} headings, gold located {found_n}/{len(located_all)}, ingest {t_ingest:.1f}s", flush=True)

    with open(out_dir / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(out_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=1, ensure_ascii=False)

    explained = [r for r in rows if "kind" in r]
    pos = [r for r in rows if r["has_answer"]]
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None
    summary = {
        "contracts": len(ids),
        "ingestion": {
            "pages_mean": mean([d["pages"] for d in docs]), "words_ratio_to_cuad": mean([d["words"] / d["cuad_words"] for d in docs]),
            "headings_mean": mean([d["headings"] for d in docs]), "contracts_without_headings": sum(d["headings"] == 0 for d in docs),
            "gold_located": round(sum(d["gold_located"] for d in docs) / max(1, sum(d["gold_spans"] for d in docs)), 3),
            "seconds_mean": mean([d["t_ingest"] for d in docs]),
        },
        "retrieval_gold_in_top10": round(sum(r["retr_hit"] for r in pos) / len(pos), 3) if pos else None,
        "extraction_from_pdf": summarise(rows, "pdf_"),
        "extraction_from_cuad_text": summarise(rows, "text_"),
        "extraction_seconds_mean": mean([r["t_extract"] for r in rows]),
        "accepted_with_section": round(sum(bool(r.get("section")) for r in rows if r["accepted"]) / max(1, sum(r["accepted"] for r in rows)), 3),
        "accepted_with_pages": round(sum(bool(r.get("pages")) for r in rows if r["accepted"]) / max(1, sum(r["accepted"] for r in rows)), 3),
    }
    if explained:
        summary["rewrites"] = {
            "n": len(explained),
            "all_checks_passed": mean([float(r["checks_passed"]) for r in explained]),
            "numbers_passed": mean([float(r["numbers_ok"]) for r in explained]),
            "negation_passed": mean([float(r["negation_ok"]) for r in explained]),
            "negation_passed_when_negated": mean([float(r["negation_ok"]) for r in explained if r["negation_applies"]]),
            "nli_passed": mean([float(r["nli_ok"]) for r in explained]),
            "nli_coverage_mean": mean([r["nli_coverage"] for r in explained if r["nli_coverage"] is not None]),
            "fkgl_clause_mean": mean([r["fkgl_clause"] for r in explained if r["fkgl_clause"] is not None]),
            "fkgl_rewrite_mean": mean([r["fkgl_rewrite"] for r in explained if r["fkgl_rewrite"] is not None]),
            "length_ratio_mean": mean([r["rewrite_words"] / max(1, r["clause_words"]) for r in explained]),
            "seconds_gemma_mean": mean([r["t_gemma"] for r in explained]),
            "seconds_checks_mean": mean([r["t_checks"] for r in explained]),
            "on_correct_spans": {
                "n": sum(r["hit"] for r in explained),
                "all_checks_passed": mean([float(r["checks_passed"]) for r in explained if r["hit"]]),
            },
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
