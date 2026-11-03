"""Check that the app reproduces each team component exactly.

  questions  the 10 CUAD questions in config.py equal the ones in qa.jsonl.gz
  chunking   re-chunking every CUAD contract gives Theo's chunks (offsets and ids)
  bm25       re-ranking every exported question gives the same top 10 and scores
             as bm25_top10_{train,val,test}.jsonl.gz
  code       the functions copied from Grpproj (prompt, decoding cleanup, number /
             negation / NLI checks, clause expansion) are character-identical to it
  expand     clause_span returns expand_to_clause's passage on every gold span
  deberta    the extractor reproduces the saved test predictions (GPU; --skip-gpu to skip)

    python scripts/check_parity.py [--skip-gpu]
Exits with status 1 if any check fails.
"""
import argparse
import ast
import collections
import gzip
import json
import re
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jargonaut import checks, expand, simplifier
from jargonaut.chunking import Chunk, chunk_spans
from jargonaut.config import CATEGORIES, CUAD_QUESTIONS, DEBERTA_RUN
from jargonaut.retrieval import ChunkIndex

PREP = Path("/data/prep/outputs")
BM25_DIR = Path("/data/evidence/BM25")
GRPPROJ = Path("/grpproj")
RESULTS = {}


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def report(name, ok, detail):
    RESULTS[name] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def check_questions(qa):
    bad = [(r["question_id"]) for r in qa if r["category"] in CUAD_QUESTIONS and r["question"] != CUAD_QUESTIONS[r["category"]]]
    seen = {r["category"] for r in qa}
    report("questions", not bad and set(CATEGORIES) <= seen,
           f"{sum(r['category'] in CUAD_QUESTIONS for r in qa)} rows checked, {len(bad)} mismatches")


def check_chunking(contracts, chunks_by_contract):
    bad = 0
    for cid, text in contracts.items():
        mine = [(f"{cid}_c{k:04d}", s, e) for k, (s, e) in enumerate(chunk_spans(text))]
        theirs = [(c["chunk_id"], c["char_start"], c["char_end"]) for c in chunks_by_contract[cid]]
        bad += mine != theirs
    n = sum(map(len, chunks_by_contract.values()))
    report("chunking", bad == 0, f"{len(contracts)} contracts, {n} chunks, {bad} contracts differ")


def check_bm25(chunks_by_contract):
    indexes, n, bad_ids, bad_scores = {}, 0, 0, 0
    for split in ("train", "val", "test"):
        for r in read_jsonl_gz(BM25_DIR / f"bm25_top10_{split}.jsonl.gz"):
            cid = r["contract_id"]
            if cid not in indexes:
                indexes[cid] = ChunkIndex([Chunk(c["chunk_id"], c["chunk_index"], c["char_start"], c["char_end"], c["text"])
                                           for c in chunks_by_contract[cid]])
            top = indexes[cid].top_k(r["question"])
            want = sorted(r["candidates"], key=lambda c: c["rank"])
            n += 1
            bad_ids += [c.chunk_id for c, _ in top] != [c["chunk_id"] for c in want]
            bad_scores += [round(s, 4) for _, s in top] != [c["bm25_score"] for c in want]
    report("bm25", bad_ids == 0 and bad_scores == 0,
           f"{n} questions, {bad_ids} with a different top 10, {bad_scores} with different scores")


def _top_level_sources(path):
    """name -> source text of every top-level def, class and assignment in a file, without running it."""
    src = path.read_text(encoding="utf-8")
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(src, node)
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        out[f"{node.name}.{item.name}"] = textwrap.dedent(ast.get_source_segment(src, item))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.get_source_segment(src, node)
    return out


def check_code():
    if not GRPPROJ.is_dir():
        report("code", False, f"{GRPPROJ} is not mounted")
        return
    pairs = [
        (GRPPROJ / "jargonaut/common.py", Path(simplifier.__file__),
         ["SYSTEM_PROMPT", "build_messages", "render_prompt", "clean_output"]),
        (GRPPROJ / "jargonaut/metrics.py", Path(checks.__file__),
         ["_NUMBER_WORDS", "_LEADING_NUMBERING", "_CLAUSE_HEADING", "_REF_WORD", "_NOT_UNIT", "_CROSS_REFERENCE",
          "_NUMBER", "_NEGATION", "_NEGATION_IDIOMS", "numbers", "number_recall", "number_precision",
          "negation_kept", "split_sentences", "NLIScorer.score"]),
        (GRPPROJ / "scripts/02_build_pipeline_set.py", Path(expand.__file__), ["expand_to_clause"]),
    ]
    differ = []
    for theirs_path, mine_path, names in pairs:
        theirs, mine = _top_level_sources(theirs_path), _top_level_sources(mine_path)
        differ += [f"{mine_path.name}:{n}" for n in names if theirs.get(n) is None or theirs.get(n) != mine.get(n)]
    # The settings the app hard-codes (config.py) against Grpproj's
    yaml_text = (GRPPROJ / "configs/e4b.yaml").read_text(encoding="utf-8")
    settings_ok = (re.search(r"model_id:\s*google/gemma-4-E4B-it", yaml_text) and re.search(r"max_new_tokens:\s*512", yaml_text)
                   and re.search(r"load_in_4bit:\s*true", yaml_text) and "MIN_WORDS, MAX_WORDS = 40, 300" in
                   (GRPPROJ / "scripts/02_build_pipeline_set.py").read_text(encoding="utf-8"))
    report("code", not differ and bool(settings_ok),
           f"{sum(len(n) for _, _, n in pairs)} copied definitions compared, differing: {differ or 'none'}; "
           f"e4b.yaml / clause limits {'match' if settings_ok else 'DIFFER'}")


def check_expand(contracts, qa):
    # Grpproj's own function, taken from its file and run without importing the script
    grp_src = _top_level_sources(GRPPROJ / "scripts/02_build_pipeline_set.py").get("expand_to_clause") if GRPPROJ.is_dir() else None
    original = None
    if grp_src:
        ns = {"re": re, "MIN_WORDS": 40, "MAX_WORDS": 300}
        exec(grp_src, ns)
        original = ns["expand_to_clause"]
    n = bad = bad_orig = bad_offsets = 0
    for r in qa:
        if r["category"] not in CUAD_QUESTIONS:
            continue
        text = contracts.get(r["contract_id"])
        for a in r["answers"]:
            s, e = a["answer_start"], a["answer_start"] + len(a["text"])
            want = expand.expand_to_clause(text, s, e)
            got = expand.clause_span(text, s, e)
            n += 1
            if want is None or got is None:
                bad += (want is None) != (got is None)
                continue
            passage, ps, pe = got
            bad += passage != want
            bad_orig += original is not None and original(text, s, e) != want
            # the offsets must cover the passage: same text once whitespace is ignored
            bad_offsets += "".join(text[ps:pe].split()) != "".join(passage.split())
    report("expand", bad == 0 and bad_orig == 0 and bad_offsets == 0,
           f"{n} gold spans, {bad} passage mismatches, {bad_orig} vs Grpproj's function, {bad_offsets} offset mismatches")


def check_deberta():
    from jargonaut.extractor import Extractor

    preds = {}
    with open(DEBERTA_RUN / "test_preds.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            preds[p["question_id"]] = p
    records = [r for r in read_jsonl_gz(BM25_DIR / "bm25_top10_test.jsonl.gz") if r["question_id"] in preds]
    ex = Extractor()
    t = time.time()
    n = accept_diff = span_diff = 0
    max_delta = 0.0
    for r in records:
        cands = [(Chunk(c["chunk_id"], 0, c["char_start"], c["char_end"], c["text"]), c["bm25_score"])
                 for c in sorted(r["candidates"], key=lambda c: c["rank"])[:10]]
        e = ex.extract(r["question"], cands)
        want = preds[r["question_id"]]
        n += 1
        got_diff = e.best.diff if e.best else -1e9
        max_delta = max(max_delta, abs(got_diff - want["confidence"]))
        accept_diff += e.accepted != want["answerable"]
        if e.accepted and want["answerable"]:
            span_diff += (e.best.start, e.best.end, e.best.chunk_id) != (want["char_start"], want["char_end"], want["chunk_id"])
    report("deberta", accept_diff == 0 and span_diff == 0 and max_delta < 0.05,
           f"{n} test questions in {time.time() - t:.0f} s, {accept_diff} answerability disagreements, "
           f"{span_diff} different spans, max |score difference| {max_delta:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    qa = read_jsonl_gz(PREP / "processed/qa.jsonl.gz")
    contracts = {r["contract_id"]: r["context"] for r in read_jsonl_gz(PREP / "processed/contracts.jsonl.gz")}
    chunks_by_contract = collections.defaultdict(list)
    for c in read_jsonl_gz(PREP / "chunks/chunks_words_m300_o60/chunks.jsonl.gz"):
        chunks_by_contract[c["contract_id"]].append(c)
    for cs in chunks_by_contract.values():
        cs.sort(key=lambda c: c["chunk_index"])

    check_questions(qa)
    check_chunking(contracts, chunks_by_contract)
    check_code()
    check_expand(contracts, qa)
    check_bm25(chunks_by_contract)
    if not args.skip_gpu:
        check_deberta()
    print(f"\n{sum(RESULTS.values())}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS.values()) else 1)


if __name__ == "__main__":
    main()
