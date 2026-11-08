"""Step 8: check the LLM-written rewrites from step 7's shards and merge the good ones with the human pairs.

Input : data/processed/aug_candidates.jsonl              the clauses (step 7)
        data/processed/aug_rewrites/shard_NN.jsonl        agent output, one line per clause: {"id", "reference"}
        data/processed/aug_rewrites/revised_NN.jsonl      optional second passes; these override shard_ entries
Output: data/processed/aug_accepted.jsonl                  pairs that pass every check
        data/processed/aug_rejected.jsonl                  pairs that fail, with the reason
        data/processed/simplegal_train_aug.jsonl           human train pairs + accepted pairs; "origin" says which

A rewrite is accepted only if, measured with the same functions as step 5:
  number_recall == 1      every date, amount and quantity in the clause survives
  number_precision == 1   nothing numeric was invented
  negation_kept != 0      a clause with a negation cue still has one
  fkgl <= --max-fkgl      it reads at or below grade 11 (the human references' 85th percentile)
  compression in range    not an empty summary, not a copy
  no chatbot preamble, no commentary about the passage itself, not identical to the source,
  at least two sentences' worth of text
"""
import argparse
import json
import re
from collections import Counter

from jargonaut import metrics
from jargonaut.common import PROCESSED, read_jsonl, write_jsonl

PREAMBLE = re.compile(r"^(here is|here's|sure|certainly|below is|this clause|the clause|rewrite:|simplified)", re.I)
# Bullets, numbered lists, headings and bold. "[***]" redaction tokens copied from CUAD are allowed.
MARKUP = re.compile(r"^\s*([-*•]|\d+[.)])\s|^#{1,6}\s|\*\*[^*\]]+\*\*", re.M)
# Commentary about the passage itself ("The text starts mid-sentence") instead of a rewrite of it.
META = re.compile(
    r"\b(?:the (?:text|clause|source|passage|list|sentence|excerpt|provision) (?:above |here )?(?:starts|begins|continues|ends|is cut|is incomplete|is truncated|picks up|appears to)|mid-sentence|cut off|cut-off|truncated|the rewrite|this rewrite|the original text|the date entered in the blank|the blank date)\b", re.I
)


def check(source, reference, args):
    """Return None if the pair is acceptable, else the name of the failed check."""
    if not reference.strip():
        return "empty"
    if PREAMBLE.match(reference.strip()):
        return "preamble"
    if MARKUP.search(reference):
        return "markup"
    if META.search(reference):
        return "meta"
    if metrics.compression(source, reference) < args.min_compression:
        return "too_short"
    if metrics.compression(source, reference) > args.max_compression:
        return "too_long"
    if re.sub(r"\W+", " ", reference.lower()).strip() == re.sub(r"\W+", " ", source.lower()).strip():
        return "unchanged"
    if metrics.number_recall(source, reference) not in (None, 1.0):
        return "number_dropped"
    if metrics.number_precision(source, reference) not in (None, 1.0):
        return "number_invented"
    if metrics.negation_kept(source, reference) == 0.0:
        return "negation_lost"
    grade = metrics.fkgl(reference)
    if grade is None or grade > args.max_fkgl:
        return "too_hard"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-fkgl", type=float, default=11.0,
                        help="the human Basic references have median 7.8 and 19%% above 10")
    parser.add_argument("--min-compression", type=float, default=0.35)
    parser.add_argument("--max-compression", type=float, default=1.1)
    args = parser.parse_args()

    candidates = {r["id"]: r for r in read_jsonl(PROCESSED / "aug_candidates.jsonl")}
    # shard_NN.jsonl holds the first drafts; revised_NN.jsonl holds second passes on rejected
    # items. Files are read in sorted order and a later file replaces an earlier one per id,
    # so "revised_" (which sorts after "shard_") wins.
    files = sorted((PROCESSED / "aug_rewrites").glob("shard_*.jsonl")) + sorted(
        (PROCESSED / "aug_rewrites").glob("revised_*.jsonl"))
    if not files:
        parser.error(f"no rewrite files in {PROCESSED / 'aug_rewrites'}")

    rewrites, bad_lines, replaced = {}, 0, 0
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    item_id, reference = row["id"], str(row["reference"]).strip()
                except (ValueError, KeyError, TypeError):
                    bad_lines += 1
                    continue
                if item_id not in candidates:
                    bad_lines += 1
                    continue
                replaced += item_id in rewrites
                rewrites[item_id] = reference
        print(f"read {file.name}")
    if replaced:
        print(f"{replaced} drafts replaced by revised files")

    accepted, rejected, reasons = [], [], Counter()
    for item_id, reference in rewrites.items():
        cand = candidates[item_id]
        reason = check(cand["source"], reference, args)
        pair = {"id": item_id, "origin": cand["origin"], "contract": cand["contract"],
                "category": cand["category"], "source": cand["source"], "reference": reference}
        if reason:
            reasons[reason] += 1
            rejected.append({**pair, "reason": reason})
        else:
            accepted.append(pair)

    write_jsonl(PROCESSED / "aug_accepted.jsonl", accepted)
    write_jsonl(PROCESSED / "aug_rejected.jsonl", rejected)
    human = [{**r, "origin": "simplegal"} for r in read_jsonl(PROCESSED / "simplegal_train.jsonl")]
    merged = human + [{"id": p["id"], "source": p["source"], "reference": p["reference"], "origin": p["origin"]}
                      for p in accepted]
    write_jsonl(PROCESSED / "simplegal_train_aug.jsonl", merged)

    total = len(accepted) + len(rejected)
    print(f"\n{total} rewrites checked ({bad_lines} unparseable or unknown lines skipped)")
    print(f"accepted {len(accepted)} ({100 * len(accepted) / max(1, total):.0f}%), rejected {len(rejected)}")
    for reason, n in reasons.most_common():
        print(f"  {reason:17s} {n}")
    if accepted:
        grades = [metrics.fkgl(p["reference"]) for p in accepted]
        comps = [metrics.compression(p["source"], p["reference"]) for p in accepted]
        print(f"accepted: mean fkgl {sum(grades) / len(grades):.1f}, mean compression {sum(comps) / len(comps):.2f}, "
              f"origins {dict(Counter(p['origin'] for p in accepted))}")
    print(f"merged train set: {len(merged)} pairs -> {PROCESSED / 'simplegal_train_aug.jsonl'}")


if __name__ == "__main__":
    main()
