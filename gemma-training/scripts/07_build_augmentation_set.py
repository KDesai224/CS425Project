"""Step 7: sample clauses for synthetic training pairs from the CUAD and ContractNLI *train* splits.

The 139 SimpLegal-Pref pairs are few. This script draws extra clauses from the
train (and ContractNLI dev) splits, never test, and writes them in shards so that
rewrites can be produced in parallel by LLM agents and then checked by step 8.

Output: data/processed/aug_candidates.jsonl   every drawn clause
        data/processed/aug_shards/shard_NN.jsonl   the same rows, interleaved into shards
        each line: {"id", "origin", "contract", "category", "length_bucket", "source"}

Diversity controls, in order:
  - round-robin over clause category, so rare categories are not swamped
  - rotation over three length buckets (40-80, 80-150, 150-300 words) inside each category
  - at most --per-contract clauses from one contract
  - any passage sharing a ten-word shingle with SimpLegal-Pref (all splits) or the
    pipeline test set is excluded, so nothing held out leaks into training

Shards are interleaved, so each shard holds every category and length bucket.
"""
import argparse
import importlib.util
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from jargonaut.common import PROCESSED, read_jsonl, write_jsonl

# Reuse the clause expansion, overlap normalisation and download helpers from step 2.
_spec = importlib.util.spec_from_file_location("build_pipeline_set", Path(__file__).with_name("02_build_pipeline_set.py"))
_step2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step2)
expand_to_clause, normalise, fetch_zip = _step2.expand_to_clause, _step2.normalise, _step2.fetch_zip
CUAD_URL, CNLI_URL, SKIP_CATEGORIES = _step2.CUAD_URL, _step2.CNLI_URL, _step2.SKIP_CATEGORIES

BUCKETS = ["short", "medium", "long"]


def length_bucket(text):
    n = len(text.split())
    return "short" if n < 80 else "medium" if n < 150 else "long"


def cuad_train_candidates():
    data = json.load(fetch_zip(CUAD_URL, "cuad_data.zip").open("train_separate_questions.json"))["data"]
    for contract in data:
        context = contract["paragraphs"][0]["context"]
        for qa in contract["paragraphs"][0]["qas"]:
            category = re.search(r'related to "([^"]+)"', qa["question"])
            category = category.group(1) if category else "Unknown"
            if category in SKIP_CATEGORIES or not qa["answers"]:
                continue
            answer = max(qa["answers"], key=lambda a: len(a["text"]))
            start = answer["answer_start"]
            source = expand_to_clause(context, start, start + len(answer["text"]))
            if source:
                yield {"origin": "cuad", "contract": contract["title"], "category": category, "source": source}


def contractnli_train_candidates():
    archive = fetch_zip(CNLI_URL, "contract-nli.zip")
    for member in ("contract-nli/train.json", "contract-nli/dev.json"):
        data = json.load(io.TextIOWrapper(archive.open(member), encoding="utf-8"))
        for doc in data["documents"]:
            text, spans = doc["text"], doc["spans"]
            for label, annotation in doc["annotation_sets"][0]["annotations"].items():
                idx = annotation["spans"]
                if not idx or max(idx) - min(idx) > 3:
                    continue
                source = expand_to_clause(text, spans[min(idx)][0], spans[max(idx)][1])
                if source:
                    yield {
                        "origin": "contractnli", "contract": doc["file_name"],
                        "category": data["labels"][label]["short_description"], "source": source,
                    }


def held_out_shingles():
    """Ten-word shingles of every passage the model must never train on."""
    shingles = set()
    for name in ("simplegal_train", "simplegal_val", "simplegal_test", "pipeline_test"):
        for row in read_jsonl(PROCESSED / f"{name}.jsonl"):
            words = normalise(row["source"]).split()
            shingles.update(" ".join(words[i : i + 10]) for i in range(0, max(1, len(words) - 9), 5))
    return shingles


def sample(candidates, n, rng, per_contract):
    """Round-robin over categories, rotating length buckets within each, with a per-contract cap."""
    pool = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        pool[c["category"]][length_bucket(c["source"])].append(c)
    for buckets in pool.values():
        for items in buckets.values():
            rng.shuffle(items)
    picked, used, turn = [], Counter(), defaultdict(int)
    while len(picked) < n and any(any(b.values()) for b in pool.values()):
        for category in sorted(pool):
            for _ in range(len(BUCKETS)):
                items = pool[category][BUCKETS[turn[category] % len(BUCKETS)]]
                turn[category] += 1
                found = None
                while items and found is None:
                    c = items.pop()
                    if used[c["contract"]] < per_contract:
                        found = c
                if found:
                    used[found["contract"]] += 1
                    picked.append(found)
                    break
            if len(picked) == n:
                break
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cuad", type=int, default=805, help="700 target x 1.15 overdraw for the quality gate")
    parser.add_argument("--n-contractnli", type=int, default=345, help="300 target x 1.15 overdraw")
    parser.add_argument("--per-contract", type=int, default=3)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    shingles = held_out_shingles()

    def is_clean(c):
        words = normalise(c["source"]).split()
        return not any(" ".join(words[i : i + 10]) in shingles for i in range(max(1, len(words) - 9)))

    rows = []
    for name, candidates, n in (
        ("CUAD train", cuad_train_candidates(), args.n_cuad),
        ("ContractNLI train+dev", contractnli_train_candidates(), args.n_contractnli),
    ):
        unique = {}
        for c in candidates:
            unique.setdefault(c["source"], c)
        clean = [c for c in unique.values() if is_clean(c)]
        picked = sample(clean, n, rng, args.per_contract)
        print(f"{name}: {len(unique)} unique passages, {len(unique) - len(clean)} dropped for overlap with "
              f"held-out sets, {len(picked)} drawn from {len({c['contract'] for c in picked})} contracts")
        rows.extend(picked)

    for i, row in enumerate(rows):
        row["id"] = f"aug-{row['origin']}-{i:04d}"
        row["length_bucket"] = length_bucket(row["source"])
    rows = [{"id": r["id"], "origin": r["origin"], "contract": r["contract"], "category": r["category"],
             "length_bucket": r["length_bucket"], "source": r["source"]} for r in rows]
    write_jsonl(PROCESSED / "aug_candidates.jsonl", rows)

    n_shards = -(-len(rows) // args.shard_size)
    shards = [rows[i::n_shards] for i in range(n_shards)]
    shard_dir = PROCESSED / "aug_shards"
    for i, shard in enumerate(shards):
        write_jsonl(shard_dir / f"shard_{i:02d}.jsonl", shard)

    buckets = Counter(r["length_bucket"] for r in rows)
    print(f"Wrote {len(rows)} candidates to {PROCESSED / 'aug_candidates.jsonl'} and {n_shards} shards of "
          f"<= {args.shard_size} to {shard_dir}")
    print("length buckets: " + ", ".join(f"{b} {buckets[b]}" for b in BUCKETS)
          + f" | categories: {len({r['category'] for r in rows})}")


if __name__ == "__main__":
    main()
