"""Step 2: build the pipeline test set from CUAD and ContractNLI test contracts.

SimpLegal-Pref paragraphs are not what the simplifier sees in the product. There
it receives evidence found by the extractor. This script samples gold evidence
spans from the *test* splits of CUAD and ContractNLI and expands each one to its
surrounding clause, giving realistic inputs. There are no reference rewrites,
so this set is scored with readability, faithfulness and human ratings only.

Output: data/processed/pipeline_test.jsonl
        each line: {"id", "origin", "contract", "category", "question", "span", "source"}

Any passage that shares text with a SimpLegal-Pref paragraph is excluded, so
nothing the simplifier was trained on can appear here.
"""
import argparse
import io
import json
import random
import re
import urllib.request
import zipfile
from collections import defaultdict

from jargonaut.common import PROCESSED, RAW, read_jsonl, write_jsonl

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CNLI_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
# These CUAD categories mark title-page facts (a name, a date), not clauses.
SKIP_CATEGORIES = {"Document Name", "Parties", "Agreement Date"}
MIN_WORDS, MAX_WORDS = 40, 300


def fetch_zip(url, name):
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return zipfile.ZipFile(path)


def normalise(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def expand_to_clause(text, start, end):
    """Grow [start, end) to whole paragraphs until the passage has MIN_WORDS words,
    then cut it back to MAX_WORDS around the span if it is too long."""
    blocks = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]
    inside = [i for i, (s, e) in enumerate(blocks) if s < end and e > start]
    if not inside:
        return None
    lo, hi = inside[0], inside[-1]
    words = lambda: len(text[blocks[lo][0] : blocks[hi][1]].split())
    while words() < MIN_WORDS and (lo > 0 or hi < len(blocks) - 1):
        if hi < len(blocks) - 1:
            hi += 1
        if words() < MIN_WORDS and lo > 0:
            lo -= 1
    passage_start, passage_end = blocks[lo][0], blocks[hi][1]
    if len(text[passage_start:passage_end].split()) > MAX_WORDS:
        # Keep the span and add context on both sides up to the limit, starting
        # and ending on a sentence boundary where there is one.
        budget = max(0, MAX_WORDS - len(text[start:end].split())) // 2
        before = " ".join(text[passage_start:start].split(" ")[-budget:]) if budget else ""
        after = " ".join(text[end:passage_end].split(" ")[:budget]) if budget else ""
        before = re.sub(r"^.*?[.;:]\s+(?=[A-Z(\"])", "", before, count=1, flags=re.S)
        last_stop = after.rfind(". ")
        if not after.rstrip().endswith(".") and last_stop >= 0:
            after = after[: last_stop + 1]
        return (before + text[start:end] + after).strip()
    return re.sub(r"\s*\n\s*", "\n", text[passage_start:passage_end]).strip()


def cuad_candidates():
    data = json.load(fetch_zip(CUAD_URL, "cuad_data.zip").open("test.json"))["data"]
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
                yield {
                    "origin": "cuad", "contract": contract["title"], "category": category,
                    "question": qa["question"], "span": answer["text"], "source": source,
                }


def contractnli_candidates():
    archive = fetch_zip(CNLI_URL, "contract-nli.zip")
    data = json.load(io.TextIOWrapper(archive.open("contract-nli/test.json"), encoding="utf-8"))
    for doc in data["documents"]:
        text, spans = doc["text"], doc["spans"]
        for label, annotation in doc["annotation_sets"][0]["annotations"].items():
            idx = annotation["spans"]
            # Only evidence that sits together in the document makes one coherent passage.
            if not idx or max(idx) - min(idx) > 3:
                continue
            start, end = spans[min(idx)][0], spans[max(idx)][1]
            source = expand_to_clause(text, start, end)
            if source:
                yield {
                    "origin": "contractnli", "contract": doc["file_name"],
                    "category": data["labels"][label]["short_description"],
                    "question": data["labels"][label]["hypothesis"],
                    "span": text[start:end], "source": source,
                }


def sample(candidates, n, rng, per_contract=2):
    """Round-robin over categories so no clause type or contract dominates."""
    by_category = defaultdict(list)
    for c in candidates:
        by_category[c["category"]].append(c)
    for items in by_category.values():
        rng.shuffle(items)
    picked, used = [], defaultdict(int)
    while len(picked) < n and any(by_category.values()):
        for category in sorted(by_category):
            items = by_category[category]
            while items:
                c = items.pop()
                if used[c["contract"]] < per_contract and c["source"] not in {p["source"] for p in picked}:
                    used[c["contract"]] += 1
                    picked.append(c)
                    break
            if len(picked) == n:
                break
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cuad", type=int, default=80)
    parser.add_argument("--n-contractnli", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Ten-word shingles of every SimpLegal paragraph, for the overlap check.
    shingles = set()
    for split in ("train", "val", "test"):
        for row in read_jsonl(PROCESSED / f"simplegal_{split}.jsonl"):
            words = normalise(row["source"]).split()
            shingles.update(" ".join(words[i : i + 10]) for i in range(0, max(1, len(words) - 9), 5))

    def is_clean(candidate):
        words = normalise(candidate["source"]).split()
        return not any(" ".join(words[i : i + 10]) in shingles for i in range(max(1, len(words) - 9)))

    rows = []
    for name, candidates, n in (
        ("CUAD", cuad_candidates(), args.n_cuad),
        ("ContractNLI", contractnli_candidates(), args.n_contractnli),
    ):
        candidates = list(candidates)
        clean = [c for c in candidates if is_clean(c)]
        picked = sample(clean, n, rng)
        print(f"{name}: {len(candidates)} candidate spans, {len(candidates) - len(clean)} dropped for "
              f"overlap with SimpLegal-Pref, {len(picked)} sampled")
        rows.extend(picked)

    for i, row in enumerate(rows):
        row["id"] = f"pipeline-{row['origin']}-{i:03d}"
    write_jsonl(PROCESSED / "pipeline_test.jsonl", [{"id": r.pop("id"), **r} for r in rows])
    words = sorted(len(r["source"].split()) for r in rows)
    print(f"Wrote {len(rows)} passages | words: min {words[0]}, median {words[len(words) // 2]}, max {words[-1]}")


if __name__ == "__main__":
    main()
