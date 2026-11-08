"""Step 1: download SimpLegal-Pref and write fixed train / validation / test splits.

Input : SimpLegalPref.xlsx from the authors' GitHub repository
        (columns: Index, Original, Skilled, Intermediate, Basic)
Output: data/processed/simplegal_{train,val,test}.jsonl
        each line: {"id", "source", "reference"}  (Original paragraph -> Basic rewrite)

The file has no contract identifier, so paragraphs cannot be grouped by source
document; the split is a seeded shuffle instead. The test split is written once
and must never be used for tuning.
"""
import argparse
import random
import re
import urllib.request

import openpyxl

from jargonaut.common import PROCESSED, RAW, write_jsonl

URL = "https://github.com/GSLD-SimpLegal/FromJargonToClarity/raw/main/GSLD/Data/SimpLegal-Pref/SimpLegalPref.xlsx"
PREAMBLE = re.compile(r"^(here is|here's)[^\n:]*:\s*", re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    xlsx = RAW / "SimpLegalPref.xlsx"
    if not xlsx.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, xlsx)

    sheet = openpyxl.load_workbook(xlsx, read_only=True).active
    rows = sheet.iter_rows(values_only=True)
    header = [str(c).strip() for c in next(rows)]
    col = {name: header.index(name) for name in ("Index", "Original", "Basic")}

    pairs, seen, skipped, preambles = [], set(), 0, 0
    for row in rows:
        source = str(row[col["Original"]] or "").strip()
        reference = str(row[col["Basic"]] or "").strip()
        # A few rewrites open with a chatbot preamble ("Here is the simplified text:").
        reference, n = PREAMBLE.subn("", reference)
        preambles += n
        # Rows without a Basic rewrite, and repeated paragraphs, are left out.
        if not source or not reference or source in seen:
            skipped += 1
            continue
        seen.add(source)
        pairs.append({"id": f"simplegal-{row[col['Index']]}", "source": source, "reference": reference})

    random.Random(args.seed).shuffle(pairs)
    splits = {
        "test": pairs[: args.n_test],
        "val": pairs[args.n_test : args.n_test + args.n_val],
        "train": pairs[args.n_test + args.n_val :],
    }
    for name, split in splits.items():
        write_jsonl(PROCESSED / f"simplegal_{name}.jsonl", split)
        words = sorted(len(p["source"].split()) for p in split)
        print(f"{name:5s} {len(split):3d} pairs | source words: median {words[len(words) // 2]}, max {words[-1]}")
    print(f"Skipped {skipped} rows (empty Basic rewrite or duplicate); removed {preambles} preambles.")


if __name__ == "__main__":
    main()
