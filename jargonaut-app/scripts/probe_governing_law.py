"""Diagnostic for eval_fictional.py: is Governing Law missed because of how the clause is worded?

On the fictional contracts DeBERTa finds the right governing-law sentence every
time but scores the short form ("New York law governs.") below the threshold.
This swaps only that sentence for other wordings of the same choice of law and
scores each version again with the Governing Law question. Each contract is read
as text (the PDF's own text, so the "as written" row is the control) and nothing
else changes. It is a diagnostic only: no threshold or model setting is changed.

    python scripts/probe_governing_law.py [--dir eval/fictional]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jargonaut.config import CUAD_QUESTIONS
from jargonaut.extractor import Extractor
from jargonaut.pipeline import open_contract

WORDINGS = {
    "passive, short": "This Agreement is governed by {state} law.",
    "active, laws of the State of": "The laws of the State of {state} govern this Agreement.",
    "passive, laws of the State of": "This Agreement shall be governed by the laws of the State of {state}.",
    "CUAD boilerplate": ("This Agreement shall be governed by and construed in accordance with the laws of the "
                         "State of {state}, without regard to its conflict of laws principles."),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="eval/fictional")
    args = parser.parse_args()
    dir_ = Path(args.dir)
    blind = json.loads((dir_ / "blind_map.json").read_text())
    gold = {r["doc"]: r["spans"][0] for r in map(json.loads, open(dir_ / "gold.jsonl", encoding="utf-8"))
            if r["category"] == "Governing Law"}
    extractor = Extractor()
    question = CUAD_QUESTIONS["Governing Law"]

    def score(text, sentence):
        contract = open_contract(text.encode("utf-8"), "contract.txt")
        ext = extractor.extract(question, contract.index.top_k(question))
        i = contract.doc.text.find(sentence)
        on = bool(ext.best and i >= 0 and ext.best.start < i + len(sentence) and ext.best.end > i)
        return {"score": round(ext.best.diff, 2), "accepted": ext.accepted, "on_sentence": on}

    rows = []
    for d in sorted(blind):
        text = open_contract((dir_ / "pdfs" / f"{d}.pdf").read_bytes(), f"{d}.pdf").doc.text
        sentence = gold[d]
        state = re.search(r"(New York|Wisconsin|Texas|Indiana|Delaware)", sentence).group(1)
        row = {"doc": d, "file": blind[d], "as written": {"sentence": sentence, **score(text, sentence)}}
        for name, template in WORDINGS.items():
            new = template.format(state=state)
            row[name] = {"sentence": new, **score(text.replace(sentence, new), new)}
        rows.append(row)
        print(d, f"{blind[d]:36}", "  ".join(f"{row[k]['score']:6.2f}{'*' if row[k]['accepted'] else ' '}"
                                             for k in ["as written", *WORDINGS]), flush=True)

    out = {"threshold": extractor.threshold, "wordings": WORDINGS, "rows": rows,
           "accepted": {k: sum(r[k]["accepted"] for r in rows) for k in ["as written", *WORDINGS]},
           "on_sentence": {k: sum(r[k]["on_sentence"] for r in rows) for k in ["as written", *WORDINGS]}}
    (dir_ / "probe_governing_law.json").write_text(json.dumps(out, indent=2))
    print("columns: as written |", " | ".join(WORDINGS), "  (* = accepted)")
    print(json.dumps({"accepted": out["accepted"], "on_sentence": out["on_sentence"]}, indent=2))


if __name__ == "__main__":
    main()
