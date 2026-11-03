"""Ask questions about a contract from the command line.

    python ask.py contract.pdf "Can I cancel this early?"          one question
    python ask.py contract.pdf --topic Insurance                    one category, no routing
    python ask.py contract.pdf                                      interactive conversation
    python ask.py contract.pdf "Which law applies?" --json          the full answer as JSON
    python ask.py contract.pdf --no-gemma ...                       skip Gemma (clauses only, faster start)
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

from jargonaut.config import CATEGORIES, TOPIC_LABELS


def show(answer, as_json=False):
    a = answer.to_dict()
    if as_json:
        print(json.dumps(a, indent=2, ensure_ascii=False))
        return
    wrap = lambda s, indent="  ": textwrap.fill(" ".join(s.split()), 100, initial_indent=indent, subsequent_indent=indent)
    print()
    if a["topic"]:
        conf = a["confidence"] or {}
        prob = f", P(evidence) {conf['probability']:.2f}" if conf.get("probability") is not None else ""
        score = f"score {conf['score']} vs threshold {conf['threshold']}{prob}" if conf.get("score") is not None else ""
        print(f"[{a['topic']}] {score}")
    if a["kind"] == "answer":
        print("In plain English:"); print(wrap(a["rewrite"]))
    elif a["kind"] == "unchecked":
        print("WARNING:", a["message"])
    else:
        print(a["message"])
    if a["clause"]:
        print("From the contract:"); print(wrap(a["clause"]))
        where = " > ".join(a["section"] or []) or "section not found"
        pages = ", ".join(map(str, a["pages"])) if a["pages"] else "n/a"
        print(f"  ({where}; page {pages})")
    if a["checks"] and "numbers" in a["checks"]:
        c = a["checks"]
        cov = c["nli"].get("nli_coverage")
        print(f"Checks: numbers {'ok' if c['numbers']['passed'] else 'FAILED'}, negation "
              f"{'ok' if c['negation']['passed'] else 'FAILED'}, NLI coverage "
              f"{'n/a' if cov is None else f'{cov:.2f}'} ({'ok' if c['nli']['passed'] else 'FAILED'})")
        if a["kind"] == "unchecked" and a["rewrite"]:
            print("Rewrite that failed the checks:"); print(wrap(a["rewrite"]))
    if a["timings"]:
        print("Time:", ", ".join(f"{k} {v:.2f}s" for k, v in a["timings"].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("contract")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--topic", choices=CATEGORIES, help="answer this category directly")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-gemma", action="store_true")
    args = parser.parse_args()

    from jargonaut.pipeline import Conversation, Jargonaut, open_contract

    path = Path(args.contract)
    contract = open_contract(path.read_bytes(), path.name)
    s = contract.summary()
    print(f"{s['name']}: {s['pages'] or '?'} pages, {s['words']} words, {s['chunks']} passages, "
          f"{s['headings']} headings", file=sys.stderr)
    for w in s["warnings"]:
        print("  note:", w, file=sys.stderr)
    system = Jargonaut(load_simplifier=not args.no_gemma, log=lambda m: print(m, file=sys.stderr))
    conversation = Conversation(system, contract)
    if args.topic:
        show(conversation.reply_category(args.topic), args.json)
    elif args.question:
        show(conversation.reply(" ".join(args.question)), args.json)
    else:
        print("Topics:", "; ".join(TOPIC_LABELS[c] for c in CATEGORIES), file=sys.stderr)
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            show(conversation.reply(q), args.json)


if __name__ == "__main__":
    main()
