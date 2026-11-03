"""Evaluate the question router (step 2) on eval/router_questions.jsonl.

The 220 questions were written by a separate agent from the CUAD category
descriptions alone, without seeing the router: 16 per category and 60 out of scope
(q161-q200 are other contract topics, including deliberate near-misses; q201-q220
have nothing to do with the contract), split 110 dev / 110 test. The router was
designed on dev; test was run once, at the end.

A prediction is correct if it is the label or one of its also_acceptable topics; for
out-of-scope questions, if the router says out of scope. A clarification question
counts as correct when the right topic is one of the two options offered.

    python scripts/eval_router.py --split dev [--errors] [--sweep]          Gemma router (the app's)
    python scripts/eval_router.py --split dev --fallback [--errors]         keyword rules + NLI
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jargonaut.router import CLARIFY_AT, GEMMA_OPTIONS, MARGIN, OOS, THRESHOLD, decide, decide_gemma, rule_hits

ROOT = Path(__file__).resolve().parent.parent


def outcome(q, route):
    if route.method == "clarify":
        return "clarify", q["label"] in route.options
    pred = route.category or OOS
    return pred, pred in {q["label"], *q["also_acceptable"]}


def summarise(rows):
    ins = [r for r in rows if r["label"] != OOS]
    oos = [r for r in rows if r["label"] == OOS]
    share = lambda xs, f: round(sum(map(f, xs)) / len(xs), 3) if xs else None
    return {
        "n": len(rows), "accuracy": share(rows, lambda r: r["ok"]),
        "in_scope": {"n": len(ins), "correct": share(ins, lambda r: r["ok"]),
                     "wrong_topic": share(ins, lambda r: not r["ok"] and r["pred"] not in (OOS, "clarify")),
                     "said_out_of_scope": share(ins, lambda r: r["pred"] == OOS),
                     "clarify": share(ins, lambda r: r["pred"] == "clarify")},
        "out_of_scope": {"n": len(oos), "rejected": share(oos, lambda r: r["pred"] == OOS),
                         "near_miss_rejected": share([r for r in oos if int(r["id"][1:]) <= 200], lambda r: r["pred"] == OOS),
                         "unrelated_rejected": share([r for r in oos if int(r["id"][1:]) > 200], lambda r: r["pred"] == OOS)},
        "by_method": {m: {"n": c, "correct": share([r for r in rows if r["method"] == m], lambda r: r["ok"])}
                      for m, c in Counter(r["method"] for r in rows).items()},
    }


def run(questions, decide_one):
    rows = []
    for q in questions:
        route = decide_one(q)
        pred, ok = outcome(q, route)
        top = sorted(route.scores.items(), key=lambda kv: -kv[1])[:2] if route.scores else []
        rows.append({**q, "pred": pred, "ok": ok, "method": route.method, "hits": route.rule_hits,
                     "top": [(k, round(v, 3)) for k, v in top]})
    return rows, summarise(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--fallback", action="store_true", help="evaluate the rules + NLI fallback instead")
    parser.add_argument("--errors", action="store_true", help="list the misrouted questions")
    parser.add_argument("--sweep", action="store_true", help="try other thresholds")
    parser.add_argument("--rescore", action="store_true", help="recompute the cached model scores")
    args = parser.parse_args()
    questions = [json.loads(l) for l in open(ROOT / "eval/router_questions.jsonl", encoding="utf-8") if l.strip()]
    questions = [q for q in questions if q["split"] == args.split]
    name = "fallback" if args.fallback else "gemma"

    # Model scores are cached per split; the cache is tied to the prompt wording.
    version = hashlib.sha1(json.dumps(GEMMA_OPTIONS if not args.fallback else "nli-v1").encode()).hexdigest()[:8]
    cache_path = ROOT / f"eval/router_cache_{name}_{args.split}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() and not args.rescore else {}
    if cache.get("version") != version:
        cache = {"version": version}
    missing = [q for q in questions if q["id"] not in cache]
    if missing:
        if args.fallback:
            from jargonaut.checks import NLIScorer
            from jargonaut.router import RuleRouter

            router = RuleRouter(NLIScorer())
            for q in missing:
                cache[q["id"]] = router.nli_scores(q["question"])
        else:
            from jargonaut.router import GemmaRouter
            from jargonaut.simplifier import Simplifier

            router = GemmaRouter(Simplifier())
            for q in missing:
                cache[q["id"]] = router.probabilities(q["question"])
        cache_path.write_text(json.dumps(cache))

    if args.fallback:
        settings = {"threshold": THRESHOLD, "margin": MARGIN}
        one = lambda q, t=THRESHOLD, m=MARGIN: decide(rule_hits(q["question"]), *cache[q["id"]], t, m)
    else:
        settings = {"clarify_at": CLARIFY_AT}

        def one(q, c=CLARIFY_AT):
            route = decide_gemma(cache[q["id"]], c)
            route.rule_hits = rule_hits(q["question"])
            return route

    rows, summary = run(questions, one)
    result = {"router": name, "split": args.split, **settings, **summary}
    print(json.dumps(result, indent=2))
    if args.sweep:
        if args.fallback:
            print("\nthreshold margin accuracy in_scope oos_rejected clarify")
            for t in (0.3, 0.5, 0.7, 0.9):
                for m in (0.0, 0.05, 0.1):
                    _, s = run(questions, lambda q: one(q, t, m))
                    print(f"{t:9.2f} {m:6.2f} {s['accuracy']:8.3f} {s['in_scope']['correct']:8.3f} "
                          f"{s['out_of_scope']['rejected']:12.3f} {s['in_scope']['clarify']:7.3f}")
        else:
            print("\nclarify_at accuracy in_scope oos_rejected clarify")
            for c in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 1.01):
                _, s = run(questions, lambda q: one(q, c))
                print(f"{c:10.2f} {s['accuracy']:8.3f} {s['in_scope']['correct']:8.3f} "
                      f"{s['out_of_scope']['rejected']:12.3f} {s['in_scope']['clarify']:7.3f}")
    if args.errors:
        print("\nMisrouted:")
        for r in rows:
            if not r["ok"]:
                print(f"{r['id']} [{r['label']} -> {r['pred']} via {r['method']}; rules {r['hits']}; top {r['top']}]\n"
                      f"    {r['question']}")
    (ROOT / f"eval/router_results_{name}_{args.split}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
